"""Optuna search over the unified model's ``modality_dropout_probs``.

Objective: MINIMIZE the best ``val/MAE`` of a unified pretrain — and ``val/MAE``
is the MEAN waveform MAE across all trained directions (ecg_ppg/ecg/ppg), so the
search optimises "all three directions good" directly. Single objective.

One trial = one full unified pretrain launched via ``train.sh`` (DDP/torchrun, all
GPUs). Trials run SEQUENTIALLY (each grabs the whole node). Pruning: ASHA reads the
per-epoch ``[val] MAE mean=`` line that ``Trainer.validate()`` prints on rank 0, and
kills a losing prob-allocation after a few epochs instead of burning a full run —
the biggest lever when each trial costs hours.

Search space (2-simplex reparam, ecg_ppg kept >= 0.20 as the flagship):
    p_ecg, p_ppg ~ U[0.15, 0.40]   ->   p_ecg_ppg = 1 - p_ecg - p_ppg  in [0.20, 0.70]

PREREQUISITE (see CLAUDE.md discussion): run the ecg/ppg specialists first to learn
each direction's ceiling. If ppg is info-limited (its specialist also fails AAMI),
no prob mix recovers it and this search mostly trades ecg_ppg away — don't run it
blind.

    pip install optuna
    python tune_modality_probs.py --n-trials 8 --search-epochs 150 --nproc gpu

Resumable: re-run the same command; the SQLite study (``--storage``) continues.
Each trial's training run also logs its own SwanLab run (gpu.yaml use_swanlab=true),
so per-trial curves live in SwanLab; the study DB holds the (probs -> best_val) map.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import subprocess
from typing import List

logger = logging.getLogger("tune_modality_probs")

REPO = os.path.dirname(os.path.abspath(__file__))
_VAL_RE = re.compile(r"MAE mean=([0-9]+\.[0-9]+)")  # matches Trainer.validate()'s rank-0 log line


def _suggest_probs(trial) -> List[float]:
    """2-simplex reparam -> [ecg_ppg, ecg, ppg] (MODALITY_ORDER). ecg_ppg >= 0.20."""
    p_ecg = trial.suggest_float("p_ecg", 0.15, 0.40)
    p_ppg = trial.suggest_float("p_ppg", 0.15, 0.40)
    p_ecg_ppg = 1.0 - p_ecg - p_ppg
    return [round(p_ecg_ppg, 4), round(p_ecg, 4), round(p_ppg, 4)]


def _read_best_val(trial_dir: str) -> float:
    """Best mean val/MAE from the run's best checkpoint (Trainer saves best_val)."""
    import torch

    ckpt = os.path.join(trial_dir, "checkpoint_best.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint_best.pth in {trial_dir} (run never improved?)")
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    return float(obj["best_val"])


def _run_training(probs: List[float], name: str, out_root: str, epochs: int, nproc: str, trial) -> None:
    """Launch one unified pretrain; stream rank-0 output, ASHA-prune on per-epoch MAE.

    run_name=<name> makes exp_dir = out_root/<name> AND names the SwanLab run, so the
    on-disk dir and the SwanLab run both carry the probs.
    """
    a, b, c = probs
    cmd = [
        "bash", "train.sh", "--nproc", str(nproc),
        "data.modality_dropout=true",
        f"data.modality_dropout_probs=[{a},{b},{c}]",  # no shell -> brackets pass literally
        f"training.output_dir={out_root}",
        f"training.run_name={name}",                    # exp_dir basename + SwanLab experiment_name
        f"training.epochs={epochs}",                    # cap trial cost (winner retrains to full)
        "training.val_freq_epoch=1",                    # 1 val/epoch -> our step counter == epoch (pruner alignment)
    ]
    logger.info("trial %d: %s -> %s", trial.number, name, " ".join(cmd))
    # start_new_session: own process group so we can kill the whole torchrun on prune.
    proc = subprocess.Popen(
        cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, start_new_session=True,
    )
    step = 0
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            m = _VAL_RE.search(line)
            if not m:
                continue
            step += 1  # one val round ~= one epoch (val_freq_epoch=1)
            trial.report(float(m.group(1)), step)
            if trial.should_prune():
                _kill(proc)
                raise _optuna().TrialPruned(f"pruned at epoch ~{step}")
        if proc.wait() != 0:
            raise RuntimeError(f"train.sh exited {proc.returncode} (trial {trial.number})")
    finally:
        if proc.poll() is None:
            _kill(proc)


def _kill(proc: subprocess.Popen) -> None:
    """Terminate the whole torchrun process group AND reap it, so no zombie / lingering
    CUDA worker holds GPU memory into the next trial. No-op if already dead (so the
    finally-block re-call doesn't SIGTERM an unrelated recycled pgid)."""
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=60)
        return
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait(timeout=10)  # reap the killed group (avoid zombies / stuck GPU mem)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass


def _optuna():
    try:
        import optuna
    except ImportError as e:  # keep the dep optional until the search is actually run
        raise SystemExit("optuna not installed: pip install optuna") from e
    return optuna


def _run_name(trial_number: int, probs: List[float]) -> str:
    """Probs-encoded, dir-unique run name, e.g. tune_t00_ep34_e33_p33 (probs x100).

    The t<n> prefix keeps the exp_dir unique even when two nearby trials round to the
    same probs string (so checkpoints never clobber each other)."""
    a, b, c = (round(p * 100) for p in probs)
    return f"tune_t{trial_number:02d}_ep{a}_e{b}_p{c}"


def make_objective(out_root: str, epochs: int, nproc: str):
    def objective(trial) -> float:
        probs = _suggest_probs(trial)
        name = _run_name(trial.number, probs)
        _run_training(probs, name, out_root, epochs, nproc, trial)
        best = _read_best_val(os.path.join(out_root, name))  # exp_dir = out_root/<name>
        logger.info("trial %d done: %s  best val/MAE=%.4f", trial.number, name, best)
        return best

    return objective


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Optuna search over modality_dropout_probs (min mean val/MAE)")
    ap.add_argument("--n-trials", type=int, default=8, help="total trials (each = one full pretrain)")
    ap.add_argument("--search-epochs", type=int, default=150,
                    help="epochs cap PER trial during the search (winner retrains to full convergence)")
    ap.add_argument("--nproc", default="gpu", help="GPUs per trial (gpu = all; trials run sequentially)")
    ap.add_argument("--out-root", default="output/tune_probs", help="parent dir for per-trial run dirs")
    ap.add_argument("--storage", default="sqlite:///optuna_modality.db", help="Optuna storage (resumable)")
    ap.add_argument("--study-name", default="modality_probs")
    args = ap.parse_args()

    optuna = _optuna()
    os.makedirs(args.out_root, exist_ok=True)
    study = optuna.create_study(
        study_name=args.study_name, storage=args.storage, load_if_exists=True,
        direction="minimize",
        # TPE after seed points; swap to optuna.samplers.GPSampler() for more
        # sample-efficient BO at this tiny budget (needs botorch).
        sampler=optuna.samplers.TPESampler(n_startup_trials=3, seed=42),
        # ASHA reads the per-epoch reported MAE; n_warmup_steps lets each trial run a
        # few epochs (MAE drops fast early) before it's eligible to be killed.
        pruner=optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=20),
    )
    # Seed the search with the informed hand-picked allocations discussed (reparam'd):
    #   uniform [0.34,0.33,0.33] | flagship-tilt [0.5,0.25,0.25] | lift-weak [0.2,0.4,0.4]
    for p_ecg, p_ppg in [(0.33, 0.33), (0.25, 0.25), (0.40, 0.40)]:
        study.enqueue_trial({"p_ecg": p_ecg, "p_ppg": p_ppg}, skip_if_exists=True)

    # n_trials is a TOTAL budget, not a per-invocation count: on resume, only run the
    # trials still missing (a re-run after a crash tops up to n_trials, not +n_trials).
    done = sum(1 for t in study.trials if t.state.is_finished())
    remaining = max(0, args.n_trials - done)
    logger.info("study has %d finished trial(s); running %d more (target %d)", done, remaining, args.n_trials)
    if remaining:
        # catch=(RuntimeError,): a single trial whose train.sh fails is marked FAILED
        # and the sweep continues, instead of one transient error aborting a multi-day run.
        study.optimize(
            make_objective(args.out_root, args.search_epochs, args.nproc),
            n_trials=remaining, catch=(RuntimeError,),
        )

    # End-of-sweep breakdown by trial state (COMPLETE / PRUNED / FAIL / ...).
    states: dict = {}
    for t in study.trials:
        states[t.state.name] = states.get(t.state.name, 0) + 1
    logger.info("sweep finished: %d trial(s) total — %s", len(study.trials),
                ", ".join(f"{k}={v}" for k, v in sorted(states.items())))

    # best_trial raises if NO trial COMPLETEd (e.g. all FAILED via catch= / all pruned).
    if not any(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials):
        logger.warning("no completed trials yet — nothing to report (check the trial logs / GPU env)")
        return
    best = study.best_trial
    probs = _suggest_probs(optuna.trial.FixedTrial(best.params))
    logger.info("BEST trial %d: probs=%s  mean val/MAE=%.4f", best.number, probs, best.value)
    logger.info("Retrain the winner to full convergence: "
                "bash train.sh data.modality_dropout=true 'data.modality_dropout_probs=%s'", probs)


if __name__ == "__main__":
    main()
