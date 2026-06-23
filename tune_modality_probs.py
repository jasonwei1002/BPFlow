"""Optuna search over the unified model's ``data.task_probs`` (5-task system).

Objective: MINIMIZE the best ``val/MAE`` of a unified pretrain — and ``val/MAE``
is the MEAN waveform MAE (mmHg) across the trained ->ABP directions only
(ecg_ppg2abp / ecg2abp / ppg2abp; ``Trainer`` reports this as ``best_val``). So the
search optimises "all ->ABP directions good" directly. Single objective.

WHY THE BRIDGE TASKS ARE NOT SEARCHED
-------------------------------------
The model trains 5 tasks: ``[ecg_ppg2abp, ecg2abp, ppg2abp, ppg2ecg, ecg2ppg]``
(this exact order — it MUST match ``base.yaml: data.tasks``). The two bridge
directions ppg2ecg / ecg2ppg live in [0,1] units and are *deliberately excluded*
from the decision metric. If the search were free to allocate their probability
mass, the optimiser would drive both to ~0 (every unit spent on a bridge is mass
stolen from the scored ->ABP tasks), starving the bridge directions and destroying
the unified multi-direction model. So the bridge probs are held at a FIXED reserve
(``--bridge-ppg2ecg`` / ``--bridge-ecg2ppg``); only the ->ABP budget is searched.
Raise a bridge reserve manually if you want that direction trained harder — tuning
bridges *for their own metric* needs a multi-objective run, out of scope here.

SEARCH SPACE (budget-relative reparam; flagship kept >= 0.20)
------------------------------------------------------------
    abp_budget = 1 - bridge_ppg2ecg - bridge_ecg2ppg            (e.g. 0.75)
    p_ecg_ppg2abp = U[flag_lo, flag_hi]                          (flagship, abs)
    ppg_share     = U[ppg_share_lo, ppg_share_hi]               (ppg2abp's slice of
                                                                  the non-flagship budget)
    R          = abp_budget - p_ecg_ppg2abp
    p_ppg2abp  = R * ppg_share        # biased high: ppg2abp is the bottleneck
    p_ecg2abp  = R * (1 - ppg_share)
    -> task_probs = [p_ecg_ppg2abp, p_ecg2abp, p_ppg2abp, bridge_ppg2ecg, bridge_ecg2ppg]
This guarantees sum == 1, flagship in [flag_lo, flag_hi], and lets ppg2abp grow far
above its current 0.20 (the hypothesis under test).

One trial = one full unified pretrain via ``train.sh`` (DDP/torchrun, all GPUs).
Trials run SEQUENTIALLY (each grabs the whole node). Pruning: ASHA reads the per-epoch
``[val] MAE mean=`` line ``Trainer.validate()`` prints on rank 0 and kills a losing
allocation after a few epochs instead of burning a full run — the biggest lever when
each trial costs hours.

PREREQUISITE: the ppg2abp specialist PASSES AAMI (SBP SDE ~6.99) while the unified
ppg2abp FAILS (SDE ~7.99) at only 20% mass — so ppg2abp is NOT info-limited and more
mass should help. That is exactly what this search probes. (If a direction's
specialist itself failed, no prob mix would recover it — don't search blind.)

    pip install optuna
    python tune_modality_probs.py --n-trials 8 --search-epochs 300 --nproc gpu

WHY ``--search-epochs`` DEFAULTS TO 300 (NOT A TIGHT CAP)
--------------------------------------------------------
The real pretrain's val/MAE is only ~96% converged at epoch 150 (still ~0.96 mmHg
above its early-stop best at ~ep250) — larger than the sub-mmHg gap this search must
resolve. Truncating at 150 would rank survivors on an un-converged metric and pick
the wrong winner. So the cap is set ABOVE the natural early-stop (early_stop_patience
ends a run ~ep250): early-stop — not the cap — ends each trial at true convergence,
while ASHA still kills clearly-losing allocations by ~ep40-60 (72-86% of the drop is
in by then). Per-surviving-trial cost == one real pretrain, not more.

Resumable: re-run the same command; the SQLite study (``--storage``) continues. Each
trial also logs its own SwanLab run (gpu.yaml use_swanlab=true), so per-trial curves
live in SwanLab; the study DB holds the (task_probs -> best_val) map.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import subprocess
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:  # type-only imports (keep optuna/torch optional at runtime)
    from types import ModuleType

    from optuna.trial import BaseTrial, Trial  # Trial: live; BaseTrial: also FixedTrial

logger = logging.getLogger("tune_modality_probs")

REPO = os.path.dirname(os.path.abspath(__file__))
_VAL_RE = re.compile(r"MAE mean=([0-9]+\.[0-9]+)")  # matches Trainer.validate()'s rank-0 log line

# The config train.sh loads (it inherits base.yaml's data.tasks). main() asserts
# this config's task order matches TASK_ORDER below.
TRAIN_CONFIG = "bpflow/config/gpu.yaml"
# The 5-task unified set in the exact positional order _suggest_probs assumes
# (flagship / ecg2abp / ppg2abp / bridge / bridge). main() verifies this equals
# bpflow.data.TASK_ORDER AND TRAIN_CONFIG's data.tasks, so a reordered or resized
# task set fails loudly at startup instead of silently misaligning the probs.
TASK_ORDER = ["ecg_ppg2abp", "ecg2abp", "ppg2abp", "ppg2ecg", "ecg2ppg"]


@dataclass(frozen=True)
class SearchSpace:
    """Fixed bridge reserve + the searched ->ABP allocation ranges."""
    bridge_ppg2ecg: float
    bridge_ecg2ppg: float
    flag_lo: float
    flag_hi: float
    ppg_share_lo: float
    ppg_share_hi: float

    @property
    def abp_budget(self) -> float:
        return round(1.0 - self.bridge_ppg2ecg - self.bridge_ecg2ppg, 6)


def _suggest_probs(trial: "BaseTrial", space: SearchSpace) -> List[float]:
    """Sample a 5-task task_probs vector in TASK_ORDER (bridges fixed, ->ABP searched)."""
    flagship = trial.suggest_float("p_flagship", space.flag_lo, space.flag_hi)
    ppg_share = trial.suggest_float("ppg_share", space.ppg_share_lo, space.ppg_share_hi)
    r = space.abp_budget - flagship  # non-flagship ->ABP budget
    p_ppg2abp = r * ppg_share
    p_ecg2abp = r * (1.0 - ppg_share)
    probs = [flagship, p_ecg2abp, p_ppg2abp, space.bridge_ppg2ecg, space.bridge_ecg2ppg]
    return [round(p, 4) for p in probs]


def _read_best_val(trial_dir: str) -> float:
    """Best mean ->ABP val/MAE from the run's best checkpoint (Trainer saves best_val)."""
    import torch

    ckpt = os.path.join(trial_dir, "checkpoint_best.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"no checkpoint_best.pth in {trial_dir} (run never improved?)")
    obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    return float(obj["best_val"])


def _run_training(probs: List[float], name: str, out_root: str, epochs: int, nproc: str, trial: "Trial") -> None:
    """Launch one unified pretrain; stream rank-0 output, ASHA-prune on per-epoch MAE.

    Only ``data.task_probs`` is overridden — ``data.tasks`` is left at the base.yaml
    5-task default so the probs stay positionally aligned to TASK_ORDER. run_name=<name>
    makes exp_dir = out_root/<name> AND names the SwanLab run, so both carry the probs.
    """
    probs_str = "[" + ",".join(str(p) for p in probs) + "]"  # no shell -> brackets pass literally
    cmd = [
        "bash", "train.sh", "--nproc", str(nproc),
        f"data.task_probs={probs_str}",
        f"training.output_dir={out_root}",
        f"training.run_name={name}",                    # exp_dir basename + SwanLab experiment_name
        f"training.epochs={epochs}",                    # cap trial cost (winner retrains to full)
        "training.val_freq_epoch=1",                    # 1 val/epoch -> step counter == epoch (pruner alignment)
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


def _optuna() -> "ModuleType":
    try:
        import optuna
    except ImportError as e:  # keep the dep optional until the search is actually run
        raise SystemExit("optuna not installed: pip install optuna") from e
    return optuna


def _run_name(trial_number: int, probs: List[float]) -> str:
    """Probs-encoded, dir-unique run name, e.g. tune_t00_fl22_pp30 (flagship & ppg2abp x100).

    The t<n> prefix keeps the exp_dir unique even when two nearby trials round to the
    same probs string (so checkpoints never clobber each other)."""
    flag, ppg2abp = round(probs[0] * 100), round(probs[2] * 100)
    return f"tune_t{trial_number:02d}_fl{flag}_pp{ppg2abp}"


def make_objective(space: SearchSpace, out_root: str, epochs: int, nproc: str) -> "Callable[[Trial], float]":
    def objective(trial: "Trial") -> float:
        probs = _suggest_probs(trial, space)
        name = _run_name(trial.number, probs)
        _run_training(probs, name, out_root, epochs, nproc, trial)
        best = _read_best_val(os.path.join(out_root, name))  # exp_dir = out_root/<name>
        logger.info("trial %d done: %s  probs=%s  best ->ABP val/MAE=%.4f",
                    trial.number, name, probs, best)
        return best

    return objective


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Optuna search over data.task_probs (min mean ->ABP val/MAE)")
    ap.add_argument("--n-trials", type=int, default=8, help="total trials (each = one full pretrain)")
    ap.add_argument("--search-epochs", type=int, default=300,
                    help="epochs CEILING per trial; set above the ~ep250 natural early-stop so "
                         "early-stop (not this cap) ends each trial at convergence. <200 ranks "
                         "survivors on an un-converged metric — see module docstring.")
    ap.add_argument("--nproc", default="gpu", help="GPUs per trial (gpu = all; trials run sequentially)")
    ap.add_argument("--out-root", default="output/tune_probs", help="parent dir for per-trial run dirs")
    ap.add_argument("--storage", default="sqlite:///optuna_taskprobs.db", help="Optuna storage (resumable)")
    ap.add_argument("--study-name", default="task_probs")
    # Fixed bridge reserve (NOT searched — see module docstring). Default biases the
    # weaker direction (ppg2ecg, r~0.77) over the strong one (ecg2ppg, r~0.97).
    ap.add_argument("--bridge-ppg2ecg", type=float, default=0.15, help="fixed prob mass for ppg2ecg")
    ap.add_argument("--bridge-ecg2ppg", type=float, default=0.10, help="fixed prob mass for ecg2ppg")
    # ->ABP search ranges.
    ap.add_argument("--flag-lo", type=float, default=0.20, help="flagship (ecg_ppg2abp) lower bound")
    ap.add_argument("--flag-hi", type=float, default=0.45, help="flagship (ecg_ppg2abp) upper bound")
    ap.add_argument("--ppg-share-lo", type=float, default=0.35,
                    help="ppg2abp's min share of the non-flagship ->ABP budget")
    ap.add_argument("--ppg-share-hi", type=float, default=0.70,
                    help="ppg2abp's max share of the non-flagship ->ABP budget")
    args = ap.parse_args()

    # Guard the positional probs<->tasks alignment this tuner assumes (the reparam in
    # _suggest_probs hardcodes the 5-task layout). Verify TASK_ORDER matches both the
    # canonical bpflow.data list AND TRAIN_CONFIG's data.tasks (what train.sh loads),
    # so a reordered/resized task set fails HERE instead of silently misaligning the
    # probs or erroring late inside a spawned trial. (Imports are local to keep
    # torch/bpflow off the --help path.)
    from bpflow.data import TASK_ORDER as _CANON_TASK_ORDER
    from bpflow.trainer_utils import load_config

    cfg_tasks = list(load_config(TRAIN_CONFIG).data.tasks)
    if TASK_ORDER != list(_CANON_TASK_ORDER) or cfg_tasks != TASK_ORDER:
        raise SystemExit(
            f"task-order mismatch: TASK_ORDER={TASK_ORDER}, bpflow.data.TASK_ORDER="
            f"{list(_CANON_TASK_ORDER)}, {TRAIN_CONFIG} data.tasks={cfg_tasks}. This tuner "
            "assumes the 5-task unified set in that exact order; update TASK_ORDER or the config."
        )

    space = SearchSpace(
        bridge_ppg2ecg=args.bridge_ppg2ecg, bridge_ecg2ppg=args.bridge_ecg2ppg,
        flag_lo=args.flag_lo, flag_hi=args.flag_hi,
        ppg_share_lo=args.ppg_share_lo, ppg_share_hi=args.ppg_share_hi,
    )
    if space.abp_budget - space.flag_hi < 0.05:
        raise SystemExit(
            f"abp_budget ({space.abp_budget}) too small for flag_hi ({args.flag_hi}); "
            "lower --flag-hi or the bridge reserves."
        )
    logger.info("search space: bridges(ppg2ecg=%.2f, ecg2ppg=%.2f) abp_budget=%.2f "
                "flagship in [%.2f,%.2f] ppg_share in [%.2f,%.2f]; tasks=%s",
                space.bridge_ppg2ecg, space.bridge_ecg2ppg, space.abp_budget,
                space.flag_lo, space.flag_hi, space.ppg_share_lo, space.ppg_share_hi, TASK_ORDER)

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
    # Seed with informed allocations (in search-param space, so they respect any bridge
    # reserve): current-baseline-ish | rebalance-to-ppg | aggressive-ppg+low-flagship.
    for p_flagship, ppg_share in [(0.30, 0.50), (0.24, 0.62), (0.20, 0.65)]:
        if space.flag_lo <= p_flagship <= space.flag_hi and space.ppg_share_lo <= ppg_share <= space.ppg_share_hi:
            study.enqueue_trial({"p_flagship": p_flagship, "ppg_share": ppg_share}, skip_if_exists=True)
        else:
            logger.warning(
                "seed (p_flagship=%.2f, ppg_share=%.2f) is outside the search bounds "
                "[flag %.2f-%.2f, ppg_share %.2f-%.2f] — NOT enqueued",
                p_flagship, ppg_share, space.flag_lo, space.flag_hi,
                space.ppg_share_lo, space.ppg_share_hi,
            )

    # n_trials is a TOTAL budget, not a per-invocation count: on resume, only run the
    # trials still missing (a re-run after a crash tops up to n_trials, not +n_trials).
    done = sum(1 for t in study.trials if t.state.is_finished())
    remaining = max(0, args.n_trials - done)
    logger.info("study has %d finished trial(s); running %d more (target %d)", done, remaining, args.n_trials)
    if remaining:
        # A single bad trial is marked FAILED and the sweep continues, instead of one
        # error aborting a multi-day run. Covers train.sh non-zero exit (RuntimeError)
        # AND _read_best_val failures on a trial that never improved — no
        # checkpoint_best.pth (FileNotFoundError) or a checkpoint missing best_val
        # (KeyError). Optuna handles TrialPruned itself, independent of this tuple.
        study.optimize(
            make_objective(space, args.out_root, args.search_epochs, args.nproc),
            n_trials=remaining, catch=(RuntimeError, FileNotFoundError, KeyError),
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
    probs = _suggest_probs(optuna.trial.FixedTrial(best.params), space)
    logger.info("BEST trial %d: task_probs=%s (order %s)  mean ->ABP val/MAE=%.4f",
                best.number, probs, TASK_ORDER, best.value)
    logger.info("Retrain the winner to full convergence: "
                "bash train.sh 'data.task_probs=%s'", probs)


if __name__ == "__main__":
    main()
