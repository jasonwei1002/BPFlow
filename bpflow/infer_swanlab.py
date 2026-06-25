"""Log multi-seed infer test metrics as a SwanLab run inside the pipeline's group.

run_uci.sh / run_pulsedb.sh call this after the infer stage. The pipeline's
pretrain and finetune each log their own SwanLab run under a shared group; the
orchestrator owns that group identity and passes it in via ``--group``. This
creates a THIRD run in the same group for the held-out test scores, so pretrain
/ finetune / infer sit together on the dashboard.

Never fatal: a missing swanlab or any SDK error is logged and skipped so it
can't break the pipeline.

    python -m bpflow.infer_swanlab \
        --summary output/uci_infer_<ts>/summary.json \
        --group   uci_<ts>_<slug> \
        --ckpt    output/uci_ft_<ts>_<slug>/checkpoint_best.pth
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Keep ``/`` (SwanLab metric grouping), ``.`` and ``-``; collapse everything else.
_SAFE_RE = re.compile(r"[^0-9A-Za-z_/.\-]+")


def _safe_key(key: str) -> str:
    """SwanLab-safe metric name, e.g. ``SBP/BHS/<=5mmHg`` -> ``SBP/BHS/le5mmHg``."""
    key = key.replace("<=", "le").replace(">=", "ge")
    return _SAFE_RE.sub("_", key).strip("_")


def _test_metrics(summary: dict) -> dict:
    """``summary.json`` ``{path: {mean,std,...}}`` ->
    ``{test/<path>: mean, test/<path>_std: std}``. The single-task ``_single/``
    prefix is stripped so keys read ``test/waveform/MAE``."""
    out: dict = {}
    for path, stat in summary.items():
        if not isinstance(stat, dict) or "mean" not in stat:
            continue
        key = path[len("_single/"):] if path.startswith("_single/") else path
        key = _safe_key(key)
        out[f"test/{key}"] = float(stat["mean"])
        out[f"test/{key}_std"] = float(stat["std"])
    return out


def log_test_run(summary_path: str, project: str = "bpflow", group: str = "",
                 ckpt: str = "") -> bool:
    """Create a SwanLab run (in ``group``) and log ``test/*`` metrics. True if logged."""
    try:
        import swanlab
    except ImportError:
        logger.warning("swanlab not installed; skipping infer SwanLab logging.")
        return False
    metrics = _test_metrics(json.loads(Path(summary_path).read_text()))
    if not metrics:
        logger.warning("no numeric metrics in %s; nothing to log.", summary_path)
        return False
    run_name = Path(summary_path).resolve().parent.name  # e.g. uci_infer_<ts>
    init_kwargs: dict = {
        "project": project,
        "mode": "online",  # grouped cloud run; offline wouldn't reach the dashboard
        "experiment_name": run_name,
        "description": "BPFlow held-out test scores",
        "config": {"infer_ckpt": ckpt} if ckpt else {},
    }
    if group:
        init_kwargs["group"] = group
    try:
        swanlab.init(**init_kwargs)
        swanlab.log(metrics)
        swanlab.finish()
    except Exception as e:  # noqa: BLE001 — logging must never break the pipeline
        logger.warning("SwanLab infer logging failed (%s); metrics remain in %s.", e, summary_path)
        return False
    logger.info("Logged %d test/* metrics to SwanLab run '%s' (project=%s, group=%s).",
                len(metrics), run_name, project, group or "<none>")
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Log infer test metrics as a SwanLab run in the pipeline group")
    ap.add_argument("--summary", required=True, help="Path to the multi-seed summary.json")
    ap.add_argument("--group", default="", help="SwanLab group (the orchestrator's pipeline id)")
    ap.add_argument("--project", default="bpflow", help="SwanLab project")
    ap.add_argument("--ckpt", default="", help="Scored checkpoint, recorded as run config provenance")
    args = ap.parse_args()
    log_test_run(args.summary, args.project, args.group, args.ckpt)


if __name__ == "__main__":
    main()
