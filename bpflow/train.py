"""BPFlow training entrypoint.

Single GPU / CPU:
    python -m bpflow.train --config bpflow/config/pulsedb.yaml
Multi-GPU (DDP):
    torchrun --nproc_per_node=N -m bpflow.train --config bpflow/config/pulsedb.yaml
"""

import argparse
import logging
import os

from omegaconf import OmegaConf

from .trainer import Trainer
from .trainer_utils import load_config

logger = logging.getLogger(__name__)


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="BPFlow training entrypoint")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--init-ckpt", default=None,
        help="Pretrained checkpoint to initialize weights from (finetune); "
             "overrides training.init_from_ckpt.",
    )
    parser.add_argument(
        "--resume", default=None,
        help="Existing run dir (e.g. output/20260101_120000) to resume IN PLACE: "
             "reuse it as the output dir, load checkpoint_latest.pth, and continue "
             "the same SwanLab run. Overrides training.resume_dir.",
    )
    # Leftover args are dotted key=value config overrides (see _overrides_from_extra),
    # e.g. `data.finetune_train_ratio=0.25 training.lr=3e-5`.
    return parser.parse_known_args()


def _overrides_from_extra(extra: list[str]) -> dict:
    """Turn leftover CLI tokens into a config-override dict via OmegaConf dotlist.

    e.g. ["data.finetune_train_ratio=0.25", "training.lr=3e-5"] ->
         {"data": {"finetune_train_ratio": 0.25}, "training": {"lr": 3e-05}}.
    OmegaConf infers each value's type; a value whose type clashes with the
    structured schema — or a key the schema lacks (a typo) — fails loudly when
    load_config merges it into the struct-mode config.
    """
    bad = [tok for tok in extra if tok.startswith("-") or "=" not in tok]
    if bad:
        raise SystemExit(
            f"unrecognized argument(s): {bad}. Pass config overrides as dotted "
            "key=value, e.g. data.finetune_train_ratio=0.25 training.lr=3e-5"
        )
    return OmegaConf.to_container(OmegaConf.from_dotlist(extra), resolve=False)  # type: ignore[return-value]


def main() -> None:
    # Only rank 0 prints INFO; other DDP ranks stay quiet (WARNING+) so the
    # terminal isn't multiplied by the number of GPUs.
    rank = int(os.environ.get("RANK", 0))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args, extra = parse_args()
    overrides = _overrides_from_extra(extra)
    cfg = load_config(args.config, overrides or None)
    if args.init_ckpt is not None:
        cfg.training.init_from_ckpt = args.init_ckpt
    if args.resume is not None:
        cfg.training.resume_dir = args.resume
    if rank == 0 and overrides:
        logger.info("Config overrides from CLI: %s", overrides)
    trainer = Trainer(cfg)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
