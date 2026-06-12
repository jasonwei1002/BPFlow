"""BPFlow training entrypoint.

Single GPU / CPU:
    python -m bpflow.train --config bpflow/config/gpu.yaml
Multi-GPU (DDP):
    torchrun --nproc_per_node=N -m bpflow.train --config bpflow/config/gpu.yaml
"""

import argparse
import logging
import os

from .trainer import Trainer
from .trainer_utils import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    # Only rank 0 prints INFO; other DDP ranks stay quiet (WARNING+) so the
    # terminal isn't multiplied by the number of GPUs.
    rank = int(os.environ.get("RANK", 0))
    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)
    if args.init_ckpt is not None:
        cfg.training.init_from_ckpt = args.init_ckpt
    if args.resume is not None:
        cfg.training.resume_dir = args.resume
    trainer = Trainer(cfg)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
