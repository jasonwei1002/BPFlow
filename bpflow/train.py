"""BPFlow training entrypoint.

Single GPU / CPU:
    python -m bpflow.train --config bpflow/config/gpu.yaml
Multi-GPU (DDP):
    torchrun --nproc_per_node=N -m bpflow.train --config bpflow/config/gpu.yaml
"""

import argparse
import logging

from .trainer import Trainer
from .trainer_utils import load_config

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BPFlow training entrypoint")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    args = parse_args()
    cfg = load_config(args.config)
    trainer = Trainer(cfg)
    try:
        trainer.train()
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
