"""Entry point: train one baseline on one direction.

Usage:
    python -m bpflow_baselines.train --model nabnet --direction ecg2abp \
        [--config bpflow_baselines/config/nabnet.yaml] [--init-ckpt X] \
        [--resume DIR] [dotted.key=value ...]

Routes P2E-WGAN to the WGAN-GP trainer; everything else to the supervised one.
The config defaults to bpflow_baselines/config/<model>.yaml.
"""

from __future__ import annotations

import argparse
import logging
import os

from .config import load_config, overrides_from_extra

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_DEFAULT_CFG = "bpflow_baselines/config/{model}.yaml"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   choices=["mdvisco", "nabnet", "patchtst", "ppg2abp", "p2e_wgan", "wavenet"])
    p.add_argument("--direction", required=True,
                   choices=["ecg2abp", "ppg2abp", "ecg2ppg", "ppg2ecg", "ecg_ppg2abp"])
    p.add_argument("--config", default=None)
    p.add_argument("--init-ckpt", default=None)
    p.add_argument("--resume", default=None)
    args, extra = p.parse_known_args()

    cfg_path = args.config or _DEFAULT_CFG.format(model=args.model)
    if not os.path.exists(cfg_path):
        raise SystemExit(f"config not found: {cfg_path}")
    overrides = overrides_from_extra(extra)
    cfg = load_config(cfg_path, overrides=overrides)
    cfg.model.name = args.model
    cfg.baseline.direction = args.direction
    if args.init_ckpt:
        cfg.training.init_from_ckpt = args.init_ckpt
    if args.resume:
        cfg.training.resume_dir = args.resume

    if args.model == "p2e_wgan":
        from . import gan_trainer as runner
    else:
        from . import trainer as runner
    runner.train(cfg)


if __name__ == "__main__":
    main()
