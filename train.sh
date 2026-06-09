#!/usr/bin/env bash
# Train BPFlow.  Usage: bash train.sh   (override: bash train.sh --config ...)
set -euo pipefail
cd "$(dirname "$0")"
python -m bpflow.train --config bpflow/config/gpu.yaml "$@"
