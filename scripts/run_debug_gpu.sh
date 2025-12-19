#!/usr/bin/env bash
set -euo pipefail

module load Python/3.11.5-GCCcore-13.2.0 || true
module load CUDA/12.3.0 || true
module load cuDNN/8.9.7.29-CUDA-12.3.0 || true

source venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/venv/lib/nvidia_libs:$(pwd)/venv/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

SID="${1:?usage: $0 <shot_id>}"
python scripts/debug_shot.py --config config/config.yaml --shot "${SID}"
