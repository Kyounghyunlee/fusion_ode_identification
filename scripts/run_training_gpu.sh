#!/usr/bin/env bash
set -euo pipefail

# Load modules (HPC environment)
# We use || true to avoid failing if 'module' is not defined in the subshell
module load Python/3.11.5-GCCcore-13.2.0 || true
module load CUDA/12.3.0 || true
module load cuDNN/8.9.7.29-CUDA-12.3.0 || true

# 1. Activate venv
source venv/bin/activate

# 2. Force JAX to use our local pip-installed NVIDIA libs (CUDA 11 compat mode)
# We also need to ensure NCCL is found. It might be in venv/lib/python3.11/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=$(pwd)/venv/lib/nvidia_libs:$(pwd)/venv/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}

echo "Starting training on GPU..."
./venv/bin/python -m scripts.train_ode_physics_manifold_hpc "$@"
