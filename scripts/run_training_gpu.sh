#!/usr/bin/env bash
set -euo pipefail

# Load modules (HPC environment)
# Make failures visible but non-fatal so we still run on plain hosts.
module load Python/3.11.5-GCCcore-13.2.0 && echo "[module] Python/3.11.5 loaded" || echo "[module] Python/3.11.5 missing or failed"
module load CUDA/12.3.0 && echo "[module] CUDA/12.3.0 loaded" || echo "[module] CUDA/12.3.0 missing or failed"
module load cuDNN/8.9.7.29-CUDA-12.3.0 && echo "[module] cuDNN/8.9.7.29-CUDA-12.3.0 loaded" || echo "[module] cuDNN/8.9.7.29-CUDA-12.3.0 missing or failed"

# 1. Activate venv
source venv/bin/activate

# Tame XLA/TF logging noise and disable async collectives to reduce rendezvous chatter
export TF_CPP_MIN_LOG_LEVEL=1
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_async_collectives=false"

# 2. Force JAX to use our local pip-installed NVIDIA libs (CUDA 11 compat mode)
# Guard the paths to avoid polluting LD_LIBRARY_PATH with non-existent dirs.
LD_PATHS=()
if [ -d "$(pwd)/venv/lib/nvidia_libs" ]; then
	LD_PATHS+=("$(pwd)/venv/lib/nvidia_libs")
fi
if [ -d "$(pwd)/venv/lib/python3.11/site-packages/nvidia/nccl/lib" ]; then
	LD_PATHS+=("$(pwd)/venv/lib/python3.11/site-packages/nvidia/nccl/lib")
fi
if [ ${#LD_PATHS[@]} -gt 0 ]; then
	export LD_LIBRARY_PATH="$(IFS=:; echo "${LD_PATHS[*]}")${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

# Force CUDA backend to avoid ROCm/TPU probes
export JAX_PLATFORMS=cuda

run_with_filtered_stderr() {
	# Filter a known-noisy rendezvous warning while preserving everything else.
	"$@" 2> >(grep -vE 'external/xla/xla/service/rendezvous\.cc:(31|36)' >&2)
}

if [[ "${1:-}" == "--python" ]]; then
	shift
	echo "Starting Python on GPU..."
	run_with_filtered_stderr ./venv/bin/python "$@"
	exit $?
fi

# Convenience: allow `./scripts/run_training_gpu.sh -c '...'` and `... -m module`.
# (These are python flags, not train_tokamak_ode_hpc.py flags.)
if [[ "${1:-}" == "-c" || "${1:-}" == "-m" ]]; then
	echo "Starting Python on GPU..."
	run_with_filtered_stderr ./venv/bin/python "$@"
	exit $?
fi

echo "Starting training on GPU..."
run_with_filtered_stderr ./venv/bin/python -m train_tokamak_ode_hpc "$@"
