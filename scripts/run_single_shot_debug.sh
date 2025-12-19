#!/usr/bin/env bash
set -euo pipefail

# Single-shot sweep: short Adam-only runs per pack, no L-BFGS, no restarts.
# Uses the same module/venv setup as run_training_gpu.sh.

module load Python/3.11.5-GCCcore-13.2.0 || true
module load CUDA/12.3.0 || true
module load cuDNN/8.9.7.29-CUDA-12.3.0 || true

source venv/bin/activate
export LD_LIBRARY_PATH="$(pwd)/venv/lib/nvidia_libs:$(pwd)/venv/lib/python3.11/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"

BASE_CFG="config/config.yaml"
SHOTS=$(ls data/*_torax_training.npz 2>/dev/null | sed -E 's#.*/([0-9]+)_torax_training\.npz#\1#' | sort -n)

if [ -z "${SHOTS}" ]; then
  echo "No *_torax_training.npz files found under data/." >&2
  exit 1
fi

echo "Running short single-shot sweeps for shots: ${SHOTS}";

for SID in ${SHOTS}; do
  echo "=== Shot ${SID} ==="
  TMP_CFG=$(mktemp)
  SID_ENV="${SID}" BASE_CFG="${BASE_CFG}" TMP_CFG="${TMP_CFG}" \
  python - <<'PY'
import os, yaml
sid = int(os.environ['SID_ENV'])
base = os.environ['BASE_CFG']
out = os.environ['TMP_CFG']
with open(base) as f:
    cfg = yaml.safe_load(f)
cfg.setdefault('data', {})['shots'] = [sid]
tr = cfg.setdefault('training', {})
tr['num_restarts'] = 1
tr['total_steps'] = 10
tr['warmup_steps'] = min(tr.get('warmup_steps', 1), 3)
tr['batch_size'] = 2  # divisible by 2 GPUs; adjust if device count differs
tr['ema_decay'] = 0.0
tr['lbfgs_finetune'] = False
tr['lbfgs_maxiter'] = 0
tr['lbfgs_history'] = 0
tr['lbfgs_batch_shots'] = 0
tr['lbfgs_tol'] = 1.0
with open(out, 'w') as f:
    yaml.safe_dump(cfg, f)
PY
  ./scripts/run_training_gpu.sh --config "${TMP_CFG}" --debug_one_shot "${SID}" --throw || { echo "Shot ${SID} failed"; rm -f "${TMP_CFG}"; exit 1; }
  rm -f "${TMP_CFG}"
  echo "Shot ${SID} done"
done

echo "Sweep complete."
