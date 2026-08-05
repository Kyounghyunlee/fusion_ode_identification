#!/usr/bin/env bash
# Experiment queue (CPU). Runs PAR jobs concurrently with pinned thread
# counts; each run early-stops on the grouped-validation plateau.
# Usage: scripts/run_experiments.sh [PAR]
set -u
cd "$(dirname "$0")/.."
PAR="${1:-3}"
THREADS=$(( $(nproc) / PAR ))
PY=.venv/bin/python

run() {
  local id="$1"; shift
  if [ -f "models/$id/train_summary.json" ]; then echo "== $id done, skip"; return; fi
  echo "== start $id : $*"
  PYTHONPATH="$PWD" JAX_PLATFORMS=cpu \
  OMP_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" \
  XLA_FLAGS="--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=$THREADS" \
    $PY train.py --config config/config.yaml --model-id "$id" "$@" > "logs/exp_$id.log" 2>&1
  PYTHONPATH="$PWD" JAX_PLATFORMS=cpu OMP_NUM_THREADS="$THREADS" \
    $PY scripts/evaluate_model.py --config config/config.yaml --model-id "$id" \
      --role val --no-plots >> "logs/exp_$id.log" 2>&1
  echo "== done  $id"
}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 20; done; }

# --- Stage 1: decisive comparison (drive set x topology x seed) ---
for s in 0 1 2; do
  throttle; run "e_ext_free_s$s"  --seed "$s" --override model.drive_set=extended & sleep 2
  throttle; run "e_bas_free_s$s"  --seed "$s" --override model.drive_set=basic & sleep 2
  throttle; run "e_ext_mono_s$s"  --seed "$s" --override model.drive_set=extended --override model.beta_mode=nonpositive & sleep 2
done
wait

# --- Stage 2: ablations on the extended/free configuration ---
throttle; run "e_ext_nochi_s0"    --seed 0 --override model.drive_set=extended --override model.delta_chi_off=true & sleep 2
throttle; run "e_ext_nosrc_s0"    --seed 0 --override model.drive_set=extended --override model.source_off=true & sleep 2
throttle; run "e_ext_nodalpha_s0" --seed 0 --override model.drive_set=extended --override training.lambda_dalpha=0.0 & sleep 2
throttle; run "e_ext_noregime_s0" --seed 0 --override model.drive_set=extended --override training.lambda_regime=0.0 & sleep 2
wait

echo "QUEUE COMPLETE"
