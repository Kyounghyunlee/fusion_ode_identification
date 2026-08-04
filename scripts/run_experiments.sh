#!/usr/bin/env bash
# Sequential experiment queue (CPU). Each run early-stops on val plateau.
# Usage: scripts/run_experiments.sh
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD" JAX_PLATFORMS=cpu
PY=.venv/bin/python

run() {
  local id="$1"; shift
  if [ -f "models/$id/train_summary.json" ]; then
    echo "== $id already done, skipping"
    return
  fi
  echo "== $id : $*"
  $PY train.py --config config/config.yaml --model-id "$id" "$@" \
      > "logs/exp_$id.log" 2>&1
  $PY scripts/evaluate_model.py --config config/config.yaml --model-id "$id" \
      --role val --no-plots >> "logs/exp_$id.log" 2>&1
}

# E1/E2: free vs constrained-monostable, 3 screening seeds each
for s in 0 1 2; do
  run "v2_free_s$s" --seed "$s"
  run "v2_mono_s$s" --seed "$s" --override model.beta_mode=nonpositive
done

# E3-E6: ablations (seed 0)
run "v2_nochi_s0"    --seed 0 --override model.delta_chi_off=true
run "v2_nosrc_s0"    --seed 0 --override model.source_off=true
run "v2_nodalpha_s0" --seed 0 --override training.lambda_dalpha=0.0
run "v2_noregime_s0" --seed 0 --override training.lambda_regime=0.0

echo "QUEUE COMPLETE"
