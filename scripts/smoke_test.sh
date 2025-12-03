#!/usr/bin/env bash
set -euo pipefail

ENV="${PWD}/torax_env/bin/python"
SHOT=${SHOT:-30421}
PACK="data/${SHOT}_torax_training.npz"
OUTDIR="outputs/${SHOT}_torax_training_4dvar"

echo ">>> Smoke test: eq_only training on shot ${SHOT}"
$ENV -m scripts.train_4dvar \
  "$PACK" \
  --steps 20 \
  --lr 5e-3 \
  --use-ne \
  --mode eq_only

echo ">>> Tail of loss.csv"
 tail -n 10 "${OUTDIR}/loss.csv" || { echo "loss.csv missing"; exit 1; }

if grep -qi 'nan' "${OUTDIR}/loss.csv"; then
  echo "ERROR: NaN encountered in loss.csv"
  exit 1
fi

echo ">>> Running simulation interface (simulate_shot.py)"
$ENV -m scripts.simulate_shot "$PACK" --norm-stats "${OUTDIR}/normalization.json" --use-ne

echo ">>> Smoke test passed."
