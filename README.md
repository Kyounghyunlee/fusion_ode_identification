# Fusion ODE Identification

A physics-informed neural ODE that learns reduced-order electron-temperature
transport for MAST discharges, with a **bistable cusp normal-form latent** that
gives the L/H confinement regime a precise mathematical meaning.

The PDE side is a conservative finite-volume diffusion operator with a
differentiable diffusivity profile `chi(rho, z)`; the closure is a small MLP
residual source. The scalar latent `z(t)` obeys

```
tau * dz/dt = a(u) + b * z - z^3        (b > 0)
```

driven by an affine map `a(u)` of the actuators only (`P_nbi`, `|Ip|`,
`nebar`). The two stable branches of the cusp are the L and H regimes;
transitions are saddle-node (fold) bifurcations of the latent dynamics, so
hysteresis is intrinsic and the distance of the drive from the folds is a
closed-form, control-ready transition margin. The D-alpha measurement never
enters the dynamics: it supervises a small learned observation head, which
keeps the identified regime state falsifiable instead of proxy-following.

Training and evaluation run end-to-end in JAX (float64) with a custom IMEX
theta-method integrator (Thomas-algorithm tridiagonal implicit diffusion +
explicit source/latent), so reverse-mode autodiff goes through the rollout.

Where to look:
- **Design and physics**: [docs/PHYSICS_INFORMED_TOKAMAK_ODE.md](docs/PHYSICS_INFORMED_TOKAMAK_ODE.md)
- **Code architecture**: [docs/code_architecture.md](docs/code_architecture.md)
- **Training pack format**: [docs/training_data_pack.md](docs/training_data_pack.md)
- **Paper draft** (dynamical-systems framing): [paper/main.tex](paper/main.tex)

## Setup (local workstation)

```bash
cd ~/Research/fusion_ode_identification
uv venv .venv --python 3.12 --native-tls
UV_NATIVE_TLS=1 uv pip install -p .venv -r requirements.txt
source scripts/env_local.sh   # TLS proxy CA bundle, PYTHONPATH, JAX x64, venv
```

## Data pipeline

1. **Download** MAST Level-2 groups (anonymous S3) for the shot list:

```bash
python preprocessing/download_data.py --shots 27574 27866 27759 ...
```

The current training set is 36 shots: the M8 session 27567-27587 plus
logbook-verified L-H transition shots found through the
[mastapp.site](https://mastapp.site) metadata API (including back-transition
and dithering cases: 27866, 28088, 27452, ...). Ohmic and non-transitioning
shots are kept deliberately as L-only negative examples.

2. **Build training packs** with strict QA and regime labels:

```bash
python -m preprocessing.build_training_pack --discover \
    --qa-grade fail \
    --qa-summary data/sanity_summary_v4.csv \
    --qa-plots data/plots/strict_v4
```

QA keeps only stable per-shot/per-rho `T_e` columns (median >= 50 eV, bounded
jumps/span, >= 4 stable edge columns). Regime labels (0 unknown / 1 L /
2 transition / 3 H) come from a dwell-time-constrained Otsu split of the
**lower envelope** of D-alpha, gated to the current flat-top, with an entry
sharpness test that rejects slow drifts. Detected transition times match the
session logbook where available (e.g. 27759: 0.244 s vs "247 ms").

3. **Inspect packs** (optional): `python scripts/inspect_data.py`

## Training

```bash
# production (CPU, ~2-3 h for 6000 steps)
JAX_PLATFORMS=cpu python train.py --config config/config_cusp.yaml

# quick debug run
JAX_PLATFORMS=cpu python train.py --config config/config_cusp.yaml --total-steps 50
```

Device note (RTX 5080 + 14-core CPU, float64): CPU and GPU tie at batch 8
(~1.4 s/step); XLA-GPU core-dumps compiling the gradient for batch >= 10, and
GeForce FP64 throughput makes larger GPU batches pointless — so CPU is the
default. Checkpoints (`*_best.eqx`, `*_best_ema.eqx`) are written whenever the
validation loss (which includes the regime and D-alpha terms) improves.
Resume with `--resume_ckpt <path> --resume_step_offset <step>`.

## Evaluation

```bash
python scripts/evaluate_model.py --config config/config_cusp.yaml --model-id cusp_run_v1
```

Writes to `logs/<model_id>/evaluation/`:
- `evaluation_report.json` — profile fit metrics plus the **quantitative L/H
  assessment**: per-shot accuracy / F1 / AUC / Brier of `p_H` against the
  labels, transition-time error, and cusp bifurcation diagnostics (fold
  amplitude, drive range, bistable fraction).
- `bifurcation_shot_<id>.npz` — drive, latent, saddle, and basin trajectories
  per shot (consumed by the paper figure script).
- per-shot fit plots.

## Paper

```bash
python paper/scripts/make_figures.py --model-id cusp_run_v1
cd paper && latexmk -pdf main.tex
```

## Dependency management

Edit `requirements.in`, then `uv pip compile requirements.in
--output-file requirements.txt` and `uv pip install -p .venv -r
requirements.txt`.
