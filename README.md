# Fusion ODE Identification

Data-driven identification of low-dimensional L-H transition dynamics for
MAST discharges, built for eventual real-time regime assessment and control.

A 1D conservative electron-temperature transport equation (finite-volume,
learned residual source) is coupled to a scalar regime latent governed by the
**full unfolding of the cubic normal form**

```
tau * dz/dt = a(u) + c1 * z + c2 * z^2 - z^3
```

with `c1`, `c2` unconstrained: whether the identified dynamics are bistable
(folds, hysteresis) or monostable (a sharp threshold without memory) is
decided by the data, not assumed. The drive `a(u)` is affine in the actuators
(`P_nbi`, `|Ip|`, `nebar`) in **fixed physical units** (MW, MA, 1e19 m^-3),
monotone in power — so the model is causal and the identified threshold is a
power threshold in watts. D-alpha never enters the dynamics; it supervises a
small learned observation head. The calibrated regime probability
`p_H = sigmoid(k*z + k0)` doubles as the barrier coordinate that suppresses
edge transport in the PDE.

Everything runs end-to-end in JAX (float64) through a differentiable IMEX
theta-method integrator.

- **Paper** (detailed model, data, and assessment description): [paper/main.tex](paper/main.tex)
- **Code architecture**: [docs/code_architecture.md](docs/code_architecture.md)
- **Training pack format**: [docs/training_data_pack.md](docs/training_data_pack.md)

## Setup

```bash
uv venv .venv --python 3.12 --native-tls
UV_NATIVE_TLS=1 uv pip install -p .venv -r requirements.txt
source scripts/env_local.sh   # TLS proxy CA bundle, PYTHONPATH, JAX x64, venv
```

## Pipeline

```bash
# 1. download MAST Level-2 data (anonymous S3)
python preprocessing/download_data.py --shots 27574 27759 ...

# 2. build strict training packs (QA + weak regime labels)
python -m preprocessing.build_training_pack --discover --qa-grade fail \
    --qa-summary data/sanity_summary.csv --qa-plots data/plots/qa

# 3. train (CPU, ~1.5 h for the 2000-step schedule)
JAX_PLATFORMS=cpu python train.py --config config/config.yaml

# 4. evaluate: fit metrics, regime classification, normal-form diagnostics
python scripts/evaluate_model.py --config config/config.yaml --model-id nf_run_v1

# 5. paper figures + build
python paper/scripts/make_figures.py --model-id nf_run_v1
cd paper && latexmk -pdf main.tex
```

The corpus is 63 quality-screened discharges (52 with a labeled L-H
transition, 11 L-only negatives including ohmic discharges), selected from
the public MAST catalogue (mastapp.site) by session-log annotations. Regime
labels come from a dwell-constrained bimodal split of the D-alpha lower
envelope with drift rejection; detected transition times match session-log
annotations to ~20 ms.

Device note (float64 pipeline): at the usable batch size the 14-core CPU and
an RTX 5080 are tied, and XLA-GPU is unstable compiling larger-batch
gradients — train on CPU.
