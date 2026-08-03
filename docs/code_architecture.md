# Code Architecture

This repository trains and evaluates a physics-informed electron-temperature
transport ROM from strict MAST training packs, with a bistable cusp latent for
quantitative L/H regime assessment. The active branch is IMEX-only,
uniform-grid-only, single-device (jit + vmap), and float64.

## Pipeline

1. `preprocessing/download_data.py` downloads raw NetCDF diagnostics
   (anonymous S3, MAST Level-2 Zarr).
2. `preprocessing/build_training_pack.py` converts raw diagnostics into strict
   `*_torax_training.npz` packs, including dwell-constrained L/H regime labels.
3. `fusion_ode_identification.data.load_data` loads packs, broadcasts scalar
   density, builds boundary conditions, pads time, and returns `ShotBundle`.
4. `train.py` trains `HybridField` with JAX/Equinox/Optax on one device.
5. `scripts/evaluate_model.py` loads the preferred checkpoint, runs full
   trajectories, writes fit metrics, regime-classification metrics, and
   bifurcation diagnostics, and regenerates plots.
6. `paper/scripts/make_figures.py` turns evaluation artifacts into the paper
   figures.

## Package Map

| Path | Role |
|---|---|
| `fusion_ode_identification/model.py` | `HybridField`, `SourceNN`, diffusivity, latent designs (`cusp`, `barrier_v1`, `cubic`) |
| `fusion_ode_identification/data.py` | Pack loading, interpolation, padding, masks, scalar-density broadcast |
| `fusion_ode_identification/loss.py` | IMEX rollout loss + regime/D-alpha terms, diagnostics |
| `fusion_ode_identification/imex_solver.py` | Fixed-step theta-method solver |
| `fusion_ode_identification/interp.py` | JAX linear interpolation helper |
| `fusion_ode_identification/types.py` | Shared `NamedTuple` schemas |
| `fusion_ode_identification/regime_metrics.py` | Quantitative L/H metrics and cusp bifurcation diagnostics |
| `preprocessing/build_training_pack.py` | Strict pack builder, QA artifacts, regime labeler |
| `train.py` | Single-device training entrypoint |
| `scripts/evaluate_model.py` | Evaluation reports and plots |
| `scripts/smoke_*.py`, `scripts/check_bc.py` | Regression smoke checks |

## Data Contract

Active controls:

```python
CONTROL_NAMES = ["P_nbi", "Ip", "nebar", "D_alpha"]
```

`nebar` is the profile-derived scalar density (`ne_profile_scalar`), not the
summary-file line average. The loader broadcasts it over rho so
`ShotBundle.ne_vals` keeps shape `(shots, time, rho)`.

Regime labels in the packs: `0` unknown/no plasma, `1` L, `2` transition
window, `3` H. They are produced by `estimate_regime_labels`: rolling
15th-percentile lower envelope of D-alpha (ELM-robust), flat-top gating on
`|Ip|` and `nebar`, Otsu split with separation guards, >= 20 ms dwell, an
L-lead requirement (no H at gate opening), and a 25 ms entry-sharpness test
that rejects slow drifts. Multiple H segments (back-transitions) are allowed.

## Model

`HybridField` combines:

1. Conservative finite-volume diffusion using `chi(rho, z)` and `Vprime`.
2. A residual source MLP evaluated pointwise at `(rho, Te, ne, controls, z)`.
3. A scalar latent ODE (`model.latent_design`).

### Latent designs

- **`cusp` (active)** — `CuspLatentDynamics`:
  `tau * dz/dt = a(u) + b z - z^3` with `b > 0`, drive
  `a(u) = softclip(w . u[:3] + w0)` over normalized `P_nbi, Ip, nebar` only.
  Bistable for `|a| < a_fold = 2 (b/3)^{3/2}`; regimes are the two stable
  branches, transitions are fold crossings, hysteresis is intrinsic. The
  barrier coordinate is `z_b = sigmoid(3 z / sqrt(b))`; the regime logit is
  `k z / sqrt(b)`. D-alpha is predicted by a learned head
  `sigmoid(MLP(z_b, controls, Te_edge, ne_edge))` and never drives the latent.
- `barrier_v1` (legacy) — relaxation toward a D-alpha-derived target; kept for
  comparison. Its aux head is the parameter-free `1 - z_b`, which pins the
  latent to the proxy.
- `cubic` (legacy) — symmetric cubic driven by normalized controls.

## Loss

Weighted pseudo-Huber data term on masked interior radii (gated by
`reliable_mask`), plus:

| Term | Meaning |
|---|---|
| `lambda_src` | Residual source magnitude penalty |
| `lambda_z`, `lambda_zreg` | Latent smoothness / magnitude penalties |
| `lambda_regime` (+ `lambda_pH`) | Regime BCE against pack labels (clean L/H samples only) |
| `lambda_dalpha` | Observation-head misfit against normalized D-alpha |

For `config_cusp.yaml`: `lambda_regime = 5e-2`, `lambda_dalpha = 1.0`. The
validation loss used for checkpoint selection includes the regime and D-alpha
terms, so checkpoints are selected for regime quality too.

## Evaluation

`scripts/evaluate_model.py` reports, per shot and pooled:

- profile fit (MSE / MAE, whole-mask and reliable annulus),
- **regime classification**: accuracy, F1, AUC, Brier of
  `p_H = sigmoid(regime_logit)` against clean-L/H labels,
- **transition timing**: first sustained `p_H > 0.5` upcrossing vs label,
- **bifurcation diagnostics** (cusp only): `b`, `tau`, `a_fold`, drive range,
  bistable fraction, basin membership; per-shot
  `bifurcation_shot_<id>.npz` artifacts.

## Smoke Workflow

```bash
JAX_PLATFORMS=cpu python scripts/smoke_time_padding_strict.py
JAX_PLATFORMS=cpu python scripts/smoke_diffusion_sanity.py --config config/config_cusp.yaml --shot 27578
JAX_PLATFORMS=cpu python scripts/check_bc.py --config config/config_cusp.yaml --shot 27567
JAX_PLATFORMS=cpu python train.py --config config/config_cusp.yaml --total-steps 20
```
