# Code Architecture

Single-device (jit + vmap), IMEX-only, uniform-grid, float64.

## Pipeline

1. `preprocessing/download_data.py` — MAST Level-2 NetCDF download (anonymous S3).
2. `preprocessing/build_training_pack.py` — strict QA + weak regime labels -> `*_torax_training.npz`.
3. `fusion_ode_identification.data.load_data` — pack loading, regridding, boundary traces, padding -> `ShotBundle`.
4. `train.py` — single-device training.
5. `scripts/evaluate_model.py` — fit metrics, regime classification, normal-form diagnostics, plots, and npz artifacts.
6. `paper/scripts/make_figures.py` — publication figures from the evaluation artifacts.

## Package Map

| Path | Role |
|---|---|
| `fusion_ode_identification/model.py` | `HybridField` (PDE backbone), `SourceNN`, `NormalFormLatent` |
| `fusion_ode_identification/data.py` | Pack loading, masks, boundary construction, fixed-scale controls |
| `fusion_ode_identification/loss.py` | Differentiable rollout loss (profile + regime + observation terms) |
| `fusion_ode_identification/imex_solver.py` | Fixed-substep IMEX theta-method, Thomas tridiagonal solve |
| `fusion_ode_identification/interp.py` | JAX linear interpolation |
| `fusion_ode_identification/types.py` | `ShotBundle`, `LossCfg`, `IMEXConfig` |
| `fusion_ode_identification/regime_metrics.py` | AUC/F1/Brier, transition timing, normal-form fold/basin diagnostics |

## Model

- Backbone: conservative FVM diffusion with `chi(rho, z)` (edge value suppressed
  by `p_H`), Neumann axis, Dirichlet edge trace; residual source MLP
  (tanh, zero-initialized output).
- Latent: `NormalFormLatent`, `tau * dz/dt = a(u) + c1*z + c2*z^2 - z^3`;
  drive affine in physically scaled actuators (monotone in power);
  `p_H = sigmoid(k*z + k0)`; learned D-alpha head
  `sigmoid(MLP(p_H, controls, Te_edge, ne_edge))`.
- All control inputs use `CONTROL_SCALES` (MW, MA, 1e19 m^-3, a.u.) — no
  per-shot statistics; inference is causal.
- Bistability is a *result*: bistable iff `c2^2 + 3*c1 > 0`
  (`regime_metrics.normal_form_diagnostics` reports folds/margins, NaN when
  monostable).

## Loss (config/config.yaml)

Pseudo-Huber profile misfit on measured samples in the reliable annulus
(inverse-coverage weighted) + source magnitude penalty + latent smoothness /
magnitude + regime BCE against weak labels (`lambda_regime`) + observation
head misfit (`lambda_dalpha`). The profile term is O(700); regime/observation
weights of 20 keep those terms at a few percent, which the small latent needs.
Validation loss includes all terms, so checkpoint selection reflects regime
quality.

## Regime labels (pack builder)

`estimate_regime_labels`: rolling 15th-percentile lower envelope of D-alpha
(ELM-robust), flat-top gate on `|Ip|`/`nebar`, Otsu split with separation
guards, >= 20 ms dwell, L-lead requirement, 25 ms entry-sharpness test
(rejects drifts). Codes: 0 unknown, 1 L, 2 transition window, 3 H. Multiple
H segments (back-transitions) allowed.

## Smoke workflow

```bash
JAX_PLATFORMS=cpu python train.py --config config/config.yaml --total-steps 20
python scripts/evaluate_model.py --config config/config.yaml --model-id nf_run_v1
```
