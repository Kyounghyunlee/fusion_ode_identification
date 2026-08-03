# Code Architecture

This repository trains and evaluates a physics-informed electron-temperature
transport ROM from strict MAST training packs. The active branch is IMEX-only,
uniform-grid-only, and uses a small scalar input set.

## Pipeline

1. `preprocessing/download_data.py` downloads raw NetCDF diagnostics.
2. `preprocessing/build_training_pack.py` converts raw diagnostics into strict
   `*_torax_training.npz` packs.
3. `fusion_ode_identification.data.load_data` loads packs, broadcasts scalar
   density, builds boundary conditions, pads time, and returns `ShotBundle`.
4. `train_tokamak_ode_hpc.py` trains `HybridField` with JAX/Equinox/Optax.
5. `scripts/evaluate_model.py` loads the preferred checkpoint, runs full
   trajectories, writes metrics, and regenerates plots.

## Package Map

| Path | Role |
|---|---|
| `fusion_ode_identification/model.py` | `HybridField`, `SourceNN`, diffusivity, latent dynamics |
| `fusion_ode_identification/data.py` | Pack loading, interpolation, padding, masks, scalar-density broadcast |
| `fusion_ode_identification/loss.py` | IMEX rollout loss, D-alpha auxiliary loss, diagnostics |
| `fusion_ode_identification/imex_solver.py` | Fixed-step theta-method solver |
| `fusion_ode_identification/interp.py` | JAX linear interpolation helper |
| `fusion_ode_identification/types.py` | Shared `NamedTuple` schemas |
| `preprocessing/build_training_pack.py` | Strict pack builder and QA artifacts |
| `scripts/evaluate_model.py` | Evaluation reports and plots |
| `scripts/smoke_*.py` | Regression smoke checks |

## Current Data Contract

Active controls are:

```python
CONTROL_NAMES = ["P_nbi", "Ip", "nebar", "D_alpha"]
```

`nebar` is not the summary-file line average. It is the profile-derived scalar
density saved as `ne_profile_scalar` and `ne_profile_scalar_ts`. The loader
broadcasts the Thomson-time scalar over rho so `ShotBundle.ne_vals` retains shape
`(shots,time,rho)` while carrying only one density value per time.

Removed controls:

```text
S_gas S_rec S_nbi
```

The D-alpha trace now enters directly as a scalar control and as the main latent
transition evidence.

## Data Loading Details

`load_data(config)` returns:

```python
bundle, rho_rom, rho_cap, obs_idx
```

Important `ShotBundle` fields:

| Field | Meaning |
|---|---|
| `ts_t` | Padded Thomson time base |
| `ts_Te` | Filled temperature array used for numeric rollout/loss |
| `ts_Te_raw` | NaN-preserving temperature array used for measured-only plotting |
| `mask` | Strict measured `T_e` support after per-rho QA |
| `reliable_mask` | Corpus reliable annulus mask computed after regridding |
| `Te_edge` | Dirichlet boundary trace from outermost observed `T_e` |
| `ctrl_vals` | Four scalar controls on the profile time base |
| `ne_vals` | Broadcast representative scalar density |
| `dalpha_ts` | D-alpha on the profile time base |
| `z0` | Per-shot latent initial condition |
| `t_len` | Valid unpadded length |

The runtime grid is uniform: `rho_rom = linspace(0, 1, N)`. The final node is a
Dirichlet boundary and is excluded from data-loss supervision.

## Model

`HybridField` combines:

1. Conservative finite-volume diffusion using `chi(rho,z)` and `Vprime`.
2. A residual source MLP evaluated pointwise at `(rho, Te, ne, controls, z)`.
3. A scalar latent ODE.

The active latent design is `barrier_v1`. Its barrier coordinate is
`z_b = sigmoid(zeta)`. Higher `z_b` means stronger H-mode/barrier evidence and
lower edge diffusivity through the existing diffusivity profile.

Latent features have size 6:

```text
1 - norm(D_alpha)
-d/dt norm(D_alpha)
+d/dt norm(Te_edge)
+d/dt norm(ne_edge)
norm(P_nbi)
norm(Ip)
```

The feature smoother uses short windows so transitions in these sub-second shots
are not averaged away. The D-alpha auxiliary head predicts `norm(D_alpha)` as
`1 - z_b`, directly tying the latent coordinate to the measured L-H proxy.

## Loss

`loss.py` computes a weighted pseudo-Huber data term on masked interior radii,
gated by `reliable_mask`, plus:

| Term | Meaning |
|---|---|
| `lambda_src` | Residual source magnitude penalty |
| `lambda_z` | Latent smoothness penalty |
| `lambda_zreg` | Latent magnitude penalty |
| `lambda_regime`, `lambda_pH` | Heuristic regime BCE supervision |
| `lambda_dalpha` | D-alpha auxiliary supervision |

For `config_v3*.yaml`, `lambda_dalpha = 10.0` so the latent is materially
supervised by D-alpha during training.

## Evaluation

`scripts/evaluate_model.py`:

- Prefers `_best_ema.eqx`, then `_best.eqx`, then `_finetuned.eqx`.
- Clears stale `shot_*.png` files before writing new plots.
- Uses measured-only heatmaps where unobserved regions are white.
- Reports whole-mask, annulus, D-alpha, latent, and physics consistency metrics.
- Overlays `1 - norm(D_alpha)` and the barrier latent on overview/latent plots.

## Smoke Workflow

On the login node use CPU smoke tests:

```bash
JAX_PLATFORMS=cpu python scripts/smoke_time_padding_strict.py
JAX_PLATFORMS=cpu python scripts/smoke_valid_window.py --config config/config_v3_debug.yaml
JAX_PLATFORMS=cpu python scripts/smoke_padding_freeze.py --config config/config_v3_debug.yaml
JAX_PLATFORMS=cpu python scripts/smoke_diffusion_sanity.py --config config/config_v3_debug.yaml --shot 27578
python scripts/smoke_checkpoint_selection.py
JAX_PLATFORMS=cpu python -m train_tokamak_ode_hpc --config config/config_v3_debug.yaml --device cpu
JAX_PLATFORMS=cpu python scripts/evaluate_model.py --config config/config_v3_debug.yaml --model-id production_run_v3_debug --data-check
```

Validated strict-pack smoke state:

```text
18 packs loaded
ShotBundle Te shape = (18, 102, 65)
control shape = (18, 102, 4)
density broadcast check = true
training/evaluation ok fraction = 1.000
```

## GPU Training

Use the wrapper on a compute node:

```bash
./scripts/run_training_gpu.sh --config config/config_v3.yaml
```

Do not run full GPU training on the login node. CPU smoke is fine there; real
training should run under an allocated GPU session.
