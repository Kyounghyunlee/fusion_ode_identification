# Training Data Pack

This document describes the current `schema_version=3` training packs produced by
`preprocessing/build_training_pack.py`. The current policy is deliberately strict:
we keep only trustworthy measured `T_e` radial signals inside each shot, and we
prefer fewer packs over training on damaged channels.

## Canonical Build

From the repository root:

```bash
python -m preprocessing.build_training_pack \
  --discover \
  --qa-grade fail \
  --qa-summary data/sanity_summary_v3.csv \
  --qa-plots data/plots/strict_iter3
```

The validated strict rebuild produced 18 packs:

```text
27567 27568 27569 27570 27573 27574 27575 27576 27577
27578 27579 27580 27581 27582 27584 27585 27586 27587
```

Shots `27571`, `27572`, and `27594` are not written by the strict defaults
because too few stable edge rho columns survive. This is intentional.

## Raw Inputs

Required per shot under `data/<shot>/`:

| File | Use |
|---|---|
| `equilibrium.nc` | Flux coordinate, geometry metadata, and `Vprime` |
| `thomson_scattering.nc` | Electron temperature and density profiles |
| `summary.nc` | Scalar summary traces such as `Ip`, `P_nbi`, and optional diagnostics |

Optional per shot:

| File | Use |
|---|---|
| `d_alpha.nc` | Preferred D-alpha sidecar |
| `spectrometer_visible.nc` | Fallback D-alpha source |

`gas_injection.nc` is no longer part of the training control vector.

## Strict Time-Slice QA

Before radial channel screening, Thomson time slices are cleaned with:

| Parameter | Default | Meaning |
|---|---:|---|
| `--time-min-channels` | `8` | Minimum finite Thomson channels in a time slice |
| `--time-min-rho-span` | `0.08` | Minimum radial span before the slice is usable |
| `--qa-mad-window` | `11` | Rolling MAD window |
| `--qa-k-mad` | `6.0` | Spike rejection multiplier |

This removes impossible values and local spikes without rejecting a whole shot
just because a few slices are bad.

## Per-Rho `T_e` Signal QA

The most important filter is per-shot and per-rho. A rho column is retained only
if its own measured `T_e(t)` behavior is stable enough.

| Parameter | Default | Meaning |
|---|---:|---|
| `--rho-hard-min` | `0.72` | Always drop observed columns below this rho |
| `--rho-low-cutoff` | `0.78` | Low-rho zone checked for flat artifacts |
| `--rho-outer-min` | `0.78` | Edge-support threshold for write eligibility |
| `--rho-min-col-coverage` | `0.35` | Minimum fraction of time slices observed |
| `--rho-flat-span-ev` | `25.0` | Low-rho flat-signal absolute span threshold |
| `--rho-flat-rel-span` | `0.08` | Low-rho flat-signal relative span threshold |
| `--rho-min-te-median` | `50.0` | Drop cold near-zero retained columns |
| `--rho-max-te-rel-jump-p95` | `0.60` | Drop jumpy columns by 95th percentile relative jump |
| `--rho-max-te-rel-span` | `1.80` | Drop excessive robust relative span columns |
| `--rho-min-edge-cols` | `4` | Minimum stable edge columns to write a pack |
| `--rho-min-kept-cols` | `4` | Minimum stable columns to keep per written shot |

The validated rebuild retained 110 trusted radial columns across 18 packs. Global
retained support starts at `rho=0.859375`; no core or low-rho filled profile is
treated as supervised data.

Useful QA artifacts:

```text
data/sanity_summary_v3.csv
data/plots/strict_iter3/rho_coverage_heatmap.png
data/plots/strict_iter3/rho_086_kept_columns.png
```

## Scalar Controls

The active model control vector is intentionally small:

```python
CONTROL_NAMES = ["P_nbi", "Ip", "nebar", "D_alpha"]
```

The pack builder sanitizes these signals before writing:

| Signal | Treatment |
|---|---|
| `P_nbi` | Interpolated to summary time, non-finite values filled, clipped nonnegative |
| `Ip` | Signed raw trace is saved as `Ip_signed_raw`; active `Ip` is absolute magnitude |
| `nebar` | Replaced by the representative profile-density scalar described below |
| `D_alpha` | Summed from D-alpha channels, filled, clipped nonnegative |

The old proxy controls `S_gas`, `S_rec`, and `S_nbi` have been removed from pack
generation and from the model input. `P_rad` may still be stored as an optional
diagnostic, but it is not part of `CONTROL_NAMES`.

## Representative Density Scalar

Radial density is not used as a true vector control. Instead, each Thomson time
slice gets one representative density value computed from the overlap of stable
`T_e` columns and valid `n_e` columns.

Defaults:

| Parameter | Default | Meaning |
|---|---:|---|
| `--density-rho-min` | `0.78` | Lower preferred rho bound |
| `--density-rho-max` | `0.98` | Upper preferred rho bound |
| `--density-min-cols` | `3` | Minimum valid columns for the preferred average |

The builder writes:

| Key | Shape | Meaning |
|---|---:|---|
| `ne_profile_scalar_ts` | `(n_ts,)` | Profile-derived scalar density on Thomson time |
| `ne_profile_scalar_mask_ts` | `(n_ts,)` | Whether preferred density support was available |
| `ne_profile_scalar` | `(n_summary,)` | Same scalar interpolated to summary time |
| `nebar` | `(n_summary,)` | Active scalar density control, equal to `ne_profile_scalar` |

At load time, `fusion_ode_identification.data.load_data` broadcasts this scalar
back across rho to satisfy the existing model API. This keeps density simple and
prevents noisy radial density structure from pretending to be trustworthy input.

## Core Pack Keys

| Key | Shape | Notes |
|---|---:|---|
| `t` | `(n_summary,)` | Summary/control time base |
| `t_ts` | `(n_ts,)` | Thomson time base |
| `rho` | `(n_rho,)` | Pack rho grid, normally 65 nodes |
| `Te` | `(n_ts,n_rho)` | Invalid entries are `0.0`; use `Te_mask` for support |
| `ne` | `(n_ts,n_rho)` | Raw profile density with invalid entries zeroed |
| `Te_mask` | `(n_ts,n_rho)` | Strict per-rho measured support |
| `ne_mask` | `(n_ts,n_rho)` | Density support after the retained-rho policy |
| `Vprime` | `(n_rho,)` | Geometry for finite-volume diffusion |
| `P_nbi`, `Ip`, `nebar`, `D_alpha` | `(n_summary,)` | Active scalar controls |
| `D_alpha_channels` | `(n_summary,n_channels)` | Nonnegative channel traces when available |
| `regime`, `regime_score` | `(n_summary,)` | Heuristic L/transition/H labels for diagnostics/loss |
| `transition_time` | `(1,)` | Estimated transition time |

## QA Metadata Keys

The builder writes enough metadata to audit every retained column:

| Key | Meaning |
|---|---|
| `rho_signal_keep_mask` | Boolean retained-rho mask |
| `rho_signal_drop_reason` | Per-rho reason string |
| `rho_signal_Te_col_coverage` | Per-rho measured `T_e` coverage |
| `rho_signal_Te_span_eV` | Robust `T_e` span |
| `rho_signal_Te_median_eV` | Robust `T_e` median |
| `rho_signal_Te_rel_span` | Robust span divided by scale |
| `rho_signal_Te_rel_jump_p95` | 95th percentile relative adjacent jump |
| `rho_quality_*` | Threshold values used for the pack |
| `Te_mask_mean`, `ne_mask_mean` | Overall retained coverage |

## Validation Snapshot

The strict rebuild was validated with:

```text
18 packs
110 retained Te columns
min retained rho = 0.859375
max retained Te relative jump p95 = 0.5986
max retained Te relative span = 1.7827
min retained Te median = 56.8 eV
no non-finite or negative active scalar controls
```

Smoke-tested commands are listed in [code_architecture.md](code_architecture.md).
