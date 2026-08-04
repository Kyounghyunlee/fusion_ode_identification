"""
Evaluation Script for Physics-Consistent Manifold Model
Loads a trained model and generates comparison plots (Model vs Observation).
"""

import sys

import os
import json
import argparse
from typing import List, NamedTuple

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import equinox as eqx


def _existing_with_mtime(paths: List[str]):
    out = []
    for p in paths:
        if os.path.exists(p):
            try:
                out.append((p, float(os.path.getmtime(p))))
            except Exception:
                out.append((p, float("nan")))
    return out


def _select_checkpoint_by_preference(
    preference: List[str],
    best_ema: List[str],
    best: List[str],
    finetuned: List[str],
) -> str:
    pref = [str(x).strip().lower() for x in preference if str(x).strip()]
    if not pref:
        pref = ["best_ema", "best", "finetuned", "newest"]

    groups = {
        "best_ema": best_ema,
        "best": best,
        "finetuned": finetuned,
    }

    all_candidates = best_ema + best + finetuned
    existing_all = _existing_with_mtime(all_candidates)
    if existing_all:
        existing_sorted = sorted(existing_all, key=lambda x: x[1], reverse=True)
        print("[eval] Existing candidate checkpoints (newest first):")
        for p, mtime in existing_sorted:
            print(f"  - {p} (mtime={mtime:.0f})")

    for token in pref:
        if token in groups:
            for p in groups[token]:
                if os.path.exists(p):
                    print(f"[eval] Selected checkpoint by preference ({token}): {p}")
                    return p
        elif token == "newest":
            if not existing_all:
                continue
            selected = sorted(existing_all, key=lambda x: x[1], reverse=True)[0][0]
            print(f"[eval] Selected checkpoint by fallback (newest): {selected}")
            return selected
        else:
            raise ValueError(f"Unknown checkpoint preference token: {token!r}")

    raise FileNotFoundError(f"No checkpoint found. Tried: {all_candidates}")


def _sanitize_name(name: str) -> str:
    import re

    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name

from fusion_ode_identification.model import build_hybrid_model
from fusion_ode_identification.data import load_data
from fusion_ode_identification.types import ShotBundle, IMEXConfig
from fusion_ode_identification.imex_solver import IMEXIntegrator
from fusion_ode_identification.interp import LinearInterpolation
from fusion_ode_identification.regime_metrics import (
    regime_classification_metrics,
    transition_time_error,
    normal_form_diagnostics,
)

jax.config.update("jax_enable_x64", True)


class EvalBundle(NamedTuple):
    ts_t: jnp.ndarray
    ts_Te: jnp.ndarray
    ts_Te_raw: jnp.ndarray
    mask: jnp.ndarray
    reliable_mask: jnp.ndarray
    Te0: jnp.ndarray
    z0: float
    shot_id: int
    rho: jnp.ndarray
    Vprime: jnp.ndarray
    ctrl_t: jnp.ndarray
    ctrl_vals: jnp.ndarray
    ctrl_means: jnp.ndarray
    ctrl_stds: jnp.ndarray
    regime_ts: jnp.ndarray
    ne_vals: jnp.ndarray
    Te_edge: jnp.ndarray
    dalpha_ts: jnp.ndarray
    obs_idx: jnp.ndarray

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_eval_bundles(stacked: ShotBundle) -> List[EvalBundle]:
    """Unstack the padded ShotBundle produced by training load_data into per-shot bundles."""

    bundles: List[EvalBundle] = []
    n_shots = stacked.ts_t.shape[0]

    for i in range(n_shots):
        t_len = int(stacked.t_len[i])

        ts_t = jnp.asarray(stacked.ts_t[i, :t_len])
        ts_Te = jnp.asarray(stacked.ts_Te[i, :t_len])
        ts_Te_raw = jnp.asarray(stacked.ts_Te_raw[i, :t_len])
        mask = jnp.asarray(stacked.mask[i, :t_len])
        reliable_mask = jnp.asarray(stacked.reliable_mask[i])
        Te0 = jnp.asarray(stacked.Te0[i])
        z0 = float(stacked.z0[i])
        rho = jnp.asarray(stacked.rho_rom[i])
        Vprime = jnp.asarray(stacked.Vprime_rom[i])
        ctrl_t = jnp.asarray(stacked.ctrl_t[i, :t_len])
        ctrl_vals = jnp.asarray(stacked.ctrl_vals[i, :t_len])
        ctrl_means = jnp.asarray(stacked.ctrl_means[i])
        ctrl_stds = jnp.asarray(stacked.ctrl_stds[i])
        regime_ts = jnp.asarray(stacked.regime_ts[i, :t_len])
        ne_vals = jnp.asarray(stacked.ne_vals[i, :t_len])
        Te_edge = jnp.asarray(stacked.Te_edge[i, :t_len])
        dalpha_ts = jnp.asarray(stacked.dalpha_ts[i, :t_len])
        obs_idx = jnp.asarray(stacked.obs_idx[i])
        shot_id = int(stacked.shot_id[i])

        bundles.append(
            EvalBundle(
                ts_t=ts_t,
                ts_Te=ts_Te,
                ts_Te_raw=ts_Te_raw,
                mask=mask,
                reliable_mask=reliable_mask,
                Te0=Te0,
                z0=z0,
                shot_id=shot_id,
                rho=rho,
                Vprime=Vprime,
                ctrl_t=ctrl_t,
                ctrl_vals=ctrl_vals,
                ctrl_means=ctrl_means,
                ctrl_stds=ctrl_stds,
                regime_ts=regime_ts,
                ne_vals=ne_vals,
                Te_edge=Te_edge,
                dalpha_ts=dalpha_ts,
                obs_idx=obs_idx,
            )
        )

    return bundles

def load_model(model_path, config):
    """Recreate the trained model structure for deserialization."""
    key = jax.random.PRNGKey(0)
    model = build_hybrid_model(config, key)
    return eqx.tree_deserialise_leaves(model_path, model)


def _observation_weight_grid(mask, reliable_mask=None):
    mask = mask.astype(jnp.float64)
    if reliable_mask is not None:
        mask = mask * reliable_mask[None, :].astype(jnp.float64)
    col_cov = jnp.mean(mask, axis=0)
    has_obs = col_cov > 0
    inv = jnp.where(has_obs, 1.0 / (col_cov + 1e-8), 0.0)
    inv_sum = jnp.sum(inv)
    col_weight = jnp.where(
        inv_sum > 0,
        inv / (inv_sum + 1e-8),
        jnp.ones_like(col_cov) / jnp.maximum(col_cov.size, 1),
    )
    return mask * col_weight[None, :]


def masked_error_metrics_weighted(pred, obs, mask, reliable_mask=None):
    # Exclude Dirichlet boundary node (last column) to match training loss.
    if pred.shape[-1] >= 2:
        pred = pred[:, :-1]
        obs = obs[:, :-1]
        mask = mask[:, :-1]
        if reliable_mask is not None:
            reliable_mask = reliable_mask[:-1]
    weight_grid = _observation_weight_grid(mask, reliable_mask=reliable_mask)
    resid = (pred - obs) ** 2
    abs_resid = jnp.sqrt(resid)
    denom = jnp.maximum(jnp.abs(obs), 50.0)
    wsum = jnp.sum(weight_grid) + 1e-8
    mse = jnp.sum(weight_grid * resid) / wsum
    mae_eV = jnp.sum(weight_grid * abs_resid) / wsum
    mae_pct = 100.0 * jnp.sum(weight_grid * (abs_resid / denom)) / wsum
    return float(mse), float(mae_eV), float(mae_pct)

def run_inference(model, bundle: EvalBundle, imex_cfg: IMEXConfig):
    t0, t1 = float(bundle.ts_t[0]), float(bundle.ts_t[-1])

    # Evaluate controls at Te grid once; solver uses cheap blending across substeps.
    ctrl_interp = LinearInterpolation(ts=bundle.ctrl_t, ys=bundle.ctrl_vals)
    ctrl_vals_ts = ctrl_interp.evaluate(bundle.ts_t)
    ctrl_norm_ts = (ctrl_vals_ts - bundle.ctrl_means) / (bundle.ctrl_stds + 1e-6)
    ctrl_norm_ts = jnp.clip(ctrl_norm_ts, -10.0, 10.0)
    ne_edge_ts = bundle.ne_vals[:, -1]
    latent_features_ts = ctrl_norm_ts

    rho = bundle.rho
    Vprime = jnp.clip(bundle.Vprime, 1e-6, None)
    dr = jnp.diff(rho)
    dr = jnp.clip(dr, 1e-6 * jnp.max(dr) + 1e-12, None)
    Vprime_face = 0.5 * (Vprime[:-1] + Vprime[1:])
    Vprime_cell = 0.5 * (Vprime[:-1] + Vprime[1:])
    denom_raw = Vprime_cell * dr
    denom_floor = jnp.maximum(1e-4 * jnp.max(denom_raw), 1e-10)
    denom = jnp.maximum(denom_raw, denom_floor)
    ode_args_geom = (rho, Vprime, dr, Vprime_face, Vprime_cell, denom)

    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, jnp.array([bundle.z0], dtype=jnp.float64)])

    integrator = IMEXIntegrator(
        theta=imex_cfg.theta,
        dt_base=imex_cfg.dt_base,
        max_steps=imex_cfg.max_steps,
        rtol=imex_cfg.rtol,
        atol=imex_cfg.atol,
        substeps=getattr(imex_cfg, "substeps", 1),
    )

    sol = integrator.integrate(
        t_span=(t0, t1),
        y0=y0,
        saveat=bundle.ts_t,
        model=model,
        Te_edge_ts=bundle.Te_edge,
        ctrl_norm_ts=ctrl_norm_ts,
        ne_ts=bundle.ne_vals,
        args=ode_args_geom,
        latent_features_ts=latent_features_ts,
    )

    ys_clean = jnp.nan_to_num(sol.ys, nan=0.0, posinf=0.0, neginf=0.0)
    Te_hats = ys_clean[:, :-1]
    zs = ys_clean[:, -1]

    Te_bc_ts = bundle.Te_edge

    def reconstruct(Te_hat_row, bc_val):
        return jnp.append(Te_hat_row, bc_val / model.Te_scale) * model.Te_scale

    Te_model = jax.vmap(reconstruct)(Te_hats, Te_bc_ts)
    return Te_model, zs

def analyze_physics_components(model, bundle: EvalBundle, Te_model, zs):
    """Compute mean magnitudes for total tendency and NN source for quick diagnostics."""

    ctrl_interp = LinearInterpolation(ts=bundle.ctrl_t, ys=bundle.ctrl_vals)
    ctrl_vals_ts = ctrl_interp.evaluate(bundle.ts_t)
    ctrl_norm_ts = (ctrl_vals_ts - bundle.ctrl_means) / (bundle.ctrl_stds + 1e-6)
    ctrl_norm_ts = jnp.clip(ctrl_norm_ts, -10.0, 10.0)

    div_vals = jax.vmap(
        lambda Te_row, z_val: model.compute_divergence_from_values(bundle.rho, bundle.Vprime, Te_row, z_val)
    )(Te_model, zs)
    src_vals = jax.vmap(
        lambda Te_row, z_val, cn, ne: model.compute_source_from_values(bundle.rho, Te_row, z_val, ne, cn)
    )(Te_model, zs, ctrl_norm_ts, bundle.ne_vals)

    total = div_vals + src_vals
    return jnp.mean(jnp.abs(total)), jnp.mean(jnp.abs(src_vals))

def _regime_states(regime_ts) -> np.ndarray:
    regime_np = np.asarray(regime_ts, dtype=float)
    states = np.zeros_like(regime_np, dtype=np.int8)
    states[(regime_np >= 0.5) & (regime_np < 1.5)] = 1
    states[(regime_np >= 1.5) & (regime_np < 2.5)] = 2
    states[regime_np >= 2.5] = 3
    return states


def _boxcar_smooth(values: np.ndarray, width: int) -> np.ndarray:
    if values.size < 3:
        return values
    width = max(1, min(int(width), int(values.size)))
    if width <= 1:
        return values
    filt = np.ones(width, dtype=float) / float(width)
    return np.convolve(values, filt, mode="same")


def compute_dalpha_stats(ts, dalpha, regime_ts) -> dict:
    ts_np = np.asarray(ts, dtype=float)
    dalpha_np = np.asarray(dalpha, dtype=float)
    if dalpha_np.size == 0:
        return {
            "available": False,
            "finite_fraction": 0.0,
        }

    finite = np.isfinite(dalpha_np)
    finite_fraction = float(np.mean(finite))
    if not np.any(finite):
        return {
            "available": False,
            "finite_fraction": finite_fraction,
        }

    dalpha_filled = dalpha_np.copy()
    if not np.all(finite):
        dalpha_filled[~finite] = np.interp(
            ts_np[~finite],
            ts_np[finite],
            dalpha_np[finite],
            left=dalpha_np[finite][0],
            right=dalpha_np[finite][-1],
        )

    mean = float(np.mean(dalpha_filled))
    std = float(np.std(dalpha_filled))
    norm = (dalpha_filled - mean) / (std + 1.0e-6)
    slope = np.gradient(_boxcar_smooth(norm, 101), ts_np) if ts_np.size > 1 else np.zeros_like(norm)

    regime_state = _regime_states(regime_ts)
    transition_idx = np.flatnonzero((regime_state[:-1] < 2) & (regime_state[1:] >= 2)) if regime_state.size > 1 else np.array([], dtype=int)
    if transition_idx.size > 0:
        center = int(transition_idx[0] + 1)
    else:
        center = int(np.argmax(-slope)) if slope.size > 0 else 0
    lo = max(0, center - 5)
    hi = min(slope.size, center + 6)

    return {
        "available": True,
        "finite_fraction": finite_fraction,
        "min": float(np.min(dalpha_filled)),
        "max": float(np.max(dalpha_filled)),
        "mean": mean,
        "std": std,
        "hmode_fraction": float(np.mean(regime_state == 3)),
        "transition_fraction": float(np.mean(regime_state == 2)),
        "transition_time": float(ts_np[center]) if ts_np.size > 0 else float("nan"),
        "normalized_slope_min": float(np.min(slope)) if slope.size > 0 else 0.0,
        "normalized_slope_max": float(np.max(slope)) if slope.size > 0 else 0.0,
        "transition_window_slope_min": float(np.min(slope[lo:hi])) if hi > lo else 0.0,
        "transition_window_slope_max": float(np.max(slope[lo:hi])) if hi > lo else 0.0,
    }


def _add_regime_band(ax, ts_np: np.ndarray, regime_state: np.ndarray) -> None:
    colors = {
        0: (0.92, 0.92, 0.92, 0.35),
        1: (0.25, 0.47, 0.85, 0.14),
        2: (0.55, 0.55, 0.55, 0.16),
        3: (0.85, 0.25, 0.25, 0.14),
    }
    if ts_np.size == 0 or regime_state.size == 0:
        return
    start = 0
    while start < regime_state.size:
        state = int(regime_state[start])
        end = start + 1
        while end < regime_state.size and int(regime_state[end]) == state:
            end += 1
        x0 = float(ts_np[start])
        x1 = float(ts_np[end - 1])
        if end < ts_np.size:
            x1 = float(0.5 * (ts_np[end - 1] + ts_np[end]))
        elif ts_np.size > 1:
            x1 = float(ts_np[-1] + 0.5 * (ts_np[-1] - ts_np[-2]))
        ax.axvspan(x0, x1, color=colors.get(state, colors[0]), lw=0.0)
        start = end


def _unit_range(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values)
    filled = values.copy()
    if not np.all(finite):
        idx = np.flatnonzero(finite)
        filled[~finite] = np.interp(np.flatnonzero(~finite), idx, values[finite])
    vmin = float(np.nanmin(filled))
    vmax = float(np.nanmax(filled))
    return (filled - vmin) / (vmax - vmin + 1.0e-6)


def _padded_ylim(values: np.ndarray, lo: float = None, hi: float = None) -> tuple:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0 if lo is None else lo, 1.0 if hi is None else hi)
    ymin = float(np.min(finite)) if lo is None else min(float(lo), float(np.min(finite)))
    ymax = float(np.max(finite)) if hi is None else max(float(hi), float(np.max(finite)))
    if ymax <= ymin:
        pad = max(0.05, abs(ymax) * 0.1)
        return ymin - pad, ymax + pad
    pad = 0.08 * (ymax - ymin)
    return ymin - pad, ymax + pad


def compute_regime_consistency(regime_ts, zs, z_barrier=None) -> dict:
    regime_state = _regime_states(regime_ts)
    latent_np = np.asarray(z_barrier if z_barrier is not None else zs, dtype=float)
    finite = np.isfinite(latent_np)
    out = {"available": bool(np.any(finite))}
    if not out["available"]:
        return out
    h_mask = (regime_state == 3) & finite
    l_mask = (regime_state == 1) & finite
    out["latent_min"] = float(np.min(latent_np[finite]))
    out["latent_max"] = float(np.max(latent_np[finite]))
    out["latent_span"] = float(out["latent_max"] - out["latent_min"])
    out["hmode_median"] = float(np.median(latent_np[h_mask])) if np.any(h_mask) else float("nan")
    out["lmode_median"] = float(np.median(latent_np[l_mask])) if np.any(l_mask) else float("nan")
    out["h_minus_l_median"] = (
        float(out["hmode_median"] - out["lmode_median"])
        if np.isfinite(out["hmode_median"]) and np.isfinite(out["lmode_median"])
        else float("nan")
    )
    return out


def compute_dalpha_latent_alignment(dalpha_ts, z_barrier) -> dict:
    if z_barrier is None:
        return {"available": False}
    dalpha_np = np.asarray(dalpha_ts, dtype=float)
    z_np = np.asarray(z_barrier, dtype=float)
    if dalpha_np.size != z_np.size or dalpha_np.size == 0:
        return {"available": False}
    evidence = 1.0 - _unit_range(dalpha_np)
    finite = np.isfinite(evidence) & np.isfinite(z_np)
    if np.count_nonzero(finite) < 3:
        return {"available": False}
    evidence_use = evidence[finite]
    z_use = z_np[finite]
    evidence_std = float(np.std(evidence_use))
    z_std = float(np.std(z_use))
    corr = float("nan")
    if evidence_std > 1.0e-9 and z_std > 1.0e-9:
        corr = float(np.corrcoef(evidence_use, z_use)[0, 1])
    return {
        "available": True,
        "pearson_corr_with_1_minus_norm_dalpha": corr,
        "rmse_vs_1_minus_norm_dalpha": float(np.sqrt(np.mean((z_use - evidence_use) ** 2))),
        "mean_abs_error_vs_1_minus_norm_dalpha": float(np.mean(np.abs(z_use - evidence_use))),
    }


def plot_results(ts, rho, Te_obs_raw, mask, Te_model, zs, shot_id, plots_dir, regime_ts=None, dalpha_ts=None, z_barrier=None):
    ts_np = np.asarray(ts)
    rho_np = np.asarray(rho)
    Te_obs_np = np.asarray(Te_obs_raw)
    mask_np = np.asarray(mask)
    Te_model_np = np.asarray(Te_model)
    zs_np = np.asarray(zs)
    z_display_np = np.asarray(z_barrier, dtype=float) if z_barrier is not None else zs_np
    z_display_label = "Barrier latent" if z_barrier is not None else "Latent z"

    Te_obs_plot = np.where(mask_np > 0.5, Te_obs_np, np.nan)
    finite_obs = Te_obs_plot[np.isfinite(Te_obs_plot)]
    if finite_obs.size > 0:
        vmin = float(np.min(finite_obs))
        vmax = float(np.max(finite_obs))
    else:
        finite_model = Te_model_np[np.isfinite(Te_model_np)]
        if finite_model.size > 0:
            vmin = float(np.min(finite_model))
            vmax = float(np.max(finite_model))
        else:
            vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    obs_cmap = matplotlib.colormaps["inferno"].copy()
    obs_cmap.set_bad("white")
    model_cmap = matplotlib.colormaps["inferno"].copy()
    model_cmap.set_bad("white")
    Te_model_plot = np.where(np.isfinite(Te_model_np), np.clip(Te_model_np, vmin, vmax), np.nan)

    # 1. Temperature Profile Evolution (Heatmap)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Obs
    c1 = ax[0].pcolormesh(ts_np, rho_np, Te_obs_plot.T, shading="auto", cmap=obs_cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title(f"Shot {shot_id}: Measured-only Te")
    ax[0].set_ylabel("rho")
    plt.colorbar(c1, ax=ax[0])

    # Model
    c2 = ax[1].pcolormesh(ts_np, rho_np, Te_model_plot.T, shading="auto", cmap=model_cmap, vmin=vmin, vmax=vmax)
    ax[1].set_title(f"Shot {shot_id}: Model Te")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("rho")
    plt.colorbar(c2, ax=ax[1])

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"shot_{shot_id}_heatmap.png"))
    plt.close()

    regime_state = _regime_states(regime_ts) if regime_ts is not None else np.zeros(ts_np.shape, dtype=np.int8)
    dalpha_np = np.asarray(dalpha_ts, dtype=float) if dalpha_ts is not None else np.zeros_like(ts_np)

    fig, ax = plt.subplots(3, 1, figsize=(11, 12), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0, 1.0]})
    c1 = ax[0].pcolormesh(ts_np, rho_np, Te_obs_plot.T, shading="auto", cmap=obs_cmap, vmin=vmin, vmax=vmax)
    ax[0].set_title(f"Shot {shot_id}: Measured Te, D_alpha, and latent state")
    ax[0].set_ylabel("rho")
    plt.colorbar(c1, ax=ax[0], label="Te [eV]")

    _add_regime_band(ax[1], ts_np, regime_state)
    ax[1].plot(ts_np, dalpha_np, color="black", linewidth=1.4, label="D_alpha")
    ax[1].set_ylabel("D_alpha")
    ax[1].grid(True, alpha=0.25)
    if np.any((regime_state == 1) | (regime_state == 3)):
        ax[1].legend(loc="best")
    else:
        ax[1].set_title("Regime undecided")

    _add_regime_band(ax[2], ts_np, regime_state)
    if dalpha_np.size == z_display_np.size:
        ax[2].plot(ts_np, 1.0 - _unit_range(dalpha_np), color="tab:orange", linewidth=1.0, linestyle="--", alpha=0.75, label="1 - norm(D_alpha)")
    ax[2].plot(ts_np, z_display_np, color="tab:blue", linewidth=1.8, label=z_display_label)
    if z_barrier is not None:
        ax[2].axhline(0.5, color="tab:red", linewidth=0.8, linestyle=":", alpha=0.6)
        ax[2].set_ylim(*_padded_ylim(z_display_np, lo=0.0, hi=1.0))
    else:
        ax[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax[2].set_ylim(*_padded_ylim(z_display_np))
    ax[2].set_xlabel("Time (s)")
    ax[2].set_ylabel("latent")
    ax[2].grid(True, alpha=0.25)
    ax[2].legend(loc="best")

    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, f"shot_{shot_id}_overview.png"))
    plt.close(fig)

    plt.figure(figsize=(10, 4))
    _add_regime_band(plt.gca(), ts_np, regime_state)
    if dalpha_np.size == z_display_np.size:
        plt.plot(ts_np, 1.0 - _unit_range(dalpha_np), label="1 - norm(D_alpha)", color="tab:orange", linestyle="--", alpha=0.75)
    plt.plot(ts_np, z_display_np, label=z_display_label, color="tab:blue")
    if z_barrier is not None:
        plt.axhline(0.5, color="tab:red", linewidth=0.8, linestyle=":", alpha=0.6)
        plt.ylim(*_padded_ylim(z_display_np, lo=0.0, hi=1.0))
    else:
        plt.ylim(*_padded_ylim(z_display_np))
    plt.xlabel("Time (s)")
    plt.ylabel("latent")
    plt.title(f"Shot {shot_id}: Latent Coordinate Evolution")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"shot_{shot_id}_latent.png"))
    plt.close()


def _select_measured_trace_indices(mask_np: np.ndarray, obs_idx, max_traces: int) -> np.ndarray:
    n_cols = int(mask_np.shape[1])
    interior_cols = max(n_cols - 1, 1)
    coverage = mask_np[:, :interior_cols].mean(axis=0)
    measured_cols = np.flatnonzero(coverage > 0.0)

    if measured_cols.size > 0:
        if measured_cols.size <= max_traces:
            return measured_cols.astype(int)

        chunks = np.array_split(measured_cols, max_traces)
        return np.array([int(chunk[len(chunk) // 2]) for chunk in chunks if chunk.size > 0], dtype=int)

    fallback = np.array(obs_idx, dtype=int)
    fallback = fallback[(fallback >= 0) & (fallback < interior_cols)]
    if fallback.size == 0:
        fallback = np.arange(min(max_traces, interior_cols), dtype=int)
    return fallback[:max_traces]


def plot_time_series(ts, rho, Te_obs, Te_model, mask, obs_idx, shot_id, plots_dir, max_traces: int = 6):
    ts_np = np.asarray(ts)
    rho_np = np.asarray(rho)
    Te_obs_np = np.asarray(Te_obs)
    Te_model_np = np.asarray(Te_model)
    mask_np = np.asarray(mask)

    idxs = _select_measured_trace_indices(mask_np, obs_idx, max_traces)

    n_rows = idxs.size
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)

    measured_labeled = False
    for ax, idx in zip(axes, idxs):
        ax.plot(ts_np, Te_model_np[:, idx], label="Model", linewidth=2.0, color="tab:blue", zorder=2)

        obs_mask = mask_np[:, idx] > 0.5
        if np.any(obs_mask):
            ax.plot(
                ts_np[obs_mask],
                Te_obs_np[obs_mask, idx],
                label="Measured" if not measured_labeled else None,
                color="tab:orange",
                linewidth=1.25,
                marker="o",
                markersize=3.2,
                alpha=0.95,
                zorder=3,
            )
            measured_labeled = True

        ax.set_ylabel(f"Te @ rho={rho_np[idx]:.2f}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    if measured_labeled:
        axes[0].legend(loc="best")

    fig.suptitle(f"Shot {shot_id}: Model vs Measured Te (time series)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(plots_dir, f"shot_{shot_id}_timeseries.png"))
    plt.close(fig)

def summarize_data(bundles):
    stats = []
    for b in bundles:
        cov = float(b.mask.mean())
        n_rho = b.mask.shape[1]
        n_t = b.mask.shape[0]
        stats.append((b.shot_id, cov, n_t, n_rho))
    return stats


def main():
    ap = argparse.ArgumentParser(description="Evaluate trained physics manifold model")
    ap.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    ap.add_argument("--model-id", default=None, help="Override model_id from config")
    ap.add_argument("--role", default="val", choices=["train", "val", "test", "all"],
                    help="Which grouped-split role to evaluate (data/split.json); 'test' is the locked set")
    ap.add_argument("--no-plots", action="store_true", help="Skip per-shot plot generation")
    ap.add_argument("--data-check", action="store_true", help="Print mask coverage summary before eval")
    args = ap.parse_args()

    config_cli = load_config(args.config)

    model_id = args.model_id or config_cli["output"].get("model_id", "default_run")
    base_log_dir = config_cli["output"].get("log_dir", "logs")
    saved_cfg_path = os.path.join(base_log_dir, model_id, "config.yaml")

    config = config_cli
    if os.path.exists(saved_cfg_path):
        try:
            with open(saved_cfg_path, "r") as f:
                config = yaml.safe_load(f)
            print(f"[eval] Using saved training config: {saved_cfg_path}")
        except Exception as e:
            print(f"[eval] Could not load saved training config; using CLI config. Reason: {e}")

    config.setdefault("output", {})
    config["output"]["model_id"] = model_id

    base_save_dir = config["output"]["save_dir"]
    base_log_dir = config["output"].get("log_dir", "logs")

    model_dir = os.path.join(base_save_dir, model_id)
    log_dir = os.path.join(base_log_dir, model_id)

    raw = config["output"]["model_name"]
    safe = _sanitize_name(raw)
    best_ema = [
        os.path.join(model_dir, f"{raw}_best_ema.eqx"),
        os.path.join(model_dir, f"{safe}_best_ema.eqx"),
    ]
    best = [
        os.path.join(model_dir, f"{raw}_best.eqx"),
        os.path.join(model_dir, f"{safe}_best.eqx"),
    ]
    finetuned = [
        os.path.join(model_dir, f"{raw}_finetuned.eqx"),
        os.path.join(model_dir, f"{safe}_finetuned.eqx"),
    ]

    # Preference order can be overridden via config.output.checkpoint_preference
    # (comma-separated), but defaults to: best_ema,best,finetuned,newest
    pref_str = config.get("output", {}).get("checkpoint_preference", "best_ema,best,finetuned,newest")
    preference = [x.strip() for x in str(pref_str).split(",")]

    candidates_all = best_ema + best + finetuned
    try:
        model_path = _select_checkpoint_by_preference(preference, best_ema=best_ema, best=best, finetuned=finetuned)
    except FileNotFoundError:
        print(f"Model not found. Tried: {candidates_all}. Run training first.")
        return

    print("Loading Data...")
    stacked_bundles, rho_rom, _, _ = load_data(config)
    eval_bundles = build_eval_bundles(stacked_bundles)
    split_path = config.get("data", {}).get("split", "data/split.json")
    if args.role != "all" and os.path.exists(split_path):
        with open(split_path) as f:
            split_roles = json.load(f)
        keep = set(split_roles[args.role])
        eval_bundles = [b for b in eval_bundles if b.shot_id in keep]
        print(f"[eval] role={args.role}: {len(eval_bundles)} shots")
    if args.data_check:
        cov_stats = summarize_data(eval_bundles)
        print("Mask coverage (mean over grid) and shapes:")
        for sid, cov, n_t, n_r in cov_stats:
            print(f"  shot {sid}: cov={cov:.3f}, t={n_t}, rho={n_r}")
    
    print("Loading Model...")
    model = load_model(model_path, config)

    solver_name = str(config.get("training", {}).get("solver", "imex")).lower()
    if solver_name != "imex":
        raise ValueError(f"This branch is IMEX-only, but training.solver={solver_name!r}")

    imex_dict = config.get("training", {}).get(
        "imex",
        {
            "theta": 1.0,
            "dt_base": 0.001,
            "max_steps": 50000,
            "rtol": 1.0e-4,
            "atol": 1.0e-6,
            "substeps": 1,
        },
    )
    imex_cfg = IMEXConfig(
        theta=float(imex_dict.get("theta", 0.7)),
        dt_base=float(imex_dict.get("dt_base", 1e-3)),
        max_steps=int(imex_dict.get("max_steps", 50000)),
        rtol=float(imex_dict.get("rtol", 1e-4)),
        atol=float(imex_dict.get("atol", 1e-6)),
        substeps=int(imex_dict.get("substeps", 1)),
    )
    
    eval_dir = os.path.join(log_dir, "evaluation")
    plots_dir = os.path.join(eval_dir, "plots")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    for filename in os.listdir(plots_dir):
        if filename.startswith("shot_") and filename.endswith(".png"):
            os.remove(os.path.join(plots_dir, filename))
    
    print("Running Inference...")
    
    report = {
        "model_config": config['model'],
        "training_config": config['training'],
        "observability": {
            "reliable_cov_min": float(config.get("data", {}).get("reliable_cov_min", 0.10)),
            "reliable_rho_min": float(config.get("data", {}).get("reliable_rho_min", 0.80)),
        },
        "shot_metrics": {}
    }
    
    total_mse = 0.0
    total_mae_eV = 0.0
    total_mae_pct = 0.0
    annulus_total_mse = 0.0
    annulus_total_mae_eV = 0.0
    annulus_total_mae_pct = 0.0
    outside_total_mse = 0.0
    outside_total_mae_eV = 0.0
    outside_total_mae_pct = 0.0
    
    for bundle in eval_bundles:
        print(f"Evaluating Shot {bundle.shot_id}...")
        Te_model, zs = run_inference(model, bundle, imex_cfg)
        z_barrier = jax.vmap(model.barrier_coordinate)(zs)

        # Match the training-time weighting so offline evaluation is comparable.
        mse, mae_eV, mae_pct = masked_error_metrics_weighted(Te_model, bundle.ts_Te, bundle.mask)
        annulus_mse, annulus_mae_eV, annulus_mae_pct = masked_error_metrics_weighted(
            Te_model,
            bundle.ts_Te,
            bundle.mask,
            bundle.reliable_mask,
        )
        outside_mse, outside_mae_eV, outside_mae_pct = masked_error_metrics_weighted(
            Te_model,
            bundle.ts_Te,
            bundle.mask,
            1.0 - bundle.reliable_mask,
        )
        print(
            "  Legacy MSE: {:.4f} | MAE: {:.2f} eV | MAE%: {:.2f} | "
            "Annulus MAE: {:.2f} eV | Outside-annulus MAE: {:.2f} eV".format(
                mse,
                mae_eV,
                mae_pct,
                annulus_mae_eV,
                outside_mae_eV,
            )
        )
        total_mse += mse
        total_mae_eV += mae_eV
        total_mae_pct += mae_pct
        annulus_total_mse += annulus_mse
        annulus_total_mae_eV += annulus_mae_eV
        annulus_total_mae_pct += annulus_mae_pct
        outside_total_mse += outside_mse
        outside_total_mae_eV += outside_mae_eV
        outside_total_mae_pct += outside_mae_pct
        
        # Quantitative L/H regime assessment
        regime_ts_np = np.asarray(bundle.regime_ts)
        # Same clean-L/clean-H masking rule as the training loader.
        regime_mask_np = (
            ((regime_ts_np > 0.5) & (regime_ts_np < 1.5)) | ((regime_ts_np > 2.5) & (regime_ts_np < 3.5))
        ).astype(float)
        regime_logits_np = np.asarray(jax.vmap(model.compute_regime_logit)(zs))
        regime_class = regime_classification_metrics(regime_logits_np, regime_ts_np, regime_mask_np)

        # Labeled transition time: first sample where the label reaches H (3).
        ts_np = np.asarray(bundle.ts_t)
        h_idx = np.where(regime_ts_np > 2.5)[0]
        t_label = float(ts_np[h_idx[0]]) if h_idx.size else float("nan")
        transition_timing = transition_time_error(ts_np, regime_logits_np, t_label)

        bifurcation_summary = None
        ctrl_interp_diag = LinearInterpolation(ts=bundle.ctrl_t, ys=bundle.ctrl_vals)
        drive_features_diag = (ctrl_interp_diag.evaluate(bundle.ts_t) - bundle.ctrl_means) / (bundle.ctrl_stds + 1e-6)
        bif = normal_form_diagnostics(model.latent, np.asarray(drive_features_diag), np.asarray(zs))
        if bif is not None:
            bifurcation_summary = {
                "beta": bif["beta"],
                "tau_s": bif["tau"],
                "bistable": bool(bif["bistable"]),
                "a_fold_low": bif["a_fold_low"],
                "a_fold_high": bif["a_fold_high"],
                "drive_min": float(np.min(bif["a_t"])),
                "drive_max": float(np.max(bif["a_t"])),
                "bistable_fraction": bif["bistable_fraction"],
                "h_basin_fraction": float(np.mean(bif["basin"] > 0)),
            }
            np.savez(
                os.path.join(eval_dir, f"bifurcation_shot_{bundle.shot_id}.npz"),
                ts=ts_np,
                z=np.asarray(zs),
                a_t=bif["a_t"],
                c1=bif["c1"],
                c2=bif["c2"],
                a_fold_low=bif["a_fold_low"],
                a_fold_high=bif["a_fold_high"],
                z_saddle=bif["z_saddle"],
                basin=bif["basin"],
                regime_ts=regime_ts_np,
                regime_logits=regime_logits_np,
                dalpha_ts=np.asarray(bundle.dalpha_ts),
            )

        # Compact fit artifact for downstream figures (paper/scripts).
        np.savez(
            os.path.join(eval_dir, f"fit_shot_{bundle.shot_id}.npz"),
            ts=ts_np,
            rho=np.asarray(bundle.rho),
            Te_model=np.asarray(Te_model),
            Te_obs=np.asarray(bundle.ts_Te_raw),
            mask=np.asarray(bundle.mask),
            z_barrier=np.asarray(z_barrier),
            dalpha_ts=np.asarray(bundle.dalpha_ts),
        )

        # Physics Diagnostics
        diff_mag, source_mag = analyze_physics_components(model, bundle, Te_model, zs)
        
        # Latent Stats
        z_min, z_max = float(jnp.min(zs)), float(jnp.max(zs))
        z_std = float(jnp.std(zs))
        z_barrier_stats = None
        if z_barrier is not None:
            z_barrier_stats = {
                "min": float(jnp.min(z_barrier)),
                "max": float(jnp.max(z_barrier)),
                "std": float(jnp.std(z_barrier)),
            }
        
        metrics = {
            "mse": mse,
            "mae_eV": mae_eV,
            "mae_pct": mae_pct,
            "annulus_metrics": {
                "mse": annulus_mse,
                "mae_eV": annulus_mae_eV,
                "mae_pct": annulus_mae_pct,
            },
            "outside_annulus_metrics": {
                "mse": outside_mse,
                "mae_eV": outside_mae_eV,
                "mae_pct": outside_mae_pct,
            },
            "z_stats": {"min": z_min, "max": z_max, "std": z_std},
            "z_barrier_stats": z_barrier_stats,
            "dalpha_stats": compute_dalpha_stats(bundle.ts_t, bundle.dalpha_ts, bundle.regime_ts),
            "regime_consistency": compute_regime_consistency(bundle.regime_ts, zs, z_barrier),
            "dalpha_latent_alignment": compute_dalpha_latent_alignment(bundle.dalpha_ts, z_barrier),
            "regime_classification": regime_class,
            "transition_timing": transition_timing,
            "bifurcation": bifurcation_summary,
            "physics_consistency": {
                "diffusion_magnitude": float(diff_mag),
                "source_magnitude": float(source_mag),
                "source_ratio": float(source_mag / (diff_mag + 1e-6))
            }
        }
        report["shot_metrics"][str(bundle.shot_id)] = metrics
        
        rho_vals = np.array(bundle.rho)
        if args.no_plots:
            continue
        plot_results(
            bundle.ts_t,
            rho_vals,
            bundle.ts_Te_raw,
            bundle.mask,
            Te_model,
            zs,
            bundle.shot_id,
            plots_dir,
            regime_ts=bundle.regime_ts,
            dalpha_ts=bundle.dalpha_ts,
            z_barrier=z_barrier,
        )
        plot_time_series(bundle.ts_t, rho_vals, bundle.ts_Te_raw, Te_model, bundle.mask, bundle.obs_idx, bundle.shot_id, plots_dir)
        
    # Pooled regime-classification summary over shots.
    per_shot_cls = [m.get("regime_classification", {}) for m in report["shot_metrics"].values()]
    per_shot_cls = [c for c in per_shot_cls if c and c.get("n_scored", 0) > 1]
    if per_shot_cls:
        def _mean_of(key):
            vals = [c[key] for c in per_shot_cls if key in c and np.isfinite(c[key])]
            return float(np.mean(vals)) if vals else float("nan")
        tt_errors = [
            abs(m["transition_timing"]["transition_time_error_s"])
            for m in report["shot_metrics"].values()
            if m.get("transition_timing") and np.isfinite(m["transition_timing"].get("transition_time_error_s", float("nan")))
        ]
        report["regime_classification_summary"] = {
            "n_shots_scored": len(per_shot_cls),
            "mean_accuracy": _mean_of("accuracy"),
            "mean_f1": _mean_of("f1"),
            "mean_auc": _mean_of("auc"),
            "mean_brier": _mean_of("brier"),
            "mean_abs_transition_time_error_s": float(np.mean(tt_errors)) if tt_errors else float("nan"),
        }

    report["overall_metrics"] = {
        "mean_mse": total_mse / len(eval_bundles),
        "mean_mae_eV": total_mae_eV / len(eval_bundles),
        "mean_mae_pct": total_mae_pct / len(eval_bundles),
        "annulus_mean_mse": annulus_total_mse / len(eval_bundles),
        "annulus_mean_mae_eV": annulus_total_mae_eV / len(eval_bundles),
        "annulus_mean_mae_pct": annulus_total_mae_pct / len(eval_bundles),
        "outside_annulus_mean_mse": outside_total_mse / len(eval_bundles),
        "outside_annulus_mean_mae_eV": outside_total_mae_eV / len(eval_bundles),
        "outside_annulus_mean_mae_pct": outside_total_mae_pct / len(eval_bundles),
    }
    
    # Save Report
    with open(os.path.join(eval_dir, f"evaluation_report_{args.role}.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {eval_dir}")

if __name__ == "__main__":
    main()
