"""Debug utilities for single-shot evaluation and artifact export."""
# fusion_ode_identification/debug.py

import os
import re
from typing import Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from .data import load_data
from .loss import eval_shot_trajectory_imex
from .interp import LinearInterpolation
from .model import HybridField, LatentDynamics, SourceNN
from .types import LossCfg, IMEXConfig

jax.config.update("jax_enable_x64", True)


def sanitize_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    return name


def find_best_checkpoint(cfg) -> str:
    model_id = cfg["output"]["model_id"]
    save_dir = cfg["output"]["save_dir"]
    model_dir = os.path.join(save_dir, model_id)
    raw = cfg["output"]["model_name"]
    safe = sanitize_name(raw)

    use_finetuned = bool(cfg.get("training", {}).get("lbfgs_finetune", False))

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

    candidates = best_ema + best + (finetuned if use_finetuned else [])

    existing = []
    for p in candidates:
        if os.path.exists(p):
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = float("nan")
            existing.append((p, mtime))
    if len(existing) > 0:
        existing_sorted = sorted(existing, key=lambda x: x[1], reverse=True)
        print("[debug] Existing candidate checkpoints (newest first):")
        for p, mtime in existing_sorted:
            print(f"  - {p} (mtime={mtime:.0f})")

    for p in candidates:
        if os.path.exists(p):
            print(f"[debug] Using checkpoint: {p}")
            return p
    raise FileNotFoundError(f"No checkpoint found. Tried: {candidates}")


def build_loss_cfg(cfg, solver_throw_override: bool = False) -> LossCfg:
    lcb = {
        "huber_delta": float(cfg["training"].get("huber_delta", 5.0)),
        "lambda_src": float(cfg["training"].get("lambda_src", 1e-4)),
        "src_delta": float(cfg["training"].get("src_delta", 5.0)),
        "lambda_w": float(cfg["training"].get("lambda_w", 1e-5)),
        "model_error_delta": float(cfg["training"].get("model_error_delta", 10.0)),
        "lambda_z": float(cfg["training"].get("lambda_z", 1e-4)),
        "lambda_zreg": float(cfg["training"].get("lambda_zreg", 1e-4)),
        "throw_solver": bool(cfg["training"].get("throw_solver", False)),
        "rtol": float(cfg["training"].get("rtol", 1e-3)),
        "atol": float(cfg["training"].get("atol", 1e-3)),
    }
    if solver_throw_override:
        lcb["throw_solver"] = True

    loss_cfg = LossCfg(
        huber_delta=lcb["huber_delta"],
        lambda_src=lcb["lambda_src"],
        src_delta=lcb["src_delta"],
        lambda_w=lcb["lambda_w"],
        model_error_delta=lcb["model_error_delta"],
        lambda_z=lcb["lambda_z"],
        lambda_zreg=lcb["lambda_zreg"],
        throw_solver=lcb["throw_solver"],
        rtol=lcb["rtol"],
        atol=lcb["atol"],
    )
    return loss_cfg


def build_imex_cfg(cfg) -> IMEXConfig:
    imex_dict = cfg.get("training", {}).get(
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
    return IMEXConfig(
        theta=float(imex_dict["theta"]),
        dt_base=float(imex_dict["dt_base"]),
        max_steps=int(imex_dict["max_steps"]),
        rtol=float(imex_dict["rtol"]),
        atol=float(imex_dict["atol"]),
        substeps=int(imex_dict.get("substeps", 1)),
    )


def build_model_template(cfg, key) -> HybridField:
    layers = int(cfg.get("model", {}).get("layers", 64))
    depth = int(cfg.get("model", {}).get("depth", 3))
    latent_gain = float(cfg.get("model", {}).get("latent_gain", 1.0))
    source_scale = float(cfg.get("model", {}).get("source_scale", 3.0e5))
    divergence_clip = float(cfg.get("model", {}).get("divergence_clip", 1.0e6))

    key_nn, key_mu = jax.random.split(key)

    return HybridField(
        nn=SourceNN(key_nn, source_scale=source_scale, layers=layers, depth=depth),
        latent=LatentDynamics(
            alpha=jnp.array(1.0, dtype=jnp.float64),
            beta=jnp.array(1.0, dtype=jnp.float64),
            gamma=jnp.array(1.0, dtype=jnp.float64),
            mu_weights=jax.random.normal(key_mu, (3,), dtype=jnp.float64) * 0.01,
            mu_bias=jnp.array(0.0, dtype=jnp.float64),
            mu_ref=jnp.array(0.0, dtype=jnp.float64),
        ),
        latent_gain=latent_gain,
        divergence_clip=divergence_clip,
    )


def make_debug_plot_and_npz(bundle0, ev, out_png: str, out_npz: str, div_inf_ts=None, src_inf_ts=None, div_ts=None, src_ts=None):
    L = int(np.array(bundle0.t_len))
    ts = np.array(bundle0.ts_t[:L], dtype=float)
    Te_data = np.array(bundle0.ts_Te[:L], dtype=float)
    mask = np.array(bundle0.mask[:L], dtype=float)
    Te_edge = np.array(bundle0.Te_edge[:L], dtype=float)
    Te_model = np.array(ev.Te_model[:L], dtype=float)
    z_ts = np.array(ev.z_ts[:L], dtype=float)
    rho_rom = np.array(bundle0.rho_rom, dtype=float)
    shot_id = int(np.array(bundle0.shot_id))
    edge_idx = int(np.array(bundle0.edge_idx))
    rho_edge = float(np.array(bundle0.rho_edge))

    obs = np.array(bundle0.obs_idx, dtype=int)
    if obs.ndim > 1:
        obs = obs[0]

    nrows = 4 if (div_inf_ts is not None and src_inf_ts is not None) else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(12, 12 if nrows == 4 else 10), sharex=True)

    ax = axes[0]
    for j in obs:
        ax.plot(ts, Te_model[:, j])
        m = mask[:, j] > 0.5
        if np.any(m):
            ax.scatter(ts[m], Te_data[m, j], s=8)
    ax.set_ylabel("Te [eV]")
    Te_edge_min = float(np.min(Te_edge)) if Te_edge.size else 0.0
    Te_edge_max = float(np.max(Te_edge)) if Te_edge.size else 0.0
    Te_edge_ptp = Te_edge_max - Te_edge_min

    ax.set_title(
        f"shot {shot_id} | ok={int(np.array(ev.ok))} | loss={float(np.array(ev.loss)):.3g} "
        f"| mae_eV={float(np.array(ev.mae_eV)):.2f} | mae_pct={float(np.array(ev.mae_pct)):.2f} "
        f"| Te_edge_ptp={Te_edge_ptp:.1f} (min={Te_edge_min:.1f}, max={Te_edge_max:.1f}) "
        f"| rho_edge={rho_edge:.4f} (idx={edge_idx})"
    )
    ax.grid(True)

    ax = axes[1]
    ax.plot(ts, z_ts)
    ax.set_ylabel("z(t)")
    ax.grid(True)

    ax = axes[2]
    ax.plot(ts, Te_edge, label="Te_edge (BC)")
    ax.plot(ts, Te_model[:, -1], label="Te_model edge")
    ax.set_ylabel("Te_edge [eV]")
    ax.set_xlabel("t")
    ax.legend()
    ax.grid(True)

    if nrows == 4:
        ax = axes[3]
        ax.plot(ts, np.array(div_inf_ts[:L], dtype=float), label="max|div| (eV/s)")
        ax.plot(ts, np.array(src_inf_ts[:L], dtype=float), label="max|src| (eV/s)")
        ax.set_ylabel("magnitude")
        ax.set_xlabel("t")
        ax.legend()
        ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    np.savez(
        out_npz,
        shot_id=shot_id,
        ok=int(np.array(ev.ok)),
        loss=float(np.array(ev.loss)),
        mae_eV=float(np.array(ev.mae_eV)),
        mae_pct=float(np.array(ev.mae_pct)),
        ts=ts,
        rho_rom=rho_rom,
        obs_idx=obs,
        Te_model=Te_model,
        Te_data=Te_data,
        mask=mask,
        Te_edge=Te_edge,
        z_ts=z_ts,
        Te_model_edge=Te_model[:, -1],
        edge_idx=edge_idx,
        rho_edge=rho_edge,
        div_inf_ts=np.array(div_inf_ts[:L], dtype=float) if div_inf_ts is not None else None,
        src_inf_ts=np.array(src_inf_ts[:L], dtype=float) if src_inf_ts is not None else None,
        div_ts=np.array(div_ts[:L], dtype=float) if div_ts is not None else None,
        src_ts=np.array(src_ts[:L], dtype=float) if src_ts is not None else None,
    )


def run_debug_shot(config_path: str, shot_id: int, ckpt_path: str = None, out_dir: str = "out", solver_throw: bool = False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("data", {})
    cfg["data"]["shots"] = [shot_id]

    bundles, rho_rom, rho_cap, obs_idx = load_data(cfg)
    bundle0 = jax.tree_util.tree_map(lambda x: x[0], bundles)

    if ckpt_path is None:
        ckpt_path = find_best_checkpoint(cfg)

    os.makedirs(out_dir, exist_ok=True)

    seed = int(cfg.get("training", {}).get("seed", 0))
    key = jax.random.PRNGKey(seed)
    template = build_model_template(cfg, key)
    model_loaded = eqx.tree_deserialise_leaves(ckpt_path, template)

    loss_cfg = build_loss_cfg(cfg, solver_throw_override=solver_throw)
    imex_cfg = build_imex_cfg(cfg)
    ev = eval_shot_trajectory_imex(model_loaded, bundle0, loss_cfg, imex_cfg)

    # Build ode_args matching eval_shot_trajectory (valid window only)
    L = int(np.array(bundle0.t_len))
    L = max(1, L)
    ts_t_full = bundle0.ts_t[:L]
    ctrl_t_full = bundle0.ctrl_t[:L]
    ctrl_vals_full = bundle0.ctrl_vals[:L]
    ne_vals_full = bundle0.ne_vals[:L]
    Te_edge_full = bundle0.Te_edge[:L]

    ctrl_interp = LinearInterpolation(ts=ctrl_t_full, ys=ctrl_vals_full)
    ne_interp = LinearInterpolation(ts=ts_t_full, ys=ne_vals_full)
    Te_bc_interp = LinearInterpolation(ts=ts_t_full, ys=Te_edge_full)
    ode_args = (
        bundle0.rho_rom,
        bundle0.Vprime_rom,
        ctrl_interp,
        bundle0.ctrl_means,
        bundle0.ctrl_stds,
        ne_interp,
        Te_bc_interp,
    )

    def _div_src_at(ti, Te_row, zi):
        div = model_loaded.compute_divergence_only(ti, Te_row, zi, ode_args)
        src = model_loaded.compute_source(ti, Te_row, zi, ode_args)
        return div, src

    div_ts, src_ts = jax.vmap(_div_src_at)(ev.ts_t, ev.Te_model, ev.z_ts)
    div_inf_ts = jnp.max(jnp.abs(div_ts), axis=1)
    src_inf_ts = jnp.max(jnp.abs(src_ts), axis=1)

    out_png = os.path.join(out_dir, f"debug_shot_{shot_id}.png")
    out_npz = os.path.join(out_dir, f"debug_shot_{shot_id}.npz")
    make_debug_plot_and_npz(bundle0, ev, out_png, out_npz, div_inf_ts=div_inf_ts, src_inf_ts=src_inf_ts, div_ts=div_ts, src_ts=src_ts)
    print(f"[debug] Wrote: {out_png}")
    print(f"[debug] Wrote: {out_npz}")
