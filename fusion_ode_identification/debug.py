"""Debug utilities for single-shot evaluation and artifact export."""
# fusion_ode_identification/debug.py

import os
import re
from typing import Tuple, Sequence, List

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


def _existing_with_mtime(paths: Sequence[str]):
    out = []
    for p in paths:
        if os.path.exists(p):
            try:
                out.append((p, float(os.path.getmtime(p))))
            except Exception:
                out.append((p, float("nan")))
    return out


def _select_checkpoint_by_preference(
    preference: Sequence[str],
    best_ema: Sequence[str],
    best: Sequence[str],
    finetuned: Sequence[str],
) -> str:
    pref = [str(x).strip().lower() for x in preference if str(x).strip()]
    if len(pref) == 0:
        pref = ["best_ema", "best", "finetuned", "newest"]

    groups = {
        "best_ema": list(best_ema),
        "best": list(best),
        "finetuned": list(finetuned),
    }

    all_candidates: List[str] = list(best_ema) + list(best) + list(finetuned)
    existing_all = _existing_with_mtime(all_candidates)
    if existing_all:
        existing_sorted = sorted(existing_all, key=lambda x: x[1], reverse=True)
        print("[debug] Existing candidate checkpoints (newest first):")
        for p, mtime in existing_sorted:
            print(f"  - {p} (mtime={mtime:.0f})")

    for token in pref:
        if token in groups:
            for p in groups[token]:
                if os.path.exists(p):
                    print(f"[debug] Selected checkpoint by preference ({token}): {p}")
                    return p
        elif token == "newest":
            if not existing_all:
                continue
            selected = sorted(existing_all, key=lambda x: x[1], reverse=True)[0][0]
            print(f"[debug] Selected checkpoint by fallback (newest): {selected}")
            return selected
        else:
            raise ValueError(f"Unknown checkpoint preference token: {token!r}")

    raise FileNotFoundError(f"No checkpoint found. Tried: {all_candidates}")


def find_best_checkpoint(cfg) -> str:
    model_id = cfg["output"]["model_id"]
    save_dir = cfg["output"]["save_dir"]
    model_dir = os.path.join(save_dir, model_id)
    raw = cfg["output"]["model_name"]
    safe = sanitize_name(raw)

    finetuned = [
        os.path.join(model_dir, f"{raw}_finetuned.eqx"),
        os.path.join(model_dir, f"{safe}_finetuned.eqx"),
    ]
    best = [
        os.path.join(model_dir, f"{raw}_best.eqx"),
        os.path.join(model_dir, f"{safe}_best.eqx"),
    ]
    best_ema = [
        os.path.join(model_dir, f"{raw}_best_ema.eqx"),
        os.path.join(model_dir, f"{safe}_best_ema.eqx"),
    ]

    pref_str = (
        cfg.get("output", {}).get("checkpoint_preference")
        or cfg.get("debug", {}).get("checkpoint_preference")
        or "best_ema,best,finetuned,newest"
    )
    preference = [x.strip() for x in str(pref_str).split(",")]

    return _select_checkpoint_by_preference(preference, best_ema=best_ema, best=best, finetuned=finetuned)


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
    Vprime_rom = np.array(bundle0.Vprime_rom, dtype=float)
    shot_id = int(np.array(bundle0.shot_id))
    edge_idx = int(np.array(bundle0.edge_idx))
    rho_edge = float(np.array(bundle0.rho_edge))

    # Geometry diagnostics used by the diffusion operator.
    dr = np.diff(rho_rom) if rho_rom.size >= 2 else np.array([], dtype=float)
    min_dr = float(np.min(dr)) if dr.size else 0.0
    dr_floor = float(1e-6 * np.max(dr) + 1e-12) if dr.size else 0.0
    min_Vprime = float(np.min(Vprime_rom)) if Vprime_rom.size else 0.0
    Vprime_cell = 0.5 * (Vprime_rom[:-1] + Vprime_rom[1:]) if Vprime_rom.size >= 2 else np.array([], dtype=float)
    denom_raw = Vprime_cell * dr if dr.size else np.array([], dtype=float)
    min_denom = float(np.min(denom_raw)) if denom_raw.size else 0.0
    denom_floor = float(max(1e-4 * float(np.max(denom_raw)) if denom_raw.size else 0.0, 1e-10))

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
        Vprime_rom=Vprime_rom,
        obs_idx=obs,
        Te_model=Te_model,
        Te_data=Te_data,
        mask=mask,
        Te_edge=Te_edge,
        z_ts=z_ts,
        Te_model_edge=Te_model[:, -1],
        edge_idx=edge_idx,
        rho_edge=rho_edge,
        min_dr=min_dr,
        dr_floor=dr_floor,
        min_Vprime=min_Vprime,
        min_denom=min_denom,
        denom_floor=denom_floor,
        div_inf_ts=np.array(div_inf_ts[:L], dtype=float) if div_inf_ts is not None else None,
        src_inf_ts=np.array(src_inf_ts[:L], dtype=float) if src_inf_ts is not None else None,
        div_ts=np.array(div_ts[:L], dtype=float) if div_ts is not None else None,
        src_ts=np.array(src_ts[:L], dtype=float) if src_ts is not None else None,
    )


def run_debug_shot(config_path: str, shot_id: int, ckpt_path: str = None, out_dir: str = "out", solver_throw: bool = False):
    with open(config_path, "r") as f:
        cfg_cli = yaml.safe_load(f)

    cfg = cfg_cli
    try:
        model_id = cfg_cli["output"].get("model_id")
        base_log_dir = cfg_cli["output"].get("log_dir", "logs")
        saved_cfg_path = os.path.join(base_log_dir, model_id, "config.yaml")
        if model_id is not None and os.path.exists(saved_cfg_path):
            with open(saved_cfg_path, "r") as f:
                cfg = yaml.safe_load(f)
            print(f"[debug] Using saved training config: {saved_cfg_path}")
    except Exception as e:
        print(f"[debug] Could not load saved training config; using CLI config. Reason: {e}")

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

    # Evaluate controls on Te grid once.
    ctrl_interp = LinearInterpolation(ts=ctrl_t_full, ys=ctrl_vals_full)
    ctrl_vals_ts = ctrl_interp.evaluate(ts_t_full)
    ctrl_norm_ts = (ctrl_vals_ts - bundle0.ctrl_means) / (bundle0.ctrl_stds + 1e-6)
    ctrl_norm_ts = jnp.clip(ctrl_norm_ts, -10.0, 10.0)

    div_ts = jax.vmap(lambda Te_row, zi: model_loaded.compute_divergence_from_values(bundle0.rho_rom, bundle0.Vprime_rom, Te_row, zi))(
        ev.Te_model,
        ev.z_ts,
    )
    src_ts = jax.vmap(lambda Te_row, zi, cn, ne: model_loaded.compute_source_from_values(bundle0.rho_rom, Te_row, zi, ne, cn))(
        ev.Te_model,
        ev.z_ts,
        ctrl_norm_ts,
        ne_vals_full,
    )
    div_inf_ts = jnp.max(jnp.abs(div_ts), axis=1)
    src_inf_ts = jnp.max(jnp.abs(src_ts), axis=1)

    out_png = os.path.join(out_dir, f"debug_shot_{shot_id}.png")
    out_npz = os.path.join(out_dir, f"debug_shot_{shot_id}.npz")
    make_debug_plot_and_npz(bundle0, ev, out_png, out_npz, div_inf_ts=div_inf_ts, src_inf_ts=src_inf_ts, div_ts=div_ts, src_ts=src_ts)
    print(f"[debug] Wrote: {out_png}")
    print(f"[debug] Wrote: {out_npz}")
