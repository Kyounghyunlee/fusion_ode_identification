"""
Evaluation Script for Physics-Consistent Manifold Model
Loads a trained model and generates comparison plots (Model vs Observation).
"""

# scripts/evaluate_model.py

"""
Evaluation Script for Physics-Consistent Manifold Model
Loads a trained model and generates comparison plots (Model vs Observation).
"""

import os
import json
import argparse
from typing import List, NamedTuple

import yaml
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import equinox as eqx
import diffrax

from fusion_ode_identification.model import HybridField, SourceNN, LatentDynamics
from fusion_ode_identification.data import load_data
from fusion_ode_identification.types import ShotBundle
from fusion_ode_identification.loss import MAX_SOLVER_STEPS

jax.config.update("jax_enable_x64", True)


class EvalBundle(NamedTuple):
    ts_t: jnp.ndarray
    ts_Te: jnp.ndarray
    mask: jnp.ndarray
    Te0: jnp.ndarray
    z0: float
    shot_id: int
    rho: jnp.ndarray
    Vprime: jnp.ndarray
    ctrl_t: jnp.ndarray
    ctrl_vals: jnp.ndarray
    ctrl_means: jnp.ndarray
    ctrl_stds: jnp.ndarray
    ne_vals: jnp.ndarray
    Te_edge: jnp.ndarray
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
        mask = jnp.asarray(stacked.mask[i, :t_len])
        Te0 = jnp.asarray(stacked.Te0[i])
        z0 = float(stacked.z0[i])
        rho = jnp.asarray(stacked.rho_rom[i])
        Vprime = jnp.asarray(stacked.Vprime_rom[i])
        ctrl_t = jnp.asarray(stacked.ctrl_t[i, :t_len])
        ctrl_vals = jnp.asarray(stacked.ctrl_vals[i, :t_len])
        ctrl_means = jnp.asarray(stacked.ctrl_means[i])
        ctrl_stds = jnp.asarray(stacked.ctrl_stds[i])
        ne_vals = jnp.asarray(stacked.ne_vals[i, :t_len])
        Te_edge = jnp.asarray(stacked.Te_edge[i, :t_len])
        obs_idx = jnp.asarray(stacked.obs_idx[i])
        shot_id = int(stacked.shot_id[i])

        bundles.append(
            EvalBundle(
                ts_t=ts_t,
                ts_Te=ts_Te,
                mask=mask,
                Te0=Te0,
                z0=z0,
                shot_id=shot_id,
                rho=rho,
                Vprime=Vprime,
                ctrl_t=ctrl_t,
                ctrl_vals=ctrl_vals,
                ctrl_means=ctrl_means,
                ctrl_stds=ctrl_stds,
                ne_vals=ne_vals,
                Te_edge=Te_edge,
                obs_idx=obs_idx,
            )
        )

    return bundles

def load_model(model_path, config):
    """Recreate the trained model structure for deserialization."""
    key = jax.random.PRNGKey(0)

    layers = int(config.get("model", {}).get("layers", 64))
    depth = int(config.get("model", {}).get("depth", 3))
    latent_gain = float(config.get("model", {}).get("latent_gain", 1.0))
    source_scale = float(config.get("model", {}).get("source_scale", 3.0e5))

    model = HybridField(
        nn=SourceNN(key, source_scale=source_scale, layers=layers, depth=depth),
        latent=LatentDynamics(
            alpha=jnp.array(1.0, dtype=jnp.float64),
            beta=jnp.array(1.0, dtype=jnp.float64),
            gamma=jnp.array(1.0, dtype=jnp.float64),
            mu_weights=jnp.zeros(3, dtype=jnp.float64),
            mu_bias=jnp.array(0.0, dtype=jnp.float64),
            mu_ref=jnp.array(0.0, dtype=jnp.float64),
        ),
        latent_gain=latent_gain,
    )

    return eqx.tree_deserialise_leaves(model_path, model)


def masked_mse_weighted(pred, obs, mask):
    mask = mask.astype(jnp.float64)
    col_weight = jnp.mean(mask, axis=0)
    col_weight = jnp.where(
        jnp.sum(col_weight) > 0,
        col_weight / (jnp.sum(col_weight) + 1e-8),
        jnp.ones_like(col_weight) / col_weight.size,
    )
    weight_grid = mask * col_weight
    resid = (pred - obs) ** 2
    return float(jnp.sum(weight_grid * resid) / (jnp.sum(weight_grid) + 1e-8))

def _build_ode_args(bundle: EvalBundle):
    ctrl_interp = diffrax.LinearInterpolation(ts=bundle.ctrl_t, ys=bundle.ctrl_vals)
    ne_interp = diffrax.LinearInterpolation(ts=bundle.ts_t, ys=bundle.ne_vals)
    Te_bc_interp = diffrax.LinearInterpolation(ts=bundle.ts_t, ys=bundle.Te_edge)
    return (
        bundle.rho,
        bundle.Vprime,
        ctrl_interp,
        bundle.ctrl_means,
        bundle.ctrl_stds,
        ne_interp,
        Te_bc_interp,
    )


def run_inference(model, bundle: EvalBundle):
    t0, t1 = float(bundle.ts_t[0]), float(bundle.ts_t[-1])

    ode_args = _build_ode_args(bundle)

    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, jnp.array([bundle.z0], dtype=jnp.float64)])

    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)
    saveat = diffrax.SaveAt(ts=bundle.ts_t)

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        solver,
        t0,
        t1,
        y0=y0,
        dt0=1e-4,
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=MAX_SOLVER_STEPS,
        throw=False,
        args=ode_args,
    )

    ys_clean = jnp.nan_to_num(sol.ys, nan=0.0, posinf=0.0, neginf=0.0)
    Te_hats = ys_clean[:, :-1]
    zs = ys_clean[:, -1]

    Te_bc_ts = ode_args[-1].evaluate(bundle.ts_t)

    def reconstruct(Te_hat_row, bc_val):
        return jnp.append(Te_hat_row, bc_val / model.Te_scale) * model.Te_scale

    Te_model = jax.vmap(reconstruct)(Te_hats, Te_bc_ts)
    return Te_model, zs

def analyze_physics_components(model, bundle: EvalBundle, Te_model, zs):
    """Compute mean magnitudes for total tendency and NN source for quick diagnostics."""

    ode_args = _build_ode_args(bundle)

    def total_tendency(t, Te_row, z_val):
        return model.compute_physics_tendency(t, Te_row, z_val, ode_args)

    def source_only(t, Te_row, z_val):
        return model.compute_source(t, Te_row, z_val, ode_args)

    f_vals = jax.vmap(total_tendency)(bundle.ts_t, Te_model, zs)
    s_vals = jax.vmap(source_only)(bundle.ts_t, Te_model, zs)

    return jnp.mean(jnp.abs(f_vals)), jnp.mean(jnp.abs(s_vals))

def plot_results(ts, rho, Te_obs, Te_model, zs, shot_id, plots_dir):
    ts_np = np.asarray(ts)
    rho_np = np.asarray(rho)
    Te_obs_np = np.asarray(Te_obs)
    Te_model_np = np.asarray(Te_model)
    zs_np = np.asarray(zs)

    # 1. Temperature Profile Evolution (Heatmap)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    # Obs
    c1 = ax[0].contourf(ts_np, rho_np, Te_obs_np.T, levels=20, cmap='inferno')
    ax[0].set_title(f"Shot {shot_id}: Observed Te")
    ax[0].set_ylabel("rho")
    plt.colorbar(c1, ax=ax[0])
    
    # Model
    c2 = ax[1].contourf(ts_np, rho_np, Te_model_np.T, levels=20, cmap='inferno')
    ax[1].set_title(f"Shot {shot_id}: Model Te")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("rho")
    plt.colorbar(c2, ax=ax[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"shot_{shot_id}_heatmap.png"))
    plt.close()
    
    # 2. Latent Dynamics
    plt.figure(figsize=(10, 4))
    plt.plot(ts_np, zs_np, label='Latent z')
    plt.title(f"Shot {shot_id}: Latent Coordinate Evolution")
    plt.xlabel("Time (s)")
    plt.ylabel("z")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f"shot_{shot_id}_latent.png"))
    plt.close()


def plot_time_series(ts, rho, Te_obs, Te_model, mask, obs_idx, shot_id, plots_dir, max_traces: int = 6):
    ts_np = np.asarray(ts)
    rho_np = np.asarray(rho)
    Te_obs_np = np.asarray(Te_obs)
    Te_model_np = np.asarray(Te_model)
    mask_np = np.asarray(mask)

    idxs = np.array(obs_idx)
    if idxs.size == 0:
        idxs = np.arange(min(5, Te_obs_np.shape[1]))
    idxs = idxs[:max_traces]

    n_rows = idxs.size
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 3 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)

    obs_labeled = False
    for ax, idx in zip(axes, idxs):
        ax.plot(ts_np, Te_model_np[:, idx], label="Model", linewidth=2.0, color="tab:blue")

        obs_mask = mask_np[:, idx] > 0.5
        if np.any(obs_mask):
            ax.scatter(ts_np[obs_mask], Te_obs_np[obs_mask, idx], label="Observed" if not obs_labeled else None, color="tab:orange", s=14, alpha=0.8)
            obs_labeled = True

        ax.set_ylabel(f"Te @ rho={rho_np[idx]:.2f}")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    if obs_labeled:
        axes[0].legend(loc="best")

    fig.suptitle(f"Shot {shot_id}: Model vs Observation (time series)")
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
    ap.add_argument("--data-check", action="store_true", help="Print mask coverage summary before eval")
    args = ap.parse_args()

    config = load_config(args.config)
    
    model_id = args.model_id or config['output'].get('model_id', 'default_run')
    base_save_dir = config['output']['save_dir']
    base_log_dir = config['output'].get('log_dir', 'logs')
    
    model_dir = os.path.join(base_save_dir, model_id)
    log_dir = os.path.join(base_log_dir, model_id)
    
    model_path = os.path.join(model_dir, f"{config['output']['model_name']}_best.eqx")
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Run training first.")
        return

    print("Loading Data...")
    stacked_bundles, rho_rom, _, _ = load_data(config)
    eval_bundles = build_eval_bundles(stacked_bundles)
    if args.data_check:
        cov_stats = summarize_data(eval_bundles)
        print("Mask coverage (mean over grid) and shapes:")
        for sid, cov, n_t, n_r in cov_stats:
            print(f"  shot {sid}: cov={cov:.3f}, t={n_t}, rho={n_r}")
    
    print("Loading Model...")
    model = load_model(model_path, config)
    
    eval_dir = os.path.join(log_dir, "evaluation")
    plots_dir = os.path.join(eval_dir, "plots")
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print("Running Inference...")
    
    report = {
        "model_config": config['model'],
        "training_config": config['training'],
        "shot_metrics": {}
    }
    
    total_mse = 0.0
    
    for bundle in eval_bundles:
        print(f"Evaluating Shot {bundle.shot_id}...")
        Te_model, zs = run_inference(model, bundle)

        # Calculate weighted MSE (mask + per-column coverage)
        mse = masked_mse_weighted(Te_model, bundle.ts_Te, bundle.mask)
        print(f"  MSE: {mse:.4f}")
        total_mse += mse
        
        # Physics Diagnostics
        diff_mag, source_mag = analyze_physics_components(model, bundle, Te_model, zs)
        
        # Latent Stats
        z_min, z_max = float(jnp.min(zs)), float(jnp.max(zs))
        z_std = float(jnp.std(zs))
        
        metrics = {
            "mse": mse,
            "z_stats": {"min": z_min, "max": z_max, "std": z_std},
            "physics_consistency": {
                "diffusion_magnitude": float(diff_mag),
                "source_magnitude": float(source_mag),
                "source_ratio": float(source_mag / (diff_mag + 1e-6))
            }
        }
        report["shot_metrics"][str(bundle.shot_id)] = metrics
        
        rho_vals = np.array(bundle.rho)
        plot_results(bundle.ts_t, rho_vals, bundle.ts_Te, Te_model, zs, bundle.shot_id, plots_dir)
        plot_time_series(bundle.ts_t, rho_vals, bundle.ts_Te, Te_model, bundle.mask, bundle.obs_idx, bundle.shot_id, plots_dir)
        
    report["overall_metrics"] = {
        "mean_mse": total_mse / len(eval_bundles)
    }
    
    # Save Report
    with open(os.path.join(eval_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {eval_dir}")

if __name__ == "__main__":
    main()
