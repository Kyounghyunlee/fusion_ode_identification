"""
Evaluation Script for Physics-Consistent Manifold Model
Loads a trained model and generates comparison plots (Model vs Observation).
"""

"""
Evaluation Script for Physics-Consistent Manifold Model
Loads a trained model and generates comparison plots (Model vs Observation).
"""

import os
import json
import argparse
import yaml
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import equinox as eqx
import diffrax

# Import model definitions from the training script
from train_ode_physics_manifold_hpc import HybridField, SourceNN, EquilibriumManifold, LatentDynamics, load_data, ShotBundle

jax.config.update("jax_enable_x64", True)

def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_model(model_path, rho):
    # Re-instantiate the model structure (must match training script exactly)
    # We need dummy keys/values just to build the skeleton
    key = jax.random.PRNGKey(0)
    
    # Basis (must match training)
    centers = jnp.linspace(0.1, 0.9, 5)
    diff = jnp.abs(rho[:, None] - centers[None, :])
    Phi = jnp.clip(1.0 - diff / 0.18, 0.0, 1.0)
    
    model = HybridField(
        nn=SourceNN(key),
        manifold=EquilibriumManifold(
            Phi=Phi,
            base_coef=jnp.zeros(5),
            latent_coef=jnp.zeros(5),
            latent_gain=1.0
        ),
        latent=LatentDynamics(
            alpha=1.0, beta=1.0, gamma=1.0,
            mu_weights=jnp.zeros(3), mu_bias=0.0, mu_ref=0.0
        )
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

def run_inference(model, bundle: ShotBundle):
    t0, t1 = bundle.ts_t[0], bundle.ts_t[-1]
    # Evolve nodes 0..N-2
    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, jnp.array([bundle.z0], dtype=jnp.float64)])

    solver = diffrax.Tsit5()
    saveat = diffrax.SaveAt(ts=bundle.ts_t)
    
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(model), solver, t0, t1, y0=y0, dt0=5e-4,
        saveat=saveat, max_steps=100000, throw=False, args=bundle.ode_args
    )
    
    Te_hats = sol.ys[:, :-1]
    zs = sol.ys[:, -1]
    Te_bc = bundle.ode_args[-1]
    
    def reconstruct(Te_hat_row):
        return jnp.append(Te_hat_row, Te_bc / model.Te_scale) * model.Te_scale

    Te_model = jax.vmap(reconstruct)(Te_hats)
    return Te_model, zs

def analyze_physics_components(model, bundle, Te_model, zs):
    """
    Decompose the tendency into Diffusion and Source terms to check physics consistency.
    """
    # We need to evaluate the tendency at each time step
    # This is computationally expensive but necessary for diagnostics
    
    def get_terms(t, Te, z):
        # Re-use the logic from HybridField.compute_physics_tendency
        # But we need to extract the intermediate values
        args = bundle.ode_args
        (rho_vals, Vp_vals, control_ts, control_vals, control_means, control_stds, 
         ne_ts, ne_vals, Te_bc) = args
        rho = jnp.asarray(rho_vals)
        
        # 1. Diffusivity
        chi_edge = model.chi_edge_base - model.chi_edge_drop * model.manifold.gate(z)
        chi_edge = jnp.clip(chi_edge, 0.1, 5.0)
        w_ped = jax.nn.sigmoid((rho - model.ped_center) / model.ped_width)
        chi = model.chi_core + w_ped * (chi_edge - model.chi_core)

        # 2. Flux (FVM)
        dr_face = jnp.maximum(jnp.diff(rho), 1e-6)
        grad_T = jnp.diff(Te) / dr_face
        chi_face = 0.5 * (chi[:-1] + chi[1:])
        flux_face = -chi_face * grad_T

        # 3. Divergence
        Vp = jnp.clip(jnp.asarray(Vp_vals), 1e-6, None)
        Vp_face = 0.5 * (Vp[:-1] + Vp[1:])
        total_flux_face = Vp_face * flux_face
        flux_diff = total_flux_face - jnp.concatenate([jnp.array([0.0]), total_flux_face[:-1]])
        
        dr_dual = 0.5 * (rho[2:] - rho[:-2])
        dr_0 = 0.5 * rho[1]
        dr_cell = jnp.concatenate([jnp.array([dr_0]), dr_dual])
        
        divergence = flux_diff / (Vp[:-1] * dr_cell)

        # 4. Source NN
        control_interp = jnp.stack([jnp.interp(t, control_ts, control_vals[:, i]) for i in range(control_vals.shape[-1])])
        control_norm = (control_interp - control_means) / (control_stds + 1e-6)
        ne_interp = jnp.clip(jnp.stack([jnp.interp(t, ne_ts, ne_vals[:, j]) for j in range(ne_vals.shape[-1])]), 1e17, model.ne_scale)

        S_nn = jax.vmap(lambda r, T, n: model.nn(r, T / model.Te_scale, n / model.ne_scale, control_norm, z))(rho[:-1], Te[:-1], ne_interp[:-1])
        
        return jnp.mean(jnp.abs(divergence)), jnp.mean(jnp.abs(S_nn))

    # Vectorize over time
    vmap_terms = jax.vmap(get_terms)
    diff_mags, source_mags = vmap_terms(bundle.ts_t, Te_model, zs)
    
    return jnp.mean(diff_mags), jnp.mean(source_mags)

def plot_results(ts, rho, Te_obs, Te_model, zs, shot_id, save_dir):
    # 1. Temperature Profile Evolution (Heatmap)
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    
    # Obs
    c1 = ax[0].contourf(ts, rho, Te_obs.T, levels=20, cmap='inferno')
    ax[0].set_title(f"Shot {shot_id}: Observed Te")
    ax[0].set_ylabel("rho")
    plt.colorbar(c1, ax=ax[0])
    
    # Model
    c2 = ax[1].contourf(ts, rho, Te_model.T, levels=20, cmap='inferno')
    ax[1].set_title(f"Shot {shot_id}: Model Te")
    ax[1].set_xlabel("Time (s)")
    ax[1].set_ylabel("rho")
    plt.colorbar(c2, ax=ax[1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"shot_{shot_id}_heatmap.png"))
    plt.close()
    
    # 2. Latent Dynamics
    plt.figure(figsize=(10, 4))
    plt.plot(ts, zs, label='Latent z')
    plt.title(f"Shot {shot_id}: Latent Coordinate Evolution")
    plt.xlabel("Time (s)")
    plt.ylabel("z")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"shot_{shot_id}_latent.png"))
    plt.close()

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
    bundles, rho_ref = load_data(config)
    if args.data_check:
        cov_stats = summarize_data(bundles)
        print("Mask coverage (mean over grid) and shapes:")
        for sid, cov, n_t, n_r in cov_stats:
            print(f"  shot {sid}: cov={cov:.3f}, t={n_t}, rho={n_r}")
    
    print("Loading Model...")
    model = load_model(model_path, rho_ref)
    
    eval_dir = os.path.join(log_dir, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    print("Running Inference...")
    
    report = {
        "model_config": config['model'],
        "training_config": config['training'],
        "shot_metrics": {}
    }
    
    total_mse = 0.0
    
    for bundle in bundles:
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
        
        rho_vals = np.array(bundle.ode_args[0])
        plot_results(bundle.ts_t, rho_vals, bundle.ts_Te, Te_model, zs, bundle.shot_id, eval_dir)
        
    report["overall_metrics"] = {
        "mean_mse": total_mse / len(bundles)
    }
    
    # Save Report
    with open(os.path.join(eval_dir, "evaluation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {eval_dir}")

if __name__ == "__main__":
    main()
