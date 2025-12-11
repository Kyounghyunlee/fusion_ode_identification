"""
HPC Training Script for Physics-Consistent Manifold Learning
Implements strategies from PHYSICS_CONSISTENT_MANIFOLD.md:
- Random Restarts (Ensemble)
- Advanced Optimizers (AdamW/Lion)
- Learning Rate Schedules (Warmup + Cosine Decay)
- Gradient Clipping
- Physics Loss Annealing
"""

import argparse
import os
import time
import yaml
import glob
from typing import NamedTuple, List

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

# Enable 64-bit precision for stiff ODE stability
jax.config.update("jax_enable_x64", True)

# --- Configuration Loading ---
def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# --- Data Structures ---
CONTROL_NAMES = ["P_nbi", "Ip", "nebar", "S_gas", "S_rec", "S_nbi"]

class ShotBundle(NamedTuple):
    ts_t: jnp.ndarray
    ts_Te: jnp.ndarray
    mask: jnp.ndarray
    Te0: jnp.ndarray
    z0: float
    ode_args: tuple
    shot_id: int

# --- Model Definitions (Self-Contained) ---
class SourceNN(eqx.Module):
    mlp: eqx.nn.MLP
    source_scale: float

    def __init__(self, key, source_scale: float = 1.0):
        layers = 64
        depth = 3
        in_size = 1 + 1 + 1 + len(CONTROL_NAMES) + 1 # rho, Te, ne, controls, z
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=layers,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        # Initialize output layer to zero for stability
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].weight, self.mlp, jnp.zeros_like(self.mlp.layers[-1].weight))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].bias, self.mlp, jnp.zeros_like(self.mlp.layers[-1].bias))
        self.source_scale = source_scale

    def __call__(self, rho, Te_val, ne_val, controls, z):
        x = jnp.concatenate([
            jnp.array([rho], dtype=jnp.float64),
            jnp.array([Te_val], dtype=jnp.float64),
            jnp.array([ne_val], dtype=jnp.float64),
            controls,
            jnp.array([z], dtype=jnp.float64),
        ])
        return jnp.tanh(self.mlp(x)[0]) * self.source_scale

class EquilibriumManifold(eqx.Module):
    Phi: jnp.ndarray
    base_coef: jnp.ndarray
    latent_coef: jnp.ndarray
    latent_gain: float

    def __call__(self, z: float) -> jnp.ndarray:
        gate = jax.nn.sigmoid(self.latent_gain * z)
        features = self.base_coef + gate * self.latent_coef
        return self.Phi @ features

    def gate(self, z: float) -> float:
        return jax.nn.sigmoid(self.latent_gain * z)

class LatentDynamics(eqx.Module):
    alpha: float
    beta: float
    gamma: float
    mu_weights: jnp.ndarray
    mu_bias: float
    mu_ref: float

    def __call__(self, z: float, controls: jnp.ndarray) -> float:
        mu = jnp.dot(controls[:3], self.mu_weights) + self.mu_bias
        return self.alpha * (mu - self.mu_ref) - self.beta * z - self.gamma * z ** 3

class HybridField(eqx.Module):
    nn: SourceNN
    manifold: EquilibriumManifold
    latent: LatentDynamics

    Te_scale: float = 1000.0
    ne_scale: float = 1e19
    chi_core: float = 0.6
    chi_edge_base: float = 2.0
    chi_edge_drop: float = 1.0
    ped_center: float = 0.85
    ped_width: float = 0.08

    def __call__(self, t, y, args):
        (rho_vals, Vp_vals, control_ts, control_vals, control_means, control_stds, 
         ne_ts, ne_vals, Te_bc) = args

        delta_hat = y[:-1]
        z = y[-1]

        # Reconstruct full Temperature (Interior + Boundary)
        # delta_hat corresponds to nodes 0..N-2
        # Te_bc corresponds to node N-1
        Te_total = jnp.append(delta_hat * self.Te_scale, Te_bc)

        # Compute Physics Tendency
        dTedt = self.compute_physics_tendency(t, Te_total, z, args)
        
        # Derivative Clipping (Stability)
        dTedt = jnp.clip(dTedt, -1e4, 1e4)
        ddelta_dt = dTedt / self.Te_scale

        # Latent Dynamics
        control_interp = jnp.stack([jnp.interp(t, control_ts, control_vals[:, i]) for i in range(control_vals.shape[-1])])
        control_norm = (control_interp - control_means) / (control_stds + 1e-6)
        z_dot = self.latent(z, control_norm)
        
        return jnp.concatenate([ddelta_dt, jnp.array([z_dot], dtype=jnp.float64)])

    def compute_physics_tendency(self, t, Te_total, z, args):
        (rho_vals, Vp_vals, control_ts, control_vals, control_means, control_stds, 
         ne_ts, ne_vals, Te_bc) = args
        rho = jnp.asarray(rho_vals)
        
        # 1. Diffusivity
        chi_edge = self.chi_edge_base - self.chi_edge_drop * self.manifold.gate(z)
        chi_edge = jnp.clip(chi_edge, 0.1, 5.0)
        w_ped = jax.nn.sigmoid((rho - self.ped_center) / self.ped_width)
        chi = self.chi_core + w_ped * (chi_edge - self.chi_core)

        # 2. Flux (FVM on 1D grid)
        # Calculate gradients on faces 0..N-2
        dr_face = jnp.maximum(jnp.diff(rho), 1e-6)
        grad_T = jnp.diff(Te_total) / dr_face
        
        chi_face = 0.5 * (chi[:-1] + chi[1:])
        flux_face = -chi_face * grad_T

        # 3. Divergence
        Vp = jnp.clip(jnp.asarray(Vp_vals), 1e-6, None)
        Vp_face = 0.5 * (Vp[:-1] + Vp[1:])
        total_flux_face = Vp_face * flux_face
        
        # Flux entering from left for Node 0 is 0 (Neumann)
        flux_diff = total_flux_face - jnp.concatenate([jnp.array([0.0]), total_flux_face[:-1]])
        
        # Cell volumes
        dr_dual = 0.5 * (rho[2:] - rho[:-2])
        dr_0 = 0.5 * rho[1]
        dr_cell = jnp.concatenate([jnp.array([dr_0]), dr_dual])
        
        divergence = flux_diff / (Vp[:-1] * dr_cell)

        # 4. Source NN
        control_interp = jnp.stack([jnp.interp(t, control_ts, control_vals[:, i]) for i in range(control_vals.shape[-1])])
        control_norm = (control_interp - control_means) / (control_stds + 1e-6)
        ne_interp = jnp.clip(jnp.stack([jnp.interp(t, ne_ts, ne_vals[:, j]) for j in range(ne_vals.shape[-1])]), 1e17, self.ne_scale)

        # Vectorized NN evaluation on evolved nodes
        S_nn = jax.vmap(lambda r, T, n: self.nn(r, T / self.Te_scale, n / self.ne_scale, control_norm, z))(rho[:-1], Te_total[:-1], ne_interp[:-1])

        return divergence + S_nn

# --- Loss Function ---
def shot_loss(model, bundle: ShotBundle, lambda_phy: float):
    t0, t1 = bundle.ts_t[0], bundle.ts_t[-1]
    # Evolve nodes 0..N-2
    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, jnp.array([bundle.z0], dtype=jnp.float64)])

    # Solver setup
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)
    saveat = diffrax.SaveAt(ts=bundle.ts_t)
    
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(model), solver, t0, t1, y0=y0, dt0=5e-4,
        stepsize_controller=controller, saveat=saveat, max_steps=100000, throw=False, args=bundle.ode_args
    )

    # Data Loss
    Te_hats = sol.ys[:, :-1]
    zs = sol.ys[:, -1]
    Te_bc = bundle.ode_args[-1]
    
    def reconstruct(Te_hat_row):
        return jnp.append(Te_hat_row, Te_bc / model.Te_scale) * model.Te_scale

    Te_model = jax.vmap(reconstruct)(Te_hats)
    resid = bundle.mask * (Te_model - bundle.ts_Te)
    mse = jnp.sum(resid ** 2) / (jnp.sum(bundle.mask) + 1e-8)
    
    # Physics Loss (Slow Manifold)
    def physics_residual(t, z):
        T_man = model.manifold(z).at[-1].set(Te_bc) # Enforce BC
        tendency = model.compute_physics_tendency(t, T_man, z, bundle.ode_args)
        return jnp.mean(tendency ** 2)

    loss_slow = jnp.mean(jax.vmap(physics_residual)(bundle.ts_t, zs))
    return mse + lambda_phy * loss_slow + 1e-4 * jnp.mean(zs ** 2)

def batch_loss(model, bundles, lambda_phy):
    # Sum loss over a batch of shots
    losses = jnp.stack([shot_loss(model, b, lambda_phy) for b in bundles])
    return jnp.mean(losses)

# --- Data Loading ---
def load_data(config):
    data_dir = config['data']['data_dir']
    shot_list = config['data']['shots']
    
    if shot_list == "all":
        files = glob.glob(os.path.join(data_dir, "*_torax_training.npz"))
    else:
        files = [os.path.join(data_dir, f"{s}_torax_training.npz") for s in shot_list]
    
    bundles = []
    print(f"Loading {len(files)} shots...")
    
    # Load one file to get grid info
    ref_data = np.load(files[0])
    rho = jnp.array(ref_data['rho'])
    
    for f in files:
        d = np.load(f)
        # Basic preprocessing (simplified for brevity)
        ts_t = jnp.array(d['t'])
        ts_Te = jnp.array(d['Te'])
        mask = jnp.array(d['Te_mask'])
        
        # Controls
        ctrl_vals = jnp.stack([d[k] for k in CONTROL_NAMES], axis=-1)
        ctrl_means = jnp.mean(ctrl_vals, axis=0)
        ctrl_stds = jnp.std(ctrl_vals, axis=0)
        
        # Density
        ne_vals = jnp.array(d['ne'])
        
        # Initial Condition
        Te0 = ts_Te[0]
        # Fill NaNs in IC with synthetic profile if needed
        if jnp.isnan(Te0).any():
             Te0 = 100.0 * (1.0 - rho**2) + 10.0
             
        # Args
        # Use mean edge temperature for BC
        Te_bc_val = jnp.nanmean(ts_Te[:, -1])
        if jnp.isnan(Te_bc_val):
            Te_bc_val = 50.0

        ode_args = (rho, jnp.array(d['Vprime']), ts_t, ctrl_vals, ctrl_means, ctrl_stds, 
                    ts_t, ne_vals, Te_bc_val)
        
        shot_id = int(os.path.basename(f).split('_')[0])
        bundles.append(ShotBundle(ts_t, ts_Te, mask, Te0, 0.0, ode_args, shot_id))
        
    return bundles, rho

# --- Main Training Loop ---
def main():
    config = load_config()
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    
    print(f"Running on device: {jax.devices()[0]}")
    
    bundles, rho = load_data(config)
    
    # Basis for Manifold
    centers = jnp.linspace(0.1, 0.9, 5)
    diff = jnp.abs(rho[:, None] - centers[None, :])
    Phi = jnp.clip(1.0 - diff / 0.18, 0.0, 1.0)
    
    best_loss = float('inf')
    
    for restart in range(config['training']['num_restarts']):
        print(f"\n=== Restart {restart + 1}/{config['training']['num_restarts']} ===")
        key = jax.random.PRNGKey(int(time.time()) + restart)
        key_nn, key_man = jax.random.split(key)
        
        # Initialize Model
        model = HybridField(
            nn=SourceNN(key_nn),
            manifold=EquilibriumManifold(
                Phi=Phi,
                base_coef=jnp.zeros(5) + 100.0, # Start with flat 100eV
                latent_coef=jnp.zeros(5),
                latent_gain=1.0
            ),
            latent=LatentDynamics(
                alpha=1.0, beta=1.0, gamma=1.0,
                mu_weights=jnp.zeros(3), mu_bias=0.0, mu_ref=0.0
            )
        )
        
        # Optimizer & Schedule
        total_steps = config['training']['total_steps']
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-5,
            peak_value=config['training']['learning_rate'],
            warmup_steps=config['training']['warmup_steps'],
            decay_steps=total_steps,
            end_value=1e-6
        )
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(config['training']['grad_clip']),
            getattr(optax, config['training']['optimizer'])(learning_rate=schedule, weight_decay=config['training']['weight_decay'])
        )
        
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        
        @eqx.filter_value_and_grad
        def compute_loss(model, batch_bundles, lambda_phy):
            return batch_loss(model, batch_bundles, lambda_phy)
        
        @eqx.filter_jit
        def make_step(model, opt_state, batch_bundles, lambda_phy):
            loss, grads = compute_loss(model, batch_bundles, lambda_phy)
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return loss, model, opt_state

        # Training Loop
        start_time = time.time()
        for step in range(total_steps):
            # Physics Annealing
            lambda_phy = min(
                config['training']['lambda_phy_max'], 
                config['training']['lambda_phy_max'] * (step / config['training']['lambda_phy_warmup_steps'])
            )
            
            # Simple Batching (Random Sample)
            batch_indices = np.random.choice(len(bundles), config['training']['batch_size'])
            batch_bundles = [bundles[i] for i in batch_indices]
            
            loss, model, opt_state = make_step(model, opt_state, batch_bundles, lambda_phy)
            
            if step % 100 == 0:
                print(f"Step {step}: Loss = {loss:.4f}, Lambda_Phy = {lambda_phy:.4f}")
                
        # Save if best
        if loss < best_loss:
            best_loss = loss
            save_path = os.path.join(config['output']['save_dir'], f"{config['output']['model_name']}_best.eqx")
            eqx.tree_serialise_leaves(save_path, model)
            print(f"New best model saved to {save_path} (Loss: {loss:.4f})")
            
    print(f"Training Complete. Best Loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()
