"""
Train a physics-consistent equilibrium manifold.
This script implements the approach defined in PHYSICS_CONSISTENT_MANIFOLD.md.
It unfreezes the manifold shape coefficients and adds a stationarity loss term.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, NamedTuple

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

jax.config.update("jax_enable_x64", True)

CONTROL_NAMES = ["P_nbi", "Ip", "nebar", "S_gas", "S_rec", "S_nbi"]


class ShotBundle(NamedTuple):
    ts_t: jnp.ndarray
    ts_Te: jnp.ndarray
    mask: jnp.ndarray
    Te0: jnp.ndarray
    z0: float
    ode_args: tuple


class SourceNN(eqx.Module):
    mlp: eqx.nn.MLP
    source_scale: float

    def __init__(self, key, source_scale: float = 1.0):
        layers = 64
        depth = 3
        in_size = 1 + 1 + 1 + len(CONTROL_NAMES) + 1
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=layers,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        # Initialize to near-zero
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
        # Unpack args
        (rho_vals, Vp_vals, control_ts, control_vals, control_means, control_stds, 
         ne_ts, ne_vals, Te_bc) = args

        delta_hat = y[:-1]
        z = y[-1]

        # Reconstruct full Temperature
        Te_hat_full = jnp.concatenate([
            delta_hat[:1],
            delta_hat,
            jnp.array([Te_bc / self.Te_scale], dtype=jnp.float64),
        ])
        Te_total = Te_hat_full * self.Te_scale

        # Compute Physics Tendency (Diffusion + Source)
        dTedt = self.compute_physics_tendency(t, Te_total, z, args)
        
        # Clip and scale
        dTedt = jnp.clip(dTedt, -1e4, 1e4)
        ddelta_dt = dTedt / self.Te_scale

        # Latent Dynamics
        # We need controls for latent dynamics too
        control_interp = jnp.stack([
            jnp.interp(t, control_ts, control_vals[:, i])
            for i in range(control_vals.shape[-1])
        ])
        control_norm = (control_interp - control_means) / (control_stds + 1e-6)
        
        z_dot = self.latent(z, control_norm)
        
        return jnp.concatenate([ddelta_dt, jnp.array([z_dot], dtype=jnp.float64)])

    def compute_physics_tendency(self, t, Te_total, z, args):
        """
        Computes F(T, u) = Diffusion(T) + S_nn(T, u).
        Te_total is the full profile (including ghost/boundary) or handled internally?
        In __call__, Te_total has shape (N+2,) or (N,)?
        
        In __call__:
        delta_hat is (N,). 
        Te_hat_full is (1) + (N) + (1) = N+2.
        So Te_total is size N+2 (Ghost Left, 0..N-1, Boundary Right).
        
        Wait, let's check the original script's indexing.
        Original:
        grad_T = (Te_total[1:] - Te_total[:-1]) / dr_face
        flux = -chi * grad_T
        
        Yes, Te_total includes ghost nodes.
        """
        (rho_vals, Vp_vals, control_ts, control_vals, control_means, control_stds, 
         ne_ts, ne_vals, Te_bc) = args

        rho = jnp.asarray(rho_vals)
        
        # 1. Diffusivity
        chi_edge = self.chi_edge_base - self.chi_edge_drop * self.manifold.gate(z)
        chi_edge = jnp.clip(chi_edge, 0.1, 5.0)
        w_ped = jax.nn.sigmoid((rho - self.ped_center) / self.ped_width)
        chi = self.chi_core + w_ped * (chi_edge - self.chi_core)

        # 2. Flux
        dr_face = jnp.diff(rho)
        dr_face = jnp.maximum(dr_face, 1e-6)
        grad_T = (Te_total[1:] - Te_total[:-1]) / dr_face
        chi_face = 0.5 * (chi[1:] + chi[:-1])
        flux = -chi_face * grad_T

        # 3. Divergence
        Vp = jnp.clip(jnp.asarray(Vp_vals), 1e-6, None)
        Vp_face = 0.5 * (Vp[1:] + Vp[:-1])
        dr_node = rho[2:] - rho[:-2]
        dr_node = jnp.maximum(dr_node, 1e-6)
        flux_right = Vp_face[1:] * flux[1:]
        flux_left = Vp_face[:-1] * flux[:-1]
        diff_term = (flux_right - flux_left) / (Vp[1:-1] * dr_node)

        # 4. Source NN
        control_interp = jnp.stack([
            jnp.interp(t, control_ts, control_vals[:, i])
            for i in range(control_vals.shape[-1])
        ])
        control_norm = (control_interp - control_means) / (control_stds + 1e-6)

        ne_interp = jnp.stack([
            jnp.interp(t, ne_ts, ne_vals[:, j])
            for j in range(ne_vals.shape[-1])
        ])
        ne_interp = jnp.clip(ne_interp, 1e17, self.ne_scale)

        rho_int = rho[1:-1]
        Te_int = Te_total[1:-1]
        ne_int = ne_interp[1:-1]

        def eval_nn(rho_pt, Te_pt, ne_pt):
            return self.nn(rho_pt, Te_pt / self.Te_scale, ne_pt / self.ne_scale, control_norm, z)

        S_nn = jax.vmap(eval_nn)(rho_int, Te_int, ne_int)

        return diff_term + S_nn


def shot_loss(model, bundle: ShotBundle, lambda_phy: float):
    t0 = bundle.ts_t[0]
    t1 = bundle.ts_t[-1]
    y0 = jnp.concatenate([
        bundle.Te0[1:-1] / model.Te_scale,
        jnp.array([bundle.z0], dtype=jnp.float64),
    ])

    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)
    saveat = diffrax.SaveAt(ts=bundle.ts_t)
    
    # 1. Solve Trajectory
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        solver,
        t0,
        t1,
        y0=y0,
        dt0=5e-4,
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=100000,
        throw=False,
        args=bundle.ode_args,
    )

    Te_hats = sol.ys[:, :-1]
    zs = sol.ys[:, -1]
    ts = bundle.ts_t

    # 2. Data Loss
    Te_bc = bundle.ode_args[-1]
    def reconstruct(Te_hat_row):
        Te_full = jnp.concatenate([
            Te_hat_row[:1],
            Te_hat_row,
            jnp.array([Te_bc / model.Te_scale], dtype=jnp.float64),
        ])
        return Te_full * model.Te_scale

    Te_model = jax.vmap(reconstruct)(Te_hats)
    mask = bundle.mask.astype(jnp.float64)
    resid = mask * (Te_model - bundle.ts_Te)
    mse = jnp.sum(resid ** 2) / (jnp.sum(mask) + 1e-8)
    
    # 3. Physics / Slow Manifold Loss
    # Evaluate F(T_manifold(z), u) at each time step
    
    def physics_residual_at_step(t, z):
        # Construct T on the manifold
        T_man_core = model.manifold(z) # shape (N,)
        
        # T_man_core has size 65 (Nodes 0..64).
        # We enforce the boundary condition at the edge (Node 64) to match the solver.
        T_man_full = T_man_core.at[-1].set(Te_bc)
        
        # Compute tendency
        # Note: compute_physics_tendency returns diff_term + S_nn
        # This corresponds to dTe/dt at the internal nodes.
        tendency = model.compute_physics_tendency(t, T_man_full, z, bundle.ode_args)
        return jnp.mean(tendency ** 2)

    # vmap over time
    phy_losses = jax.vmap(physics_residual_at_step)(ts, zs)
    loss_slow = jnp.mean(phy_losses)

    z_reg = 1e-4 * jnp.mean(zs ** 2)
    
    return mse + lambda_phy * loss_slow + z_reg


def loss_fn(model, bundles: Iterable[ShotBundle], lambda_phy: float):
    shot_losses = [shot_loss(model, bundle, lambda_phy) for bundle in bundles]
    return jnp.sum(jnp.stack(shot_losses)) / len(bundles)


# ... (build_hat_basis, parse_pack, collect_pack_paths same as before) ...
def build_hat_basis(rho, centers=None, width=0.18):
    if centers is None:
        centers = jnp.linspace(0.1, 0.9, 5)
    rho_grid = jnp.asarray(rho)
    centers = jnp.asarray(centers)
    diff = jnp.abs(rho_grid[:, None] - centers[None, :])
    hats = jnp.clip(1.0 - diff / width, 0.0, 1.0)
    norms = jnp.linalg.norm(hats, axis=0, keepdims=True)
    norms = jnp.where(norms < 1e-6, 1.0, norms)
    return hats / norms

def parse_pack(path):
    data = np.load(path)
    rho = data["rho"].astype(np.float64)
    ts_t = data.get("t_ts", data["t"]).astype(np.float64)
    Te = data["Te"].astype(np.float64)
    if "Te_mask" in data:
        mask = data["Te_mask"].astype(bool)
    else:
        mask = np.isfinite(Te)
    mask &= np.isfinite(Te)
    mask &= Te > 1.0
    fallback = np.isfinite(Te) & (Te > 1.0)
    ts_mask = mask if mask.sum() >= max(1, fallback.sum() // 10) else fallback

    Te0 = Te[0].copy()
    if np.nanmin(Te0) <= 10.0 or not np.all(np.isfinite(Te0)):
        edge_val = 10.0
        core_val = 100.0
        Te0 = edge_val + (core_val - edge_val) * (1.0 - rho ** 2)
        Te[0] = Te0

    def control_signal(name):
        if name in data:
            return np.interp(ts_t, data["t"], np.nan_to_num(data[name]))
        return np.zeros_like(ts_t)

    control_arr = np.stack(
        [control_signal(name) for name in CONTROL_NAMES],
        axis=-1,
    )
    control_means = np.nanmean(control_arr, axis=0)
    control_stds = np.nanstd(control_arr, axis=0)
    control_stds = np.where(control_stds < 1e-3, 1.0, control_stds)

    ne_arr = np.nan_to_num(data["ne"], nan=1e19)
    if ne_arr.shape[0] != ts_t.shape[0]:
        raise ValueError("ne profile does not share the Te time grid")

    Vp = np.nan_to_num(data.get("Vprime", np.ones_like(rho)), nan=1.0)
    Vp = np.maximum(Vp, 1e-3)

    return (
        ShotBundle(
            ts_t=jnp.array(ts_t, dtype=jnp.float64),
            ts_Te=jnp.array(np.nan_to_num(Te), dtype=jnp.float64),
            mask=jnp.array(ts_mask, dtype=bool),
            Te0=jnp.array(Te0, dtype=jnp.float64),
            z0=0.0,
            ode_args=(
                jnp.array(rho, dtype=jnp.float64),
                jnp.array(Vp, dtype=jnp.float64),
                jnp.array(ts_t, dtype=jnp.float64),
                jnp.array(control_arr, dtype=jnp.float64),
                jnp.array(control_means, dtype=jnp.float64),
                jnp.array(control_stds, dtype=jnp.float64),
                jnp.array(ts_t, dtype=jnp.float64),
                jnp.array(ne_arr, dtype=jnp.float64),
                jnp.array(float(Te0[-1]), dtype=jnp.float64),
            ),
        ),
        rho,
    )

def collect_pack_paths(explicit_paths: list[str] | None) -> list[Path]:
    if explicit_paths:
        return [Path(p) for p in explicit_paths]
    data_dir = Path("data")
    if not data_dir.exists():
        raise FileNotFoundError("data/ directory not found for automatic pack discovery")
    packs = sorted(data_dir.glob("*_torax_training*.npz"))
    if not packs:
        raise FileNotFoundError("no training packs found under data/")
    return packs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="*", help="Training pack paths")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda_phy", type=float, default=1e-2, help="Weight for physics stationarity loss")
    ap.add_argument("--out", default="outputs/physics_manifold")
    args = ap.parse_args()

    pack_paths = collect_pack_paths(args.packs or None)
    print(f"Training packs ({len(pack_paths)}): {[p.name for p in pack_paths]}")
    shot_bundles = []
    rho = None
    for pack_path in pack_paths:
        bundle, shot_rho = parse_pack(pack_path)
        shot_bundles.append(bundle)
        if rho is None:
            rho = shot_rho
        elif not np.allclose(shot_rho, rho):
            raise ValueError("Pack %s uses a different rho grid" % pack_path)

    if rho is None:
        raise ValueError("No packs provided")
    shot_bundles = tuple(shot_bundles)

    # Initialize Manifold
    basis = build_hat_basis(rho)
    Phi_pinv = jnp.linalg.pinv(basis)
    Te0_vec = shot_bundles[0].Te0
    base_coef = Phi_pinv @ Te0_vec
    latent_coef = jnp.linspace(0.0, 0.3, basis.shape[-1])
    latent_coef = latent_coef * (jnp.max(Te0_vec) - jnp.min(Te0_vec))

    manifold = EquilibriumManifold(
        Phi=jnp.array(basis, dtype=jnp.float64),
        base_coef=base_coef,
        latent_coef=latent_coef,
        latent_gain=3.0,
    )

    latent = LatentDynamics(
        alpha=jnp.array(1.0, dtype=jnp.float64),
        beta=jnp.array(1.0, dtype=jnp.float64),
        gamma=jnp.array(0.5, dtype=jnp.float64),
        mu_weights=jnp.array([0.6, 0.3, 0.1], dtype=jnp.float64),
        mu_bias=jnp.array(0.0, dtype=jnp.float64),
        mu_ref=jnp.array(0.0, dtype=jnp.float64),
    )

    key = jax.random.PRNGKey(0)
    nn_init = SourceNN(key, source_scale=30.0)
    
    # Construct Field
    field = HybridField(nn=nn_init, manifold=manifold, latent=latent)

    # Partitioning: Train NN, manifold latent_coef, and latent dynamics params
    # We create a filter spec that is True for trainable parameters
    filter_spec = jax.tree_util.tree_map(lambda _: False, field)
    
    # Enable NN parameters (only arrays)
    filter_spec = eqx.tree_at(
        lambda m: m.nn, 
        filter_spec, 
        replace_fn=lambda m: jax.tree_util.tree_map(lambda x: eqx.is_array(x), m)
    )
    
    # Enable Manifold latent_coef
    filter_spec = eqx.tree_at(
        lambda m: m.manifold.latent_coef,
        filter_spec,
        True
    )

    # Enable latent dynamics scalars/weights
    for getter in (
        lambda m: m.latent.alpha,
        lambda m: m.latent.beta,
        lambda m: m.latent.gamma,
        lambda m: m.latent.mu_weights,
        lambda m: m.latent.mu_bias,
        lambda m: m.latent.mu_ref,
    ):
        filter_spec = eqx.tree_at(getter, filter_spec, True)

    params, static = eqx.partition(field, filter_spec)

    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(args.lr))
    opt_state = optimizer.init(params)

    @eqx.filter_jit
    def step(params, opt_state):
        def compute_loss(p):
            model = eqx.combine(p, static)
            return loss_fn(model, shot_bundles, args.lambda_phy)

        loss, grads = jax.value_and_grad(compute_loss)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    os.makedirs(args.out, exist_ok=True)
    print(f"Starting physics-consistent training (lambda_phy={args.lambda_phy})...")
    for step_idx in range(args.steps):
        start = time.time()
        params, opt_state, loss = step(params, opt_state)
        if step_idx % 10 == 0:
            duration = time.time() - start
            print(f"Step {step_idx}: loss={loss:.4f} time={duration:.2f}s")

    trained_model = eqx.combine(params, static)
    eqx.tree_serialise_leaves(os.path.join(args.out, "model.eqx"), trained_model)
    print("Training complete.")

if __name__ == "__main__":
    main()
