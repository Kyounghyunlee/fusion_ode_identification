"""
Differentiable Physics training for TORAX using Diffrax (Method of Lines).

This script replaces the manual Euler time-stepper with a robust ODE solver (Diffrax).
It solves the coupled system:
    dTe/dt = div(chi * grad(Te - Te_profile)) + S_Te
    dne/dt = div(D * grad(ne)) + S_ne
    dz/dt  = f(z, u)

Usage:
    python -m scripts.train_ode data/30421_torax_training.npz --steps 500 --lr 1e-2 --transport-ne
"""

import argparse
import csv
import json
import os
import time
from typing import Dict, Tuple, Optional

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax.tree_util import tree_map, tree_leaves

from scripts.normalization import compute_stats, normalize_inputs, save_stats

# -----------------------------------------------------------------------------
# Physics & Numerics Helpers
# -----------------------------------------------------------------------------

def hat_basis(rho: jnp.ndarray, knots: jnp.ndarray) -> jnp.ndarray:
    """Piecewise-linear hat basis on [0,1]. Returns Phi with shape (Nrho, K)."""
    Nr = rho.shape[0]
    K = knots.shape[0]
    idx_right = jnp.searchsorted(knots, rho, side="right")
    idx_left = jnp.clip(idx_right - 1, 0, K - 1)
    idx_right = jnp.clip(idx_right, 0, K - 1)
    kL = knots[idx_left]
    kR = knots[idx_right]
    denom = jnp.maximum(kR - kL, 1e-12)
    wR = (rho - kL) / denom
    wL = 1.0 - wR
    Phi = jnp.zeros((Nr, K))
    Phi = Phi.at[jnp.arange(Nr), idx_left].add(wL)
    Phi = Phi.at[jnp.arange(Nr), idx_right].add(wR)
    return Phi

def sigmoid(x):
    return 1.0 / (1.0 + jnp.exp(-x))

def get_control_at_t(t: float, controls: diffrax.LinearInterpolation):
    """Extract control vector at time t."""
    # controls(t) returns shape (n_channels,)
    return controls.evaluate(t)

# -----------------------------------------------------------------------------
# Vector Field (The Physics)
# -----------------------------------------------------------------------------

class ToraxField(eqx.Module):
    """
    Defines the coupled ODE system dy/dt = F(t, y, args).
    State y is a flat vector: [Te_norm (Nr-2), ne_norm (Nr-2), z (1)].
    
    We evolve NORMALIZED variables to ensure numerical stability.
    Te_hat = Te / Te_scale
    ne_hat = ne / ne_scale
    
    We only evolve INTERIOR points to avoid singularity at rho=0 and fixed BC at rho=1.
    """
    params: Dict[str, jnp.ndarray]
    rho: jnp.ndarray
    Vprime: jnp.ndarray
    controls: diffrax.LinearInterpolation
    Phi: jnp.ndarray
    dr: jnp.ndarray
    Vp_face: jnp.ndarray
    use_ne: bool
    transport_ne: bool
    shape_scale: bool
    
    # Normalization scales (fixed constants)
    Te_scale: float = 1000.0 # eV
    ne_scale: float = 1e19   # m^-3
    Te_bc: float
    ne_bc: float
    
    def __init__(self, params, rho, Vprime, controls, use_ne, transport_ne, shape_scale, Te_bc, ne_bc):
        self.params = params
        self.rho = rho
        self.Vprime = Vprime
        self.controls = controls
        self.use_ne = use_ne
        self.transport_ne = transport_ne
        self.shape_scale = shape_scale
        self.Te_bc = Te_bc
        self.ne_bc = ne_bc
        
        # Precompute geometry
        self.Phi = hat_basis(rho, params["knots"])
        self.dr = rho[1:] - rho[:-1]
        
        # Use analytical Vprime to avoid data artifacts at boundary
        Vp_anal = 14.0 * rho
        Vp = jnp.maximum(Vprime, Vp_anal * 0.5)
        Vp = jnp.maximum(Vp, 0.1) # Hard clamp
        
        # Vp at faces (approx)
        self.Vp_face = 0.5 * (Vp[1:] + Vp[:-1])
        self.Vprime = Vp 

    def __call__(self, t, y, args):
        # y contains only INTERIOR points
        # Full grid size is Nr
        # y size is (Nr - 2) for each species
        Nr = self.rho.shape[0]
        N_interior = Nr - 2
        
        # Unpack NORMALIZED state (Interior only)
        Te_hat_int = y[:N_interior]
        # Evolve Te directly. Clamp for physics calculations only.
        # Te_hat_int = jax.nn.softplus(Te_hat_int) + 1e-4
        
        # Reconstruct full Te profile for gradient calculation
        # BC: Axis (Neumann, dT/dr=0) -> Te[0] = Te[1]
        # BC: Edge (Dirichlet) -> Te[-1] = Fixed (from data)
        Te_edge_hat = self.Te_bc
        
        # Concatenate: [Te_int[0], Te_int, Te_edge]
        # Te_int[0] is the value at rho[1]. We set rho[0] value to same.
        Te_hat = jnp.concatenate([
            Te_hat_int[0:1], # Te[0] = Te[1]
            Te_hat_int,      # Te[1]...Te[N-2]
            jnp.array([Te_edge_hat]) # Te[N-1]
        ])
        
        offset = N_interior
        if self.use_ne:
            ne_hat_int = y[offset:offset+N_interior]
            # ne_hat_int = jax.nn.softplus(ne_hat_int) + 1e-4
            
            ne_edge_hat = self.ne_bc
            ne_hat = jnp.concatenate([
                ne_hat_int[0:1],
                ne_hat_int,
                jnp.array([ne_edge_hat])
            ])
            offset += N_interior
        else:
            ne_hat = None
            ne = None
            
        z = y[offset]
        
        # Convert to physical units for physics calculation
        # Ensure positivity for physics terms (chi, sources)
        Te = jnp.maximum(Te_hat, 1e-4) * self.Te_scale
        if ne_hat is not None:
            ne = jnp.maximum(ne_hat, 1e-4) * self.ne_scale
        
        # Get controls at time t
        u = self.controls.evaluate(t)
        P_nbi = u[0]
        Ip = u[1]
        nebar = u[2]
        S_gas = u[3]
        S_rec = u[4]
        S_nbi = u[5]
        
        # --- 1. Latent Dynamics (z) ---
        mu = P_nbi
        if "w_mu" in self.params:
            mu_feat = jnp.stack([P_nbi, Ip, nebar])
            mu = jnp.dot(self.params["w_mu"], mu_feat) + self.params.get("b_mu", 0.0)
            
        alpha = self.params["alpha"]
        beta = jnp.exp(self.params["raw_beta"]) if "raw_beta" in self.params else self.params["beta"]
        gamma = jnp.exp(self.params["raw_gamma"]) if "raw_gamma" in self.params else self.params["gamma"]
        mu_c = self.params["mu_c"]
        
        dzdt = alpha * (mu - mu_c) - beta * z - gamma * (z ** 3)
        
        # --- 2. Te Dynamics ---
        # Profile shape
        sig = sigmoid(self.params["ksig"] * z)
        
        if self.shape_scale and ("W_A_Te" in self.params):
            feats = jnp.stack([z, P_nbi, Ip, nebar, 1.0])
            A_Te = jnp.dot(self.params["W_A_Te"], feats) + self.params["b_A_Te"]
            B_Te = jnp.dot(self.params["W_B_Te"], feats) + self.params["b_B_Te"]
            Te_prof = A_Te * self.params["phi_shape_Te"] + B_Te
        else:
            Te_prof = self.Phi @ (self.params["b0_Te"] + sig * self.params["b1_Te"] + P_nbi * self.params["bu_Te"])
            
        # Chi profile - Ensure it is positive and bounded
        if "chi_core" in self.params:
            rho_ped = jnp.abs(self.params["rho_ped"])
            w_ped = jnp.abs(self.params["w_ped"])
            w_ped_r = jnp.exp(-((self.rho - rho_ped) ** 2) / (2.0 * w_ped ** 2))
            chi_edge_L = jnp.abs(self.params["chi_edge_L"])
            chi_edge_H = jnp.abs(self.params["chi_edge_H"])
            chi_edge = chi_edge_L + sig * (chi_edge_H - chi_edge_L)
            chi_r = jnp.abs(self.params["chi_core"]) + w_ped_r * (chi_edge - jnp.abs(self.params["chi_core"]))
        else:
            chi_eff = jnp.abs(self.params["chi_H"]) + sigmoid(self.params["kdiff"] * z) * (jnp.abs(self.params["chi_L"]) - jnp.abs(self.params["chi_H"]))
            chi_r = jnp.full_like(self.rho, chi_eff)
        
        # Clamp Chi to avoid explosion or negative diffusion
        chi_r = jnp.clip(chi_r, 0.1, 100.0)
            
        # Diffusion: Flux = -chi * grad(Te - Te_prof)
        # Note: We compute gradients in PHYSICAL units
        Xp = Te - Te_prof
        chi_face = 0.5 * (chi_r[1:] + chi_r[:-1])
        
        # Gradient at faces
        # grad[i] is between node i and i+1
        grad_Xp = (Xp[1:] - Xp[:-1]) / jnp.maximum(self.dr, 1e-6)
        Flux_Te = -chi_face * grad_Xp
        
        # Divergence at INTERIOR nodes (1 to Nr-2)
        F_right = Flux_Te[1:Nr-1] # Indices 1 to Nr-2
        F_left  = Flux_Te[0:Nr-2] # Indices 0 to Nr-3
        
        # Vp_face has size Nr-1.
        Vp_f_right = self.Vp_face[1:Nr-1]
        Vp_f_left  = self.Vp_face[0:Nr-2]
        
        # Vp at nodes 1 to Nr-2
        Vp_nodes = self.Vprime[1:-1]
        dr_nodes = 0.5 * (self.dr[0:Nr-2] + self.dr[1:Nr-1])
        
        # Conservative divergence
        # Ensure Vp_nodes is not too small (it shouldn't be for interior nodes, but safety first)
        denom = jnp.maximum(Vp_nodes * dr_nodes, 1e-6)
        diff_term = (F_right * Vp_f_right - F_left * Vp_f_left) / denom
        
        dTedt_int = diff_term
        
        # Convert dTedt to normalized units
        dTe_hat_dt = dTedt_int / self.Te_scale
        
        # --- 3. ne Dynamics ---
        dne_hat_dt = jnp.zeros(N_interior) if self.use_ne else None
        
        if self.use_ne:
            if self.transport_ne:
                # D profile
                D_core = jnp.abs(self.params.get("D_core", 0.1))
                D_edge = jnp.abs(self.params.get("D_edge", 1.0))
                D_r = D_core + (self.rho**2) * (D_edge - D_core)
                D_r = jnp.clip(D_r, 0.01, 50.0) # Clamp D
                D_face = 0.5 * (D_r[1:] + D_r[:-1])
                
                # Sources
                shape_edge = jnp.exp(-((self.rho - 1.0)**2) / (2 * 0.05**2))
                shape_core = jnp.exp(-(self.rho**2) / (2 * 0.3**2))
                vol_int = lambda f: jnp.sum(f * self.Vprime) * (self.rho[1] - self.rho[0])
                shape_edge = shape_edge / (vol_int(shape_edge) + 1e-6)
                shape_core = shape_core / (vol_int(shape_core) + 1e-6)
                
                k_gas = jnp.abs(self.params.get("k_gas", 1.0))
                k_rec = jnp.abs(self.params.get("k_rec", 1.0))
                k_nbi = jnp.abs(self.params.get("k_nbi", 1.0))
                
                # Sources (Physical units)
                S_tot = (k_gas * S_gas * 1e22 * shape_edge + 
                         k_rec * S_rec * shape_edge + 
                         k_nbi * S_nbi * 1e20 * shape_core)
                
                # Clamp sources to reasonable values
                S_tot = jnp.clip(S_tot, -1e24, 1e24)
                
                # Diffusion (Physical units)
                grad_ne = (ne[1:] - ne[:-1]) / jnp.maximum(self.dr, 1e-6)
                Flux_ne = -D_face * grad_ne
                
                F_ne_right = Flux_ne[1:Nr-1]
                F_ne_left  = Flux_ne[0:Nr-2]
                
                div_ne_int = (F_ne_right * Vp_f_right - F_ne_left * Vp_f_left) / denom
                
                # Source at interior points
                S_tot_int = S_tot[1:-1]
                
                dnedt_int = div_ne_int + S_tot_int
                
                # Normalize
                dne_hat_dt = dnedt_int / self.ne_scale
                
            else:
                if self.shape_scale and ("W_A_ne" in self.params):
                    feats = jnp.stack([z, P_nbi, Ip, nebar, 1.0])
                    A_ne = jnp.dot(self.params["W_A_ne"], feats) + self.params["b_A_ne"]
                    B_ne = jnp.dot(self.params["W_B_ne"], feats) + self.params["b_B_ne"]
                    ne_prof = A_ne * self.params["phi_shape_ne"] + B_ne
                else:
                    ne_prof = self.Phi @ (self.params["b0_ne"] + sig * self.params["b1_ne"] + P_nbi * self.params["bu_ne"])
                
                dnedt = -100.0 * (ne - ne_prof)
                dne_hat_dt = dnedt[1:-1] / self.ne_scale

        # Pack derivatives
        dy = [dTe_hat_dt]
        if self.use_ne:
            dy.append(dne_hat_dt)
        dy.append(jnp.array([dzdt]))
        
        res = jnp.concatenate(dy)
        # Debug print
        # jax.debug.print("t={t}, max_dy={m}", t=t, m=jnp.max(jnp.abs(res)))
        return res

# -----------------------------------------------------------------------------
# Training Loop
# -----------------------------------------------------------------------------

def loss_fn(
    params,
    rho,
    ts_t, # Time points for data
    ts_Te, # Data Te
    ts_ne, # Data ne
    ts_mask_Te,
    ts_mask_ne,
    inputs_controls, # LinearInterpolation
    z0,
    Te0,
    ne0,
    args
):
    # 1. Solve ODE
    # Time span: from t[0] to t[-1]
    t0 = ts_t[0]
    t1 = ts_t[-1]
    
    # Initial state (NORMALIZED)
    Te_scale = 1000.0
    ne_scale = 1e19
    
    # Slice to interior points
    Te0_int = Te0[1:-1]
    
    y0_list = [Te0_int / Te_scale]
    if args.use_ne:
        ne0_int = ne0[1:-1]
        y0_list.append(ne0_int / ne_scale)
    y0_list.append(jnp.array([z0]))
    y0 = jnp.concatenate(y0_list)
    
    # Define vector field
    Te_bc_val = Te0[-1] / Te_scale
    ne_bc_val = ne0[-1] / ne_scale if args.use_ne else 0.0
    
    field = ToraxField(
        params, rho, args.Vprime, inputs_controls, 
        args.use_ne, args.transport_ne, args.shape_scale,
        Te_bc_val, ne_bc_val
    )
    
    # Solver options
    # Use Explicit solver for better stability with rough initial conditions
    solver = diffrax.Tsit5()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)
    
    # Save at data time points
    saveat = diffrax.SaveAt(ts=ts_t)
    
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        solver,
        t0,
        t1,
        dt0=1e-5, # Smaller initial step guess
        y0=y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=10000, # Allow more steps
        throw=False 
    )
    
    # Check for failure
    # if sol.result != diffrax.RESULTS.successful:
    #     return jnp.inf
        
    ys = sol.ys # (Nt, StateDim) - NORMALIZED
    
    # Unpack solution and Denormalize
    Nr = rho.shape[0]
    N_interior = Nr - 2
    
    Te_model_hat_int = ys[:, :N_interior]
    
    # Reconstruct full profile for loss
    # We need to pad back to Nr
    # Te[0] = Te[1], Te[-1] = Te_bc
    
    # Vectorized reconstruction
    def reconstruct(T_int):
        return jnp.concatenate([T_int[0:1], T_int, jnp.array([Te_bc_val])])
    
    Te_model_hat = jax.vmap(reconstruct)(Te_model_hat_int)
    Te_model = Te_model_hat * Te_scale
    
    offset = N_interior
    if args.use_ne:
        ne_model_hat_int = ys[:, offset:offset+N_interior]
        ne_model_hat = jax.vmap(lambda x: jnp.concatenate([x[0:1], x, jnp.array([ne_bc_val])]))(ne_model_hat_int)
        ne_model = ne_model_hat * ne_scale
        offset += N_interior
    else:
        ne_model = None
    z_model = ys[:, offset]
    
    # 2. Compute Loss
    # Masked MSE
    def masked_mse(model, data, mask, sigma):
        resid = (model - data) / sigma
        resid = jnp.where(mask, resid, 0.0)
        return 0.5 * jnp.sum(resid**2) / jnp.maximum(jnp.sum(mask), 1.0)

    sigma_Te = jnp.exp(params["log_sigma_Te"]) if args.learn_noise else params["sigma_Te"]
    loss = masked_mse(Te_model, ts_Te, ts_mask_Te, sigma_Te)
    
    if args.use_ne and ne_model is not None:
        sigma_ne = jnp.exp(params["log_sigma_ne"]) if args.learn_noise else params["sigma_ne"]
        loss += masked_mse(ne_model, ts_ne, ts_mask_ne, sigma_ne)
        
        # Nebar penalty
        nebar_model = jnp.mean(ne_model, axis=1)
        # Get nebar data from controls (it's the 3rd channel, index 2)
        # But controls are continuous, we need data at ts_t
        # We can evaluate controls at ts_t
        u_eval = jax.vmap(inputs_controls.evaluate)(ts_t)
        nebar_data = u_eval[:, 2] # Normalized nebar
        # Note: nebar_data in controls is normalized. nebar_model is normalized.
        # We should compare them.
        loss += 0.5 * args.lambda_global * jnp.mean((nebar_model - nebar_data)**2)

    # Regularization
    reg = 1e-4 * (
        jnp.sum(params["b0_Te"]**2) + jnp.sum(params["b1_Te"]**2) + jnp.sum(params["bu_Te"]**2) +
        params["chi_H"]**2 + params["chi_L"]**2 + params["alpha"]**2
    )
    loss += reg
    loss += args.lambda_z0 * (z0**2)
    
    return loss

# -----------------------------------------------------------------------------
# Main Driver
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="+")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--use-ne", action="store_true")
    ap.add_argument("--transport-ne", action="store_true")
    ap.add_argument("--shape-scale", action="store_true")
    ap.add_argument("--learn-noise", action="store_true")
    ap.add_argument("--lambda-z0", type=float, default=1e-3)
    ap.add_argument("--lambda-global", type=float, default=1e-4)
    args = ap.parse_args()
    
    # Logic fix: transport_ne implies use_ne
    if args.transport_ne:
        args.use_ne = True
    
    # Load Data
    print(f"Loading {len(args.packs)} packs...")
    datasets = []
    
    # Special handling: If using 30421, try to load clean targets from simulate_outputs.npz
    clean_targets = None
    if os.path.exists("simulate_outputs.npz"):
        print("Found simulate_outputs.npz. Will use it for Te targets if timestamps match.")
        clean_data = np.load("simulate_outputs.npz")
        clean_targets = {
            "t": clean_data["t"],
            "Te": clean_data["Te"]
        }

    for path in args.packs:
        d = np.load(path)
        # Subsample for speed if needed
        stride = 1
        t = d["t_ts"][::stride]
        rho = d["rho"]
        
        # Try to use clean targets
        if clean_targets is not None and len(t) == len(clean_targets["t"]):
            print(f"Using clean Te from simulate_outputs.npz for {path}")
            Te = clean_targets["Te"][::stride]
            # simulate_outputs.npz doesn't have ne, so we might need to fallback or ignore
            ne = None 
        else:
            Te = d["Te"][::stride]
            ne = d["ne"][::stride] if "ne" in d else None
        
        # SANITIZE DATA: Clamp Te to be strictly positive
        # Negative values or zeros cause solver crashes and physics violations
        Te = np.nan_to_num(Te, nan=0.0)
        Te = np.maximum(Te, 10.0) # Minimum 10 eV
        
        if ne is not None:
            ne = np.nan_to_num(ne, nan=0.0)
            ne = np.maximum(ne, 1e17) # Minimum density
        
        # Inputs for controls
        # We need to align inputs (on 't') to 't_ts' or just create a continuous interpolation
        t_sum = d["t"]
        P_nbi = d["P_nbi"]
        Ip = d.get("Ip", np.zeros_like(t_sum))
        nebar = d.get("nebar", np.zeros_like(t_sum))
        S_gas = d.get("S_gas", np.zeros_like(t_sum)) * 1e-22 # Scale
        S_rec = d.get("S_rec", np.zeros_like(t_sum))
        S_nbi = d.get("S_nbi", np.zeros_like(t_sum)) * 1e-20 # Scale
        
        # Create interpolation
        # Stack controls: (Nt, 6)
        controls_arr = np.stack([P_nbi, Ip, nebar, S_gas, S_rec, S_nbi], axis=-1)
        # Handle NaNs
        controls_arr = np.nan_to_num(controls_arr)
        
        # Create Diffrax interpolation
        # t_sum must be strictly increasing
        controls = diffrax.LinearInterpolation(ts=t_sum, ys=controls_arr)
        
        # Handle NaNs in initial condition
        Te0_raw = Te[0]
        # Check for NaNs OR unphysically low values (which we just clamped, but checking again for logic flow)
        if np.isnan(Te0_raw).any() or np.min(Te0_raw) <= 15.0: # If it was clamped to 10, it hits this.
            # Construct synthetic profile
            # Find valid edge
            valid_idx = np.where(np.isfinite(Te0_raw))[0]
            if len(valid_idx) > 0:
                edge_val = Te0_raw[valid_idx[-1]]
            else:
                edge_val = 50.0 # Fallback eV
            
            # Parabolic profile: T = T_edge + (T_core - T_edge) * (1 - rho^2)
            # Guess T_core = 10 * T_edge
            core_val = 10.0 * edge_val
            Te0_clean = edge_val + (core_val - edge_val) * (1.0 - rho**2)
            print(f"Warning: Te[0] bad. Using synthetic parabolic profile (Edge={edge_val:.2f}, Core={core_val:.2f}).")
        else:
            Te0_clean = Te0_raw

        if ne is not None:
            ne0_raw = ne[0]
            if np.isnan(ne0_raw).any():
                valid_idx = np.where(np.isfinite(ne0_raw))[0]
                if len(valid_idx) > 0:
                    edge_val = ne0_raw[valid_idx[-1]]
                else:
                    edge_val = 0.1
                core_val = 5.0 * edge_val
                ne0_clean = edge_val + (core_val - edge_val) * (1.0 - rho**2)
            else:
                ne0_clean = ne0_raw
        else:
            ne0_clean = np.zeros_like(rho)

        datasets.append({
            "t": t,
            "rho": rho,
            "Te": np.nan_to_num(Te), # Replace NaNs with 0 for safety in loss
            "ne": np.nan_to_num(ne) if ne is not None else None,
            "controls": controls,
            "Vprime": d.get("Vprime", np.ones_like(rho)),
            "Te_mask": d.get("Te_mask", np.isfinite(Te))[::stride],
            "ne_mask": d.get("ne_mask", np.isfinite(ne))[::stride] if ne is not None else None,
            "Te0": Te0_clean,
            "ne0": ne0_clean,
            "ne_scale_val": float(np.nanstd(ne)) if ne is not None else 1.0
        })

    # Init Params
    K = 8
    key = jax.random.PRNGKey(0)
    params = {
        "knots": jnp.linspace(0.0, 1.0, K),
        "b0_Te": jax.random.normal(key, (K,)) * 0.1,
        "b1_Te": jax.random.normal(key, (K,)) * 0.1,
        "bu_Te": jax.random.normal(key, (K,)) * 0.01,
        "ksig": jnp.array(1.0),
        "chi_H": jnp.array(2.0),
        "chi_L": jnp.array(10.0),
        "kdiff": jnp.array(1.0),
        "alpha": jnp.array(10.0), # Faster latent dynamics
        "beta": jnp.array(1.0),
        "gamma": jnp.array(1.0),
        "mu_c": jnp.array(0.0),
        "sigma_Te": jnp.array(1.0),
        "sigma_ne": jnp.array(1.0),
        "log_sigma_Te": jnp.array(0.0),
        "log_sigma_ne": jnp.array(0.0),
        # Transport params
        "D_core": jnp.array(0.1),
        "D_edge": jnp.array(1.0),
        "k_gas": jnp.array(1.0),
        "k_rec": jnp.array(1.0),
        "k_nbi": jnp.array(1.0),
        # Shape scale
        "phi_shape_Te": jnp.exp(-3.0 * datasets[0]["rho"]),
        "W_A_Te": jax.random.normal(key, (1, 5)) * 0.1,
        "W_B_Te": jax.random.normal(key, (1, 5)) * 0.1,
        "b_A_Te": jnp.array(0.0),
        "b_B_Te": jnp.array(0.0),
    }
    
    # Per-shot z0
    z0_vec = jnp.zeros(len(datasets))
    
    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(args.lr)
    )
    opt_state = optimizer.init(params)
    
    # Training Loop
    print("Starting training with Diffrax...")
    
    @eqx.filter_jit
    def update(params, z0_vec, opt_state):
        loss_val = 0.0
        grads_acc = tree_map(lambda x: jnp.zeros_like(x), params)
        z0_grads = jnp.zeros_like(z0_vec)
        
        for i, ds in enumerate(datasets):
            # Wrap loss to get grads wrt params and z0
            def loss_wrapper(p_in, z):
                # Copy params to avoid side effects
                p = p_in.copy()
                # Inject ne_scale_val into params
                args_i = argparse.Namespace(**vars(args))
                args_i.Vprime = ds["Vprime"]
                p["ne_scale_val"] = jnp.array(ds["ne_scale_val"])
                
                return loss_fn(
                    p, ds["rho"], ds["t"], ds["Te"], ds["ne"],
                    ds["Te_mask"], ds["ne_mask"], ds["controls"],
                    z, ds["Te0"], ds["ne0"], args_i
                )
            
            l, (g_p, g_z) = jax.value_and_grad(loss_wrapper, argnums=(0, 1))(params, z0_vec[i])
            loss_val += l
            grads_acc = tree_map(lambda a, b: a + b, grads_acc, g_p)
            z0_grads = z0_grads.at[i].set(g_z)
            
        updates, opt_state = optimizer.update(grads_acc, opt_state, params)
        params = optax.apply_updates(params, updates)
        z0_vec = z0_vec - args.lr * z0_grads # Simple SGD for z0
        
        return params, z0_vec, opt_state, loss_val

    for step in range(args.steps):
        start = time.time()
        params, z0_vec, opt_state, loss = update(params, z0_vec, opt_state)
        end = time.time()
        if step % 10 == 0:
            print(f"Step {step}: Loss = {loss:.4f} (Time: {end-start:.2f}s)")
            
    # Save results
    out_dir = "outputs/ode_training"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save params
    serializable_params = {k: (v.tolist() if hasattr(v, "tolist") else float(v)) for k, v in params.items()}
    with open(os.path.join(out_dir, "params.json"), "w") as f:
        json.dump(serializable_params, f, indent=2)
        
    print(f"Training complete. Results saved to {out_dir}")

if __name__ == "__main__":
    main()
