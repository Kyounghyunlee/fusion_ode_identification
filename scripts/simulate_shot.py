"""
Simulate a shot using the Diffrax ODE solver and trained parameters.
Replaces the old simulate_shot.py.

Usage:
    python -m scripts.simulate_shot data/30421_torax_training.npz --params outputs/ode_training/params.json --transport-ne
"""

import argparse
import json
import os
import numpy as np
import jax
import jax.numpy as jnp
import diffrax
import equinox as eqx

from scripts.train_ode import ToraxField

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", help="Path to <shot>_torax_training.npz")
    ap.add_argument("--params", required=True, help="Path to params.json")
    ap.add_argument("--use-ne", action="store_true")
    ap.add_argument("--transport-ne", action="store_true")
    ap.add_argument("--shape-scale", action="store_true")
    ap.add_argument("--out", default="simulate_outputs.npz", help="Output file")
    args = ap.parse_args()

    # 1. Load Data
    print(f"Loading {args.pack}...")
    d = np.load(args.pack)
    t = d["t_ts"] # Simulation time points (can be finer if needed, but we output at these)
    rho = d["rho"]
    
    # Inputs for controls
    t_sum = d["t"]
    P_nbi = d["P_nbi"]
    Ip = d.get("Ip", np.zeros_like(t_sum))
    nebar = d.get("nebar", np.zeros_like(t_sum))
    S_gas = d.get("S_gas", np.zeros_like(t_sum)) * 1e-22
    S_rec = d.get("S_rec", np.zeros_like(t_sum))
    S_nbi = d.get("S_nbi", np.zeros_like(t_sum)) * 1e-20
    
    controls_arr = np.stack([P_nbi, Ip, nebar, S_gas, S_rec, S_nbi], axis=-1)
    controls_arr = np.nan_to_num(controls_arr)
    controls = diffrax.LinearInterpolation(ts=t_sum, ys=controls_arr)
    
    Vprime = d.get("Vprime", np.ones_like(rho))
    
    # Initial conditions
    Te0 = np.nan_to_num(d["Te"][0])
    ne0 = np.nan_to_num(d["ne"][0]) if "ne" in d else np.zeros_like(rho)
    
    # 2. Load Params
    print(f"Loading params from {args.params}...")
    with open(args.params) as f:
        params_raw = json.load(f)
    # Convert to JAX arrays
    params = {k: jnp.asarray(v) for k, v in params_raw.items()}
    
    # 3. Setup Solver
    # z0: use value from params if available (vector), else 0
    z0 = 0.0
    if "z0_vec" in params:
        z0_vec = params["z0_vec"]
        if z0_vec.ndim > 0:
            z0 = z0_vec[0] # Assume first shot if vector
        else:
            z0 = z0_vec
            
    y0_list = [Te0]
    if args.use_ne:
        y0_list.append(ne0)
    y0_list.append(jnp.array([z0]))
    y0 = jnp.concatenate(y0_list)
    
    field = ToraxField(
        params, rho, Vprime, controls, 
        args.use_ne, args.transport_ne, args.shape_scale
    )
    
    solver = diffrax.Kvaerno5()
    stepsize_controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)
    saveat = diffrax.SaveAt(ts=t)
    
    print("Running simulation...")
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        solver,
        t[0],
        t[-1],
        dt0=1e-4,
        y0=y0,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        max_steps=10000,
    )
    
    if sol.result != diffrax.RESULTS.successful:
        print("Warning: Simulation might have failed or hit max steps.")
        
    ys = sol.ys
    Nr = rho.shape[0]
    Te_sim = ys[:, :Nr]
    offset = Nr
    if args.use_ne:
        ne_sim = ys[:, offset:offset+Nr]
        offset += Nr
    else:
        ne_sim = None
    z_sim = ys[:, offset]
    
    # 4. Save
    np.savez_compressed(
        args.out, 
        t=t, 
        rho=rho, 
        Te=Te_sim, 
        ne=ne_sim if ne_sim is not None else np.array([]), 
        z=z_sim
    )
    print(f"Saved simulation to {args.out}")

if __name__ == "__main__":
    main()
