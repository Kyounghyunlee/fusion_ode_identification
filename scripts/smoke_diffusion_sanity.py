#!/usr/bin/env python
"""Lightweight diffusion operator sanity checks.

Run via:
  ./scripts/run_training_gpu.sh --python scripts/smoke_diffusion_sanity.py --config config/config_debug.yaml --shot 27567

Checks:
- Constant Te profile => explicit diffusion ~ 0
- Implicit operator boundary coupling sign sanity
"""

import argparse
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fusion_ode_identification.data import load_data
from fusion_ode_identification.imex_solver import apply_diffusion_explicit, build_diffusion_matrix_implicit


def _assert(name: str, cond: bool):
    if not cond:
        raise SystemExit(f"FAIL: {name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config_debug.yaml")
    p.add_argument("--shot", type=int, default=None)
    p.add_argument("--tol", type=float, default=1e-10)
    args = p.parse_args()

    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.shot is not None:
        cfg.setdefault("data", {})
        cfg["data"]["shots"] = [int(args.shot)]

    bundles, _rho_rom, _rho_cap, _obs_idx = load_data(cfg)
    b0 = jax.tree_util.tree_map(lambda x: x[0], bundles)

    rho = jnp.asarray(b0.rho_rom)
    Vprime = jnp.asarray(b0.Vprime_rom)
    N = int(rho.shape[0])

    # Use a benign chi profile for the operator-only sanity check.
    chi = jnp.ones((N,), dtype=jnp.float64)

    # 1) Constant profile => diffusion ~ 0
    Te_const = jnp.ones((N,), dtype=jnp.float64) * 123.4
    div = apply_diffusion_explicit(rho, Vprime, chi, Te_const)
    max_abs = float(jnp.max(jnp.abs(div)))
    print(f"explicit diffusion const-profile max|div|={max_abs:.3e}")
    _assert("explicit diffusion const-profile ~0", max_abs <= args.tol)

    # 2) Boundary coupling sign sanity for implicit operator
    dt = 1e-3
    theta = 1.0
    A, b_bc = build_diffusion_matrix_implicit(rho, Vprime, chi, dt=dt, theta=theta)

    # Construct a RHS with only boundary forcing: rhs = dt*theta*b_bc*T_edge
    T_edge = 1.0
    rhs = dt * theta * b_bc * T_edge
    x = jnp.linalg.solve(A, rhs)

    x_min = float(jnp.min(x))
    x_max = float(jnp.max(x))
    print(f"implicit bc-only solve: min={x_min:.3e} max={x_max:.3e}")

    _assert("implicit bc-only solve nonnegative", x_min >= -1e-12)
    _assert("implicit bc-only solve bounded", x_max <= T_edge * 1.01 + 1e-12)

    print("PASS")


if __name__ == "__main__":
    main()
