"""Shared data structures for tokamak ODE training and debugging."""
# fusion_ode_identification/types.py
from typing import NamedTuple

import jax.numpy as jnp


class ShotBundle(NamedTuple):
    ts_t: jnp.ndarray
    ts_Te: jnp.ndarray
    mask: jnp.ndarray
    obs_idx: jnp.ndarray
    regime_ts: jnp.ndarray
    regime_mask: jnp.ndarray
    Te0: jnp.ndarray
    z0: jnp.ndarray
    latent_idx: jnp.ndarray
    latent_proj: jnp.ndarray
    shot_id: jnp.ndarray
    t_len: jnp.ndarray  # scalar per-shot after slicing
    rho_rom: jnp.ndarray
    Vprime_rom: jnp.ndarray
    ctrl_t: jnp.ndarray
    ctrl_vals: jnp.ndarray
    ctrl_means: jnp.ndarray
    ctrl_stds: jnp.ndarray
    ne_vals: jnp.ndarray
    Te_edge: jnp.ndarray
    edge_idx: jnp.ndarray
    rho_edge: jnp.ndarray


class ShotEval(NamedTuple):
    ok: jnp.ndarray
    loss: jnp.ndarray
    mae_eV: jnp.ndarray
    mae_pct: jnp.ndarray
    Te_model: jnp.ndarray
    resid: jnp.ndarray
    ts_t: jnp.ndarray
    obs_idx: jnp.ndarray
    z_ts: jnp.ndarray  # latent trajectory for debugging


class LossCfg(NamedTuple):
    huber_delta: float
    lambda_src: float
    src_delta: float
    lambda_w: float
    model_error_delta: float
    lambda_z: float
    lambda_zreg: float
    throw_solver: bool


class IMEXConfig(NamedTuple):
    """IMEX solver configuration."""
    theta: float
    dt_base: float
    max_steps: int
    rtol: float
    atol: float
    substeps: int
