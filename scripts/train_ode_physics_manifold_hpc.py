"""
HPC Training Script for Physics-Consistent Manifold Learning
"""

import argparse
import os
import time
import yaml
import glob
import logging
import functools
from typing import NamedTuple, List, Dict, Any

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pathlib

jax.config.update("jax_enable_x64", True)

MAX_SOLVER_STEPS = 200000
CONTROL_NAMES = ["P_nbi", "Ip", "nebar", "S_gas", "S_rec", "S_nbi"]


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ShotBundle(NamedTuple):
    ts_t: jnp.ndarray
    ts_Te: jnp.ndarray
    mask: jnp.ndarray
    obs_idx: jnp.ndarray
    regime_ts: jnp.ndarray
    regime_mask: jnp.ndarray
    Te0: jnp.ndarray
    z0: float
    latent_idx: jnp.ndarray
    latent_proj: jnp.ndarray
    shot_id: jnp.ndarray
    t_len: int
    rho_rom: jnp.ndarray
    Vprime_rom: jnp.ndarray
    ctrl_t: jnp.ndarray
    ctrl_vals: jnp.ndarray
    ctrl_means: jnp.ndarray
    ctrl_stds: jnp.ndarray
    ne_vals: jnp.ndarray
    Te_edge: jnp.ndarray


class LossCfg(NamedTuple):
    huber_delta: float
    lambda_src: float
    src_delta: float
    lambda_w: float
    model_error_delta: float
    lambda_z: float
    lambda_zreg: float
    throw_solver: bool


class SourceNN(eqx.Module):
    mlp: eqx.nn.MLP
    source_scale: float

    def __init__(self, key, source_scale: float = 1.0, layers: int = 64, depth: int = 3):
        in_size = 1 + 1 + 1 + len(CONTROL_NAMES) + 1  # rho, Te, ne, controls, z
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=layers,
            depth=depth,
            activation=jax.nn.tanh,
            key=key,
        )
        # zero-init final layer for stability
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].weight, self.mlp, jnp.zeros_like(self.mlp.layers[-1].weight))
        self.mlp = eqx.tree_at(lambda m: m.layers[-1].bias, self.mlp, jnp.zeros_like(self.mlp.layers[-1].bias))
        self.source_scale = float(source_scale)

    def __call__(self, rho, Te_val, ne_val, controls, z):
        x = jnp.concatenate(
            [
                jnp.array([rho], dtype=jnp.float64),
                jnp.array([Te_val], dtype=jnp.float64),
                jnp.array([ne_val], dtype=jnp.float64),
                controls.astype(jnp.float64),
                jnp.array([z], dtype=jnp.float64),
            ]
        )
        return self.mlp(x)[0] * self.source_scale


class LatentDynamics(eqx.Module):
    alpha: jnp.ndarray
    beta: jnp.ndarray
    gamma: jnp.ndarray
    mu_weights: jnp.ndarray
    mu_bias: jnp.ndarray
    mu_ref: jnp.ndarray

    def __call__(self, z: float, controls: jnp.ndarray) -> float:
        mu = jnp.dot(controls[:3], self.mu_weights) + self.mu_bias
        alpha_eff = jax.nn.softplus(self.alpha)
        beta_eff = jax.nn.softplus(self.beta)
        gamma_eff = jax.nn.softplus(self.gamma)
        return alpha_eff * (mu - self.mu_ref) - beta_eff * z - gamma_eff * z**3


class HybridField(eqx.Module):
    nn: SourceNN
    latent: LatentDynamics
    latent_gain: jnp.ndarray

    Te_scale: float = 1000.0
    ne_scale: float = 1e19
    chi_core: jnp.ndarray
    chi_edge_base: jnp.ndarray
    chi_edge_drop: jnp.ndarray
    ped_center: float = 0.85
    ped_width: float = 0.08

    def __init__(
        self,
        nn: SourceNN,
        latent: LatentDynamics,
        latent_gain: float = 1.0,
        chi_core: float = 0.6,
        chi_edge_base: float = 2.0,
        chi_edge_drop: float = 1.0,
    ):
        self.nn = nn
        self.latent = latent
        self.latent_gain = jnp.array(latent_gain, dtype=jnp.float64)
        self.chi_core = jnp.array(chi_core, dtype=jnp.float64)
        self.chi_edge_base = jnp.array(chi_edge_base, dtype=jnp.float64)
        self.chi_edge_drop = jnp.array(chi_edge_drop, dtype=jnp.float64)

    def __call__(self, t, y, args):
        (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_fn) = args

        Te_hat = y[:-1]
        z = jnp.clip(y[-1], -10.0, 10.0)

        Te_bc = Te_bc_fn.evaluate(t)
        Te_total = jnp.append(Te_hat * self.Te_scale, Te_bc)
        Te_total = jnp.clip(Te_total, 0.0, 5000.0)

        dTedt = self.compute_physics_tendency(t, Te_total, z, args)

        # derivative clipping
        limit = 1e4
        dTedt = limit * jnp.tanh(dTedt / limit)
        dTe_hat_dt = dTedt / self.Te_scale

        control_vals = ctrl_interp.evaluate(t)
        control_norm = (control_vals - control_means) / (control_stds + 1e-6)
        z_dot = self.latent(z, control_norm)

        rhs = jnp.concatenate([dTe_hat_dt, jnp.array([z_dot], dtype=jnp.float64)])
        rhs = jnp.where(jnp.isfinite(rhs), rhs, 0.0)
        return rhs

    def compute_physics_tendency(self, t, Te_total, z, args):
        (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_fn) = args
        rho = jnp.asarray(rho_vals, dtype=jnp.float64)

        assert rho.shape[0] == Te_total.shape[0]

        # diffusivity profile
        chi_edge = self.chi_edge_base - self.chi_edge_drop * jax.nn.sigmoid(self.latent_gain * z)
        chi_edge = jnp.clip(chi_edge, 0.1, 5.0)
        w_ped = jax.nn.sigmoid((rho - self.ped_center) / self.ped_width)
        chi = self.chi_core + w_ped * (chi_edge - self.chi_core)

        # conservative flux (N-1 faces for N cells with fixed boundary)
        Vprime = jnp.clip(jnp.asarray(Vprime_vals, dtype=jnp.float64), 1e-6, None)
        assert Vprime.shape[0] == rho.shape[0]

        dr = jnp.clip(jnp.diff(rho), 1e-8, None)
        grad_T = jnp.diff(Te_total) / dr
        chi_face = 0.5 * (chi[:-1] + chi[1:])
        Vprime_face = 0.5 * (Vprime[:-1] + Vprime[1:])
        flux_face = -Vprime_face * chi_face * grad_T

        flux_in = jnp.concatenate([jnp.array([0.0], dtype=jnp.float64), flux_face[:-1]])
        flux_out = flux_face
        
        # Volume element for divergence (cell average)
        Vprime_cell = 0.5 * (Vprime[:-1] + Vprime[1:])
        divergence = (flux_out - flux_in) / (Vprime_cell * dr)

        # source NN
        control_vals = ctrl_interp.evaluate(t)
        control_norm = (control_vals - control_means) / (control_stds + 1e-6)
        ne_vals = jnp.clip(ne_interp.evaluate(t), 1e17, 1e21)

        S_nn = jax.vmap(
            lambda r, T, n: self.nn(r, T / self.Te_scale, n / self.ne_scale, control_norm, z)
        )(rho[:-1], Te_total[:-1], ne_vals[:-1])

        assert divergence.shape == (rho.shape[0] - 1,)
        assert S_nn.shape == (rho.shape[0] - 1,)

        return divergence + S_nn

    def compute_source(self, t, Te_total, z, args):
        (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_fn) = args
        rho = jnp.asarray(rho_vals, dtype=jnp.float64)

        control_vals = ctrl_interp.evaluate(t)
        control_norm = (control_vals - control_means) / (control_stds + 1e-6)
        ne_vals = jnp.clip(ne_interp.evaluate(t), 1e17, 1e21)

        S_nn = jax.vmap(
            lambda r, T, n: self.nn(r, T / self.Te_scale, n / self.ne_scale, control_norm, z)
        )(rho[:-1], Te_total[:-1], ne_vals[:-1])
        return S_nn


def pseudo_huber(r, delta):
    delta = jnp.asarray(delta, dtype=jnp.float64)
    return (delta * delta) * (jnp.sqrt(1.0 + (r / delta) ** 2) - 1.0)


def shot_loss(model, bundle: ShotBundle, loss_cfg: LossCfg):
    # Use full padded arrays; create time mask from t_len
    t_len = bundle.t_len
    ts_t_full = bundle.ts_t
    ctrl_t_full = bundle.ctrl_t
    ctrl_vals_full = bundle.ctrl_vals
    ne_vals_full = bundle.ne_vals
    Te_edge_full = bundle.Te_edge

    T_max = ts_t_full.shape[0]
    time_mask = (jnp.arange(T_max, dtype=jnp.int32) < t_len).astype(jnp.float64)
    
    # Construct interpolants on the full (padded) sequences
    ctrl_interp = diffrax.LinearInterpolation(ts=ctrl_t_full, ys=ctrl_vals_full)
    ne_interp = diffrax.LinearInterpolation(ts=ts_t_full, ys=ne_vals_full)
    Te_bc_interp = diffrax.LinearInterpolation(ts=ts_t_full, ys=Te_edge_full)
    
    ode_args = (
        bundle.rho_rom,
        bundle.Vprime_rom,
        ctrl_interp,
        bundle.ctrl_means,
        bundle.ctrl_stds,
        ne_interp,
        Te_bc_interp,
    )
    
    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, jnp.array([bundle.z0], dtype=jnp.float64)])
    
    # dynamic final time: t_end = ts_t_full[t_len-1]
    t_end = jax.lax.dynamic_index_in_dim(ts_t_full, jnp.maximum(t_len - 1, 0), keepdims=False)

    solver = diffrax.Kvaerno5()
    controller = diffrax.PIDController(rtol=1e-3, atol=1e-3)
    saveat = diffrax.SaveAt(ts=ts_t_full)
    dt0 = 1e-4
    max_steps = MAX_SOLVER_STEPS
    
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(model),
        solver,
        t0=ts_t_full[0],
        t1=t_end,
        y0=y0,
        dt0=dt0,
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=max_steps,
        throw=loss_cfg.throw_solver,
        args=ode_args,
    )
    
    is_success = (sol.result == diffrax.RESULTS.successful)
    ys = jnp.nan_to_num(sol.ys, nan=0.0, posinf=0.0, neginf=0.0)
    Te_hats = ys[:, :-1]
    zs = ys[:, -1]

    Te_bc_ts = Te_bc_interp.evaluate(ts_t_full)

    def reconstruct(Te_hat_row, bc_val):
        return jnp.append(Te_hat_row, bc_val / model.Te_scale) * model.Te_scale

    Te_model = jax.vmap(reconstruct)(Te_hats, Te_bc_ts)

    # observation loss with time mask
    mask_obs = bundle.mask[:, bundle.obs_idx].astype(jnp.float64)
    col_weight = jnp.mean(mask_obs, axis=0)
    col_weight = jnp.where(
        jnp.sum(col_weight) > 0,
        col_weight / (jnp.sum(col_weight) + 1e-8),
        jnp.ones_like(col_weight) / col_weight.size,
    )
    weight_grid = (mask_obs * col_weight) * time_mask[:, None]
    resid = Te_model[:, bundle.obs_idx] - bundle.ts_Te[:, bundle.obs_idx]

    huber_delta = loss_cfg.huber_delta
    obs_loss = jnp.sum(weight_grid * pseudo_huber(resid, huber_delta)) / (jnp.sum(weight_grid) + 1e-8)

    # source regularisation
    def eval_source(ti, Te_row, zi):
        return model.compute_source(ti, Te_row, zi, ode_args)

    S_nn_vals = jax.vmap(eval_source)(ts_t_full, Te_model, zs)
    lambda_src = loss_cfg.lambda_src
    src_delta = loss_cfg.src_delta
    
    # Broadcast time mask for source penalty
    src_penalty = lambda_src * jnp.sum(time_mask[:, None] * pseudo_huber(S_nn_vals, src_delta)) / (jnp.sum(time_mask) * S_nn_vals.shape[1] + 1e-8)

    w_loss = 0.0

    # regime supervision
    reg_mask = (bundle.regime_mask * time_mask).astype(jnp.float64)
    y_regime = jnp.where(bundle.regime_ts > 2.5, 1.0, 0.0)
    z_prob = jax.nn.sigmoid(zs)
    lambda_z = loss_cfg.lambda_z
    
    z_loss_num = jnp.sum(reg_mask * (z_prob - y_regime) ** 2)
    z_loss_den = jnp.sum(reg_mask) + 1e-8
    z_loss = lambda_z * z_loss_num / z_loss_den

    z_reg = loss_cfg.lambda_zreg * jnp.mean(zs**2)

    total_loss = obs_loss + src_penalty + w_loss + z_loss + z_reg
    return jnp.where(is_success, total_loss, 1e12)


def batch_loss(model, bundles, loss_cfg):
    losses = jnp.stack([shot_loss(model, b, loss_cfg) for b in bundles])
    return jnp.mean(losses)


def pad_to_max(arrays, mode='constant', constant_values=0.0):
    max_len = max(x.shape[0] for x in arrays)
    out = []
    for x in arrays:
        pad_len = max_len - x.shape[0]
        if pad_len > 0:
            if mode == 'edge':
                # repeat last element
                padding = jnp.repeat(x[-1:], pad_len, axis=0)
                out.append(jnp.concatenate([x, padding], axis=0))
            else:
                padding = jnp.full((pad_len,) + x.shape[1:], constant_values, dtype=x.dtype)
                out.append(jnp.concatenate([x, padding], axis=0))
        else:
            out.append(x)
    return jnp.stack(out)


def load_data(config):
    data_dir = config["data"]["data_dir"]
    shot_list = config["data"]["shots"]
    if shot_list == "all":
        files = glob.glob(os.path.join(data_dir, "*_torax_training.npz"))
    else:
        files = [os.path.join(data_dir, f"{s}_torax_training.npz") for s in shot_list]

    if len(files) == 0:
        raise FileNotFoundError(f"No training packs found in {data_dir}")

    print(f"Loading {len(files)} shots...")

    ref_data = np.load(files[0])
    rho_ref_np = np.array(ref_data["rho"], dtype=float)

    def interp_profile_to_grid(values_t_s, mask_t_s, rho_src, rho_dst):
        Nt = values_t_s.shape[0]
        out_val = np.zeros((Nt, rho_dst.size), dtype=float)
        out_mask = np.zeros_like(out_val)
        for i in range(Nt):
            row_val = values_t_s[i]
            row_mask = mask_t_s[i].astype(bool) & np.isfinite(row_val) & np.isfinite(rho_src)
            if np.count_nonzero(row_mask) == 0:
                continue
            rs = rho_src[row_mask]
            vs = row_val[row_mask]
            ms = mask_t_s[i][row_mask]
            idx = np.argsort(rs)
            rs, vs, ms = rs[idx], vs[idx], ms[idx]
            out_val[i] = np.interp(rho_dst, rs, vs, left=0.0, right=0.0)
            out_mask[i] = np.interp(rho_dst, rs, ms, left=0.0, right=0.0)
        out_mask = (out_mask > 0.5).astype(float)
        return out_val, out_mask

    raw_shots: List[dict] = []
    cov_list = []

    for f in files:
        d = np.load(f)

        ts_t_full = np.array(d["t_ts"], dtype=float)
        ts_Te_raw = np.array(d["Te"], dtype=float)
        mask_raw = np.array(d["Te_mask"], dtype=float)
        ne_raw = np.array(d["ne"], dtype=float)
        ne_mask_raw = np.array(d.get("ne_mask", np.isfinite(ne_raw)), dtype=float)

        mask_full = np.where(np.isnan(ts_Te_raw), 0.0, mask_raw)
        ts_Te_full = np.nan_to_num(ts_Te_raw, nan=0.0, posinf=0.0, neginf=0.0)
        ne_mask_full = np.where(np.isnan(ne_raw), 0.0, ne_mask_raw)
        ne_full = np.nan_to_num(ne_raw, nan=0.0, posinf=0.0, neginf=0.0)

        ctrl_t_full = np.array(d["t"], dtype=float)
        ctrl_vals_full = np.stack([np.array(d[k], dtype=float) for k in CONTROL_NAMES], axis=-1)
        ctrl_vals_full = np.nan_to_num(ctrl_vals_full, nan=0.0, posinf=0.0, neginf=0.0)
        regime_full = np.array(d.get("regime", np.zeros_like(ctrl_t_full)), dtype=float)

        cov_list.append((os.path.basename(f), float(np.mean(mask_full)), mask_full.shape[0], mask_full.shape[1]))

        rho_src = np.array(d["rho"], dtype=float)
        if (rho_src.shape[0] != rho_ref_np.size) or (not np.allclose(rho_src, rho_ref_np)):
            ts_Te_full, mask_full = interp_profile_to_grid(ts_Te_full, mask_full, rho_src, rho_ref_np)
            ne_full, ne_mask_full = interp_profile_to_grid(ne_full, ne_mask_full, rho_src, rho_ref_np)
            Vprime_src = np.array(d["Vprime"], dtype=float)
            Vprime_full = np.interp(rho_ref_np, rho_src, Vprime_src, left=Vprime_src[0], right=Vprime_src[-1])
        else:
            Vprime_full = np.array(d["Vprime"], dtype=float)

        # overlapping time window
        t0 = max(ts_t_full[0], ctrl_t_full[0])
        t1 = min(ts_t_full[-1], ctrl_t_full[-1])
        ts_mask = (ts_t_full >= t0) & (ts_t_full <= t1)
        ctrl_mask = (ctrl_t_full >= t0) & (ctrl_t_full <= t1)

        ts_t = ts_t_full[ts_mask]
        ts_Te = ts_Te_full[ts_mask]
        mask = mask_full[ts_mask]
        ne_vals = ne_full[ts_mask]
        ne_mask = ne_mask_full[ts_mask]

        # enforce monotone time (diffrax requirement)
        order = np.argsort(ts_t)
        ts_t, ts_Te, mask, ne_vals, ne_mask = (
            ts_t[order],
            ts_Te[order],
            mask[order],
            ne_vals[order],
            ne_mask[order],
        )
        keep = np.concatenate([[True], np.diff(ts_t) > 0])
        ts_t, ts_Te, mask, ne_vals, ne_mask = (
            ts_t[keep],
            ts_Te[keep],
            mask[keep],
            ne_vals[keep],
            ne_mask[keep],
        )

        regime_ts = np.interp(ts_t, ctrl_t_full, regime_full)
        regime_mask = (
            ((regime_ts > 0.5) & (regime_ts < 1.5)) | ((regime_ts > 2.5) & (regime_ts < 3.5))
        ).astype(np.float64)

        ctrl_t_trim = ctrl_t_full[ctrl_mask]
        ctrl_vals_trim = ctrl_vals_full[ctrl_mask]

        # resample controls to TS times
        ctrl_vals_ts = np.stack(
            [
                np.interp(ts_t, ctrl_t_trim, ctrl_vals_trim[:, i], left=ctrl_vals_trim[0, i], right=ctrl_vals_trim[-1, i])
                for i in range(ctrl_vals_trim.shape[-1])
            ],
            axis=-1,
        )
        ctrl_means = ctrl_vals_ts.mean(axis=0)
        ctrl_stds = ctrl_vals_ts.std(axis=0)

        raw_shots.append(
            dict(
                ts_t=jnp.array(ts_t),
                ts_Te=ts_Te,
                mask=mask,
                ne_vals=ne_vals,
                ne_mask=ne_mask,
                Vprime=Vprime_full,
                regime_ts=jnp.array(regime_ts),
                regime_mask=jnp.array(regime_mask),
                ctrl_t=jnp.array(ts_t),
                ctrl_vals=jnp.array(ctrl_vals_ts),
                ctrl_means=jnp.array(ctrl_means),
                ctrl_stds=jnp.array(ctrl_stds),
                shot_id=int(os.path.basename(f).split("_")[0]),
            )
        )

    # intersection observed columns
    col_cov_stack = np.stack([s["mask"].mean(axis=0) for s in raw_shots], axis=0)
    col_cov_min = col_cov_stack.min(axis=0)
    tau_cap = float(config["data"].get("intersection_rho_threshold", 0.05))
    I_cap = np.flatnonzero(col_cov_min >= tau_cap)
    if I_cap.size < 2:
        k = min(max(5, rho_ref_np.size // 4), rho_ref_np.size)
        I_cap = np.argsort(col_cov_min)[::-1][:k]
        I_cap = np.sort(I_cap)
    rho_cap = rho_ref_np[I_cap]

    # ROM grid
    n_int = int(config["data"].get("rom_n_interior", 8))
    rho_cap_min = float(rho_cap.min()) if rho_cap.size > 0 else float(rho_ref_np[1])
    if n_int > 0:
        k_int = np.arange(n_int)
        x = 0.5 * (1 - np.cos(np.pi * (k_int + 1) / (n_int + 1)))
        rho_int = rho_cap_min * x
    else:
        rho_int = np.array([], dtype=float)
    rho_rom = np.unique(np.sort(np.concatenate([np.array([0.0]), rho_int, rho_cap, np.array([1.0])])))

    def find_obs_idx(rho_cap_vals, rho_rom_vals):
        idxs = []
        for r in rho_cap_vals:
            idxs.append(int(np.argmin(np.abs(rho_rom_vals - r))))
        if len(idxs) == 0:
            return np.arange(min(5, rho_rom_vals.size), dtype=np.int32)
        # deduplicate
        seen = set()
        out = []
        for j in idxs:
            if j not in seen:
                seen.add(j)
                out.append(j)
        return np.array(out, dtype=np.int32)

    obs_idx = find_obs_idx(rho_cap, rho_rom)
    if obs_idx.size == 0:
        obs_idx = np.arange(min(5, rho_rom.size), dtype=np.int32)

    bundles_list = []
    cap_cov_list = []

    for shot in raw_shots:
        ts_t = shot["ts_t"]
        ts_Te_rom, mask_rom = interp_profile_to_grid(shot["ts_Te"], shot["mask"], rho_ref_np, rho_rom)
        ne_rom, _ = interp_profile_to_grid(shot["ne_vals"], shot["ne_mask"], rho_ref_np, rho_rom)

        Te0 = ts_Te_rom[0]
        if (not np.isfinite(Te0).any()) or (mask_rom[0].sum() == 0):
            Te0 = 100.0 * (1.0 - rho_rom**2) + 10.0

        # edge boundary
        m_edge = mask_rom[:, -1]
        Te_edge = ts_Te_rom[:, -1]
        Te_edge_filled = np.where(m_edge > 0, Te_edge, np.nan)
        if np.all(np.isnan(Te_edge_filled)):
            Te_edge_filled = np.full_like(Te_edge, 50.0)
        else:
            ok = np.isfinite(Te_edge_filled)
            Te_edge_filled = np.where(
                ok,
                Te_edge_filled,
                np.interp(np.array(ts_t), np.array(ts_t)[ok], Te_edge_filled[ok], left=Te_edge_filled[ok][0], right=Te_edge_filled[ok][-1]),
            )

        # SAFE CORE: Use Toroidal Vprime but fix the core singularity
        Vprime_rom = np.interp(rho_rom, rho_ref_np, shot["Vprime"])
        if np.allclose(Vprime_rom, 1.0):
            Vprime_rom = 2.0 * rho_rom
        
        # Robust core floor
        Vprime_rom = np.clip(Vprime_rom, 0.0, None)
        if Vprime_rom.size > 1:
            core_floor = 0.125 * max(Vprime_rom[1], 1e-6)
            Vprime_rom[0] = max(Vprime_rom[0], core_floor)
        Vprime_rom = np.clip(Vprime_rom, 1e-6, None)

        cap_cov = float(mask_rom[:, obs_idx].mean()) if obs_idx.size > 0 else 0.0
        cap_cov_list.append((shot["shot_id"], cap_cov))

        bundles_list.append({
            "ts_t": ts_t,
            "ts_Te": jnp.array(ts_Te_rom),
            "mask": jnp.array(mask_rom),
            "obs_idx": jnp.array(obs_idx, dtype=jnp.int32),
            "regime_ts": shot["regime_ts"],
            "regime_mask": shot["regime_mask"],
            "Te0": jnp.array(Te0),
            "z0": 0.0,
            "latent_idx": jnp.array([], dtype=jnp.int32),
            "latent_proj": jnp.zeros((0, 0), dtype=jnp.float64),
            "rho_rom": jnp.array(rho_rom),
            "Vprime_rom": jnp.array(Vprime_rom),
            "ctrl_t": shot["ctrl_t"],
            "ctrl_vals": shot["ctrl_vals"],
            "ctrl_means": shot["ctrl_means"],
            "ctrl_stds": shot["ctrl_stds"],
            "ne_vals": jnp.array(ne_rom),
            "Te_edge": jnp.array(Te_edge_filled),
            "shot_id": jnp.array(shot["shot_id"]),
            "t_len": len(ts_t),
        })

    # Stack and Pad
    ts_t_stack = pad_to_max([b["ts_t"] for b in bundles_list], mode='edge')
    ts_Te_stack = pad_to_max([b["ts_Te"] for b in bundles_list], mode='constant')
    mask_stack = pad_to_max([b["mask"] for b in bundles_list], mode='constant')
    regime_ts_stack = pad_to_max([b["regime_ts"] for b in bundles_list], mode='edge')
    regime_mask_stack = pad_to_max([b["regime_mask"] for b in bundles_list], mode='constant')
    Te0_stack = jnp.stack([b["Te0"] for b in bundles_list])
    z0_stack = jnp.array([b["z0"] for b in bundles_list])
    
    rho_rom_stack = jnp.stack([b["rho_rom"] for b in bundles_list])
    Vprime_rom_stack = jnp.stack([b["Vprime_rom"] for b in bundles_list])
    
    ctrl_t_stack = pad_to_max([b["ctrl_t"] for b in bundles_list], mode='edge')
    ctrl_vals_stack = pad_to_max([b["ctrl_vals"] for b in bundles_list], mode='edge')
    ctrl_means_stack = jnp.stack([b["ctrl_means"] for b in bundles_list])
    ctrl_stds_stack = jnp.stack([b["ctrl_stds"] for b in bundles_list])
    
    ne_vals_stack = pad_to_max([b["ne_vals"] for b in bundles_list], mode='edge')
    Te_edge_stack = pad_to_max([b["Te_edge"] for b in bundles_list], mode='edge')
    
    shot_id_stack = jnp.stack([b["shot_id"] for b in bundles_list])
    obs_idx_stack = jnp.stack([b["obs_idx"] for b in bundles_list])
    t_len_stack = jnp.array([b["t_len"] for b in bundles_list], dtype=jnp.int32)
    
    latent_idx_stack = jnp.stack([b["latent_idx"] for b in bundles_list]) if bundles_list[0]["latent_idx"].size > 0 else jnp.zeros((len(bundles_list), 0), dtype=jnp.int32)
    latent_proj_stack = jnp.stack([b["latent_proj"] for b in bundles_list]) if bundles_list[0]["latent_proj"].size > 0 else jnp.zeros((len(bundles_list), 0, 0), dtype=jnp.float64)

    stacked_bundle = ShotBundle(
        ts_t_stack,
        ts_Te_stack,
        mask_stack,
        obs_idx_stack,
        regime_ts_stack,
        regime_mask_stack,
        Te0_stack,
        z0_stack,
        latent_idx_stack,
        latent_proj_stack,
        shot_id_stack,
        t_len_stack,
        rho_rom_stack,
        Vprime_rom_stack,
        ctrl_t_stack,
        ctrl_vals_stack,
        ctrl_means_stack,
        ctrl_stds_stack,
        ne_vals_stack,
        Te_edge_stack,
    )

    print(f"[data] Loaded and stacked {len(bundles_list)} shots.")
    print(f"       Max time steps: {ts_t_stack.shape[1]}")
    
    return stacked_bundle, rho_rom, rho_cap, obs_idx


def print_model_and_rho_summary(model: HybridField, rho_rom: jnp.ndarray, rho_cap: jnp.ndarray, obs_idx: jnp.ndarray):
    # model parameter count (approx)
    leaves = jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_inexact_array))
    n_params = int(sum(x.size for x in leaves if hasattr(x, "size")))
    print("\n[model]")
    print(f"  params (inexact leaves): {n_params:,}")
    print(f"  Te_scale={model.Te_scale}, ne_scale={model.ne_scale}, latent_gain={model.latent_gain}")
    print(f"  chi_core={model.chi_core}, chi_edge_base={model.chi_edge_base}, chi_edge_drop={model.chi_edge_drop}")
    print("\n[rho grids]")
    print(f"  rho_rom: N={rho_rom.size}, values={np.array2string(np.array(rho_rom), precision=4, separator=', ')}")
    print(f"  rho_cap: N={rho_cap.size}, values={np.array2string(np.array(rho_cap), precision=4, separator=', ')}")
    print(f"  obs_idx (into rho_rom): N={obs_idx.size}, idx={np.array(obs_idx)}")
    if obs_idx.size > 0:
        rho_obs = np.array(rho_rom)[np.array(obs_idx)]
        print(f"  rho_obs (rho_rom[obs_idx]): N={rho_obs.size}, values={np.array2string(rho_obs, precision=4, separator=', ')}")
    print("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--debug_one_shot", type=int, default=None, help="Override config data.shots with single shot")
    parser.add_argument("--throw", action="store_true", help="Force solver throw=True for debug")
    parser.add_argument("--lbfgs_finetune", action="store_true", help="Run optional single-device L-BFGS finetune after AdamW")
    parser.add_argument("--lbfgs_smoke", action="store_true", help="Quick L-BFGS smoke test: small batch and few iterations")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.debug_one_shot is not None:
        config.setdefault("data", {})
        config["data"]["shots"] = [int(args.debug_one_shot)]

    if args.throw:
        config.setdefault("training", {})
        config["training"]["throw_solver"] = True

    model_id = config["output"].get("model_id", "default_run")
    model_dir = os.path.join(config["output"]["save_dir"], model_id)
    log_dir = os.path.join(config["output"].get("log_dir", "logs"), model_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    failure_dir = os.path.join(log_dir, "failures")
    os.makedirs(failure_dir, exist_ok=True)

    log_every = int(config["training"].get("log_every", 50))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w"), logging.StreamHandler()],
    )

    logging.info(f"Starting training: model_id={model_id}")
    devices = jax.devices()
    n_devices = len(devices)
    logging.info(f"Devices: {devices} ({n_devices} total)")

    want_gpu = str(config.get("system", {}).get("device", "cpu")).lower() == "gpu"
    if want_gpu and not any(d.platform == "gpu" for d in devices):
        raise RuntimeError("GPU requested but JAX is on CPU. Fix CUDA runtime/JAX CUDA wheels.")

    logging.info(f"Config:\n{yaml.dump(config)}")

    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Optional LBFGS smoke: override a few finetune settings to be very small
    if args.lbfgs_smoke:
        config.setdefault("training", {})
        config["training"]["lbfgs_finetune"] = True
        config["training"]["lbfgs_maxiter"] = 5
        config["training"]["lbfgs_history"] = 5
        config["training"]["lbfgs_batch_shots"] = 1
        config["training"]["lbfgs_tol"] = 1e-4

    # Load data (stacked)
    all_bundles, rho_rom, rho_cap, obs_idx = load_data(config)
    n_shots = all_bundles.ts_t.shape[0]
    logging.info(f"Loaded {n_shots} shots (stacked).")

    base_seed = int(config["training"].get("seed", 0))
    best_loss = float("inf")
    best_step = -1

    # model hyperparams from config (fallback to defaults)
    layers = int(config.get("model", {}).get("layers", 64))
    depth = int(config.get("model", {}).get("depth", 3))
    latent_gain = float(config.get("model", {}).get("latent_gain", 1.0))
    source_scale = float(config.get("model", {}).get("source_scale", 3.0e5))

    # loss weights (explicit defaults)
    loss_cfg_base = dict(
        huber_delta=float(config["training"].get("huber_delta", 5.0)),
        lambda_src=float(config["training"].get("lambda_src", 1e-4)),
        src_delta=float(config["training"].get("src_delta", 5.0)),
        lambda_w=float(config["training"].get("lambda_w", 1e-5)),
        model_error_delta=float(config["training"].get("model_error_delta", 10.0)),
        lambda_z=float(config["training"].get("lambda_z", 1e-4)),
        lambda_zreg=float(config["training"].get("lambda_zreg", 1e-4)),
        throw_solver=bool(config["training"].get("throw_solver", False)),
    )

    for restart in range(int(config["training"]["num_restarts"])):
        key = jax.random.PRNGKey(base_seed)
        key = jax.random.fold_in(key, restart)
        key_nn, _ = jax.random.split(key)

        # Initialize model on host/single device first
        model = HybridField(
            nn=SourceNN(key_nn, source_scale=source_scale, layers=layers, depth=depth),
            latent=LatentDynamics(
                alpha=jnp.array(1.0, dtype=jnp.float64),
                beta=jnp.array(1.0, dtype=jnp.float64),
                gamma=jnp.array(1.0, dtype=jnp.float64),
                mu_weights=jax.random.normal(key, (3,), dtype=jnp.float64) * 0.01,
                mu_bias=jnp.array(0.0, dtype=jnp.float64),
                mu_ref=jnp.array(0.0, dtype=jnp.float64),
            ),
            latent_gain=latent_gain,
        )

        # print structure + rho summary (requested)
        print_model_and_rho_summary(model, rho_rom, rho_cap, obs_idx)

        total_steps = int(config["training"]["total_steps"])
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-5,
            peak_value=float(config["training"]["learning_rate"]),
            warmup_steps=int(config["training"]["warmup_steps"]),
            decay_steps=total_steps,
            end_value=1e-6,
        )
        opt_name = str(config["training"]["optimizer"]).lower()
        if not hasattr(optax, opt_name):
            raise ValueError(f"Unknown optax optimizer: {opt_name}")

        optimizer = optax.chain(
            optax.clip_by_global_norm(float(config["training"]["grad_clip"])),
            getattr(optax, opt_name)(learning_rate=schedule, weight_decay=float(config["training"]["weight_decay"])),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

        # Partition model into learnable params (arrays) and static structure
        model_params, model_static = eqx.partition(model, eqx.is_inexact_array)

        # Replicate model parameters and optimizer state across devices
        model_params = jax.device_put_replicated(model_params, devices)
        opt_state = jax.device_put_replicated(opt_state, devices)

        # Optional EMA of parameters across steps (device-wise)
        ema_decay = float(config["training"].get("ema_decay", 0.0))
        if ema_decay > 0.0:
            ema_params = jax.tree_map(lambda x: x, model_params)
        else:
            ema_params = None

        @functools.partial(jax.pmap, axis_name='devices', static_broadcasted_argnums=(3,))
        def make_step(params, st, batch_bundles, loss_cfg):
            # Recombine model
            m = eqx.combine(params, model_static)

            # batch_bundles is (DeviceBatch, ...)
            # We vmap over the device batch
            
            def compute_loss_vmap(m, bundles, cfg):
                # bundles is a ShotBundle where leaves are (Batch, ...)
                # vmap shot_loss over this
                losses = jax.vmap(lambda b: shot_loss(m, b, cfg))(bundles)
                return jnp.mean(losses)

            loss, grads = eqx.filter_value_and_grad(compute_loss_vmap)(m, batch_bundles, loss_cfg)
            
            # Check for NaNs in gradients
            grad_leaves = jax.tree_util.tree_leaves(eqx.filter(grads, eqx.is_inexact_array))
            grad_is_nan = jnp.any(jnp.stack([jnp.any(jnp.isnan(x)) for x in grad_leaves]))
            
            # Cast to float for safe reduction across devices
            grad_is_nan_float = grad_is_nan.astype(jnp.float32)
            grad_is_nan_float = jax.lax.pmax(grad_is_nan_float, axis_name='devices')
            grad_is_nan = grad_is_nan_float > 0.5
            
            # Average gradients across devices
            grads = jax.lax.pmean(grads, axis_name='devices')
            loss = jax.lax.pmean(loss, axis_name='devices')
            
            # Partition model into learnable params (arrays) and static structure (functions/metadata)
            params, static = eqx.partition(m, eqx.is_inexact_array)
            
            def do_update(g, s, p):
                updates, new_s = optimizer.update(g, s, p)
                new_p = eqx.apply_updates(p, updates)
                return new_p, new_s

            def skip_update(g, s, p):
                return p, s

            new_params, new_state = jax.lax.cond(
                grad_is_nan,
                skip_update,
                do_update,
                grads, st, params
            )
            
            # We return params, not the full model, because static parts can't be returned from pmap
            grad_norm = jnp.sqrt(sum(jnp.sum(x**2) for x in grad_leaves))
            grad_norm = jax.lax.pmean(grad_norm, axis_name='devices')
            
            return loss, new_params, new_state, grad_norm, grad_is_nan

        start_time = time.time()
        global_batch_size = int(config["training"]["batch_size"])
        if global_batch_size % n_devices != 0:
            raise ValueError(f"Batch size {global_batch_size} not divisible by {n_devices} devices")
        # Clamp batch size when shots are few to avoid heavy duplication
        if n_shots < global_batch_size:
            clamped = max(n_devices, n_shots * n_devices)
            if clamped != global_batch_size:
                logging.info(f"Clamping batch_size from {global_batch_size} to {clamped} for {n_shots} shots")
                global_batch_size = clamped
        device_batch_size = global_batch_size // n_devices

        for step in range(total_steps):
            batch_indices = np.random.choice(n_shots, global_batch_size, replace=(global_batch_size > n_shots))
            
            # Slice and reshape for pmap
            batch_bundle = jax.tree_map(lambda x: x[batch_indices], all_bundles)
            sharded_bundle = jax.tree_map(lambda x: x.reshape((n_devices, device_batch_size) + x.shape[1:]), batch_bundle)

            # Build a hashable loss config for static broadcast (no dynamic fields)
            lcb = loss_cfg_base
            loss_cfg_step = LossCfg(
                huber_delta=lcb["huber_delta"],
                lambda_src=lcb["lambda_src"],
                src_delta=lcb["src_delta"],
                lambda_w=lcb["lambda_w"],
                model_error_delta=lcb["model_error_delta"],
                lambda_z=lcb["lambda_z"],
                lambda_zreg=lcb["lambda_zreg"],
                throw_solver=lcb["throw_solver"],
            )

            # Pass loss config as static-broadcasted to all devices
            loss, params, opt_state, grad_norm, grad_is_nan = make_step(model_params, opt_state, sharded_bundle, loss_cfg_step)
            model_params = params

            # Update EMA if enabled
            if ema_params is not None:
                ema_params = jax.tree_map(lambda new, ema: ema_decay * ema + (1.0 - ema_decay) * new, model_params, ema_params)
            
            # Take values from first device for logging
            loss_val = float(loss[0])
            grad_norm_val = float(grad_norm[0])
            is_nan_val = bool(grad_is_nan[0])

            if (step % log_every) == 0 or step == total_steps - 1:
                elapsed = time.time() - start_time
                current_lr = float(schedule(step))
                logging.info(f"[restart {restart}] step {step}/{total_steps} loss={loss_val:.6g} grad={grad_norm_val:.4e} lr={current_lr:.2e} nan={is_nan_val} elapsed={elapsed:.1f}s")

            if loss_val < best_loss:
                best_loss = loss_val
                best_step = restart * total_steps + step
                # Save model from first device
                params_save = jax.tree_map(lambda x: x[0], model_params)
                model_save = eqx.combine(params_save, model_static)
                save_path = os.path.join(model_dir, f"{config['output']['model_name']}_best.eqx")
                eqx.tree_serialise_leaves(save_path, model_save)
                logging.info(f"New best model saved: {save_path} loss={best_loss:.6g} step={best_step}")

                # Save EMA snapshot as well if enabled
                if ema_params is not None:
                    ema_params_save = jax.tree_map(lambda x: x[0], ema_params)
                    ema_model_save = eqx.combine(ema_params_save, model_static)
                    ema_path = os.path.join(model_dir, f"{config['output']['model_name']}_best_ema.eqx")
                    eqx.tree_serialise_leaves(ema_path, ema_model_save)
                    logging.info(f"EMA snapshot saved: {ema_path}")

    logging.info(f"Training complete. best_loss={best_loss:.6g} best_step={best_step}")

    # Optional: single-device L-BFGS finetune on a fixed small subset
    if args.lbfgs_finetune or bool(config.get("training", {}).get("lbfgs_finetune", False)):
        try:
            from jaxopt import LBFGS  # type: ignore
        except Exception as e:
            logging.warning("L-BFGS finetune requested but jaxopt is not installed. Skipping. Try: pip install jaxopt")
            return

        # Locate best checkpoint
        best_path = os.path.join(model_dir, f"{config['output']['model_name']}_best.eqx")
        if not os.path.exists(best_path):
            logging.warning(f"No best checkpoint found at {best_path}; skipping L-BFGS finetune.")
            return

        # Recreate a template model and load best weights
        key_nn = jax.random.PRNGKey(int(config["training"].get("seed", 0)))
        template_model = HybridField(
            nn=SourceNN(key_nn, source_scale=source_scale, layers=layers, depth=depth),
            latent=LatentDynamics(
                alpha=jnp.array(1.0, dtype=jnp.float64),
                beta=jnp.array(1.0, dtype=jnp.float64),
                gamma=jnp.array(1.0, dtype=jnp.float64),
                mu_weights=jnp.zeros((3,), dtype=jnp.float64),
                mu_bias=jnp.array(0.0, dtype=jnp.float64),
                mu_ref=jnp.array(0.0, dtype=jnp.float64),
            ),
            latent_gain=latent_gain,
        )
        model_best = eqx.tree_deserialise_leaves(best_path, template_model)

        # Fixed, small batch of shots for deterministic gradients
        n_shots = all_bundles.ts_t.shape[0]
        k = int(config.get("training", {}).get("lbfgs_batch_shots", 1))
        k = max(1, min(k, n_shots))
        idxs = jnp.arange(k)
        fixed_bundle = jax.tree_map(lambda x: x[idxs], all_bundles)

        # Loss configuration (same as training, static)
        lcb = dict(loss_cfg_base)
        loss_cfg_ft = LossCfg(
            huber_delta=lcb["huber_delta"],
            lambda_src=lcb["lambda_src"],
            src_delta=lcb["src_delta"],
            lambda_w=lcb["lambda_w"],
            model_error_delta=lcb["model_error_delta"],
            lambda_z=lcb["lambda_z"],
            lambda_zreg=lcb["lambda_zreg"],
            throw_solver=lcb["throw_solver"],
        )

        # Select trainable block: SourceNN last layer + LatentDynamics
        def extract_train_vars(m: HybridField):
            w = m.nn.mlp.layers[-1].weight
            b = m.nn.mlp.layers[-1].bias
            ld = m.latent
            return (w, b, ld.alpha, ld.beta, ld.gamma, ld.mu_weights, ld.mu_bias, ld.mu_ref)

        def set_train_vars(m: HybridField, vars_tuple):
            (w, b, alpha, beta, gamma, mu_w, mu_b, mu_ref) = vars_tuple
            m1 = eqx.tree_at(lambda mm: mm.nn.mlp.layers[-1].weight, m, w)
            m1 = eqx.tree_at(lambda mm: mm.nn.mlp.layers[-1].bias, m1, b)
            m1 = eqx.tree_at(lambda mm: mm.latent.alpha, m1, alpha)
            m1 = eqx.tree_at(lambda mm: mm.latent.beta, m1, beta)
            m1 = eqx.tree_at(lambda mm: mm.latent.gamma, m1, gamma)
            m1 = eqx.tree_at(lambda mm: mm.latent.mu_weights, m1, mu_w)
            m1 = eqx.tree_at(lambda mm: mm.latent.mu_bias, m1, mu_b)
            m1 = eqx.tree_at(lambda mm: mm.latent.mu_ref, m1, mu_ref)
            return m1

        base_vars = extract_train_vars(model_best)

        # Objective over trainable vars only (others frozen)
        def objective(train_vars):
            m_full = set_train_vars(model_best, train_vars)
            losses = jax.vmap(lambda b: shot_loss(m_full, b, loss_cfg_ft))(fixed_bundle)
            return jnp.mean(losses)

        # Configure L-BFGS
        maxiter = int(config.get("training", {}).get("lbfgs_maxiter", 50))
        history = int(config.get("training", {}).get("lbfgs_history", 10))
        tol = float(config.get("training", {}).get("lbfgs_tol", 1e-6))
        solver = LBFGS(fun=objective, maxiter=maxiter, tol=tol, history_size=history)

        logging.info(f"[lbfgs] Starting finetune on {k} shot(s): maxiter={maxiter}, history={history}, tol={tol}")
        vars_opt, state = solver.run(base_vars)
        loss_ft = float(state.value)
        logging.info(f"[lbfgs] Finetune complete. loss={loss_ft:.6g}, iters={int(state.iter_num)}")

        model_ft = set_train_vars(model_best, vars_opt)
        save_path_ft = os.path.join(model_dir, f"{config['output']['model_name']}_finetuned.eqx")
        eqx.tree_serialise_leaves(save_path_ft, model_ft)
        logging.info(f"[lbfgs] Saved finetuned model: {save_path_ft}")


if __name__ == "__main__":
    main()
