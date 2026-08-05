"""Loss functions and trajectory evaluation."""
# fusion_ode_identification/loss.py

import jax
import jax.numpy as jnp

from .types import LossCfg, ShotBundle, ShotEval, IMEXConfig
from .imex_solver import IMEXIntegrator
from .interp import LinearInterpolation
from .model import normalize_observed_signal, smooth_clamp

# Fixed robust scales for loss normalization (training-corpus magnitudes,
# frozen; see paper Sec. identification). Losses are averages of O(1)
# quantities so task weights are interpretable.
S_TE = 100.0   # eV
S_SRC = 1.0e4  # eV/s


def pseudo_huber(r, delta):
    delta = jnp.asarray(delta, dtype=jnp.float64)
    return (delta * delta) * (jnp.sqrt(1.0 + (r / delta) ** 2) - 1.0)


# This branch is IMEX-only.
# The previous Diffrax-based time integrators have been removed.


# ==================== IMEX-Based Loss and Evaluation ====================


def _observation_weight_grid(mask_obs, time_mask=None, reliable_mask=None):
    mask_use = mask_obs.astype(jnp.float64)
    if time_mask is not None:
        mask_use = mask_use * time_mask[:, None].astype(jnp.float64)
        denom = jnp.sum(time_mask.astype(jnp.float64))
    else:
        denom = jnp.array(mask_use.shape[0], dtype=jnp.float64)
    if reliable_mask is not None:
        mask_use = mask_use * reliable_mask[None, :].astype(jnp.float64)

    denom = jnp.maximum(denom, 1.0)
    col_cov = jnp.sum(mask_use, axis=0) / denom
    has_obs = col_cov > 0
    inv = jnp.where(has_obs, 1.0 / (col_cov + 1e-8), 0.0)
    inv_sum = jnp.sum(inv)
    col_weight = jnp.where(
        inv_sum > 0,
        inv / (inv_sum + 1e-8),
        jnp.ones_like(col_cov) / jnp.maximum(col_cov.size, 1),
    )
    return mask_use * col_weight[None, :]


# Controls are scaled by fixed physical constants (see model.CONTROL_SCALES),
# so the same normalized series feeds both the source network and the latent
# drive; no separate latent feature construction is needed.


def shot_loss_imex(model, bundle: ShotBundle, loss_cfg: LossCfg, imex_cfg: IMEXConfig):
    """
    Shot loss using IMEX time integration.
    
    Args:
        model: HybridField model
        bundle: Shot data bundle
        loss_cfg: Loss configuration
        imex_cfg: IMEX solver configuration (theta, dt_base, etc.)
    """
    t_len = jnp.asarray(bundle.t_len, dtype=jnp.int32)
    ts_t_full = bundle.ts_t
    ctrl_t_full = bundle.ctrl_t
    ctrl_vals_full = bundle.ctrl_vals
    ne_vals_full = bundle.ne_vals
    Te_edge_full = bundle.Te_edge
    dalpha_full = bundle.dalpha_ts
    ts_Te_full = bundle.ts_Te
    mask_full = bundle.mask
    reliable_mask_full = bundle.reliable_mask

    T_max = ts_t_full.shape[0]
    time_mask = (jnp.arange(T_max, dtype=jnp.int32) < t_len).astype(jnp.float64)
    # Freeze integration after the valid time window so padded tail can't destabilize the solver.
    active_mask = jnp.arange(T_max - 1, dtype=jnp.int32) < jnp.maximum(t_len - 1, 0)

    # Evaluate controls at Te time grid once; solver uses cheap blending within intervals.
    ctrl_interp = LinearInterpolation(ts=ctrl_t_full, ys=ctrl_vals_full)
    ctrl_vals_ts = ctrl_interp.evaluate(ts_t_full)
    ctrl_norm_ts = (ctrl_vals_ts - bundle.ctrl_means) / (bundle.ctrl_stds + 1e-6)
    ctrl_norm_ts = jnp.clip(ctrl_norm_ts, -10.0, 10.0)
    ne_edge_ts = ne_vals_full[:, -1]
    # Latent drive uses its own physically scaled feature set.
    latent_features_ts = bundle.drive_feats

    # Precompute static geometry factors once per shot (used by diffusion operator).
    rho = bundle.rho_rom
    Vprime = jnp.clip(bundle.Vprime_rom, 1e-6, None)
    dr = jnp.diff(rho)
    dr = jnp.clip(dr, 1e-6 * jnp.max(dr) + 1e-12, None)
    Vprime_face = 0.5 * (Vprime[:-1] + Vprime[1:])
    from .imex_solver import cell_volumes
    denom_raw = cell_volumes(rho, Vprime, dr)
    denom_floor = jnp.maximum(1e-4 * jnp.max(denom_raw), 1e-10)
    denom = jnp.maximum(denom_raw, denom_floor)
    ode_args_geom = (rho, Vprime, dr, Vprime_face, Vprime_face, denom)

    # Causal initialization: lowest equilibrium of the latent vector field
    # at the initial drive (gated discharges start in state L).
    z0 = model.latent.initial_state(latent_features_ts[0])
    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, jnp.atleast_1d(z0)])

    t0 = ts_t_full[0]
    t1 = ts_t_full[-1]

    # Create IMEX integrator
    integrator = IMEXIntegrator(
        theta=imex_cfg.theta,
        dt_base=imex_cfg.dt_base,
        max_steps=imex_cfg.max_steps,
        rtol=imex_cfg.rtol,
        atol=imex_cfg.atol,
        substeps=getattr(imex_cfg, "substeps", 1),
    )

    # Integrate
    sol = integrator.integrate(
        t_span=(t0, t1),
        y0=y0,
        saveat=ts_t_full,
        model=model,
        Te_edge_ts=Te_edge_full,
        ctrl_norm_ts=ctrl_norm_ts,
        ne_ts=ne_vals_full,
        latent_features_ts=latent_features_ts,
        args=ode_args_geom,
        active_mask=active_mask,
    )

    is_success = sol.success

    # Geometry diagnostics (computed outside the solver for logging/debug).
    rho_diag = bundle.rho_rom
    Vprime_diag = jnp.clip(bundle.Vprime_rom, 0.0, None)
    dr_diag = jnp.diff(rho_diag)
    dr_floor = 1e-6 * jnp.max(dr_diag) + 1e-12
    min_dr = jnp.min(dr_diag)
    vprime_floor = 1e-6
    min_Vprime = jnp.min(Vprime_diag)
    Vprime_cell = 0.5 * (Vprime_diag[:-1] + Vprime_diag[1:])
    denom_raw = Vprime_cell * dr_diag
    denom_floor = jnp.maximum(1e-4 * jnp.max(denom_raw), 1e-10)
    min_denom = jnp.min(denom_raw)
    dr_floor_hit = (min_dr < dr_floor).astype(jnp.float64)
    vprime_floor_hit = (min_Vprime < vprime_floor).astype(jnp.float64)
    denom_floor_hit = (min_denom < denom_floor).astype(jnp.float64)

    def success_branch(_):
        ys = jnp.nan_to_num(sol.ys, nan=0.0, posinf=0.0, neginf=0.0)
        Te_hats = ys[:, :-1]
        zs = ys[:, -1]

        Te_bc_ts = Te_edge_full

        def reconstruct(Te_hat_row, bc_val):
            Te_hat_row = smooth_clamp(Te_hat_row, 0.0, 5000.0 / model.Te_scale)
            return jnp.append(Te_hat_row, bc_val / model.Te_scale) * model.Te_scale

        Te_model = jax.vmap(reconstruct)(Te_hats, Te_bc_ts)

        # Supervise all interior radii (exclude Dirichlet boundary node).
        # Weight by actual data availability per radius to handle sparse observability.
        N = ts_Te_full.shape[1]
        use_cols = jnp.arange(N - 1, dtype=jnp.int32)

        tm = time_mask.astype(jnp.float64)
        tm2 = tm[:, None]

        weight_grid = _observation_weight_grid(
            mask_full[:, use_cols],
            time_mask=tm,
            reliable_mask=reliable_mask_full[use_cols],
        )

        resid = Te_model[:, use_cols] - ts_Te_full[:, use_cols]

        abs_resid = jnp.abs(resid)
        Te_data = ts_Te_full[:, use_cols]
        denom = jnp.maximum(jnp.abs(Te_data), 50.0)

        wsum = jnp.sum(weight_grid) + 1e-8
        mae_eV = jnp.sum(weight_grid * abs_resid) / wsum
        mae_pct = 100.0 * jnp.sum(weight_grid * (abs_resid / denom)) / wsum

        huber_delta = loss_cfg.huber_delta
        obs_loss = jnp.sum(weight_grid * pseudo_huber(resid / S_TE, huber_delta)) / wsum

        S_nn_vals = jax.vmap(lambda Te_row, zi, cn, ne: model.compute_source_from_values(bundle.rho_rom, Te_row, zi, ne, cn))(
            Te_model,
            zs,
            ctrl_norm_ts,
            ne_vals_full,
        )
        lambda_src = loss_cfg.lambda_src
        src_delta = loss_cfg.src_delta

        src_wsum = (jnp.sum(tm) * S_nn_vals.shape[1]) + 1e-8
        src_penalty = lambda_src * jnp.sum(tm2 * pseudo_huber(S_nn_vals / S_SRC, src_delta)) / src_wsum

        # Physics diagnostics: mean magnitudes (time-masked for padded arrays)
        div_vals = jax.vmap(lambda Te_row, zi: model.compute_divergence_from_values(bundle.rho_rom, bundle.Vprime_rom, Te_row, zi))(
            Te_model,
            zs,
        )
        div_wsum = (jnp.sum(tm) * div_vals.shape[1]) + 1e-8
        mean_abs_div = jnp.sum(tm2 * jnp.abs(div_vals)) / div_wsum
        mean_abs_src = jnp.sum(tm2 * jnp.abs(S_nn_vals)) / src_wsum
        src_over_diff = mean_abs_src / (mean_abs_div + 1e-8)

        # z regularization
        # - lambda_z: smoothness in time (encourage slowly-varying latent)
        # - lambda_zreg: magnitude penalty (keep z bounded)
        z_reg = loss_cfg.lambda_zreg * (jnp.sum(tm * (zs**2)) / (jnp.sum(tm) + 1e-8))
        dz = zs[1:] - zs[:-1]
        tm_dz = tm[:-1]
        z_smooth = loss_cfg.lambda_z * (jnp.sum(tm_dz * (dz**2)) / (jnp.sum(tm_dz) + 1e-8))

        regime_mask = bundle.regime_mask.astype(jnp.float64) * tm
        regime_target = jnp.where(bundle.regime_ts > 2.0, 1.0, 0.0)
        regime_logits = jax.vmap(model.compute_regime_logit)(zs)
        regime_bce = jnp.maximum(regime_logits, 0.0) - regime_logits * regime_target + jnp.log1p(jnp.exp(-jnp.abs(regime_logits)))
        regime_weight = loss_cfg.lambda_regime + loss_cfg.lambda_pH
        regime_penalty = regime_weight * (jnp.sum(regime_mask * regime_bce) / (jnp.sum(regime_mask) + 1e-8))

        dalpha_target = normalize_observed_signal(dalpha_full)
        dalpha_hat = jax.vmap(lambda zi, cn, Tee, nee: model.compute_aux_dalpha_hat(zi, cn, Tee, nee))(
            zs,
            ctrl_norm_ts,
            Te_edge_full,
            ne_edge_ts,
        )
        dalpha_penalty = loss_cfg.lambda_dalpha * (jnp.sum(tm * ((dalpha_hat - dalpha_target) ** 2)) / (jnp.sum(tm) + 1e-8))

        total_loss = obs_loss + src_penalty + z_reg + z_smooth + regime_penalty + dalpha_penalty

        diag = jnp.array(
            [
                bundle.shot_id.astype(jnp.float64),
                t_len.astype(jnp.float64),
                jnp.array(imex_cfg.dt_base, dtype=jnp.float64),
                mae_eV,
                mae_pct,
                mean_abs_div,
                mean_abs_src,
                src_over_diff,
                min_dr,
                min_Vprime,
                min_denom,
                dr_floor_hit,
                vprime_floor_hit,
                denom_floor_hit,
            ],
            dtype=jnp.float64,
        )

        return total_loss, jnp.array(1, dtype=jnp.int32), diag

    def fail_branch(_):
        diag = jnp.array(
            [
                bundle.shot_id.astype(jnp.float64),
                t_len.astype(jnp.float64),
                jnp.array(imex_cfg.dt_base, dtype=jnp.float64),
                jnp.array(0.0, dtype=jnp.float64),
                jnp.array(0.0, dtype=jnp.float64),
                jnp.array(0.0, dtype=jnp.float64),
                jnp.array(0.0, dtype=jnp.float64),
                jnp.array(0.0, dtype=jnp.float64),
                min_dr,
                min_Vprime,
                min_denom,
                dr_floor_hit,
                vprime_floor_hit,
                denom_floor_hit,
            ],
            dtype=jnp.float64,
        )
        return jax.lax.stop_gradient(jnp.array(1e12, dtype=jnp.float64)), jnp.array(0, dtype=jnp.int32), diag

    return jax.lax.cond(is_success, success_branch, fail_branch, operand=None)


def batch_loss_imex(model, bundles, loss_cfg, imex_cfg: IMEXConfig):
    """Batch loss using IMEX integration."""
    losses, oks, _ = jax.vmap(lambda b: shot_loss_imex(model, b, loss_cfg, imex_cfg))(bundles)
    return jnp.mean(losses), jnp.mean(oks)


def eval_shot_trajectory_imex(model, bundle: ShotBundle, loss_cfg: LossCfg, imex_cfg: IMEXConfig) -> ShotEval:
    """
    Evaluate single shot trajectory using IMEX integration.
    """
    L = int(jnp.asarray(bundle.t_len))
    L = max(1, L)

    ts_t_full = bundle.ts_t[:L]
    ctrl_t_full = bundle.ctrl_t[:L]
    ctrl_vals_full = bundle.ctrl_vals[:L]
    ne_vals_full = bundle.ne_vals[:L]
    Te_edge_full = bundle.Te_edge[:L]
    dalpha_full = bundle.dalpha_ts[:L]
    ts_Te_full = bundle.ts_Te[:L]
    mask_full = bundle.mask[:L]
    reliable_mask_full = bundle.reliable_mask

    ctrl_interp = LinearInterpolation(ts=ctrl_t_full, ys=ctrl_vals_full)
    ctrl_vals_ts = ctrl_interp.evaluate(ts_t_full)
    ctrl_norm_ts = (ctrl_vals_ts - bundle.ctrl_means) / (bundle.ctrl_stds + 1e-6)
    ctrl_norm_ts = jnp.clip(ctrl_norm_ts, -10.0, 10.0)
    ne_edge_ts = ne_vals_full[:, -1]
    # Latent drive uses its own physically scaled feature set.
    latent_features_ts = bundle.drive_feats

    rho = bundle.rho_rom
    Vprime = jnp.clip(bundle.Vprime_rom, 1e-6, None)
    dr = jnp.diff(rho)
    dr = jnp.clip(dr, 1e-6 * jnp.max(dr) + 1e-12, None)
    Vprime_face = 0.5 * (Vprime[:-1] + Vprime[1:])
    from .imex_solver import cell_volumes
    denom_raw = cell_volumes(rho, Vprime, dr)
    denom_floor = jnp.maximum(1e-4 * jnp.max(denom_raw), 1e-10)
    denom = jnp.maximum(denom_raw, denom_floor)
    ode_args_geom = (rho, Vprime, dr, Vprime_face, Vprime_face, denom)

    # Causal initialization: lowest equilibrium of the latent vector field
    # at the initial drive (gated discharges start in state L).
    z0 = model.latent.initial_state(latent_features_ts[0])
    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, jnp.atleast_1d(z0)])

    t0 = ts_t_full[0]
    t1 = ts_t_full[-1]

    # Create IMEX integrator
    integrator = IMEXIntegrator(
        theta=imex_cfg.theta,
        dt_base=imex_cfg.dt_base,
        max_steps=imex_cfg.max_steps,
        rtol=imex_cfg.rtol,
        atol=imex_cfg.atol,
        substeps=getattr(imex_cfg, "substeps", 1),
    )

    # Integrate
    sol = integrator.integrate(
        t_span=(t0, t1),
        y0=y0,
        saveat=ts_t_full,
        model=model,
        Te_edge_ts=Te_edge_full,
        ctrl_norm_ts=ctrl_norm_ts,
        ne_ts=ne_vals_full,
        latent_features_ts=latent_features_ts,
        args=ode_args_geom,
    )

    use_cols = jnp.arange(ts_Te_full.shape[1] - 1, dtype=jnp.int32)

    if not sol.success:
        return ShotEval(
            ok=jnp.array(0, dtype=jnp.int32),
            loss=jnp.array(1e12, dtype=jnp.float64),
            mae_eV=jnp.array(0.0, dtype=jnp.float64),
            mae_pct=jnp.array(0.0, dtype=jnp.float64),
            Te_model=jnp.zeros_like(ts_Te_full),
            resid=jnp.zeros((ts_Te_full.shape[0], use_cols.shape[0]), dtype=jnp.float64),
            ts_t=ts_t_full,
            obs_idx=use_cols,
            z_ts=jnp.zeros_like(ts_t_full),
        )

    ys = jnp.nan_to_num(sol.ys, nan=0.0, posinf=0.0, neginf=0.0)
    Te_hats = ys[:, :-1]
    zs = ys[:, -1]

    Te_bc_ts = Te_edge_full

    def reconstruct(Te_hat_row, bc_val):
        Te_hat_row = jnp.clip(Te_hat_row, 0.0, 5000.0 / model.Te_scale)
        return jnp.append(Te_hat_row, bc_val / model.Te_scale) * model.Te_scale

    Te_model = jax.vmap(reconstruct)(Te_hats, Te_bc_ts)

    weight_grid = _observation_weight_grid(mask_full[:, use_cols], reliable_mask=reliable_mask_full[use_cols])
    resid = Te_model[:, use_cols] - ts_Te_full[:, use_cols]

    abs_resid = jnp.abs(resid)
    Te_data = ts_Te_full[:, use_cols]
    denom = jnp.maximum(jnp.abs(Te_data), 50.0)
    mae_eV = jnp.sum(weight_grid * abs_resid) / (jnp.sum(weight_grid) + 1e-8)
    mae_pct = 100.0 * jnp.sum(weight_grid * (abs_resid / denom)) / (jnp.sum(weight_grid) + 1e-8)

    huber_delta = loss_cfg.huber_delta
    obs_loss = jnp.sum(weight_grid * pseudo_huber(resid / S_TE, huber_delta)) / (jnp.sum(weight_grid) + 1e-8)

    S_nn_vals = jax.vmap(lambda Te_row, zi, cn, ne: model.compute_source_from_values(bundle.rho_rom, Te_row, zi, ne, cn))(
        Te_model,
        zs,
        ctrl_norm_ts,
        ne_vals_full,
    )
    lambda_src = loss_cfg.lambda_src
    src_delta = loss_cfg.src_delta
    src_penalty = lambda_src * jnp.sum(pseudo_huber(S_nn_vals / S_SRC, src_delta)) / (S_nn_vals.size + 1e-8)

    z_reg = loss_cfg.lambda_zreg * jnp.mean(zs**2)
    dz = zs[1:] - zs[:-1]
    z_smooth = loss_cfg.lambda_z * jnp.mean(dz**2)

    # Regime and D-alpha terms mirror the training loss so that checkpoint
    # selection (validation loss) also reflects L/H discrimination quality.
    regime_mask = bundle.regime_mask[:L].astype(jnp.float64)
    regime_target = jnp.where(bundle.regime_ts[:L] > 2.0, 1.0, 0.0)
    regime_logits = jax.vmap(model.compute_regime_logit)(zs)
    regime_bce = jnp.maximum(regime_logits, 0.0) - regime_logits * regime_target + jnp.log1p(jnp.exp(-jnp.abs(regime_logits)))
    regime_weight = loss_cfg.lambda_regime + loss_cfg.lambda_pH
    regime_penalty = regime_weight * (jnp.sum(regime_mask * regime_bce) / (jnp.sum(regime_mask) + 1e-8))

    dalpha_target = normalize_observed_signal(dalpha_full)
    dalpha_hat = jax.vmap(lambda zi, cn, Tee, nee: model.compute_aux_dalpha_hat(zi, cn, Tee, nee))(
        zs,
        ctrl_norm_ts,
        Te_edge_full,
        ne_edge_ts,
    )
    dalpha_penalty = loss_cfg.lambda_dalpha * jnp.mean((dalpha_hat - dalpha_target) ** 2)

    total_loss = obs_loss + src_penalty + z_reg + z_smooth + regime_penalty + dalpha_penalty

    return ShotEval(
        ok=jnp.array(1, dtype=jnp.int32),
        loss=total_loss,
        mae_eV=mae_eV,
        mae_pct=mae_pct,
        Te_model=Te_model,
        resid=resid,
        ts_t=ts_t_full,
        obs_idx=use_cols,
        z_ts=zs,
    )

