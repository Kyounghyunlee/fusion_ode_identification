"""Loss functions and trajectory evaluation."""
# fusion_ode_identification/loss.py

import jax
import jax.numpy as jnp

from .types import LossCfg, ShotBundle, ShotEval, IMEXConfig
from .imex_solver import IMEXIntegrator
from .interp import LinearInterpolation
from .model import smooth_clamp


def pseudo_huber(r, delta):
    delta = jnp.asarray(delta, dtype=jnp.float64)
    return (delta * delta) * (jnp.sqrt(1.0 + (r / delta) ** 2) - 1.0)


# This branch is IMEX-only.
# The previous Diffrax-based time integrators have been removed.


# ==================== IMEX-Based Loss and Evaluation ====================


def shot_loss_imex(model, bundle: ShotBundle, loss_cfg: LossCfg, imex_cfg: IMEXConfig):
    """
    Shot loss using IMEX time integration.
    
    Args:
        model: HybridField model
        bundle: Shot data bundle
        loss_cfg: Loss configuration
        imex_cfg: IMEX solver configuration (theta, dt_base, etc.)
    """
    t_len = bundle.t_len
    ts_t_full = bundle.ts_t
    ctrl_t_full = bundle.ctrl_t
    ctrl_vals_full = bundle.ctrl_vals
    ne_vals_full = bundle.ne_vals
    Te_edge_full = bundle.Te_edge
    ts_Te_full = bundle.ts_Te
    mask_full = bundle.mask

    T_max = ts_t_full.shape[0]
    time_mask = (jnp.arange(T_max, dtype=jnp.int32) < t_len).astype(jnp.float64)

    ctrl_interp = LinearInterpolation(ts=ctrl_t_full, ys=ctrl_vals_full)
    ne_interp = LinearInterpolation(ts=ts_t_full, ys=ne_vals_full)
    Te_bc_interp = LinearInterpolation(ts=ts_t_full, ys=Te_edge_full)

    ode_args = (
        bundle.rho_rom,
        bundle.Vprime_rom,
        ctrl_interp,
        bundle.ctrl_means,
        bundle.ctrl_stds,
        ne_interp,
        Te_bc_interp,
    )

    z0_arr = jnp.atleast_1d(jnp.asarray(bundle.z0, dtype=jnp.float64))
    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, z0_arr])

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
        Te_edge_interp=lambda t: Te_bc_interp.evaluate(t),
        args=ode_args,
    )

    is_success = sol.success

    def success_branch(_):
        ys = jnp.nan_to_num(sol.ys, nan=0.0, posinf=0.0, neginf=0.0)
        Te_hats = ys[:, :-1]
        zs = ys[:, -1]

        Te_bc_ts = Te_bc_interp.evaluate(sol.ts)

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

        mask_use = mask_full[:, use_cols].astype(jnp.float64) * tm2

        col_cov = jnp.mean(mask_use, axis=0)  # (N-1,)
        col_weight = jnp.where(
            jnp.sum(col_cov) > 0,
            col_cov / (jnp.sum(col_cov) + 1e-8),
            jnp.ones_like(col_cov) / col_cov.size,
        )
        weight_grid = mask_use * col_weight[None, :]

        resid = Te_model[:, use_cols] - ts_Te_full[:, use_cols]

        abs_resid = jnp.abs(resid)
        Te_data = ts_Te_full[:, use_cols]
        denom = jnp.maximum(jnp.abs(Te_data), 50.0)

        wsum = jnp.sum(weight_grid) + 1e-8
        mae_eV = jnp.sum(weight_grid * abs_resid) / wsum
        mae_pct = 100.0 * jnp.sum(weight_grid * (abs_resid / denom)) / wsum

        huber_delta = loss_cfg.huber_delta
        obs_loss = jnp.sum(weight_grid * pseudo_huber(resid, huber_delta)) / wsum

        def eval_source(ti, Te_row, zi):
            return model.compute_source(ti, Te_row, zi, ode_args)

        S_nn_vals = jax.vmap(eval_source)(sol.ts, Te_model, zs)
        lambda_src = loss_cfg.lambda_src
        src_delta = loss_cfg.src_delta

        src_wsum = (jnp.sum(tm) * S_nn_vals.shape[1]) + 1e-8
        src_penalty = lambda_src * jnp.sum(tm2 * pseudo_huber(S_nn_vals, src_delta)) / src_wsum

        # Physics diagnostics: mean magnitudes (time-masked for padded arrays)
        def eval_divergence(ti, Te_row, zi):
            return model.compute_divergence_only(ti, Te_row, zi, ode_args)

        div_vals = jax.vmap(eval_divergence)(sol.ts, Te_model, zs)
        div_wsum = (jnp.sum(tm) * div_vals.shape[1]) + 1e-8
        mean_abs_div = jnp.sum(tm2 * jnp.abs(div_vals)) / div_wsum
        mean_abs_src = jnp.sum(tm2 * jnp.abs(S_nn_vals)) / src_wsum
        src_over_diff = mean_abs_src / (mean_abs_div + 1e-8)

        # z regularization
        z_reg = loss_cfg.lambda_zreg * (jnp.sum(tm * (zs**2)) / (jnp.sum(tm) + 1e-8))

        total_loss = obs_loss + src_penalty + z_reg

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
    ts_Te_full = bundle.ts_Te[:L]
    mask_full = bundle.mask[:L]

    ctrl_interp = LinearInterpolation(ts=ctrl_t_full, ys=ctrl_vals_full)
    ne_interp = LinearInterpolation(ts=ts_t_full, ys=ne_vals_full)
    Te_bc_interp = LinearInterpolation(ts=ts_t_full, ys=Te_edge_full)

    ode_args = (
        bundle.rho_rom,
        bundle.Vprime_rom,
        ctrl_interp,
        bundle.ctrl_means,
        bundle.ctrl_stds,
        ne_interp,
        Te_bc_interp,
    )

    z0_arr = jnp.atleast_1d(jnp.asarray(bundle.z0, dtype=jnp.float64))
    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, z0_arr])

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
        Te_edge_interp=lambda t: Te_bc_interp.evaluate(t),
        args=ode_args,
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

    Te_bc_ts = Te_bc_interp.evaluate(sol.ts)

    def reconstruct(Te_hat_row, bc_val):
        Te_hat_row = jnp.clip(Te_hat_row, 0.0, 5000.0 / model.Te_scale)
        return jnp.append(Te_hat_row, bc_val / model.Te_scale) * model.Te_scale

    Te_model = jax.vmap(reconstruct)(Te_hats, Te_bc_ts)

    mask_obs = mask_full[:, use_cols].astype(jnp.float64)
    col_cov = jnp.mean(mask_obs, axis=0)
    col_weight = jnp.where(
        jnp.sum(col_cov) > 0,
        col_cov / (jnp.sum(col_cov) + 1e-8),
        jnp.ones_like(col_cov) / col_cov.size,
    )
    weight_grid = mask_obs * col_weight[None, :]
    resid = Te_model[:, use_cols] - ts_Te_full[:, use_cols]

    abs_resid = jnp.abs(resid)
    Te_data = ts_Te_full[:, use_cols]
    denom = jnp.maximum(jnp.abs(Te_data), 50.0)
    mae_eV = jnp.sum(weight_grid * abs_resid) / (jnp.sum(weight_grid) + 1e-8)
    mae_pct = 100.0 * jnp.sum(weight_grid * (abs_resid / denom)) / (jnp.sum(weight_grid) + 1e-8)

    huber_delta = loss_cfg.huber_delta
    obs_loss = jnp.sum(weight_grid * pseudo_huber(resid, huber_delta)) / (jnp.sum(weight_grid) + 1e-8)

    def eval_source(ti, Te_row, zi):
        return model.compute_source(ti, Te_row, zi, ode_args)

    S_nn_vals = jax.vmap(eval_source)(sol.ts, Te_model, zs)
    lambda_src = loss_cfg.lambda_src
    src_delta = loss_cfg.src_delta
    src_penalty = lambda_src * jnp.sum(pseudo_huber(S_nn_vals, src_delta)) / (S_nn_vals.size + 1e-8)

    z_reg = loss_cfg.lambda_zreg * jnp.mean(zs**2)

    total_loss = obs_loss + src_penalty + z_reg

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

