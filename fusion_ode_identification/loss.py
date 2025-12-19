"""Loss functions and trajectory evaluation."""
# fusion_ode_identification/loss.py

import diffrax
import jax
import jax.numpy as jnp

from .types import LossCfg, ShotBundle, ShotEval

MAX_SOLVER_STEPS = 50000


def pseudo_huber(r, delta):
    delta = jnp.asarray(delta, dtype=jnp.float64)
    return (delta * delta) * (jnp.sqrt(1.0 + (r / delta) ** 2) - 1.0)


def softclip(x, limit):
    limit = jnp.asarray(limit, dtype=jnp.float64)
    return limit * (x / (limit + jnp.abs(x)))


def shot_loss(model, bundle: ShotBundle, loss_cfg: LossCfg, solver_name: str):
    t_len = bundle.t_len
    ts_t_full = bundle.ts_t
    ctrl_t_full = bundle.ctrl_t
    ctrl_vals_full = bundle.ctrl_vals
    ne_vals_full = bundle.ne_vals
    Te_edge_full = bundle.Te_edge
    ts_Te_full = bundle.ts_Te
    mask_full = bundle.mask
    regime_ts_full = bundle.regime_ts
    regime_mask_full = bundle.regime_mask

    T_max = ts_t_full.shape[0]
    time_mask = (jnp.arange(T_max, dtype=jnp.int32) < t_len).astype(jnp.float64)

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

    z0_arr = jnp.atleast_1d(jnp.asarray(bundle.z0, dtype=jnp.float64))
    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, z0_arr])

    t0 = ts_t_full[0]
    # IMPORTANT: solve over the *padded* window to keep SaveAt(ts=ts_t_full) valid and strictly increasing.
    # All losses below are masked by time_mask to use only the real window (t_len).
    t1 = ts_t_full[-1]

    dts_full = jnp.diff(ts_t_full)
    mask_dts = time_mask[1:] * time_mask[:-1]
    dt_mean = jnp.where(
        jnp.sum(mask_dts) > 0,
        jnp.sum(dts_full * mask_dts) / (jnp.sum(mask_dts) + 1e-8),
        jnp.array(1e-3, dtype=jnp.float64),
    )
    dt0 = jnp.clip(0.5 * dt_mean, 0.05 * dt_mean, 2.0 * dt_mean)
    dt0 = jnp.clip(dt0, 1e-8, 1.0)
    dt0 = jnp.minimum(dt0, 0.1 * (t1 - t0 + 1e-8))

    solver_name = str(solver_name).lower()
    if solver_name == "kencarp3":
        solver = diffrax.KenCarp3()

        def rhs_nonstiff(t, y, args):
            (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_interp) = args

            Te_hat = jnp.clip(y[:-1], 0.0, 5000.0 / model.Te_scale)
            z = jnp.clip(y[-1], -10.0, 10.0)

            Te_bc = Te_bc_interp.evaluate(t)
            Te_total = jnp.append(Te_hat * model.Te_scale, Te_bc)
            Te_total = jnp.clip(Te_total, 0.0, 5000.0)

            src_raw = model.compute_source(t, Te_total, z, args)
            control_norm = model._control_norm(t, ctrl_interp, control_means, control_stds)
            z_dot = model.latent(z, control_norm)

            dTe_hat_dt_src = softclip(src_raw, 1e4) / model.Te_scale
            rhs = jnp.concatenate([dTe_hat_dt_src, jnp.array([z_dot], dtype=jnp.float64)])
            return jnp.where(jnp.isfinite(rhs), rhs, 0.0)

        def rhs_stiff(t, y, args):
            Te_hat = jnp.clip(y[:-1], 0.0, 5000.0 / model.Te_scale)
            z = jnp.clip(y[-1], -10.0, 10.0)

            Te_bc = Te_bc_interp.evaluate(t)
            Te_total = jnp.append(Te_hat * model.Te_scale, Te_bc)
            Te_total = jnp.clip(Te_total, 0.0, 5000.0)

            div_raw = model.compute_divergence_only(t, Te_total, z, args)
            dTe_hat_dt_div = softclip(div_raw, 1e4) / model.Te_scale
            rhs = jnp.concatenate([dTe_hat_dt_div, jnp.array([0.0], dtype=jnp.float64)])
            return jnp.where(jnp.isfinite(rhs), rhs, 0.0)

        terms = diffrax.MultiTerm(
            diffrax.ODETerm(rhs_nonstiff),
            diffrax.ODETerm(rhs_stiff),
        )
    else:
        if solver_name == "tsit5":
            solver = diffrax.Tsit5()
        else:
            solver = diffrax.Kvaerno5()

        def rhs_full(t, y, args):
            return model(t, y, args)

        terms = diffrax.ODETerm(rhs_full)

    controller = diffrax.PIDController(rtol=float(loss_cfg.rtol), atol=float(loss_cfg.atol))
    ts_eval = ts_t_full
    saveat = diffrax.SaveAt(ts=ts_eval)
    max_steps = MAX_SOLVER_STEPS

    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        y0=y0,
        dt0=dt0,
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=max_steps,
        throw=loss_cfg.throw_solver,
        args=ode_args,
    )
    
    is_success = (sol.result == diffrax.RESULTS.successful)

    def success_branch(_):
        ys = jnp.nan_to_num(sol.ys, nan=0.0, posinf=0.0, neginf=0.0)
        Te_hats = ys[:, :-1]
        zs = ys[:, -1]

        Te_bc_ts = Te_bc_interp.evaluate(ts_eval)

        def reconstruct(Te_hat_row, bc_val):
            Te_hat_row = jnp.clip(Te_hat_row, 0.0, 5000.0 / model.Te_scale)
            return jnp.append(Te_hat_row, bc_val / model.Te_scale) * model.Te_scale

        Te_model = jax.vmap(reconstruct)(Te_hats, Te_bc_ts)

        # Use fixed obs_idx columns (boundary excluded in data loader).
        use_cols = bundle.obs_idx
        use_cols = use_cols[0] if use_cols.ndim > 1 else use_cols
        use_cols = use_cols.astype(jnp.int32)

        # ---- VALID-WINDOW MASKING ----
        tm = time_mask.astype(jnp.float64)  # (T_max,)
        tm2 = tm[:, None]                   # (T_max, 1)

        mask_obs = mask_full[:, use_cols].astype(jnp.float64) * tm2
        resid = (Te_model[:, use_cols] - ts_Te_full[:, use_cols]) * tm2

        abs_resid = jnp.abs(resid)
        Te_data = ts_Te_full[:, use_cols]
        denom = jnp.maximum(jnp.abs(Te_data), 50.0)

        wsum = jnp.sum(mask_obs) + 1e-8
        mae_eV = jnp.sum(mask_obs * abs_resid) / wsum
        mae_pct = 100.0 * jnp.sum(mask_obs * (abs_resid / denom)) / wsum

        huber_delta = loss_cfg.huber_delta
        obs_loss = jnp.sum(mask_obs * pseudo_huber(resid, huber_delta)) / wsum

        def eval_source(ti, Te_row, zi):
            return model.compute_source(ti, Te_row, zi, ode_args)

        S_nn_vals = jax.vmap(eval_source)(ts_eval, Te_model, zs)  # (T_max, N-1)
        lambda_src = loss_cfg.lambda_src
        src_delta = loss_cfg.src_delta

        # Source penalty only over valid times
        src_wsum = (jnp.sum(tm) * S_nn_vals.shape[1]) + 1e-8
        src_penalty = lambda_src * jnp.sum(tm2 * pseudo_huber(S_nn_vals, src_delta)) / src_wsum

        w_loss = 0.0

        # z supervision only over valid times
        reg_mask = regime_mask_full.astype(jnp.float64) * tm
        y_regime = jnp.where(regime_ts_full > 2.5, 1.0, 0.0)
        z_prob = jax.nn.sigmoid(zs)
        lambda_z = loss_cfg.lambda_z

        z_loss_num = jnp.sum(reg_mask * (z_prob - y_regime) ** 2)
        z_loss_den = jnp.sum(reg_mask) + 1e-8
        z_loss = lambda_z * z_loss_num / z_loss_den

        # z regularisation only over valid times
        z_reg = loss_cfg.lambda_zreg * (jnp.sum(tm * (zs**2)) / (jnp.sum(tm) + 1e-8))

        total_loss = obs_loss + src_penalty + w_loss + z_loss + z_reg

        Te_edge_ts = Te_edge_full
        Te_edge_min = jnp.min(jnp.where(time_mask > 0, Te_edge_ts, jnp.inf))
        Te_edge_max = jnp.max(jnp.where(time_mask > 0, Te_edge_ts, -jnp.inf))
        Te_edge_ptp = Te_edge_max - Te_edge_min

        diag = jnp.array(
            [
                bundle.shot_id.astype(jnp.float64),
                t_len.astype(jnp.float64),
                dt0.astype(jnp.float64),
                mae_eV,
                mae_pct,
                Te_edge_min,
                Te_edge_max,
                Te_edge_ptp,
            ],
            dtype=jnp.float64,
        )

        return total_loss, jnp.array(1, dtype=jnp.int32), diag

    def fail_branch(_):
        diag = jnp.array(
            [
                bundle.shot_id.astype(jnp.float64),
                t_len.astype(jnp.float64),
                dt0.astype(jnp.float64),
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


def batch_loss(model, bundles, loss_cfg, solver_name: str):
    losses, oks, _ = jax.vmap(lambda b: shot_loss(model, b, loss_cfg, solver_name))(bundles)
    return jnp.mean(losses), jnp.mean(oks)


def eval_shot_trajectory(model, bundle: ShotBundle, loss_cfg: LossCfg, solver_name: str) -> ShotEval:
    # ---- DEBUG/EVAL MUST USE VALID WINDOW ONLY ----
    L = int(jnp.asarray(bundle.t_len))  # host int is fine here (not pmapped/vmapped)
    L = max(1, L)

    ts_t_full = bundle.ts_t[:L]
    ctrl_t_full = bundle.ctrl_t[:L]
    ctrl_vals_full = bundle.ctrl_vals[:L]
    ne_vals_full = bundle.ne_vals[:L]
    Te_edge_full = bundle.Te_edge[:L]
    ts_Te_full = bundle.ts_Te[:L]
    mask_full = bundle.mask[:L]
    regime_ts_full = bundle.regime_ts[:L]
    regime_mask_full = bundle.regime_mask[:L]

    # In eval mode we do not need a time_mask; arrays are already truncated.

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

    z0_arr = jnp.atleast_1d(jnp.asarray(bundle.z0, dtype=jnp.float64))
    y0 = jnp.concatenate([bundle.Te0[:-1] / model.Te_scale, z0_arr])

    t0 = ts_t_full[0]
    t1 = ts_t_full[-1]

    dts_full = jnp.diff(ts_t_full)
    dt_mean = jnp.where(
        dts_full.size > 0,
        jnp.sum(dts_full) / (dts_full.size + 1e-8),
        jnp.array(1e-3, dtype=jnp.float64),
    )
    dt0 = jnp.clip(0.5 * dt_mean, 0.05 * dt_mean, 2.0 * dt_mean)
    dt0 = jnp.clip(dt0, 1e-8, 1.0)
    dt0 = jnp.minimum(dt0, 0.1 * (t1 - t0 + 1e-8))

    solver_name = str(solver_name).lower()
    if solver_name == "kencarp3":
        solver = diffrax.KenCarp3()

        def rhs_nonstiff(t, y, args):
            (rho_vals, Vprime_vals, ctrl_interp, control_means, control_stds, ne_interp, Te_bc_interp) = args

            Te_hat = jnp.clip(y[:-1], 0.0, 5000.0 / model.Te_scale)
            z = jnp.clip(y[-1], -10.0, 10.0)

            Te_bc = Te_bc_interp.evaluate(t)
            Te_total = jnp.append(Te_hat * model.Te_scale, Te_bc)
            Te_total = jnp.clip(Te_total, 0.0, 5000.0)

            src_raw = model.compute_source(t, Te_total, z, args)
            control_norm = model._control_norm(t, ctrl_interp, control_means, control_stds)
            z_dot = model.latent(z, control_norm)

            dTe_hat_dt_src = softclip(src_raw, 1e4) / model.Te_scale
            rhs = jnp.concatenate([dTe_hat_dt_src, jnp.array([z_dot], dtype=jnp.float64)])
            return jnp.where(jnp.isfinite(rhs), rhs, 0.0)

        def rhs_stiff(t, y, args):
            Te_hat = jnp.clip(y[:-1], 0.0, 5000.0 / model.Te_scale)
            z = jnp.clip(y[-1], -10.0, 10.0)

            Te_bc = Te_bc_interp.evaluate(t)
            Te_total = jnp.append(Te_hat * model.Te_scale, Te_bc)
            Te_total = jnp.clip(Te_total, 0.0, 5000.0)

            div_raw = model.compute_divergence_only(t, Te_total, z, args)
            dTe_hat_dt_div = softclip(div_raw, 1e4) / model.Te_scale

            rhs = jnp.concatenate([dTe_hat_dt_div, jnp.array([0.0], dtype=jnp.float64)])
            return jnp.where(jnp.isfinite(rhs), rhs, 0.0)

        terms = diffrax.MultiTerm(diffrax.ODETerm(rhs_nonstiff), diffrax.ODETerm(rhs_stiff))
    else:
        solver = diffrax.Tsit5() if solver_name == "tsit5" else diffrax.Kvaerno5()
        terms = diffrax.ODETerm(lambda t, y, args: model(t, y, args))

    controller = diffrax.PIDController(rtol=float(loss_cfg.rtol), atol=float(loss_cfg.atol))
    ts_eval = ts_t_full
    saveat = diffrax.SaveAt(ts=ts_eval)

    sol = diffrax.diffeqsolve(
        terms,
        solver,
        t0=t0,
        t1=t1,
        y0=y0,
        dt0=dt0,
        stepsize_controller=controller,
        saveat=saveat,
        max_steps=MAX_SOLVER_STEPS,
        throw=loss_cfg.throw_solver,
        args=ode_args,
    )

    use_cols = jnp.arange(ts_Te_full.shape[1] - 1, dtype=jnp.int32)

    if sol.result != diffrax.RESULTS.successful:
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

    Te_bc_ts = Te_bc_interp.evaluate(ts_eval)

    def reconstruct(Te_hat_row, bc_val):
        Te_hat_row = jnp.clip(Te_hat_row, 0.0, 5000.0 / model.Te_scale)
        return jnp.append(Te_hat_row, bc_val / model.Te_scale) * model.Te_scale

    Te_model = jax.vmap(reconstruct)(Te_hats, Te_bc_ts)

    mask_obs = mask_full[:, use_cols].astype(jnp.float64)
    weight_grid = mask_obs
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

    S_nn_vals = jax.vmap(eval_source)(ts_eval, Te_model, zs)
    lambda_src = loss_cfg.lambda_src
    src_delta = loss_cfg.src_delta
    src_penalty = lambda_src * jnp.sum(pseudo_huber(S_nn_vals, src_delta)) / (S_nn_vals.size + 1e-8)

    reg_mask = regime_mask_full.astype(jnp.float64)
    y_regime = jnp.where(regime_ts_full > 2.5, 1.0, 0.0)
    z_prob = jax.nn.sigmoid(zs)
    lambda_z = loss_cfg.lambda_z
    z_loss_num = jnp.sum(reg_mask * (z_prob - y_regime) ** 2)
    z_loss_den = jnp.sum(reg_mask) + 1e-8
    z_loss = lambda_z * z_loss_num / z_loss_den

    z_reg = loss_cfg.lambda_zreg * jnp.mean(zs**2)

    total_loss = obs_loss + src_penalty + z_loss + z_reg

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
