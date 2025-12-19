"""
HPC Training Script for Physics-Consistent Manifold Learning
"""
# train_tokamak_ode_hpc.py

import argparse
import os
import time
import yaml
import logging
import functools
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from fusion_ode_identification.data import load_data, log_data_scale
from fusion_ode_identification.debug import build_loss_cfg as build_debug_loss_cfg, build_model_template, find_best_checkpoint, make_debug_plot_and_npz
from fusion_ode_identification.model import HybridField, LatentDynamics, SourceNN
from fusion_ode_identification.loss import eval_shot_trajectory, shot_loss
from fusion_ode_identification.types import LossCfg

jax.config.update("jax_enable_x64", True)


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_lr_schedule(cfg: dict):
    """Robust LR schedule that never passes invalid decay_steps to optax."""
    tr = cfg.setdefault("training", {})
    total_steps = int(tr.get("total_steps", 1000))
    warmup_steps = int(tr.get("warmup_steps", 0))

    init_lr = float(tr.get("init_lr", 1e-5))
    peak_lr = float(tr.get("learning_rate", 2e-4))
    end_lr = float(tr.get("end_lr", 1e-6))

    warmup_steps = max(0, warmup_steps)
    total_steps = max(1, total_steps)

    decay_steps = total_steps - warmup_steps
    if total_steps <= warmup_steps or decay_steps <= 0:
        warm = optax.linear_schedule(init_value=init_lr, end_value=peak_lr, transition_steps=max(1, total_steps))
        hold = optax.constant_schedule(peak_lr)
        return optax.join_schedules([warm, hold], boundaries=[total_steps])

    return optax.warmup_cosine_decay_schedule(
        init_value=init_lr,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_lr,
    )


def print_model_and_rho_summary(model: HybridField, rho_rom: jnp.ndarray, rho_cap: jnp.ndarray, obs_idx: jnp.ndarray):
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


def _time_block(name: str, fn):
    t0 = time.time()
    out = fn()
    # Force sync: otherwise GPU work is async and logs look “quiet”.
    out = jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        out,
    )
    dt = time.time() - t0
    logging.info(f"[timing] {name}: {dt:.2f}s")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--debug_one_shot", type=int, default=None, help="Override config data.shots with single shot")
    parser.add_argument("--debug_eval_only", action="store_true", help="Load a checkpoint, export debug artifacts for one shot, then exit")
    parser.add_argument("--debug_eval_shot", type=int, default=None, help="Shot id for debug_eval_only (default: first loaded)")
    parser.add_argument("--debug_ckpt", type=str, default=None, help="Checkpoint path override for debug_eval_only")
    parser.add_argument("--debug_solver_throw", action="store_true", help="Force solver throw=True during debug eval")
    parser.add_argument("--throw", action="store_true", help="Force solver throw=True for debug")
    parser.add_argument("--lbfgs_finetune", action="store_true", help="Run optional single-device L-BFGS finetune after AdamW")
    parser.add_argument("--lbfgs_smoke", action="store_true", help="Quick L-BFGS smoke test: small batch and few iterations")
    args = parser.parse_args()

    # Configure logging early (before any logging.* calls) so INFO shows up.
    # `force=True` is important because earlier logging calls can otherwise
    # cause basicConfig to be ignored.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
        force=True,
    )

    config = load_config(args.config)

    if args.debug_one_shot is not None:
        config.setdefault("data", {})
        config["data"]["shots"] = [int(args.debug_one_shot)]

    if args.throw:
        config.setdefault("training", {})
        config["training"]["throw_solver"] = True

    tr = config.setdefault("training", {})
    ts = int(tr.get("total_steps", 1000))
    ws = int(tr.get("warmup_steps", 0))
    if ts <= 0:
        raise ValueError(f"training.total_steps must be > 0, got {ts}")
    if ws < 0:
        raise ValueError(f"training.warmup_steps must be >= 0, got {ws}")
    if ws >= ts:
        logging.warning(f"[lr] warmup_steps ({ws}) >= total_steps ({ts}); using warmup+hold schedule (no cosine decay).")

    model_id = config["output"].get("model_id", "default_run")
    model_dir = os.path.join(config["output"]["save_dir"], model_id)
    log_dir = os.path.join(config["output"].get("log_dir", "logs"), model_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    failure_dir = os.path.join(log_dir, "failures")
    os.makedirs(failure_dir, exist_ok=True)

    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    log_every = int(config["training"].get("log_every", 50))

    # Add a file handler in addition to the already-configured stream handler.
    root_logger = logging.getLogger()
    root_logger.addHandler(logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w"))

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

    if args.lbfgs_smoke:
        config.setdefault("training", {})
        config["training"]["lbfgs_finetune"] = True
        config["training"]["lbfgs_maxiter"] = 5
        config["training"]["lbfgs_history"] = 5
        config["training"]["lbfgs_batch_shots"] = 1
        config["training"]["lbfgs_tol"] = 1e-4

    all_bundles, rho_rom, rho_cap, obs_idx = load_data(config)
    n_shots = all_bundles.ts_t.shape[0]
    logging.info(f"Loaded {n_shots} shots (stacked).")

    log_data_scale(all_bundles, obs_idx)

    logging.info(
        "bundle shapes: ts_t %s, ts_Te %s, mask %s, rho_rom %s, Vprime_rom %s, t_len %s",
        all_bundles.ts_t.shape,
        all_bundles.ts_Te.shape,
        all_bundles.mask.shape,
        all_bundles.rho_rom.shape,
        all_bundles.Vprime_rom.shape,
        all_bundles.t_len.shape,
    )

    if args.debug_eval_only:
        cfg_for_debug = yaml.safe_load(yaml.dump(config))
        shot_array = np.array(all_bundles.shot_id)
        if args.debug_eval_shot is not None:
            match = np.where(shot_array == int(args.debug_eval_shot))[0]
            if match.size == 0:
                raise ValueError(f"Shot {args.debug_eval_shot} not present in loaded bundles: {shot_array}")
            idx0 = int(match[0])
            shot_id = int(args.debug_eval_shot)
        else:
            idx0 = 0
            shot_id = int(shot_array[idx0])

        cfg_for_debug.setdefault("data", {})
        cfg_for_debug["data"]["shots"] = [shot_id]

        bundle0 = jax.tree_util.tree_map(lambda x: x[idx0], all_bundles)
        ckpt_path = args.debug_ckpt or find_best_checkpoint(cfg_for_debug)
        seed = int(cfg_for_debug.get("training", {}).get("seed", 0))
        key = jax.random.PRNGKey(seed)
        template = build_model_template(cfg_for_debug, key)
        model_loaded = eqx.tree_deserialise_leaves(ckpt_path, template)

        npz_path = os.path.join(config["data"]["data_dir"], f"{shot_id}_torax_training.npz")
        try:
            with np.load(npz_path) as d:
                rho_raw = d["rho"].astype(float)
                cov_raw = d["Te_mask"].astype(float).mean(axis=0)
            print("rho tail:", rho_raw[-5:])
            print("Te_mask coverage tail:", cov_raw[-5:])
            print("Te_mask coverage last 3:", cov_raw[-3:])
        except Exception as e:
            print(f"[debug] Could not load raw NPZ for shot {shot_id}: {e}")

        print_model_and_rho_summary(model_loaded, rho_rom, rho_cap, obs_idx)

        loss_cfg_debug, solver_name_debug = build_debug_loss_cfg(cfg_for_debug, solver_throw_override=args.debug_solver_throw)
        ev = eval_shot_trajectory(model_loaded, bundle0, loss_cfg_debug, solver_name_debug)

        out_png = os.path.join(out_dir, f"debug_shot_{shot_id}.png")
        out_npz = os.path.join(out_dir, f"debug_shot_{shot_id}.npz")
        make_debug_plot_and_npz(bundle0, ev, out_png, out_npz)
        logging.info("Debug artifacts written: %s, %s", out_png, out_npz)
        return

    base_seed = int(config["training"].get("seed", 0))
    global_best_val_loss = float("inf")
    global_best_step = -1
    failure_logged = 0

    layers = int(config.get("model", {}).get("layers", 64))
    depth = int(config.get("model", {}).get("depth", 3))
    latent_gain = float(config.get("model", {}).get("latent_gain", 1.0))
    source_scale = float(config.get("model", {}).get("source_scale", 3.0e5))
    divergence_clip = float(config.get("model", {}).get("divergence_clip", 1.0e6))

    loss_cfg_base = dict(
        huber_delta=float(config["training"].get("huber_delta", 5.0)),
        lambda_src=float(config["training"].get("lambda_src", 1e-4)),
        src_delta=float(config["training"].get("src_delta", 5.0)),
        lambda_w=float(config["training"].get("lambda_w", 1e-5)),
        model_error_delta=float(config["training"].get("model_error_delta", 10.0)),
        lambda_z=float(config["training"].get("lambda_z", 1e-4)),
        lambda_zreg=float(config["training"].get("lambda_zreg", 1e-4)),
        throw_solver=bool(config["training"].get("throw_solver", False)),
        solver=str(config["training"].get("solver", "kvaerno5")),
        rtol=float(config["training"].get("rtol", 1e-3)),
        atol=float(config["training"].get("atol", 1e-3)),
    )

    for restart in range(int(config["training"]["num_restarts"])):
        key = jax.random.PRNGKey(base_seed)
        key = jax.random.fold_in(key, restart)
        key_nn, key_mu = jax.random.split(key)

        model = HybridField(
            nn=SourceNN(key_nn, source_scale=source_scale, layers=layers, depth=depth),
            latent=LatentDynamics(
                alpha=jnp.array(1.0, dtype=jnp.float64),
                beta=jnp.array(1.0, dtype=jnp.float64),
                gamma=jnp.array(1.0, dtype=jnp.float64),
                mu_weights=jax.random.normal(key_mu, (3,), dtype=jnp.float64) * 0.01,
                mu_bias=jnp.array(0.0, dtype=jnp.float64),
                mu_ref=jnp.array(0.0, dtype=jnp.float64),
            ),
            latent_gain=latent_gain,
            divergence_clip=divergence_clip,
        )

        print_model_and_rho_summary(model, rho_rom, rho_cap, obs_idx)

        total_steps = int(config["training"]["total_steps"])
        schedule = build_lr_schedule(config)
        opt_name = str(config["training"]["optimizer"]).lower()
        if not hasattr(optax, opt_name):
            raise ValueError(f"Unknown optax optimizer: {opt_name}")

        optimizer = optax.chain(
            optax.clip_by_global_norm(float(config["training"]["grad_clip"])),
            getattr(optax, opt_name)(learning_rate=schedule, weight_decay=float(config["training"]["weight_decay"])),
        )

        model_params_host, model_static = eqx.partition(model, eqx.is_inexact_array)
        opt_state = optimizer.init(model_params_host)
        model_params = jax.device_put_replicated(model_params_host, devices)
        opt_state = jax.device_put_replicated(opt_state, devices)

        test_params, _ = eqx.partition(model, eqx.is_inexact_array)
        _ = optimizer.init(test_params)

        ema_decay = float(config["training"].get("ema_decay", 0.0))
        if ema_decay > 0.0:
            ema_params = jax.tree_util.tree_map(lambda x: x, model_params)
        else:
            ema_params = None

        UPD_DTYPE = jnp.float32
        GRAD_DTYPE = jnp.float64

        def tree_all_finite(tree):
            leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
            if len(leaves) == 0:
                return jnp.array(True)
            flags = [jnp.all(jnp.isfinite(x)) for x in leaves]
            return jnp.all(jnp.stack(flags))

        def safe_global_norm(tree, dtype=jnp.float64):
            leaves = jax.tree_util.tree_leaves(eqx.filter(tree, eqx.is_inexact_array))
            if len(leaves) == 0:
                return jnp.array(0.0, dtype=dtype)
            ss = jnp.array(0.0, dtype=dtype)
            for x in leaves:
                xx = x.astype(dtype)
                ss = ss + jnp.sum(xx * xx)
            return jnp.sqrt(ss)

        @functools.partial(jax.pmap, axis_name="devices", static_broadcasted_argnums=(3, 4))
        def make_step(params, st, batch_bundles, loss_cfg, solver_name):
            m = eqx.combine(params, model_static)

            def loss_fn(m, bundles, cfg, solver_name):
                losses, oks, diags = jax.vmap(lambda b: shot_loss(m, b, cfg, solver_name))(bundles)
                ok_mean = jnp.mean(oks)
                return jnp.mean(losses), (ok_mean, diags, oks)

            (loss, (ok_rate, diags, oks)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(m, batch_bundles, loss_cfg, solver_name)

            grad_norm_local = safe_global_norm(grads, dtype=GRAD_DTYPE)
            any_bad = jnp.logical_or(~tree_all_finite(grads), ~jnp.isfinite(grad_norm_local))
            grad_is_nan = jax.lax.pmax(any_bad.astype(jnp.int32), axis_name="devices") > 0

            grads = jax.lax.pmean(grads, axis_name="devices")
            loss = jax.lax.pmean(loss, axis_name="devices")
            ok_rate = jax.lax.pmean(ok_rate, axis_name="devices")

            def do_update(g, s, p):
                updates, new_s = optimizer.update(g, s, p)
                new_p = eqx.apply_updates(p, updates)
                upd_norm = safe_global_norm(updates, dtype=jnp.float64).astype(UPD_DTYPE)
                return new_p, new_s, upd_norm

            def skip_update(g, s, p):
                return p, s, jnp.array(0.0, dtype=UPD_DTYPE)

            new_params, new_state, upd_norm = jax.lax.cond(
                grad_is_nan,
                skip_update,
                do_update,
                grads,
                st,
                params,
            )

            grad_norm = safe_global_norm(grads, dtype=GRAD_DTYPE)
            grad_norm = jax.lax.pmean(grad_norm, axis_name="devices")
            upd_norm = jax.lax.pmean(upd_norm, axis_name="devices")

            return loss, ok_rate, new_params, new_state, grad_norm, upd_norm, grad_is_nan, diags, oks

        start_time = time.time()

        rng = np.random.default_rng(base_seed + 1000 * restart)
        all_idx = np.arange(n_shots)
        rng.shuffle(all_idx)

        if n_shots < 3:
            val_idx = all_idx
            train_idx = all_idx
        else:
            n_val = max(2, int(0.2 * n_shots))
            n_val = min(n_val, n_shots - 1)
            val_idx = all_idx[:n_val]
            train_idx = all_idx[n_val:]

        train_pool = train_idx
        target_bs = int(config["training"].get("batch_size", n_shots))
        target_bs = min(target_bs, train_pool.size)
        if target_bs % n_devices != 0:
            target_bs = target_bs - (target_bs % n_devices)
        target_bs = max(n_devices, target_bs)

        global_batch_size = target_bs
        if global_batch_size % n_devices != 0:
            raise ValueError(f"global_batch_size {global_batch_size} must be divisible by n_devices {n_devices}")
        if global_batch_size > train_pool.size:
            raise ValueError(f"global_batch_size {global_batch_size} exceeds train set size {train_pool.size}")

        device_batch_size = global_batch_size // n_devices
        if device_batch_size < 1:
            raise ValueError("device_batch_size must be >= 1")

        @functools.partial(jax.jit, static_argnums=(2, 3))
        def eval_loss_on_indices(params_single, idxs, loss_cfg, solver_name):
            m = eqx.combine(params_single, model_static)
            b = jax.tree_util.tree_map(lambda x: x[idxs], all_bundles)
            losses, oks, diags = jax.vmap(lambda bb: shot_loss(m, bb, loss_cfg, solver_name))(b)
            return jnp.mean(losses), losses, jnp.mean(oks), oks, diags

        best_val_loss = float("inf")
        best_val_step = -1

        np_rng = np.random.default_rng(base_seed + 12345 * restart)

        solver_name = str(loss_cfg_base["solver"]).lower()
        loss_cfg_step = LossCfg(
            huber_delta=loss_cfg_base["huber_delta"],
            lambda_src=loss_cfg_base["lambda_src"],
            src_delta=loss_cfg_base["src_delta"],
            lambda_w=loss_cfg_base["lambda_w"],
            model_error_delta=loss_cfg_base["model_error_delta"],
            lambda_z=loss_cfg_base["lambda_z"],
            lambda_zreg=loss_cfg_base["lambda_zreg"],
            throw_solver=loss_cfg_base["throw_solver"],
            rtol=loss_cfg_base["rtol"],
            atol=loss_cfg_base["atol"],
        )

        logging.info(f"[lr] lr(step0)={float(schedule(0)):.3e} lr(step_end)={float(schedule(total_steps - 1)):.3e}")

        # ---- One-time warmup to make the first compile visible ----
        print("[compile] Warming up (first pmap compile)...", flush=True)
        logging.info("[compile] Warming up (first pmap compile)...")
        warm_indices = train_idx[:global_batch_size]
        warm_bundle = jax.tree_util.tree_map(lambda x: x[warm_indices], all_bundles)
        warm_sharded = jax.tree_util.tree_map(
            lambda x: x.reshape((n_devices, device_batch_size) + x.shape[1:]),
            warm_bundle,
        )

        _time_block(
            "make_step compile+run",
            lambda: make_step(model_params, opt_state, warm_sharded, loss_cfg_step, solver_name),
        )
        print("[compile] Warmup done. Starting loop.", flush=True)
        logging.info("[compile] Warmup done. Starting loop.")

        for step in range(total_steps):
            train_pool = train_idx
            use_full_dataset = global_batch_size == train_pool.size

            if use_full_dataset:
                perm = np_rng.permutation(train_pool)
                r = perm.size % n_devices
                if r != 0:
                    pad = n_devices - r
                    perm = np.concatenate([perm, perm[:pad]])
                batch_indices = perm
            else:
                batch_indices = np_rng.choice(train_pool, global_batch_size, replace=False)

            if step == 0:
                uniq, cnt = np.unique(batch_indices, return_counts=True)
                logging.info(f"Batch index counts (step0): {dict(zip(uniq.tolist(), cnt.tolist()))}")

            batch_bundle = jax.tree_util.tree_map(lambda x: x[batch_indices], all_bundles)
            sharded_bundle = jax.tree_util.tree_map(lambda x: x.reshape((n_devices, device_batch_size) + x.shape[1:]), batch_bundle)

            loss, ok_rate, params, opt_state, grad_norm, upd_norm, grad_is_nan, diags, oks = make_step(model_params, opt_state, sharded_bundle, loss_cfg_step, solver_name)
            model_params = params

            if ema_params is not None and not bool(grad_is_nan[0]):
                ema_params = jax.tree_util.tree_map(lambda new, ema: ema_decay * ema + (1.0 - ema_decay) * new, model_params, ema_params)
            
            loss_val = float(loss[0])
            ok_rate_val = float(ok_rate[0])
            grad_norm_val = float(grad_norm[0])
            is_nan_val = bool(grad_is_nan[0])

            if failure_logged < 5:
                oks_host = np.array(oks[0])
                diags_host = np.array(diags[0])
                fail_mask = oks_host < 0.5
                if np.any(fail_mask):
                    for row in diags_host[fail_mask][: max(0, 5 - failure_logged)]:
                        shot_id_v, t_len_v, dt0_v, mae_eV_v, mae_pct_v = row.tolist()[:5]
                        fname = os.path.join(failure_dir, f"fail_{failure_logged:03d}.npz")
                        np.savez(
                            fname,
                            restart=restart,
                            step=step,
                            shot_id=shot_id_v,
                            t_len=t_len_v,
                            dt0=dt0_v,
                            mae_eV=mae_eV_v,
                            mae_pct=mae_pct_v,
                        )
                        failure_logged += 1
                        if failure_logged >= 5:
                            break

            if (step % log_every) == 0 or step == total_steps - 1:
                elapsed = time.time() - start_time
                current_lr = float(schedule(step))
                params0 = jax.tree_util.tree_map(lambda x: x[0], model_params)
                val_mean, val_vec, val_ok, val_oks, val_diags = eval_loss_on_indices(params0, jnp.array(val_idx), loss_cfg_step, solver_name)
                val_mean_f = float(val_mean)
                val_p90 = float(jnp.percentile(val_vec, 90.0))
                val_max = float(jnp.max(val_vec))
                val_mask = jnp.where(val_oks > 0.5, 1.0, 0.0)
                mae_eV_val = float(jnp.sum(val_diags[:, 3] * val_mask) / (jnp.sum(val_mask) + 1e-8))
                mae_pct_val = float(jnp.sum(val_diags[:, 4] * val_mask) / (jnp.sum(val_mask) + 1e-8))
                logging.info(f"[restart {restart}] step {step}/{total_steps} train={loss_val:.6g} val={val_mean_f:.6g} val_p90={val_p90:.6g} val_max={val_max:.6g} ok={ok_rate_val:.3f} val_ok={float(val_ok):.3f} mae_eV={mae_eV_val:.3f} mae_pct={mae_pct_val:.3f} grad={grad_norm_val:.4e} upd={float(upd_norm[0]):.4e} lr={current_lr:.2e} nan={is_nan_val} elapsed={elapsed:.1f}s")

            if ((step % log_every) == 0 or step == total_steps - 1):
                if val_mean_f < best_val_loss:
                    best_val_loss = val_mean_f
                    best_val_step = restart * total_steps + step
                    params_save = jax.tree_util.tree_map(lambda x: x[0], model_params)
                    model_save = eqx.combine(params_save, model_static)
                    save_path = os.path.join(model_dir, f"{config['output']['model_name']}_best.eqx")
                    eqx.tree_serialise_leaves(save_path, model_save)
                    logging.info(f"New best (val) model saved: {save_path} val_loss={best_val_loss:.6g} step={best_val_step}")

                    if best_val_loss < global_best_val_loss:
                        global_best_val_loss = best_val_loss
                        global_best_step = best_val_step

                    if ema_params is not None:
                        ema_params_save = jax.tree_util.tree_map(lambda x: x[0], ema_params)
                        ema_model_save = eqx.combine(ema_params_save, model_static)
                        ema_path = os.path.join(model_dir, f"{config['output']['model_name']}_best_ema.eqx")
                        eqx.tree_serialise_leaves(ema_path, ema_model_save)
                        logging.info(f"EMA snapshot (val-best) saved: {ema_path}")

    logging.info(f"Training complete. best_val_loss={global_best_val_loss:.6g} best_step={global_best_step}")

    if args.lbfgs_finetune or bool(config.get("training", {}).get("lbfgs_finetune", False)):
        try:
            from jaxopt import LBFGS  # type: ignore
        except Exception:
            logging.warning("L-BFGS finetune requested but jaxopt is not installed. Skipping. Try: pip install jaxopt")
            return

        best_path = os.path.join(model_dir, f"{config['output']['model_name']}_best.eqx")
        if not os.path.exists(best_path):
            logging.warning(f"No best checkpoint found at {best_path}; skipping L-BFGS finetune.")
            return

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

        n_shots = all_bundles.ts_t.shape[0]
        k = int(config.get("training", {}).get("lbfgs_batch_shots", 1))
        k = max(1, min(k, n_shots))
        idxs = jnp.arange(k)
        fixed_bundle = jax.tree_util.tree_map(lambda x: x[idxs], all_bundles)

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
            rtol=lcb["rtol"],
            atol=lcb["atol"],
        )

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

        def objective(train_vars):
            m_full = set_train_vars(model_best, train_vars)
            solver_name_ft = str(loss_cfg_base["solver"]).lower()
            losses, oks, _ = jax.vmap(lambda b: shot_loss(m_full, b, loss_cfg_ft, solver_name_ft))(fixed_bundle)
            return jnp.mean(losses)

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
