"""Single-device training entrypoint (jit + vmap).

Replaces the legacy pmap-based HPC script: JAX >= 0.11 removed the primitives
it relied on, and this project now targets a single-GPU/CPU workstation.

Usage:
    python train.py --config config/config_cusp.yaml
    JAX_PLATFORMS=cpu python train.py --config ... --total-steps 50
"""

import argparse
import logging
import os
import shutil
import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import optax
import yaml

from fusion_ode_identification.data import load_data
from fusion_ode_identification.loss import shot_loss_imex
from fusion_ode_identification.model import build_hybrid_model
from fusion_ode_identification.types import IMEXConfig, LossCfg


def sanitize_name(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name)


def build_lr_schedule(tr: dict):
    total = int(tr.get("total_steps", 1000))
    warmup = min(int(tr.get("warmup_steps", 0)), max(total - 1, 1))
    peak = float(tr.get("learning_rate", 2e-4))
    if total - warmup <= 1:
        return optax.constant_schedule(peak)
    return optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=peak,
        warmup_steps=max(warmup, 1),
        decay_steps=total,
        end_value=0.05 * peak,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_cusp.yaml")
    ap.add_argument("--total-steps", type=int, default=None, help="Override training.total_steps")
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--resume_step_offset", type=int, default=0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

    with open(args.config) as f:
        config = yaml.safe_load(f)
    tr = config["training"]
    if args.total_steps is not None:
        tr["total_steps"] = int(args.total_steps)

    model_id = config["output"].get("model_id", "default_run")
    model_dir = os.path.join(config["output"]["save_dir"], model_id)
    log_dir = os.path.join(config["output"].get("log_dir", "logs"), model_id)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(os.path.join(log_dir, "training.log"), mode="w"))
    shutil.copyfile(args.config, os.path.join(log_dir, "config.yaml"))

    device = jax.devices()[0]
    logging.info(f"Device: {device} (platform={device.platform})")

    logging.info("Loading data ...")
    all_bundles, rho_rom, shot_ids, _ = load_data(config)
    n_shots = int(all_bundles.ts_t.shape[0])
    logging.info(f"Loaded {n_shots} shots: {shot_ids}")

    key = jax.random.PRNGKey(int(tr.get("seed", 0)))
    model = build_hybrid_model(config, key)
    if args.resume_ckpt:
        model = eqx.tree_deserialise_leaves(args.resume_ckpt, model)
        logging.info(f"Resumed weights from {args.resume_ckpt}")
    params, static = eqx.partition(model, eqx.is_inexact_array)

    schedule = build_lr_schedule(tr)
    step_offset = int(args.resume_step_offset)
    optimizer = optax.chain(
        optax.clip_by_global_norm(float(tr.get("grad_clip", 1e3))),
        optax.adamw(
            learning_rate=lambda step: schedule(step + step_offset),
            weight_decay=float(tr.get("weight_decay", 0.0)),
        ),
    )
    opt_state = optimizer.init(params)

    loss_cfg = LossCfg(
        huber_delta=float(tr.get("huber_delta", 10.0)),
        lambda_src=float(tr.get("lambda_src", 1e-4)),
        src_delta=float(tr.get("src_delta", 10.0)),
        lambda_z=float(tr.get("lambda_z", 1e-4)),
        lambda_zreg=float(tr.get("lambda_zreg", 1e-5)),
        lambda_regime=float(tr.get("lambda_regime", 1e-3)),
        lambda_dalpha=float(tr.get("lambda_dalpha", 1.0)),
        lambda_pH=float(tr.get("lambda_pH", 0.0)),
        throw_solver=bool(tr.get("throw_solver", False)),
    )
    imx = tr.get("imex", {})
    imex_cfg = IMEXConfig(
        theta=float(imx.get("theta", 0.7)),
        dt_base=float(imx.get("dt_base", 1e-3)),
        max_steps=int(imx.get("max_steps", 50000)),
        rtol=float(imx.get("rtol", 1e-4)),
        atol=float(imx.get("atol", 1e-6)),
        substeps=int(imx.get("substeps", 5)),
    )

    # Train/validation split (seeded, ~20% validation, >= 2 shots).
    rng = np.random.default_rng(int(tr.get("seed", 0)))
    all_idx = rng.permutation(n_shots)
    if n_shots < 3:
        train_idx = val_idx = all_idx
    else:
        n_val = min(max(2, int(0.2 * n_shots)), n_shots - 1)
        val_idx, train_idx = all_idx[:n_val], all_idx[n_val:]
    batch_size = min(int(tr.get("batch_size", 8)), train_idx.size)
    logging.info(f"Split: {train_idx.size} train / {val_idx.size} val; batch_size={batch_size}")

    @eqx.filter_jit
    def train_step(params, opt_state, batch):
        def loss_fn(p):
            m = eqx.combine(p, static)
            losses, oks, _ = jax.vmap(lambda b: shot_loss_imex(m, b, loss_cfg, imex_cfg))(batch)
            return jnp.mean(losses), jnp.mean(oks)

        (loss, ok_rate), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        flat = jax.tree_util.tree_leaves(grads)
        grad_bad = jnp.any(jnp.array([jnp.any(~jnp.isfinite(g)) for g in flat]))
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = eqx.apply_updates(params, updates)
        # Skip the update entirely on non-finite gradients.
        new_params, new_opt_state = jax.tree_util.tree_map(
            lambda new, old: jnp.where(grad_bad, old, new), (new_params, new_opt_state), (params, opt_state)
        )
        return loss, ok_rate, new_params, new_opt_state, grad_bad

    @eqx.filter_jit
    def eval_loss(params, idxs):
        m = eqx.combine(params, static)
        batch = jax.tree_util.tree_map(lambda x: x[idxs], all_bundles)
        losses, oks, _ = jax.vmap(lambda b: shot_loss_imex(m, b, loss_cfg, imex_cfg))(batch)
        return jnp.mean(losses), jnp.mean(oks)

    ema_decay = float(tr.get("ema_decay", 0.0))
    ema_params = params if ema_decay > 0 else None

    raw_name = config["output"].get("model_name", "model")
    safe_name = sanitize_name(raw_name)
    best_path = os.path.join(model_dir, f"{safe_name}_best.eqx")
    best_ema_path = os.path.join(model_dir, f"{safe_name}_best_ema.eqx")

    total_steps = int(tr["total_steps"])
    log_every = int(tr.get("log_every", 50))
    val_idx_j = jnp.asarray(val_idx)

    logging.info("[compile] Warming up train_step ...")
    t0 = time.time()
    warm = jax.tree_util.tree_map(lambda x: x[train_idx[:batch_size]], all_bundles)
    out = train_step(params, opt_state, warm)
    jax.block_until_ready(out[0])
    logging.info(f"[compile] train_step compile+run: {time.time() - t0:.1f}s")

    best_val = float("inf")
    best_val_ema = float("inf")
    t_loop = time.time()
    for step in range(total_steps):
        batch_idx = rng.choice(train_idx, batch_size, replace=False)
        batch = jax.tree_util.tree_map(lambda x: x[batch_idx], all_bundles)
        loss, ok_rate, params, opt_state, grad_bad = train_step(params, opt_state, batch)
        if ema_params is not None:
            ema_params = jax.tree_util.tree_map(
                lambda e, p: ema_decay * e + (1.0 - ema_decay) * p, ema_params, params
            )

        if step % log_every == 0 or step == total_steps - 1:
            val_loss, val_ok = eval_loss(params, val_idx_j)
            val_loss = float(val_loss)
            msg = (
                f"step={step + step_offset} loss={float(loss):.4f} ok={float(ok_rate):.2f} "
                f"val={val_loss:.4f} val_ok={float(val_ok):.2f} lr={float(schedule(step + step_offset)):.2e} "
                f"elapsed={time.time() - t_loop:.0f}s"
            )
            if bool(grad_bad):
                msg += " [non-finite grads: update skipped]"
            logging.info(msg)
            if val_loss < best_val and float(val_ok) == 1.0:
                best_val = val_loss
                eqx.tree_serialise_leaves(best_path, eqx.combine(params, static))
                logging.info(f"New best (val) saved: {best_path} val_loss={val_loss:.4f} step={step + step_offset}")
            if ema_params is not None:
                val_loss_ema, val_ok_ema = eval_loss(ema_params, val_idx_j)
                val_loss_ema = float(val_loss_ema)
                if val_loss_ema < best_val_ema and float(val_ok_ema) == 1.0:
                    best_val_ema = val_loss_ema
                    eqx.tree_serialise_leaves(best_ema_path, eqx.combine(ema_params, static))
                    logging.info(
                        f"New best (val, EMA) saved: {best_ema_path} val_loss={val_loss_ema:.4f} step={step + step_offset}"
                    )

    logging.info(f"Training complete. best_val={best_val:.4f} best_val_ema={best_val_ema:.4f}")


if __name__ == "__main__":
    main()
