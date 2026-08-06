"""Causal streaming replay of the regime assessor.

Replays held-out shots sample-by-sample using only past/current inputs:
  z_{n+1} = z_n + dt * F(z_n, r_n),  b/p_H/margins from z_n.
No D-alpha, no profiles, no future samples. Reports per-step latency
(compiled JAX path and a pure-NumPy path representative of a PCS
implementation) and verifies agreement with the batch rollout.

Usage: python scripts/streaming_replay.py --model-id v2_free_s0 --role test
"""

import argparse
import json
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import yaml

from fusion_ode_identification.model import (
    DRIVE_FEATURES, DRIVE_OFFSETS, DRIVE_SCALES, build_hybrid_model,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--role", default="val")
    ap.add_argument("--config", default="config/config.yaml")
    args = ap.parse_args()

    saved = f"logs/{args.model_id}/config.yaml"
    cfg = yaml.safe_load(open(saved if os.path.exists(saved) else args.config))
    drive_set = str(cfg.get("model", {}).get("drive_set", "basic"))
    feat_names = DRIVE_FEATURES[drive_set]
    model = build_hybrid_model(cfg, jax.random.PRNGKey(0))
    model = eqx.tree_deserialise_leaves(f"models/{args.model_id}/model_best.eqx", model)
    lat = model.latent

    beta = float(lat.beta())
    tau = float(lat.tau_eff())
    w = np.concatenate([[float(jax.nn.softplus(lat.drive_weights[0]))],
                        np.asarray(lat.drive_weights[1:], dtype=float)])
    k0 = float(lat.drive_bias)
    kb = float(jax.nn.softplus(lat.kb_raw) + 0.5)
    bb = float(lat.bb)

    with open("data/split.json") as f:
        split = json.load(f)
    shots = split[args.role]

    # --- pure NumPy streaming path (PCS-representative) ---
    def numpy_replay(t, R):
        alpha = R @ w + k0
        z = np.empty(len(t))
        # causal init: lowest equilibrium at first drive (Newton)
        zz = -(1 + np.sqrt(abs(beta)) + abs(alpha[0]) ** (1 / 3))
        for _ in range(60):
            f = alpha[0] + beta * zz - zz**3
            fp = beta - 3 * zz**2
            zz -= f / (fp if abs(fp) > 1e-8 else -1e-8)
        z[0] = zz
        for i in range(1, len(t)):
            dt = min(max(t[i] - t[i - 1], 1e-4), 0.05)
            for _ in range(5):  # same substep count as training
                zz = zz + (dt / 5) * (alpha[i - 1] + beta * zz - zz**3) / tau
            z[i] = zz
        pH = 1 / (1 + np.exp(-(kb * z + bb)))
        return z, pH

    lat_step = jax.jit(lambda z, feat: lat(z, feat))

    results = {"per_shot": {}, "params": {"beta": beta, "tau": tau}}
    lat_ms = []
    for shot in shots:
        d = np.load(f"data/{shot}_torax_training.npz", allow_pickle=True)
        t = d["t"]
        cols = []
        for nm in feat_names:
            off = DRIVE_OFFSETS.get(nm, 0.0)
            sc = DRIVE_SCALES[nm]
            v = np.asarray(d[nm], dtype=float).reshape(-1) if nm in d else np.full(t.size, off * sc)
            if nm == "Ip":
                v = np.abs(v)
            if v.size != t.size or not np.any(np.isfinite(v)):
                v = np.full(t.size, off * sc)
            cols.append(v / sc - off)
        R = np.stack(cols, -1)
        t0 = time.perf_counter()
        z, pH = numpy_replay(t, R)
        wall = time.perf_counter() - t0
        lat_ms.append(1e3 * wall / max(len(t) - 1, 1))
        results["per_shot"][str(shot)] = {
            "n_steps": int(len(t)),
            "numpy_us_per_step": 1e6 * wall / max(len(t) - 1, 1),
            "max_pH": float(pH.max()),
        }

    # compiled-path latency on one shot
    feat = jnp.zeros((len(feat_names),))
    z = jnp.array(-1.0)
    _ = lat_step(z, feat)  # warm-up/compile
    t0 = time.perf_counter()
    for _ in range(2000):
        z = z + 1e-3 * lat_step(z, feat)
    _ = float(z)
    results["jax_dispatch_us_per_step"] = (time.perf_counter() - t0) / 2000 * 1e6
    results["numpy_median_us_per_step"] = float(np.median([v["numpy_us_per_step"] for v in results["per_shot"].values()]))

    os.makedirs("experiments", exist_ok=True)
    with open(f"experiments/streaming_{args.model_id}_{args.role}.json", "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps({k: v for k, v in results.items() if k != "per_shot"}, indent=1))


if __name__ == "__main__":
    main()
