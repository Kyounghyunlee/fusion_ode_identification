"""Smoke test: padded tail freeze prevents solver failure.

Usage:
  python scripts/smoke_padding_freeze.py --config config/config_debug.yaml

This loads a few shots, forces t_len smaller (keeping padding intact), and asserts
shot_loss_imex() stays finite/ok because integration freezes after the valid window.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import yaml
import jax
import jax.numpy as jnp

from fusion_ode_identification.data import load_data
from fusion_ode_identification.debug import build_imex_cfg, build_loss_cfg, build_model_template
from fusion_ode_identification.loss import shot_loss_imex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/config_debug.yaml")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    bundles, _, _, _ = load_data(cfg)
    bundle0 = jax.tree_util.tree_map(lambda x: x[0], bundles)

    L = int(jnp.asarray(bundle0.t_len))
    new_L = max(2, L // 2)
    bundle_half = bundle0._replace(t_len=jnp.asarray(new_L, dtype=bundle0.t_len.dtype))

    key = jax.random.PRNGKey(int(cfg.get("training", {}).get("seed", 0)))
    model = build_model_template(cfg, key)

    loss_cfg = build_loss_cfg(cfg)
    imex_cfg = build_imex_cfg(cfg)

    loss, ok, _ = shot_loss_imex(model, bundle_half, loss_cfg, imex_cfg)

    loss_f = float(loss)
    ok_i = int(ok)
    assert ok_i == 1, f"expected ok=1, got ok={ok_i}"
    assert loss_f == loss_f and abs(loss_f) < 1e30, f"unexpected loss: {loss_f}"

    print(f"OK: padding freeze works (t_len {L} -> {new_L}), loss={loss_f:.6g}")


if __name__ == "__main__":
    main()
