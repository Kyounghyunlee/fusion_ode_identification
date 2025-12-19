#!/usr/bin/env python
# scripts/smoke_valid_window.py
import argparse
import yaml
import jax

from fusion_ode_identification.data import load_data
from fusion_ode_identification.debug import build_loss_cfg, build_model_template
from fusion_ode_identification.loss import eval_shot_trajectory


def main():
    parser = argparse.ArgumentParser(description="Smoke test: ensure eval uses valid window only")
    parser.add_argument("--config", default="config/config_debug.yaml", help="Path to config YAML")
    parser.add_argument("--shot", type=int, default=None, help="Optional shot id override")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.shot is not None:
        cfg.setdefault("data", {})
        cfg["data"]["shots"] = [int(args.shot)]

    bundles, _, _, _ = load_data(cfg)
    bundle0 = jax.tree_util.tree_map(lambda x: x[0], bundles)

    seed = int(cfg.get("training", {}).get("seed", 0))
    key = jax.random.PRNGKey(seed)
    model = build_model_template(cfg, key)
    loss_cfg, solver_name = build_loss_cfg(cfg, solver_throw_override=False)

    ev = eval_shot_trajectory(model, bundle0, loss_cfg, solver_name)

    L = int(bundle0.t_len)
    assert ev.ts_t.shape[0] == L, f"ts_t length mismatch: {ev.ts_t.shape[0]} vs {L}"
    assert ev.Te_model.shape[0] == L, f"Te_model length mismatch: {ev.Te_model.shape[0]} vs {L}"

    print(f"valid window OK: L={L} t0={float(ev.ts_t[0]):.6g} t1={float(ev.ts_t[-1]):.6g}")


if __name__ == "__main__":
    main()
