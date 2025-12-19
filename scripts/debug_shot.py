#!/usr/bin/env python3
# scripts/debug_shot.py
import argparse

from fusion_ode_identification.debug import run_debug_shot


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--shot", type=int, required=True)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--out_dir", default="out")
    p.add_argument("--throw", action="store_true")
    args = p.parse_args()

    run_debug_shot(
        config_path=args.config,
        shot_id=args.shot,
        ckpt_path=args.ckpt,
        out_dir=args.out_dir,
        solver_throw=args.throw,
    )


if __name__ == "__main__":
    main()
