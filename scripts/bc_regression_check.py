#!/usr/bin/env python3
"""
BC regression script: boundary-condition (Te_edge) sanity check for a single shot.

Alias for the repo's BC check so tools (and Copilot) can find it by name.
"""
# scripts/bc_regression_check.py

import argparse
import sys
import numpy as np
import yaml

from fusion_ode_identification.data import load_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config_debug.yaml")
    p.add_argument("--shot", type=int, default=None, help="Shot id to inspect (default: first in config)")
    p.add_argument("--ptp_tol", type=float, default=1.0, help="Minimum allowed Te_edge peak-to-peak (eV)")
    args = p.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.shot is not None:
        cfg.setdefault("data", {})
        cfg["data"]["shots"] = [int(args.shot)]

    bundles, _, _, _ = load_data(cfg)
    b0 = {k: np.array(v[0]) for k, v in zip(bundles._fields, bundles)}

    edge_bc_mode = cfg.get("data", {}).get("edge_bc_mode", "use_last_observed")
    edge_idx = int(b0["edge_idx"])
    rho_edge = float(b0["rho_edge"])

    L = int(b0["t_len"])
    Te_edge = np.array(b0["Te_edge"][:L]).astype(float)

    if Te_edge.shape[0] != L:
        print(f"FAIL: Te_edge length {Te_edge.shape[0]} != t_len {L}", file=sys.stderr)
        sys.exit(2)

    Te_edge_min = float(np.min(Te_edge)) if Te_edge.size else 0.0
    Te_edge_max = float(np.max(Te_edge)) if Te_edge.size else 0.0
    Te_edge_ptp = Te_edge_max - Te_edge_min

    print(f"edge_bc_mode={edge_bc_mode} edge_idx={edge_idx} rho_edge={rho_edge:.4f}")
    print(f"Te_edge stats: min={Te_edge_min:.2f} max={Te_edge_max:.2f} ptp={Te_edge_ptp:.2f}")

    if Te_edge_ptp < args.ptp_tol:
        print(f"FAIL: Te_edge peak-to-peak {Te_edge_ptp:.2f} < tol {args.ptp_tol}", file=sys.stderr)
        sys.exit(1)

    Te_edge_from_data = np.array(b0["ts_Te"][:L, edge_idx], dtype=float)
    mask_edge = np.array(b0["mask"][:L, edge_idx] > 0.5, dtype=bool)
    if mask_edge.any():
        data_vals = Te_edge_from_data[mask_edge]
        print(
            f"data edge node stats (masked): min={float(data_vals.min()):.2f} "
            f"max={float(data_vals.max()):.2f} ptp={float(data_vals.ptp()):.2f}"
        )
    else:
        print("data edge node stats (masked): no valid points")

    print("PASS")


if __name__ == "__main__":
    main()
