# scripts/check_bc.py
# BC regression script (Te_edge boundary-condition sanity check)
"""Quick BC regression check for a single shot."""

import argparse
import os
import sys
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from fusion_ode_identification.data import load_data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/config_debug.yaml")
    p.add_argument("--shot", type=int, default=None, help="Shot id to inspect (default: first in config)")
    p.add_argument("--ptp_tol", type=float, default=1.0, help="Minimum allowed Te_edge peak-to-peak (eV)")
    args = p.parse_args()

    cfg_override = {"data": {}, "training": {}}
    if args.shot is not None:
        cfg_override["data"]["shots"] = [int(args.shot)]

    # Minimal load: relies on load_data handling overrides in config file
    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data", {}).update(cfg_override["data"])

    bundles, rho_rom, rho_cap, obs_idx = load_data(cfg)
    b0 = {k: np.array(v[0]) for k, v in zip(bundles._fields, bundles)}

    edge_bc_mode = cfg.get("data", {}).get("edge_bc_mode", "use_last_observed")
    edge_idx = int(b0["edge_idx"])
    rho_edge = float(b0["rho_edge"])

    Te_edge = np.array(b0["Te_edge"][: int(b0["t_len"])]).astype(float)
    Te_edge_min = float(np.min(Te_edge)) if Te_edge.size else 0.0
    Te_edge_max = float(np.max(Te_edge)) if Te_edge.size else 0.0
    Te_edge_ptp = Te_edge_max - Te_edge_min

    print(f"edge_bc_mode={edge_bc_mode} edge_idx={edge_idx} rho_edge={rho_edge:.4f}")
    print(f"Te_edge stats: min={Te_edge_min:.2f} max={Te_edge_max:.2f} ptp={Te_edge_ptp:.2f}")

    if Te_edge_ptp < args.ptp_tol:
        print(f"FAIL: Te_edge peak-to-peak {Te_edge_ptp:.2f} < tol {args.ptp_tol}", file=sys.stderr)
        sys.exit(1)

    # Compare to outermost observed rho point per timestep (robust for sparse edge coverage)
    ts_Te = np.array(b0["ts_Te"][: int(b0["t_len"])], dtype=float)
    mask = np.array(b0["mask"][: int(b0["t_len"])], dtype=float)
    Te_last_obs = np.full((ts_Te.shape[0],), np.nan, dtype=float)
    for i in range(ts_Te.shape[0]):
        m = (mask[i] > 0.5) & np.isfinite(ts_Te[i])
        if not np.any(m):
            continue
        j1 = np.where(m)[0][-1]
        Te_last_obs[i] = float(ts_Te[i, j1])

    m_ok = np.isfinite(Te_last_obs) & np.isfinite(Te_edge)
    if np.any(m_ok):
        diff = Te_edge[m_ok] - Te_last_obs[m_ok]
        mae = float(np.mean(np.abs(diff)))
        mx = float(np.max(np.abs(diff)))
        print(f"Te_edge vs last-observed: N={int(m_ok.sum())} MAE={mae:.3f} max|diff|={mx:.3f}")
    else:
        print("Te_edge vs last-observed: no valid points")

    print("PASS")


if __name__ == "__main__":
    main()
