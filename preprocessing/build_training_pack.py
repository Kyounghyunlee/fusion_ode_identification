"""
Build TORAX-ready training packs from downloaded NetCDFs produced by
download_data.py. For each shot, we:

1) Load equilibrium.nc → compute flux coordinate (rho) and geometry scalars.
2) Load thomson_scattering.nc → map Te, ne onto a fixed rho grid.
3) Load summary.nc → collect global time-series (Ip, nebar, powers).
4) Save data/<shot>_torax_training.npz with arrays and geometry.

Usage:
  python -m scripts.build_training_pack --shots 30420 30421 30422
  python -m scripts.build_training_pack --shot 30421
  python -m scripts.build_training_pack --discover
"""

# preprocessing/build_training_pack.py

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import fsspec
import zarr

MIN_COL_COVERAGE = float(os.getenv("PACK_MIN_COL_COVERAGE", "0.0"))

from .geometry import (
    choose_itime,
    compute_rho_scalars,
    extract_geom_params,
    rho_from_RZ,
    volume_derivatives,
)


def find_shots_in_data(root: str = "data") -> List[int]:
    shots: List[int] = []
    if not os.path.isdir(root):
        return shots
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if name.isdigit() and os.path.isdir(p):
            if os.path.exists(os.path.join(p, "equilibrium.nc")):
                shots.append(int(name))
    return sorted(shots)


def get_var(ds: xr.Dataset, candidates: List[str]) -> Optional[xr.DataArray]:
    for c in candidates:
        if c in ds:
            return ds[c]
    return None


def interp_fill_1d(t: np.ndarray, arr: np.ndarray) -> np.ndarray:
    """Fill NaNs in a 1D signal via linear interpolation and edge carry.

    If all values are NaN, returns zeros. Assumes t is 1D, monotonically
    increasing. This is used to avoid propagating NaNs into control arrays
    saved in the packs.
    """
    out = np.array(arr, dtype=float)
    finite = np.isfinite(out)
    if not np.any(finite):
        return np.zeros_like(out)
    idx = np.flatnonzero(finite)
    out[~finite] = np.interp(t[~finite], t[idx], out[idx])
    # Carry edges
    out[: idx[0]] = out[idx[0]]
    out[idx[-1] + 1 :] = out[idx[-1]]
    return out


def infer_ts_radial_coordinate(ts: xr.Dataset) -> Optional[str]:
    # Prefer an explicit rho-like coordinate
    for cname in ("rho", "rho_ts", "psi_N", "psiN", "psi_norm"):
        if cname in ts.coords:
            return cname
    # Otherwise, we may have channel-based positions (R and maybe Z)
    if "R" in ts and ("Z" in ts or "Z_midplane" in ts or "Z_channel" in ts):
        return None  # will compute rho from R/Z positions
    # MAST TS uses 'major_radius' as R-like coordinate with midplane Z≈0
    if "major_radius" in ts.coords or "major_radius" in ts:
        return None
    # Fallback: if there's a single non-time dimension, assume it's radial
    non_time_dims = [d for d in ts.dims if d != "time"]
    if len(non_time_dims) == 1:
        return non_time_dims[0]
    return None


def profiles_to_rho_grid(
    rho_src: np.ndarray,
    values_t_s: np.ndarray,
    rho_dst: np.ndarray,
    values_mask_t_s: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate time-by-sample profiles (and optional masks) to a target rho grid.

    values_t_s: shape (Nt, Ns)
    values_mask_t_s: optional mask with same shape; when absent, mask is inferred
    from finite values after interpolation.
    Returns (values_interp, mask_interp).
    """
    Nt = values_t_s.shape[0]
    out = np.full((Nt, rho_dst.size), np.nan, dtype=float)
    out_mask = np.zeros_like(out)
    have_mask = values_mask_t_s is not None
    for t in range(Nt):
        v = values_t_s[t]
        vm = np.isfinite(rho_src) & np.isfinite(v)
        if np.count_nonzero(vm) == 0:
            continue
        rs = rho_src[vm]
        vs = v[vm]
        idx = np.argsort(rs)
        rs_sorted = rs[idx]
        vs_sorted = vs[idx]
        # no extrapolation; set nan outside hull
        out[t] = np.interp(rho_dst, rs_sorted, vs_sorted, left=np.nan, right=np.nan)

        if have_mask:
            mrow = values_mask_t_s[t]
            m_valid = np.isfinite(rho_src) & np.isfinite(mrow)
            if np.count_nonzero(m_valid) > 0:
                rs_m = rho_src[m_valid]
                ms_sorted = mrow[m_valid][np.argsort(rs_m)]
                rs_m_sorted = np.sort(rs_m)
                out_mask[t] = np.interp(rho_dst, rs_m_sorted, ms_sorted, left=0.0, right=0.0)
        else:
            out_mask[t] = np.isfinite(out[t]).astype(float)

    if have_mask:
        out_mask = (out_mask > 0.5).astype(float)
    return out, out_mask


# --- Sanity reporting helpers (formerly in scripts/sanity_check_packs.py) ---
CTRL_KEYS = ["P_nbi", "Ip", "nebar", "S_gas", "S_rec", "S_nbi"]
STATE_KEYS = ["Te", "ne"]
MASK_KEYS = ["Te_mask", "ne_mask"]


def _fmt_1d(name: str, arr: np.ndarray) -> str:
    total = arr.size
    n_nan = np.count_nonzero(~np.isfinite(arr))
    pct = 100.0 * n_nan / max(total, 1)
    return f"{name}: NaNs={n_nan} ({pct:.2f}%) len={total}"


def _fmt_2d(name: str, arr: np.ndarray) -> str:
    total = arr.size
    n_nan = np.count_nonzero(~np.isfinite(arr))
    pct = 100.0 * n_nan / max(total, 1)
    n_rows_nan = np.count_nonzero(~np.all(np.isfinite(arr), axis=1)) if arr.ndim == 2 else 0
    return f"{name}: NaNs={n_nan} ({pct:.2f}%), rows_with_nan={n_rows_nan}/{arr.shape[0]}, shape={arr.shape}"


def _fmt_mask(name: str, arr: np.ndarray) -> str:
    ones = int(np.sum(arr))
    total = arr.size
    return f"{name}: ones={ones} zeros={total - ones} ({100.0 * ones / max(total, 1):.2f}% valid)"


def sanity_report(path: str) -> Tuple[str, Tuple[int, bool, float, float]]:
    d = np.load(path)
    t = d["t"]
    t_ts = d["t_ts"]
    rho = d["rho"] if "rho" in d else None
    rho_fb = bool(d.get("rho_fallback_used", False))
    psi_axis = d.get("psi_axis", np.nan)
    psi_edge = d.get("psi_edge", np.nan)
    report: List[str] = []

    report.append(f"time summary: [{t.min():.4f}, {t.max():.4f}] len={len(t)}")
    report.append(f"time ts:      [{t_ts.min():.4f}, {t_ts.max():.4f}] len={len(t_ts)}")
    report.append(f"overlap:       [{max(t.min(), t_ts.min()):.4f}, {min(t.max(), t_ts.max()):.4f}]")
    report.append(f"rho fallback: {rho_fb}; psi_axis={psi_axis:.4g} psi_edge={psi_edge:.4g}")

    for key in STATE_KEYS:
        arr = d[key]
        report.append(_fmt_2d(key, arr))
        finite = np.isfinite(arr)
        if finite.any():
            vals = arr[finite]
            report.append(
                f"  {key} finite stats: min={vals.min():.3g} median={np.median(vals):.3g} max={vals.max():.3g}"
            )

    for key in MASK_KEYS:
        if key in d:
            m = d[key]
            report.append(_fmt_mask(key, m))
            if rho is not None and m.ndim == 2:
                cols = np.any(m, axis=0)
                rows = np.any(m, axis=1)
                n_cols = cols.size
                n_rows = rows.size
                n_cols_valid = int(np.sum(cols))
                n_rows_valid = int(np.sum(rows))
                rho_min = float(rho[cols].min()) if n_cols_valid else float("nan")
                rho_max = float(rho[cols].max()) if n_cols_valid else float("nan")
                col_cov = 100.0 * n_cols_valid / max(1, n_cols)
                row_cov = 100.0 * n_rows_valid / max(1, n_rows)
                report.append(
                    f"  coverage: cols {n_cols_valid}/{n_cols} ({col_cov:.2f}%), "
                    f"rows {n_rows_valid}/{n_rows} ({row_cov:.2f}%), "
                    f"rho span[{rho_min:.3f},{rho_max:.3f}]"
                )

    report.append("controls:")
    for key in CTRL_KEYS:
        if key in d:
            report.append("  " + _fmt_1d(key, d[key]))

    for extra in ["P_rad", "W_tot", "P_ohm", "P_tot", "H98", "q95", "li", "beta_n", "B_t0"]:
        if extra in d:
            report.append("  " + _fmt_1d(extra, d[extra]))

    def _is_mono(x):
        return np.all(np.diff(x) >= -1e-9)
    report.append(f"t mono={_is_mono(t)}, t_ts mono={_is_mono(t_ts)}")

    Te_cov = float(d["Te_mask"].mean()) if "Te_mask" in d else float("nan")
    ne_cov = float(d["ne_mask"].mean()) if "ne_mask" in d else float("nan")
    shot = int(os.path.basename(path).split("_")[0])
    return "\n".join(report), (shot, rho_fb, Te_cov, ne_cov)


def load_ts_from_zarr(
    shot: int,
    rho_fn,
    rho_torax: np.ndarray,
    manifest_path: Optional[str] = None,
) -> Optional[Dict[str, np.ndarray]]:
    """Load Thomson profiles directly from zarr (ayc/aye) and map to rho grid.

    Returns dict with t_ts, Te_rho_t, ne_rho_t, Te_mask, ne_mask or None on failure.
    """

    manifest_path = manifest_path or os.path.join("data", f"{shot}_zarr_manifest.json")
    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path) as f:
        mani = json.load(f)
    url = mani.get("url")
    endpoint = mani.get("endpoint")
    te_path = mani.get("thomson_te_path") or "ayc/te"
    if not url or not endpoint:
        return None

    group = te_path.split("/")[0]
    ne_path = te_path.replace("/te", "/ne") if "/te" in te_path else None
    radius_path = f"{group}/radius"
    time_path = f"{group}/time"

    mapper = fsspec.get_mapper(url, anon=True, client_kwargs={"endpoint_url": endpoint})
    try:
        root = zarr.open_consolidated(mapper, mode="r")
    except Exception:
        root = zarr.open(mapper, mode="r")

    if te_path not in root:
        return None
    te_arr = np.asarray(root[te_path][:])

    ne_arr = None
    for cand in [ne_path, f"{group}/ne", f"{group}/ne_core", f"{group}/ne_raw"]:
        if cand and cand in root:
            ne_arr = np.asarray(root[cand][:])
            break
    if ne_arr is None:
        return None

    if radius_path not in root:
        return None
    radius = np.asarray(root[radius_path][:])
    if radius.shape[0] != te_arr.shape[0]:
        raise ValueError("radius/te time dimension mismatch")

    if time_path in root:
        t_ts = np.asarray(root[time_path][:])
    else:
        t_ts = np.arange(te_arr.shape[0])

    Nt = te_arr.shape[0]
    Te_rho_t = np.full((Nt, rho_torax.size), np.nan, dtype=float)
    ne_rho_t = np.full_like(Te_rho_t)

    for t_idx in range(Nt):
        rho_chan = rho_fn(radius[t_idx], np.zeros_like(radius[t_idx]))
        # Fix B3: Interpolate per-time directly using that time's rho_chan
        # Te
        v_te = te_arr[t_idx]
        m_te = np.isfinite(rho_chan) & np.isfinite(v_te)
        if np.count_nonzero(m_te) > 0:
            rs = rho_chan[m_te]
            vs = v_te[m_te]
            idx = np.argsort(rs)
            Te_rho_t[t_idx] = np.interp(rho_torax, rs[idx], vs[idx], left=np.nan, right=np.nan)
        
        # ne
        v_ne = ne_arr[t_idx]
        m_ne = np.isfinite(rho_chan) & np.isfinite(v_ne)
        if np.count_nonzero(m_ne) > 0:
            rs = rho_chan[m_ne]
            vs = v_ne[m_ne]
            idx = np.argsort(rs)
            ne_rho_t[t_idx] = np.interp(rho_torax, rs[idx], vs[idx], left=np.nan, right=np.nan)

    return {
        "t_ts": t_ts,
        "Te_rho_t": Te_rho_t,
        "ne_rho_t": ne_rho_t,
        "Te_mask": Te_mask,
        "ne_mask": ne_mask,
    }


def build_one_shot(shot: int, data_root: str = "data", Nrho: int = 65, use_zarr: bool = False) -> str:
    shot_dir = os.path.join(data_root, str(shot))
    eq_path = os.path.join(shot_dir, "equilibrium.nc")
    ts_path = os.path.join(shot_dir, "thomson_scattering.nc")
    sm_path = os.path.join(shot_dir, "summary.nc")

    if not (os.path.exists(eq_path) and os.path.exists(ts_path) and os.path.exists(sm_path)):
        raise FileNotFoundError(f"Missing one or more NetCDF files for shot {shot}")

    eq = xr.load_dataset(eq_path)
    ts = xr.load_dataset(ts_path)
    summ = xr.load_dataset(sm_path)

    # Geometry and rho normalisation (with fallback if equilibrium is degenerate)
    it = choose_itime(eq)
    geom = extract_geom_params(eq, it)

    rho_fallback_used = False
    psi_axis_val = float("nan")
    psi_edge_val = float("nan")

    fallback_meta = {"rho_fallback_method": "psi", "rho_r_min": float("nan"), "rho_r_max": float("nan")}

    def _rho_from_R_linear(R: np.ndarray) -> np.ndarray:
        # Simple linear normalisation of R when psi is unusable
        r_candidates = ("major_radius", "R", "R_grid", "Rcoord", "R_grid_1d")
        r_vals = None
        for cand in r_candidates:
            if cand in eq.coords:
                r_vals = np.asarray(eq.coords[cand].values)
                break
            if cand in eq:
                r_vals = np.asarray(eq[cand].values)
                break
        if r_vals is None and "major_radius" in ts:
            r_vals = np.asarray(ts["major_radius"].values)
        if r_vals is None:
            r_min, r_max = 0.0, 1.0
        else:
            r_min = float(np.nanmin(r_vals))
            r_max = float(np.nanmax(r_vals))
            if not np.isfinite(r_min) or not np.isfinite(r_max) or abs(r_max - r_min) < 1e-9:
                r_min, r_max = 0.0, 1.0
        fallback_meta.update({"rho_fallback_method": "linear_R", "rho_r_min": r_min, "rho_r_max": r_max})
        return np.clip((R - r_min) / (r_max - r_min + 1e-6), 0.0, 1.0)

    try:
        scalars = compute_rho_scalars(eq, it)
        psi_axis_val = scalars["psi_axis"]
        psi_edge_val = scalars["psi_edge"]
        rho_fn = lambda r, z: rho_from_RZ(eq, r, z, itime=it)
    except Exception:
        rho_fallback_used = True
        rho_fn = lambda r, z: _rho_from_R_linear(r)

    rho_eq, V, Vprime = volume_derivatives(eq, it) if not rho_fallback_used else (None, None, None)

    # TORAX rho grid
    if rho_eq is not None and rho_eq.size >= Nrho:
        idx = np.linspace(0, rho_eq.size - 1, Nrho).astype(int)
        rho_torax = rho_eq[idx]
    else:
        rho_torax = np.linspace(0.0, 1.0, Nrho)

    # Interpolate Vprime onto TORAX rho grid
    if rho_eq is not None and Vprime is not None:
        Vprime_torax = np.interp(rho_torax, rho_eq, Vprime)
    else:
        Vprime_torax = np.ones_like(rho_torax)

    # Thomson scattering profiles
    used_zarr = False
    Te_units = ""
    ne_units = ""

    if use_zarr:
        try:
            zarr_payload = load_ts_from_zarr(shot, rho_fn, rho_torax)
        except Exception as e:
            zarr_payload = None
            print(f"  !! Zarr Thomson failed for shot {shot}: {e}")
        if zarr_payload:
            used_zarr = True
            t_ts = zarr_payload["t_ts"]
            Te_rho_t = zarr_payload["Te_rho_t"]
            ne_rho_t = zarr_payload["ne_rho_t"]
            Te_mask = zarr_payload["Te_mask"]
            ne_mask = zarr_payload["ne_mask"]
            Nt = Te_rho_t.shape[0]
            Te_units = zarr_payload.get("Te_units", "unknown_zarr")
            ne_units = zarr_payload.get("ne_units", "unknown_zarr")

    if not used_zarr:
        # Variables: try common names
        Te_da = get_var(ts, ["Te", "T_e", "te", "Te_eV", "t_e"])  # units may vary
        ne_da = get_var(ts, ["ne", "n_e", "ne_cm3", "ne_m3"])  # units may vary
        if Te_da is None or ne_da is None:
            raise KeyError("Could not find Te/ne in thomson_scattering.nc")

        # Time alignment: assume ts has time dimension named 'time'
        if ("time" not in Te_da.dims) and ("time" not in ne_da.dims):
            raise KeyError("Expected 'time' dimension in Thomson scattering variables")

        ts_sizes = getattr(ts, "sizes", {})
        Nt = ts_sizes.get("time")
        if Nt is None:
            Nt = Te_da.sizes.get("time")
        if Nt is None:
            Nt = ne_da.sizes.get("time", 0)

        # Determine radial-like axis for TS
        rho_coord_name = infer_ts_radial_coordinate(ts)
        if rho_coord_name is not None and rho_coord_name in ts.coords:
            rho_ts = ts.coords[rho_coord_name].values

            def to_time_samples(da: xr.DataArray) -> np.ndarray:
                dims = list(da.dims)
                if "time" in dims:
                    dims_no_time = [d for d in dims if d != "time"]
                    if dims_no_time:
                        arr = da.transpose("time", *dims_no_time).values
                        return arr.reshape(Nt, -1)
                    else:
                        arr = da.transpose("time").values
                        return arr.reshape(Nt, -1)
                else:
                    arr = da.values.reshape(1, -1)
                    return np.tile(arr, (Nt, 1))

            Te_ts = to_time_samples(Te_da)
            ne_ts = to_time_samples(ne_da)

            Te_rho_t, Te_mask = profiles_to_rho_grid(rho_ts, Te_ts, rho_torax)
            ne_rho_t, ne_mask = profiles_to_rho_grid(rho_ts, ne_ts, rho_torax)

        else:
            R_da = get_var(ts, ["R", "R_midplane", "R_channel", "major_radius"])
            Z_da = get_var(ts, ["Z", "Z_midplane", "Z_channel"])
            if R_da is None:
                raise KeyError("Thomson dataset missing R coordinate for channels and no rho given")

            def to_time_samples_fill(da: xr.DataArray) -> np.ndarray:
                if "time" in da.dims:
                    dims_no_time = [d for d in da.dims if d != "time"]
                    if dims_no_time:
                        arr = da.transpose("time", *dims_no_time).values
                    else:
                        arr = da.transpose("time").values[..., None]
                    return arr.reshape(Nt, -1)
                else:
                    arr = np.array(da.values).reshape(1, -1)
                    return np.tile(arr, (Nt, 1))

            R_t_s = to_time_samples_fill(R_da)
            if Z_da is None:
                Z_t_s = np.zeros_like(R_t_s)
            else:
                Z_t_s = to_time_samples_fill(Z_da)

            Te_t_s = to_time_samples_fill(Te_da)
            ne_t_s = to_time_samples_fill(ne_da)

            Te_rho_t = np.full((Nt, rho_torax.size), np.nan, dtype=float)
            ne_rho_t = np.full_like(Te_rho_t, np.nan)
            Te_mask = np.zeros_like(Te_rho_t)
            ne_mask = np.zeros_like(ne_rho_t)
            for t_idx in range(Nt):
                rho_chan = rho_fn(R_t_s[t_idx], Z_t_s[t_idx])
                vals_te, mask_te = profiles_to_rho_grid(rho_chan, Te_t_s[t_idx : t_idx + 1], rho_torax)
                vals_ne, mask_ne = profiles_to_rho_grid(rho_chan, ne_t_s[t_idx : t_idx + 1], rho_torax)
                Te_rho_t[t_idx] = vals_te[0]
                Te_mask[t_idx] = mask_te[0]
                ne_rho_t[t_idx] = vals_ne[0]
                ne_mask[t_idx] = mask_ne[0]

        t_ts = ts["time"].values if "time" in ts.coords else np.arange(Nt)

    # Combine upstream TS masks with finite checks (do not overwrite semantics)
    if 'Te_mask' not in locals():
        Te_mask = np.ones_like(Te_rho_t, dtype=bool)
    if 'ne_mask' not in locals():
        ne_mask = np.ones_like(ne_rho_t, dtype=bool)

    Te_mask = ((Te_mask > 0.5) & np.isfinite(Te_rho_t)).astype(float)
    ne_mask = ((ne_mask > 0.5) & np.isfinite(ne_rho_t)).astype(float)

    # Summary signals
    t = get_var(summ, ["time"]).values
    Ip_da = get_var(summ, ["ip", "Ip"])
    Ip = Ip_da.values if Ip_da is not None else np.full_like(t, np.nan)
    nebar_da = get_var(summ, ["line_average_n_e", "ne_bar", "nebar"])  # m^-3 typically
    nebar = nebar_da.values if nebar_da is not None else np.full_like(t, np.nan)
    P_nbi_da = get_var(summ, ["power_nbi", "P_NBI", "pnbi"])
    P_rad_da = get_var(summ, ["power_radiated", "P_rad", "prad"])
    P_nbi = P_nbi_da.values if P_nbi_da is not None else np.full_like(t, np.nan)
    P_rad = P_rad_da.values if P_rad_da is not None else np.full_like(t, np.nan)

    # Preserve raw (pre-filled) copies
    P_nbi_raw = np.array(P_nbi, copy=True)
    P_rad_raw = np.array(P_rad, copy=True)

    # Fill NaNs in key control signals to avoid gaps downstream
    Ip = interp_fill_1d(t, Ip)
    nebar = interp_fill_1d(t, nebar)
    P_nbi = interp_fill_1d(t, P_nbi)
    P_rad = interp_fill_1d(t, P_rad)

    # Extended Summary Signals (Level-2)
    W_tot_da = get_var(summ, ["W_tot", "w_tot", "stored_energy", "energy_total"])
    P_ohm_da = get_var(summ, ["p_ohm", "power_ohmic", "P_ohm"])
    P_tot_da = get_var(summ, ["p_tot", "power_total", "P_tot"])
    ne_line_da = get_var(summ, ["n_e_line", "ne_line", "line_average_density", "line_average_n_e"])
    H98_da = get_var(summ, ["H98", "H_98", "h98", "h_factor_98y2"])

    W_tot = W_tot_da.values if W_tot_da is not None else None
    P_ohm = P_ohm_da.values if P_ohm_da is not None else None
    P_tot = P_tot_da.values if P_tot_da is not None else None
    ne_line = ne_line_da.values if ne_line_da is not None else None
    H98 = H98_da.values if H98_da is not None else None

    # Magnetics / Equilibrium Scalars (try summary first, then equilibrium)
    q95_da = get_var(summ, ["q95", "q_95"])
    if q95_da is None:
        q95_da = get_var(eq, ["q95", "q_95"])
    q95 = q95_da.values if q95_da is not None else None

    li_da = get_var(summ, ["li", "li_3", "internal_inductance"])
    if li_da is None:
        li_da = get_var(eq, ["li", "li_3"])
    li = li_da.values if li_da is not None else None

    beta_n_da = get_var(summ, ["beta_n", "beta_N", "beta_normalised"])
    if beta_n_da is None:
        beta_n_da = get_var(eq, ["beta_n", "beta_N"])
    beta_n = beta_n_da.values if beta_n_da is not None else None

    B_t0_da = get_var(summ, ["B_t0", "b_t0", "toroidal_field_center"])
    if B_t0_da is None:
        B_t0_da = get_var(eq, ["B_t0", "b_t0"])
    B_t0 = B_t0_da.values if B_t0_da is not None else None

    # Particle Sources (Optional)
    gas_path = os.path.join(shot_dir, "gas_injection.nc")
    spec_path = os.path.join(shot_dir, "spectrometer_visible.nc")
    
    S_gas = np.zeros_like(t)
    S_rec = np.zeros_like(t)
    S_nbi = np.zeros_like(t)

    # 1. Gas Puffing (S_gas)
    if os.path.exists(gas_path):
        try:
            gas_ds = xr.load_dataset(gas_path)
            total_inj = get_var(gas_ds, ["total_injected", "flow_rate_total"])
            if total_inj is not None:
                t_gas = gas_ds["time"].values
                # Compute rate: d(count)/dt if cumulative, else use flow rate directly
                # Check units or name. "total_injected" sounds cumulative.
                # "flow_rate_total" sounds like rate.
                # Assuming cumulative for "total_injected" based on previous context.
                if "total_injected" in gas_ds:
                    rate_gas = np.gradient(total_inj.values, t_gas)
                else:
                    rate_gas = total_inj.values
                
                # Interpolate to summary time
                S_gas = np.interp(t, t_gas, rate_gas, left=0.0, right=0.0)
                S_gas = np.maximum(S_gas, 0.0)
        except Exception as e:
            print(f"  !! Failed to load gas injection for shot {shot}: {e}")

    # 2. Recycling (S_rec) - D_alpha proxy
    if os.path.exists(spec_path):
        try:
            spec_ds = xr.load_dataset(spec_path)
            # Try to find D_alpha
            dalpha = get_var(spec_ds, ["filter_spectrometer_dalpha_voltage", "d_alpha", "D_alpha"])
            if dalpha is not None:
                # Sum over channels if multiple exist
                if "dalpha_channel" in dalpha.dims:
                    dalpha_sum = dalpha.sum(dim="dalpha_channel")
                elif "channel" in dalpha.dims:
                    dalpha_sum = dalpha.sum(dim="channel")
                else:
                    dalpha_sum = dalpha
                
                t_spec = spec_ds["time"].values
                # Interpolate to summary time
                S_rec_raw = np.interp(t, t_spec, dalpha_sum.values, left=0.0, right=0.0)
                S_rec = np.maximum(S_rec_raw, 0.0)
        except Exception as e:
            print(f"  !! Failed to load spectrometer for shot {shot}: {e}")

    # 3. NBI Fueling (S_nbi)
    # Approx: P_nbi [W] / (E_beam [eV] * e)
    # Assuming E_beam ~ 75 keV for MAST
    E_beam_eV = 75000.0
    e_charge = 1.60217663e-19
    S_nbi = np.maximum(P_nbi, 0.0) / (E_beam_eV * e_charge)

    # Extract time arrays
    t_summary = get_var(summ, ["time"]).values
    if not used_zarr:
        t_ts = ts["time"].values if "time" in ts.coords else np.arange(Nt)

    # Simple regime labelling on summary grid (t_summary)
    # 0 = unknown, 1 = L-mode, 2 = transition, 3 = H-mode
    regime = np.zeros_like(t_summary, dtype=np.int8)

    # crude heuristic: look at nebar and P_NBI threshold & derivative
    if nebar_da is not None and P_nbi_da is not None:
        def _smooth(x: np.ndarray, k: int = 5) -> np.ndarray:
            if k <= 1:
                return x
            k = min(k, x.size)
            filt = np.ones(k) / k
            return np.convolve(x, filt, mode="same")

        ne_norm = (nebar - np.nanmin(nebar)) / (np.nanmax(nebar) - np.nanmin(nebar) + 1e-6)
        P_nbi_norm = (P_nbi - np.nanmin(P_nbi)) / (np.nanmax(P_nbi) - np.nanmin(P_nbi) + 1e-6)
        ne_s = _smooth(ne_norm)
        pnbi_s = _smooth(P_nbi_norm)
        d_ne = np.gradient(ne_s, t_summary)
        d_pnbi = np.gradient(pnbi_s, t_summary)
        score = np.abs(d_ne) + 0.3 * np.abs(d_pnbi)
        trans_idx = int(np.argmax(score))
        regime[trans_idx] = 2
        w = max(2, regime.size // 20)
        regime[: max(0, trans_idx - w)] = 1
        regime[min(regime.size - 1, trans_idx + w) :] = 3

    # Optionally drop rho columns with zero coverage to reduce dimensionality
    if MIN_COL_COVERAGE > 0.0:
        # Fix B2: Use actual coverage threshold
        col_cov = np.maximum(Te_mask.mean(axis=0), ne_mask.mean(axis=0))
        keep_idx = np.where(col_cov >= MIN_COL_COVERAGE)[0]
        n_before = Te_mask.shape[1]
        if keep_idx.size > 0:
            Te_rho_t = Te_rho_t[:, keep_idx]
            ne_rho_t = ne_rho_t[:, keep_idx]
            Te_mask = Te_mask[:, keep_idx]
            ne_mask = ne_mask[:, keep_idx]
            rho_torax = rho_torax[keep_idx]
            if Vprime_torax is not None and Vprime_torax.shape[0] == n_before:
                Vprime_torax = Vprime_torax[keep_idx]

    # Coverage diagnostics (post any column drop)
    Te_mask_col_cov = Te_mask.mean(axis=0).astype(np.float32)
    Te_mask_row_cov = Te_mask.mean(axis=1).astype(np.float32)
    ne_mask_col_cov = ne_mask.mean(axis=0).astype(np.float32)
    ne_mask_row_cov = ne_mask.mean(axis=1).astype(np.float32)

    Te_rho_t = np.nan_to_num(Te_rho_t, nan=0.0, posinf=0.0, neginf=0.0)
    ne_rho_t = np.nan_to_num(ne_rho_t, nan=0.0, posinf=0.0, neginf=0.0)

    # Coverage summaries
    Te_mask_mean = float(Te_mask.mean())
    ne_mask_mean = float(ne_mask.mean())
    edge_mask = rho_torax >= 0.8
    Te_mask_mean_edge = float(Te_mask[:, edge_mask].mean()) if edge_mask.any() else float('nan')
    ne_mask_mean_edge = float(ne_mask[:, edge_mask].mean()) if edge_mask.any() else float('nan')

    # Save NPZ bundle
    out_npz = os.path.join(data_root, f"{shot}_torax_training.npz")
    payload = dict(
        t=t_summary,
        t_ts=t_ts,
        rho=rho_torax,
        Te=Te_rho_t,
        ne=ne_rho_t,
        Te_mask=Te_mask,
        ne_mask=ne_mask,
        Te_mask_col_cov=Te_mask_col_cov,
        Te_mask_row_cov=Te_mask_row_cov,
        ne_mask_col_cov=ne_mask_col_cov,
        ne_mask_row_cov=ne_mask_row_cov,
        Ip=Ip,
        nebar=nebar,
        P_nbi=P_nbi,
        P_rad=P_rad,
        P_nbi_raw=P_nbi_raw,
        P_rad_raw=P_rad_raw,
        S_gas=S_gas,
        S_rec=S_rec,
        S_nbi=S_nbi,
        Vprime=Vprime_torax,
        regime=regime,
        schema_version=2,
        psi_axis=psi_axis_val,
        psi_edge=psi_edge_val,
        rho_fallback_used=rho_fallback_used,
        rho_fallback_method=fallback_meta["rho_fallback_method"],
        rho_r_min=fallback_meta["rho_r_min"],
        rho_r_max=fallback_meta["rho_r_max"],
        Te_units=Te_units if Te_units else (str(Te_da.attrs.get("units", "")) if 'Te_da' in locals() and Te_da is not None else ""),
        ne_units=ne_units if ne_units else (str(ne_da.attrs.get("units", "")) if 'ne_da' in locals() and ne_da is not None else ""),
        Te_mask_mean=Te_mask_mean,
        ne_mask_mean=ne_mask_mean,
        Te_mask_mean_edge=Te_mask_mean_edge,
        ne_mask_mean_edge=ne_mask_mean_edge,
        **geom,
    )

    # Attach optional signals only if present and not all-NaN
    optional = {
        "W_tot": W_tot,
        "P_ohm": P_ohm,
        "P_tot": P_tot,
        "ne_line": ne_line,
        "H98": H98,
        "q95": q95,
        "li": li,
        "beta_n": beta_n,
        "B_t0": B_t0,
    }
    for key, arr in optional.items():
        if arr is None:
            continue
        if arr.shape != t.shape:
            print(f"  .. dropping {key} (shape mismatch {arr.shape} vs {t.shape})")
            continue
        finite = np.isfinite(arr)
        if np.any(finite):
            payload[key] = interp_fill_1d(t, arr)
        else:
            print(f"  .. dropping {key} (all NaN or missing)")

    np.savez_compressed(out_npz, **payload)

    return out_npz


def main():
    ap = argparse.ArgumentParser(description="Build TORAX training packs from NetCDFs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--shot", type=int, help="Single shot number to process")
    g.add_argument("--shots", type=int, nargs="+", help="List of shots")
    g.add_argument("--discover", action="store_true", help="Discover shots under data/<shot>")
    ap.add_argument("--Nrho", type=int, default=65, help="Number of rho grid points (default: 65)")
    ap.add_argument("--use-zarr", action="store_true", help="Use zarr Thomson data (ayc/aye) if manifest exists")
    args = ap.parse_args()

    if args.discover:
        shots = find_shots_in_data("data")
    elif args.shot is not None:
        shots = [args.shot]
    else:
        shots = args.shots

    created: List[str] = []
    summary_rows: List[Tuple[int, bool, float, float]] = []
    for shot in shots:
        print(f"[build] Shot {shot}")
        try:
            path = build_one_shot(shot, data_root="data", Nrho=args.Nrho, use_zarr=args.use_zarr)
            print(f"  -> {path}")
            # Inline sanity report
            try:
                rep, row = sanity_report(path)
                print(rep)
                summary_rows.append(row)
            except Exception as rep_exc:
                print(f"  !! Sanity report failed for {shot}: {rep_exc}")
            created.append(path)
        except Exception as e:
            print(f"  !! Failed for shot {shot}: {e}")

    if summary_rows:
        summary_rows.sort()
        print("=== Coverage summary ===")
        print("shot fallback Te_mask_cov ne_mask_cov")
        for row in summary_rows:
            print(f"{row[0]} {row[1]} {row[2]:.3f} {row[3]:.3f}")

    if not created:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
