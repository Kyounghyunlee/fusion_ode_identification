"""
Build TORAX-ready training packs from downloaded NetCDFs produced by
download_data.py. For each shot, we:

1) Load equilibrium.nc → compute flux coordinate (rho) and geometry scalars.
2) Load thomson_scattering.nc → map Te, ne onto a fixed rho grid.
3) Load summary.nc → collect global time-series (Ip, nebar, powers).
4) Save data/<shot>_torax_training.npz with arrays and geometry.

Usage:
    python -m preprocessing.build_training_pack --shots 30420 30421 30422
    python -m preprocessing.build_training_pack --shot 30421
    python -m preprocessing.build_training_pack --discover
    python preprocessing/build_training_pack.py --discover
"""

# preprocessing/build_training_pack.py

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr  # Loads NetCDF files

if __package__ in (None, ""):
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from preprocessing.geometry import (
        choose_itime,
        compute_rho_scalars,
        extract_geom_params,
        rho_from_RZ,
        volume_derivatives,
    )
else:
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


def get_var(ds: xr.Dataset, candidates: List[str]) -> Optional[xr.DataArray]: # Try multiple candidate names for a variable in the dataset, returning the first match.
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
    # Carry edges : these are actually redundant with the left/right fill in np.interp, but just to be safe
    out[: idx[0]] = out[idx[0]]
    out[idx[-1] + 1 :] = out[idx[-1]]
    return out


def sort_unique_series(t: np.ndarray, arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    arr = np.asarray(arr, dtype=float)
    order = np.argsort(t)
    t = t[order]
    arr = arr[order]
    if t.size > 1:
        keep = np.concatenate([[True], np.diff(t) > 0])
        t = t[keep]
        arr = arr[keep]
    return t, arr


def find_first_existing(paths: List[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def extract_dalpha_arrays(ds: xr.Dataset) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return D-alpha time, channel matrix, channel names, and aggregate signal."""
    dalpha_da = get_var(ds, ["D_alpha", "filter_spectrometer_dalpha_voltage", "d_alpha", "D_alpha_sum"])
    if dalpha_da is None:
        raise KeyError("Could not find D-alpha variable")

    time_dim = next((dim for dim in dalpha_da.dims if "time" in dim.lower()), None)
    if time_dim is None:
        time_dim = dalpha_da.dims[-1]

    other_dims = [dim for dim in dalpha_da.dims if dim != time_dim]
    time_vals = ds.coords[time_dim].values if time_dim in ds.coords else np.arange(dalpha_da.sizes[time_dim], dtype=float)

    if len(other_dims) == 0:
        channel_vals = np.asarray(dalpha_da.transpose(time_dim).values, dtype=float)[None, :]
        channel_names = np.array([str(dalpha_da.attrs.get("uda_name", dalpha_da.name or "D_alpha"))], dtype="U64")
    elif len(other_dims) == 1:
        channel_dim = other_dims[0]
        channel_vals = np.asarray(dalpha_da.transpose(channel_dim, time_dim).values, dtype=float)
        if channel_dim in ds.coords:
            channel_names = np.asarray(ds.coords[channel_dim].values).astype("U64")
        else:
            channel_names = np.array([f"{dalpha_da.name or 'D_alpha'}_{i:02d}" for i in range(channel_vals.shape[0])], dtype="U64")
    else:
        stacked = dalpha_da.stack(dalpha_channel=other_dims).transpose("dalpha_channel", time_dim)
        channel_vals = np.asarray(stacked.values, dtype=float)
        channel_names = np.asarray([str(v) for v in stacked.coords["dalpha_channel"].values], dtype="U64")

    order = np.argsort(time_vals)
    time_vals = np.asarray(time_vals, dtype=float)[order]
    channel_vals = channel_vals[:, order]
    if time_vals.size > 1:
        keep = np.concatenate([[True], np.diff(time_vals) > 0])
        time_vals = time_vals[keep]
        channel_vals = channel_vals[:, keep]

    if "D_alpha_sum" in ds:
        dalpha_sum = np.asarray(ds["D_alpha_sum"].transpose(time_dim).values, dtype=float)[order]
        if time_vals.size > 1:
            dalpha_sum = dalpha_sum[keep]
    else:
        dalpha_sum = np.nansum(channel_vals, axis=0)

    return time_vals, channel_vals, channel_names, dalpha_sum


def interp_channels_to_time(t_src: np.ndarray, values_c_t: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    """Interpolate channel-by-time signals onto a target time base."""
    out = np.zeros((t_dst.size, values_c_t.shape[0]), dtype=float)
    for idx, row in enumerate(values_c_t):
        valid = np.isfinite(t_src) & np.isfinite(row)
        if np.count_nonzero(valid) == 0:
            continue
        src_t = t_src[valid]
        src_v = row[valid]
        out[:, idx] = np.interp(t_dst, src_t, src_v, left=src_v[0], right=src_v[-1])
    return out


def estimate_regime_labels(
    t: np.ndarray,
    nebar: np.ndarray,
    P_nbi: np.ndarray,
    D_alpha: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Estimate L/H transition timing using D-alpha drop plus actuator rises."""
    regime = np.zeros_like(t, dtype=np.int8)
    score = np.zeros_like(t, dtype=float)
    if t.size < 5:
        return regime, score, float("nan")

    def _smooth(x: np.ndarray, k: int) -> np.ndarray:
        if x.size < 3:
            return x
        k = min(k, x.size)
        if k <= 1:
            return x
        filt = np.ones(k, dtype=float) / float(k)
        return np.convolve(x, filt, mode="same")

    def _norm(x: np.ndarray) -> Optional[np.ndarray]:
        if not np.any(np.isfinite(x)):
            return None
        x_filled = interp_fill_1d(t, x)
        span = float(np.nanmax(x_filled) - np.nanmin(x_filled))
        if span < 1e-9:
            return None
        return (x_filled - np.nanmin(x_filled)) / (span + 1e-6)

    score_terms = []

    d_alpha_norm = _norm(D_alpha)
    if d_alpha_norm is not None:
        d_alpha_s = _smooth(d_alpha_norm, 101)
        score_terms.append(1.2 * np.maximum(-np.gradient(d_alpha_s, t), 0.0))

    ne_norm = _norm(nebar)
    if ne_norm is not None:
        ne_s = _smooth(ne_norm, 31)
        score_terms.append(0.7 * np.maximum(np.gradient(ne_s, t), 0.0))

    pnbi_norm = _norm(P_nbi)
    if pnbi_norm is not None:
        pnbi_s = _smooth(pnbi_norm, 31)
        score_terms.append(0.3 * np.maximum(np.gradient(pnbi_s, t), 0.0))

    if not score_terms:
        return regime, score, float("nan")

    score = np.sum(score_terms, axis=0)
    pad = max(10, score.size // 20)
    if score.size > 2 * pad:
        score[:pad] = 0.0
        score[-pad:] = 0.0

    trans_idx = int(np.argmax(score))
    if not np.isfinite(score[trans_idx]) or score[trans_idx] <= 0.0:
        return regime, score, float("nan")

    regime[:] = 1
    width = max(2, regime.size // 20)
    lo = max(0, trans_idx - width)
    hi = min(regime.size, trans_idx + width + 1)
    regime[lo:hi] = 2
    regime[hi:] = 3
    return regime, score, float(t[trans_idx])


def infer_ts_radial_coordinate(ts: xr.Dataset) -> Optional[str]: 
    """Heuristic to infer which coordinate in the Thomson scattering dataset corresponds to the radial-like position (rho or similar)."""
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
    rho_src: np.ndarray, # 1D array of source rho positions for the profiles (shape (Ns,))
    values_t_s: np.ndarray, # 2D array of profile values with shape (Nt, Ns) where Nt is the number of time samples and Ns is the number of spatial samples in the original profiles.
    rho_dst: np.ndarray, # 1D array of target rho positions for interpolation (shape (Nrho,))
    values_mask_t_s: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate time-by-sample profiles (and optional masks) to a target rho grid.

    values_t_s: shape (Nt, Ns)
    values_mask_t_s: optional mask with same shape; when absent, mask is inferred
    from finite values after interpolation.
    Returns (values_interp, mask_interp).
    """
    Nt = values_t_s.shape[0]
    out = np.full((Nt, rho_dst.size), np.nan, dtype=float) # Pre fill with NaNs to identify missing data after interpolation
    out_mask = np.zeros_like(out) # Create mask array initialized to zeros; will set to 1 where valid data exists after interpolation
    have_mask = values_mask_t_s is not None # True or False depending on whether an explicit mask was provided
    for t in range(Nt):
        v = values_t_s[t]
        vm = np.isfinite(rho_src) & np.isfinite(v)
        if np.count_nonzero(vm) == 0:
            continue # No valid data for this time sample; leave as NaN/0 in output
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


def _fmt_1d(name: str, arr: np.ndarray) -> str: # Format a 1D array for reporting, showing number of NaNs and basic stats.
    total = arr.size
    n_nan = np.count_nonzero(~np.isfinite(arr))
    pct = 100.0 * n_nan / max(total, 1)
    return f"{name}: NaNs={n_nan} ({pct:.2f}%) len={total}"


def _fmt_2d(name: str, arr: np.ndarray) -> str: # Format a 2D array for reporting, showing number of NaNs, rows with NaNs, and shape.
    total = arr.size
    n_nan = np.count_nonzero(~np.isfinite(arr))
    pct = 100.0 * n_nan / max(total, 1)
    n_rows_nan = np.count_nonzero(~np.all(np.isfinite(arr), axis=1)) if arr.ndim == 2 else 0
    return f"{name}: NaNs={n_nan} ({pct:.2f}%), rows_with_nan={n_rows_nan}/{arr.shape[0]}, shape={arr.shape}"


def _fmt_mask(name: str, arr: np.ndarray) -> str: # Format a mask array for reporting, showing number of ones, zeros, and percentage of valid entries.
    ones = int(np.sum(arr))
    total = arr.size
    return f"{name}: ones={ones} zeros={total - ones} ({100.0 * ones / max(total, 1):.2f}% valid)"


def sanity_report(path: str) -> Tuple[str, Tuple[int, bool, float, float]]:
    d = np.load(path)
    t = d["t"]
    t_ts = d["t_ts"] # Thomson scattering time samples
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

    for extra in ["P_rad", "W_tot", "P_ohm", "P_tot", "H98", "q95", "li", "beta_n", "B_t0", "D_alpha", "regime_score", "transition_time"]:
        if extra in d:
            report.append("  " + _fmt_1d(extra, d[extra]))

    def _is_mono(x): # Check if array is monotonically non-decreasing 
        return np.all(np.diff(x) >= -1e-9)
    report.append(f"t mono={_is_mono(t)}, t_ts mono={_is_mono(t_ts)}")

    Te_cov = float(d["Te_mask"].mean()) if "Te_mask" in d else float("nan")
    ne_cov = float(d["ne_mask"].mean()) if "ne_mask" in d else float("nan")
    shot = int(os.path.basename(path).split("_")[0])
    return "\n".join(report), (shot, rho_fb, Te_cov, ne_cov)


def build_one_shot(shot: int, data_root: str = "data", Nrho: int = 65) -> str:
    shot_dir = os.path.join(data_root, str(shot))
    eq_path = os.path.join(shot_dir, "equilibrium.nc")
    ts_path = os.path.join(shot_dir, "thomson_scattering.nc")
    sm_path = os.path.join(shot_dir, "summary.nc")

    if not (os.path.exists(eq_path) and os.path.exists(ts_path) and os.path.exists(sm_path)):
        raise FileNotFoundError(f"Missing one or more NetCDF files for shot {shot}")

    eq = xr.load_dataset(eq_path) # Load equilibrium dataset; used for geometry and flux coordinates
    ts = xr.load_dataset(ts_path) # Load Thomson scattering dataset; used for Te and ne profiles
    summ = xr.load_dataset(sm_path) # Load summary dataset; used for global time-series like Ip, nebar, powers

    # Geometry and rho normalisation (with fallback if equilibrium is degenerate)
    it = choose_itime(eq) # pick representative time index for equilibrium; middle of time dimension.
    geom = extract_geom_params(eq, it) # Extract geometry parameters (R_major, a_minor, kappa, delta) from equilibrium at chosen time index

    rho_fallback_used = False
    psi_axis_val = float("nan")
    psi_edge_val = float("nan")

    fallback_meta = {"rho_fallback_method": "psi", "rho_r_min": float("nan"), "rho_r_max": float("nan")}

    def _rho_from_R_linear(R: np.ndarray) -> np.ndarray: # Rho fallback function
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

    # Thomson scattering profiles from NetCDF
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

            def to_time_samples(da: xr.DataArray) -> np.ndarray: # Convert DataArray to 2D time-by-sample array, aligning time dimension if present. If no time dimension, replicate across time samples.
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

    # Extract units from NetCDF metadata
    Te_units = str(Te_da.attrs.get("units", "")) if Te_da is not None else ""
    ne_units = str(ne_da.attrs.get("units", "")) if ne_da is not None else ""

    # Combine upstream TS masks with finite checks (keep as bool)
    Te_mask = (Te_mask > 0.5) & np.isfinite(Te_rho_t)
    ne_mask = (ne_mask > 0.5) & np.isfinite(ne_rho_t)

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
    dalpha_path = os.path.join(shot_dir, "d_alpha.nc")
    spec_path = os.path.join(shot_dir, "spectrometer_visible.nc")

    S_gas = np.zeros_like(t)
    S_rec = np.zeros_like(t)
    S_nbi = np.zeros_like(t)
    D_alpha = np.zeros_like(t)
    D_alpha_channels = np.zeros((t.size, 0), dtype=float)
    D_alpha_channel_names = np.array([], dtype="U64")

    # 1. Gas Puffing (S_gas)
    if os.path.exists(gas_path):
        try:
            gas_ds = xr.load_dataset(gas_path)
            total_inj = get_var(gas_ds, ["total_injected", "flow_rate_total"])
            if total_inj is not None:
                t_gas = gas_ds["time"].values
                gas_vals = total_inj.values
                t_gas, gas_vals = sort_unique_series(t_gas, gas_vals)
                # Compute rate: d(count)/dt if cumulative, else use flow rate directly
                # Check units or name. "total_injected" sounds cumulative.
                # "flow_rate_total" sounds like rate.
                # Assuming cumulative for "total_injected" based on previous context.
                if "total_injected" in gas_ds:
                    rate_gas = np.gradient(interp_fill_1d(t_gas, gas_vals), t_gas)
                else:
                    rate_gas = interp_fill_1d(t_gas, gas_vals)
                rate_gas = np.nan_to_num(rate_gas, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Interpolate to summary time
                S_gas = np.interp(t, t_gas, rate_gas, left=0.0, right=0.0)
                S_gas = np.maximum(np.nan_to_num(S_gas, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        except Exception as e:
            print(f"  !! Failed to load gas injection for shot {shot}: {e}")

    # 2. Recycling (S_rec) and raw D-alpha signal
    dalpha_source_path = find_first_existing([dalpha_path, spec_path])
    if dalpha_source_path is not None:
        try:
            dalpha_ds = xr.load_dataset(dalpha_source_path)
            t_dalpha, dalpha_channels_native, D_alpha_channel_names, _dalpha_sum_native = extract_dalpha_arrays(dalpha_ds)
            D_alpha_channels = interp_channels_to_time(t_dalpha, dalpha_channels_native, t)
            D_alpha = np.sum(D_alpha_channels, axis=1)
            D_alpha = interp_fill_1d(t, D_alpha)
            S_rec = np.maximum(D_alpha, 0.0)
        except Exception as e:
            print(f"  !! Failed to load D-alpha for shot {shot}: {e}")

    # 3. NBI Fueling (S_nbi)
    # Approx: P_nbi [W] / (E_beam [eV] * e)
    # Assuming E_beam ~ 75 keV for MAST
    E_beam_eV = 75000.0
    e_charge = 1.60217663e-19
    S_nbi = np.maximum(P_nbi, 0.0) / (E_beam_eV * e_charge)

    # Extract time arrays
    t_summary = get_var(summ, ["time"]).values

    # Simple regime labelling on summary grid (t_summary)
    # 0 = unknown, 1 = L-mode, 2 = transition, 3 = H-mode
    regime, regime_score, transition_time = estimate_regime_labels(t_summary, nebar, P_nbi, D_alpha)

    # Coverage diagnostics
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
        D_alpha=D_alpha,
        D_alpha_channels=D_alpha_channels,
        D_alpha_channel_names=D_alpha_channel_names,
        S_gas=S_gas,
        S_rec=S_rec,
        S_nbi=S_nbi,
        Vprime=Vprime_torax,
        regime=regime,
        regime_score=regime_score,
        transition_time=transition_time,
        schema_version=3,
        psi_axis=psi_axis_val,
        psi_edge=psi_edge_val,
        rho_fallback_used=rho_fallback_used,
        rho_fallback_method=fallback_meta["rho_fallback_method"],
        rho_r_min=fallback_meta["rho_r_min"],
        rho_r_max=fallback_meta["rho_r_max"],
        Te_units=Te_units,
        ne_units=ne_units,
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
            path = build_one_shot(shot, data_root="data", Nrho=args.Nrho)
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
