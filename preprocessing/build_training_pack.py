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
import csv
import os
import sys
from dataclasses import dataclass
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


def _smooth_time_window(x: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
    """Moving average over an approximately fixed time window (seconds)."""
    if x.size < 3:
        return x
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0:
        return x
    k = int(np.clip(round(window_s / dt), 1, x.size))
    if k <= 1:
        return x
    pad_l = k // 2
    pad_r = k - 1 - pad_l
    x_pad = np.concatenate([np.repeat(x[:1], pad_l), x, np.repeat(x[-1:], pad_r)])
    return np.convolve(x_pad, np.ones(k) / k, mode="valid")


def _rolling_quantile(x: np.ndarray, t: np.ndarray, window_s: float, q: float) -> np.ndarray:
    """Rolling quantile over an approximately fixed time window (seconds).

    Used to extract the lower envelope (baseline) of D-alpha: ELM bursts are
    short positive spikes, so a low quantile tracks the inter-ELM level.
    """
    dt = float(np.median(np.diff(t)))
    if not np.isfinite(dt) or dt <= 0 or x.size < 5:
        return x.copy()
    half = max(1, int(round(0.5 * window_s / dt)))
    out = np.empty_like(x, dtype=float)
    for i in range(x.size):
        lo = max(0, i - half)
        hi = min(x.size, i + half + 1)
        out[i] = np.nanquantile(x[lo:hi], q)
    return out


def _otsu_threshold(x: np.ndarray, n_bins: int = 128) -> float:
    """Two-class variance-maximizing threshold (Otsu) on a 1D sample."""
    x = x[np.isfinite(x)]
    hist, edges = np.histogram(x, bins=n_bins)
    hist = hist.astype(float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    m0 = np.cumsum(hist * centers)
    mu0 = np.divide(m0, w0, out=np.zeros_like(m0), where=w0 > 0)
    mu1 = np.divide(m0[-1] - m0, w1, out=np.zeros_like(m0), where=w1 > 0)
    between = w0 * w1 * (mu0 - mu1) ** 2
    return float(centers[int(np.argmax(between))])


def estimate_regime_labels(
    t: np.ndarray,
    nebar: np.ndarray,
    P_nbi: np.ndarray,
    D_alpha: np.ndarray,
    Ip: Optional[np.ndarray] = None,
    min_dwell_s: float = 0.02,
    transition_halfwidth_s: float = 0.003,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Label L/H regime from D-alpha with a dwell-time-constrained bimodal split.

    Method: smooth D-alpha over ~4 ms, split its distribution (within the
    plasma window, gated on Ip and nebar) with an Otsu threshold, and accept
    low-D-alpha (H-regime) segments only if they persist for >= min_dwell_s.
    Multiple segments are allowed, so H->L back-transitions are labelled too.

    Returns:
        regime: int8 array; 0 = unknown/no plasma, 1 = L, 2 = transition, 3 = H
        score: H-evidence in [0, 1] (distance of smoothed D-alpha below threshold)
        transition_time: time of the first L->H switch (nan if none)
    """
    regime = np.zeros_like(t, dtype=np.int8)
    score = np.zeros_like(t, dtype=float)
    if t.size < 5 or not np.any(np.isfinite(D_alpha)):
        return regime, score, float("nan")

    dalpha = interp_fill_1d(t, np.asarray(D_alpha, dtype=float))
    # Lower envelope of D-alpha: robust to ELM bursts, tracks the inter-ELM
    # baseline whose step-down is the actual L->H signature.
    dalpha_s = _rolling_quantile(dalpha, t, 0.015, 0.15)

    # Flat-top gate: only label where the discharge is established. This must
    # exclude the current ramp, where D-alpha is low simply because recycling
    # has not built up yet (not because of confinement).
    gate = np.ones_like(t, dtype=bool)
    if Ip is not None and np.any(np.isfinite(Ip)):
        ip_abs = np.abs(interp_fill_1d(t, np.asarray(Ip, dtype=float)))
        gate &= ip_abs > 0.6 * np.nanpercentile(ip_abs, 95)
    if np.any(np.isfinite(nebar)):
        ne_f = interp_fill_1d(t, np.asarray(nebar, dtype=float))
        gate &= ne_f > 0.2 * np.nanpercentile(ne_f, 95)
    if np.count_nonzero(gate) < 10:
        return regime, score, float("nan")

    gated = dalpha_s[gate]
    lo, hi = np.nanpercentile(gated, [1, 99])
    span = hi - lo
    if span < 1e-12:
        return regime, score, float("nan")
    norm_s = np.clip((dalpha_s - lo) / span, 0.0, 1.0)
    thr = _otsu_threshold(norm_s[gate])
    # Guards against fabricating a transition on unimodal data: the threshold
    # must sit inside the central range, both classes must be populated, and
    # the class means must be genuinely separated.
    def _all_L():
        regime[gate] = 1
        return regime, score, float("nan")

    if not (0.05 < thr < 0.95):
        return _all_L()
    low = norm_s[gate] < thr
    low_frac = float(np.mean(low))
    if low_frac < 0.05 or low_frac > 0.90:
        return _all_L()
    mean_low = float(np.mean(norm_s[gate][low]))
    mean_high = float(np.mean(norm_s[gate][~low]))
    if (mean_high - mean_low) < 0.20:
        return _all_L()

    score = np.clip((thr - norm_s) / max(thr, 1e-6), 0.0, 1.0)

    # H candidate = low D-alpha inside the gate; enforce minimum dwell time.
    # An H run must also be preceded by an L reference inside the gate: right
    # at gate opening D-alpha is still building up, so a low baseline there is
    # ramp physics, not confinement.
    h_cand = (norm_s < thr) & gate
    dt = float(np.median(np.diff(t)))
    min_dwell_n = max(2, int(round(min_dwell_s / max(dt, 1e-9))))
    gate_start = int(np.argmax(gate))
    min_L_lead_n = max(min_dwell_n, int(round(0.02 / max(dt, 1e-9))))
    sharp_n = max(2, int(round(0.010 / max(dt, 1e-9))))  # 10 ms contrast windows
    h_ok = np.zeros_like(h_cand)
    i = 0
    n = h_cand.size
    while i < n:
        if h_cand[i]:
            j = i
            while j < n and h_cand[j]:
                j += 1
            accept = (j - i) >= min_dwell_n and (i - gate_start) >= min_L_lead_n
            if accept:
                # Entry sharpness: the baseline must step DOWN across the run
                # entry. A slowly drifting baseline that happens to straddle
                # the Otsu threshold is not a confinement transition.
                pre = norm_s[max(gate_start, i - sharp_n) : i]
                post = norm_s[i : min(j, i + sharp_n)]
                accept = pre.size > 0 and post.size > 0 and (float(np.mean(pre)) - float(np.mean(post))) >= 0.12
            if accept:
                h_ok[i:j] = True
            i = j
        else:
            i += 1

    regime[gate] = 1
    regime[h_ok] = 3

    # Mark short transition windows around every regime switch inside the gate.
    half_n = max(1, int(round(transition_halfwidth_s / max(dt, 1e-9))))
    switches = np.where(np.diff(regime.astype(int)) != 0)[0]
    transition_time = float("nan")
    for s_idx in switches:
        if regime[s_idx] in (1, 3) and regime[s_idx + 1] in (1, 3):
            if np.isnan(transition_time) and regime[s_idx] == 1 and regime[s_idx + 1] == 3:
                transition_time = float(t[s_idx + 1])
            lo_i = max(0, s_idx - half_n + 1)
            hi_i = min(n, s_idx + 1 + half_n)
            regime[lo_i:hi_i] = 2

    return regime, score, transition_time


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
CTRL_KEYS = ["P_nbi", "Ip", "nebar", "D_alpha"]
STATE_KEYS = ["Te", "ne"]
MASK_KEYS = ["Te_mask", "ne_mask"]
QA_GRADE_ORDER = {"fail": 0, "caution": 1, "pass": 2}


@dataclass(frozen=True)
class TimeSliceQualityConfig:
    n_chan_min: int = 8
    delta_rho_min: float = 0.08
    mad_window: int = 11
    k_mad: float = 6.0


@dataclass(frozen=True)
class RhoSignalQualityConfig:
    enabled: bool = True
    hard_min_rho: float = 0.72
    low_rho_cutoff: float = 0.78
    outer_rho_min: float = 0.78
    min_col_coverage: float = 0.35
    low_rho_flat_span_eV: float = 25.0
    low_rho_flat_rel_span: float = 0.08
    min_te_median_eV: float = 50.0
    max_te_rel_jump_p95: float = 0.60
    max_te_rel_span: float = 1.80
    min_edge_cols: int = 4
    min_kept_cols: int = 4


@dataclass(frozen=True)
class RepresentativeScalarConfig:
    density_rho_min: float = 0.78
    density_rho_max: float = 0.98
    min_density_cols: int = 3


def _robust_span_and_median(values: np.ndarray) -> Tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    median = float(np.median(values))
    if values.size == 1:
        return 0.0, median, 0.0
    q10, q90 = np.nanpercentile(values, [10.0, 90.0])
    span = float(q90 - q10)
    rel_span = span / max(abs(median), 50.0)
    return span, median, rel_span


def build_rho_signal_quality(
    rho: np.ndarray,
    Te_rho_t: np.ndarray,
    Te_mask: np.ndarray,
    ne_rho_t: np.ndarray,
    ne_mask: np.ndarray,
    cfg: RhoSignalQualityConfig,
) -> Dict[str, object]:
    """Build a per-shot radial signal mask that drops weak rho channels, not shots."""
    rho = np.asarray(rho, dtype=float)
    Te_rho_t = np.asarray(Te_rho_t, dtype=float)
    ne_rho_t = np.asarray(ne_rho_t, dtype=float)
    Te_mask_bool = np.asarray(Te_mask, dtype=bool) & np.isfinite(Te_rho_t)
    ne_mask_bool = np.asarray(ne_mask, dtype=bool) & np.isfinite(ne_rho_t)

    te_cov = Te_mask_bool.mean(axis=0) if Te_mask_bool.size else np.zeros_like(rho)
    ne_cov = ne_mask_bool.mean(axis=0) if ne_mask_bool.size else np.zeros_like(rho)
    te_span = np.full_like(rho, np.nan, dtype=float)
    te_median = np.full_like(rho, np.nan, dtype=float)
    te_rel_span = np.full_like(rho, np.nan, dtype=float)
    te_rel_jump_p95 = np.full_like(rho, np.nan, dtype=float)

    for col_idx in range(rho.size):
        vals = Te_rho_t[:, col_idx][Te_mask_bool[:, col_idx]]
        span, median, rel_span = _robust_span_and_median(vals)
        te_span[col_idx] = span
        te_median[col_idx] = median
        te_rel_span[col_idx] = rel_span
        if vals.size >= 3:
            jump = np.abs(np.diff(vals)) / np.maximum(np.abs(vals[:-1]), 50.0)
            te_rel_jump_p95[col_idx] = float(np.nanpercentile(jump, 95.0)) if jump.size else 0.0
        elif vals.size > 0:
            te_rel_jump_p95[col_idx] = 0.0

    if not cfg.enabled:
        keep = te_cov > 0.0
        reason = np.where(keep, "kept", "no_te")
    else:
        has_te = te_cov >= float(cfg.min_col_coverage)
        hard_low = rho < float(cfg.hard_min_rho)
        low_rho = rho < float(cfg.low_rho_cutoff)
        flat_low = (
            low_rho
            & has_te
            & (
                (np.nan_to_num(te_span, nan=0.0) <= float(cfg.low_rho_flat_span_eV))
                | (np.nan_to_num(te_rel_span, nan=0.0) <= float(cfg.low_rho_flat_rel_span))
            )
        )
        low_median = has_te & (np.nan_to_num(te_median, nan=0.0) < float(cfg.min_te_median_eV))
        jumpy = has_te & (np.nan_to_num(te_rel_jump_p95, nan=np.inf) > float(cfg.max_te_rel_jump_p95))
        excessive_span = has_te & (np.nan_to_num(te_rel_span, nan=np.inf) > float(cfg.max_te_rel_span))
        keep = has_te & ~hard_low & ~flat_low & ~low_median & ~jumpy & ~excessive_span

        min_kept = max(0, int(cfg.min_kept_cols))
        if min_kept > 0 and int(np.count_nonzero(keep)) < min_kept:
            candidates = np.flatnonzero((rho >= float(cfg.outer_rho_min)) & has_te & ~low_median & ~jumpy & ~excessive_span)
            if candidates.size > 0:
                score = rho[candidates] + 0.25 * te_cov[candidates] - 0.5 * np.nan_to_num(te_rel_jump_p95[candidates], nan=0.0)
                chosen = candidates[np.argsort(score)[-min(min_kept, candidates.size) :]]
                keep[chosen] = True

        edge_ok = keep & (rho >= float(cfg.outer_rho_min))
        enough_edge = int(np.count_nonzero(edge_ok)) >= int(cfg.min_edge_cols)
        if not enough_edge:
            keep[:] = False

        reason = np.full(rho.shape, "kept", dtype="U24")
        reason[~has_te] = "low_coverage"
        reason[hard_low] = "low_rho_hard_drop"
        reason[flat_low] = "low_rho_flat"
        reason[low_median] = "low_te_median"
        reason[jumpy] = "jumpy_te"
        reason[excessive_span] = "excessive_te_span"
        if not enough_edge:
            reason[has_te] = "too_few_stable_edge_cols"
        reason[keep] = "kept"

    return {
        "keep": keep.astype(bool),
        "reason": reason.astype("U24"),
        "Te_col_coverage": te_cov.astype(np.float32),
        "ne_col_coverage": ne_cov.astype(np.float32),
        "Te_col_span_eV": te_span.astype(np.float32),
        "Te_col_median_eV": te_median.astype(np.float32),
        "Te_col_rel_span": te_rel_span.astype(np.float32),
        "Te_col_rel_jump_p95": te_rel_jump_p95.astype(np.float32),
        "n_kept": int(np.count_nonzero(keep)),
        "n_dropped": int(keep.size - np.count_nonzero(keep)),
        "first_kept_rho": float(rho[keep][0]) if np.any(keep) else float("nan"),
        "last_kept_rho": float(rho[keep][-1]) if np.any(keep) else float("nan"),
        "edge_cols_kept": int(np.count_nonzero(keep & (rho >= float(cfg.outer_rho_min)))),
        "low_rho_cols_dropped": int(np.count_nonzero((~keep) & (rho < float(cfg.low_rho_cutoff)) & (te_cov > 0.0))),
    }


def _row_trimmed_mean(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if values.size < 4:
        return float(np.median(values))
    lo, hi = np.nanpercentile(values, [20.0, 80.0])
    trimmed = values[(values >= lo) & (values <= hi)]
    if trimmed.size == 0:
        return float(np.median(values))
    return float(np.mean(trimmed))


def build_representative_density_scalar(
    t_ts: np.ndarray,
    rho: np.ndarray,
    ne_rho_t: np.ndarray,
    ne_mask: np.ndarray,
    Te_mask: np.ndarray,
    rho_keep: np.ndarray,
    cfg: RepresentativeScalarConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Collapse ne(rho,t) to one robust scalar per time using trusted overlapping rho support."""
    t_ts = np.asarray(t_ts, dtype=float)
    rho = np.asarray(rho, dtype=float)
    ne_rho_t = np.asarray(ne_rho_t, dtype=float)
    valid = np.asarray(ne_mask, dtype=bool) & np.asarray(Te_mask, dtype=bool) & np.isfinite(ne_rho_t)
    valid &= (ne_rho_t >= 1.0e17) & (ne_rho_t <= 1.0e21)
    preferred_cols = (
        np.asarray(rho_keep, dtype=bool)
        & (rho >= float(cfg.density_rho_min))
        & (rho <= float(cfg.density_rho_max))
    )
    if int(np.count_nonzero(preferred_cols)) < int(cfg.min_density_cols):
        preferred_cols = np.asarray(rho_keep, dtype=bool) & (rho >= float(cfg.density_rho_min))
    if int(np.count_nonzero(preferred_cols)) < int(cfg.min_density_cols):
        preferred_cols = np.asarray(rho_keep, dtype=bool)

    out = np.full((ne_rho_t.shape[0],), np.nan, dtype=float)
    mask = np.zeros_like(out, dtype=bool)
    for idx in range(ne_rho_t.shape[0]):
        row_valid = valid[idx] & preferred_cols
        if int(np.count_nonzero(row_valid)) < int(cfg.min_density_cols):
            row_valid = valid[idx] & np.asarray(rho_keep, dtype=bool)
        if int(np.count_nonzero(row_valid)) == 0:
            continue
        out[idx] = _row_trimmed_mean(ne_rho_t[idx, row_valid])
        mask[idx] = np.isfinite(out[idx])

    if np.any(mask):
        out[~mask] = np.interp(t_ts[~mask], t_ts[mask], out[mask], left=out[mask][0], right=out[mask][-1])
    else:
        out[:] = 0.0

    meta = {
        "ne_profile_scalar_valid_fraction": float(np.mean(mask)) if mask.size else 0.0,
        "ne_profile_scalar_rho_min": float(np.min(rho[preferred_cols])) if np.any(preferred_cols) else float("nan"),
        "ne_profile_scalar_rho_max": float(np.max(rho[preferred_cols])) if np.any(preferred_cols) else float("nan"),
        "ne_profile_scalar_cols": int(np.count_nonzero(preferred_cols)),
    }
    return out.astype(np.float64), mask.astype(bool), meta


def sanitize_nonnegative_signal(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), 0.0)


def profiles_to_rho_grid_timevarying(
    rho_t_s: np.ndarray,
    values_t_s: np.ndarray,
    rho_dst: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate time-varying channel coordinates onto a fixed rho grid."""
    Nt = values_t_s.shape[0]
    out = np.full((Nt, rho_dst.size), np.nan, dtype=float)
    out_mask = np.zeros_like(out)
    for t_idx in range(Nt):
        rho_src = np.asarray(rho_t_s[t_idx], dtype=float)
        values = np.asarray(values_t_s[t_idx], dtype=float)
        valid = np.isfinite(rho_src) & np.isfinite(values)
        if np.count_nonzero(valid) == 0:
            continue
        rs = rho_src[valid]
        vs = values[valid]
        order = np.argsort(rs)
        rs = rs[order]
        vs = vs[order]
        rs, uniq_idx = np.unique(rs, return_index=True)
        vs = vs[uniq_idx]
        out[t_idx] = np.interp(rho_dst, rs, vs, left=np.nan, right=np.nan)
        out_mask[t_idx] = np.isfinite(out[t_idx]).astype(float)
    return out, out_mask


def _row_rho_span(rho_t_s: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    span = np.zeros((valid_mask.shape[0],), dtype=float)
    for idx in range(valid_mask.shape[0]):
        row_valid = valid_mask[idx]
        if np.count_nonzero(row_valid) == 0:
            continue
        rho_row = np.asarray(rho_t_s[idx, row_valid], dtype=float)
        span[idx] = float(np.nanmax(rho_row) - np.nanmin(rho_row)) if rho_row.size > 0 else 0.0
    return span


def _rolling_mad_spike_mask(values_t_s: np.ndarray, window: int, k_mad: float) -> np.ndarray:
    """Return a boolean mask of time-local outliers for each Thomson channel."""
    values = np.asarray(values_t_s, dtype=float)
    bad = np.zeros_like(values, dtype=bool)
    half = max(int(window) // 2, 1)
    for chan_idx in range(values.shape[1]):
        series = values[:, chan_idx]
        finite_idx = np.flatnonzero(np.isfinite(series))
        if finite_idx.size < 3:
            continue
        for time_idx in finite_idx:
            lo = max(0, time_idx - half)
            hi = min(values.shape[0], time_idx + half + 1)
            window_vals = series[lo:hi]
            window_vals = window_vals[np.isfinite(window_vals)]
            if window_vals.size < 3:
                continue
            median = float(np.median(window_vals))
            mad = 1.4826 * float(np.median(np.abs(window_vals - median)))
            scale = max(mad, 1.0e-6 * max(abs(median), 1.0))
            if abs(float(series[time_idx]) - median) > float(k_mad) * scale:
                bad[time_idx, chan_idx] = True
    return bad


def _outermost_valid_trace(rho_t_s: np.ndarray, values_t_s: np.ndarray) -> np.ndarray:
    out = np.full((values_t_s.shape[0],), np.nan, dtype=float)
    for idx in range(values_t_s.shape[0]):
        valid = np.isfinite(rho_t_s[idx]) & np.isfinite(values_t_s[idx])
        if np.count_nonzero(valid) == 0:
            continue
        rho_row = rho_t_s[idx, valid]
        values_row = values_t_s[idx, valid]
        outer_idx = int(np.argmax(rho_row))
        out[idx] = float(values_row[outer_idx])
    return out


def run_shot_quality_screen(
    t_ts: np.ndarray,
    rho_t_s: np.ndarray,
    Te_t_s: np.ndarray,
    ne_t_s: np.ndarray,
    n_chan_min: int = 8,
    delta_rho_min: float = 0.08,
    mad_window: int = 11,
    k_mad: float = 6.0,
    te_bounds: Tuple[float, float] = (5.0, 5000.0),
    ne_bounds: Tuple[float, float] = (1.0e17, 1.0e21),
) -> Dict[str, object]:
    """Apply the v3 Thomson QA screen and return filtered arrays plus grade metrics."""
    t_ts = np.asarray(t_ts, dtype=float)
    rho_t_s = np.asarray(rho_t_s, dtype=float)
    Te_raw = np.asarray(Te_t_s, dtype=float)
    ne_raw = np.asarray(ne_t_s, dtype=float)

    raw_valid_te = np.isfinite(rho_t_s) & np.isfinite(Te_raw)
    raw_n_chan = np.sum(raw_valid_te, axis=1)
    raw_rho_span = _row_rho_span(rho_t_s, raw_valid_te)
    raw_keep = (raw_n_chan >= int(n_chan_min)) & (raw_rho_span >= float(delta_rho_min))

    Te_filtered = np.array(Te_raw, copy=True)
    ne_filtered = np.array(ne_raw, copy=True)

    te_bad_mad = _rolling_mad_spike_mask(Te_filtered, window=mad_window, k_mad=k_mad)
    ne_bad_mad = _rolling_mad_spike_mask(ne_filtered, window=mad_window, k_mad=k_mad)
    mad_removed_count = int(np.count_nonzero(te_bad_mad) + np.count_nonzero(ne_bad_mad))
    Te_filtered[te_bad_mad] = np.nan
    ne_filtered[ne_bad_mad] = np.nan

    te_bad_bounds = np.isfinite(Te_filtered) & ((Te_filtered < float(te_bounds[0])) | (Te_filtered > float(te_bounds[1])))
    ne_bad_bounds = np.isfinite(ne_filtered) & ((ne_filtered < float(ne_bounds[0])) | (ne_filtered > float(ne_bounds[1])))
    hard_bounds_count = int(np.count_nonzero(te_bad_bounds) + np.count_nonzero(ne_bad_bounds))
    Te_filtered[te_bad_bounds] = np.nan
    ne_filtered[ne_bad_bounds] = np.nan

    filtered_valid_te = np.isfinite(rho_t_s) & np.isfinite(Te_filtered)
    filtered_n_chan = np.sum(filtered_valid_te, axis=1)
    filtered_rho_span = _row_rho_span(rho_t_s, filtered_valid_te)
    keep_time = raw_keep & (filtered_n_chan >= int(n_chan_min)) & (filtered_rho_span >= float(delta_rho_min))

    total_samples = int(np.count_nonzero(np.isfinite(Te_raw)) + np.count_nonzero(np.isfinite(ne_raw)))
    mad_removed_fraction = float(mad_removed_count / total_samples) if total_samples > 0 else 0.0
    kept_fraction = float(np.mean(keep_time)) if keep_time.size > 0 else 0.0

    if kept_fraction >= 0.80 and mad_removed_fraction <= 0.005 and hard_bounds_count == 0:
        grade = "pass"
    elif kept_fraction >= 0.60 and mad_removed_fraction <= 0.02 and hard_bounds_count <= 1:
        grade = "caution"
    else:
        grade = "fail"

    return {
        "grade": grade,
        "n_chan_min": int(n_chan_min),
        "delta_rho_min": float(delta_rho_min),
        "t_ts_raw": t_ts,
        "t_ts": t_ts[keep_time],
        "rho_t_s": rho_t_s[keep_time],
        "Te_t_s": Te_filtered[keep_time],
        "ne_t_s": ne_filtered[keep_time],
        "n_time_total": int(t_ts.size),
        "n_time_kept": int(np.count_nonzero(keep_time)),
        "kept_fraction": kept_fraction,
        "mad_removed_fraction": mad_removed_fraction,
        "mad_removed_count": mad_removed_count,
        "hard_bounds_count": hard_bounds_count,
        "raw_n_chan_min": int(np.min(raw_n_chan)) if raw_n_chan.size > 0 else 0,
        "raw_n_chan_median": float(np.median(raw_n_chan)) if raw_n_chan.size > 0 else 0.0,
        "filtered_n_chan_median": float(np.median(filtered_n_chan[keep_time])) if np.any(keep_time) else 0.0,
        "raw_keep": raw_keep,
        "keep_time": keep_time,
        "raw_n_chan": raw_n_chan,
        "filtered_n_chan": filtered_n_chan,
        "raw_te_edge": _outermost_valid_trace(rho_t_s, Te_raw),
        "filtered_te_edge": _outermost_valid_trace(rho_t_s, Te_filtered),
        "te_bad_counts_per_time": np.sum(te_bad_mad, axis=1),
        "ne_bad_counts_per_time": np.sum(ne_bad_mad, axis=1),
    }


def qa_grade_allows(grade: str, min_grade: str) -> bool:
    return QA_GRADE_ORDER[str(grade)] >= QA_GRADE_ORDER[str(min_grade)]


def write_qa_plot(shot: int, qa: Dict[str, object], plot_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    ts = np.asarray(qa["t_ts_raw"], dtype=float)
    raw_keep = np.asarray(qa["raw_keep"], dtype=bool)
    keep_time = np.asarray(qa["keep_time"], dtype=bool)
    raw_n_chan = np.asarray(qa["raw_n_chan"], dtype=float)
    filtered_n_chan = np.asarray(qa["filtered_n_chan"], dtype=float)
    raw_te_edge = np.asarray(qa["raw_te_edge"], dtype=float)
    filtered_te_edge = np.asarray(qa["filtered_te_edge"], dtype=float)
    te_bad_counts = np.asarray(qa["te_bad_counts_per_time"], dtype=float)
    ne_bad_counts = np.asarray(qa["ne_bad_counts_per_time"], dtype=float)

    has_rho_quality = "rho_quality_rho" in qa
    nrows = 4 if has_rho_quality else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(11, 11 if has_rho_quality else 9), sharex=False)
    axes[0].plot(ts, raw_n_chan, label="raw valid Te channels", color="tab:blue")
    axes[0].plot(ts, filtered_n_chan, label="post-filter valid Te channels", color="tab:orange")
    axes[0].axhline(float(qa.get("n_chan_min", 8.0)), color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    axes[0].set_ylabel("channels")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(ts, raw_te_edge, label="raw outermost Te", color="tab:green")
    axes[1].plot(ts, filtered_te_edge, label="filtered outermost Te", color="tab:red")
    axes[1].set_ylabel("outermost Te [eV]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    axes[2].plot(ts, te_bad_counts, label="Te MAD removals", color="tab:purple")
    axes[2].plot(ts, ne_bad_counts, label="ne MAD removals", color="tab:brown")
    axes[2].fill_between(ts, 0.0, 1.0, where=raw_keep, transform=axes[2].get_xaxis_transform(), color="tab:blue", alpha=0.10)
    axes[2].fill_between(ts, 0.0, 1.0, where=keep_time, transform=axes[2].get_xaxis_transform(), color="tab:red", alpha=0.10)
    axes[2].set_ylabel("removed samples")
    axes[2].set_xlabel("time [s]")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best")

    if has_rho_quality:
        rho = np.asarray(qa["rho_quality_rho"], dtype=float)
        keep = np.asarray(qa["rho_quality_keep"], dtype=bool)
        te_cov = np.asarray(qa["rho_quality_Te_cov"], dtype=float)
        te_span = np.asarray(qa["rho_quality_Te_span_eV"], dtype=float)
        ax = axes[3]
        ax.plot(rho, te_cov, color="tab:blue", marker="o", markersize=3.0, label="Te coverage")
        ax.fill_between(rho, 0.0, 1.0, where=keep, color="tab:green", alpha=0.12, label="kept rho")
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel("coverage")
        ax.set_xlabel("rho")
        ax.grid(True, alpha=0.25)
        ax2 = ax.twinx()
        ax2.plot(rho, te_span, color="tab:orange", linewidth=1.1, label="Te span")
        ax2.set_ylabel("Te span [eV]")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best")

    fig.suptitle(
        f"Shot {shot} QA: grade={qa['grade']} kept={qa['n_time_kept']}/{qa['n_time_total']} "
        f"mad_removed={qa['mad_removed_fraction']:.3%} hard_bounds={qa['hard_bounds_count']}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(plot_dir, f"{shot}_qa.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_qa_summary_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "shot",
        "qa_grade",
        "written",
        "n_time_total",
        "n_time_kept",
        "kept_fraction",
        "mad_removed_fraction",
        "mad_removed_count",
        "hard_bounds_count",
        "time_min_channels",
        "time_min_rho_span",
        "raw_n_chan_min",
        "raw_n_chan_median",
        "filtered_n_chan_median",
        "rho_cols_kept",
        "rho_cols_dropped",
        "rho_first_kept",
        "rho_edge_cols_kept",
        "rho_low_cols_dropped",
        "rho_fallback_used",
        "ne_profile_scalar_valid_fraction",
        "Te_mask_cov",
        "ne_mask_cov",
        "path",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


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

    for extra in ["P_rad", "W_tot", "P_ohm", "P_tot", "H98", "q95", "li", "beta_n", "B_t0", "regime_score", "transition_time"]:
        if extra in d:
            report.append("  " + _fmt_1d(extra, d[extra]))

    def _is_mono(x): # Check if array is monotonically non-decreasing 
        return np.all(np.diff(x) >= -1e-9)
    report.append(f"t mono={_is_mono(t)}, t_ts mono={_is_mono(t_ts)}")

    Te_cov = float(d["Te_mask"].mean()) if "Te_mask" in d else float("nan")
    ne_cov = float(d["ne_mask"].mean()) if "ne_mask" in d else float("nan")
    shot = int(os.path.basename(path).split("_")[0])
    return "\n".join(report), (shot, rho_fb, Te_cov, ne_cov)


def build_one_shot(
    shot: int,
    data_root: str = "data",
    Nrho: int = 65,
    qa_min_grade: str = "fail",
    allow_degraded: bool = False,
    qa_plots_dir: Optional[str] = None,
    time_quality_cfg: Optional[TimeSliceQualityConfig] = None,
    rho_quality_cfg: Optional[RhoSignalQualityConfig] = None,
    scalar_cfg: Optional[RepresentativeScalarConfig] = None,
) -> Tuple[Optional[str], Dict[str, object]]:
    if time_quality_cfg is None:
        time_quality_cfg = TimeSliceQualityConfig()
    if rho_quality_cfg is None:
        rho_quality_cfg = RhoSignalQualityConfig()
    if scalar_cfg is None:
        scalar_cfg = RepresentativeScalarConfig()

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
            rho_t_s = np.broadcast_to(np.asarray(rho_ts, dtype=float)[None, :], Te_ts.shape).copy()

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
            rho_t_s = np.vstack([rho_fn(R_t_s[t_idx], Z_t_s[t_idx]) for t_idx in range(Nt)])

        t_ts = ts["time"].values if "time" in ts.coords else np.arange(Nt)

    qa = run_shot_quality_screen(
        t_ts,
        rho_t_s,
        Te_ts if rho_coord_name is not None and rho_coord_name in ts.coords else Te_t_s,
        ne_ts if rho_coord_name is not None and rho_coord_name in ts.coords else ne_t_s,
        n_chan_min=int(time_quality_cfg.n_chan_min),
        delta_rho_min=float(time_quality_cfg.delta_rho_min),
        mad_window=int(time_quality_cfg.mad_window),
        k_mad=float(time_quality_cfg.k_mad),
    )
    qa_summary: Dict[str, object] = {
        "shot": int(shot),
        "qa_grade": str(qa["grade"]),
        "written": False,
        "n_time_total": int(qa["n_time_total"]),
        "n_time_kept": int(qa["n_time_kept"]),
        "kept_fraction": float(qa["kept_fraction"]),
        "mad_removed_fraction": float(qa["mad_removed_fraction"]),
        "mad_removed_count": int(qa["mad_removed_count"]),
        "hard_bounds_count": int(qa["hard_bounds_count"]),
        "time_min_channels": int(time_quality_cfg.n_chan_min),
        "time_min_rho_span": float(time_quality_cfg.delta_rho_min),
        "raw_n_chan_min": int(qa["raw_n_chan_min"]),
        "raw_n_chan_median": float(qa["raw_n_chan_median"]),
        "filtered_n_chan_median": float(qa["filtered_n_chan_median"]),
        "rho_cols_kept": 0,
        "rho_cols_dropped": 0,
        "rho_first_kept": float("nan"),
        "rho_edge_cols_kept": 0,
        "rho_low_cols_dropped": 0,
        "rho_fallback_used": bool(rho_fallback_used),
        "ne_profile_scalar_valid_fraction": float("nan"),
        "Te_mask_cov": float("nan"),
        "ne_mask_cov": float("nan"),
        "path": "",
    }

    out_npz = os.path.join(data_root, f"{shot}_torax_training.npz")
    if not qa_grade_allows(str(qa["grade"]), str(qa_min_grade)) and not allow_degraded:
        if os.path.exists(out_npz):
            os.remove(out_npz)
            print(f"  .. removed stale pack for shot {shot}: {out_npz}")
        print(
            f"  .. skipping pack write for shot {shot}: grade={qa['grade']} < min_grade={qa_min_grade} "
            f"(use --allow-degraded to override)"
        )
        return None, qa_summary

    t_ts = np.asarray(qa["t_ts"], dtype=float)
    rho_t_s = np.asarray(qa["rho_t_s"], dtype=float)
    Te_filtered_t_s = np.asarray(qa["Te_t_s"], dtype=float)
    ne_filtered_t_s = np.asarray(qa["ne_t_s"], dtype=float)
    if t_ts.size == 0:
        raise RuntimeError(f"Shot {shot} has no Thomson slices left after QA screening")

    Te_rho_t, Te_mask = profiles_to_rho_grid_timevarying(rho_t_s, Te_filtered_t_s, rho_torax)
    ne_rho_t, ne_mask = profiles_to_rho_grid_timevarying(rho_t_s, ne_filtered_t_s, rho_torax)

    # Extract units from NetCDF metadata
    Te_units = str(Te_da.attrs.get("units", "")) if Te_da is not None else ""
    ne_units = str(ne_da.attrs.get("units", "")) if ne_da is not None else ""

    # Combine upstream TS masks with finite checks (keep as bool)
    Te_mask = (Te_mask > 0.5) & np.isfinite(Te_rho_t)
    ne_mask = (ne_mask > 0.5) & np.isfinite(ne_rho_t)

    rho_quality = build_rho_signal_quality(rho_torax, Te_rho_t, Te_mask, ne_rho_t, ne_mask, rho_quality_cfg)
    rho_keep = np.asarray(rho_quality["keep"], dtype=bool)
    Te_mask = Te_mask & rho_keep[None, :]
    ne_mask = ne_mask & rho_keep[None, :]
    Te_rho_t = np.where(Te_mask, Te_rho_t, np.nan)
    ne_rho_t = np.where(ne_mask, ne_rho_t, np.nan)

    qa_summary["rho_cols_kept"] = int(rho_quality["n_kept"])
    qa_summary["rho_cols_dropped"] = int(rho_quality["n_dropped"])
    qa_summary["rho_first_kept"] = float(rho_quality["first_kept_rho"])
    qa_summary["rho_edge_cols_kept"] = int(rho_quality["edge_cols_kept"])
    qa_summary["rho_low_cols_dropped"] = int(rho_quality["low_rho_cols_dropped"])

    qa.update(
        {
            "rho_quality_rho": rho_torax,
            "rho_quality_keep": rho_keep,
            "rho_quality_Te_cov": rho_quality["Te_col_coverage"],
            "rho_quality_Te_span_eV": rho_quality["Te_col_span_eV"],
        }
    )
    if qa_plots_dir is not None:
        write_qa_plot(shot, qa, qa_plots_dir)

    if int(rho_quality["n_kept"]) <= 0:
        if os.path.exists(out_npz):
            os.remove(out_npz)
        print(f"  .. skipping pack write for shot {shot}: no stable rho columns survived strict QA")
        return None, qa_summary

    # Summary signals
    t = get_var(summ, ["time"]).values
    Ip_da = get_var(summ, ["ip", "Ip"])
    Ip = Ip_da.values if Ip_da is not None else np.full_like(t, np.nan)
    nebar_da = get_var(summ, ["line_average_n_e", "ne_bar", "nebar"])  # m^-3 typically
    nebar_summary_raw = nebar_da.values if nebar_da is not None else np.full_like(t, np.nan)
    P_nbi_da = get_var(summ, ["power_nbi", "P_NBI", "pnbi"])
    P_rad_da = get_var(summ, ["power_radiated", "P_rad", "prad"])
    P_nbi = P_nbi_da.values if P_nbi_da is not None else np.full_like(t, np.nan)
    P_rad = P_rad_da.values if P_rad_da is not None else np.full_like(t, np.nan)

    # Preserve raw (pre-filled) copies
    P_nbi_raw = np.array(P_nbi, copy=True)
    P_rad_raw = np.array(P_rad, copy=True)

    # Fill NaNs in key control signals to avoid gaps downstream
    Ip_signed_raw = np.array(Ip, copy=True)
    Ip = np.abs(interp_fill_1d(t, Ip))
    nebar_raw = interp_fill_1d(t, nebar_summary_raw)
    P_nbi = sanitize_nonnegative_signal(interp_fill_1d(t, P_nbi))
    P_rad = sanitize_nonnegative_signal(interp_fill_1d(t, P_rad))

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

    # D-alpha diagnostic and L-H transition signal.
    dalpha_path = os.path.join(shot_dir, "d_alpha.nc")
    spec_path = os.path.join(shot_dir, "spectrometer_visible.nc")

    D_alpha = np.zeros_like(t)
    D_alpha_channels = np.zeros((t.size, 0), dtype=float)
    D_alpha_channel_names = np.array([], dtype="U64")

    dalpha_source_path = find_first_existing([dalpha_path, spec_path])
    if dalpha_source_path is not None:
        try:
            dalpha_ds = xr.load_dataset(dalpha_source_path)
            t_dalpha, dalpha_channels_native, D_alpha_channel_names, _dalpha_sum_native = extract_dalpha_arrays(dalpha_ds)
            D_alpha_channels = interp_channels_to_time(t_dalpha, dalpha_channels_native, t)
            D_alpha_channels = sanitize_nonnegative_signal(D_alpha_channels)
            D_alpha = sanitize_nonnegative_signal(interp_fill_1d(t, np.sum(D_alpha_channels, axis=1)))
        except Exception as e:
            print(f"  !! Failed to load D-alpha for shot {shot}: {e}")

    # Extract time arrays
    t_summary = get_var(summ, ["time"]).values

    ne_profile_scalar_ts, ne_profile_scalar_mask_ts, ne_scalar_meta = build_representative_density_scalar(
        t_ts,
        rho_torax,
        ne_rho_t,
        ne_mask,
        Te_mask,
        rho_keep,
        scalar_cfg,
    )
    nebar = np.interp(t_summary, t_ts, ne_profile_scalar_ts, left=ne_profile_scalar_ts[0], right=ne_profile_scalar_ts[-1])
    qa_summary["ne_profile_scalar_valid_fraction"] = float(ne_scalar_meta["ne_profile_scalar_valid_fraction"])

    # Simple regime labelling on summary grid (t_summary)
    # 0 = unknown, 1 = L-mode, 2 = transition, 3 = H-mode
    regime, regime_score, transition_time = estimate_regime_labels(t_summary, nebar, P_nbi, D_alpha, Ip=Ip)

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
        rho_signal_keep_mask=rho_keep,
        rho_signal_drop_reason=np.asarray(rho_quality["reason"], dtype="U24"),
        rho_signal_Te_col_coverage=rho_quality["Te_col_coverage"],
        rho_signal_ne_col_coverage=rho_quality["ne_col_coverage"],
        rho_signal_Te_span_eV=rho_quality["Te_col_span_eV"],
        rho_signal_Te_median_eV=rho_quality["Te_col_median_eV"],
        rho_signal_Te_rel_span=rho_quality["Te_col_rel_span"],
        rho_signal_Te_rel_jump_p95=rho_quality["Te_col_rel_jump_p95"],
        rho_quality_hard_min_rho=float(rho_quality_cfg.hard_min_rho),
        rho_quality_low_rho_cutoff=float(rho_quality_cfg.low_rho_cutoff),
        rho_quality_outer_rho_min=float(rho_quality_cfg.outer_rho_min),
        rho_quality_min_col_coverage=float(rho_quality_cfg.min_col_coverage),
        rho_quality_low_rho_flat_span_eV=float(rho_quality_cfg.low_rho_flat_span_eV),
        rho_quality_low_rho_flat_rel_span=float(rho_quality_cfg.low_rho_flat_rel_span),
        rho_quality_min_te_median_eV=float(rho_quality_cfg.min_te_median_eV),
        rho_quality_max_te_rel_jump_p95=float(rho_quality_cfg.max_te_rel_jump_p95),
        rho_quality_max_te_rel_span=float(rho_quality_cfg.max_te_rel_span),
        Ip=Ip,
        Ip_signed_raw=Ip_signed_raw,
        nebar=nebar,
        nebar_raw=nebar_raw,
        ne_profile_scalar=nebar,
        ne_profile_scalar_ts=ne_profile_scalar_ts,
        ne_profile_scalar_mask_ts=ne_profile_scalar_mask_ts,
        **ne_scalar_meta,
        P_nbi=P_nbi,
        P_rad=P_rad,
        P_nbi_raw=P_nbi_raw,
        P_rad_raw=P_rad_raw,
        D_alpha=D_alpha,
        D_alpha_channels=D_alpha_channels,
        D_alpha_channel_names=D_alpha_channel_names,
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
        qa_grade=np.array(str(qa["grade"]), dtype="U16"),
        qa_kept_fraction=float(qa["kept_fraction"]),
        qa_mad_removed_fraction=float(qa["mad_removed_fraction"]),
        qa_mad_removed_count=int(qa["mad_removed_count"]),
        qa_hard_bounds_count=int(qa["hard_bounds_count"]),
        qa_n_time_total=int(qa["n_time_total"]),
        qa_n_time_kept=int(qa["n_time_kept"]),
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

    qa_summary["written"] = True
    qa_summary["Te_mask_cov"] = float(Te_mask_mean)
    qa_summary["ne_mask_cov"] = float(ne_mask_mean)
    qa_summary["path"] = out_npz
    return out_npz, qa_summary


def main():
    ap = argparse.ArgumentParser(description="Build TORAX training packs from NetCDFs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--shot", type=int, help="Single shot number to process")
    g.add_argument("--shots", type=int, nargs="+", help="List of shots")
    g.add_argument("--discover", action="store_true", help="Discover shots under data/<shot>")
    ap.add_argument("--Nrho", type=int, default=65, help="Number of rho grid points (default: 65)")
    ap.add_argument("--qa-grade", choices=["fail", "caution", "pass"], default="fail", help="Minimum QA grade required to write a pack")
    ap.add_argument("--allow-degraded", action="store_true", help="Force-write shots below --qa-grade if enough data remains after QA")
    ap.add_argument("--qa-summary", default=None, help="Optional CSV path for shot-level QA summary")
    ap.add_argument("--qa-plots", default=None, help="Optional directory for per-shot QA plots")
    ap.add_argument("--time-min-channels", type=int, default=8, help="Minimum finite Thomson channels required to keep a time slice")
    ap.add_argument("--time-min-rho-span", type=float, default=0.08, help="Minimum per-slice rho span required before per-rho masking")
    ap.add_argument("--qa-mad-window", type=int, default=11, help="Rolling window for Thomson MAD spike filtering")
    ap.add_argument("--qa-k-mad", type=float, default=6.0, help="MAD multiplier for Thomson spike filtering")
    ap.add_argument("--disable-rho-quality", action="store_true", help="Disable per-rho signal quality masking")
    ap.add_argument("--rho-hard-min", type=float, default=0.72, help="Always drop observed rho columns below this value")
    ap.add_argument("--rho-low-cutoff", type=float, default=0.78, help="Treat columns below this rho as low-rho candidates for flat-signal removal")
    ap.add_argument("--rho-outer-min", type=float, default=0.78, help="Outer rho threshold used for edge-focused fallback/diagnostics")
    ap.add_argument("--rho-min-col-coverage", type=float, default=0.35, help="Minimum per-column Te coverage to keep a rho signal")
    ap.add_argument("--rho-flat-span-ev", type=float, default=25.0, help="Low-rho robust Te span threshold for flat-signal removal")
    ap.add_argument("--rho-flat-rel-span", type=float, default=0.08, help="Low-rho relative Te span threshold for flat-signal removal")
    ap.add_argument("--rho-min-te-median", type=float, default=50.0, help="Drop rho columns whose robust Te median is below this value")
    ap.add_argument("--rho-max-te-rel-jump-p95", type=float, default=0.60, help="Drop rho columns with excessive 95th percentile relative Te jumps")
    ap.add_argument("--rho-max-te-rel-span", type=float, default=1.80, help="Drop rho columns with excessive robust relative Te span")
    ap.add_argument("--rho-min-edge-cols", type=int, default=4, help="Minimum stable edge rho columns required to write a pack")
    ap.add_argument("--rho-min-kept-cols", type=int, default=4, help="Minimum stable rho columns to keep per written shot")
    ap.add_argument("--density-rho-min", type=float, default=0.78, help="Lower rho bound for representative profile-density scalar")
    ap.add_argument("--density-rho-max", type=float, default=0.98, help="Upper rho bound for representative profile-density scalar")
    ap.add_argument("--density-min-cols", type=int, default=3, help="Minimum overlapping rho columns for representative profile-density scalar")
    args = ap.parse_args()

    time_quality_cfg = TimeSliceQualityConfig(
        n_chan_min=int(args.time_min_channels),
        delta_rho_min=float(args.time_min_rho_span),
        mad_window=int(args.qa_mad_window),
        k_mad=float(args.qa_k_mad),
    )

    rho_quality_cfg = RhoSignalQualityConfig(
        enabled=not bool(args.disable_rho_quality),
        hard_min_rho=float(args.rho_hard_min),
        low_rho_cutoff=float(args.rho_low_cutoff),
        outer_rho_min=float(args.rho_outer_min),
        min_col_coverage=float(args.rho_min_col_coverage),
        low_rho_flat_span_eV=float(args.rho_flat_span_ev),
        low_rho_flat_rel_span=float(args.rho_flat_rel_span),
        min_te_median_eV=float(args.rho_min_te_median),
        max_te_rel_jump_p95=float(args.rho_max_te_rel_jump_p95),
        max_te_rel_span=float(args.rho_max_te_rel_span),
        min_edge_cols=int(args.rho_min_edge_cols),
        min_kept_cols=int(args.rho_min_kept_cols),
    )

    scalar_cfg = RepresentativeScalarConfig(
        density_rho_min=float(args.density_rho_min),
        density_rho_max=float(args.density_rho_max),
        min_density_cols=int(args.density_min_cols),
    )

    if args.discover:
        shots = find_shots_in_data("data")
    elif args.shot is not None:
        shots = [args.shot]
    else:
        shots = args.shots

    created: List[str] = []
    summary_rows: List[Tuple[int, bool, float, float]] = []
    qa_rows: List[Dict[str, object]] = []
    for shot in shots:
        print(f"[build] Shot {shot}")
        try:
            path, qa_summary = build_one_shot(
                shot,
                data_root="data",
                Nrho=args.Nrho,
                qa_min_grade=args.qa_grade,
                allow_degraded=bool(args.allow_degraded),
                qa_plots_dir=args.qa_plots,
                time_quality_cfg=time_quality_cfg,
                rho_quality_cfg=rho_quality_cfg,
                scalar_cfg=scalar_cfg,
            )
            qa_rows.append(qa_summary)
            if path is None:
                continue
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

    if args.qa_summary is not None:
        write_qa_summary_csv(args.qa_summary, qa_rows)
        print(f"[build] Wrote QA summary: {args.qa_summary}")

    if qa_rows:
        print("=== QA summary ===")
        print("shot grade written kept/total mad_removed hard_bounds")
        for row in sorted(qa_rows, key=lambda item: int(item["shot"])):
            print(
                f"{int(row['shot'])} {row['qa_grade']} {bool(row['written'])} "
                f"{int(row['n_time_kept'])}/{int(row['n_time_total'])} "
                f"{float(row['mad_removed_fraction']):.3%} {int(row['hard_bounds_count'])} "
                f"rho_kept={int(row['rho_cols_kept'])} edge={int(row['rho_edge_cols_kept'])}"
            )

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
