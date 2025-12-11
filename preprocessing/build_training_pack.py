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

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr

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
) -> np.ndarray:
    """Interpolate time-by-sample profiles to a target rho grid.

    values_t_s: shape (Nt, Ns)
    """
    Nt = values_t_s.shape[0]
    out = np.full((Nt, rho_dst.size), np.nan, dtype=float)
    # Ensure monotonic and finite source
    m = np.isfinite(rho_src)
    rho_src = rho_src[m]
    for t in range(Nt):
        v = values_t_s[t]
        vm = m & np.isfinite(v)
        if np.count_nonzero(vm) == 0:
            continue
        rs = rho_src[vm]
        vs = v[vm]
        # sort by rho
        idx = np.argsort(rs)
        rs_sorted = rs[idx]
        vs_sorted = vs[idx]
        left_val = vs_sorted[0]
        right_val = vs_sorted[-1]
        out[t] = np.interp(rho_dst, rs_sorted, vs_sorted, left=left_val, right=right_val)
    return out


def build_one_shot(shot: int, data_root: str = "data", Nrho: int = 65) -> str:
    shot_dir = os.path.join(data_root, str(shot))
    eq_path = os.path.join(shot_dir, "equilibrium.nc")
    ts_path = os.path.join(shot_dir, "thomson_scattering.nc")
    sm_path = os.path.join(shot_dir, "summary.nc")

    if not (os.path.exists(eq_path) and os.path.exists(ts_path) and os.path.exists(sm_path)):
        raise FileNotFoundError(f"Missing one or more NetCDF files for shot {shot}")

    eq = xr.load_dataset(eq_path)
    ts = xr.load_dataset(ts_path)
    summ = xr.load_dataset(sm_path)

    # Geometry and rho normalisation
    it = choose_itime(eq)
    scalars = compute_rho_scalars(eq, it)
    geom = extract_geom_params(eq, it)
    rho_eq, V, Vprime = volume_derivatives(eq, it)

    # TORAX rho grid
    if rho_eq is not None and rho_eq.size >= Nrho:
        # Subsample equilibrium rho to preserve flux surface alignment
        idx = np.linspace(0, rho_eq.size - 1, Nrho).astype(int)
        rho_torax = rho_eq[idx]
    else:
        rho_torax = np.linspace(0.0, 1.0, Nrho)

    # Interpolate Vprime onto TORAX rho grid
    Vprime_torax = None
    if rho_eq is not None and Vprime is not None:
        # Simple 1D interp; assume rho_eq in [0,1]
        Vprime_torax = np.interp(rho_torax, rho_eq, Vprime)
    else:
        Vprime_torax = np.ones_like(rho_torax)

    # Thomson scattering profiles
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
        # Case A: TS already has rho-like coordinate
        rho_ts = ts.coords[rho_coord_name].values
        # Align shapes to (time, samples)
        # Choose the second dimension (non-time) for interpolation
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

        Te_rho_t = profiles_to_rho_grid(rho_ts, Te_ts, rho_torax)
        ne_rho_t = profiles_to_rho_grid(rho_ts, ne_ts, rho_torax)

    else:
        # Case B: Need to compute rho from (R,Z) channel positions
        # Try to find R and Z per channel; allow some naming variants
        R_da = get_var(ts, ["R", "R_midplane", "R_channel", "major_radius"])  # expect dims (time?, channel)
        Z_da = get_var(ts, ["Z", "Z_midplane", "Z_channel"])  # may be constant or per-time
        if R_da is None:
            raise KeyError("Thomson dataset missing R coordinate for channels and no rho given")

        # Make (time, samples) arrays for R, Z, Te, ne
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
            # Use midplane Z=0 if not provided
            Z_t_s = np.zeros_like(R_t_s)
        else:
            Z_t_s = to_time_samples_fill(Z_da)

        Te_t_s = to_time_samples_fill(Te_da)
        ne_t_s = to_time_samples_fill(ne_da)

        # Compute rho per time/channel from equilibrium once per time sample index
        # For simplicity, use single geometry at it for all times
        # If TS provides only major_radius, assume midplane Z=0
        rho_chan = rho_from_RZ(eq, R_t_s[0], Z_t_s[0], itime=it)
        Te_rho_t = profiles_to_rho_grid(rho_chan, Te_t_s, rho_torax)
        ne_rho_t = profiles_to_rho_grid(rho_chan, ne_t_s, rho_torax)

    # Summary signals
    t = get_var(summ, ["time"]).values
    Ip = get_var(summ, ["ip", "Ip"]).values if get_var(summ, ["ip", "Ip"]) is not None else np.full_like(t, np.nan)
    nebar_da = get_var(summ, ["line_average_n_e", "ne_bar", "nebar"])  # m^-3 typically
    nebar = nebar_da.values if nebar_da is not None else np.full_like(t, np.nan)
    P_nbi_da = get_var(summ, ["power_nbi", "P_NBI", "pnbi"])
    P_rad_da = get_var(summ, ["power_radiated", "P_rad", "prad"])
    P_nbi = P_nbi_da.values if P_nbi_da is not None else np.full_like(t, np.nan)
    P_rad = P_rad_da.values if P_rad_da is not None else np.full_like(t, np.nan)

    # Extended Summary Signals (Level-2)
    W_tot = get_var(summ, ["W_tot", "w_tot", "stored_energy", "energy_total"]).values if get_var(summ, ["W_tot", "w_tot", "stored_energy", "energy_total"]) is not None else np.full_like(t, np.nan)
    P_ohm = get_var(summ, ["p_ohm", "power_ohmic", "P_ohm"]).values if get_var(summ, ["p_ohm", "power_ohmic", "P_ohm"]) is not None else np.full_like(t, np.nan)
    P_tot = get_var(summ, ["p_tot", "power_total", "P_tot"]).values if get_var(summ, ["p_tot", "power_total", "P_tot"]) is not None else np.full_like(t, np.nan)
    ne_line = get_var(summ, ["n_e_line", "ne_line", "line_average_density", "line_average_n_e"]).values if get_var(summ, ["n_e_line", "ne_line", "line_average_density", "line_average_n_e"]) is not None else np.full_like(t, np.nan)
    H98 = get_var(summ, ["H98", "H_98", "h98", "h_factor_98y2"]).values if get_var(summ, ["H98", "H_98", "h98", "h_factor_98y2"]) is not None else np.full_like(t, np.nan)

    # Magnetics / Equilibrium Scalars (try summary first, then equilibrium)
    q95 = get_var(summ, ["q95", "q_95"])
    if q95 is None: q95 = get_var(eq, ["q95", "q_95"])
    q95 = q95.values if q95 is not None else np.full_like(t, np.nan)

    li = get_var(summ, ["li", "li_3", "internal_inductance"])
    if li is None: li = get_var(eq, ["li", "li_3"])
    li = li.values if li is not None else np.full_like(t, np.nan)

    beta_n = get_var(summ, ["beta_n", "beta_N", "beta_normalised"])
    if beta_n is None: beta_n = get_var(eq, ["beta_n", "beta_N"])
    beta_n = beta_n.values if beta_n is not None else np.full_like(t, np.nan)

    B_t0 = get_var(summ, ["B_t0", "b_t0", "toroidal_field_center"])
    if B_t0 is None: B_t0 = get_var(eq, ["B_t0", "b_t0"])
    B_t0 = B_t0.values if B_t0 is not None else np.full_like(t, np.nan)

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
    t_ts = ts["time"].values if "time" in ts.coords else np.arange(Nt)

    # Simple regime labelling on summary grid (t_summary)
    # 0 = unknown, 1 = L-mode, 2 = transition, 3 = H-mode
    regime = np.zeros_like(t_summary, dtype=np.int8)

    # crude heuristic: look at nebar and P_NBI threshold & derivative
    if nebar_da is not None and P_nbi_da is not None:
        ne_norm = (nebar - np.nanmin(nebar)) / (np.nanmax(nebar) - np.nanmin(nebar) + 1e-6)
        d_ne = np.gradient(ne_norm, t_summary)
        # "transition" when derivative peaks
        trans_idx = np.argmax(np.abs(d_ne))
        regime[trans_idx] = 2
        # mark some window before as L and after as H
        w = max(2, regime.size // 20)
        regime[: max(0, trans_idx - w)] = 1
        regime[min(regime.size - 1, trans_idx + w) :] = 3

    # Masks for valid profile samples (finite values)
    Te_mask = np.isfinite(Te_rho_t)
    ne_mask = np.isfinite(ne_rho_t)

    # Save NPZ bundle
    out_npz = os.path.join(data_root, f"{shot}_torax_training.npz")
    np.savez_compressed(
        out_npz,
        t=t_summary,
        t_ts=t_ts,
        rho=rho_torax,
        Te=Te_rho_t,
        ne=ne_rho_t,
        Te_mask=Te_mask,
        ne_mask=ne_mask,
        Ip=Ip,
        nebar=nebar,
        P_nbi=P_nbi,
        P_rad=P_rad,
        P_nbi_raw=P_nbi,
        P_rad_raw=P_rad,
        S_gas=S_gas,
        S_rec=S_rec,
        S_nbi=S_nbi,
        Vprime=Vprime_torax,
        regime=regime,
        # Extended fields
        W_tot=W_tot,
        P_ohm=P_ohm,
        P_tot=P_tot,
        ne_line=ne_line,
        H98=H98,
        q95=q95,
        li=li,
        beta_n=beta_n,
        B_t0=B_t0,
        schema_version=2,
        psi_axis=scalars["psi_axis"],
        psi_edge=scalars["psi_edge"],
        **geom,
    )

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
    for shot in shots:
        print(f"[build] Shot {shot}")
        try:
            path = build_one_shot(shot, data_root="data", Nrho=args.Nrho)
            print(f"  -> {path}")
            created.append(path)
        except Exception as e:
            print(f"  !! Failed for shot {shot}: {e}")

    if not created:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
