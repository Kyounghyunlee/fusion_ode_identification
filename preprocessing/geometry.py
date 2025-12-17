"""
Geometry utilities to derive flux coordinate rho from equilibrium data
and extract tokamak scalar geometry parameters for TORAX.

Assumptions:
- equilibrium.nc contains variables like psirz(time,R,Z), R(time?) axis, LCFS traces.
- Coordinates are named "R" and "Z" for the poloidal plane. Adjust if needed.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr


def choose_itime(eq: xr.Dataset) -> int:
    """Pick a representative time index (middle of the time dimension)."""
    if "time" not in eq.sizes:
        return 0
    return int(eq.sizes["time"] // 2)


def _get_psi_2d(eq: xr.Dataset, itime: int) -> xr.DataArray:
    """Return psi(R,Z) at a given time, supporting common naming variants."""
    for name in ("psirz", "psi", "psi_pol", "psi_poloidal"):
        if name in eq:
            da = eq[name].isel(time=itime) if "time" in eq[name].dims else eq[name]
            return da
    raise KeyError("No psi-like 2D field found (psirz/psi/psi_pol)")


def _get_RZ_coords(eq: xr.Dataset) -> Tuple[str, str]:
    """Find the names of R and Z coordinates used in psi(R,Z)."""
    # Common names including MAST conventions
    r_candidates = ("R", "R_grid", "Rcoord", "R_grid_1d", "major_radius")
    z_candidates = ("Z", "Z_grid", "Zcoord", "Z_grid_1d", "z")
    r_name = next((n for n in r_candidates if n in eq.coords or n in eq), None)
    z_name = next((n for n in z_candidates if n in eq.coords or n in eq), None)
    if not r_name or not z_name:
        raise KeyError("Could not determine R/Z coordinate names in equilibrium dataset")
    return r_name, z_name


def _psi_axis_value(psi2d: xr.DataArray) -> float:
    """Approximate psi at the magnetic axis as the minimum value of psi(R,Z)."""
    return float(np.nanmin(psi2d.values))


def _psi_edge_value(eq: xr.Dataset, psi2d: xr.DataArray, itime: int) -> float:
    """Estimate psi at the LCFS by sampling psi along the LCFS contour.

    Requires R_lcfs(time,npts), Z_lcfs(time,npts) in equilibrium dataset.
    If not available, falls back to mean psi on outer boundary of the grid.
    """
    psi_max = float(np.nanmax(psi2d.values))
    lcfs_r_keys = ("R_lcfs", "lcfs_r")
    lcfs_z_keys = ("Z_lcfs", "lcfs_z")

    try:
        r_key = next(k for k in lcfs_r_keys if k in eq)
        z_key = next(k for k in lcfs_z_keys if k in eq)
        R_lcfs = eq[r_key]
        Z_lcfs = eq[z_key]

        # Align time dimension if present
        if "time" in R_lcfs.dims:
            R_lcfs = R_lcfs.isel(time=itime)
        if "time" in Z_lcfs.dims:
            Z_lcfs = Z_lcfs.isel(time=itime)

        r_name, z_name = _get_RZ_coords(eq)
        psi_s = psi2d.interp({r_name: R_lcfs, z_name: Z_lcfs})
        psi_edge = float(np.nanmax(psi_s.values))
        return psi_max if not np.isfinite(psi_edge) or psi_edge < psi_max else psi_edge
    except Exception:
        # Fallback: average psi on the rectangular boundary
        arr = psi2d.values
        top = arr[-1, :]
        bottom = arr[0, :]
        left = arr[:, 0]
        right = arr[:, -1]
        boundary = np.concatenate([top, bottom, left, right])
        psi_edge = float(np.nanmax(boundary))
        return psi_max if not np.isfinite(psi_edge) or psi_edge < psi_max else psi_edge


def compute_rho_scalars(eq: xr.Dataset, itime: Optional[int] = None) -> Dict[str, float]:
    """Compute psi_axis and psi_edge scalars used to normalise psi → rho.

    Returns a dict with keys: psi_axis, psi_edge.
    """
    it = choose_itime(eq) if itime is None else itime
    psi2d = _get_psi_2d(eq, it)
    psi_axis = _psi_axis_value(psi2d)
    psi_edge = _psi_edge_value(eq, psi2d, it)
    if not np.isfinite(psi_edge - psi_axis) or abs(psi_edge - psi_axis) < 1e-12:
        raise ValueError("Degenerate psi normalization: psi_edge ≈ psi_axis")
    return {"psi_axis": float(psi_axis), "psi_edge": float(psi_edge)}


def rho_from_RZ(eq: xr.Dataset, R: np.ndarray, Z: np.ndarray, itime: Optional[int] = None) -> np.ndarray:
    """Compute rho for arrays of (R,Z) points using psirz normalization.

    rho = sqrt((psi - psi_axis) / (psi_edge - psi_axis)) clipped to [0,1].
    """
    it = choose_itime(eq) if itime is None else itime
    psi2d = _get_psi_2d(eq, it)
    scalars = compute_rho_scalars(eq, it)
    r_name, z_name = _get_RZ_coords(eq)
    psi_vals = psi2d.interp({r_name: ("pts", R), z_name: ("pts", Z)}).values
    psiN = (psi_vals - scalars["psi_axis"]) / (scalars["psi_edge"] - scalars["psi_axis"])
    rho = np.sqrt(np.clip(psiN, 0.0, 1.0))
    return rho


def extract_geom_params(eq: xr.Dataset, itime: Optional[int] = None) -> Dict[str, float]:
    """Extract simple tokamak geometry scalars from equilibrium dataset.

    Returns dict with keys: R_major, a_minor, kappa, delta.
    """
    it = choose_itime(eq) if itime is None else itime
    R_axis = float(eq["R_axis"].isel(time=it).item()) if "R_axis" in eq else float("nan")

    try:
        R_lcfs = eq["R_lcfs"].isel(time=it).values
        Z_lcfs = eq["Z_lcfs"].isel(time=it).values
        a_minor = float(np.max(R_lcfs - R_axis)) if np.isfinite(R_axis) else float("nan")
        kappa = float((np.nanmax(Z_lcfs) - np.nanmin(Z_lcfs)) / (2.0 * a_minor)) if np.isfinite(a_minor) and a_minor > 0 else float("nan")
        i_top = int(np.nanargmax(Z_lcfs))
        delta = float((R_lcfs[i_top] - R_axis) / a_minor) if np.isfinite(a_minor) and a_minor > 0 else float("nan")
    except Exception:
        a_minor = float("nan")
        kappa = float("nan")
        delta = float("nan")

    return {
        "R_major": R_axis,
        "a_minor": a_minor,
        "kappa": kappa,
        "delta": delta,
    }


def volume_derivatives(eq: xr.Dataset, itime: Optional[int] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Return (rho, V, Vprime) if available from equilibrium dataset.

    V := flux_surface_volume(time, rho). Computes Vprime = dV/drho via np.gradient.
    Returns (None, None, None) if not enough info is present.
    """
    it = choose_itime(eq) if itime is None else itime
    if "flux_surface_volume" in eq and "rho" in eq.dims:
        V = eq["flux_surface_volume"].isel(time=it).values
        rho = eq["rho"].isel(time=it).values if "time" in eq["rho"].dims else eq["rho"].values
        Vprime = np.gradient(V, rho)
        return rho, V, Vprime
    return None, None, None
