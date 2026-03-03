#!/usr/bin/env python3
"""Check whether NPZ files were built from NetCDF or Zarr sources."""

import numpy as np
import os
from pathlib import Path


def check_npz_source(npz_path: str) -> str:
    """Determine if NPZ was built from NetCDF or Zarr.
    
    Returns:
        "zarr" if built from zarr
        "netcdf" if built from NetCDF
        "unknown" if cannot determine
    """
    try:
        d = np.load(npz_path)
        
        # Key indicator: Te_units and ne_units
        te_units = d.get("Te_units", b"").decode() if isinstance(d.get("Te_units", ""), bytes) else str(d.get("Te_units", ""))
        ne_units = d.get("ne_units", b"").decode() if isinstance(d.get("ne_units", ""), bytes) else str(d.get("ne_units", ""))
        
        # Zarr path sets these to "unknown_zarr" (see line 408)
        if te_units == "unknown_zarr" or ne_units == "unknown_zarr":
            return "zarr"
        
        # NetCDF path either has actual units or empty string
        # Since zarr is the only one that explicitly sets "unknown_zarr"
        return "netcdf"
        
    except Exception as e:
        print(f"  Error reading {npz_path}: {e}")
        return "unknown"


def main():
    data_dir = Path("data")
    
    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        return
    
    # Find all NPZ files
    npz_files = sorted(data_dir.glob("*_torax_training.npz"))
    
    if not npz_files:
        print("No training pack NPZ files found in data/")
        return
    
    print(f"Found {len(npz_files)} training pack(s)\n")
    print(f"{'Shot':<10} {'Source':<10} {'Te_units':<20} {'ne_units':<20}")
    print("-" * 70)
    
    netcdf_count = 0
    zarr_count = 0
    
    for npz_file in npz_files:
        shot = npz_file.stem.split("_")[0]
        source = check_npz_source(str(npz_file))
        
        # Get units for display
        try:
            d = np.load(str(npz_file))
            te_units = d.get("Te_units", b"").decode() if isinstance(d.get("Te_units", ""), bytes) else str(d.get("Te_units", ""))
            ne_units = d.get("ne_units", b"").decode() if isinstance(d.get("ne_units", ""), bytes) else str(d.get("ne_units", ""))
            te_units = te_units if te_units else "(empty)"
            ne_units = ne_units if ne_units else "(empty)"
        except:
            te_units = "error"
            ne_units = "error"
        
        print(f"{shot:<10} {source:<10} {te_units:<20} {ne_units:<20}")
        
        if source == "netcdf":
            netcdf_count += 1
        elif source == "zarr":
            zarr_count += 1
    
    print("-" * 70)
    print(f"\nSummary: {netcdf_count} from NetCDF, {zarr_count} from Zarr")
    
    if netcdf_count > 0 and zarr_count == 0:
        print("✓ All training packs were built from NetCDF files")
    elif zarr_count > 0:
        print("⚠ Some training packs were built from Zarr (may have issues)")


if __name__ == "__main__":
    main()
