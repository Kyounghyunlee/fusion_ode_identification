#!/usr/bin/env python3
"""
Inspect thomson_scattering.nc to see coordinate system for Te/ne.
"""

import xarray as xr
import numpy as np

shot = 27567
ts_path = f"data/{shot}/thomson_scattering.nc"

print(f"Loading {ts_path}")
ts = xr.load_dataset(ts_path)

print("\n" + "="*70)
print("DATASET OVERVIEW")
print("="*70)
print(ts)

print("\n" + "="*70)
print("DIMENSIONS")
print("="*70)
for dim, size in ts.dims.items():
    print(f"  {dim}: {size}")

print("\n" + "="*70)
print("COORDINATES")
print("="*70)
for coord_name in ts.coords:
    coord = ts.coords[coord_name]
    print(f"\n{coord_name}:")
    print(f"  dims: {coord.dims}")
    print(f"  shape: {coord.shape}")
    print(f"  dtype: {coord.dtype}")
    if coord.size < 100:
        print(f"  values: {coord.values}")
    else:
        print(f"  values[0:5]: {coord.values.flat[0:5]}")
        print(f"  values[-5:]: {coord.values.flat[-5:]}")

print("\n" + "="*70)
print("DATA VARIABLES")
print("="*70)
for var_name in ts.data_vars:
    var = ts[var_name]
    print(f"\n{var_name}:")
    print(f"  dims: {var.dims}")
    print(f"  shape: {var.shape}")
    print(f"  dtype: {var.dtype}")
    print(f"  attrs: {dict(var.attrs)}")
    if var.size < 20:
        print(f"  values:\n{var.values}")

# Search for Te/ne variables
print("\n" + "="*70)
print("TEMPERATURE & DENSITY VARIABLES")
print("="*70)

te_candidates = ["Te", "T_e", "te", "Te_eV", "t_e", "electron_temperature"]
ne_candidates = ["ne", "n_e", "ne_cm3", "ne_m3", "electron_density"]

te_found = None
ne_found = None

for cand in te_candidates:
    if cand in ts:
        te_found = cand
        break

for cand in ne_candidates:
    if cand in ts:
        ne_found = cand
        break

if te_found:
    print(f"\nFound Te as '{te_found}':")
    te_var = ts[te_found]
    print(f"  dims: {te_var.dims}")
    print(f"  shape: {te_var.shape}")
    print(f"  attrs: {dict(te_var.attrs)}")
else:
    print("\nTe NOT FOUND")

if ne_found:
    print(f"\nFound ne as '{ne_found}':")
    ne_var = ts[ne_found]
    print(f"  dims: {ne_var.dims}")
    print(f"  shape: {ne_var.shape}")
    print(f"  attrs: {dict(ne_var.attrs)}")
else:
    print("\nne NOT FOUND")

# Check for radial coordinate indicators
print("\n" + "="*70)
print("RADIAL COORDINATE ANALYSIS")
print("="*70)

rho_like = ["rho", "rho_ts", "psi_N", "psiN", "psi_norm"]
rz_like = ["R", "Z", "R_midplane", "R_channel", "major_radius", "Z_midplane", "Z_channel"]

print("\nFlux coordinates (rho-like):")
for name in rho_like:
    if name in ts.coords:
        print(f"  ✓ {name} (coord)")
    elif name in ts:
        print(f"  ✓ {name} (variable)")
    else:
        print(f"  ✗ {name}")

print("\nPhysical coordinates (R,Z-like):")
for name in rz_like:
    if name in ts.coords:
        print(f"  ✓ {name} (coord)")
    elif name in ts:
        print(f"  ✓ {name} (variable)")
    else:
        print(f"  ✗ {name}")

# Inference
print("\n" + "="*70)
print("COORDINATE SYSTEM INFERENCE")
print("="*70)

has_rho = any(name in ts.coords or name in ts for name in rho_like)
has_rz = any(name in ts.coords or name in ts for name in rz_like)

if has_rho and not has_rz:
    print("→ Data stored in FLUX COORDINATES (ρ)")
    print("  Te, ne likely indexed by (time, rho) or similar")
elif has_rz and not has_rho:
    print("→ Data stored in PHYSICAL COORDINATES (R, Z)")
    print("  Te, ne likely indexed by (time, channel) with R,Z positions")
elif has_rho and has_rz:
    print("→ Data has BOTH coordinate systems")
    print("  Need to check which one Te/ne actually use")
    if te_found:
        te_dims = ts[te_found].dims
        print(f"  Te dims: {te_dims}")
        for dim in te_dims:
            if dim in rho_like:
                print(f"    → Uses flux coordinate '{dim}'")
            elif dim in rz_like:
                print(f"    → Uses physical coordinate '{dim}'")
else:
    print("→ UNCLEAR: Neither clear rho nor R,Z found")
    print("  May use implicit channel indexing")

print("\n" + "="*70)
