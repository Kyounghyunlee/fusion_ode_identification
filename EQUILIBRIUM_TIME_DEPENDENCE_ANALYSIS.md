# Equilibrium Time-Dependence Analysis

## Summary

**Your equilibrium data IS time-dependent, but your pipeline treats it as STATIC.**

## What the Data Contains

From `data/27567/equilibrium.nc`:

- **Time dimension**: 111 timesteps covering [-0.095, 0.455] seconds
- **Time-dependent variables** (35 total):
  - `psi(z, major_radius, time)`: 2D poloidal flux field - **CRITICAL**
  - `magnetic_axis_r(time)`, `magnetic_axis_z(time)`: Magnetic axis position
  - `lcfs_r(time)`, `lcfs_z(time)`: Last closed flux surface coordinates
  - `q(profile_r, time)`: Safety factor profile
  - Geometry: `elongation(time)`, `triangularity_upper/lower(time)`, etc.

## How Your Pipeline Currently Uses Equilibrium

### 1. Preprocessing Stage (`preprocessing/build_training_pack.py`)

**Line 340-342:**
```python
# Geometry and rho normalisation (with fallback if equilibrium is degenerate)
it = choose_itime(eq)  # ← Picks MIDDLE timepoint
geom = extract_geom_params(eq, it)
```

**Line 375-380** (`preprocessing/geometry.py:choose_itime`):
```python
def choose_itime(eq: xr.Dataset) -> int:
    """Pick a representative time index (middle of the time dimension)."""
    if "time" not in eq.sizes:
        return 0
    return int(eq.sizes["time"] // 2)  # ← Fixed at timestep ~55/111
```

**Result**: 
- `psi_axis` and `psi_edge` computed from **t ≈ 0.18s only**
- `rho` (flux coordinate) mapping frozen at middle timepoint
- `Vprime` (dV/drho) computed from equilibrium at middle timepoint

### 2. Training Data Structure

**From `preprocessing/build_training_pack.py:381`:**
```python
rho_eq, V, Vprime = volume_derivatives(eq, it) if not rho_fallback_used else (None, None, None)
```

**Saved to `.npz` file:**
- `Vprime`: 1D array (65 values) - **STATIC**
- `rho`: 1D array - **STATIC**
- No time-dependent equilibrium information

### 3. Model Runtime (`fusion_ode_identification/model.py`)

**Line 140-142:**
```python
def compute_physics_tendency(self, t, Te_total, z, args):
    (rho_vals, Vprime_vals, ctrl_interp, ...) = args
    Vprime = jnp.clip(_as64(Vprime_vals), 1e-6, None)  # ← CONSTANT array
```

**Result**: The ODE uses **fixed** `Vprime` throughout the entire discharge evolution.

## Implications

### ✅ What Works (Probably Fine)

If your shots have:
1. **Flat-top phases**: Equilibrium relatively stable
2. **Small geometry changes**: Axis doesn't move much
3. **Limited time window**: Training focused on quasi-steady periods

### ⚠️ Potential Issues

1. **Ramp-up/down phases**: 
   - Plasma current changing → `psi` field evolving
   - Using middle-timepoint `rho` mapping may mismatch early/late profiles
   
2. **ELMs or disruptions**:
   - Rapid boundary changes → `Vprime` varies
   - Static geometry won't capture this

3. **Pedestal evolution**:
   - H-mode pedestal position tied to equilibrium
   - Fixed `rho` grid may not track moving pedestal

4. **Accuracy of transport coefficients**:
   - If true `Vprime(t)` varies by >20%, learned χ may compensate incorrectly

## Verification: Check Equilibrium Variability

Run this to see how much your equilibrium actually changes:

```python
import xarray as xr
import numpy as np

eq = xr.load_dataset("data/27567/equilibrium.nc")
psi = eq['psi'].values  # (65, 65, 111)

# Check axis position variation
R_axis = eq['magnetic_axis_r'].values
Z_axis = eq['magnetic_axis_z'].values
print(f"R_axis range: [{R_axis.min():.3f}, {R_axis.max():.3f}] m (Δ={R_axis.max()-R_axis.min():.3f})")
print(f"Z_axis range: [{Z_axis.min():.3f}, {Z_axis.max():.3f}] m (Δ={Z_axis.max()-Z_axis.min():.3f})")

# Check psi_axis and psi_edge variation
psi_axis = np.array([psi[:,:,t].min() for t in range(111)])
psi_edge = np.array([psi[:,:,t].max() for t in range(111)])
psi_span = psi_edge - psi_axis
print(f"psi normalization span: [{psi_span.min():.3f}, {psi_span.max():.3f}] (variation: {100*(psi_span.max()-psi_span.min())/psi_span.mean():.1f}%)")

# Check elongation variation
kappa = eq['elongation'].values
print(f"Elongation κ: [{kappa.min():.3f}, {kappa.max():.3f}] (Δ={kappa.max()-kappa.min():.3f})")
```

**If variations are < 5%**: Current static approach is reasonable.  
**If variations are > 20%**: Consider implementing time-dependent equilibrium.

## Recommendations

### Option 1: Accept Current Limitation (Easiest)

**When appropriate:**
- Training only on flat-top phases
- Equilibrium verified to be quasi-static
- Model performs well in validation

**Document in your paper/code:**
> "We assume quasi-static MHD equilibrium and use the equilibrium reconstruction from the discharge midpoint for flux coordinate mapping throughout the simulation."

### Option 2: Interpolate Equilibrium in Time (Medium Effort)

**Changes needed:**
1. Store time-dependent `psi(t)`, `psi_axis(t)`, `psi_edge(t)` in `.npz` packs
2. Modify `preprocessing/build_training_pack.py` to save:
   ```python
   np.savez(
       ...,
       psi_axis_t=psi_axis_t,  # (Nt_eq,)
       psi_edge_t=psi_edge_t,  # (Nt_eq,)
       t_eq=t_eq,              # (Nt_eq,)
       Vprime_t=Vprime_t,      # (Nrho, Nt_eq)
   )
   ```
3. Update `model.py` to interpolate `Vprime(t)` during ODE solve
4. Recompute `rho` mapping for each Thomson profile timepoint

**Complexity:** ~2-3 days implementation

### Option 3: Full Time-Dependent Geometry (High Effort)

**Full physics-accurate approach:**
- Recompute flux surfaces at each ODE timestep
- Track pedestal position in moving coordinates
- Account for geometry changes in transport operators

**Complexity:** ~1-2 weeks, major refactor

## Current Status

✅ **Static equilibrium approach**: Fully implemented and working  
⚠️ **Time-dependent equilibrium**: Not implemented  
📊 **Impact assessment**: Needs measurement (run variability check above)

## Next Steps

1. **Quantify equilibrium variability** for your shots (run check script above)
2. **Decide if limitation is acceptable** for your science goals
3. **Document assumption** in code and publications
4. **Consider upgrade** if modeling ramp-up/down phases or seeing unexplained errors

---

**Date**: 2025-12-22  
**Analyzed by**: Check of `preprocessing/build_training_pack.py`, `preprocessing/geometry.py`, `fusion_ode_identification/model.py`, and data structures.
