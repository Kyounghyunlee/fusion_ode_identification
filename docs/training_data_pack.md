# Training Pack Build Process

## Overview

This document describes **exactly** how the training packs were built for the
current dataset (shots 27567–27594), which code paths were taken, and what is
stored in each `.npz` file. It is based on inspection of the actual saved files
and the full source of `preprocessing/build_training_pack.py` and
`preprocessing/geometry.py`.

Recommended invocation from the repo root:

```bash
python -m preprocessing.build_training_pack --discover
```

---

## 1. Input Data Per Shot

For each shot, three NetCDF files are required under `data/<shot>/`:

| File | Purpose |
|------|---------|
| `equilibrium.nc` | Magnetic equilibrium → flux coordinates, geometry, `Vprime` |
| `thomson_scattering.nc` | TS diagnostics → `Te`, `ne` profiles vs. time |
| `summary.nc` | Global time-series → `Ip`, `nebar`, `P_nbi`, `P_rad`, etc. |

Optional files are also attempted:

| File | Purpose |
|------|---------|
| `gas_injection.nc` | Gas puffing rate → `S_gas` |
| `d_alpha.nc` | Compact D-alpha sidecar extracted from the visible spectrometer diagnostic |
| `spectrometer_visible.nc` | Fallback source for D-alpha if `d_alpha.nc` is absent |

---

## 2. Step-by-Step Build Process

### Step 1: Load All NetCDF Files

```python
eq   = xr.load_dataset("data/<shot>/equilibrium.nc")
ts   = xr.load_dataset("data/<shot>/thomson_scattering.nc")
summ = xr.load_dataset("data/<shot>/summary.nc")
```

---

### Step 2: Geometry and Rho Coordinate

#### 2a. Choose Representative Equilibrium Time Index

```python
it = choose_itime(eq)
```

`choose_itime` picks the **middle equilibrium snapshot**:

$$
i_t = \left\lfloor \frac{N_{\mathrm{eq}}}{2} \right\rfloor
$$

where $N_{\mathrm{eq}}$ is the length of the equilibrium time dimension.
For the present dataset, this means the pack is built from **one frozen-in-time magnetic equilibrium per shot**, not from a time-varying sequence of equilibria.

Mathematically, every equilibrium-derived quantity used downstream is evaluated from

$$
\mathcal{E}_{\mathrm{shot}} = \mathcal{E}(t_{i_t}).
$$

That snapshot defines:
- the flux geometry used to interpret radial position,
- the normalized radial coordinate $\rho$,
- the equilibrium volume profile $V(\rho)$,
- and the geometric weight $V'(\rho)=dV/d\rho$ used later in transport.

This is a **quasi-static flux-coordinate construction**: the shot has many diagnostic time samples, but all of them are mapped onto the same equilibrium geometry.

#### 2b. Extract Geometry Scalars

```python
geom = extract_geom_params(eq, it)
```

The equilibrium is treated as an **axisymmetric poloidal cross-section**. In the present files, the relevant coordinates are the major-radius-like horizontal coordinate and vertical coordinate:

$$
(R, Z) \in \mathbb{R}^2,
$$

with the poloidal flux represented as a scalar field

$$
\psi = \psi(R, Z; t_{i_t}).
$$

In the MAST-style files inspected here, this field is stored on a rectilinear grid with coordinates equivalent to:
- `major_radius` for $R$,
- `z` for $Z$,
- `time` for the equilibrium time index.

`extract_geom_params(eq, it)` then tries to derive a few scalar summaries of the same frozen equilibrium. The intended formulas are:

$$
R_{\mathrm{major}} = R_{\mathrm{axis}},
$$

$$
a_{\mathrm{minor}} = \max(R_{\mathrm{LCFS}} - R_{\mathrm{axis}}),
$$

$$
\kappa = \frac{Z_{\max} - Z_{\min}}{2 a_{\mathrm{minor}}},
$$

$$
\delta = \frac{R_{\mathrm{LCFS}}(Z_{\max}) - R_{\mathrm{axis}}}{a_{\mathrm{minor}}}.
$$

For downstream training in this repository, these scalar shape descriptors are **not the primary geometric objects**. The objects that actually matter are the flux mapping and the volume derivative $V'(\rho)$. The scalar geometry is therefore best viewed as metadata, while the effective transport geometry is carried by the coordinate transform and by $V'(\rho)$.

#### 2c. Rho Coordinate — PSI-BASED PATH (no fallback used)

```python
scalars     = compute_rho_scalars(eq, it)
psi_axis    = scalars["psi_axis"]
psi_edge    = scalars["psi_edge"]
rho_fn      = lambda r, z: rho_from_RZ(eq, r, z, itime=it)
```

This is the **core coordinate construction actually used** by the pack builder.

The code first extracts two normalization constants from the frozen flux map:

$$
\psi_{\mathrm{axis}} = \min_{R,Z} \psi(R,Z),
$$

and an edge value $\psi_{\mathrm{edge}}$ estimated from the last closed flux surface (LCFS) contour, or, failing that, from the outer boundary of the $\psi(R,Z)$ grid.

These define the normalized poloidal flux

$$
\psi_N(R,Z) = \frac{\psi(R,Z)-\psi_{\mathrm{axis}}}{\psi_{\mathrm{edge}}-\psi_{\mathrm{axis}}}.
$$

The radial coordinate used by the training pack is then the square-root flux radius

$$
\rho(R,Z) = \sqrt{\operatorname{clip}(\psi_N(R,Z), 0, 1)}.
$$

Why the square root? Because normalized poloidal flux is naturally an **area-like flux-surface label**, whereas transport models usually want a **radius-like coordinate**.

In a simple large-aspect-ratio tokamak picture, the enclosed poloidal flux grows approximately like the square of minor radius:

$$
\psi_N \propto r^2.
$$

So if one used $\psi_N$ directly as the radial coordinate, equal steps in that coordinate would correspond more nearly to equal increments of enclosed flux or cross-sectional area, not equal increments of physical radius. Taking the square root restores a coordinate that behaves roughly like a normalized minor radius:

$$
\rho \sim \frac{r}{a}.
$$

This has three practical consequences:

1. $\rho=0.5$ is interpreted roughly as "halfway out in minor radius," not "half the enclosed normalized flux." 
2. Radial gradients $\partial/\partial\rho$ become easier to interpret as axis-to-edge profile gradients.
3. The resulting coordinate is closer to the conventions commonly used for profile transport, pedestal location, and equilibrium-derived geometry tables.

So the chain is:

$$
\psi_N \;\text{is a normalized flux label},
\qquad
\rho=\sqrt{\psi_N} \;\text{is the corresponding radius-like label}.
$$

This is why the code does **not** use $\rho=\psi_N$. It uses the square root specifically to convert a flux fraction into a coordinate that behaves more like radial distance from the magnetic axis.

#### 2c.1 Coordinate and Symbol Clarification: `Z` vs `z`

There are two different symbols in this project that look similar but mean completely different things:

- $Z$ is the **geometric vertical coordinate** in the poloidal cross-section.
- $z$ is the **latent scalar state** used later by the ROM.

So:

$$
Z \;\text{is spatial geometry},
\qquad
z \;\text{is dynamical latent state}.
$$

More concretely:

- $(R,Z)$ tells you where a point sits in the 2D poloidal plane of the equilibrium.
- $\psi_N(R,Z)$ and $\rho(R,Z)$ are computed from that geometry.
- $z(t)$ is introduced only later, in the transport model, as a low-dimensional variable that modulates edge transport.

The current ROM uses $z$ as an order-parameter-like scalar for confinement state. In the implemented model, it enters the diffusivity profile through

$$
\chi = \chi(\rho,z),
$$

so changing $z$ changes the **transport coefficient profile**, especially near the edge, but does not specify a spatial point inside the plasma.

This means `z` is not part of the equilibrium coordinate system at all. It is a state variable of the learned dynamical model.

#### 2c.2 Are $(\rho,z)$ Coordinates of the Plasma?

No. Your intuition is correct: $(\rho,z)$ is **not** a one-to-one coordinate system for points inside the torus.

There are two separate reasons.

First, lowercase $z$ is not a spatial coordinate. It is a single global latent variable, shared by the whole plasma state at a given time. So the pair $(\rho,z)$ means roughly:

- radial location $\rho$,
- plus global confinement state $z(t)$.

That pair labels the arguments of the diffusivity function $\chi(\rho,z)$, not the position of a particle or grid point in 3D space.

Second, even $\rho$ by itself is not one-to-one with spatial location. A fixed value of $\rho$ corresponds to an entire flux surface, not to a unique point. In an axisymmetric tokamak, many spatial points share the same $\rho$.

If one wanted a spatial coordinate system for the full torus, one would need something like:

$$
(\rho, \theta, \varphi)
$$

or, in geometric coordinates,

$$
(R, Z, \varphi),
$$

where $\theta$ is a poloidal angle and $\varphi$ is the toroidal angle.

By contrast:

- $(R,Z)$ describes a point in the 2D poloidal cross-section,
- $(R,Z,\varphi)$ describes a point in the full 3D torus,
- $\rho$ labels a flux surface,
- $z$ describes the inferred confinement/transport state of the discharge.

So the statement

$$
\chi = \chi(\rho,z)
$$

should be read as:

"the diffusivity depends on where you are radially in flux space, and on the current latent confinement state,"

not as

"$(\rho,z)$ are spatial coordinates of the plasma."

At this point it is useful to separate two quantities that are easy to confuse:

- $\rho$ is a **coordinate**. It tells you **where** you are in the plasma, measured in normalized flux radius.
- $\chi$ is a **transport coefficient**. It tells you **how easily heat diffuses** at that location.

So they do very different jobs:

$$
\rho \;\text{labels position},
\qquad
\chi \;\text{labels transport strength}.
$$

The important structural point is that the pack builder computes and stores $\rho$, but it does **not** compute or store $\chi$. The diffusivity $\chi$ belongs to the downstream transport model, where it is defined **as a function on the rho grid**:

$$
\chi = \chi(\rho, z).
$$

In other words, $\rho$ is the independent radial variable, while $\chi$ is one of the coefficients living on that variable. A good mental picture is:

- $\rho$ answers: "which flux surface is this?"
- $\chi(\rho,z)$ answers: "how fast does heat spread across this flux surface neighborhood?"

This is the standard tokamak-style convention in which:
- $\rho=0$ corresponds to the magnetic axis,
- $\rho=1$ corresponds approximately to the plasma edge / LCFS,
- and equal increments in $\rho$ correspond to equal increments in **normalized flux radius**, not equal physical distance in $R$.

That distinction is important. The coordinate used here is **not** Euclidean radius and not simple major radius normalization. It is a flux surface label derived from the equilibrium solution itself. Two points with the same $\rho$ are intended to lie on the same magnetic surface, even if their geometric positions $(R,Z)$ differ.

To evaluate $\rho$ at arbitrary points, the code performs **2D interpolation of the equilibrium flux field** on the rectilinear $(R,Z)$ grid:

1. Select the 2D slice $\psi(R,Z;t_{i_t})$.
2. Interpolate $\psi$ to the requested point(s) $(R_k,Z_k)$.
3. Normalize with $\psi_{\mathrm{axis}}$ and $\psi_{\mathrm{edge}}$.
4. Apply the square root to obtain $\rho_k$.

In code terms, this is done by `xarray.DataArray.interp(...)`, which here acts as a **piecewise linear interpolant on the rectilinear equilibrium mesh**. So the effective map is:

$$
(R_k,Z_k) \xrightarrow[]{\text{bilinear / separable linear interp on grid}} \psi_k
\xrightarrow[]{\text{normalize}} \psi_{N,k}
\xrightarrow[]{\sqrt{\cdot}} \rho_k.
$$

For the current dataset, this psi-based construction succeeded for all shots, so the training packs are genuinely built in a flux coordinate system rather than in a geometric-radius approximation.

#### 2d. Volume Derivatives

```python
rho_eq, V, Vprime = volume_derivatives(eq, it)
```

Once the equilibrium flux coordinate is fixed, the builder obtains a 1D equilibrium volume profile

$$
V = V(\rho)
$$

and differentiates it numerically to obtain

$$
V'(\rho) = \frac{dV}{d\rho}.
$$

This quantity is central for transport because the radial heat equation is written in conservative form on flux surfaces. In a flux coordinate, diffusion is not just scaled by $\chi$ and radial gradients; it is also weighted by the flux-surface geometry. In the ROM used later, $V'(\rho)$ appears in the finite-volume diffusion operator as the geometric factor that converts face fluxes into cell-wise divergence.

The cleanest way to see the distinct roles of $\rho$, $V'(\rho)$, and $\chi(\rho,z)$ is through the transport operator used later in the ROM:

$$
\frac{\partial T_e}{\partial t}
=
\frac{1}{V'(\rho)}
\frac{\partial}{\partial \rho}
\left(
V'(\rho)\,\chi(\rho,z)\,\frac{\partial T_e}{\partial \rho}
\right)
+ \cdots
$$

Here:
- $\rho$ is the axis-to-edge coordinate along which differentiation is performed,
- $\partial T_e/\partial \rho$ is the temperature gradient with respect to that coordinate,
- $\chi(\rho,z)$ multiplies that gradient and controls the diffusive flux magnitude,
- $V'(\rho)$ supplies the geometry of the flux surfaces.

So the division is:

- The training pack provides the **grid and geometry**: $\rho$ and $V'(\rho)$.
- The ROM later learns or evaluates the **transport strength** on that grid: $\chi(\rho,z)$.

This is why $\rho$ and $\chi$ should not be thought of as competing quantities. They are not alternatives. The model needs both: one to describe **where** the state lives, and one to describe **how** it evolves there.

#### 2d.1 Comparison of the Main Radial / Geometric Objects

The following objects appear close together in the pipeline, but they have different meanings:

| Object | Type | What it represents | One-to-one with spatial point? | Used for |
|------|------|--------------------|-------------------------------|----------|
| $\psi(R,Z)$ | 2D scalar field | Poloidal flux on the equilibrium grid | No | Raw equilibrium field |
| $\psi_N(R,Z)$ | 2D scalar field | Normalized flux label | No | Flux-surface labeling |
| $\rho(R,Z)=\sqrt{\psi_N}$ | radius-like flux label | Normalized minor-radius-like coordinate | No | Radial coordinate for profiles and transport |
| $V(\rho)$ | 1D profile | Enclosed plasma volume versus flux radius | Not applicable | Geometry summary |
| $V'(\rho)$ | 1D profile | Geometric weight $dV/d\rho$ | Not applicable | Conservative transport discretization |
| $\chi(\rho,z)$ | 1D coefficient field at fixed $z$ | Heat diffusivity profile | Not by itself | Transport strength in the ROM |
| $Z$ | geometric coordinate | Vertical position in the poloidal plane | Yes, in 2D with $R$ | Equilibrium geometry |
| $z$ | latent scalar state | Global confinement / transport state | No | Modulates edge diffusivity |

The most important separation is:

- $\psi$, $\psi_N$, $\rho$, $V$, and $V'$ come from the **equilibrium geometry / preprocessing**.
- $\chi(\rho,z)$ and $z(t)$ belong to the **downstream transport model**.

So the pack builder constructs the radial geometry first, and the ROM later places a diffusivity model on top of that geometry.

Numerically, `volume_derivatives(...)` returns three aligned 1D arrays:
- `rho_eq`: the equilibrium-native radial coordinate,
- `V`: enclosed plasma volume as a function of `rho_eq`,
- `Vprime`: a numerical gradient of `V` with respect to `rho_eq`.

So the equilibrium contributes not only a coordinate label, but also the measure with which radial transport is discretized.

---

### Step 3: Build TORAX Rho Grid

Because `rho_eq` was available and `rho_eq.size >= 65` for all shots:

```python
idx       = np.linspace(0, rho_eq.size - 1, 65).astype(int)
rho_torax = rho_eq[idx]
```

This step constructs the radial grid that will actually be saved into the pack and later used for profile data and transport modeling.

The procedure is:

1. Start from the equilibrium-native grid `rho_eq`.
2. Select 65 indices uniformly in **index space**, not uniformly in `rho` value.
3. Use the selected `rho_eq[idx]` values as the final pack grid `rho_torax`.

In symbols, if the native equilibrium grid has length $N_{\mathrm{eq},\rho}$, the selected indices are approximately

$$
i_j = \left\lfloor j\,\frac{N_{\mathrm{eq},\rho}-1}{64} \right\rfloor, \qquad j=0,\dots,64,
$$

and the stored grid is

$$
\rho_j^{\mathrm{torax}} = \rho_{\mathrm{eq}}[i_j].
$$

This gives a **non-uniform but equilibrium-consistent 1D flux grid**. It is not a plain `linspace(0,1,65)`. The spacing between neighboring points reflects the native equilibrium sampling, so geometric information from the equilibrium is retained in the radial mesh.

Grid spans `[0, 1]`:
```
rho: shape=(65,), min=0, max=1
```

`Vprime` is then interpolated onto this grid:
```python
Vprime_torax = np.interp(rho_torax, rho_eq, Vprime)
```

This is a **1D piecewise linear interpolation** in the flux coordinate:

$$
V'_{\mathrm{torax}}(\rho_j^{\mathrm{torax}})
=
\operatorname{interp}\left(
\rho_j^{\mathrm{torax}};
\rho_{\mathrm{eq}},
V'(\rho_{\mathrm{eq}})
\right).
$$

So there are two conceptually distinct interpolation layers in the equilibrium preprocessing:

1. **2D interpolation in $(R,Z)$** to evaluate flux and convert spatial points into the flux coordinate $\rho$.
2. **1D interpolation in $\rho$** to transfer geometric weights such as $V'(\rho)$ from the equilibrium-native radial mesh onto the final 65-point pack grid.

That separation is important. The first interpolation defines the coordinate system itself; the second interpolation transfers equilibrium-derived 1D geometry onto the modeling grid.

---

### Step 4: Thomson Scattering Profiles

#### 4a. Variable Discovery

```python
Te_da = get_var(ts, ["Te", "T_e", "te", "Te_eV", "t_e"])
ne_da = get_var(ts, ["ne", "n_e", "ne_cm3", "ne_m3"])
```

Units stored in the pack:
```
Te_units = "eV"
ne_units = (10-char string, likely "m^-3")
```

#### 4b. Radial Coordinate Path — Explicit Rho Coordinate

`infer_ts_radial_coordinate(ts)` is called, checking in order:
1. Explicit rho-like coord: `"rho"`, `"rho_ts"`, `"psi_N"`, `"psiN"`, `"psi_norm"`
2. Channel-based `R`/`Z` positions
3. `major_radius` coordinate
4. Single non-time dimension fallback

**Path taken for all shots:** explicit rho coordinate was found in `ts.coords`.
Evidence: `Te.shape=(111, 65)` matches `rho.shape=(65,)`, confirming that the Thomson data already arrives as profiles parameterized by a radial coordinate, rather than as independent channel locations that must be converted from $(R,Z)$ point-by-point.

```python
rho_ts = ts.coords[rho_coord_name].values   # shape (65,)

Te_ts = to_time_samples(Te_da)              # shape (111, 65)
ne_ts = to_time_samples(ne_da)              # shape (111, 65)

Te_rho_t, Te_mask = profiles_to_rho_grid(rho_ts, Te_ts, rho_torax)
ne_rho_t, ne_mask = profiles_to_rho_grid(rho_ts, ne_ts, rho_torax)
```

Here the logic is simpler than in the equilibrium step because the TS system is already providing a 1D radial coordinate per channel. No 2D equilibrium interpolation is needed in the actual path used for these packs.

`to_time_samples(...)` performs a pure array-layout operation:

1. Move the `time` dimension to axis 0.
2. Flatten the remaining non-time dimensions into a single sample axis.

So if the original Thomson variable is already shaped like

$$
T_e(t, s),
$$

where $s$ indexes radial channels, then `to_time_samples` produces an array

$$
\mathbf{T}^{\mathrm{TS}} \in \mathbb{R}^{N_t \times N_s}
$$

with rows corresponding to time and columns corresponding to source radial sample locations `rho_ts[s]`.

In the present dataset:
- $N_t = 111$ Thomson time slices,
- $N_s = 65$ Thomson radial samples.

So at each time $t_n$, the builder has a discrete profile

$$
\left\{\bigl(\rho^{\mathrm{TS}}_s,\, T_e(t_n, \rho^{\mathrm{TS}}_s)\bigr)\right\}_{s=1}^{65}
$$

and it wants to re-express that profile on the equilibrium-derived target grid

$$
\left\{\rho^{\mathrm{torax}}_j\right\}_{j=1}^{65}.
$$

The R/Z channel path (`to_time_samples_fill` + per-timestep `rho_fn` call) was **not used** for this dataset.

#### 4c. Profile Interpolation: `profiles_to_rho_grid`

This is the main **profile remapping stage** from diagnostic coordinates onto the modeling coordinates.

For each Thomson time slice $t_n$, the function `profiles_to_rho_grid(...)` does the following:

1. Start from the discrete source profile values

$$
v_s = T_e(t_n, \rho^{\mathrm{TS}}_s)
$$

or similarly for $n_e$.

2. Keep only entries where both the source coordinate and the source value are finite:

$$
\mathcal{I}_{\mathrm{valid}} = \{ s : \rho^{\mathrm{TS}}_s \text{ finite and } v_s \text{ finite} \}.
$$

3. Sort the valid samples by increasing source radius. This is required because 1D interpolation assumes an ordered independent variable.

4. Define a piecewise linear interpolant in the source radial coordinate and evaluate it on each target grid point $\rho^{\mathrm{torax}}_j$:

$$
v^{\mathrm{torax}}_j
=
\operatorname{interp}\left(
\rho^{\mathrm{torax}}_j;
\rho^{\mathrm{TS}}_{\mathrm{sorted}},
v_{\mathrm{sorted}}
\right).
$$

5. Apply **no extrapolation** beyond the convex hull of the valid TS source points. Concretely:

$$
\rho^{\mathrm{torax}}_j < \min \rho^{\mathrm{TS}}_{\mathrm{valid}}
\quad \text{or} \quad
\rho^{\mathrm{torax}}_j > \max \rho^{\mathrm{TS}}_{\mathrm{valid}}
\;
\Longrightarrow
\;
v^{\mathrm{torax}}_j = \mathrm{NaN}.
$$

So the interpolated profile is only trusted on the radial interval actually covered by valid TS measurements at that time.

6. Build a validity mask on the target grid. When no explicit mask is passed, the target mask is simply

$$
m^{\mathrm{torax}}_j = \mathbf{1}\{v^{\mathrm{torax}}_j \text{ is finite}\}.
$$

This means the target-grid mask is not an independent measurement quality metric; it is the indicator of whether interpolation produced a finite value at that $(t_n,\rho_j)$ location.

In words, the interpolation is:
- linear in the 1D TS radial coordinate,
- local between neighboring valid source points,
- and conservative in the sense that it does **not** invent profile values outside measured radial support.

This procedure is done independently for every time slice, so the valid radial interval can change with time as TS coverage changes.

**Result:**
```
Te:      shape=(111, 65), min=0, max=681.4 eV
ne:      shape=(111, 65), min=0, max=5.697e+19 m^-3
Te_mask: shape=(111, 65), dtype=bool
ne_mask: shape=(111, 65), dtype=bool
```

Coverage is low (~8–18% valid entries per shot), as TS diagnostics do not
cover all rho positions at all times. Core channels generally have better
coverage than the edge.

#### 4d. Final Masking and NaN Replacement

```python
Te_mask = (Te_mask > 0.5) & np.isfinite(Te_rho_t)
ne_mask = (ne_mask > 0.5) & np.isfinite(ne_rho_t)

Te_rho_t = np.nan_to_num(Te_rho_t, nan=0.0, posinf=0.0, neginf=0.0)
ne_rho_t = np.nan_to_num(ne_rho_t, nan=0.0, posinf=0.0, neginf=0.0)
```

**Important**: invalid entries are replaced with `0.0` in the saved arrays.
Always use `Te_mask` / `ne_mask` to identify real data.

---

### Step 5: Summary Signals

Time axis from summary:
```
t: shape=(2207,), range=[-0.095, 0.4565] s
```

Signals extracted and NaN-gaps filled via `interp_fill_1d` (linear
interpolation + edge carry):

| Key | Source names tried | Present |
|-----|--------------------|---------|
| `Ip` | `ip`, `Ip` | ✅ shape (2207,) |
| `nebar` | `line_average_n_e`, `ne_bar`, `nebar` | ✅ shape (2207,) |
| `P_nbi` | `power_nbi`, `P_NBI`, `pnbi` | ✅ shape (2207,) |
| `P_rad` | `power_radiated`, `P_rad`, `prad` | ✅ shape (2207,) |
| `ne_line` | `n_e_line`, `ne_line`, `line_average_n_e` | ✅ shape (2207,) |
| `W_tot` | `W_tot`, `stored_energy`, etc. | ❌ not found in summary.nc |
| `P_ohm` | `p_ohm`, `power_ohmic`, etc. | ❌ not found |
| `P_tot` | `p_tot`, `power_total`, etc. | ❌ not found |
| `H98` | `H98`, `h_factor_98y2`, etc. | ❌ not found |
| `q95` | `q95`, `q_95` | ❌ not found |
| `li` | `li`, `li_3`, etc. | ❌ not found |
| `beta_n` | `beta_n`, `beta_N`, etc. | ❌ not found |
| `B_t0` | `B_t0`, `b_t0`, etc. | ❌ not found |

Raw (pre-fill) copies are also saved:
- `P_nbi_raw` (2207,)
- `P_rad_raw` (2207,)

---

### Step 6: Particle Sources

#### S_gas (gas puffing)
Loaded from `gas_injection.nc` if present. The variable `total_injected` is
treated as cumulative: `rate = d(total_injected)/dt` via `np.gradient`.
Interpolated onto summary time `t`. Clipped to `>= 0`.
`S_gas: shape=(2207,)` — initialized to zeros if file missing or load fails.

#### D_alpha (explicit diagnostic signal)
The builder now prefers `d_alpha.nc`, which is a compact sidecar extracted from
the visible spectrometer group. If that file is absent, it falls back to
`spectrometer_visible.nc` and looks for `filter_spectrometer_dalpha_voltage`,
`d_alpha`, or `D_alpha`.

For the current MAST shots inspected here, the D-alpha dataset contains three
channels:

- `XIM_DA/HM10/R`
- `XIM_DA/HM10/T`
- `XIM_DA/HU10/T`

All channels are first interpolated onto the summary time grid `t`, then summed
to form a compact discharge-level D-alpha trace:

$$
D_{\alpha}(t_n) = \sum_c D_{\alpha,c}(t_n).
$$

Saved keys:

- `D_alpha`: shape `(2207,)`, summed/interpolated D-alpha on summary time
- `D_alpha_channels`: shape `(2207, 3)` for current shots
- `D_alpha_channel_names`: string array of channel labels

#### S_rec (recycling proxy)
`S_rec` is now the clipped recycling proxy derived from `D_alpha`:

$$
S_{\mathrm{rec}}(t) = \max(D_{\alpha}(t), 0).
$$

`S_rec: shape=(2207,)` — initialized to zeros if unavailable.

#### S_nbi (NBI fueling)
Computed analytically, assuming MAST beam energy of 75 keV:
```python
S_nbi = P_nbi / (75000.0 * 1.602e-19)   # [particles/s]
```
`S_nbi: shape=(2207,)`

---

### Step 7: Regime Labelling

A D-alpha-assisted heuristic labels each summary time point based on the
combination of:

- a **drop** in D-alpha,
- a **rise** in line-averaged density,
- and a **rise** in NBI power.

| Value | Meaning |
|-------|---------|
| 0 | Unknown |
| 1 | L-mode |
| 2 | Transition |
| 3 | H-mode |

The transition score is of the form

$$
	ext{score}(t)
=
1.2\,\max\!\left(-\frac{d\tilde D_{\alpha}}{dt},0\right)
+ 0.7\,\max\!\left(\frac{d\tilde n_e}{dt},0\right)
+ 0.3\,\max\!\left(\frac{d\widetilde P_{\mathrm{NBI}}}{dt},0\right),
$$

where each signal is normalized and smoothed first. The first and last `Nt/20`
points are masked out before the `argmax` is taken, which avoids the previous
edge-effect failure mode where the transition could collapse onto the start of
the shot.

A window of `Nt/20` around the detected peak is marked as transition; before is
L-mode, after is H-mode.

```
regime: shape=(2207,), dtype=int8
regime_score: shape=(2207,)
transition_time: shape=(1,)
```

**Note**: this is still a heuristic, not a validated confinement-database label,
but it tracks the D-alpha drop far better than the previous pure `nebar`/`P_nbi`
gradient score.

---

### Step 8: Save NPZ

Output: `data/<shot>_torax_training.npz`
Compression: `np.savez_compressed`
Schema version: `2`

---

## 3. Complete Key Reference

### Time Axes

| Key | Shape | Description |
|-----|-------|-------------|
| `t` | (2207,) | Summary time grid (seconds) |
| `t_ts` | (111,) | Thomson scattering time grid (seconds) |

### Radial Grid

| Key | Shape | Description |
|-----|-------|-------------|
| `rho` | (65,) | TORAX rho grid sampled from equilibrium, range [0, 1] |

### State Profiles (TS time × rho grid)

| Key | Shape | Description |
|-----|-------|-------------|
| `Te` | (111, 65) | Electron temperature [eV], 0 where invalid |
| `ne` | (111, 65) | Electron density [m⁻³], 0 where invalid |
| `Te_mask` | (111, 65) | Bool: True where Te is valid data |
| `ne_mask` | (111, 65) | Bool: True where ne is valid data |
| `Te_mask_col_cov` | (65,) | Per-rho fraction of valid Te time steps |
| `Te_mask_row_cov` | (111,) | Per-time fraction of valid Te rho points |
| `ne_mask_col_cov` | (65,) | Per-rho fraction of valid ne time steps |
| `ne_mask_row_cov` | (111,) | Per-time fraction of valid ne rho points |

### Coverage Scalars

| Key | Shape | Description |
|-----|-------|-------------|
| `Te_mask_mean` | scalar | Global fraction of valid Te entries |
| `Te_mask_mean_edge` | scalar | Fraction valid for rho >= 0.8 |
| `ne_mask_mean` | scalar | Global fraction of valid ne entries |
| `ne_mask_mean_edge` | scalar | Fraction valid for rho >= 0.8 |

### Summary / Control Signals (on summary time grid)

| Key | Shape | Description |
|-----|-------|-------------|
| `Ip` | (2207,) | Plasma current [A], NaN-filled |
| `nebar` | (2207,) | Line-averaged density [m⁻³], NaN-filled |
| `P_nbi` | (2207,) | NBI heating power [W], NaN-filled |
| `P_rad` | (2207,) | Radiated power [W], NaN-filled |
| `P_nbi_raw` | (2207,) | P_nbi before NaN-filling |
| `P_rad_raw` | (2207,) | P_rad before NaN-filling |
| `D_alpha` | (2207,) | Summed D-alpha signal on summary time grid |
| `D_alpha_channels` | (2207, Nc) | Per-channel D-alpha traces on summary time grid |
| `D_alpha_channel_names` | (Nc,) | Channel labels for `D_alpha_channels` |
| `ne_line` | (2207,) | Line-averaged electron density |
| `S_gas` | (2207,) | Gas injection rate [particles/s] |
| `S_rec` | (2207,) | Clipped recycling proxy from D-alpha |
| `S_nbi` | (2207,) | NBI fueling rate [particles/s] |
| `regime` | (2207,) | Confinement regime label (int8, 0–3) |
| `regime_score` | (2207,) | D-alpha-assisted transition score |
| `transition_time` | (1,) | Estimated transition time on summary grid |

### Geometry (scalars from equilibrium)

| Key | Description | Status |
|-----|-------------|--------|
| `R_major` | Major radius of magnetic axis [m] | ✅ present |
| `a_minor` | Minor radius [m] | ❌ `nan` (LCFS extraction failed) |
| `kappa` | Elongation | ❌ `nan` |
| `delta` | Triangularity | ❌ `nan` |
| `Vprime` | (65,) dV/drho on TORAX rho grid | ✅ present |

### Equilibrium Metadata

| Key | Description | Example (shot 27567) |
|-----|-------------|----------------------|
| `psi_axis` | Poloidal flux at magnetic axis [Wb/rad] | -0.19972 |
| `psi_edge` | Poloidal flux at plasma edge [Wb/rad] | 0.04495 |
| `rho_fallback_used` | Whether linear-R fallback was used | `False` |
| `rho_fallback_method` | Rho method used | `"psi"` |
| `rho_r_min` | Fallback r_min (unused) | `nan` |
| `rho_r_max` | Fallback r_max (unused) | `nan` |

### Units and Versioning

| Key | Value |
|-----|-------|
| `Te_units` | `"eV"` |
| `ne_units` | `"m^-3"` (10-char string) |
| `schema_version` | `3` |

---

## 4. Which Code Paths Were Taken

| Decision point | Path taken | Evidence |
|----------------|-----------|----------|
| Rho computation | **PSI-based** (`rho_from_RZ`) | `rho_fallback_used=False` all shots |
| TORAX rho grid | **Sampled from equilibrium** `rho_eq` | `Vprime` present, `rho_eq` available |
| TS radial coordinate | **Explicit rho coord** in `ts.coords` | `Te.shape=(111,65)` matches `rho.shape=(65,)` |
| TS interpolation helper | **`to_time_samples`** (rho-coord branch) | Rho coord found, R/Z path not needed |
| D-alpha source | **Preferred `d_alpha.nc`, fallback `spectrometer_visible.nc`** | Builder checks for `d_alpha.nc` first |
| Geometry extraction | **Partially failed** | `a_minor`, `kappa`, `delta` all `nan`; `R_major` succeeded |
| Volume derivatives | **Succeeded** | `Vprime.shape=(65,)` present |
| Optional summary signals | **Most absent** | Only `ne_line` and `P_rad` beyond core |
| Regime label path | **D-alpha-assisted score** | Transition timing follows D-alpha drop instead of edge gradients |

---

## 5. Important Notes for Downstream Use

1. **Always use masks**: `Te` and `ne` store `0.0` where data is invalid.
   Use `Te_mask` and `ne_mask` to identify real measurements.

2. **Two separate time grids**: `t` has 2207 points (summary), `t_ts` has 111
   points (TS). `Te` and `ne` live on `t_ts`; all controls live on `t`.
   Interpolate between them as needed.

3. **Geometry scalars are partial**: `a_minor`, `kappa`, `delta` are `nan` for
   all shots. Only `R_major` and `Vprime` are usable from geometry.

4. **Low TS coverage**: ~8–18% of `(time × rho)` entries are valid per shot.
   Core tends to have better coverage than edge (`rho >= 0.8`).

5. **D-alpha enters twice**: the raw/interpolated signal is saved as `D_alpha`
   for diagnostics and labeling, while `S_rec` remains the clipped proxy used by
   the current training controls.

6. **NBI fueling is approximate**: `S_nbi` assumes MAST beam energy of 75 keV.
   Adjust `E_beam_eV` in `build_training_pack.py` for other machines.

7. **Regime labels are heuristic**: `regime` is still a rough estimate, not
   validated against a confinement mode database, but the D-alpha-assisted score
   avoids the previous start-of-shot failure mode.

8. **Optional equilibrium-series scalars are shape-checked**: if arrays such as
   `q95` or `li` are found on a non-summary time base, they are dropped instead
   of being silently misaligned onto `t`.
