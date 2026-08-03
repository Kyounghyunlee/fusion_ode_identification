# Physics-Consistent Transport ROM (Uniform Grid)

**Status:** Active — implemented in `train_tokamak_ode_hpc.py` and `fusion_ode_identification/*`.  
**Pipeline:** Standalone training pipeline using TORAX-compatible pack format (see §3.0). TORAX is **not** required to run this ROM.  
**Scope of this document:** the physics, discretisation, loss, numerics, and forward-looking engineering plan. Code-level API documentation lives next to the sources.  
**Audience:** plasma physicists who want to know what is actually being solved; ML engineers who want to know why the model is shaped the way it is; and new contributors who need to trace a concept to a file and line.

### Reading order

- **If you only read three things:** §0.4 (key symbols), §1.3 (the one-paragraph justification of the hybrid PDE-ODE form), and §13 (what to work on next).
- **If you are debugging the solver:** §6.3.1 (tridiagonal), §9 (IMEX and safeguards), and §10 (diagnostics).
- **If you are writing a paper about the results:** §1–§7 and §14 (references).

### Table of contents

- [0. Scope and notation](#0-scope-and-notation)
- [1. Model architecture and why it is structured this way](#1-model-architecture-and-why-it-is-structured-this-way)
- [2. Pipeline overview (data → grid → BCs → geometry → bundles)](#2-pipeline-overview-data--grid--bcs--geometry--bundles)
- [3. Data, geometry, and TORAX relationship](#3-data-geometry-and-torax-relationship)
- [4. Governing model (continuous → semi-discrete)](#4-governing-model-continuous--semi-discrete)
- [5. Boundary conditions](#5-boundary-conditions-neumann-and-dirichlet-continuous-and-discrete)
- [6. Spatial discretisation](#6-spatial-discretisation-uniform-fvm-with-bc-insertion)
- [7. Loss terms and training objective](#7-loss-terms-and-training-objective-with-bc-consistent-supervision)
- [8. Why uniform grids](#8-why-uniform-grids-and-how-this-interacts-with-stiffness--bcs)
- [9. Numerical stability and solver choice](#9-numerical-stability-and-solver-choice-imex-with-bcs-built-in)
- [10. Logging and diagnostics](#10-logging-and-diagnostics-bc-aware-checks)
- [11. Configuration knobs](#11-configuration-knobs-grouped-matched-to-configconfigyaml)
- [12. Data inventory](#12-data-inventory)
- [13. Future improvement plan](#13-future-improvement-plan)
- [14. References](#14-references-updated)
- [Appendix A. Code map](#appendix-a-code-map)

---

## Executive summary (one page)

We learn a **physics-consistent neural ODE** reduced-order model (ROM) for tokamak **electron temperature transport** on a 1D flux coordinate $\rho\in[0,1]$. The ROM evolves the **physical profile state** directly, rather than a latent embedding of it:

- Physical state: $\mathbf{T}(t)\in\mathbb{R}^{N}$ (electron temperature on a uniform radial grid).
- Latent order parameter: $z(t)\in\mathbb{R}$ (a scalar regime coordinate that modulates edge diffusivity, L-mode ↔ H-mode).
- Inputs at each time $t$: density profile $\mathbf{n}_e(t)$ and a 6-dim actuator vector $\mathbf{u}(t)=[P_{\mathrm{nbi}},I_p,\bar n_e,S_{\mathrm{gas}},S_{\mathrm{rec}},S_{\mathrm{nbi}}]$.

The model combines three interpretable pieces:
1) A **conservative finite-volume diffusion operator** built from the flux form of the transport PDE, with an explicit, differentiable $\chi(\rho,z)$ (sigmoid pedestal profile centred at $\rho\!\approx\!0.85$) and experimentally supplied $V'(\rho)$;
2) A **learned residual source** $\mathbf{S}_\theta$ — a small tanh MLP, zero-initialised on the output layer, evaluated pointwise at each $\rho_i$ on normalised inputs $(\rho_i,T_i/\texttt{Te\_scale},n_{e,i}/\texttt{ne\_scale},\tilde{\mathbf{u}},z)$;
3) A low-dimensional latent ODE $\dot z=\alpha(\mu(\mathbf{u})-\mu_{\mathrm{ref}})-\beta z-\gamma z^3$ that is coupled to the profile only through $\chi_{\mathrm{edge}}(z)$.

**Why a hybrid PDE-ODE rather than a generic neural ODE:** the stiff part of tokamak core-edge transport is geometrically structured (divergence of a flux), while the closure terms (anomalous transport, edge physics, recycling) are the parts we do not know. Keeping the stiff operator interpretable and letting the residual be learned gives a model that is (a) identifiable from sparse Thomson data, (b) stable under long rollouts because its linear part is dissipative, and (c) inspectable — we can read off the learned diffusivity profile and source magnitude and compare them to physical expectations.

**Stiffness and time integration:** the discrete diffusion operator has $|\lambda_{\max}|\sim \chi/(\Delta\rho)^2$, so explicit RK would require $\Delta t=\mathcal{O}((\Delta\rho)^2)$. We instead use an **IMEX $\theta$-method** (Thomas-algorithm tridiagonal implicit diffusion, explicit residual source and latent). The solver runs a **fixed** number of substeps per Thomson interval (default `substeps=5`, `theta=0.7`) so that loop bounds are static and reverse-mode autodiff through the rollout is well-defined under `jit`/`vmap`.

**Boundary conditions are first-class:**
- Axis ($\rho=0$): **homogeneous Neumann / symmetry**, implemented as **zero left boundary face flux**, $F_{-1/2}=0$.
- Edge ($\rho=1$): **Dirichlet**, with $T_{\mathrm{edge}}(t)$ built from masked data. The solver state is interior-only — $y=[\hat{\mathbf{T}}_{\mathrm{int}},z]$ with $\hat{\mathbf{T}}=\mathbf{T}/\texttt{Te\_scale}$ — and the boundary value is reconstructed and appended at every RHS / implicit-matrix evaluation.

This document describes the full modelling and training pipeline, with special emphasis on **how Neumann and Dirichlet BCs enter the discrete operator and the semi-discrete ODE form**, and on the numerical choices that make the rollout differentiable and trainable end-to-end.

---

## 0. Scope and notation

### 0.1 Coordinates and grids
- Radial coordinate: $\rho\in[0,1]$ (normalised flux radius).
- Uniform ROM grid:
  $$
  \boldsymbol{\rho}_{\mathrm{rom}}=\mathrm{linspace}(0,1,N),\qquad
  \rho_i=\frac{i}{N-1},\ i=0,\dots,N-1,
  $$
  with constant spacing
  $$
  \Delta\rho=\rho_{i+1}-\rho_i=\frac{1}{N-1}.
  $$
- Observed indices:
  $$
  \mathcal{I}_{\mathrm{obs}}=\{0,1,\dots,N-2\}.
  $$
  The last node $i=N-1$ is reserved for the Dirichlet boundary and is excluded from the data loss.

### 0.2 State, inputs, and geometry
- State: electron temperature profile $\mathbf{T}(t)\in\mathbb{R}^{N}$; latent scalar $z(t)\in\mathbb{R}$.
- Density: $\mathbf{n}_e(t)\in\mathbb{R}^{N}$ (regridded onto $\boldsymbol{\rho}_{\mathrm{rom}}$).
- Actuators: $\mathbf{u}(t)\in\mathbb{R}^M$ (summary signals, interpolated to the profile time base as needed).
- Geometry: $V'(\rho)$ sampled on $\boldsymbol{\rho}_{\mathrm{rom}}$ when available; fallback $V'(\rho)=2\rho$ (toroidal approximation) with safe-core clamping.

### 0.3 Discrete conventions
Bold symbols are vectors sampled on $\boldsymbol{\rho}_{\mathrm{rom}}$. Norms are Euclidean unless stated. Expectations are empirical over batches and time.

### 0.4 Key symbols (glossary)

| Symbol | Meaning | Where it is set |
|---|---|---|
| $N$ | Number of ROM grid nodes (incl. boundary) | `data.uniform_n_rho` (defaults to NPZ `rho` length) |
| $\rho_i$ | $i/(N-1)$, uniform flux-radius grid | `data.py::load_data` |
| $T_i(t)$ | Electron temperature at node $i$ [eV] | `ShotBundle.ts_Te` |
| $T_{\mathrm{edge}}(t)$ | Dirichlet boundary trace [eV] | `ShotBundle.Te_edge` |
| $n_{e,i}(t)$ | Electron density at node $i$ [m$^{-3}$] | `ShotBundle.ne_vals` |
| $\mathbf u(t)$ | 6-dim actuator vector (raw, unscaled) | `ShotBundle.ctrl_vals` |
| $\tilde{\mathbf u}(t)$ | Per-shot z-scored actuators, clipped to $[-10,10]$ | `loss.py` |
| $V'_i$ | Volume derivative $dV/d\rho$ at node $i$ | `ShotBundle.Vprime_rom` |
| $\chi(\rho,z)$ | Heat diffusivity [m$^2$/s] | `HybridField._chi_profile` |
| $z(t)$ | Scalar latent regime coordinate | solver state `y[-1]` |
| $\hat T_i$ | Scaled temperature $T_i/\texttt{Te\_scale}$ | solver state `y[:-1]` |
| $\mathbf S_\theta$ | Learned residual source [eV/s] | `SourceNN` |
| $\texttt{Te\_scale}$ | $10^3$ eV, normalising constant | `HybridField.__init__` |
| $\texttt{ne\_scale}$ | $10^{19}$ m$^{-3}$, normalising constant | `HybridField.__init__` |
| $m_i(t)$ | Thomson validity mask at $(t,\rho_i)$ | `ShotBundle.mask` |
| $T_i^{\mathrm{raw}}(t)$ | NaN-preserving regridded $T_e$ for plotting/QA | `ShotBundle.ts_Te_raw` |
| $\mathcal I_{\mathrm{rel}}$ | Corpus-level reliable-annulus indicator | `ShotBundle.reliable_mask` |
| $\texttt{t\_len}$ | Per-shot number of valid time steps | `ShotBundle.t_len` |
| `active_mask` | Solver freeze mask for padded tail | passed to `IMEXIntegrator.integrate` |

---

## 1. Model architecture and why it is structured this way

### 1.1 The problem setting

We have $\mathcal{O}(20)$ MAST discharges with sparse, edge-localised Thomson scattering. On the stacked uniform ROM grid, more than half of the radial support is either unobserved or filled by interpolation, and the reliably measured annulus sits roughly at $\rho\in[0.8,0.97]$. The actuator traces (`P_nbi`, `Ip`, `nebar`, `S_gas`, `S_rec`, `S_nbi`) and coarse geometry ($V'$) are dense but not a closure — there is no accompanying turbulence model in the pack. Any learnt model therefore has to be **generous about inductive bias** (few parameters, lots of structure) and **strict about supervision** (only fit where Thomson is actually informative).

### 1.2 What we are *not* doing (and why)

**Transformers / sequence models on $\mathbf{T}_{t-k:t}$**: great when context is dense and interactions are long-range, but here the training signal is only a few hundred Thomson samples per shot, the stiff structure of transport is known, and we want to interpret what the model learnt. Attention on such data tends to memorise shot identity rather than transport.

**Generic latent world-models**: encode $\mathbf{T}$ into $\mathbf{h}$, evolve $\mathbf{h}$ black-box, decode. We lose physical interpretability (there is no longer a diffusivity or a flux to read off), and the stiff part of the problem — which is the part we already know analytically — becomes something the network has to rediscover from scratch.

**Modern SSMs (S4 / Mamba-like)**: efficient but the hidden state is abstract. Boundary conditions, flux conservation, and positivity of temperature all become soft, not structural.

### 1.3 What we do instead (physics-informed ODE on the physical state)

We model the evolution of the physical profile directly,
$$
\frac{d\mathbf{T}}{dt}=
\underbrace{\mathbf{D}(\mathbf{T};\chi(\rho,z),V')}_{\text{conservative diffusion (stiff, known)}}
+
\underbrace{\mathbf{S}_{\theta}(\boldsymbol{\rho},\mathbf{T},\mathbf{n}_e,\mathbf{u},z)}_{\text{learned residual (unknown)}},
\qquad
\frac{dz}{dt}=f_z(\mathbf{u},z),
$$
where $z$ modulates $\chi$ only near the plasma edge. This decomposition is the central design choice and it buys four things:

1. **Stability by construction.** The stiff operator is elliptic-dissipative for any $\chi>0$, so long rollouts cannot blow up through the linear part. The only way the model can diverge is through the bounded residual $\mathbf{S}_\theta$, and we actively penalise its magnitude.
2. **Identifiability on sparse data.** Because the diffusion piece is fixed in form, the MLP has far fewer ways to explain the data and the learning signal goes to the pieces we actually do not know.
3. **Interpretability.** We can plot the learnt $\chi(\rho,z)$ and $\mathbf{S}_\theta$ separately; we can compare core vs edge diffusivity, and we can ask whether the residual looks like a plausible heating/radiation term.
4. **A single scalar coordinate for the L–H regime.** The latent $z$ is explicitly wired to edge transport through $\chi_{\mathrm{edge}}(z)$, giving us one meaningful degree of freedom for the regime change rather than a dozen uninterpretable latents.

**Why this structure matters for analysis and control:** the state is $\mathbf{T}\in\mathbb{R}^N$, so linearisation, eigen-analysis, and critical-slowing-down diagnostics are all meaningful. Continuation/bifurcation in $z$ or in components of $\mathbf{u}$ is a well-defined operation. The same cannot be said of a 512-dim transformer hidden state.

---

## 2. Pipeline overview (data → grid → BCs → geometry → bundles)

This section describes the end-to-end pipeline as implemented in `fusion_ode_identification.data.load_data` and related modules, with explicit emphasis on BC construction.

### 2.1 Data and grid construction (uniform grid only)

**Runtime policy:** the codebase enforces `data.rho_grid_mode: "uniform"` (checked at runtime). All spatial operations use $\boldsymbol{\rho}_{\mathrm{rom}}=\mathrm{linspace}(0,1,N)$.

#### Step 1: Load and validate packs
- Read all `*_torax_training.npz` from `data.data_dir`.
- Validate rho monotonicity; reorder if needed.
- Drop invalid time slices (NaN/Inf).
- Extract time bases, profiles ($T_e,n_e$), masks, geometry, controls, optional scalars.
- If pack rho is not close to uniform $\mathrm{linspace}(0,1,N)$, log a warning (interpolation may be used).

#### Step 2: Construct the uniform ROM grid
$$
\boldsymbol{\rho}_{\mathrm{rom}}=\left[0,\frac{1}{N-1},\frac{2}{N-1},\dots,1\right].
$$
- No intersection thresholding.
- No Chebyshev clustering.
- The last node $\rho_{N-1}=1$ is reserved for Dirichlet BC.

#### Step 3: Define observed/supervised indices
$$
\mathcal{I}_{\mathrm{obs}}=\{0,1,\dots,N-2\}.
$$
- Use all interior nodes except the boundary: `obs_idx = arange(0, N-1)`.
- Boundary node excluded from data loss (Dirichlet BC is imposed, not fitted).

#### Step 4: Regrid profiles and masks
- Interpolate $T_e(t,\rho)$, $n_e(t,\rho)$, and masks to $\boldsymbol{\rho}_{\mathrm{rom}}$ when required.
- Use only finite/masked points per time slice; interpolate with constant extrapolation at boundaries (avoid injecting $T_e=0$ artefacts).
- Temporal gaps: forward-fill then backward-fill initial missing rows.

#### Step 5: Edge BC construction ($T_{\mathrm{edge}}(t)$)
Build $T_{\mathrm{edge}}(t)$ per shot:
- `use_last_observed` (default): take $T_e$ at the outermost masked-valid index at each time.
- `extrapolate_to_1`: linearly extrapolate from the last two masked-valid points to $\rho=1$ when possible; otherwise fall back to the outermost valid value.
- Interpolate linearly in time to fill gaps on the shot’s time grid.
- Absolute fallback (no valid edge anywhere on the shot): a constant $T_{\mathrm{edge}}=200$ eV.
- Final hard clip: $T_{\mathrm{edge}}(t)\in[5,5000]$ eV to guarantee a positive, finite Dirichlet trace.

#### Step 6: Geometry precomputation (one-time per shot)
Compute finite-volume arrays once:
- `dr`: face spacings $\Delta\rho_i=\rho_{i+1}-\rho_i$.
- `Vprime_face`: $V'_{i+1/2}=\frac{1}{2}(V'_i+V'_{i+1})$.
- `Vprime_cell`: a positive cell metric $\bar V'_i$ (implementation uses averaging + floors).
- `denom`: stable denominators $\max(\epsilon,\bar V'_i\,\Delta\rho)$.

These arrays are passed to the IMEX solver to avoid per-substep recomputation.

#### Step 7: Padding and stacking into `ShotBundle`
- Pad time arrays to global maximum length using `pad_time_to_max_strict` (ensure strict monotonicity after float conversions).
- Stack batched `ShotBundle` with per-shot `t_len` for loss masking.
- Padded regions are frozen via IMEX `active_mask` (no learning signal there).

#### Step 8: Initial condition and controls
- IC: first valid $T_e$ profile on $\boldsymbol{\rho}_{\mathrm{rom}}$; fallback synthetic $T(\rho)=100(1-\rho^2)+10$ eV if missing.
- Controls: z-scored and clipped signals (`P_nbi`, `Ip`, `nebar`, `S_gas`, `S_rec`, `S_nbi`) interpolated onto the profile time grid.
  `S_rec` is the D-alpha-derived recycling proxy; the raw/interpolated D-alpha signal is saved in the pack separately as `D_alpha` for mode-labeling and diagnostic analysis.

**What `D_alpha` stores in the pack:**
- Source: extracted from `d_alpha.nc` when present, otherwise from `spectrometer_visible.nc`.
- Physical meaning: a visible-spectrometer D-alpha emission trace, i.e. a line-integrated optical proxy for recycling/edge-neutral activity rather than a transport state variable.
- Representation in the pack:
  - `D_alpha`: 1D summed D-alpha time series on the summary/control grid `t`.
  - `D_alpha_channels`: 2D array with shape `(n_t, n_{\text{channels}})` containing the per-channel traces on the same time grid.
  - `D_alpha_channel_names`: string labels for the channel axis.
- Units: inherited from the underlying diagnostic and therefore best treated as relative amplitude/voltage-like units, not as an absolute particle source.

**How the model uses `D_alpha`:**
1. It is converted into `S_rec=\max(D_\alpha,0)`, and `S_rec` is the quantity that actually enters the control vector seen by the source network.
2. It is used offline during pack building to help construct the heuristic `regime` labels and `regime_score` that mark L-mode, transition, and H-mode periods.
3. The raw `D_alpha` trace is retained for diagnostics and later analysis, but the ODE does not evolve `D_alpha` as part of the model state.

**Mathematical role of `D_alpha` in this ROM:**
For a single shot, the pack stores per-channel traces $D_{\alpha,c}(t)$ and the summed signal
$$
D_\alpha(t)=\sum_{c=1}^{C} D_{\alpha,c}(t).
$$
So `D_alpha` is a scalar-valued function of time for each shot, not a single constant scalar for the whole discharge.

Within the learned source term, the model does **not** use raw `D_alpha` directly. Instead it uses the clipped recycling proxy
$$
S_{\mathrm{rec}}(t)=\max\big(D_\alpha(t),0\big),
$$
and the control vector is therefore
$$
\mathbf{u}(t)=\big[P_{\mathrm{nbi}}(t),\ I_p(t),\ \bar n_e(t),\ S_{\mathrm{gas}}(t),\ S_{\mathrm{rec}}(t),\ S_{\mathrm{nbi}}(t)\big].
$$
In other words, `D_alpha` influences the learned source term only through `S_rec`.

For regime labelling, the builder (`preprocessing.build_training_pack.estimate_regime_labels`) forms a smoothed transition score
$$
s(t)=
1.2\,\max\!\big(-\partial_t \widetilde D_\alpha(t),0\big)
+0.7\,\max\!\big(\partial_t \widetilde{\bar n}_e(t),0\big)
+0.3\,\max\!\big(\partial_t \widetilde P_{\mathrm{nbi}}(t),0\big),
$$
where tildes denote per-signal normalised ($\in[0,1]$) and boxcar-smoothed ($D_\alpha$ with $k=101$ samples; $\bar n_e,P_{\mathrm{nbi}}$ with $k=31$). Endpoint padding of $\max(10, N_t/20)$ samples is zeroed to avoid edge artefacts. The estimated transition index is then
$$
k_{LH}=\operatorname*{arg\,max}_k s(t_k).
$$
The resulting heuristic regime code is piecewise assigned on **sample indices** (not time units), with a half-width $w=\max(2,\lfloor N_t/20\rfloor)$:
$$
q(t_k)=
\begin{cases}
1, & k<k_{LH}-w, \\
2, & |k-k_{LH}|\leq w, \\
3, & k>k_{LH}+w,
\end{cases}
$$
with `1 = L-mode`, `2 = transition`, and `3 = H-mode`. If $s(t)$ never exceeds zero or the signals are missing, the builder returns $q\equiv 0$ (unknown) and the loss masks the shot out of the regime BCE. The limits of this heuristic are precisely what §13.2–13.3 argue should be replaced.

**Plasma-physics interpretation:**
The measured D-alpha brightness is a line-integrated edge/SOL optical signal,
$$
I_{D_\alpha}(t) \propto \int_{\mathrm{LOS}} \epsilon_{D_\alpha}(\mathbf{x},t)\,dl,
$$
with emissivity roughly driven by excitation and recombination of deuterium neutrals. A useful schematic form is
$$
\epsilon_{D_\alpha} \sim n_e n_0\langle\sigma v\rangle_{\mathrm{exc}} + \epsilon_{\mathrm{recomb}}.
$$
In L-mode, stronger edge transport and recycling often produce a relatively elevated D-alpha level. At the L-to-H transition, an edge transport barrier forms, particle transport to the wall/divertor is reduced, and the recycling light often drops sharply. That empirical “D-alpha drop” is why a negative slope of $D_\alpha(t)$ is commonly used as a transition marker. In H-mode, however, ELMs and divertor dynamics can still produce bursts on top of the lower baseline, so `D_alpha` should be treated as a useful proxy rather than a perfect confinement-state observable.

---

## 3. Data, geometry, and TORAX relationship

### 3.0 TORAX relationship and format compatibility

**TORAX** is a differentiable tokamak transport simulator in JAX. This ROM pipeline is **standalone**:
- Packs are named `*_torax_training.npz` because their layout is compatible (flux coordinates, geometry, uniform grids).
- Training and inference do not require TORAX.

**Conceptual difference:**
- TORAX: forward simulation with physics closure models (turbulence, sources, multi-channel coupling).
- This ROM: learns $\chi(\rho,z)$ and a residual source from sparse experimental data.

### 3.1 Equilibrium and fallbacks
- If equilibrium is present: regrid $V'(\rho)$ to $\boldsymbol{\rho}_{\mathrm{rom}}$.
- If missing/unreliable: fallback $V'(\rho)=2\rho$ and apply safe-core floors so $V'(\rho)$ stays strictly positive in the discrete denominators.

### 3.2 Quasi-static geometry (why we hold $V'(\rho)$ fixed in time)

We treat $V'(\rho)$ as **quasi-static** within each discharge window. Write
$$
V'(\rho,t)=V_0'(\rho)+\varepsilon\,\delta V'(\rho,t),\qquad \varepsilon\ll 1.
$$
Then the transport operator is
$$
\mathcal{L}[T;V'(\cdot,t)]=\mathcal{L}[T;V_0']+\varepsilon\,\Delta\mathcal{L}[T;\delta V',V_0']+O(\varepsilon^2).
$$
We keep $\mathcal{L}[T;V_0']$ and absorb the small corrections into the residual source (and/or mild changes in learned $\chi$). This improves robustness (avoids EFIT jitter injecting rough time-dependence into the stiff diffusion term) and improves identifiability (prevents degeneracy between $V'$, $\chi$, and $S_{\text{net}}$).

---

## 4. Governing model (continuous → semi-discrete)

### 4.1 Continuous framing (PDE)

We start from
$$
\frac{\partial T_e}{\partial t}
=
\frac{1}{V'(\rho)}
\frac{\partial}{\partial \rho}
\left(
V'(\rho)\,\chi(\rho,z)\,\frac{\partial T_e}{\partial\rho}
\right)
+
\mathcal{S}_{\text{net}}(\rho,T_e,n_e,\mathbf{u},z).
$$

### 4.2 Nemytskii residual source (pointwise operator)

The learned source acts pointwise in space:
$$
(\mathbf{S}_{\text{net}})_i
=
R_\theta(\rho_i,T_i,n_{e,i},\mathbf{u},z),
$$
where $R_\theta$ is a small MLP (tanh activations, zero-initialised output layer for stability).

### 4.3 Latent dynamics and coupling to diffusivity

We evolve a scalar latent
$$
\frac{dz}{dt}
=
\underbrace{\alpha_{+}}_{\mathrm{softplus}(\alpha)}\bigl(\mu(\mathbf{u})-\mu_{\text{ref}}\bigr)
-\underbrace{\beta_{+}}_{\mathrm{softplus}(\beta)} z
-\underbrace{\gamma_{+}}_{\mathrm{softplus}(\gamma)} z^3,
$$
where the positive-coefficient reparametrisation via `softplus` guarantees damping ($\beta_+,\gamma_+>0$) regardless of the raw parameter values, and
$$
\mu(\mathbf{u})=\mathbf{w}_\mu^\top\tilde{\mathbf{u}}_{1:3}+b_\mu
$$
is a learned affine map of the first three normalised controls (`P_nbi`, `Ip`, `nebar`). This is a saddle-node-bifurcation-capable reduced form in $z$ for fixed $\mathbf{u}$: for sufficiently large forcing $\mu$, the high-$z$ fixed point becomes reachable.

The latent couples back to the profile through a **differentiable sigmoid pedestal** (constructors `ped_center=0.85`, `ped_width=0.08` in `HybridField`):
$$
\chi_{\text{edge}}(z)=\operatorname{clip}\bigl(\chi_{\text{edge,base}}-\chi_{\text{edge,drop}}\,\sigma(g\,z),\ 0.1,\ 5.0\bigr),
\qquad
\chi(\rho,z)=\chi_{\text{core}}+w_{\mathrm{ped}}(\rho)\bigl(\chi_{\text{edge}}(z)-\chi_{\text{core}}\bigr),
$$
with $w_{\mathrm{ped}}(\rho)=\sigma\bigl((\rho-\texttt{ped\_center})/\texttt{ped\_width}\bigr)$ and $g=\texttt{latent\_gain}$. The post-clip on $\chi_{\mathrm{edge}}$ prevents the edge diffusivity from collapsing to zero (which would make the implicit solve near-singular) or exploding (which would make it numerically trivial).

**Numerical safeguards in the RHS** (see `HybridField.compute_rhs_components`): before forming the tendency, the interior state is smooth-clamped into $T_e\in[0,5000]$ eV and $z\in[-10,10]$; the raw diffusion divergence is `softclip`-ped at $\pm\texttt{divergence\_clip}=10^6$; and the combined $\mathbf{D}+\mathbf{S}$ is `softclip`-ped at $\pm 10^4$ before scaling by $1/\texttt{Te\_scale}$. These are gradient-preserving saturations, not hard cutoffs.

### 4.4 Semi-discrete hybrid ODE (what is integrated)

The semi-discrete system is
$$
\frac{d\mathbf{T}}{dt}
=
\mathbf{D}(\boldsymbol{\rho}_{\mathrm{rom}},V',\chi(\cdot,z))\,\mathbf{T}
+
\mathbf{S}_{\text{net}}(\boldsymbol{\rho}_{\mathrm{rom}},\mathbf{T},\mathbf{n}_e,\mathbf{u},z),
$$
$$
\frac{dz}{dt}=f_z(\mathbf{u},z).
$$

**Important implementation detail:** the solver state is interior-only plus latent:
- Solver state: $y=[\hat{\mathbf{T}}_{\text{int}},z]$ where $\hat{\mathbf{T}}=\mathbf{T}/\texttt{Te\_scale}$.
- The boundary value $T_{\text{edge}}(t)$ is appended at each RHS evaluation.

---

## 5. Boundary conditions (Neumann and Dirichlet, continuous and discrete)

This is the BC section you can treat as the “ground truth” reference.

### 5.1 Neumann at axis ($\rho=0$): symmetry / zero flux

**Rationale.** Homogeneous Neumann at the magnetic axis is not a modelling choice but a consequence of the coordinate system: $\rho=0$ is a regular interior point of the flux-surface geometry, so $\partial T_e/\partial\rho(0,t)$ must vanish by axisymmetry. In conservative form, the geometric factor $V'(\rho)\sim\rho$ already drives the flux to zero at the axis, so enforcing $F(0,t)=0$ is both physically correct and numerically the cleanest way to prevent division by $V'(0)=0$. Any non-homogeneous condition at the axis would inject or remove energy in a way inconsistent with the closed-flux-surface geometry.

**Continuous statement:**
$$
\frac{\partial T_e}{\partial \rho}(0,t)=0.
$$
Define the conductive flux
$$
F(\rho,t)=-V'(\rho)\chi(\rho,z)\,\frac{\partial T_e}{\partial \rho}(\rho,t).
$$
Then homogeneous Neumann is equivalent to
$$
F(0,t)=0.
$$

**Physical context:** the magnetic axis is a symmetry point; the radial derivative must vanish there. In conservative form, that is “no flux through the axis”.

**Discrete realisation (flux form, no ghost unknown required):**
We impose the left boundary face flux
$$
F_{-1/2}:=0.
$$
This is the clean conservative enforcement and matches your discretisation that uses face fluxes $F_{i+1/2}$.

**Ghost-node equivalence (why people talk about it):**  
If one instead enforces Neumann using a centred derivative at the boundary:
$$
\frac{\partial T}{\partial\rho}(0,t)\approx\frac{T_1-T_{-1}}{2\Delta\rho}=0
\quad\Rightarrow\quad T_{-1}=T_1,
$$
then substituting into a second-difference at $i=0$ yields the familiar doubling of the neighbour contribution. Both formulations are equivalent at the discrete level; in our code we implement the flux condition $F_{-1/2}=0$ directly.

### 5.2 Dirichlet at edge ($\rho=1$): imposed boundary trace

**Rationale.** Unlike the axis, the edge is a modelling choice. At $\rho=1$ the closed-flux region meets the scrape-off layer, whose physics (parallel transport, sheath, divertor recycling, impurity radiation) is out of scope for a core-transport ROM. We therefore *do not model* the edge and instead impose the measured $T_{\mathrm{edge}}(t)$ as a Dirichlet trace. This is the standard reduced-transport posture (see e.g. TRANSP/ASTRA/TORAX when SOL modelling is disabled) and it buys two concrete advantages:

- **Identifiability.** The learnt $\chi(\rho,z)$ and residual $\mathbf S_\theta$ describe *core* transport *given* the observed edge state; they cannot absorb SOL uncertainty.
- **Regime effects enter through forcing, not dynamics.** Pedestal formation, ELMs, and fast edge events show up as structure in $T_{\mathrm{edge}}(t)$ and are injected into the interior through the boundary forcing $\mathbf{b}_{\mathrm{edge}}(t)$, not through an extra model component.

**Trade-off.** This approach cannot predict $T_{\mathrm{edge}}(t)$ itself — it must be supplied. For closed-loop scenario prediction, one would need a companion edge model $T_{\mathrm{edge}}=g_\phi(\mathbf u,z)$ (see §13.7, milestone M7).

**Continuous statement:**
$$
T_e(1,t)=T_{\text{edge}}(t).
$$

**Physical context:** we treat SOL/divertor coupling as an externally supplied edge temperature trace rather than modelling full open-field-line physics. This is a standard ROM boundary closure when edge physics is not explicitly simulated.

**Discrete realisation (state reduction):**
We set
$$
T_{N-1}(t)\equiv T_{\text{edge}}(t)
$$
and evolve only interior nodes $0,\dots,N-2$.

**Key consequence:** the diffusion operator on interior nodes becomes **affine** in $\mathbf{T}_{\text{int}}$ (linear plus boundary forcing), which is exactly why the boundary node is excluded from the loss.

---

## 6. Spatial discretisation (uniform FVM) with BC insertion

This section provides the exact “ODE form” and shows how BCs change the matrix representation.

### 6.1 Face fluxes

For node indices $i=0,\dots,N-1$ define face-averaged coefficients
$$
\chi_{i+1/2}=\frac{\chi_i+\chi_{i+1}}{2},
\qquad
V'_{i+1/2}=\frac{V'_i+V'_{i+1}}{2}.
$$
Define the geometry-weighted face coefficient
$$
K_{i+1/2}=V'_{i+1/2}\chi_{i+1/2}.
$$
Define the discrete conductive flux through face $i+1/2$ (between nodes $i$ and $i+1$):
$$
F_{i+1/2}=-K_{i+1/2}\frac{T_{i+1}-T_i}{\Delta\rho_i},
\qquad \Delta\rho_i=\rho_{i+1}-\rho_i.
$$

### 6.2 Conservative divergence (interior update)

Let $\bar V'_i$ denote the positive cell metric used in denominators (in code: an averaged + floored version to avoid the axis singularity). The conservative divergence on interior indices $i=0,\dots,N-2$ is
$$
(\nabla\cdot F)_i
=
-\frac{F_{i+1/2}-F_{i-1/2}}{\bar V'_i\,\Delta\rho_i}.
$$

### 6.3 BC insertion in flux form

**Axis Neumann:**
$$
F_{-1/2}=0.
$$
So for $i=0$ the divergence uses only $F_{1/2}$:
$$
(\nabla\cdot F)_0=-\frac{F_{1/2}-F_{-1/2}}{\bar V'_0\,\Delta\rho_0}
=-\frac{F_{1/2}}{\bar V'_0\,\Delta\rho_0}.
$$

**Edge Dirichlet:**
$$
T_{N-1}(t)=T_{\text{edge}}(t),
$$
so the last interior face flux
$$
F_{N-3/2}=-K_{N-3/2}\frac{T_{N-1}(t)-T_{N-2}}{\Delta\rho_{N-2}}
$$
injects the boundary trace into the last interior divergence equation at $i=N-2$.

### 6.3.1 Explicit tridiagonal form used in the implicit solve

Letting
$$
k_{i+1/2}=\frac{K_{i+1/2}}{\Delta\rho_i}=\frac{V'_{i+1/2}\chi_{i+1/2}}{\Delta\rho_i},
\qquad
d_i=\bar V'_i\,\Delta\rho_i,
$$
the spatial operator $L$ (so that $\partial_t\mathbf{T}_{\mathrm{int}}=L\mathbf{T}_{\mathrm{int}}+\mathbf{b}_{\mathrm{edge}}+\mathbf{S}$) has tridiagonal entries on interior unknown indices $i=0,\dots,N-2$:
$$
(L)_{i,i-1}=\frac{k_{i-1/2}}{d_i},\qquad
(L)_{i,i}=-\frac{k_{i-1/2}+k_{i+1/2}}{d_i},\qquad
(L)_{i,i+1}=\frac{k_{i+1/2}}{d_i},
$$
with $k_{-1/2}\equiv 0$ at the axis (Neumann) and with $(L)_{N-2,N-1}$ omitted (boundary is not an unknown). The boundary forcing vector satisfies
$$
(\mathbf{b}_{\mathrm{edge}})_i=
\begin{cases}
\dfrac{k_{N-3/2}}{d_{N-2}}\,T_{\mathrm{edge}}(t), & i=N-2,\\[3pt]
0, & \text{otherwise.}
\end{cases}
$$

The implicit solve for $(I-\theta\Delta t\,L)\mathbf{T}_{\mathrm{int}}^{n+1}=\mathbf{r}$ is performed with the Thomas algorithm on the three vectors
$a_i=-\theta\Delta t\,(L)_{i,i-1}$, $b_i=1-\theta\Delta t\,(L)_{i,i}$, $c_i=-\theta\Delta t\,(L)_{i,i+1}$,
with $a_0=c_{N-2}=0$ enforced explicitly. See `build_diffusion_solve_tridiag_implicit` for the reference implementation; the denominator $d_i$ is floored at $\max(10^{-4}\max_i d_i,10^{-10})$ to prevent axis-cell singularities, and $\Delta\rho$ is floored analogously.

**Linearity requirement:** this construction assumes $L$ is linear in $\mathbf{T}$, which is why `chi` is taken to depend only on $(\rho,z)$ and **not** on $T$. Any future $T$-dependent $\chi$ breaks the linear implicit split and would require either a Newton iteration per step or moving $\chi(T)$ into the explicit part with a correspondingly smaller $\Delta t$.

### 6.4 Semi-discrete ODE: the affine form used by the solver

Putting §6.3.1 together, the semi-discrete ROM actually integrated by the IMEX solver is
$$
\boxed{\;
\frac{d\mathbf{T}_{\mathrm{int}}}{dt}
=
A\bigl(\chi(\cdot,z),V'\bigr)\,\mathbf{T}_{\mathrm{int}}
+
\mathbf{b}_{\mathrm{edge}}\bigl(t;\chi(\cdot,z),V'\bigr)
+
\mathbf{S}_{\mathrm{net,int}}\bigl(\boldsymbol{\rho},\mathbf{T},\mathbf{n}_e,\mathbf{u},z\bigr),
\qquad
\frac{dz}{dt}=f_z(\mathbf{u},z).
\;}
$$
The diffusion part is **affine** in $\mathbf{T}_{\mathrm{int}}$, with $A$ tridiagonal (§6.3.1) and $\mathbf{b}_{\mathrm{edge}}$ nonzero only in the last interior row. IMEX treats $A\mathbf{T}_{\mathrm{int}}+\mathbf{b}_{\mathrm{edge}}$ implicitly, and the residual $\mathbf{S}_{\mathrm{net,int}}$ and latent $f_z$ explicitly.

Two consequences worth noting:

- **The boundary trace is a forcing, not a state.** $T_{\mathrm{edge}}(t)$ enters exclusively through $\mathbf b_{\mathrm{edge}}$; the solver never carries it as a degree of freedom. This is what lets us safely exclude node $N-1$ from the data loss (§7.2).
- **Spectrum of $A$ is strictly dissipative.** $A$ is weakly diagonally dominant with negative diagonal and non-negative off-diagonals; all eigenvalues have non-positive real part. Together with $\chi>0$ (enforced by the $[0.1,5.0]$ clip on $\chi_{\mathrm{edge}}$), this guarantees the linear part cannot destabilise a rollout.

### 6.5 Stability floors used in code (BC-adjacent numerics)
- $\Delta\rho_i$ floored to avoid dividing by tiny spacings.
- $\bar V'_i\Delta\rho_i$ floored near the core (small-volume cells).
- $V'(\rho)$ clipped positive; safe-core floor applied so $V'(0)$ is not singular.
These floors are practical safeguards that prevent the implicit diffusion matrix from becoming ill-conditioned at the axis.

---

## 7. Loss terms and training objective (with BC-consistent supervision)

### 7.1 Composite objective
$$
\mathcal{L}_{\text{total}}
=
\mathcal{L}_{\text{data}}
+
\lambda_{\text{src}}\mathcal{L}_{\text{src}}
+
\lambda_w\mathcal{L}_{\text{model}}
+
\lambda_z\mathcal{L}_{z\text{-smooth}}
+
\lambda_{\text{regime}}\mathcal{L}_{z\text{-regime}}
+
\lambda_{z\text{reg}}\mathcal{L}_{z\text{-reg}}.
$$

### 7.2 Data term (pseudo-Huber on all interior masked radii)
We supervise **only** the interior indices $i=0,\dots,N-2$:
$$
\mathcal{L}_{\text{data}}
=
\frac{\sum_{t}\sum_{i=0}^{N-2}
m_i(t)\,w_i\,\phi_\delta\!\left(T^{\text{model}}_i(t)-T^{\text{obs}}_i(t)\right)}
{\sum_{t}\sum_{i=0}^{N-2} m_i(t)\,w_i + 10^{-8}}.
$$
- $m_i(t)\in\{0,1\}$ is the mask (valid point + within `t_len` window),
- $w_i$ are inverse-coverage normalised weights:
  $$
  w_i=\frac{(1/c_i)}{\sum_j (1/c_j)},\qquad c_i=\sum_t m_i(t),
  $$
- pseudo-Huber:
  $$
  \phi_\delta(x)=\delta^2\left(\sqrt{1+(x/\delta)^2}-1\right).
  $$

**Boundary exclusion is essential:** the node $i=N-1$ is Dirichlet-imposed and is therefore excluded from the data loss.

**Padded-time masking:** shots are stacked to a common $T_{\max}$ and each shot carries its own `t_len`. The mask $m_i(t)$ is the product of the per-point Thomson mask with $\mathbf{1}[t<\texttt{t\_len}]$, and the IMEX integrator additionally receives an `active_mask` that **freezes the state** outside the valid window (it advances the save-time cursor but keeps $y$ fixed). This ensures the padded tail contributes neither gradient to the parameters nor spurious dynamics to the solver.

### 7.3 Source magnitude penalty (pseudo-Huber, not L2)

Contrary to a naive $L^2$ penalty, the code uses a **pseudo-Huber** saturating penalty on the raw source values $(\mathbf{S}_{\mathrm{net}})_i$ with scale `src_delta`:
$$
\mathcal{L}_{\text{src}}
=
\frac{1}{N_{\mathrm{int}}\sum_t\mathbf{1}[t<\texttt{t\_len}]+\varepsilon}
\sum_{t,i}\mathbf{1}[t<\texttt{t\_len}]\,
\phi_{\delta_{\mathrm{src}}}\!\bigl((\mathbf{S}_{\mathrm{net}})_i\bigr),
\qquad
\phi_\delta(x)=\delta^2(\sqrt{1+(x/\delta)^2}-1).
$$

The choice of pseudo-Huber over $L^2$ matters during training: large transient source spikes early in training (for example when the solver explores regions where $\chi$ has not yet settled) do not generate unbounded quadratic gradient signals, so the network is not kicked out of a reasonable basin. Asymptotically $\phi_\delta(x)\sim \tfrac12 x^2$ for $|x|\ll\delta$ and $\phi_\delta(x)\sim\delta|x|-\tfrac12\delta^2$ for $|x|\gg\delta$, so the penalty is quadratic where the source is small and linear where the source is large.

This is the same penalty shape used for $\mathcal{L}_{\text{data}}$; only the scale (`huber_delta` versus `src_delta`) and the target residual (profile error versus raw source magnitude) differ.

### 7.4 Weak-constraint model error (optional)
$$
\mathcal{L}_{\text{model}}
=
\frac{1}{K-1}\sum_{k=1}^{K-1}\phi_{\delta_w}\left(T_{k+1}-\left(T_k+\Delta t_k f_k\right)\right),
$$
with $f_k$ the RHS at step $k$. Disabled by default unless `training.lambda_w>0`.

### 7.5 Latent regularisers (smoothness, optional supervision, magnitude)
- Smoothness:
  $$
  \mathcal{L}_{z\text{-smooth}}=\frac{1}{K-1}\sum_{k=1}^{K-1}(z_{k+1}-z_k)^2.
  $$
- Optional regime supervision (now driven by the D-alpha-assisted `regime` labels in the pack):
  If $q_k\in\{0,1,2,3\}$ denotes the packed regime code at time index $k$, the implementation uses
  $$
  m_k^{\mathrm{reg}}=\mathbf{1}[q_k\in\{1,3\}],\qquad r_k=\mathbf{1}[q_k=3],
  $$
  so that only confident L-mode and H-mode windows supervise the latent state, while transition and unknown windows are masked out.
  $$
  \mathcal{L}_{z\text{-regime}}=\frac{1}{K}\sum_{k=1}^{K}\operatorname{BCEWithLogits}(g\,z_k,r_k),\qquad r_k\in\{0,1\},
  $$
  where $g$ is the latent gain and $r_k=1$ denotes H-mode while $r_k=0$ denotes L-mode.
- Magnitude:
  $$
  \mathcal{L}_{z\text{-reg}}=\mathbb{E}[z^2].
  $$

---

## 8. Why uniform grids (and how this interacts with stiffness + BCs)

**Pros:**
1. Simplicity and robustness (no thresholding, no clustering).
2. All-radii supervision (interior only) with inverse-coverage weights.
3. Predictable stiffness scaling: $|\lambda_{\max}|\sim \chi/(\Delta\rho)^2$ with constant $\Delta\rho$.
4. Easy visualisation and comparison across shots.
5. TORAX-aligned conventions for possible future export.

**Trade-offs:**
- No extra pedestal resolution unless $N$ is increased.
- Stiffness scales like $N^2$ (handled by IMEX).
- Geometry/coefficients, not grid clustering, become the primary stiffness sources.

**BC interaction:** uniform grids simplify BC implementation because:
- Neumann at axis is always the same “left flux = 0” rule.
- Dirichlet at edge is always “last node fixed”; consistent across shots.

---

## 9. Numerical stability and solver choice (IMEX, with BCs built in)

### 9.1 Stiffness source (diffusion operator scaling)
For diffusion on a uniform grid, eigenvalues satisfy roughly
$$
|\lambda_{\max}|\sim \frac{\chi_{\max}}{(\Delta\rho)^2},
\qquad
\tau_{\min}\sim \frac{(\Delta\rho)^2}{\chi_{\max}}.
$$
So explicit RK would require $\Delta t\lesssim C(\Delta\rho)^2/\chi_{\max}$ for stability, even if the solution evolves on much slower diagnostic timescales.

### 9.2 IMEX $\theta$-method (our implementation)
We split
$$
\frac{d\mathbf{T}_{\text{int}}}{dt}
=
\underbrace{A\,\mathbf{T}_{\text{int}}+\mathbf{b}_{\text{edge}}(t)}_{\text{stiff diffusion + Dirichlet forcing}}
+
\underbrace{\mathbf{S}_{\text{net,int}}(\cdot)}_{\text{explicit residual}}.
$$
The IMEX $\theta$ update on a substep of size $\Delta t$ is
$$
\frac{\mathbf{T}_{\text{int}}^{n+1}-\mathbf{T}_{\text{int}}^{n}}{\Delta t}
=
\theta\left(A^{n+1}\mathbf{T}_{\text{int}}^{n+1}+\mathbf{b}_{\text{edge}}^{n+1}\right)
+
(1-\theta)\left(A^{n}\mathbf{T}_{\text{int}}^{n}+\mathbf{b}_{\text{edge}}^{n}\right)
+
\mathbf{S}_{\text{net,int}}(\mathbf{T}^n,\mathbf{u}^n,z^n).
$$
Rearranged:
$$
\left(I-\theta\Delta t\,A^{n+1}\right)\mathbf{T}_{\text{int}}^{n+1}
=
\mathbf{T}_{\text{int}}^{n}
+\Delta t(1-\theta)\left(A^{n}\mathbf{T}_{\text{int}}^{n}+\mathbf{b}_{\text{edge}}^{n}\right)
+\Delta t\,\mathbf{S}_{\text{net,int}}(\cdot)
+\Delta t\,\theta\,\mathbf{b}_{\text{edge}}^{n+1}.
$$

**BCs are handled naturally:**
- Neumann: already baked into how $A$ is assembled (via $F_{-1/2}=0$).
- Dirichlet: enters as the forcing $\mathbf{b}_{\text{edge}}(t)$ and through the last-face coefficient; no extra DOF.

### 9.3 Determinism and training convenience

- Each Thomson interval $[t_n,t_{n+1}]$ is subdivided into a **fixed** number of substeps (`training.imex.substeps`, default 5), giving per-substep step size $\Delta t=(t_{n+1}-t_n)/\texttt{substeps}$. `dt_base` is therefore unused in the current fixed-substep mode; it is retained as a placeholder for future adaptive stepping.
- Within each interval, $T_{\mathrm{edge}}$, the normalised controls $\tilde{\mathbf{u}}$, and $n_e$ are **linearly blended** between the endpoint samples rather than re-interpolated by `searchsorted` per substep. This keeps the inner loop cheap and JIT-friendly.
- Reverse-mode autodiff is well-defined because the loop is a static `fori_loop` over `substeps`, wrapped in a `scan` over the Thomson intervals. A `lax.cond` on `active_mask` freezes the state past `t_len-1` so padded tails contribute no gradient.
- Debug/eval integrates on the valid window only.

### 9.4 Numerical safeguards baked into the forward pass

The solver and model include several **soft** safeguards so that gradients stay finite even when parameters explore poor regions:

- `smooth_clamp` on $\hat T$ into $[0,5000/\texttt{Te\_scale}]$ eV and on $z$ into $[-10,10]$ (softplus-based, gradient-preserving).
- `softclip` on the raw divergence at $\pm10^6$ (`divergence_clip`) and on `div+src` at $\pm10^4$ inside `compute_rhs_components`. These limit how explosively the explicit part can grow while still letting the model learn smooth corrections.
- Hard clips on $n_e\in[10^{17},10^{21}]$ m$^{-3}$ and on normalised controls at $\pm10$.
- $\chi_{\mathrm{edge}}$ post-clipped to $[0.1,5.0]$ m$^2$/s so that $\chi$ never becomes non-positive or numerically enormous regardless of $z$.

These are **numerical choices, not physical statements**. They should be audited (via `dr_floor_hit`, `vprime_floor_hit`, `denom_floor_hit` diagnostics emitted per shot) and tightened once the model trains stably without hitting them.

---

## 10. Logging and diagnostics (BC-aware checks)

### 10.1 Per-shot diagnostics already emitted by `shot_loss_imex`

Every call to `shot_loss_imex` returns a fixed-order `diag` vector (see `fusion_ode_identification/loss.py`) which is logged batch-averaged during training and per-shot during evaluation:

| Index | Name | Meaning |
|---|---|---|
| 0 | `shot_id` | discharge number |
| 1 | `t_len` | number of valid time samples for this shot |
| 2 | `dt_base` | integrator `dt_base` (informational; unused) |
| 3 | `mae_eV` | mask-weighted mean absolute $T_e$ error [eV] |
| 4 | `mae_pct` | mask-weighted MAE normalised by $\max(\lvert T_e^{\mathrm{obs}}\rvert,50)$ [%] |
| 5 | `mean_abs_div` | time-masked mean $\lvert(\nabla\cdot F)_i\rvert$ |
| 6 | `mean_abs_src` | time-masked mean $\lvert (\mathbf S_{\mathrm{net}})_i\rvert$ |
| 7 | `src_over_diff` | ratio `mean_abs_src / mean_abs_div` (sanity check: residual should not dominate diffusion) |
| 8 | `min_dr` | smallest $\Delta\rho_i$ observed for this shot |
| 9 | `min_Vprime` | smallest $V'_i$ observed for this shot |
| 10 | `min_denom` | smallest cell-volume denominator $d_i=\bar V'_i\Delta\rho_i$ |
| 11 | `dr_floor_hit` | 1 if any $\Delta\rho_i$ hit the floor, else 0 |
| 12 | `vprime_floor_hit` | 1 if any $V'_i$ hit the floor, else 0 |
| 13 | `denom_floor_hit` | 1 if any $d_i$ hit the floor, else 0 |

**Reading the diagnostics.** `src_over_diff` above $\approx 1$ means the learnt residual is fighting diffusion rather than correcting it, and usually points to mis-scaled `source_scale` or an over-regularised $\chi$. Any non-zero `*_floor_hit` on a shot means the forward pass was kept finite by geometry floors rather than by the physics, and that shot’s loss should be treated as advisory.

### 10.2 What should additionally be logged at startup

- ROM grid size $N$, `rho_rom.min/max`, and a confirmation that the last node is excluded from loss (i.e. `obs_idx` does not contain $N-1$).
- Reliable-annulus statistics from `load_data`: number of admitted columns out of $N-1$, the first reliable $\rho$, and the active `data.reliable_cov_min` / `data.reliable_rho_min` (already printed at load time).
- Edge BC statistics per shot: min/max of $T_{\mathrm{edge}}(t)$ and the fraction of timesteps where the edge trace came from time-interpolation (gap-fill) rather than a direct Thomson measurement.
- Geometry flags: whether the $V'(\rho)=2\rho$ fallback was used, and whether the safe-core clamp fired.
- Model parameter count and the values of the fixed-at-construction hyperparameters (§11.6).

### 10.3 BC-specific sanity checks (recommended to add as unit tests)

1. **Axis Neumann:** after building $A$ from `build_diffusion_solve_tridiag_implicit`, assert $A_{0,-1}$ does not exist (tridiagonal), and that the `lower_L[0]` coefficient is exactly zero so no ghost value is referenced.
2. **Dirichlet enforcement:** assert that at every substep the reconstructed $T_{N-1}$ equals the linearly blended $T_{\mathrm{edge}}$ between $T_{\mathrm{edge}}^{n}$ and $T_{\mathrm{edge}}^{n+1}$.
3. **Affine-in-$\mathbf T_{\mathrm{int}}$ check:** for two test states $\mathbf T^{(1)}, \mathbf T^{(2)}$ at fixed $z$, verify $\mathbf D\mathbf T^{(\alpha)}-A\mathbf T^{(\alpha)}_{\mathrm{int}}=\mathbf b_{\mathrm{edge}}$ is independent of $\alpha$.
4. **Edge-trace smoothness:** flag shots whose $T_{\mathrm{edge}}(t)$ has jumps exceeding, say, 200 eV per 1 ms — large discontinuities can masquerade as solver instability.

### 10.4 Evaluation report fields (`logs/<model_id>/evaluation/evaluation_report.json`)

`scripts/evaluate_model.py` writes a per-shot block plus a top-level summary. Each per-shot block carries:

- `metrics`: legacy whole-mask MSE / MAE [eV] / MAE [%] (for backward comparison with pre-M1 runs);
- `annulus_metrics`: same three numbers restricted to the reliable annulus $\mathcal I_{\mathrm{rel}}$;
- `outside_annulus_metrics`: the complement inside the regridded support (a temporary decomposition; it will become a true filled-only metric once the strict per-channel distance gate from §13.2 lands);
- `observability`: `n_reliable_columns`, `first_reliable_rho`, and the active `reliable_cov_min` / `reliable_rho_min`.

The top-level summary aggregates `mean_mae_eV` / `mean_mae_pct` for legacy, annulus, and outside-annulus separately, so regressions in any of the three regimes are visible at a glance. §13.0 anchors the quantitative acceptance criteria to these fields.

---

## 11. Configuration knobs (grouped, matched to `config/config.yaml`)

All paths below are YAML keys in `config/config.yaml`. Defaults shown are those in the committed `production_run_v1` configuration.

### 11.1 Grid, boundary condition, and reliable-annulus gate
- `data.rho_grid_mode`: must be `"uniform"` (enforced at load time).
- `data.uniform_n_rho`: $N$. If omitted, uses the NPZ `rho` length.
- `data.edge_bc_mode`: edge boundary trace construction:
  - `"use_last_observed"` (default) — $T_{\mathrm{edge}}(t)$ is the outermost masked-valid $T_e$ at each time;
  - `"extrapolate_to_1"` — linear extrapolation from the last two observed points to $\rho=1$.
- `data.reliable_cov_min` (default `0.10`) — minimum per-column coverage on the regridded support mask required for a radial bin to enter the reliable annulus $\mathcal I_{\mathrm{rel}}$.
- `data.reliable_rho_min` (default `0.80`) — minimum $\rho$ required for a radial bin to enter $\mathcal I_{\mathrm{rel}}$.

The two thresholds together define the M1 reliable annulus used by both the loss (`shot_loss_imex` multiplies the observation mask by $\mathbf 1[i\in\mathcal I_{\mathrm{rel}}]$) and the evaluator (it reports legacy, annulus, and outside-annulus metrics side by side and renders measured-only heatmaps from `ShotBundle.ts_Te_raw`).

### 11.2 Loss (all under `training.*`)
- `training.huber_delta` (default `10.0` eV) — scale in $\phi_\delta$ for the data term.
- `training.src_delta` (default `10.0`) — scale in $\phi_\delta$ for the source penalty.
- `training.lambda_src` (default `1e-4`) — weight of $\mathcal{L}_{\text{src}}$.
- `training.lambda_w` (default `0`) — weight of the optional weak-constraint model-error term.
- `training.model_error_delta` (default `20.0`) — scale for the model-error pseudo-Huber.
- `training.lambda_z` (default `1e-4`) — latent smoothness weight.
- `training.lambda_zreg` (default `1e-5`) — latent magnitude weight.
- `training.lambda_regime` (default `1e-3`) — BCE weight on confident L/H windows only (`regime_mask`).

### 11.3 IMEX integrator (`training.imex.*`)
- `theta` (default `0.7`) — implicitness; `1.0` = implicit Euler, `0.5` = Crank-Nicolson.
- `substeps` (default `5`) — fixed substeps per Thomson interval; this is the main stability/accuracy knob.
- `max_steps` (default `50000`) — upper bound on inner loop budget.
- `dt_base`, `rtol`, `atol` — reserved for a future adaptive variant; unused in the current fixed-substep mode.

### 11.4 Optimisation (`training.*`)
- `optimizer` (default `"adamw"`), `learning_rate` (`2e-4`), `weight_decay` (`3e-5`).
- `warmup_steps`, `total_steps`, `batch_size`, `grad_clip` (default `1e3`).
- `ema_decay` (default `0.999`).
- Optional L-BFGS finetune: `lbfgs_finetune`, `lbfgs_maxiter`, `lbfgs_epochs`, `lbfgs_history`, `lbfgs_batch_shots`, `lbfgs_tol`.

### 11.5 Model (`model.*`)
- `model.layers` (default `128`), `model.depth` (default `4`) — source-network MLP geometry.
- `model.source_scale` (default `3e5`) — output multiplier applied to the MLP for the source term.
- `model.latent_gain` (default `1.0`) — gain inside $\chi_{\mathrm{edge}}(z)$ and inside the regime BCE logits.

### 11.6 Fixed-at-construction hyperparameters (see `HybridField` in `model.py`)
These are currently hard-coded in the model constructor rather than exposed via YAML:
- `Te_scale = 1000.0` eV, `ne_scale = 1e19` m$^{-3}$.
- `ped_center = 0.85`, `ped_width = 0.08` — sigmoid pedestal location and width in $\chi(\rho,z)$.
- `chi_core = 0.6`, `chi_edge_base = 2.0`, `chi_edge_drop = 1.0` m$^2$/s.
- `divergence_clip = 1e6` — `softclip` saturation on the raw diffusion divergence.

---

## 12. Data inventory (kept as in original; included for completeness)

### 12.1 Primary observables
- `Te`: Thomson scattering; target variable.
- `Te_mask`: validity mask.
- `ne`: Thomson scattering; input to the source network.

### 12.2 Actuators and control inputs
- `P_nbi`, `Ip`, `nebar`, `S_gas`, `S_rec`, `S_nbi` (dense 1D signals; z-scored + clipped; interpolated).
- `D_alpha`, `D_alpha_channels`, `D_alpha_channel_names` are stored as auxiliary diagnostic traces on the summary grid. Raw `D_alpha` is not part of `CONTROL_NAMES`; it is used to derive `S_rec` and to build regime labels for latent supervision.
- `P_rad` not used in current training script.

### 12.3 Geometry
- `rho`: flux coordinate.
- `Vprime`: $V'(\rho)=dV/d\rho$; used in conservative discretisation.

### 12.4 Optional scalars (ingested when present; not used by default)
`W_tot`, `P_ohm`, `P_tot`, `H98`, `beta_n`, `B_t0`, `q95`, `li`, `P_rad`.

### 12.5 Initial condition and latent initialisation (`Te0`, `z0`)

- **Profile IC.** `Te0 = ts_Te_rom[0]` — the first valid regridded Thomson profile. If the first row is entirely masked or non-finite, the pipeline substitutes a synthetic parabolic profile $T(\rho)=100(1-\rho^2)+10$ eV. This fallback is only a safety net; in the current 21-shot set it is rarely used, but when it fires it will produce an artificially smooth first profile and should be flagged in the per-shot diagnostics.
- **Latent IC.** `z0 = 0.0` for every shot. There is no learnt or per-shot latent initialisation yet. One consequence is that the earliest part of each rollout is effectively a “spin-up” window during which the latent is relaxing towards its attractor for the observed $\mathbf u(t)$, and this window carries relatively little information about the learnt dynamics. Evaluation that is sensitive to the first $\sim$20 ms of a shot should weight that window down or mask it out.
- **Train–eval consistency.** Both `shot_loss_imex` and `eval_shot_trajectory_imex` use identical IC construction, so evaluation metrics are directly comparable to training metrics.

---

## 13. Future improvement plan

This section turns the most important lessons from the current evaluation into a concrete, ordered engineering plan. Each milestone is scoped so that it can be merged independently, has a clear exit criterion, and does not depend on unlanded milestones further down the list. The ordering is deliberate: observability first, supervision second, latent dynamics third. Jumping ahead invites debugging a new latent on top of bad labels on top of filled-in core temperatures.

**Out-of-scope for this plan:** switching grids (remains uniform), switching solvers (remains IMEX with fixed substeps), and porting to TORAX. Those are separate larger programs.

### 13.0 Baseline numbers from `production_run_v1`

All acceptance criteria below are anchored to the held-out evaluation in [logs/production_run_v1/evaluation/evaluation_report.json](../logs/production_run_v1/evaluation/evaluation_report.json), produced by the EMA checkpoint [models/production_run_v1/tokamak ode model_best_ema.eqx](../models/production_run_v1/tokamak%20ode%20model_best_ema.eqx) on all 21 packs in `data/`. Aggregated:

| Quantity | Value | Notes |
|---|---|---|
| Legacy mean MAE on $T_e$ | $62.1$ eV | computed with the historical regridded observation mask used by the old evaluator and training loss |
| Reliable-annulus mean MAE on $T_e$ | $64.2$ eV | current M1 split, using only columns with coverage $\ge 0.10$ and $\rho\ge 0.80$ |
| Outside-annulus mean MAE on $T_e$ | $7.44$ eV | temporary decomposition over currently admitted mask support outside the reliable annulus; **not** a final "filled-only" metric |
| Legacy mean MAE% on $T_e$ | $37.9\%$ | historical whole-mask number retained for backward comparison |
| Reliable-annulus mean MAE% on $T_e$ | $38.5\%$ | current M1 split |
| MAE outlier shots | 27575, 27582, 27586 | MSE > $1.5\times 10^4$; visual inspection suggests Thomson spikes / dropouts |
| Mean source/diffusion magnitude ratio | $\approx 1.39$ | residual MLP is doing more work than the conservative diffusion operator |
| Latent $z$ excursion (median over shots) | $\max z \approx 0.06$, $\mathrm{std}\, z \approx 0.013$ | confined to a tiny neighbourhood of $0$ |
| Latent $z$ excursion (max over shots) | $\max z \approx 0.10$ on shot 27579 | even the most active shot barely leaves the linearised regime |

For the current 65-node grid this M1 split yields 12 reliable interior columns (first reliable radius at approximately $\rho=0.812$). These numbers are the targets every later milestone below must beat (or at least not regress) on its own evaluation pass.

### 13.1 What the current evaluation is actually telling us

The present evaluation should be read with care because the Thomson support is strongly edge-localized, and because M1 is only **partially** landed.

- In the current 21-shot pack set, the median innermost *actually measured* $T_e$ point is at approximately $\rho\approx 0.83$.
- The median observed radial span per time slice is only about $0.17$ in $\rho$.
- On the stacked ROM grid, aggregate $T_e$ coverage exceeds 10% only for about $\rho\geq 0.78$ and exceeds 20% only for about $\rho\geq 0.83$.
- Therefore the inner-core part of the current measured-only heatmaps is *correctly blank* after the M1 evaluator patch, but the remaining observed annulus is still a **regridded support mask**, not a strict per-channel Thomson mask.
- Visual evidence of the landed part: in [shot_27567_heatmap.png](../logs/production_run_v1/evaluation/plots/shot_27567_heatmap.png), the measured-only panel is white below roughly $\rho\approx 0.76$ and only the outer annulus is rendered. That is the intended M1 behaviour: missing support is shown as missing support.

This explains why the core of the pre-M1 evaluation heatmaps looked poor. The preprocessing uses interpolation plus fill operations to keep the solver inputs well-defined, but that numerically convenient field should not be interpreted as trustworthy core measurement where no Thomson support exists.

The remaining observability problem is now more specific and more local: [interp_profile_to_grid in data.py](../fusion_ode_identification/data.py#L99-L142) still constructs the regridded support mask by interpolating the source-channel mask and thresholding it at `> 0.5`. That means bins *between* measured Thomson channels can still be admitted once the profile is mapped to the uniform ROM grid. So M1 has fixed the gross failure mode (filled core shown and scored as truth), but it has **not** yet landed the stricter per-channel distance gate from §13.2.

### 13.1a Latent collapse: independent evidence that M3 is needed

The diagnostics in [evaluation_report.json](../logs/production_run_v1/evaluation/evaluation_report.json) show that the cubic latent
$$
\dot z=\mathrm{softplus}(\alpha)\,(\mu-\mu_{\mathrm{ref}})-\mathrm{softplus}(\beta)\,z-\mathrm{softplus}(\gamma)\,z^3
$$
([model.py `LatentDynamics`](../fusion_ode_identification/model.py#L65-L78)) is *not* expressing regime structure on this dataset. The per-shot latent excursions are below $0.1$, $\mathrm{std}\,z\sim 10^{-2}$, and visually monotone-decaying with no sign of bistability or hysteresis (e.g. [shot_27567_latent.png](../logs/production_run_v1/evaluation/plots/shot_27567_latent.png), [shot_27582_latent.png](../logs/production_run_v1/evaluation/plots/shot_27582_latent.png)). Because the model couples the profile to $z$ only through $\sigma(g\cdot z)$ with $g=\texttt{latent\_gain}=1$, an excursion of $0.06$ moves $\chi_{\mathrm{edge}}$ by less than $1.5\%$. The latent is therefore effectively switched off, and the residual MLP $\mathbf S_\theta$ is absorbing all regime-dependent behaviour — which is exactly what the source/diffusion magnitude ratio of $\approx 1.4$ in §13.0 reports.

This is independent evidence (separate from the regime-label issue in §13.2–13.3) that the cubic-damped scalar $z$ is the wrong parameterisation. M3 below replaces it with a bounded hysteretic barrier. Two cheaper intermediate diagnostics are worth running first:

- **Free-sign damping.** Drop the `softplus` on $\beta$ (allow negative damping) and refit, to test whether the optimiser is being pushed into the over-damped corner by the positivity constraint alone. If $z$ still collapses, the parameterisation, not the constraint, is the problem.
- **Latent-gain sweep.** Refit with $g\in\{1,3,10\}$ and a fixed source-magnitude penalty. If higher $g$ does not lower MAE on the reliable annulus, the residual MLP is fully shadowing the latent and §13.4's bounded barrier is the only fix.

### 13.2 Numerical logic for which $T_e$ signals should actually be used

Future work should split the profile representation into two distinct objects. The first half of this is already landed in-memory on `ShotBundle`; the second half (pack-schema and strict channel-distance gating) is still open.

- $T_e^{\mathrm{raw}}(t,\rho)$ with the original Thomson mask for supervision, plotting, and quality control.
- $T_e^{\mathrm{fill}}(t,\rho)$ only for solver housekeeping tasks such as interpolation, IC construction, and boundary completion.

The supervision mask should then be restricted to a numerically justified reliable annulus. Define a corpus-level reliable set
$$
\mathcal I_{\mathrm{rel}}=
\left\{i:\ c_i\geq c_{\min},\ \rho_i\geq \rho_{\min}\right\},
\qquad
c_i=\frac{1}{N_{\mathrm{rows}}}\sum_t m_i^{\mathrm{raw}}(t).
$$

For the current dataset, a sensible starting rule is
$$
\rho_{\min}\approx 0.80,
\qquad
c_{\min}\approx 0.10,
$$
with a stricter pedestal-focused variant using $\rho_{\min}\approx 0.83$.

Then the usable training/evaluation mask should be
$$
m_i^{\mathrm{use}}(t)=
m_i^{\mathrm{raw}}(t)
\,\mathbf 1\!\left[i\in\mathcal I_{\mathrm{rel}}\right]
\,\mathbf 1\!\left[d_i(t)\leq \Delta\rho\right],
$$
where $d_i(t)$ is the distance from bin $i$ to the nearest genuinely measured channel at time $t$.

**What is already implemented:** the current patch lands the factor $m_i^{\mathrm{raw}}(t)\,\mathbf 1[i\in\mathcal I_{\mathrm{rel}}]$ in the loss and the evaluator, and carries both `ts_Te` (filled, solver-facing) and `ts_Te_raw` (NaN-preserving, plotting-facing) on the in-memory bundle.

**What is not implemented yet:** the distance gate $\mathbf 1[d_i(t)\leq \Delta\rho]$ and a true pack-level `Te_raw` / `Te_fill` schema. At present, `ts_Te_raw` is still a regridded field supported by the interpolated mask inside the measured span.

The practical consequence is simple: if a radius is only present because of interpolation or fill, it should not participate in the loss and should not be shown as “observed” in evaluation plots.

Additional $T_e$ QC should be added before any transition redesign:

- Reject time slices with too few contiguous measured edge channels.
- Reject isolated hot pixels or single-frame spikes using a median-absolute-deviation or local-curvature test.
- Downweight or drop rows with abrupt jumps unsupported by neighboring times or channel uncertainty.
- Plot measured-only heatmaps with missing regions blank, and plot the filled solver field separately.

### 13.3 How L-H identification is usually done in experiments

Real experimental practice does not define H-mode from $D_\alpha$ alone. Instead, the transition is typically inferred from a consistent package of signatures:

1. A rapid drop in divertor/edge $D_\alpha$ or recycling light.
2. Onset of an edge pedestal in $T_e$, $n_e$, or edge pressure.
3. Improvement in global confinement indicators such as stored energy $W_{\mathrm{tot}}$, confinement time $\tau_E$, or $H_{98}$ when available.
4. A change in edge fluctuation behavior and, in ELMy H-mode, the later appearance of ELM bursts.

This is consistent with the historical H-mode observations and later reviews: H-mode is fundamentally an edge transport barrier / confinement regime, while $D_\alpha$ is a fast edge proxy rather than the definition of the state itself [18,19].

For this ROM, that implies:

- $D_\alpha$ is useful and should absolutely remain in the pipeline.
- But raw $D_\alpha$ is not a one-to-one confinement coordinate.
- It is line-integrated, view-dependent, sensitive to gas puffing and divertor conditions, and can be strongly modulated by ELMs and recycling bursts.

So the correct interpretation is: $D_\alpha$ is evidence *about* the transition, not the transition state itself.

### 13.4 Recommended next latent design

The best next model is **not** to replace the latent state with raw $D_\alpha$. The better approach is to keep an interpretable latent barrier state and use $D_\alpha$ as an auxiliary observation and transition cue.

The recommended minimal redesign is a bounded barrier variable
$$
z_b(t)\in[0,1],
$$
where $z_b\approx 0$ means L-mode-like edge transport and $z_b\approx 1$ means a strong edge barrier / H-mode-like state.

Its dynamics should be hysteretic rather than purely cubic. One practical form is
$$
\dot z_b=
\frac{1-z_b}{\tau_{\mathrm{LH}}}\,\sigma(\eta_{\mathrm{LH}})
-
\frac{z_b}{\tau_{\mathrm{HL}}}\,\sigma(\eta_{\mathrm{HL}}),
$$
with separate activation channels for L$\rightarrow$H and H$\rightarrow$L transitions. A useful feature construction is
$$
\eta_{\mathrm{LH}}=
w_u^\top \mathbf u
+w_D\big(-\partial_t\widetilde D_\alpha\big)
+w_T\,\partial_t\widetilde T_{e,\mathrm{edge}}
+w_n\,\partial_t\widetilde n_{e,\mathrm{edge}}
+b_{\mathrm{LH}},
$$
with an analogous $\eta_{\mathrm{HL}}$ using its own parameters and hysteresis logic.

The key point is that the transition latent should be driven by **edge evidence** rather than by filled core profiles.

### 13.5 Where “$z\rightarrow D_\alpha$” does make sense

The idea of using $z\rightarrow D_\alpha$ is good as an **auxiliary observation model**, not as a replacement for the latent. In other words, the model should learn
$$
\widehat D_\alpha(t)=h_\psi\big(z_b(t),\mathbf u(t),T_{e,\mathrm{edge}}(t),n_{e,\mathrm{edge}}(t)\big),
$$
and optionally also a smoothed transition score or regime probability.

This is better than making $D_\alpha$ the latent because:

- the latent remains an interpretable confinement/barrier coordinate;
- $D_\alpha$ is treated as a noisy measurement channel generated by the hidden state plus actuators;
- the model can learn that the same barrier state may correspond to different raw $D_\alpha$ amplitudes under different gas puffing or divertor conditions.

So the recommended view is:

- keep a latent state,
- make it physically interpretable,
- supervise it with better transition evidence,
- and attach $D_\alpha$ as an auxiliary output and gating signal.

### 13.6 If one latent is not enough

If later datasets are richer, the next step after the bounded barrier latent is a two-latent decomposition:

- $z_b$: confinement / barrier state;
- $z_r$: recycling / neutral / divertor response.

That split is physically cleaner because recycling and confinement are coupled but not identical. In such a model, transport coefficients should depend mainly on $z_b$, while $D_\alpha$ should couple more strongly to $z_r$.

However, for the current small and sparse dataset, the best immediate step is still the single bounded hysteretic barrier latent plus an auxiliary $D_\alpha$ head.

### 13.7 Ordered engineering milestones

Each milestone below is scoped to be mergeable on its own and has an explicit acceptance criterion. Acceptance numbers are stated relative to the §13.0 baseline so that regression is unambiguous.

**M1 — Observability hygiene (must land first).** *Touches:* [fusion_ode_identification/types.py](../fusion_ode_identification/types.py), [fusion_ode_identification/data.py](../fusion_ode_identification/data.py), [fusion_ode_identification/loss.py](../fusion_ode_identification/loss.py), [scripts/evaluate_model.py](../scripts/evaluate_model.py).
- **Status:** phase 1 is now landed. The bundle carries `ts_Te` (filled, solver-facing), `ts_Te_raw` (NaN-preserving, plotting-facing), and a corpus-level `reliable_mask`.
- In `load_data`, compute a corpus-level reliable-annulus mask $\mathcal I_{\mathrm{rel}}$ from per-column coverage ($c_{\min}=0.10$, $\rho_{\min}=0.80$) and persist it on the bundle. For the current 65-node grid this yields 12 reliable interior columns, starting at approximately $\rho=0.812$.
- In `shot_loss_imex`, multiply the current observation mask by $\mathbf 1[i\in\mathcal I_{\mathrm{rel}}]$ so the loss no longer supervises the filled core.
- In [evaluate_model.py `plot_results`](../scripts/evaluate_model.py), render a measured-only `pcolormesh` using `np.where(mask>0, Te_raw, np.nan)` with `cmap.set_bad("white")`, and report legacy, annulus, and outside-annulus metrics side-by-side in `evaluation_report.json`.
- **Remaining work in M1:** move the `Te_raw` / `Te_fill` split into the NPZ pack schema itself, and replace the current interpolated support mask with the stricter distance-to-nearest-channel gate from §13.2.
- **Exit criterion (quantitative):** the measured-only heatmap for shot 27567 contains visible NaN gaps below $\rho<0.8$; mean MAE on the reliable annulus is reported alongside the legacy metric in the report JSON; the report now decomposes the legacy 37.9 % MAPE into `annulus_*` and `outside_annulus_*` fields. A later M1 follow-up will replace `outside_annulus_*` with a true channel-distance / filled-only decomposition.

**M2 — Regime relabelling with dwell-time and per-shot QA.** *Touches:* [preprocessing/build_training_pack.py `estimate_regime_labels`](../preprocessing/build_training_pack.py#L167-L233).
- Replace the single `argmax` over $s(t)$ at L223 with a multi-signal change-point detector using $-\partial_t \widetilde D_\alpha$, $\partial_t \widetilde{\bar n}_e$, $\partial_t \widetilde P_{\mathrm{nbi}}$, and (when available) $\partial_t \widetilde W_{\mathrm{tot}}$ or $H_{98}$.
- Enforce a minimum dwell-time ($\sim 20$ ms) on both L and H segments; reject shots where no consistent H-window exists. The current `argmax` rule places the transition at the single highest-scoring sample regardless of how isolated it is in time, which on noisy $D_\alpha$ traces produces transitions inside ELM bursts.
- Emit a per-shot QA plot (normalised signals, score, detected transitions, dwell-time consistency) into `data/plots/`.
- **Exit criterion:** on the 21-shot set, at least 80 % of shots get a human-agreeable transition marker; rejected shots are logged with a reason in `data/sanity_summary.csv`.

**M3 — Bounded hysteretic barrier latent (with two free intermediate tests).** *Touches:* [model.py `LatentDynamics`, `HybridField._chi_profile`](../fusion_ode_identification/model.py#L65-L120).
- *Diagnostic step (one short training run, throwaway):* drop `softplus` on $\beta$ and rerun. If $\mathrm{std}\,z$ still stays below $0.05$, the cubic is the wrong shape and we proceed; if $\mathrm{std}\,z$ now exceeds $0.3$, log the result and *still* proceed (the cubic is intrinsically symmetric and cannot represent hysteresis).
- *Diagnostic step (one short run):* sweep `latent_gain` $\in\{1,3,10\}$ at fixed model. Record the change in mean source/diffusion ratio. The expected outcome on the current dataset is no change; if a higher gain measurably lowers the source ratio, retain that value as the §13.4 default.
- Replace the cubic damping with a bounded state $z_b\in[0,1]$ (parameterise $z_b=\sigma(\zeta)$ and evolve $\zeta$) driven by separate L$\to$H and H$\to$L activation channels, each a learned affine function of edge evidence ($-\partial_t\widetilde D_\alpha$, $\partial_t\widetilde T_{e,\mathrm{edge}}$, $\partial_t\widetilde n_{e,\mathrm{edge}}$, plus selected $\mathbf{u}$).
- Keep the IMEX split intact: the latent ODE stays fully explicit.
- Update `_chi_profile` to use $z_b$ directly rather than $\sigma(g z)$; remove the now-redundant `latent_gain`.
- **Exit criterion:** on the same train/val split, the regime BCE on held-out shots drops by at least one bit relative to the current cubic latent; mean MAE on the reliable annulus drops by at least 10 % relative to the M1 baseline; the learnt $z_b$ is visually monotone within each dwell-time-validated window.

**M4 — Auxiliary observation heads.** *Touches:* `model.py`, `loss.py`.
- Add small prediction heads $\widehat D_\alpha(t)=h_\psi(z_b,\mathbf u,T_{e,\mathrm{edge}},n_{e,\mathrm{edge}})$ and optionally a calibrated L/H probability $\widehat p_H(t)$.
- Add a soft auxiliary loss (weight $\sim 10^{-2}$, comparable to current `lambda_regime=1\mathrm{e}{-3}$ scaled by the higher signal-to-noise of $D_\alpha$) on normalised $D_\alpha$ and on the regime-labelled segments.
- **Exit criterion:** the model reproduces normalised $D_\alpha$ within 0.1 RMSE on held-out shots, and $\widehat p_H$ has calibration error $<5\%$ on validated windows.

**M5 — Numerical hygiene pass (can run in parallel with M2–M4).** *Touches:* [imex_solver.py](../fusion_ode_identification/imex_solver.py), [loss.py](../fusion_ode_identification/loss.py) diagnostics.
- Add an assertion-mode check that `dr_floor_hit`, `vprime_floor_hit`, `denom_floor_hit` are zero on every training shot; any shot that triggers a floor is excluded until its geometry is repaired.
- Add a *source-budget penalty* of the form $\lambda_{\mathrm{bal}}\,\big(\|\mathbf S_\theta\|/\|\nabla\!\cdot\!\mathbf q\|-r^\star\big)_+^2$ with $r^\star\!=\!1$ and $\lambda_{\mathrm{bal}}\!=\!10^{-3}$, so the residual cannot silently overpower the conservative diffusion operator. The §13.0 baseline ratio of $1.39$ should drop to $\leq 1.1$.
- Sweep `theta` $\in\{0.5,0.7,1.0\}$ and `substeps` $\in\{3,5,10\}$ on a fixed model; report MAE, source magnitude, and wall-clock. Pin the defaults once the Pareto-optimum is clear.
- Sensitivity scan on `ped_center` (0.83–0.90) and `ped_width` (0.04–0.12); document the chosen defaults with a one-line justification in `HybridField.__init__`.
- **Exit criterion:** a short table of $(\theta,\texttt{substeps},\texttt{ped\_center},\texttt{ped\_width})$ sensitivity lives in the evaluation report; no floor-hit diagnostic fires on the default configuration; mean source/diffusion ratio $\leq 1.1$.

**M5b — Per-shot quality screening.** *Touches:* [preprocessing/build_training_pack.py](../preprocessing/build_training_pack.py), [scripts/inspect_data.py](../scripts/inspect_data.py).
- The §13.0 outliers (27575 MSE 30.5k, 27582 MSE 54.0k, 27586 MSE 15.3k) all exceed 2.5× the cohort median MSE. Add a per-shot screen using (i) Thomson sample density, (ii) fraction of edge-channel rows with abrupt jumps unsupported by neighbouring times (median-absolute-deviation test), and (iii) the M5 floor-hit count, and write the screen result to `data/sanity_summary.csv`.
- Pack-builder should refuse to overwrite a shot pack whose screen result downgrades it; the user must explicitly opt in via `--allow-degraded`.
- **Exit criterion:** removing screened-out shots from training reduces evaluation MAE std-across-shots by at least 30 % without changing the median MAE.

**M6 — Second latent for recycling (only after M1–M4 are landed and a larger dataset is available).**
- Add $z_r\in\mathbb R$ driven primarily by $S_{\mathrm{gas}}$, $S_{\mathrm{rec}}$, and divertor/SOL evidence. Couple $\chi$ only weakly to $z_r$ (if at all); couple $\widehat D_\alpha$ strongly to $z_r$.
- **Exit criterion:** $z_r$ explains at least $30\%$ of the residual variance in $\widehat D_\alpha$ that $z_b$ alone leaves on held-out shots; otherwise reject and keep the single-latent model.

**M7 — Longer-term: controllable $T_{e,\mathrm{edge}}$.**
The current model **requires** $T_{\mathrm{edge}}(t)$ as an input (see [data.py edge BC construction](../fusion_ode_identification/data.py#L302-L340) and the Dirichlet reduction in `shot_loss_imex`) and cannot therefore do closed-loop prediction from actuators alone. A minimal remedy is a separate, lightweight edge model $T_{\mathrm{edge}}=g_\phi(\mathbf u,z_b)$ trained offline on the same packs, plugged in at inference. This is intentionally parked until M1–M4 are in place because a bad edge closure on top of bad core supervision is impossible to debug.

### 13.8 Why this ordering is non-negotiable

If M1 is skipped, every metric we optimise against is partly fitting noise in the filled core, and the §13.0 numbers cannot be trusted as a baseline. If M2 is skipped, the BCE signal in M3 points at the wrong times. If M3 is skipped and we jump to M4–M6, the auxiliary heads and the second latent inherit a cubic-damped latent that we have already shown (§13.1a) is collapsed near zero, so they will absorb the blame for whatever the latent should have been doing. M5 and M5b are deliberately structured as parallelisable hygiene work because they have to land *before* anyone reads the M3/M4 evaluation numbers as evidence — the source-budget penalty in particular fixes a confound that would otherwise let M3 "win" by silently reshaping $\mathbf S_\theta$ instead of $z_b$. The sequence above keeps each milestone *locally falsifiable* — each one has a concrete exit criterion that does not depend on the later ones.

---

## 14. References (updated)

[1] T. H. Osborne, K. H. Burrell, and R. J. Groebner. H-mode pedestal characteristics in DIII-D. *Plasma Physics and Controlled Fusion*, 40(5):845, 1998.  
[2] J. Bradbury et al. JAX: composable transformations of Python+NumPy programs. 2018.  
[3] P. Kidger. *On Neural Differential Equations*. PhD thesis, University of Oxford, 2021.  
[4] F. Felici, A. Merle, et al. TORAX: a differentiable tokamak transport simulator in JAX. *arXiv preprint arXiv:2409.10622*, 2024.  
[5] B. Coppi. Non-classical transport and the "principle of profile consistency". *Comments on Plasma Physics and Controlled Fusion*, 5(6):261–270, 1980.  
[6] H. Haken. *Synergetics: An Introduction*. Springer-Verlag, Berlin, 3rd edition, 1983.  
[7] H. Mori. Transport, collective motion, and Brownian motion. *Progress of Theoretical Physics*, 33(3):423–455, 1965.  
[8] R. Zwanzig. Memory effects in irreversible thermodynamics. *Physical Review*, 124(4):983, 1961.  
[9] L. D. Landau. On the theory of phase transitions. *Zh. Eksp. Teor. Fiz.*, 7:19–32, 1937.  
[10] J. Carr. *Applications of Centre Manifold Theory*. Springer-Verlag, New York, 1981.  
[11] A. Vaswani et al. Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 2017.  
[12] D. Ha and J. Schmidhuber. World models. *arXiv preprint arXiv:1803.10122*, 2018.  
[13] R. E. Kalman. A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1):35–45, 1960.  
[14] A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*, 2023.  
[15] E. J. Doedel et al. AUTO-07P: continuation and bifurcation software for ordinary differential equations. 2007.  
[16] M. Scheffer et al. Early-warning signals for critical transitions. *Nature*, 461(7260):53–59, 2009.  
[17] K. Hornik et al. Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5):359–366, 1989.  
[18] F. Wagner. A quarter-century of H-mode studies. *Plasma Physics and Controlled Fusion*, 49(12B):B1, 2007.  
[19] ITER Physics Expert Group on Confinement and Transport, ITER Physics Expert Group on Confinement Modelling and Database, and ITER Physics Basis Editors. Chapter 2: Plasma confinement and transport. *Nuclear Fusion*, 39(12):2175, 1999.

---

## Appendix A. Code map

This appendix maps document sections to the source of truth in the repository. When the document and the code disagree, the code wins.

| Doc section | Source | Notes |
|---|---|---|
| §1 architecture | [fusion_ode_identification/model.py](../fusion_ode_identification/model.py) | `SourceNN`, `LatentDynamics`, `HybridField` |
| §2 pipeline, §3 geometry | [fusion_ode_identification/data.py](../fusion_ode_identification/data.py) | `load_data`, `interp_profile_to_grid`, edge-BC modes |
| §2 D-alpha + regime labels | [preprocessing/build_training_pack.py](../preprocessing/build_training_pack.py) | `estimate_regime_labels`, `extract_dalpha_arrays` |
| §4.1–4.3 governing model | `HybridField.compute_rhs_components`, `HybridField._chi_profile`, `LatentDynamics.__call__` | clamps, softclip, chi profile |
| §5 boundary conditions | `HybridField.compute_rhs_components` (axis Neumann via flux form), `shot_loss_imex` (edge Dirichlet state reduction) | |
| §6 spatial discretisation | [fusion_ode_identification/imex_solver.py](../fusion_ode_identification/imex_solver.py) | `build_diffusion_solve_tridiag_implicit`, `apply_diffusion_explicit`, `solve_tridiagonal` |
| §7 loss | [fusion_ode_identification/loss.py](../fusion_ode_identification/loss.py) | `shot_loss_imex`, `pseudo_huber` |
| §9 IMEX integrator | `IMEXIntegrator.step`, `IMEXIntegrator.integrate` in `imex_solver.py` | fixed `substeps`, linear intra-interval blending |
| §10 diagnostics | `diag` tuple returned by `shot_loss_imex` | 14 fields, fixed order |
| §11 config | [config/config.yaml](../config/config.yaml) | defaults referenced here are the `production_run_v1` values |
| §12 data layout | [fusion_ode_identification/types.py](../fusion_ode_identification/types.py) | `ShotBundle`, `ShotEval`, `LossCfg`, `IMEXConfig` |
| §13 future plan | target modules listed per-milestone | M1 touches pack + data + evaluator; M3 touches `model.py` |
