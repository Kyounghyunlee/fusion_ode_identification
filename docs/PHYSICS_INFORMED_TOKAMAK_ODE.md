# Physics-Consistent Transport ROM (Uniform Grid)

**Status:** Active — implemented in `train_tokamak_ode_hpc.py` and `fusion_ode_identification/*`.  
**Pipeline:** Standalone training pipeline using TORAX-compatible pack format (see §3.0). TORAX is **not** required to run this ROM.

---

## Executive summary (one page)

We learn a **physics-consistent neural ODE** reduced-order model (ROM) for tokamak **electron temperature transport** on a 1D flux coordinate $\rho\in[0,1]$. The ROM evolves the **physical profile state** directly, rather than a latent embedding:

- Physical state: $\mathbf{T}(t)\in\mathbb{R}^{N}$ (electron temperature on a uniform radial grid).
- Latent order parameter: $z(t)\in\mathbb{R}$ (captures regime-dependent edge transport, L-mode ↔ H-mode).
- Inputs: density profile $\mathbf{n}_e(t)$ and actuators $\mathbf{u}(t)$.

The model combines:
1) A **conservative diffusion operator** obtained from a finite-volume / flux-form discretisation of the transport PDE, with interpretable $\chi(\rho,z)$ and geometry factor $V'(\rho)$;  
2) A **learned residual source** $S_\theta$ (small MLP) capturing unmodelled physics;  
3) A low-dimensional latent dynamics $\dot z=f_z(\mathbf{u},z)$.

**Stiffness:** diffusion on a fine grid produces large negative eigenvalues $|\lambda_{\max}|\sim \chi/(\Delta\rho)^2$, so explicit time stepping would require $\Delta t=O((\Delta\rho)^2)$ for stability. We therefore use an **IMEX $\theta$-method** (implicit diffusion + explicit residual/latent), yielding stable timesteps aligned with diagnostic sampling rather than CFL limits.

**Boundary conditions are first-class:**
- Axis ($\rho=0$): **homogeneous Neumann / symmetry**, implemented as **zero left boundary flux**.
- Edge ($\rho=1$): **Dirichlet**, implemented by **imposing** $T_{\text{edge}}(t)$ (built from masked data) and excluding the boundary node from training loss and solver state.

This document describes the full modelling and training pipeline, with special emphasis on **how Neumann and Dirichlet BCs enter the discrete operator and the semi-discrete ODE form**.

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

---

## 1. Model architecture and why it is structured this way

### 1.1 What we are *not* doing (alternatives, short and honest)

**Transformers**: sequence models that map histories $(\mathbf{T}_{t-k},\dots,\mathbf{T}_{t-1})$ to $\mathbf{T}_t$ using attention. Strength: long-context pattern learning. Weakness: no explicit PDE structure; expensive for real-time because each step re-processes history.

**World models / latent dynamics**: encode $\mathbf{T}$ into a latent code $\mathbf{h}$, evolve $\mathbf{h}$ with a black-box dynamics model, decode back. Strength: compression; weakness: latent is hard to interpret physically (diffusivity/flux become opaque).

**Modern SSMs (S4/Mamba-like)**: efficient sequence modelling using structured hidden states. Strength: scaling; weakness: hidden state is abstract; physical interpretability and physics-guided analysis are indirect.

### 1.2 What we do instead (physics-informed ODE on the physical state)

We model the evolution of the physical profile directly:
$$
\frac{d\mathbf{T}}{dt}=
\underbrace{\mathbf{D}(\mathbf{T};\chi,V')}_{\text{conservative diffusion (stiff)}}
+
\underbrace{\mathbf{S}_{\theta}(\boldsymbol{\rho},\mathbf{T},\mathbf{n}_e,\mathbf{u},z)}_{\text{learned residual (mild)}}.
$$
The latent variable evolves as
$$
\frac{dz}{dt}=f_z(\mathbf{u},z),
$$
and it modulates edge transport by changing $\chi(\rho,z)$ (pedestal localisation + edge drop).

**Why this structure matters (dynamical systems + control):**
- The state is $\mathbf{T}\in\mathbb{R}^N$, so linearisation and eigen-analysis are meaningful for stability/critical slowing down.
- We can interpret learned $\chi(\rho,z)$ and compare to known transport regimes.
- We can do continuation/bifurcation analysis in principle by treating $z$ or control components as parameters.

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
- `use_last_observed` (default): take $T_e$ at the outermost observed index when masked and finite.
- `extrapolate_to_1`: linearly extrapolate from the last two observed points to $\rho=1$ when possible.
- Interpolate in time (linear) to fill gaps on the shot’s time grid.
- Fallback: if undefined at any time, use 50 eV to avoid boundary degeneracy.

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
\alpha(\mu(\mathbf{u})-\mu_{\text{ref}})
-\beta z-\gamma z^3,
\qquad \alpha,\beta,\gamma>0,
$$
with $\mu(\mathbf{u})$ depending on the first three controls (current implementation: `P_nbi`, `Ip`, `nebar`).

Edge diffusivity drop:
$$
\chi_{\text{edge}}(z)=\chi_{\text{edge,base}}-\chi_{\text{edge,drop}}\;\sigma(kz),
$$
and $\chi(\rho,z)$ is blended across $\rho$ by a sigmoid pedestal envelope (localises the edge barrier).

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

**Rationale:**
The homogeneous Neumann condition at the magnetic axis is not a modeling choice but a **physical necessity** arising from the coordinate system and plasma symmetry:

1. **Geometric regularity:** In flux coordinates centered on the magnetic axis, $\rho=0$ is a regular point (not a boundary of the plasma). The temperature field must be smooth and well-defined there, which requires $\partial T_e/\partial\rho(0,t)=0$ to avoid multi-valuedness or singularities.

2. **Toroidal symmetry:** The tokamak plasma is approximately axisymmetric around the major axis. At the magnetic axis ($\rho=0$), there is no preferred radial direction, so the gradient must vanish by symmetry.

3. **Conservation form:** In the divergence form $\nabla\cdot(V'\chi\nabla T_e)$, the flux $F=-V'\chi\,\partial T_e/\partial\rho$ must remain finite as $\rho\to 0$. Since $V'(\rho)\sim\rho$ near the axis (geometric factor), we need $\partial T_e/\partial\rho\to 0$ to cancel the singularity and ensure $F(0,t)=0$.

4. **No alternative:** Any non-homogeneous Neumann or Dirichlet condition at the axis would be physically artificial and would inject/remove energy in a way inconsistent with the closed-flux-surface geometry.

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

**Rationale:**
The Dirichlet condition at the edge is a **pragmatic modeling choice** that balances physical realism with ROM scope:

1. **Open-field-line boundary:** At $\rho=1$ (the last closed flux surface / separatrix), the plasma transitions to the scrape-off layer (SOL), where field lines intersect material surfaces (divertor, limiter). The edge temperature is determined by complex SOL physics (parallel transport, sheath boundary conditions, divertor recycling, impurity radiation) that are outside the scope of a core-transport ROM.

2. **Data-driven closure:** Rather than modeling the full SOL/divertor system (which would require 2D geometry, neutral physics, and surface interactions), we use the *observed* edge temperature $T_{\text{edge}}(t)$ from Thomson scattering as a boundary condition. This implicitly encapsulates the net effect of SOL processes on the core-edge interface.

3. **Standard practice in transport modeling:** Many reduced transport models (including TRANSP, ASTRA, and some TORAX configurations) use Dirichlet or Robin conditions at the edge when SOL modeling is not included. The alternative—a Neumann (flux) condition—would require knowing the conducted power crossing the separatrix, which is equally difficult to measure directly and would still require external closure.

4. **Regime-dependent edge physics:** The edge temperature is strongly influenced by L-H mode transitions (pedestal formation), ELMs, and other edge phenomena. By imposing $T_{\text{edge}}(t)$ from data, we avoid having to explicitly model these dynamics, while still capturing their effect on the core through the boundary forcing term $\mathbf{b}_{\text{edge}}(t)$ in the diffusion operator.

5. **Identifiability:** A Dirichlet condition improves the identifiability of core transport coefficients $\chi(\rho,z)$ by decoupling them from edge/SOL uncertainties. The learned diffusivity and source terms describe *core* transport given the observed edge state, rather than conflating core and edge physics.

**Trade-off:** This approach cannot predict $T_{\text{edge}}(t)$ itself—it must be supplied (from data or from a companion edge model). For scenario prediction, one would need either (a) a statistical model for $T_{\text{edge}}(t)$ given actuators, or (b) coupling to an explicit edge/pedestal model.

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

### 6.4 Semi-discrete ODE and affine operator form

Let $\mathbf{T}_{\text{int}}=(T_0,\dots,T_{N-2})^\top$ and define the full profile
$$
\mathbf{T}=(T_0,\dots,T_{N-2},T_{\text{edge}}(t))^\top.
$$
Then diffusion on the interior can be written as
$$
\mathbf{D}(\chi,V')\,\mathbf{T}
=
A(\chi,V')\,\mathbf{T}_{\text{int}}
+
\mathbf{b}_{\text{edge}}(t;\chi,V'),
$$
where:
- $A(\chi,V')\in\mathbb{R}^{(N-1)\times(N-1)}$ is banded (tri-diagonal in the simplest 1D stencil form),
- $\mathbf{b}_{\text{edge}}(t)$ is a boundary forcing vector induced by the Dirichlet value (nonzero only near the boundary row, and potentially scaled by coefficients at the last face).

Therefore the interior ODE used by the solver is
$$
\frac{d\mathbf{T}_{\text{int}}}{dt}
=
A(\chi(\cdot,z),V')\,\mathbf{T}_{\text{int}}
+
\mathbf{b}_{\text{edge}}(t;\chi(\cdot,z),V')
+
\mathbf{S}_{\text{net,int}}(\boldsymbol{\rho},\mathbf{T},\mathbf{n}_e,\mathbf{u},z),
$$
and
$$
\frac{dz}{dt}=f_z(\mathbf{u},z).
$$

**Why this matters for numerics:** the stiff part is precisely the affine diffusion operator. IMEX treats $A\mathbf{T}_{\text{int}}+\mathbf{b}_{\text{edge}}$ implicitly, and treats the residual and latent explicitly.

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
\lambda_{z\text{-sup}}\mathcal{L}_{z\text{-regime}}
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

**Boundary exclusion is essential:** the node $i=N-1$ is Dirichlet-imposed and should not be included in the data loss.

### 7.3 Source magnitude penalty
$$
\mathcal{L}_{\text{src}}=\mathbb{E}_{t,i}\left[\left|(\mathbf{S}_{\text{net}})_i\right|^2\right].
$$

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
- Optional regime supervision:
  $$
  \mathcal{L}_{z\text{-regime}}=\frac{1}{K}\sum_{k=1}^{K}\left(\sigma(z_k)-r_k\right)^2,\qquad r_k\in\{0,1\}.
  $$
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
- Fixed number of substeps per observation interval (`training.imex.substeps`).
- Training integrates on padded time grids for batch uniformity; loss masking uses `t_len`.
- Debug/eval integrates on the valid window only.

---

## 10. Logging and diagnostics (BC-aware checks)

### 10.1 What should be logged at startup
- ROM grid size $N$ and min/max.
- `obs_idx` and confirmation that the last node is excluded from loss.
- Edge BC statistics: min/max of $T_{\text{edge}}(t)$, gap-fill fraction.
- Geometry flags: fallback usage, safe-core clamping applied.

### 10.2 BC-specific sanity checks (recommended)
1) **Axis Neumann check:** verify left flux is identically zero in the operator:
   - numerically confirm $F_{-1/2}=0$ and that no ghost value is referenced.
2) **Dirichlet enforcement check:** at every RHS evaluation time,
   $$
   T_{N-1}(t)=T_{\text{edge}}(t)
   $$
   (including substeps).
3) **Boundary forcing check:** the diffusion term is affine in $\mathbf{T}_{\text{int}}$ given fixed $\chi$ and $V'$:
   $$
   \mathbf{D}\mathbf{T}=A\mathbf{T}_{\text{int}}+\mathbf{b}_{\text{edge}}(t).
   $$
4) **Edge-trace smoothness check:** large discontinuities in $T_{\text{edge}}(t)$ can masquerade as solver instability.

---

## 11. Configuration knobs (unchanged list, grouped)

### 11.1 Grid
- `data.uniform_n_rho`: $N$.
- `data.edge_bc_mode`: edge boundary trace construction:
  - `"use_last_observed"` (default),
  - `"extrapolate_to_1"`.

### 11.2 Loss
- `loss.huber_delta`
- `loss.model_error_delta`
- `training.lambda_src`
- `training.lambda_w`
- `training.lambda_z`
- `training.lambda_zreg`
- `training.lambda_zsup` (if regime labels exist)

### 11.3 IMEX
- `training.imex.theta` (default 0.7)
- `training.imex.substeps` (default 5)
- note: fixed substeps are chosen for stable reverse-mode autodiff.

### 11.4 Model
- `model.chi_core`, `model.chi_edge_base`, `model.chi_edge_drop`
- `model.latent_gain`
- `model.source_scale`

---

## 12. Data inventory (kept as in original; included for completeness)

### 12.1 Primary observables
- `Te`: Thomson scattering; target variable.
- `Te_mask`: validity mask.
- `ne`: Thomson scattering; input to the source network.

### 12.2 Actuators and control inputs
- `P_nbi`, `Ip`, `nebar`, `S_gas`, `S_rec`, `S_nbi` (dense 1D signals; z-scored + clipped; interpolated).
- `P_rad` not used in current training script.

### 12.3 Geometry
- `rho`: flux coordinate.
- `Vprime`: $V'(\rho)=dV/d\rho$; used in conservative discretisation.

### 12.4 Optional scalars (ingested when present; not used by default)
`W_tot`, `P_ohm`, `P_tot`, `H98`, `beta_n`, `B_t0`, `q95`, `li`, `P_rad`.

---

## 13. References (unchanged)

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
