# Physics-Consistent Transport ROM (Intersection Grid)

**Status:** Active – implemented in `train_tokamak_ode_hpc.py` and `fusion_ode_identification/*`; compatible with transport analysis in TORAX [4].

## Overview

This document describes a **physics-informed neural ordinary differential equation (ODE)** approach to learning reduced-order models of tokamak electron temperature transport. Unlike purely data-driven sequence models (transformers, recurrent networks) or latent dynamics models (world models, state-space models), our approach explicitly encodes the *structure* of the underlying partial differential equation (PDE) governing heat diffusion in a magnetized plasma.

The model evolves temperature profiles $\mathbf{T}(t)$ on a reduced radial grid via a stiff ODE system that combines:
- A **physics-based diffusion operator** derived from finite-volume discretization of the heat equation, with interpretable transport coefficients,
- A **learned residual source term** (small neural network) that captures unmodeled physics (turbulent heating, atomic processes),
- A **low-dimensional latent variable** $z(t)$ that modulates edge transport, capturing regime transitions (L-mode ↔ H-mode).

This hybrid structure offers significant advantages for real-time control, state inference, and bifurcation analysis (see §1.2 below).

---

## 0. Scope and notation

- Radial coordinate: $\rho\in[0,1]$.
- Non-uniform ROM grid: $\boldsymbol{\rho}_{\text{rom}}\in\mathbb{R}^{N_{\text{rom}}}$ (intersection-supported + interior nodes).
- Observed grid (intersection across shots): $\boldsymbol{\rho}_{\cap}=\boldsymbol{\rho}_{\text{rom}}[\mathcal{I}_{\cap}]$.
- State: electron temperature profile $\mathbf{T}(t)\in\mathbb{R}^{N_{\text{rom}}}$; latent scalar $z(t)\in\mathbb{R}$.
- Density: $\mathbf{n}_e(t)\in\mathbb{R}^{N_{\text{rom}}}$ (regridded to $\boldsymbol{\rho}_{\text{rom}}$).
- Actuators: $\mathbf{u}(t)\in\mathbb{R}^M$ on the summary grid, interpolated to the profile grid.
- Geometry: $V'(\rho)$ on $\boldsymbol{\rho}_{\text{rom}}$ (equilibrium when available; otherwise we detect slab-like constants and fall back to toroidal $V'(\rho)=2\rho$ with a Safe Core floor).
- Core Singularity Fix: $V'(\rho \to 0)$ is clamped to a small finite value to prevent numerical instability in the finite volume solver.

Bold symbols denote discrete vectors on $\boldsymbol{\rho}_{\text{rom}}$. Norms are Euclidean; expectations are empirical over batches and time.

---

## 1. Model Architecture and Advantages

### 1.1 What Are the Alternatives? (Plain Language)

Before describing our approach, we briefly survey common machine learning paradigms for modeling dynamical systems:

**Transformers** (e.g., GPT-style models [11]) are sequence-to-sequence architectures that use *attention mechanisms*—a learned weighted averaging over past observations—to predict future states. Think of them as sophisticated pattern-matching machines: given a history of temperature profiles $(\mathbf{T}_{t-k}, \dots, \mathbf{T}_{t-1})$, a transformer learns to predict $\mathbf{T}_t$ by finding similar patterns in training data. They excel at long-range dependencies (e.g., "remember what happened 100 time steps ago") but do not explicitly model causality or physical laws.

**World Models** [12] learn a *compressed latent representation* of the system state (often via a variational autoencoder) and then model dynamics in that latent space. The idea: map high-dimensional profiles $\mathbf{T}\in\mathbb{R}^{N}$ to a low-dimensional code $\mathbf{h}\in\mathbb{R}^d$ ($d\ll N$), evolve $\mathbf{h}$ via a recurrent neural network (RNN) or other black-box function, and decode back to physical space. This is efficient for planning and control in robotics/games, but the latent space is typically *uninterpretable*: we lose direct access to physical quantities like diffusivity or heat flux.

**State-Space Models (SSMs)** [13] impose linear (or mildly nonlinear) dynamics on a hidden state:
$$
\mathbf{h}_{t+1} = A\mathbf{h}_t + B\mathbf{u}_t + \mathbf{w}_t,\qquad \mathbf{y}_t = C\mathbf{h}_t + \mathbf{v}_t,
$$
where $\mathbf{h}$ is a latent state, $\mathbf{u}$ are inputs (actuators), $\mathbf{y}$ are observations (sensor data), and $\mathbf{w},\mathbf{v}$ are noise terms. Modern variants (S4, Mamba [14]) use structured state matrices for efficient long-context modeling. However, the hidden state $\mathbf{h}$ is abstract: it does not correspond to physical temperature profiles or fluxes, making interpretation and physics-based analysis difficult.

### 1.2 Why Physics-Informed ODEs? (Dynamical Systems Perspective)

Our approach differs fundamentally: we model the evolution of the *physical state* $\mathbf{T}(t)$ directly via an ODE that mirrors the continuous PDE:
$$
\frac{d\mathbf{T}}{dt} = \underbrace{\mathbf{D}(\mathbf{T}; \chi)}_{\text{diffusion (known physics)}} + \underbrace{\mathbf{S}_{\theta}(\mathbf{T}, \mathbf{u}, z)}_{\text{learned residual}},\qquad \frac{dz}{dt} = f_z(\mathbf{u}, z).
$$
Here $\mathbf{D}$ is a finite-volume discretization of the diffusion operator $\nabla\cdot(V'\chi\nabla T)$, with interpretable transport coefficients $\chi(\rho,z)$, and $\mathbf{S}_{\theta}$ is a small neural network capturing unmodeled source terms.

**Advantages from a dynamical systems viewpoint:**

1. **Phase-space geometry and stability.** Because the state is the physical profile $\mathbf{T}\in\mathbb{R}^N$, we can directly apply tools from nonlinear dynamics:
   - **Lyapunov analysis:** Check if the learned system $\dot{\mathbf{y}}=f(\mathbf{y},\mathbf{u})$ has stable equilibria by linearizing $Df$ and examining eigenvalues. For transformers or world models, the "state" is a hidden embedding with no physical meaning, so Lyapunov stability is ill-defined.
   - **Invariant manifolds:** Transport PDEs often exhibit slow manifolds (e.g., stiff-layer equilibria [5,6]). Our ODE inherits this structure; black-box models do not.
   - **Bifurcation analysis:** We can perform parameter continuation on $\chi(\rho,z)$ or the latent dynamics $f_z$ to identify L→H transitions, critical control thresholds, etc. (see §1.2.4 below). For a transformer, "bifurcations" have no clear interpretation—there is no continuous parameter to vary.

2. **Real-time control and state inference.** 
   - **Computational efficiency:** Evaluating the RHS of our ODE is a single forward pass through a small MLP plus a sparse matrix–vector product (diffusion stencil). This takes ~1 ms on a CPU. A transformer must process the entire history $(\mathbf{T}_{t-k},\dots,\mathbf{T}_{t-1})$ at every step, with quadratic cost in sequence length ($O(k^2)$ for attention). For real-time control loops (~1 kHz), transformers are too slow.
   - **Observability and filtering:** Standard control theory provides tools to estimate unobserved interior states from edge measurements (Kalman filtering, moving-horizon estimation). For a physics-informed ODE, observability is tied to the diffusion operator: edge sensors "see" the interior via heat conduction. For a world model, the latent $\mathbf{h}$ is not directly observable, and there is no physics-based propagator to guide estimation.
   - **Interpretability:** Control engineers can inspect $\chi(\rho,z)$, adjust diffusivity floors, or tune the source $\mathbf{S}_{\theta}$ based on domain knowledge. A transformer's 100M+ parameters are opaque.

3. **Extrapolation and regime changes.**
   - Because diffusion is *explicitly modeled*, the system correctly captures transport even in regimes not seen during training (e.g., higher power, different geometry) as long as the residual source $\mathbf{S}_{\theta}$ remains small. Transformers rely on pattern-matching: if a new scenario is out-of-distribution, they fail unpredictably.
   - The latent $z(t)$ provides a **low-dimensional order parameter** for regime transitions. Bifurcation theory tells us when small changes in actuators $\mathbf{u}$ can trigger large jumps in $z$ (e.g., L→H transition). This is a *dynamical systems* concept; transformers have no equivalent.

4. **Bifurcation analysis and control design (detailed).**
   - **Parameter continuation:** By treating $z$ or control inputs $\mathbf{u}$ as bifurcation parameters, we can use numerical continuation methods (AUTO, MATCONT [15]) to trace solution branches, locate saddle-node or Hopf bifurcations, and identify hysteresis in the L/H transition.
   - **Critical slowing down:** Near a bifurcation, the dominant eigenvalue $\lambda_1(Df)\to 0$, causing recovery times to diverge. This is a *universal* signature of tipping points in physics [16]. For a world model, there is no notion of "dominant eigenvalue" in latent space.
   - **Control-oriented design:** If we know the system is near a saddle-node (via continuation), we can design feedback $\mathbf{u}(t)$ to stabilize unstable branches or avoid bifurcations entirely. This is impossible with black-box models.

### 1.3 Mathematical Justification (Why Hybrid Works)

Formally, the true transport PDE is:
$$
\frac{\partial T_e}{\partial t} = \frac{1}{V'(\rho)}\frac{\partial}{\partial\rho}\left(V'(\rho)\chi_{\text{turb}}(\rho,T_e,n_e,\mathbf{u})\frac{\partial T_e}{\partial\rho}\right) + S_{\text{total}}(\rho,T_e,n_e,\mathbf{u}),
$$
where $\chi_{\text{turb}}$ and $S_{\text{total}}$ are complex, closure-unknown functionals of the state. Standard transport codes (TRANSP, TORAX [4]) model these via empirical formulas (Bohm/gyroBohm scaling, etc.). Our approach *splits* the physics:
- **Known component:** We use a *simplified* diffusivity $\chi(\rho,z)$ that captures the *qualitative* structure (core vs. edge, pedestal localization) but not the full microturbulence. The latent $z$ parameterizes regime-dependent changes (L-mode: high $\chi$; H-mode: low edge $\chi$).
- **Residual learning:** The network $\mathbf{S}_{\theta}$ learns the *difference* between the true source and what the simplified diffusion alone would produce. Because diffusion already captures the dominant transport mechanism, $\mathbf{S}_{\theta}$ can be small and generalizes better than a purely black-box model.

This is a **universal approximation** result in disguise [17]: any smooth dynamics can be written as
$$
\dot{\mathbf{y}} = \mathbf{f}_{\text{known}}(\mathbf{y}) + \mathbf{f}_{\text{residual}}(\mathbf{y}),
$$
where $\mathbf{f}_{\text{known}}$ is a physics prior and $\mathbf{f}_{\text{residual}}$ is approximated by a neural network. The key insight: if $\mathbf{f}_{\text{known}}$ is a *good* prior, the residual is small and easy to learn, leading to better sample efficiency and extrapolation than purely data-driven methods.

---

## 2. Pipeline Overview

### 2.1 Data and Grid Construction

1) Build an observed intersection grid $\boldsymbol{\rho}_{\cap}$: keep columns whose *minimum across shots* mean Thomson mask coverage is above `data.intersection_rho_threshold`. If too few, fall back to top-$k$ columns by that minimum coverage.

2) Build a non-uniform ROM grid $\boldsymbol{\rho}_{\text{rom}}$: take $\boldsymbol{\rho}_{\cap}$, add interior nodes in $[0,\min \boldsymbol{\rho}_{\cap}]$ (Chebyshev by default), and always include $0$. The last ROM node is treated as the boundary node:
  - If `data.edge_bc_mode = use_last_observed` (default), the boundary node is the *outermost observed* point (typically $\rho<1$).
  - If `data.edge_bc_mode = extrapolate_to_1`, we explicitly include $\rho=1$ as the boundary node.

3) Regrid per-shot profiles/masks/geometry to $\boldsymbol{\rho}_{\text{rom}}$; carry the mapped observation indices `obs_idx` for the data term. To keep the BC consistent, `obs_idx` is clamped so it never includes the boundary column (the last ROM node).

  Edge BC (`Te_edge(t)`) is built as a 1D time trace and then time-interpolated to fill gaps:
  - `use_last_observed`: take the value at the outermost observed index (per time) when masked/finite.
  - `extrapolate_to_1`: use the last two observed points to linearly extrapolate to $\rho=1$ when possible.

4) Evolve a stiff transport ODE on $\boldsymbol{\rho}_{\text{rom}}$ with a neural residual source and a latent $z$ that modulates edge diffusivity via a sigmoid drop.
   - **Safe Core (Toroidal Geometry)**: The core volume element $V'[0]$ is clamped to a small finite value (derived from the first neighbor $V'[1]$) to avoid division by zero while preserving the toroidal geometry ($V' \propto \rho$) elsewhere. This avoids the need for a slab approximation ($V'=1$) and allows for physically consistent transport coefficients.

5) Train with a robust composite loss: pseudo-Huber data term on $\boldsymbol{\rho}_{\cap}$, source magnitude penalty, weak-constraint model-error penalty, optional regime supervision on $z$, and small $z$ regularisation. No manifold or latent subspace penalties remain.

### 2.2 Why This Grid Strategy?
- If psi-based $\rho$ or $V'$ are missing, we fall back to linear-in-$R$ $\rho\in[0,1]$; for $V'$, we detect slab-like $V'$ and use $2\rho$ (cylindrical/toroidal approximation). Flags (`rho_fallback_used`, `psi_axis`, `psi_edge`) remain in packs for audit.
- When better equilibria become available, re-run the packer to restore flux-aligned $\rho$ and $V'$; the ROM/intersection construction remains unchanged.

### 2.3 Robustness to Experimental Artifacts and Sparse Observations

Experimental tokamak data is sparse, noisy, and occasionally corrupted (Thomson dropouts, bad equilibria). The data term is applied only on the observed intersection grid $\boldsymbol{\rho}_{\cap}$ with masks $m_i(t)$ and per-column coverage weights $c_i$. Invalid points (NaNs/outliers) are removed before interpolation; the solver is free to coast through temporal gaps rather than fitting artifacts.

**Why an intersection grid (and no latent subspace)?**
- A pure mask on the full grid leaves interior nodes weakly constrained and harms identifiability. The common intersection $\boldsymbol{\rho}_{\cap}$ keeps only columns observed across shots, giving a well-conditioned data term.
- We still evolve the full non-uniform ROM grid $\boldsymbol{\rho}_{\text{rom}}$ that contains $\boldsymbol{\rho}_{\cap}$ plus added interior nodes; unobserved nodes are regularised indirectly via diffusion and the source prior—no low-rank projector is used.

### Data sparsity and observability (all channels)

- Profiles: `Te`, `ne` with masks on the Thomson grid; edge-biased coverage. The data loss uses only $\mathcal{I}_{\cap}$ with mask/coverage weights.
- Controls: `P_nbi`, `Ip`, `nebar`, `S_gas`, `S_rec`, `S_nbi` are dense 1D; z-scored and clipped. Note: `P_rad` is not used in the current training script.
- Optional scalars: `W_tot`, `P_ohm`, `P_tot`, `H98`, `beta_n`, `B_t0`, `q95`, `li` included only when finite and aligned.
- Geometry: equilibrium $\rho$, $V'$ when present; otherwise fallbacks as above.

### Expected vs. effective dimensions (current packs)

- Nominal radial nodes $N\approx65$ on the reference grid; $|\mathcal{I}_{\cap}|$ typically 12–20 edge-biased columns. ROM grid size $N_{\text{rom}} = |\boldsymbol{\rho}_{\text{rom}}|$ adds $m$ interior Chebyshev nodes plus boundaries.
- Thomson time samples ~90–120; controls ~1.7–2.3k, interpolated to the profile grid.

---

## 3. Data, Geometry, and Grids

### 3.1 Equilibrium and Fallbacks
- If equilibrium $\rho, V'$ are present, they are regridded to the common reference grid and then to $\boldsymbol{\rho}_{\text{rom}}$.
- If missing, $\rho$ falls back to linear-in-$R$ on $[0,1]$; for $V'$, we apply a toroidal fallback $V'(\rho)=2\rho$ when slab-like constants are detected, and clamp the core ($V'[0]$) to a small positive floor for stability.

### 3.2 Quasi-Static Geometry (Justification for Time-Independent $V'$)

In this framework, we treat the equilibrium metric $V'(\rho)$ as **quasi-static** (constant in time) within each discharge. Specifically, we extract $V'(\rho)$ from a representative equilibrium reconstruction (typically at mid-discharge via `choose_itime`) and hold it fixed during integration. This design choice is justified by several considerations spanning physics, numerics, and identifiability.

#### 3.2.1 Physical Justification

**Slow equilibrium evolution.** In the discharges considered (primarily L↔H transition windows), plasma shape, current, and pressure evolve on transport timescales (100s of ms to seconds), which are slow compared with the electron heat diffusion timescale (~10–100 ms). Over the training windows (typically 1–3 seconds), changes in $V'(\rho)$ are modest and smooth, making a time-averaged geometry a reasonable first-order approximation.

**Separation of scales.** The true transport PDE in flux coordinates is
$$
\frac{\partial T_e}{\partial t} = \frac{1}{V'(\rho,t)}\frac{\partial}{\partial\rho}\left(V'(\rho,t)\,\chi(\rho,z,t)\,\frac{\partial T_e}{\partial\rho}\right) + S(\rho,t).
$$
If equilibrium changes are small, we can write $V'(\rho,t) = V_0'(\rho) + \varepsilon\,\delta V'(\rho,t)$ with $\varepsilon \ll 1$. Then
$$
\mathcal{L}[T_e; V'(t)] = \mathcal{L}[T_e; V_0'] + \varepsilon\,\Delta\mathcal{L}[T_e; \delta V', V_0'] + O(\varepsilon^2).
$$
The leading-order dynamics are captured by the constant $V_0'$; time-dependent corrections enter as a small model error that can be absorbed by:
- mild adjustments to the learned diffusivity $\chi(\rho,z)$,
- the residual source term $S_{\text{net}}(\rho,T_e,n_e,\mathbf{u},z)$,
- and/or the optional weak-constraint model-error penalty (if enabled).

This is a legitimate reduced-order modeling approximation: we prioritize robust, interpretable transport coefficients over chasing noisy equilibrium fluctuations.

#### 3.2.2 Numerical Stability and Robustness

**Equilibrium reconstruction jitter.** Real-time EFIT reconstructions can exhibit frame-to-frame noise, occasional dropouts, or inconsistencies (e.g., axis/edge crossings, NaN values, slab-like fallbacks). Injecting $V'(\rho,t)$ directly into a stiff ODE solver without aggressive smoothing risks:
- Spurious gradients in time that force the adaptive solver to shrink timesteps,
- Amplification of reconstruction artifacts in the diffusion operator,
- Training instabilities when the RHS becomes rough or non-smooth in $t$.

**Safe core and metric floors.** Our "safe core" strategy (§5) clamps $V'[0]$ to avoid division by zero near the magnetic axis. This is easier to enforce and validate with a single, well-curated $V'(\rho)$ per shot than with a noisy time series requiring per-frame clamping and interpolation.

**Stiffness control.** The dominant numerical stiffness arises from diffusion on a non-uniform grid: $|\lambda_{\max}| \sim \chi/(\Delta\rho_{\min})^2$. While $V'$ affects the metric scaling, it is not the primary stiffness driver. Using a stable, time-averaged $V'$ prevents additional time-variation from destabilizing the solver without fundamentally changing the stiffness class of the problem.

#### 3.2.3 Identifiability and Interpretability

**Avoiding degeneracies.** The model has at least three "knobs" that can compensate for discrepancies between model and data:
1. Geometry metric $V'(\rho)$,
2. Diffusivity $\chi(\rho,z)$,
3. Residual source $S_{\text{net}}$.

If all three are allowed to vary freely (especially if $V'$ is time-dependent and noisy), the decomposition becomes non-unique: the network can "explain away" equilibrium errors by adjusting $\chi$ or $S_{\text{net}}$ in ways that lose physical meaning. By fixing $V'$ to a robust per-shot average, we:
- Constrain one degree of freedom, improving identifiability,
- Encourage the model to learn interpretable transport physics ($\chi$) rather than fitting geometry noise,
- Keep $S_{\text{net}}$ focused on true residual physics (unmodeled sources/sinks, closure errors) rather than becoming a "sponge" for equilibrium reconstruction artifacts.

**What does $S_{\text{net}}$ represent?** With quasi-static $V'$, the residual source absorbs:
- Unmodeled turbulent heating/transport closure errors,
- Atomic physics (ionization, radiation) not explicitly in the diffusion term,
- Residual geometry errors (small time-dependent $V'$ effects),
- Diagnostic artifacts (e.g., Thomson calibration drifts).

This is an honest "lumped uncertainty" model: we explicitly trade off geometry fidelity for robustness and interpretability. The alternative—using fully time-dependent $V'(t)$ without strong constraints on $S_{\text{net}}$—can yield better point predictions but at the cost of interpretability and extrapolation.

#### 3.2.4 Comparison with Alternatives

| Approach | Pros | Cons |
|----------|------|------|
| **Quasi-static $V'$ (ours)** | Stable integration; interpretable $\chi$; robust to EFIT jitter; clear identifiability; fast training | Small geometry errors absorbed by residual; not "maximally accurate" for shots with large shape changes |
| **Time-dependent $V'(t)$** | Captures true equilibrium evolution; potentially higher fidelity | Requires smooth, high-quality EFIT time series; can inject noise; solver may struggle with jitter; risk of $S_{\text{net}}$ learning equilibrium artifacts; harder bifurcation analysis |
| **TORAX-style implicit geometry** | Handles time-varying geometry via implicit solve; modular sources | Heavier solver machinery; not directly compatible with our sparse-data objective; identifiability still an issue if $\chi$ and $V'$ both vary |

#### 3.2.5 Current Implementation

- **Extraction:** `volume_derivatives(eq, itime)` computes $V'(\rho)$ from `flux_surface_volume` at a representative time index (typically mid-discharge, `itime = len(time) // 2`).
- **Fallback:** If equilibrium data is unavailable or unreliable, we fall back to a toroidal approximation $V'(\rho) = 2\rho$ (with safe-core clamping).
- **Regridding:** $V'$ is interpolated onto the ROM grid $\boldsymbol{\rho}_{\text{rom}}$ once per shot and held constant during integration.
- **Future work:** We may implement a smoothed time-dependent $V'(t)$ (low-pass filtered, gap-filled, with strict positivity) as an ablation study to quantify the trade-off between geometry fidelity and training stability.

#### 3.2.6 Verification and Sensitivity

To validate the quasi-static approximation, we recommend:
1. **Post-hoc check:** Compare the single $V'(\rho)$ used in training to the full time series $V'(\rho,t)$ available from EFIT; compute $\max_t \|V'(\cdot,t) - V_0'(\cdot)\|/\|V_0'\|$ to quantify time-variation.
2. **Ablation study:** Train with (a) constant fallback $V' = 2\rho$, (b) per-shot EFIT $V'$, and (c) smoothed time-dependent $V'(t)$; compare data loss, $\|S_{\text{net}}\|$, and solver step counts.
3. **Residual correlation:** Check if $S_{\text{net}}$ correlates with $\partial_\rho T_e$ (which would suggest it's compensating for diffusion/geometry errors) vs. being spatially uncorrelated (true source-like behavior).

If equilibrium evolution is found to be significant and smooth, future versions may adopt IMEX time-stepping (§15) with interpolated $V'(t)$, which handles time-varying coefficients cleanly while maintaining numerical stability.

---

### 3.3 Intersection Observed Set
Let $c_i$ be the mean mask coverage of column $i$ for shot $s$, and $\bar{c}_i$ the minimum across shots. With threshold $\tau_{\cap}=\texttt{data.intersection\_rho\_threshold}$,
$$
\mathcal{I}_{\cap}=\{ i \mid \bar{c}_i \ge \tau_{\cap}\}.
$$
If $|\mathcal{I}_{\cap}|$ is small, choose the top-$k$ columns by $\bar{c}_i$ (sorted) to ensure a usable observed grid.

### 3.4 ROM Grid Construction
- Interior nodes: $m=\texttt{data.rom\_n\_interior}$ Chebyshev-like points in $(0,\rho_{\cap,\min})$:
  $$
  x_k = 0.5\,(1-\cos(\pi (k+1)/(m+1))),\quad \rho_k = \rho_{\cap,\min}\, x_k,
  $$
  for $k=0,\dots,m-1$.
- ROM grid:
  $$
  \boldsymbol{\rho}_{\text{rom}} = \text{unique\_sort}\big(\{0\}\cup\{\rho_k\}\cup\boldsymbol{\rho}_{\cap}\cup\{1\}^{*}\big),
  $$
  where $\{1\}^{*}$ is included only when `data.edge_bc_mode = extrapolate_to_1`.
- Observed indices $\mathcal{I}_{\cap}$ are mapped into $\boldsymbol{\rho}_{\text{rom}}$ by nearest match (exact grid points after construction).

### 3.5 Regridding Profiles and Masks
For each shot, interpolate $T_e$, $n_e$, and masks from the reference grid to $\boldsymbol{\rho}_{\text{rom}}$ using only finite points per time slice; masks are interpolated as floats then thresholded $>0.5$. The observed view is a slice on $\mathcal{I}_{\cap}$.

---

## 4. Governing Model

We evolve temperature and a scalar latent:
$$
\frac{d \mathbf{T}}{dt} = \mathbf{D}(\boldsymbol{\rho}_{\text{rom}}, V')\,\mathbf{T} + \mathbf{S}_{\text{net}}(\boldsymbol{\rho}_{\text{rom}}, \mathbf{T}, \mathbf{n}_e, \mathbf{u}, z),
$$
$$
\frac{dz}{dt} = \alpha(\mu(\mathbf{u})-\mu_{\text{ref}}) - \beta z - \gamma z^3,
$$
with $\alpha,\beta,\gamma>0$ enforced by softplus. In the current implementation, $\mu(\mathbf{u})$ depends on the first three controls (`P_nbi`, `Ip`, `nebar`). The source network is a node-wise MLP on $(\rho, T, n_e, \mathbf{u}, z)$ with tanh activations and zero-initialised output layer for stability. Edge diffusivity drops with $z$ via $\chi_{\text{edge}}(z) = \chi_{\text{edge,base}} - \chi_{\text{edge,drop}}\,\sigma(k z)$; $\chi$ is blended across $\rho$ by a sigmoid pedestal [1].

The evolved state in the solver is the interior temperature nodes (scaled) plus $z$; the edge value $T_{\text{edge}}$ is fixed from masked data and appended during reconstructions.

### 4.1 Continuous Framing

The continuous model is diffusion with a latent-modulated diffusivity and a neural residual source:
$$
\frac{\partial T_e}{\partial t} = \frac{1}{V'(\rho)}\frac{\partial}{\partial \rho}\left(V'(\rho)\,\chi(\rho,z)\,\frac{\partial T_e}{\partial \rho}\right) + \mathcal{S}_{\text{net}}(\rho, T_e, n_e, \mathbf{u}, z),
$$
where $\chi$ has a sigmoid pedestal and edge drop $\chi_{\text{edge}}(z) = \chi_{\text{edge,base}} - \chi_{\text{edge,drop}}\,\sigma(k z)$. The latent ODE is $\dot z = f_z(\mathbf{u}, z)$ as above. This is discretised to the non-uniform FVM operator $\mathbf{D}$ plus the Nemytskii source $\mathbf{S}_{\text{net}}$.

### 4.2 Nemytskii Source Operator

The source acts pointwise:
$$
(\mathbf{S}_{\text{net}})_i = R_\theta(\rho_i, T_i, n_{e,i}, \mathbf{u}, z),
$$
with a small MLP $R_\theta$ (tanh, zero init on output) to ensure the residual starts near zero and learns spatially varying corrections.

### 4.3 Semi-Discrete Hybrid ODE

Combining diffusion and source yields the ODE used in training and inference:
$$
\frac{d\mathbf{T}}{dt} = \mathbf{D}\mathbf{T} + \mathbf{b}_{\text{edge}} + \mathbf{S}_{\text{net}}(\boldsymbol{\rho}_{\text{rom}}, \mathbf{T}, \mathbf{n}_e, \mathbf{u}, z),
$$
with fixed boundary contribution $\mathbf{b}_{\text{edge}}$ derived from the boundary trace $T_{\text{edge}}(t)$.

Implementation notes (current code):
- State is stored as $y=[\hat{\mathbf{T}}_{\text{int}}, z]$ where $\hat{\mathbf{T}}=\mathbf{T}/\texttt{Te\_scale}$.
- At each RHS evaluation we reconstruct the full profile by appending the boundary value from an interpolant: $\mathbf{T}=[\hat{\mathbf{T}}_{\text{int}}\,\texttt{Te\_scale},\; T_{\text{edge}}(t)]$.
- **Training** solves over the padded time grid (to keep `SaveAt(ts=...)` strictly valid) and masks losses using `t_len`.
- **Debug/eval** solves only over the valid window (no padding) for clarity.

---

## 5. Spatial Discretisation (Non-Uniform FVM)

- Faces: indices run over ROM nodes $i=0,\dots,N-1$ with a boundary node at $i=N-1$.
- Face spacings: $\Delta\rho_i = \rho_{i+1}-\rho_i$ for $i=0,\dots,N-2$.
- Face-averaged coefficients:
  $$
  \chi_{i+1/2}=\tfrac12(\chi_i+\chi_{i+1}),\qquad V'_{i+1/2}=\tfrac12(V'_i+V'_{i+1}).
  $$
- Conservative face flux (current implementation):
  $$
  F_{i+1/2} = -V'_{i+1/2}\,\chi_{i+1/2}\,\frac{T_{i+1}-T_i}{\Delta\rho_i}.
  $$
- Conservative divergence on interior cells (returns $N-1$ values for $i=0,\dots,N-2$):
  $$
  (\nabla\cdot F)_i = -\frac{F_{i+1/2}-F_{i-1/2}}{\bar V'_i\,\Delta\rho_i},\qquad \bar V'_i=\tfrac12(V'_i+V'_{i+1}),
  $$
  with the axis Neumann condition enforced by setting $F_{-1/2}=0$.

Stability floors used in code:
- $\Delta\rho_i$ is floored to avoid divisions by tiny spacings.
- The effective cell “volume” $\bar V'_i\,\Delta\rho_i$ is floored near the core (based on a small fraction of the max volume).
- $V'$ is clipped to be positive and a “safe core” floor is applied so $V'[0]$ is not singular.

Boundary conditions:
- Axis: zero-flux (Neumann) via $F_{-1/2}=0$.
- Outer boundary: Dirichlet using the supplied $T_{\text{edge}}(t)$ at the last ROM node (this last node is not necessarily $\rho=1$ unless `edge_bc_mode=extrapolate_to_1`).

---

## 6. Loss Terms (No Manifold)

- **Data term (pseudo-Huber):** on $\mathcal{I}_{\cap}$ only, with per-column coverage weights $c_i$ and time mask $m_i(t)$:
  $$
  \mathcal{L}_{\text{data}} = \frac{\sum_{t,i\in\mathcal{I}_{\cap}} m_i(t)\,c_i\,\phi_\delta(T_{\text{model},i}(t)-T_{\text{obs},i}(t))}{\sum_{t,i\in\mathcal{I}_{\cap}} m_i(t)\,c_i+10^{-8}},
  $$
  where $\phi_\delta$ is pseudo-Huber with $\delta=\texttt{loss.huber\_delta}$.
- **Source magnitude penalty:** $\lambda_{\text{src}} \mathbb{E}\big[\|\mathbf{S}_{\text{net}}\|_2^2\big]$ to keep $z$ informative.
- **Weak-constraint model error (optional):** pseudo-Huber on one-step defects $w_k = T_{k+1}-(T_k + \Delta t\,f_k)$ with weight $\lambda_w$ and scale $\texttt{loss.model\_error\_delta}$. Disabled by default in the current training script.
- **Regime supervision (optional):** MSE on $\sigma(z)$ vs. binary regime labels when present, weighted by $\lambda_z$.
- **Latent regularisation:** small $\|z\|_2^2$ term, weighted by $\lambda_{z\text{reg}}$ from config (no latent subspace penalty in the current ROM).

Total loss:
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{src}}\mathcal{L}_{\text{src}} + \lambda_w\mathcal{L}_{\text{model}} + \lambda_z\mathcal{L}_{z\text{-sup}} + \lambda_{z\text{reg}}\,\mathbb{E}[z^2].
$$

---

## 7. Data Handling and Preprocessing

- Masks: drop NaNs; masks follow Thomson grid, then regridded to $\boldsymbol{\rho}_{\text{rom}}$.
- Edge BC: masked mean on the edge column; fallback to 50 eV if undefined. Boundary values are linearly interpolated over time gaps on the shot’s time grid.
- Initial condition: first valid profile on $\boldsymbol{\rho}_{\text{rom}}$; fallback synthetic $T(\rho)=100(1-\rho^2)+10$ when missing.
- Clipping: $n_e\in[10^{17},10^{21}]$ before scaling; $T_e$ and controls are normalised/clipped to avoid outliers.

---

## 8. Configuration Knobs

- `data.intersection_rho_threshold`: coverage threshold for the observed intersection.
- `data.rom_n_interior`, `data.rom_interior_strategy`: interior node count/placement.
- `data.min_rho_coverage`: optional global prune before intersection (rarely needed now).
- `loss.huber_delta`, `loss.model_error_delta`, `training.lambda_src`, `training.lambda_w`, `training.lambda_z`.
- `model.latent_gain`, `model.chi_*`: latent-edge coupling and diffusivity envelope.

---

## 9. Logging and Diagnostics

- Report $|\boldsymbol{\rho}_{\text{rom}}|$, $|\boldsymbol{\rho}_{\cap}|$, their min/max, and the actual values at startup.
- Per-shot $\text{mask\_mean}_{\cap}$ to verify intersection coverage.
- Recommended debug mode: integrate one shot, dump $T_{\text{model}}(t,\boldsymbol{\rho}_{\cap})$ vs. data and $z(t)$.

---

## 10. Numerical Stability

- Solver: `Kvaerno5` with PID control by default (`rtol=atol=1e-3` by default), with an optional IMEX `KenCarp3` path. The solver evaluates on `SaveAt(ts=...)` and is capped at `MAX_SOLVER_STEPS=50000` in code.
- Training integrates over the padded time grid and masks all losses using `t_len` (valid-window mask). Debug/eval integrates only over the valid window.
- Derivative clipping on $dT/dt$ to $10^4$ (soft clipping).
- Non-uniform FVM widths to avoid negative cells on stretched grids; Neumann at the axis enforced via zero left flux.
- **Safe Core**: The core volume element $V'[0]$ is clamped to avoid division by zero, preserving toroidal geometry elsewhere.

---

## 11. Data Inventory (Physical Quantities and Their Role)

This section describes the experimental measurements and actuator signals used in training, with brief explanations of their physical meaning and role in tokamak operation.

### Primary Observables (State Variables)

| Quantity | Source | Physical Meaning | Role in Tokamak | Usage in Model |
| --- | --- | --- | --- | --- |
| `Te` | Thomson scattering | **Electron temperature** (keV): kinetic energy of electrons in the plasma. Hot electrons ($T_e \sim 1$–10 keV) are essential for fusion reactions. | Core heating and confinement quality indicator; edge $T_e$ drops at pedestal (H-mode). | **Target variable** for supervised loss; model predicts $T_e(t,\rho)$. |
| `Te_mask` | Thomson scattering | **Validity mask**: identifies spatial and temporal points where Thomson data is reliable (not corrupted by noise, dropouts, or reflections). | Thomson diagnostics are sparse and noisy; masks exclude bad measurements. | Applied to data loss term to avoid fitting artifacts. |
| `ne` | Thomson scattering | **Electron density** ($10^{19}$ m$^{-3}$): number of electrons per unit volume. High $n_e$ increases fusion rate ($P_{\text{fusion}} \propto n_e^2$) but can degrade confinement. | Fueling control (gas puffing) adjusts $n_e$; density limit constrains operation. | **Input feature** (time-dependent profile) for source network $\mathbf{S}_{\text{net}}(\rho,T_e,n_e,\mathbf{u},z)$. |

### Actuators and Control Inputs

| Quantity | Source | Physical Meaning | Role in Tokamak | Usage in Model |
| --- | --- | --- | --- | --- |
| `P_nbi` | NBI system logs | **Neutral Beam Injection power** (MW): energetic neutral atoms injected into plasma; ionize and deposit energy/momentum. | Primary auxiliary heating method; increases $T_e$, $T_i$, drives current, triggers L→H transitions. | Input to latent dynamics $\mu(\mathbf{u})$ and source network; dominant control actuator. |
| `Ip` | Magnetic diagnostics | **Plasma current** (MA): toroidal current flowing in plasma. Generates poloidal magnetic field for confinement (tokamak principle). | Higher $I_p$ improves confinement (via magnetic shear, $q$-profile); affects MHD stability. | Input to $\mu(\mathbf{u})$ (regime transitions) and source network; quasi-steady-state in most shots. |
| `nebar` | Interferometry | **Line-averaged density** ($10^{19}$ $m^{-3}$): line integral of $n_e$ along chord; proxy for total particle inventory. | Real-time feedback control target; determines Greenwald fraction $n_e/n_G$ (density limit). | Input to $\mu(\mathbf{u})$ and source network; complements radial $n_e$ profile from Thomson. |
| `S_gas` | Gas injection system | **Gas puffing rate** (particles/s or Torr·L/s): deuterium gas injected at edge for fueling. | Increases plasma density; controls $n_e$ evolution and particle inventory. | Input to source network; affects edge particle balance and $n_e$ profile evolution. |
| `S_rec` | Derived/modeled | **Recycling source** (particles/$s/m^3$): neutral influx from wall interactions (ionization of recycled neutrals). | Plasma-wall equilibrium; spontaneous edge fueling; affects density control. | Input to source network; represents edge particle source not directly controlled. |
| `S_nbi` | NBI system + NUBEAM | **NBI particle source** (particles/$s/m^3$): ionization profile of injected neutrals; provides both heating and fueling. | Beams deposit particles centrally; contributes to core $n_e$ and dilution. | Input to source network; couples heating and particle transport. |

### Geometry (Magnetic Equilibrium)

| Quantity | Source | Physical Meaning | Role in Tokamak | Usage in Model |
| --- | --- | --- | --- | --- |
| `rho` | EFIT equilibrium reconstruction | **Normalized radial coordinate** ($\rho \in [0,1]$): flux surface label, typically $\rho = \sqrt{(\psi - \psi_{\text{axis}})/(\psi_{\text{edge}} - \psi_{\text{axis}})}$ where $\psi$ is poloidal flux. | Maps 3D toroidal geometry to 1D; profiles $T_e(\rho)$, $n_e(\rho)$ are flux-surface-averaged quantities. | **Spatial coordinate** for model; diffusion operator acts on $\rho$. Fallback: linear-in-$R$ when EFIT unavailable. |
| `Vprime` | EFIT equilibrium | **Flux surface volume derivative** $V'(\rho) = dV/d\rho$ ($m^3$): rate of change of enclosed volume with $\rho$. In toroidal geometry, $V' \propto \rho$ near axis. | Appears in conservative form of transport equation; accounts for toroidal geometry effects. | **Metric coefficient** in FVM diffusion operator $\nabla \cdot (V' \chi \nabla T_e) / V'$. Fallback: $V'(\rho) = 2\rho$ (toroidal approx.). |

### Optional Scalars (Global Performance Metrics)

The following zero-dimensional scalars characterize overall plasma performance. They are ingested when available but **not used by default** in the current training pipeline:

- **`W_tot`** (MJ): Total stored plasma energy (thermal + kinetic). Measures confinement quality; $W_{\text{tot}} = \int (3/2)(n_e T_e + n_i T_i) dV$.
- **`P_ohm`** (MW): Ohmic heating power from plasma current resistivity. Dominant heating at low $T_e$; decreases as $T_e^{-3/2}$.
- **`P_tot`** (MW): Total input power ($P_{\text{ohm}} + P_{\text{nbi}} + P_{\text{RF}}$ etc.). Used in energy balance analysis.
- **`H98`**: H-mode confinement enhancement factor relative to IPB98(y,2) scaling law. $H98 > 1$ indicates better-than-empirical confinement.
- **`beta_n`** (%): Normalized plasma pressure $\beta_N = \beta / (I_p / (aB_t))$ where $\beta = 2\mu_0 \langle p \rangle / B_t^2$. Measures MHD stability margin.
- **`B_t0`** (T): Toroidal magnetic field at major radius $R_0$. Sets Larmor radius, trapping fraction; higher $B_t$ improves confinement.
- **`q95`**: Safety factor at 95% flux surface; $q = (r B_t)/(R_0 B_p)$ measures field line twist. Low $q_{95}$ risks MHD instabilities.
- **`li`**: Internal inductance (normalized); characterizes current profile peakedness. Affects resistive diffusion and stability.
- **`P_rad`** (MW): Total radiated power (bremsstrahlung, line radiation, impurities). Reduces net heating; $P_{\text{rad}}/P_{\text{tot}}$ is a power exhaust metric.

These scalars provide context for shot conditions but are not directly modeled; the reduced-order transport model focuses on radial profiles ($T_e$, $n_e$) and actuator inputs.

---

## 12. HPC Optimisation Quick Notes

- Use warmup + cosine decay, gradient clipping, and multi-seed restarts.
- Keep physics weights modest early; tune `lambda_w` and `lambda_src` to balance data fit vs. regularity.
- Monitor loss components and $z(t)$ to detect solver or identifiability issues.

---

## 13. HPC Optimisation Strategies (Expanded)

- **Optimisers:** Adam/AdamW as defaults; LAMB for very large batches; Lion when memory is tight.
- **Schedules:** warmup $\rightarrow$ cosine decay or exponential; keep physics weights modest early, raise $\lambda_w$ and $\lambda_{\text{src}}$ gradually.
- **Gradient control:** clip grads and keep derivative clipping at $10^4$ on $d\mathbf{T}/dt$ for solver stability.
- **Restarts:** run multi-seed jobs and keep best checkpoints; divergence often resolves by restarting with a lower LR or slightly smaller physics weights.
- **Diagnostics:** track $|\boldsymbol{\rho}_{\text{rom}}|$, $|\boldsymbol{\rho}_{\cap}|$, loss components, $z(t)$ traces, and one-shot overlays $T_{\text{model}}(t,\boldsymbol{\rho}_{\cap})$ vs. data.
- **Architecture levers:** widen the source MLP or adjust pedestal sharpness only after stabilising training; keep the source output layer zero-initialised to avoid large initial residuals.

---

## 14. Training Mechanics (Batches and Restarts)

We minimise a composite objective over shots and time. Let $\theta$ denote all learnable parameters (NN weights and latent dynamics), and let $\mathcal{L}_s(\theta)$ be the per-shot loss (data term on $\boldsymbol{\rho}_{\cap}$, source penalty, optional regime supervision, $z$ regularisation) aggregated over the shot’s time samples. The empirical objective is
$$
J(\theta) = \frac{1}{N}\sum_{s=1}^N \mathcal{L}_s(\theta),
$$
where $N$ is the number of training shots.

### Mini-batches (why we use them)
- We estimate the gradient by sampling a mini-batch $\mathcal{B}_t$ of size $B$ at step $t$:
  $$
  g_t = \frac{1}{B} \sum_{s\in\mathcal{B}_t} \nabla_\theta \mathcal{L}_s(\theta_t).
  $$
- Under uniform sampling, $\mathbb{E}[g_t] = \nabla_\theta J(\theta_t)$ (unbiased), with variance that decreases roughly like $\mathrm{Var}(g_t) \propto 1/B$.
- In our multi-GPU setup, the global batch of size $B$ is sharded across $D$ devices (each processes $B/D$ shots) and then averaged:
  $$
  g_t = \frac{1}{D}\sum_{d=1}^D g_t^{(d)}, \quad g_t^{(d)} = \frac{1}{B/D} \sum_{s\in\mathcal{B}_t^{(d)}} \nabla_\theta \mathcal{L}_s(\theta_t).
  $$
- Practical effects: larger batches reduce gradient noise (smoother updates) at the cost of more memory and potentially less exploration; small batches increase noise, which can help escape poor minima but may slow convergence.

### Restarts (why we use them)
- The objective $J(\theta)$ is nonconvex. Different initialisations $\theta_0^{(r)}$ (random seeds) can converge to different local minima.
- We run multiple restarts $r=1,\dots,R$:
  $$
  	\theta^{(r)}_\ast = \operatorname*{argmin}_{\theta} J^{(r)}(\theta),\quad r=1,\dots,R,
  $$
  and keep the best checkpoint $\theta_\ast = \operatorname*{argmin}_r J\big(\theta^{(r)}_\ast\big)$.
- Mathematically, this is multi-start optimisation; it increases the probability of reaching a lower objective value in nonconvex landscapes.

### Learning-rate schedule (warmup + cosine)
- We use a learning-rate schedule $\alpha_t$ with a warmup phase followed by cosine decay:
  $$
  \alpha_t = \begin{cases}
  \alpha_{\text{init}} + (\alpha_{\text{peak}}-\alpha_{\text{init}})\,\frac{t}{T_{\text{warm}}}, & t \le T_{\text{warm}} \\
  \alpha_{\text{end}} + \tfrac{1}{2}(\alpha_{\text{peak}}-\alpha_{\text{end}})\big(1+\cos(\pi\, \tfrac{t-T_{\text{warm}}}{T_{\text{decay}}})\big), & t > T_{\text{warm}}.
  \end{cases}
  $$
- Warmup prevents unstable large updates at the beginning when gradients/normalisations are settling; cosine decay gradually reduces step sizes to refine the solution.

## 15. ODE Solver Choice and Rationale

We integrate a stiff semi-discrete transport ODE:
$$
\frac{d\mathbf{y}}{dt} = f(t, \mathbf{y}; \theta), \quad \mathbf{y} = [\mathbf{T}_{\text{int}}, z].
$$
Stiffness arises from diffusion on a non-uniform grid: the discrete operator has large negative eigenvalues with magnitude scaling like $\|\lambda\| \sim \chi / \Delta\rho^2$, creating fast transients alongside slower profile evolution.

### 15.1 Sources of Stiffness

**Non-uniform grid spacing is the primary driver.** For diffusion, the fastest decay time is set by the smallest spatial scale:
$$
|\lambda_{\max}| \sim \frac{\chi_{\max}}{(\Delta\rho_{\min})^2}, \qquad \tau_{\min} \sim \frac{(\Delta\rho_{\min})^2}{\chi_{\max}}.
$$
On our non-uniform ROM grid $\boldsymbol{\rho}_{\text{rom}}$:
- Chebyshev interior nodes cluster near $\rho=0$ → tiny $\Delta\rho$ in the core,
- The intersection grid is often edge-biased with uneven gaps,
- The spacing ratio $\Delta\rho_{\max}/\Delta\rho_{\min}$ can reach 10–100×, creating a stiffness ratio of $10^2$–$10^4$.

**Example:** If $\Delta\rho_{\min} = 0.005$ and $\Delta\rho_{\max} = 0.05$, then diffusion modes in the clustered region decay $\sim 100\times$ faster than in the coarse region, forcing explicit methods to take prohibitively small timesteps.

**Additional stiffness sources:**
- Sigmoid pedestal in $\chi(\rho,z)$ creates sharp gradients,
- Strong edge diffusivity drop (L→H transition) amplifies local stiffness,
- Small $V'$ near the axis (before safe-core clamping) can cause large effective diffusion coefficients.

**Note:** $V'(\rho)$ affects metric scaling but is not the main stiffness driver; the $(\Delta\rho)^{-2}$ factor dominates.

### 15.2 Current Solver (Kvaerno5 + Adaptive Stepping)

- **Solver:** We use `Kvaerno5`, a stiff, high-order implicit Runge–Kutta method with adaptive step size via PID control (Diffrax [3]).
- **Adaptivity:** The controller adjusts $\Delta t$ to keep local truncation error near a target tolerance:
  $$
  \Delta t_{k+1} = \Delta t_k\,\left(\frac{\mathrm{tol}}{\mathrm{err}_k}\right)^{1/(p+1)}\cdot \text{(PI corrections)},
  $$
  where $p$ is the method order and $\mathrm{err}_k$ is the estimated error at step $k$.
- **Why not explicit?** Explicit methods require $\Delta t \lesssim 2/|\lambda_{\max}|$ for stability. With $|\lambda_{\max}| \sim 10^3$–$10^5$ (typical for our grid), explicit methods would need $\Delta t \sim 10^{-5}$–$10^{-6}$ seconds, making training impractically slow. Stiff solvers remain stable at much larger $\Delta t$ by treating fast modes implicitly.
- **Evaluation grid:** We save solutions at Thomson time stamps via `SaveAt(ts=...)`. The final integration time is the last valid profile (dynamic end time); padded time points are masked in the loss.
- **Safety limits:** `rtol=atol=1e-3`, initial `dt0=1e-4`, max steps `MAX_SOLVER_STEPS=50000`; derivatives are softly clipped at $10^4$ for additional stability.

**Trade-offs of the current approach:**
- ✓ Simple to implement; fully compatible with JAX autodiff,
- ✓ No need to manually construct Jacobians or linear systems,
- ✗ Solver adaptivity can cause wall-clock variance across shots/batches,
- ✗ Backpropagation through adaptive steps can be noisy and memory-heavy for long horizons,
- ✗ Still pays a "stiffness tax": solver must internally resolve fast modes even if we don't care about them.

### 15.3 Alternative: TORAX-Style Implicit Time-Stepping (IMEX)

**How TORAX handles stiffness [4].** TORAX is a differentiable tokamak transport simulator in JAX that solves coupled 1D PDEs (electron/ion heat, particles, current) with:
1. **Finite-volume spatial discretization** (similar to ours),
2. **Implicit time-stepping** via the $\theta$-method: $\theta=0$ (explicit Euler), $\theta=0.5$ (Crank–Nicolson), $\theta=1$ (implicit Euler, unconditionally stable),
3. **Nonlinear solvers** per timestep: linear predictor–corrector, Newton–Raphson, or optimizer-based (jaxopt),
4. **Adaptive $\Delta t$** based on $\chi_{\max}$ and stability heuristics, with backtracking if nonlinear solve fails,
5. **Differentiability** via implicit differentiation (custom VJP) to avoid backpropagating through Newton iterations.

**Key difference:** TORAX treats diffusion as an implicit linear system per timestep, not as an ODE fed to a black-box solver. This transforms stiffness from a "tiny timestep" problem to a "solve a banded linear system" problem.

**IMEX (Implicit-Explicit) for our ROM:**
Instead of integrating $\dot{\mathbf{T}} = \mathbf{D}(\mathbf{T}) + \mathbf{S}_{\text{net}}$ with Kvaerno5, we could use:
$$
\frac{\mathbf{T}^{n+1} - \mathbf{T}^n}{\Delta t} = \underbrace{\mathbf{D}(\mathbf{T}^{n+1})}_{\text{implicit (stiff diffusion)}} + \underbrace{\mathbf{S}_{\text{net}}(\mathbf{T}^n)}_{\text{explicit (mild source)}},
$$
or Crank–Nicolson (second-order accurate):
$$
\frac{\mathbf{T}^{n+1} - \mathbf{T}^n}{\Delta t} = \frac{1}{2}\big[\mathbf{D}(\mathbf{T}^{n+1}) + \mathbf{D}(\mathbf{T}^n)\big] + \mathbf{S}_{\text{net}}(\mathbf{T}^n).
$$
This rearranges to a linear system (if $\mathbf{D}$ is linear in $\mathbf{T}$ at fixed $\chi(z)$):
$$
\big(\mathbf{I} - \tfrac{\Delta t}{2}\mathbf{L}(\chi, V')\big)\mathbf{T}^{n+1} = \mathbf{T}^n + \tfrac{\Delta t}{2}\mathbf{D}(\mathbf{T}^n) + \Delta t\,\mathbf{S}_{\text{net}}(\mathbf{T}^n) + \Delta t\,\mathbf{b}_{\text{BC}},
$$
where $\mathbf{L}$ is the diffusion stencil operator (tridiagonal/banded for 1D FVM). Solving this once per timestep is $O(N_{\text{rom}})$ and stable for large $\Delta t$.

**Advantages of IMEX for our ROM:**
- **Deterministic compute:** Fixed number of linear solves per batch (no adaptive step variance),
- **Better gradients:** Backprop via implicit differentiation (solve adjoint linear system) is cleaner than unrolling adaptive ODE steps,
- **Faster training:** Fewer effective timesteps; can step directly on Thomson grid,
- **Boundary conditions:** Core Neumann BC ($F_{-1/2}=0$) and edge Dirichlet BC ($T_{\text{edge}}(t)$) are handled naturally in the linear system structure,
- **Preserves physics separation:** Diffusion is still "known structure," $S_{\text{net}}$ is still "residual learning."

**Challenges:**
- More complex to implement than "plug into Diffrax,"
- Requires constructing/solving linear systems (though JAX has good sparse solvers and tridiagonal solvers),
- If $\chi(z)$ makes $\mathbf{D}$ highly nonlinear, may need Newton iterations (but this is standard in TORAX),
- Need custom VJP for efficient backpropagation (implicit differentiation).

### 15.4 Comparison: ROM+Diffrax vs. ROM+IMEX vs. TORAX Native

| Aspect | **ROM + Diffrax (current)** | **ROM + IMEX (upgrade)** | **TORAX Native** |
|--------|----------------------------|-------------------------|------------------|
| **Integration method** | Stiff RK (Kvaerno5) adaptive | Implicit $\theta$-method, fixed/adaptive $\Delta t$ | Implicit $\theta$-method + Newton/optimizer |
| **Stiffness handling** | Automatic via stiff RK | Linear solve per step | Linear solve + nonlinear iterations |
| **Timestep control** | PID adaptive (can vary wildly) | User-controlled or adaptive (more predictable) | Adaptive with backtracking |
| **Compute per batch** | Variable (solver-dependent) | Fixed (# steps × linear solve cost) | Fixed (# steps × nonlinear solve cost) |
| **Gradient quality** | Noisy (adaptive path changes) | Cleaner (implicit diff) | Cleanest (built-in adjoint) |
| **Complexity** | Minimal (call `diffeqsolve`) | Moderate (construct system, solve, VJP) | High (full simulator stack) |
| **Physics scope** | 1 channel ($T_e$) + latent $z$ | 1 channel ($T_e$) + latent $z$ | Multi-channel (Te, Ti, ne, current) |
| **Data fit objective** | Intersection grid + robust loss (ours) | Intersection grid + robust loss (ours) | Requires data assimilation layer |
| **Interpretability** | High (explicit $\chi$, $S_{\text{net}}$) | High (same as left) | High (modular closures, but less residual freedom) |
| **Best for** | Quick prototyping; simple ROMs | Production training; control-oriented ROMs | High-fidelity forward simulation; multi-physics coupling |

**Recommendation:**
- **Current approach** (Diffrax) is fine for initial development and small-scale experiments.
- **IMEX upgrade** is the sweet spot for production training: retains our ROM design, handles stiffness properly, improves gradient stability, and keeps training time predictable.
- **TORAX native** is best if you need multi-channel physics or want to leverage TORAX's existing closures/sources, but requires heavier refactoring and a different data assimilation strategy.

### 15.5 Future Work: Grid and Solver Co-Design

To reduce stiffness without sacrificing spatial resolution:
1. **Bounded spacing ratio:** Instead of aggressive Chebyshev clustering, enforce $\Delta\rho_{\max}/\Delta\rho_{\min} \le r_{\max}$ (e.g., $r_{\max}=5$). This reduces stiffness ratio from $\sim 100$ to $\sim 25$ while keeping core resolution.
2. **Adaptive interior placement:** Cluster nodes near the pedestal (where $\partial_\rho T_e$ is largest) rather than uniformly in $(0,\rho_{\cap,\min})$.
3. **Coarse uniform interior + observed edge:** Use a small number of uniform interior nodes up to $\rho_{\cap,\min}$, then keep the observed (non-uniform) edge points. This controls state dimension $N_{\text{rom}}$ and reduces stiffness without hurting identifiability.
4. **IMEX with time-varying geometry:** If equilibrium time series are smooth and reliable, implement $V'(t)$ as a linear interpolant and re-evaluate inside IMEX steps (see §3.2.6).

---

## 16. References

[1] T. H. Osborne, K. H. Burrell, and R. J. Groebner. H-mode pedestal characteristics in DIII-D. *Plasma Physics and Controlled Fusion*, 40(5):845, 1998.

[2] J. Bradbury et al. JAX: composable transformations of Python+NumPy programs. 2018. URL: http://github.com/google/jax.

[3] P. Kidger. On Neural Differential Equations. PhD thesis, University of Oxford, 2021.

[4] F. Felici, A. Merle, et al. TORAX: a differentiable tokamak transport simulator in JAX. *arXiv preprint arXiv:2409.10622*, 2024. URL: https://github.com/google-deepmind/torax.

[5] B. Coppi. Non-classical transport and the "principle of profile consistency". *Comments on Plasma Physics and Controlled Fusion*, 5(6):261–270, 1980.

[6] H. Haken. *Synergetics: An Introduction*. Springer-Verlag, Berlin, 3rd edition, 1983.

[7] H. Mori. Transport, collective motion, and brownian motion. *Progress of Theoretical Physics*, 33(3):423–455, 1965.

[8] R. Zwanzig. Memory effects in irreversible thermodynamics. *Physical Review*, 124(4):983, 1961.

[9] L. D. Landau. On the theory of phase transitions. *Zh. Eksp. Teor. Fiz.*, 7:19–32, 1937.

[10] J. Carr. *Applications of Centre Manifold Theory*. Applied Mathematical Sciences. Springer-Verlag, New York, 1981.

[11] A. Vaswani et al. Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 2017.

[12] D. Ha and J. Schmidhuber. World models. *arXiv preprint arXiv:1803.10122*, 2018.

[13] R. E. Kalman. A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1):35–45, 1960.

[14] A. Gu and T. Dao. Mamba: Linear-time sequence modeling with selective state spaces. *arXiv preprint arXiv:2312.00752*, 2023.

[15] E. J. Doedel et al. AUTO-07P: Continuation and bifurcation software for ordinary differential equations. 2007.

[16] M. Scheffer et al. Early-warning signals for critical transitions. *Nature*, 461(7260):53–59, 2009.

[17] K. Hornik et al. Multilayer feedforward networks are universal approximators. *Neural Networks*, 2(5):359–366, 1989.
