TORAX Data Pipeline Add-ons

This repo already downloads MAST Level-2 data to NetCDF via `download_data.py`.
The scripts below turn those NetCDFs into TORAX-ready arrays and provide a
starter latent-variable simulator.

Files
- `scripts/geometry.py`: helpers to compute flux coordinate `rho` and geometry scalars (`Vprime`).
- `scripts/build_training_pack.py`: build `data/<shot>_torax_training.npz` including `Vprime`, `regime`, raw power signals, particle sources (`S_gas`, `S_rec`, `S_nbi`), and extended summary signals (`W_tot`, `P_ohm`, `H98`, `q95`, etc.).
- `scripts/train_latent_model.py`: simulate latent scalar `z(t)` using trained parameters from `params.json`.
- `scripts/train_4dvar.py`: Legacy JAX-based 4D-Var fitter with manual explicit Euler stepping.
- `scripts/train_ode.py`: **New** Differentiable Physics trainer using `Diffrax` (Method of Lines) for robust stiff integration and adjoint backpropagation.

Quick Start
1) Ensure NetCDFs exist (produced by `download_data.py`):
   - `data/<shot>/equilibrium.nc`
   - `data/<shot>/thomson_scattering.nc`
   - `data/<shot>/summary.nc`

2) Build training packs
```bash
# One shot
python -m scripts.build_training_pack --shot 30421

# Multiple shots
python -m scripts.build_training_pack --shots 30420 30421 30422

# Discover shots under data/<shot>
python -m scripts.build_training_pack --discover
```

This writes `data/<shot>_torax_training.npz`.

Downloader CLI (Updated)
------------------------

`download_data.py` now exposes a few knobs so you can refresh or skip datasets without editing the file:

- `--shots 30420 30421 ...` select one or more discharges.
- `--groups summary thomson_scattering equilibrium gas_injection spectrometer_visible` limit products (default is all five).
- `--out-root data_new` redirect output base directory.
- `--cache-dir .cache` pick the local simplecache location.
- `--max-retries 5 --retry-delay 2.0` add resilience to transient network hiccups.
- `--overwrite` force re-download even if NetCDFs already exist (otherwise they are skipped).
- `--no-plots` disable the quick-look summary PNGs when running headless.

Example full refresh (overwrites existing NetCDFs and regenerates plots):

```bash
./torax_env/bin/python download_data.py \
  --shots 30420 30421 30422 \
  --groups summary thomson_scattering equilibrium gas_injection spectrometer_visible \
  --out-root data \
  --cache-dir .cache \
  --max-retries 5 \
  --overwrite
```

End-to-End Pipeline Run
-----------------------

1. **Download / refresh NetCDFs** (command above).
2. **Build packs** (after every data refresh):

   ```bash
   ./torax_env/bin/python -m scripts.build_training_pack --shots 30420 30421 30422
   ```

3. **Single-shot sanity fit** (`eq_only` freezes transport/latent pieces so the run finishes in seconds):

   ```bash
   ./torax_env/bin/python -m scripts.train_4dvar data/30421_torax_training.npz \
     --steps 12 --lr 5e-3 --use-ne --mode eq_only
   tail -n 5 outputs/30421_torax_training_4dvar/loss.csv
   ```

4. **Optional joint smoke test** (confirms multi-shot path still works):

   ```bash
   ./torax_env/bin/python -m scripts.train_4dvar \
     data/30420_torax_training.npz data/30421_torax_training.npz \
     --steps 5 --lr 1e-3 --use-ne --mode full
   cat outputs/joint_4dvar/loss.csv
   ```

5. **Simulation / Inference** (using trained normalization stats):

   ```bash
   ./torax_env/bin/python -m scripts.simulate_shot \
     data/30421_torax_training.npz \
     --norm-stats outputs/30421_torax_training_4dvar/normalization.json \
     --use-ne
   ```

Smoke-Test Script
-----------------

`scripts/smoke_test.sh` automates the short eq_only run and fails if any NaNs show up in `loss.csv`. Use it in CI or after dependency bumps:

```bash
SHOT=30421 bash scripts/smoke_test.sh
```

It rebuilds `loss.csv`, inspects the tail for `NaN`, and surfaces a non-zero exit status if the check fails.

Cost Function Guardrails
------------------------

- **Masked residuals.** `train_4dvar` now consumes the `Te_mask`/`ne_mask` saved in each pack, so only genuinely observed points contribute to the data term.
- **Automatic scaling.** Per-pack standard deviations of `Te` and `ne` are measured on load and folded into the effective noise level (`sigma_eff = sigma_flag * std_data`). This prevents the `ne` residual from overflowing when raw data are $\\mathcal{O}(10^{19})$.
- **Global diagnostics normalization.** The same `ne` scale is applied to the line-averaged density loss so its magnitude stays comparable to the profile terms.
- **Non-finite detection.** The trainer now aborts immediately if any loss or gradient becomes non-finite, instead of silently zeroing via `nan_to_num`. When that happens you'll see which parameter produced NaNs, making it easier to diagnose bad data or learning-rate issues.

Detailed Pipeline Specification
-------------------------------

Data Harmonization
------------------

- Raw NetCDF: equilibrium (magnetic), thomson_scattering (profiles), summary (scalar diagnostics), gas_injection (fueling), spectrometer_visible (recycling).
- **Extended Signals**:
  - **Global**: `W_tot` (stored energy), `P_ohm`, `P_tot`, `ne_line` (interferometer), `H98` (confinement factor).
  - **Magnetics**: `q95`, `li`, `beta_n`, `B_t0`.
  - **Sources**: `S_gas` (total puff), `S_rec` (D-alpha proxy), `S_nbi` (beam fueling).
- Spatial Coordinate: normalized flux radius $\rho \in [0,1]$ from TS coordinate or mapped via $\psi(R,Z)$ axis/edge normalization.
- **Flux-coordinate construction:**  The axisymmetric equilibrium solver produces $\psi(R,Z)$ (the poloidal flux) defined on $\mathbb{R}^3$ via cylindrical coordinates $(R,\phi,Z)$ with symmetry in $\phi$.  We normalize each flux surface by
  $$
  \rho(\mathbf{x}) = \sqrt{\frac{\psi(R,Z) - \psi_{\text{axis}}}{\psi_{\text{sep}} - \psi_{\text{axis}}}},
  $$
  so $\rho=0$ on the magnetic axis and $\rho=1$ at the separatrix.  This map is computed by `scripts/geometry.py`, which interpolates the 2D $\psi(R,Z)$ grid onto the 3D (axisymmetric) torus and flushes the result into the pack as the `rho` array of 65 nodes.  Each node therefore represents the radial coordinate of a nested surface in the $R$–$Z$ plane, lifted uniformly along the toroidal angle $\phi$.  The same geometry module also records the Jacobian-derived quantities described below.
- **Radial discretization (65 nodes)**: Each pack stores the TS profile on a 65-point radial mesh (axis through pedestal) so we can solve the finite-volume diffusion on the exact observational nodes. Keeping 65 points trades off pedestal resolution against solver stiffness and lets the hybrid trainer reuse the same mesh when it assembles the equilibrium manifold $E(z)$ and the residual $\delta T_e$.
- Time Grids: summary time $t_{sum}$ and profile time $t_{prof}$; inputs resampled to $t_{prof}$ (nearest-neighbor; upgradeable).


State & Decomposition
---------------------

- $T_e(\rho,t) = E_{T_e}(z,u;\psi) + \delta T_e(\rho,t)$, where $E_{T_e}$ is the latent-controlled equilibrium manifold and $\delta T_e$ captures transient perturbations.
- $n_e(\rho,t) = E_{n_e}(z,u;\psi)$ (no perturbation yet).
- Latent $z(t)$ drives regime-dependent blending and transport coefficients.
  - **Equilibrium Manifolds:** We learn each equilibrium manifold $E_{T_e}$, $E_{n_e}$ jointly with the downstream transport via a hat basis $\Phi(\rho)$ and explicit shape/scale coefficients. The latent state $z$ blends core/edge structures while the control inputs modulate the scale (e.g., via $\tilde P_{NBI}$); training optimizes the coefficients $b_0$, $b_1$, and $b_u$ that multiply the shared basis vectors so the manifold adapts to both regime and inputs.
  - **Hybrid decomposition:** `scripts/train_ode_hybrid.py` makes the decomposition explicit by reconstructing $E_{T_e}(z)$ from the 65-point hat basis and tracking the residual $\delta T_e$ via the PDE + SourceNN. Keeping the manifold separate from the perturbation makes it easier to analyze stability (e.g., spectral properties of $\delta T_e$) while ensuring the latent $z$ controls the edge barrier that triggers the L-H bifurcation.


Equilibrium Manifolds
---------------------

- Hat basis $\Phi(\rho) \in \mathbb{R}^{N_\rho \times K}$ with sigmoid blend:  
  $E_{T_e} = \Phi( b_0^{Te} + \sigma(k_\sigma z) b_1^{Te} + b_u^{Te} \tilde P_{NBI} )$.  
  $E_{n_e} = \Phi( b_0^{n_e} + \sigma(k_\sigma z) b_1^{n_e} + b_u^{n_e} \tilde P_{NBI} )$.  
- Input normalization: $\tilde P_{NBI} = (P_{NBI}-\bar P)/(\mathrm{std}(P)+\varepsilon)$.
- **Shape-Scale Option**: $E = A(z,u)\phi(\rho) + B(z,u)$ where $A, B$ are learned linear maps of features.


Transport Model (Flux Divergence)
---------------------------------

The pipeline implements a 1D radial transport solver for both electron temperature ($T_e$) and electron density ($n_e$). The solver uses a finite-volume discretization on a staggered grid to ensure conservation.

### Electron Temperature ($T_e$)

The heat transport equation is modeled as:
$$ \frac{\partial T_e}{\partial t} = \frac{1}{V'} \frac{\partial}{\partial \rho} \left( V' \chi_e \frac{\partial T_e}{\partial \rho} \right) + S_{heat} $$

- **Diffusivity ($\chi_e$)**: Modeled as a profile that transitions between core and edge levels based on the latent state $z$:
  $$ \chi(\rho, z) = \chi_{core} + w_{ped}(\rho) \cdot (\chi_{edge}(z) - \chi_{core}) $$
- **Regime Dependence**: $\chi_{edge}(z)$ captures the L-mode to H-mode transition.

### Electron Density ($n_e$)

The particle transport equation is:
$$ \frac{\partial n_e}{\partial t} = \frac{1}{V'} \frac{\partial}{\partial \rho} \left( V' D \frac{\partial n_e}{\partial \rho} \right) + S_{density} $$

- **Diffusivity ($D$)**: Currently modeled with a simple parabolic profile $D(\rho) = D_{core} + \rho^2 (D_{edge} - D_{core})$.
- **Boundary Conditions**:
  - Inner ($\rho=0$): Zero flux ($\partial n_e / \partial \rho = 0$).
  - Outer ($\rho=1$): Dirichlet condition $n_e(1) = n_{edge}$ (acting as a sink).

Source Term Normalization & Geometry
------------------------------------

A critical aspect of the pipeline is the correct handling of physical units and tokamak geometry.

### Source Term Normalization

Raw particle source data (gas puff, NBI, recycling) is typically provided in units of **Total Particles per Second** ($S_{tot} \approx 10^{22} s^{-1}$). To use this in the local transport equation, it must be converted to a volumetric density rate ($m^{-3} s^{-1}$).

We define a normalized shape function $f(\rho)$ (e.g., Gaussian at the edge for gas, core for NBI) such that its volume integral is unity:
$$ \int_0^1 f(\rho) V'(\rho) d\rho = 1 $$

The local source term $S_{density}(\rho)$ is then calculated as:
$$ S_{density}(\rho) = \frac{S_{tot} \cdot f(\rho)}{V'(\rho)} $$

This ensures that $\int S_{density} dV = S_{tot}$, preserving global particle balance.

### Geometry ($V'$)

The differential volume element $V'(\rho) \equiv \frac{dV}{d\rho}$ encodes the tokamak geometry.

- **Data-Driven**: When available, $V'$ is loaded directly from the `equilibrium.nc` file.  The diagnostic supplies the toroidal volume differential via
  $$
  V'(\rho) = \frac{dV}{d\rho} = \oint_{\rho} \frac{dS}{|\nabla \rho|}
  $$
  where the line integral runs over the flux surface at normalized radius $\rho$ (and $dS$ is the surface element on that 2D surface).  In practice the equilibrium reconstruction computes $V'$ from the mesh of $(R,\psi)$ points, so the pack simply inherits the measured array.  This quantity is what feeds the finite-volume divergence denominator and the geometric prefactor $\langle|\nabla\rho|^2\rangle=V'/A(\rho)$ discussed in the log.  When the NetCDF lacks $V'$, we fall back to the large-aspect-ratio analytic estimate listed below.
- **Analytical Fallback**: If $V'$ is missing or invalid (e.g., all 1s), the pipeline approximates it using a large-aspect-ratio circular cross-section model:
  $$ V'(\rho) \approx 4\pi^2 R_0 a^2 \rho $$
  For MAST-like parameters ($R_0 \approx 0.85m, a \approx 0.65m$), this yields $V' \approx 14 \rho \, [m^3]$.

Latent Dynamics
--------------

- $\dot z = \alpha(\mu(u)-\mu_c) - \beta z + \gamma z^3$, with $\mu(u)=\tilde P_{NBI}$ (extendable to linear combo of inputs).
- **Multi-Input Drive**: $\mu(u) = w_\mu^T [P_{NBI}, I_p, \bar n_e] + b_\mu$.


Observation Operator
--------------------

- Identity on available profiles: $H(state)=\{T_e, n_e\}$.  
- Measurement: $y = H + \eta$, independent Gaussian noise variances $(\sigma_{T_e}^2, \sigma_{n_e}^2)$; optionally learned via log-sigmas.
- Hybrid residual observations: `scripts/train_ode_hybrid.py` treats $H=\mathbb{I}$ on the same 65-node grid, applies the pack's `Te_mask` together with a threshold ($T_e>1\,$eV) to select valid entries from `simulate_outputs.npz`, and compares the reconstructed manifold plus perturbation to the masked measurements so that the residual NN only sees the trustworthy data points.

Hybrid Residual Training

- `scripts/train_ode_hybrid.py` integrates $(\delta T_e, z)$ with Diffrax on the 65-point radial mesh, keeps the diffusion operator fixed, and gates the edge diffusivity through a sigmoid of $z$ so the latent variable captures the L-H barrier height.
- The equilibrium manifold $E_{T_e}(z)$ is reconstructed from a small hat basis on the same mesh, and the SourceNN takes $(\rho, T_e, n_e, \text{controls}, z)$ so the residual absorbs whatever physics the manifold does not explain. The loss is masked by the `Te_mask` plus $T_e>1$ eV and uses the clean `simulate_outputs.npz` targets, which keeps the manifold smooth while leaving $\delta T_e$ for stability diagnostics.

Cost Function
------------

$$\mathcal{J} = J_{data} + J_{reg} + J_{pwr} + J_{aux}$$

- **Data**: Masked MSE on $T_e$ and $n_e$ profiles.
- **Power Balance ($J_{pwr}$)**: Improved penalty enforcing global energy conservation:
  $$ \frac{dW_{model}}{dt} \approx P_{tot} - P_{rad} $$
  where $W_{model} = 1.5 \int n_e T_e dV$.
- **Auxiliary ($J_{aux}$):** Correlation loss forcing the latent state $z(t)$ to track confinement metrics (e.g., $z \sim H_{98}$) or regime labels.
- **Regularization**: L2 penalties on parameters.


Training Procedure
------------------

- Joint gradient accumulation across shots (batch multi-shot).  
- Explicit Euler for diffusion + latent (upgrade to semi-implicit planned).  
- Optimizer: SGD; Optax variants supported.  
- Noise params frozen unless `--learn-noise`.  
- NaN guards + clipping for stability.


Parameter Sharing
-----------------

- Shared: all equilibrium, transport, latent parameters.  
- Per-shot: $z_0^{(s)}$ (initial latent). Future: calibration offsets.


Extensibility Roadmap
---------------------

- **Implemented**: Flux divergence transport scheme (finite volume).
- **Implemented**: $n_e$ transport with particle sources ($S_{gas}, S_{nbi}, S_{rec}$).
- **TODO**: Add more channels (e.g., $T_i$, rotation).
- **TODO**: Sparse/noisy observation matrices & correlated $R$.
- **TODO**: Regime weighting via $|\dot z|$ or input variance.
- **TODO**: Probabilistic DA / uncertainty quantification.


Numerical Stability Notes
------------------------

- **Precision**: The trainer now enforces `float64` precision (`jax_enable_x64`) to handle the large dynamic range of particle source terms ($10^{22}$) and density gradients.
- **Normalized Transport**: The density transport equation is solved in normalized units ($\hat{n} = n / n_{scale}$) to keep gradients well-behaved ($O(1)$ to $O(100)$).
- **Gradient Clipping**: `stop_gradient` is applied to the diffusion term in the backward pass (TBPTT approximation) to prevent gradient explosion through time steps, while allowing full training of source coefficients (`k_gas`, `k_nbi`).
- **Unscaled Laplacian**: Reduces stiffness; still enforce CFL-like condition.  
- **Clipping**: Avoids blow-up during early parameter search.  
- **Learned Noise**: Kept positive via exponential mapping.
- **Grid compatibility**: The hybrid training grid stays at 65 nodes so the finite-volume discretization, the hat-basis equilibrium manifold, and the mask-based observations all share the same sampling; this keeps the stiff solver on a moderate-size state vector while still resolving the pedestal features needed for the L-H bifurcation.
- **Hybrid training guardrails**: `scripts/train_ode_hybrid.py` now resamples `simulate_outputs.npz`, masks to finite, positive $T_e$, and uses a clipped Tsit5 integrator so the residual-source NN only receives bounded derivatives while holding the diffusion operator fixed.


Validation Metrics
------------------

- Profile RMSE time curves (normalized).  
- Latent correlation with performance indicators.  
- Per-shot loss breakdown post joint training.


Minimal Invocation Examples
---------------------------

```bash
# Te only
python -m scripts.train_4dvar data/30421_torax_training.npz --steps 400 --lr 5e-3
# Te + ne with learned noise
python -m scripts.train_4dvar data/30421_torax_training.npz --use-ne --learn-noise --steps 400 --lr 3e-3
# Multi-shot joint (Te only)
python -m scripts.train_4dvar data/30420_torax_training.npz data/30421_torax_training.npz data/30422_torax_training.npz --steps 800 --lr 5e-3 --subsample 2
```

New Trainer Options (Extended 4D-Var)
--------------------------------------

- `--mode {full,eq_only,latent_only}`: Freeze subsets of parameters for staged curriculum.
- `--lambda-z0`: Regularization weight for latent initial states prior (push toward 0).
- `--lambda-global`: Weight for global diagnostics loss (currently line-average density approximation).
- `--shape-scale`: Use shape-scale equilibrium parameterization `E = A(z,u)*phi(rho)+B(z,u)`.
- `--optax`: Switch optimizer from manual SGD to Adam (Optax).
- Multi-input latent drive: parameters `w_mu`, `b_mu` map `[P_nbi, Ip, nebar]` to surrogate `mu`.
- Hysteresis: `beta=exp(raw_beta)`, `gamma=exp(raw_gamma)` enforce positivity for double-well potential.

Outputs Enhancements
--------------------

- Multi-shot writes per-shot diagnostics under `outputs/joint_4dvar/shots/`.
- Single-shot: `z.png`, `Te_fit.png`, optional `ne_fit.png`.

Simulation Interface
--------------------

- `scripts/simulate_shot.py`: Pure forward pass using trained parameters and normalization stats → `simulate_outputs.npz`.
- `scripts/train_latent_model.py`: Standalone simulation of latent dynamics $z(t)$ using `params.json`.

Normalization Utilities (Implemented)
-------------------------------------

- `scripts/normalization.py` supplies z-score stats (`compute_stats`, `save_stats`, `load_stats`).
- `train_4dvar.py` computes global stats from all inputs and saves them to `outputs/.../normalization.json`.
- `simulate_shot.py` loads these stats via `--norm-stats` to ensure consistent input scaling during inference.

Latent Multi-Input Extension
----------------------------

- Drive surrogate `mu(u)` now mixes normalized `[P_nbi, Ip, nebar]`. Hysteresis partially implemented; equilibrium residual remains TODO.

Physics Model & Simplifications
-------------------------------

The user may ask: *"If TORAX is a '4D' simulation, why are we solving 2D equations?"*

### 1. Dimensionality Reduction (1.5D Transport)

A full tokamak plasma evolves in 3D space $(R, Z, \phi)$ plus time $t$ (4D total). However, transport codes like TORAX, TRANSP, or RAPTOR rely on the fact that plasma flows extremely fast along magnetic field lines, equilibrating temperature and density on **flux surfaces**.

- **Reduction**: We average the equations over these flux surfaces.
- **Result**: The state depends only on the radial coordinate $\rho$ (flux label) and time $t$.
- **"1.5D"**: We solve 1D radial equations, but we use 2D geometry coefficients ($V'(\rho)$, $\langle |\nabla \rho|^2 \rangle$) derived from the magnetic equilibrium to account for the torus shape.
- **4D-Var**: The term "4D" in 4D-Var refers to optimizing a trajectory over **Space + Time**. In our case, the "Space" is 1D radial, so we optimize over the $(t, \rho)$ plane.

### 2. Current Implementation vs. Full Physics

Our current pipeline implements a **Reduced Transport Model**. Below is the comparison with a full-fidelity simulation.

#### Implemented Equations (Electron Sector)
We solve for Electron Temperature $T_e$ and Density $n_e$:
$$ \frac{\partial}{\partial t} \left( \frac{3}{2} n_e T_e \right) = \nabla \cdot (n_e \chi_e \nabla T_e) + P_{heat} $$
$$ \frac{\partial n_e}{\partial t} = \nabla \cdot (D \nabla n_e) + S_{particle} $$

#### Missing Equations (The "Full" Model)
A complete transport simulation would also include:

1.  **Ion Heat Transport ($T_i$)**:
    $$ \frac{\partial}{\partial t} \left( \frac{3}{2} n_i T_i \right) = \nabla \cdot (n_i \chi_i \nabla T_i) + P_{NBI,i} - P_{exch}(T_e, T_i) $$
    *Status*: Not implemented. Requires $T_i$ data (CXRS) and equipartition terms.

2.  **Current Diffusion (Poloidal Flux $\psi$)**:
    Governs the evolution of the safety factor $q(\rho)$ and magnetic shear.
    $$ \frac{\partial \psi}{\partial t} = \frac{\eta_{nc}}{\mu_0} \frac{1}{V'} \frac{\partial}{\partial \rho} \left( \frac{V' \langle |\nabla \rho|^2 \rangle}{R^2} \frac{\partial \psi}{\partial \rho} \right) + J_{boot} + J_{CD} $$
    *Status*: We use fixed equilibrium geometry from `equilibrium.nc`. We do not evolve $\psi$ or $q$.

3.  **Momentum Transport (Toroidal Rotation $\Omega$)**:
    $$ \frac{\partial}{\partial t} (n m R^2 \Omega) = \nabla \cdot (\chi_\phi \dots) + \tau_{NBI} $$
    *Status*: Not implemented.

4.  **Self-Consistent Equilibrium**:
    In a full code, the Grad-Shafranov equation is re-solved periodically as the pressure $P = n_e T_e + n_i T_i$ and current $J$ evolve, updating the geometry $V'(\rho)$.
    *Status*: We use fixed geometry from the initial or summary file.

5.  **Source Physics**:
    Full codes use NUBEAM (Monte Carlo) or ray-tracing to calculate exactly where NBI/RF power is deposited.
    *Status*: We use simplified Gaussian shape functions scaled by the total power $P_{NBI}(t)$.

Architecture & Gradient Flow
----------------------------

### 1. The Dynamic Loop (Coupled Physics)

The TORAX model is a **closed-loop dynamical system** where the plasma profiles, transport physics, and latent regime all interact in a continuous cycle.

The state vector at time $t$ is $S_t = \{ T_e(\rho), n_e(\rho), z \}$. These components are tightly coupled:

1.  **Latent Control ($z \to \text{Physics}$)**:
    The latent variable $z(t)$ acts as the "regime controller". It determines the magnitude and shape of the transport coefficients:
    $$ \chi_e(\rho, t) = \text{Function}(z(t)) $$
    *Mechanism*: As $z$ increases (transitioning to H-mode), $\chi_e$ at the edge drops, creating a transport barrier. The PDE solver then naturally evolves the steep temperature gradients associated with that barrier.

2.  **State Feedback ($\text{Physics} \to z$)**:
    The latent dynamics are driven by the plasma state itself. The ODE for $z$ depends on global quantities like line-averaged density $\bar{n}_e$ (derived from the $n_e$ profile):
    $$ \dot{z} = f(z, P_{NBI}, I_p, \bar{n}_e) $$
    *Mechanism*: A rise in density $n_e$ changes the drive term $\mu$, which can trigger a phase transition in $z$, which in turn alters the transport $\chi_e$, closing the feedback loop.

### 2. The Equations of Motion

The system evolves by stepping this coupled state forward in time:

1.  **PDE Step (Plasma Transport)**:
    $$ T_e^{(t+1)} = T_e^{(t)} + \Delta t \cdot \left[ \nabla \cdot (n \chi(z^{(t)}) \nabla T_e) + S_{heat} \right] $$
    $$ n_e^{(t+1)} = n_e^{(t)} + \Delta t \cdot \left[ \nabla \cdot (D(z^{(t)}) \nabla n_e) + S_{particle} \right] $$

2.  **ODE Step (Latent Dynamics)**:
    $$ z^{(t+1)} = z^{(t)} + \Delta t \cdot \left[ \alpha(\mu(u, \bar{n}_e) - \mu_c) - \beta z - \gamma z^3 \right] $$

### 3. Coupling Mechanisms

The interaction between electron temperature ($T_e$) and density ($n_e$) in this pipeline is modeled primarily through the latent regime state $z$. While full tokamak physics involves direct thermodynamic coupling, this reduced model relies on a parametric control loop.

#### Parametric Coupling (The $z$-Loop)
 The dominant coupling mechanism in this implementation is the **Regime Control Loop**:
1.  **Feedback ($n_e \to z$)**: The density profile influences the drive term $\mu(u,\bar{n}_e)$ used in the latent ODE. Crossing threshold densities tilts the system into the high-confinement branch, explicitly capturing the L-H bifurcation behavior.
2.  **Control ($z \to T_e$)**: The latent state $z$ parametrizes the equilibrium manifold and diffusivity (and therefore the learned transport coefficients), so raising $z$ enforces the L-H regime by stiffening the edge barrier while keeping the diffusion operator numerically stable.

Through this chain, the density "sets the regime," and the temperature profile responds to the resulting changes in confinement.

#### Thermodynamic Coupling (Simplification)
In a fully comprehensive physical model, the transport equations are also coupled directly through energy conservation terms:
$$ \frac{3}{2} n_e \frac{\partial T_e}{\partial t} = \nabla \cdot (n_e \chi \nabla T_e) - \frac{3}{2} T_e \frac{\partial n_e}{\partial t} $$
The final term represents **Dilution Cooling**: adding particles ($\partial_t n_e > 0$) without adding heat causes the temperature to drop.

**Current Status**: The current PDE solver simplifies this interaction by treating the $T_e$ diffusion equation independently of the $\partial_t n_e$ term. We assume that the "effective" diffusivity $\chi$ learned by the network absorbs these secondary effects for the regimes under study.

#### Independence in the Absence of $z$
It is important to note that **without the latent variable $z$, the current model equations would be uncoupled**.
*   The density $n_e(t)$ would evolve solely based on gas puffing and fixed particle diffusion.
*   The temperature $T_e(t)$ would evolve solely based on heating and fixed thermal diffusion.
*   The latent variable $z$ acts as the essential bridge, creating a one-way control loop ($n_e \to z \to T_e$) that mimics the complex regime physics of the tokamak.

### 4. Boundary Conditions & Constraints

To ensure the mathematical solution corresponds to a physical tokamak plasma, the solver enforces specific boundary conditions and global conservation laws.

#### Boundary Conditions (The "Container")
The 1D radial domain $\rho \in [0, 1]$ represents the plasma from the magnetic axis to the last closed flux surface (LCFS).

1.  **Magnetic Axis ($\rho = 0$)**:
    *   **Condition**: **Zero Flux** (Neumann).
    *   **Math**: $\left. \frac{\partial T_e}{\partial \rho} \right|_{\rho=0} = 0, \quad \left. \frac{\partial n_e}{\partial \rho} \right|_{\rho=0} = 0$.
    *   **Physics**: Due to toroidal symmetry, there is no net flow of heat or particles across the center of the plasma column. Profiles must be flat at the core.

2.  **Plasma Edge ($\rho = 1$)**:
    *   **Condition**: **Fixed Value** (Dirichlet).
    *   **Math**: $T_e(1) = T_{edge}, \quad n_e(1) = n_{edge}$.
    *   **Physics**: The edge connects to the Scrape-Off Layer (SOL) and divertor. We assume the plasma properties at this boundary are determined by external wall interactions and are effectively pinned to low values (acting as a sink).

#### Physical Constraints (The "Guardrails")
Beyond the local PDEs, the solver imposes global constraints to guide the optimization towards physically valid solutions.

1.  **Global Power Balance**:
    The total energy content of the plasma must evolve consistently with the net power input.
    $$ \frac{d W_{th}}{dt} \approx P_{heat} - P_{loss} $$
    *   **Implementation**: We add a penalty term $J_{pwr}$ to the loss function that minimizes the residual of this energy equation. This ensures that if the model learns a high temperature, it must also account for the heating source that sustains it.

2.  **Positivity**:
    Temperature and density are strictly non-negative quantities.
    *   **Implementation**: The solver applies hard clipping (e.g., $n_e \ge 0$) or soft constraints (exponential parameterization) to prevent unphysical negative values during transient solver steps.

### 5. What is Backpropagated? (Adjoint Method / BPTT)

The training process is a **Trajectory Optimization**.

1.  **Forward Pass (Integration)**:
    We run the solver from $t=0$ to $t=T_{end}$. This generates a full history of profiles:
    $$ \text{Trajectory} = \{ (T_e^{(0)}, n_e^{(0)}, z^{(0)}), \dots, (T_e^{(N)}, n_e^{(N)}, z^{(N)}) \} $$

2.  **Cost Evaluation**:
    We compare this generated trajectory to the experimental data at every time step where data exists:
    $$ J = \sum_{t=0}^N \left( || T_e^{model}(\cdot, t) - T_e^{data}(\cdot, t) ||^2 + || n_e^{model} - n_e^{data} ||^2 \right) $$

3.  **Backward Pass (Backpropagation)**:
    We compute the gradient of the total cost $J$ with respect to all learnable parameters $\theta$ (initial $z_0$, coefficients $\alpha, \beta$, baseline $\chi$).
    *   **Through the Integrator**: The gradient flows *backwards through time* (BPTT).
    *   **Chain Rule**: To know how to change $\alpha$, the gradient asks: *"How did $\alpha$ affect $z^{(t)}$? How did $z^{(t)}$ affect $\chi^{(t)}$? How did $\chi^{(t)}$ affect $T_e^{(t+1)}$? And how did that affect the error $J$?"*
    *   **Result**: The optimizer adjusts the parameters so that the *integrated physics simulation* matches the observed reality.

### 6. Advantages of ODE Solvers (Method of Lines) vs. Manual PDE Stepping

The transition from `train_4dvar.py` (manual Explicit Euler) to `train_ode.py` (Diffrax/Method of Lines) represents a fundamental shift in numerical robustness and gradient accuracy.

#### The Mathematical Problem: Stiffness
Transport equations are notoriously **stiff**. The stiffness ratio $S$ is the ratio of the fastest timescale to the slowest timescale in the system.
$$ S = \frac{\tau_{diffusion}}{\tau_{evolution}} \approx \frac{\Delta x^2 / \chi_{max}}{L^2 / \chi_{min}} $$
With high source terms ($S \sim 10^{22}$) and fine spatial grids ($\Delta x \to 0$), the diffusion timescale becomes extremely fast ($\tau_{diff} \to 0$).

*   **Explicit Solvers (Euler)**: Must satisfy the CFL condition for stability: $\Delta t < C \cdot \Delta x^2$. This forces the solver to take millions of tiny steps, making training excruciatingly slow. If $\Delta t$ is too large, the solution explodes (NaNs).
*   **Implicit Solvers (Diffrax/Kvaerno5)**: Solve a non-linear system at each step to find the next state. They are **L-stable**, meaning they can take large time steps determined only by the accuracy of the physics, not numerical stability.

#### Gradient Calculation: Adjoint Method vs. BPTT

When training a neural PDE, we need the gradient of the loss $L(y(T))$ with respect to parameters $\theta$.

**1. Backpropagation Through Time (BPTT) - Used in `train_4dvar.py`**
*   **Mechanism**: Unrolls the entire solver loop (N steps) into a massive computational graph.
*   **Memory Cost**: $O(N \cdot \text{StateSize})$. For long simulations or fine grids, this causes OOM (Out of Memory) errors.
*   **Gradient Stability**: Gradients must propagate back through thousands of multiplications. This often leads to **vanishing or exploding gradients**, making it hard to learn long-term dependencies (like the L-H transition delay).

**2. Continuous Adjoint Method - Used in `train_ode.py`**
*   **Mechanism**: Instead of differentiating the discrete solver steps, we solve the **Adjoint ODE** backwards in time:
    $$ \frac{d\lambda}{dt} = - \lambda^T \frac{\partial f}{\partial y} $$
    where $\lambda(t) = \frac{\partial L}{\partial y(t)}$ is the adjoint state (Lagrange multiplier).
*   **Memory Cost**: $O(1)$ (constant memory). The solver reconstructs the forward trajectory on the fly (or using checkpoints), so memory usage does not grow with simulation time.
*   **Accuracy**: The gradient is computed with the same adaptive precision as the forward pass.
*   **Physics-Aware**: The adjoint equation itself describes the "sensitivity transport". For diffusion, the adjoint is also a diffusion equation (running backwards), preserving the physical structure of information flow.

#### Summary of Benefits
| Feature | Manual PDE Loop (`train_4dvar.py`) | ODE Solver (`train_ode.py`) |
| :--- | :--- | :--- |
| **Stability** | Fragile (needs clipping, tiny dt) | Robust (L-stable implicit methods) |
| **Step Size** | Fixed, small (limited by CFL) | Adaptive (limited by error tolerance) |
| **Memory** | Linear $O(N_t)$ (BPTT) | Constant $O(1)$ (Adjoint) |
| **Gradients** | Prone to vanishing/exploding | Stable, high-precision |
| **Stiffness** | Fails with large sources ($10^{22}$) | Handles stiff source terms naturally |

## Control Signal Selection Rationale

- We restrict the NN input to `P_nbi`, `Ip`, `nebar`, `S_gas`, `S_rec`, and `S_nbi` because they are the only signals we can reliably extract from the Level-2 groups listed in the data catalog (`summary`, `thomson_scattering`, `equilibrium`, `gas_injection`, `spectrometer_visible`). The other datasets in the catalog (bolometers, Langmuir probes, interferometers, MSE, etc.) are valuable but either redundant with these six actuators or not uniformly available across the shots we target, so they were left out to keep the pipeline simple and consistent.
- `S_rec` comes from the spectrometer-visible D-alpha proxy (now downloaded via `download_data.py --groups spectrometer_visible`), so it tracks recycling/backflow spikes that correlate with L–H transitions; `S_gas`/`S_nbi` encode volumetric fueling from gas puffs and neutral beam injection inside `build_training_pack.py`.
- All six controls are interpolated to the common solver timeline (`ts_t`), normalized shot-by-shot, clipped where needed, and zero-filled when a diagnostic is missing, so the hybrid source sees a standalone, consistent actuator vector for every time step.

Data Quality & Validation
Recent audits of the training data packs (`data/3042*_torax_training.npz`) revealed significant quality issues:

- **Missing Profiles**: The primary training files contain >98% `NaN` values for electron temperature ($T_e$) and density ($n_e$) profiles.
- **Invalid Initial Conditions**: The first time step ($t=0$) often contains zeros or negative values, which causes numerical singularities in physics terms (e.g., resistivity $\eta \propto T_e^{-1.5}$).
- **Clean Surrogate**: A valid dataset `simulate_outputs.npz` has been identified and is currently used as the source of truth for $T_e$ targets during model development.
- **Mitigation**: The training scripts (`train_ode.py`, `train_ode_hybrid.py`) now include logic to:
  1.  Load clean targets from `simulate_outputs.npz` if available.
  2.  Synthesize parabolic initial conditions if $t=0$ data is invalid.
  3.  Mask loss functions to ignore `NaN` or unphysical values ($T_e < 1$ eV).
