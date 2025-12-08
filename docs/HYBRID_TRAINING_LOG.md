# Hybrid Physics-Machine learning Training Log

**Status:** Active (multi-shot short run validated, and working on MAST data)

## Narrative

1. **Goal:** keep the diffusion physics fixed (finite-volume flux divergence on the 65-point $\rho$ mesh) while letting a small MLP learn the residual source term that the analytic model misses. We approve `diffrax` + Equinox so the whole loop remains differentiable and the adjoint solve handles stiff diffusion.
2. **Challenge:** early versions blew up instantly—either `Kvaerno5` returned `NaN`/`inf` in the linear solve or `Tsit5` had infinite loss due to steep gradients and negative/zero temperatures in the data. That forced a full rethink of the data path, solver settings, and gradient flow.
3. **Current status:** the cheap `Tsit5` run with clipped derivatives now completes the first few steps (loss ≈ 7.5×10⁷ for `--steps 5`). No NaNs, and only a small residual-source NN is being trained while diffusion/geometry remain static.
4. **Data refresh:** shifted the download/build workflow to shots 24209–24211, now include `spectrometer_visible` so `S_rec` is populated, and verified the new packs via the notebook summary cell before running the short hybrid training smoke test (3 steps) to confirm the updated controls drive the Solver.

Diffrax/Eqinox/JAX note: we build the differentiable vector field using `diffrax.Tsit5` because it exposes adjoint-friendly time stepping, `Equinox` manages the pytree-aware model weights (manifold + source NN), and `jax` provides the just-in-time compilation, batch-friendly vmaps, and device dispatch that keep both forward and backward passes fast on CPU/GPU. Together they let us trace the PDE solve, clip gradients, and keep the entire training loop end-to-end differentiable.

## Mathematical Model

### 1. Radial Transport PDE
The core physics is governed by the 1D radial heat diffusion equation for the electron temperature $T_e(\rho, t)$. In flux coordinates, conservation of energy takes the form:
$$
\frac{\partial T_e}{\partial t} = \frac{1}{V'(\rho)} \frac{\partial}{\partial \rho} \left( V'(\rho) \chi(\rho) \frac{\partial T_e}{\partial \rho} \right) + S_{\text{net}}(\rho, T_e, n_e, u)
$$
Here, the net source function $S_{\text{net}}: [0,1] \times \mathbb{R} \times \mathbb{R} \times \mathbb{R}^6 \to \mathbb{R}$ aggregates all non-diffusive processes (heating, radiation, convection). $T_e(\rho, t) \in \mathbb{R}^+$ is the temperature profile. The radial coordinate $\rho \in [0,1]$ is defined by the normalized poloidal flux:
$$
\rho(\mathbf{x}) = \sqrt{\frac{\psi(R,Z)-\psi_{\text{axis}}}{\psi_{\text{sep}}-\psi_{\text{axis}}}}, \qquad \mathbf{x} \in \mathbb{R}^3,
$$
where $\psi_{\text{axis}}$ and $\psi_{\text{sep}}$ are the poloidal flux values at the magnetic axis and separatrix. This normalization pins the core to $\rho=0$ and the boundary to $\rho=1$.

The geometric coefficients are derived from the equilibrium reconstruction:

*   **Flux Surface**:
    *   **Definition:** A surface of constant poloidal magnetic flux $\psi(R,Z)$. In a tokamak, magnetic field lines wrap around these nested toroidal surfaces. The coordinate $\rho$ labels these surfaces, mapping the 3D structure to a 1D domain.

*   **Differential Volume ($V'(\rho)$)**:
    *   **Definition:** $V(\rho)$ is the volume enclosed by the flux surface $\rho$. $V'(\rho) = \frac{dV}{d\rho}$ is the derivative of volume with respect to the coordinate.
    *   **Role:** It acts as the Jacobian of the transformation from physical space to flux coordinates, ensuring conservation of energy in the finite-volume scheme.

*   **Thermal Diffusivity ($\chi(\rho)$)**:
    *   **Definition:** The effective coefficient governing radial heat transport. In this simplified model, it subsumes any geometric compression effects (the explicit $\langle |\nabla \rho|^2 \rangle$ term is omitted).
    *   **Model (Sigmoid Pedestal):** We use a parameterized profile to capture the L-H transition structure:
        $$ \chi(\rho) = \chi_{\text{core}} + (\chi_{\text{edge}}(z) - \chi_{\text{core}}) \cdot \sigma\left(\frac{\rho - \rho_{\text{ped}}}{w_{\text{ped}}}\right) $$
        where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the standard logistic sigmoid function. **Note:** The same function $\sigma(\cdot)$ is used for both the spatial pedestal shape (argument $\rho$) and the latent gating mechanism (argument $z$).
    *   **References:**
        *   *Pedestal Shape:* This functional form is adapted from the standard "Modified Tanh" fitting function used for H-mode pedestals (e.g., **Groebner et al., 1998**).
        *   *Dynamics:* The coupling to the latent variable $z$ is based on **Landau Theory of Phase Transitions**, where $z$ acts as the order parameter controlling the transport barrier height.
    *   **Parameters:**
        * $\chi_{\text{core}} = 0.6$: Fixed baseline diffusivity in the plasma core.
        * $\chi_{\text{edge}}(z) = \chi_{\text{edge,base}} - \chi_{\text{edge,drop}} \cdot \sigma(k z)$: The dynamic edge diffusivity. It is governed by two **fixed scalar constants**:
            * $\chi_{\text{edge,base}} = 2.0$: The baseline edge diffusivity in the L-mode limit ($z \to -\infty, \sigma \to 0$).
            * $\chi_{\text{edge,drop}} = 1.0$: The magnitude of the transport reduction in the H-mode limit ($z \to +\infty, \sigma \to 1$).
            * *Result:* $\chi_{\text{edge}}$ varies smoothly between $2.0$ (L-mode) and $1.0$ (H-mode) driven by $z(t)$.
        * $\rho_{\text{ped}} = 0.85$: The fixed radial location of the pedestal center.
        * $w_{\text{ped}} = 0.08$: The fixed width of the pedestal transition region.
        * *Note:* These values are currently hardcoded in `HybridField`.
    *   **Physical Interpretation:**
        * **Turbulence Suppression:** The drop in $\chi$ from 2.0 to 1.0 represents the suppression of edge turbulence, which is the physical mechanism behind the H-mode transport barrier.
        * **Research Context:** The spatial profile is based on the standard "Modified Tanh" fit used universally in tokamak experiments (Groebner et al.), while the dynamic control via $z$ applies Landau's phase transition theory to make the regime switching differentiable and learnable.

### 2. Finite-Volume Discretization (ODEs)
We discretize the spatial domain onto a fixed grid of $N=65$ nodes, $\rho_i$ for $i=0, \dots, 64$. The continuous field $T_e(\rho, t)$ becomes a state vector $\mathbf{T}(t) \in \mathbb{R}^{65}$.
The diffusion term is approximated using a second-order central difference scheme. First, we define the heat flux $\Gamma$ at the cell faces $i+1/2$:
$$
\Gamma_{i+1/2} = - \chi_{i+1/2} \frac{T_{i+1} - T_i}{\Delta \rho_{i+1/2}},
$$
where $\Delta \rho_{i+1/2} = \rho_{i+1} - \rho_i$. The divergence operator $\mathcal{L}: \mathbb{R}^{65} \to \mathbb{R}^{65}$ is then constructed as the net flux per unit volume:
$$
(\mathcal{L}\mathbf{T})_i = \frac{1}{V'_i \Delta \rho_i} \left( V'_{i+1/2} \Gamma_{i+1/2} - V'_{i-1/2} \Gamma_{i-1/2} \right).
$$
Boundary conditions are enforced by ghost nodes: symmetry at the axis ($\Gamma_{-1/2} = 0$) and a Dirichlet condition at the edge ($T_{N} = T_{\text{edge}}$).

### 3. Hybrid Source and Latent Dynamics
The term $S_{\text{net}}(\rho, T_e, n_e, u)$ in the PDE represents the net sum of all non-diffusive physical processes (auxiliary heating, Ohmic heating, radiation, convective transport) as well as corrections for any inaccuracies in the simplified diffusion model. Since a complete first-principles derivation of these terms is computationally intractable for control, we approximate $S_{\text{net}}$ using a hybrid data-driven approach.

**Decomposition Ansatz:**
We assume the temperature profile evolves near a low-dimensional equilibrium manifold. To operationalize this, we project the continuous source $S_{\text{net}}$ onto the finite-volume grid, yielding a vector $\mathbf{S}_{\text{net}} \in \mathbb{R}^{65}$. We then model this discrete source vector as:
$$
\mathbf{S}_{\text{net}} = \underbrace{S_{NN}(\boldsymbol{\rho}, \mathbf{T}, \mathbf{n}_e, \mathbf{u})}_{\text{Learned Residual}} + \underbrace{\frac{\partial \mathbf{T}_{\text{manifold}}}{\partial z} \frac{dz}{dt}}_{\text{Latent Structural Drive}}
$$
This formulation resolves the dimension mismatch: we are not approximating the continuous function directly, but rather its projection onto the solver grid. The neural network $S_{NN}$ is trained to close the gap between the fixed-geometry diffusion model and the observed data. By observing the state $\mathbf{T}$ and controls $\mathbf{u}$, it infers the missing power density required to match the dynamics, effectively "correcting" for both unmodeled physical sources (e.g., radiation) and geometric inaccuracies (e.g., errors in $V'$ or $\chi$).

**Role of the Neural Network ($S_{NN}$):**
The function $S_{NN}: \mathbb{R}^{65} \times \mathbb{R}^{65} \times \mathbb{R}^{65} \times \mathbb{R}^6 \to \mathbb{R}^{65}$ is a discretized mapping parameterized by a Multilayer Perceptron (MLP). It is trained to capture:
1.  **Unmodeled Physics:** Explicit sources like NBI heating profiles and radiative losses that are not included in the diffusion operator.
2.  **Model Corrections:** It compensates for the simplified 1D geometry ($V', \langle |\nabla \rho|^2 \rangle$) and the assumed diffusivity profile $\chi(\rho)$. If the fixed diffusion model under-predicts transport in a region (e.g., due to turbulence), $S_{NN}$ learns a negative "cooling" source to match the observed gradients.
3.  **Control Response:** By taking the control vector $\mathbf{u}$ as input, it learns how actuators (power, gas, current) dynamically modify the local heating rates.

Substituting this decomposition into the semi-discrete equation yields the operational ODE system:
$$
\frac{d\mathbf{T}}{dt} = \mathcal{L}(\mathbf{T}) + S_{NN}(\boldsymbol{\rho}, \mathbf{T}, \mathbf{n}_e, \mathbf{u}) + \frac{\partial \mathbf{T}_{\text{manifold}}}{\partial z} \frac{dz}{dt}.
$$

The remaining components are defined as follows:

* **Equilibrium Manifold** $\mathbf{T}_{\text{manifold}}(z)$.
  **Definition (The Slow Manifold):**
  The manifold $\mathcal{M}$ is defined as the image of the map $\Psi: \mathbb{R} \to \mathbb{R}^{65}$:
  $$
  \mathcal{M} = \{ \mathbf{T} \in \mathbb{R}^{65} \mid \mathbf{T} = \Phi(\boldsymbol{\rho}) (\mathbf{c}_{\text{base}} + \sigma(k z) \mathbf{c}_{\text{latent}}) \}
  $$
  This curve represents the **locus of all possible stationary states** (equilibria) the plasma can occupy.
  * **Base State ($z \to -\infty$):** $\mathbf{T} \approx \Phi \mathbf{c}_{\text{base}}$ (L-mode).
  * **Excited State ($z \to +\infty$):** $\mathbf{T} \approx \Phi (\mathbf{c}_{\text{base}} + \mathbf{c}_{\text{latent}})$ (H-mode).

  **Manifold Construction (Pre-computation):**
  The geometry of the manifold is fixed prior to training. The coefficients are determined by the following cost functions:
  1. **Basis ($\Phi$):** Fixed linear B-splines (hat functions) on the radial grid.
  2. **Base Profile ($\mathbf{c}_{\text{base}}$):** Determined by a Least-Squares projection of the reference shot's initial condition $\mathbf{T}_{\text{ref}}(0)$:
      $$ J_{\text{base}} = \| \mathbf{T}_{\text{ref}}(0) - \Phi \mathbf{c}_{\text{base}} \|^2 \implies \mathbf{c}_{\text{base}} = (\Phi^T \Phi)^{-1} \Phi^T \mathbf{T}_{\text{ref}}(0) $$
  3. **Latent Mode ($\mathbf{c}_{\text{latent}}$):** Currently initialized as a heuristic linear ramp.

  **Why not minimize the Physics Residual $\|\dot{\mathbf{T}}\|^2$?**
  You might ask: *If this is an equilibrium manifold, shouldn't we define it by minimizing the time derivative?*
  * **Problem:** Our analytic diffusion model is incomplete (missing radiation, exact heating profiles, etc.). Minimizing the residual of the *incomplete* physics would yield a trivial or incorrect manifold (e.g., decaying to zero).
  * **Solution (Inverse Approach):** We assume the **Experimental Data** lies on the true slow manifold. Therefore, we define the manifold geometry by minimizing **Reconstruction Error** (fitting the data).
  * **Enforcing Equilibrium:** We then train the **Residual Source** $S_{NN}$ to ensure that the physics *supports* this manifold. Effectively, the training loss forces $S_{NN}$ to cancel out the diffusion imbalance such that $\frac{d\mathbf{T}}{dt} \approx 0$ holds on the manifold states.

  **Training Cost Function (Dynamics):**
  With the manifold $\mathcal{M}$ fixed, the training process learns the *dynamics* along it (via $S_{NN}$ and potentially latent parameters). The cost function is the trajectory error:
  $$ \mathcal{L}(\theta) = \frac{1}{N_t} \sum_{t} \| \mathbf{M} (\mathbf{T}_{\text{model}}(t; \theta) - \mathbf{T}_{\text{obs}}(t)) \|^2 + \lambda \| z(t) \|^2 $$
  where $\mathbf{M}$ is the measurement mask.
  * **Note:** In the current "Smoke Test" configuration, only the weights of $S_{NN}$ are trainable ($\theta = \theta_{NN}$). The latent dynamics parameters ($\alpha, \beta, \gamma$) and manifold shape are frozen.

  **Physical Interpretation:**
  * **Manifold:** The set of points where the fast dynamics have decayed.
  * **Dynamics:** The equation $dz/dt$ describes the system sliding along this sequence of equilibria as it transitions from L-mode to H-mode.
  * **Fixed Point Selection:** While the manifold defines *all* possible shapes, the control input $\mathbf{u}$ determines which specific point $z^*$ is the stable fixed point for the current phase (where $dz/dt = 0$).

* **Latent Dynamics**.
  The scalar latent variable $z(t) \in \mathbb{R}$ evolves according to a cubic normal form:
  $$
  \frac{dz}{dt} = \alpha (\mu(\mathbf{u}) - \mu_{\text{ref}}) - \beta z - \gamma z^3,
  $$
  **Control Mechanism:**
  * $\mu(\mathbf{u})$ (**Bifurcation Parameter**): A learned linear combination of controls: $P_{NBI}$ (Power), $I_p$ (Current), and $\bar{n}_e$ (Density).
  * **Mechanism:**
    * When input power is low, $\mu(\mathbf{u})$ is small. The system has a stable fixed point at low $z$ (L-mode).
    * As you increase power ($P_{NBI}$), $\mu(\mathbf{u})$ increases.
    * When $\mu(\mathbf{u})$ crosses a threshold, the low-$z$ fixed point becomes unstable (or disappears), and the system jumps to a high-$z$ fixed point (H-mode).
  * **Hysteresis:** The cubic term $-z^3$ creates a "potential well" structure. The path from L $\to$ H is different from H $\to$ L, creating the hysteresis loop observed in real experiments.

### 4. Theoretical Justification

**Justification for a Single Variable $z$:**
You might ask why we can parameterize the entire equilibrium manifold with just a single variable $z$.

1.  **Profile Consistency (Stiffness):**
    In tokamak physics (Coppi, 1980), electron temperature profiles are observed to be "stiff." This means that due to critical gradient turbulence, the profile tends to relax to a canonical shape regardless of where you deposit heat.
    *   **Implication:** The effective degrees of freedom are very low. You don't need 65 variables to describe the equilibrium; you largely just need to know the "amplitude" (core temperature or pedestal height). The shape is fixed by transport physics.
    *   **Our Model:** $z$ captures this single degree of freedom (confinement quality), while $\Phi(\rho)$ enforces the consistent shape.

2.  **Center Manifold Theory:**
    In dynamical systems theory, when a system undergoes a bifurcation (like the L-H transition), the dynamics are dominated by the "slow" modes that become unstable, while the "fast" stable modes decay rapidly.
    *   **Slaving Principle:** The fast variables (the detailed shape of the profile, turbulence fluctuations) are "slaved" to the slow order parameter ($z$).
    *   **Feasibility:** This justifies using a 1D manifold for the transition. The complex 65-dimensional PDE state rapidly relaxes onto this 1D curve defined by $z$.

**Feasibility & Related Studies:**
*   **Is this standard?**
    *   **Physics:** Yes. This is mathematically equivalent to **Landau Theory of Phase Transitions**, which is the standard framework for describing spontaneous symmetry breaking (like L-H transition).
    *   **Reduced Models:** Codes like **RAPTOR** (Felici et al.) use a similar idea but with ad-hoc heuristics. They model the profile using a few scalar parameters (core value, edge value, width). Your approach is more rigorous because it learns the manifold shape from data rather than assuming a functional form.
    *   **Dynamical Systems:** This is a **Normal Form** approach. Instead of trying to model the millions of turbulence modes that cause the transition, we model the *topology* of the bifurcation itself.

*   **Feasibility:**
    This is highly feasible and robust because:
    1.  **It's Bounded:** The sigmoid prevents the model from predicting infinite temperatures (a common failure mode in pure Neural ODEs).
    2.  **It's Interpretable:** You can plot $z(t)$ and see exactly when the model thinks the transition happens.
    3.  **It's Data-Efficient:** The neural network only has to learn the *residual* (the small difference between the manifold and reality), not the whole physics from scratch.

## Boundary Conditions & Matrix Formulation

The finite-volume discretization converts the PDE into a system of ODEs. The boundary conditions are incorporated directly into the linear diffusion operator $\mathbf{D}$ and the source vector.

### 1. Axis Symmetry (Neumann at $\rho=0$)

**Condition:** $\left. \frac{\partial T_e}{\partial \rho} \right|_{\rho=0} = 0$.

**Detailed Derivation:**
To enforce symmetry at the magnetic axis, we utilize the concept of **ghost nodes** and **ghost faces**.

1. **Ghost Face ($\Gamma_{-1/2}$):** The face at $i=-1/2$ corresponds to the physical boundary at $\rho=0$. Since the temperature profile is symmetric ($T(\rho) = T(-\rho)$), the heat flux across the axis must be zero.
2. **Ghost Node ($T_{-1}$):** We imagine a virtual node at $i=-1$ inside the "negative radius". Symmetry implies $T_{-1} = T_0$.
3. **Gradient Calculation:** The gradient at the axis face is approximated by the central difference between the ghost node and the first physical node:
    $$ \left. \frac{\partial T}{\partial \rho} \right|_{-1/2} \approx \frac{T_0 - T_{-1}}{\Delta \rho} = \frac{T_0 - T_0}{\Delta \rho} = 0 $$
    Consequently, the flux $\Gamma_{-1/2} = -\chi \nabla T|_{-1/2}$ vanishes.

**Finite-Volume Update:**
The temperature update for the first physical node ($i=0$) is driven by the net flux divergence:
$$
\frac{dT_0}{dt} = \frac{1}{V'_0 \Delta \rho_0} (V'_{1/2} \Gamma_{1/2} - \underbrace{V'_{-1/2} \Gamma_{-1/2}}_{0})
$$
Because $\Gamma_{-1/2}=0$, the term representing flux from the left disappears. Substituting the expression for the flux from the right ($\Gamma_{1/2} = -\chi_{1/2} \frac{T_1 - T_0}{\Delta \rho_{1/2}}$):
$$
\frac{dT_0}{dt} = \underbrace{\left( \frac{V'_{1/2} \chi_{1/2}}{V'_0 \Delta \rho_0 \Delta \rho_{1/2}} \right)}_{\alpha} (T_1 - T_0)
$$

**Matrix Form:**
The first row of the linear diffusion operator $\mathbf{D}$ becomes:
$$
\mathbf{D}_{0,:} = \begin{bmatrix} -\alpha & \alpha & 0 & \dots & 0 \end{bmatrix}
$$
This formulation naturally enforces the Neumann condition by ensuring the update depends only on the difference $(T_1 - T_0)$, effectively treating the boundary as an insulating wall.

### 2. Edge Condition (Dirichlet at $\rho=1$)

**Condition:** $T_e(\rho=1) = T_{\text{edge}}(t)$.

**Implementation:**
The temperature at the boundary $T_N = T_{\text{edge}}$ is a fixed (or time-dependent) value from the dataset. The flux at the last face ($i=N-1/2$) is:
$$
\Gamma_{N-1/2} = -\chi_{N-1/2} \frac{T_{\text{edge}} - T_{N-1}}{\Delta \rho_{N-1/2}}
$$
This term introduces a dependency on $T_{\text{edge}}$ which is not part of the state vector $\mathbf{T}$. We separate this into a linear part and a boundary source vector $\mathbf{b}_{\text{edge}}$:
$$
\frac{d\mathbf{T}}{dt} = \mathbf{D}\mathbf{T} + \mathbf{b}_{\text{edge}}(T_{\text{edge}}) + \mathbf{S}_{\text{net}}
$$
where $\mathbf{b}_{\text{edge}}$ is zero everywhere except for the last node ($i=N-1$).

### 3. Control Inputs

The control vector $\mathbf{u}$ (NBI, $I_p$, density, etc.) enters the system through:

1. **Latent Dynamics:** Modulating the bifurcation parameter $\mu(\mathbf{u})$.
2. **Neural Source:** Providing context to $S_{NN}(\boldsymbol{\rho}, \mathbf{T}, \mathbf{n}_e, \mathbf{u})$ to learn heating and fueling rates.

| Channel | Usage |
| :--- | :--- |
| $P_{NBI}, I_p, \bar{n}_e$ | Drives latent regime transitions ($\mu$) and $S_{NN}$. |
| $S_{gas}, S_{rec}, S_{nbi}$ | Fueling sources for $S_{NN}$. |

## State/Observation Structure

**State evolution:** The full ODE state is $z=(T_e, n_e, z_{\text{latent}})$ with controls $u=(P_{NBI}, I_p, \bar n_e, S_{gas}, S_{rec}, S_{nbi})$. The evolution equations are:
$$
\frac{dT_e}{dt} = \underbrace{\text{diffusion}(T_e, V', \rho, \boldsymbol{\theta}_{\text{geom}}) + S_{NN}(\rho, T_e, n_e, u)}_{\text{Residual Dynamics } (\dot{\tilde{T}}_e)} + \underbrace{\Phi(\rho) \frac{d}{dt}\left[\sigma(k z_{\text{latent}}) \mathbf{c}_{\text{latent}}\right]}_{\text{Manifold Drive } (\dot{T}_{\text{manifold}})}
$$
$$
\frac{dn_e}{dt} = 0 \quad (\text{Replay from data})
$$
$$
\frac{dz_{\text{latent}}}{dt} = \alpha(\mu(u) - \mu_{\text{ref}}) - \beta z_{\text{latent}} - \gamma z_{\text{latent}}^3
$$

**Derivation of the $T_e$ Equation (Chain Rule Detail):**
The temperature profile is decomposed into a moving equilibrium manifold and a dynamic residual:
$$
T_e(\rho, t) = T_{\text{manifold}}(\rho, z_{\text{latent}}(t)) + \tilde{T}_e(\rho, t)
$$
Differentiating with respect to time $t$:
$$
\frac{dT_e}{dt} = \frac{d}{dt} T_{\text{manifold}}(\rho, z_{\text{latent}}(t)) + \frac{d\tilde{T}_e}{dt}
$$

1. **Manifold Term (Chain Rule):**
   The manifold is defined as $T_{\text{manifold}} = \Phi(\rho) (\mathbf{c}_{\text{base}} + \sigma(k z_{\text{latent}}) \mathbf{c}_{\text{latent}})$. Since $\Phi(\rho)$, $\mathbf{c}_{\text{base}}$, and $\mathbf{c}_{\text{latent}}$ are constant in time, the time derivative only acts on the sigmoid term via the chain rule:
   $$
   \begin{aligned}
   \frac{d}{dt} T_{\text{manifold}} &= \Phi(\rho) \mathbf{c}_{\text{latent}} \frac{d}{dt} \left[ \sigma(k z_{\text{latent}}(t)) \right] \\
   &= \Phi(\rho) \mathbf{c}_{\text{latent}} \cdot \sigma'(k z_{\text{latent}}) \cdot k \cdot \frac{dz_{\text{latent}}}{dt}
   \end{aligned}
   $$
   This term represents the "structural drive"—as the latent state $z$ moves (e.g., L-H transition), it forces the temperature profile to deform along the manifold.

2. **Residual Term:**
   The residual $\tilde{T}_e$ is governed by the physics (diffusion) and the neural source correction:
   $$
   \frac{d\tilde{T}_e}{dt} = \text{diffusion}(T_e) + S_{NN}(\rho, T_e, n_e, u)
   $$
   Note that diffusion acts on the *total* temperature $T_e$, not just the residual.

**Density and Controls:**

* **Density:** The density state is not governed by a differential equation: we interpolate the measured $n_e(\rho,t)$ onto the solver grid at preprocessing, represent it as $g(t)$, and then set $n_e(t)=g(t)$ during the solve.
* **Controls:** The drive $\mu(u)$ is a linear map of $(P_{NBI}, I_p, \bar n_e)$, so shots that change these controls force $z_{\text{latent}}$ to move.
 
   **Loss & observations:** the measurement operator $H(z,u)$ simply selects the masked $T_e$ entries, so the optimizer only compares the modeled Te to trusted measurements. The cost is
    $$
    \mathcal{L} = \frac{1}{\sum_i m_i + \epsilon} \sum_i m_i \left(T_{e,\text{model},i} - T_{e,\text{target},i} \right)^2 + \lambda_{\text{latent}} \frac{1}{N_t} \sum_t z_{\text{latent}}(t)^2,
    $$
    where $m_i$ comes from `ts_mask_Te`, $\epsilon=10^{-8}$ prevents division by zero, $\lambda_{\text{latent}}$ weights the latent regularizer (default $\lambda_{\text{latent}}=1$), and the equilibrium coefficients $\mathbf{c}_{\text{base}}, \mathbf{c}_{\text{latent}}$ stay fixed from Te0. This means the only trainable pieces are $S_{NN}$ and the latent controller while the diffusion backbone and observed Te remain physics-grounded.

## Numerical Stability Measures

* **Data gating:** Every `Te` time slice passes through the pack-provided `Te_mask` plus finite/positive filters. We replace bad initial profiles (NaNs, ≤10 eV) with a smooth parabolic profile so the diffusion solve never starts from a singular state.
* **Derivative clipping:** We limit $dT_e/dt$ to $\pm 10^4$ eV/s. This prevents the `Tsit5` adaptive stepper from selecting unnecessarily small steps when the solver sees momentary spikes.
* **Fixed Diffusion Coefficients:** We keep $\chi(\rho)$, $V'$, and the finite-volume stencils fixed. Allowing these to be learned would destabilize the stiff solver. Only $S_{NN}$ and the latent gate are trainable.
* **Solver settings:** Switched to `diffrax.Tsit5` with `rtol=atol=1e-3`, `dt0=5e-4`, `max_steps=100000`, and `throw=False`.

## Data Inventory and Inputs

* **Measured signals:** Each training pack contains the following arrays that we resample/interpolate before solving the ODE:

| Variable | Origin | Used for | Notes |
| :--- | :--- | :--- | :--- |
| `Te` | Thomson-event trajectory | Loss target | Masked by `Te_mask`, clipped to >1 eV. |
| `Te_mask` | Pack-provided mask | Loss filter | Marks valid Te entries. |
| `ne` | Density time series | NN input | Resampled onto `ts_t`, clipped to `[1e17, 1e19]`. |
| `P_nbi`, `Ip`, `nebar` | Control history | Latent Drive | Drives $\mu(\mathbf{u})$ for regime change. |
| `S_gas`, `S_rec`, `S_nbi` | Control history | NN Input | Fueling context for $S_{NN}$. |
| `Vprime`, `rho` | Equilibrium | Geometry | Fixed geometric coefficients. |

* **Multi-shot Training:** The script loads every pack matching `data/*_torax_training*.npz`. Each pack keeps its own `Te` mask and controls but shares the solver grid and NN weights.
* **Clean Data:** `simulate_outputs.npz` is the canonical target. We resample pack controls onto this shared timeline.

## Actionable Next Steps

1. **Loss decrease:** Run `python -m scripts.train_ode_hybrid ... --steps 50` and confirm loss reduction.
2. **Data monitors:** Log the percentage of positive (masked) Te entries.
3. **Solver diagnostics:** Instrument the vector field to return clipped derivative magnitudes.
4. **Next guard:** Add a regularizer for $S_{NN}$ deviation from zero.
