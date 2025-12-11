# Physics-Consistent Manifold Learning

**Status:** Active – implemented in `train_ode_physics_manifold.py`

## 1. Introduction

We consider a state vector $\mathbf{T}(t)\in\mathbb{R}^N$ representing the discretised electron temperature profile on a radial grid of $N=65$ nodes. This resolution is chosen to match the high-fidelity Thomson scattering diagnostics available from the MAST tokamak, ensuring that the model captures the fine-scale structure of the pedestal and core transport barriers without artificial interpolation. Its evolution is assumed to decompose into motion along a one-dimensional slow manifold and a fast transverse component,
$$
\mathbf{T}(t) = \mathbf{T}_{\mathcal{M}}(z(t)) + \tilde{\mathbf{T}}(t),
$$
where $\mathbf{T}_{\mathcal{M}}:\mathbb{R}\to\mathbb{R}^N$ is a smooth embedding of a scalar latent coordinate $z(t)\in\mathbb{R}$ defining a one-dimensional manifold $\mathcal{M}\subset\mathbb{R}^N$, and $\tilde{\mathbf{T}}(t)\in\mathbb{R}^N$ captures fast deviations from $\mathcal{M}$ that relax on shorter time scales.

This framework serves as a **physically consistent foundation model** for tokamak transport. By embedding the neural closure within a rigorous transport PDE structure, we ensure that the learned dynamics respect fundamental conservation laws and geometric constraints.

**The Role of TORAX:**
The TORAX transport solver [10] plays a critical role in this pipeline, serving as the **bridge** between the differentiable hybrid model and the physically meaningful backbone. TORAX supplies the validated numerical infrastructure—specifically the finite-volume discretisation on magnetic flux coordinates and the equilibrium geometry definitions—that grounds the neural closure in established plasma physics. Rather than learning physics from scratch, our model acts as a differentiable extension of TORAX. It preserves the solver's conservation laws and geometric dependencies ($V'(\rho)$) while replacing the rigid semi-empirical correlations (e.g., Bohm/Gyro-Bohm) with a flexible, data-driven residual operator $\mathbf{S}_{\text{net}}$. This ensures that the learned corrections are physically interpretable modifications to a standard transport simulation, rather than arbitrary black-box outputs.

Two equilibrium notions appear in the formulation. A **dynamical equilibrium** corresponds to a stationary transport state ($\partial\mathbf{T}/\partial t = 0$). A **magnetic equilibrium** refers to the background magnetic field configuration satisfying the Grad–Shafranov equation. In this work, the magnetic equilibrium is treated as a static or slowly-varying exogenous input that defines the flux-surface geometry and the specific volume $V'(\rho)$, decoupling the fast transport timescales from the slower magnetic diffusion.

The full plasma state evolves in a high-dimensional MHD state space involving temperature fields, density, magnetic fields, velocity fields, and additional coupled quantities. In this work we retain explicitly only the electron temperature profile $\mathbf{T}\in\mathbb{R}^N$ and represent the influence of all unresolved degrees of freedom through a closure operator. Motivated by the Mori–Zwanzig projection formalism [1, 2], unresolved variables induce memory terms that can be approximated, under a separation of timescales, by deterministic functionals of the resolved variables. To this end, we introduce a parameterized source term $\mathbf{S}_{\text{net}}(\boldsymbol{\rho}, \mathbf{T}, \mathbf{n}_e, \mathbf{u}, z)$ in the semi-discrete transport equation. Here, $\boldsymbol{\rho}$ is the vector of grid coordinates, $\mathbf{n}_e$ denotes the electron density profile, $\mathbf{u}$ represents the vector of external actuators (e.g., heating power, plasma current), and $z$ is the latent manifold coordinate governing the transition dynamics. This term plays several roles simultaneously: it compensates for unmodelled transport physics such as omitted nonlinearities, cross-field couplings, and higher-order transport effects; it approximates the slaving functional that maps unresolved MHD state variables to their influence on $\mathbf{T}$; and it compensates for modelling and reconstruction errors in equilibrium-derived coefficients. In this sense, $\mathbf{S}_{\text{net}}$ acts as a data-driven correction operator for missing physics, truncated degrees of freedom, and imperfect equilibrium inputs.

**Robustness to Experimental Artifacts:**
Experimental tokamak data is frequently sparse, noisy, or corrupted by diagnostic failures (e.g., Thomson scattering dropouts). To ensure the procedure is practical and robust, we introduce a binary masking operator $\mathbf{M}$ acting on the state space. This operator filters out invalid measurements (NaNs) and unphysical outliers, ensuring that the training objective is computed only on valid data points. By incorporating $\mathbf{M}$ directly into the loss formulation, the differential equation solver is allowed to "coast" through temporal gaps in the data, guided by the learned physics and conservation laws, rather than being forced to fit numerical artifacts. This decoupling of the physical simulation from the imperfections of the observation grid is crucial for training on raw experimental databases.

The restriction to a scalar latent coordinate $z$ is justified by classical results in bifurcation theory. The transition dynamics of interest are effectively codimension one and has a saddle-node structure. Near such a bifurcation, the centre manifold theorem [3] guarantees the existence of a one-dimensional invariant manifold tangent to the critical eigenspace, while Haken’s slaving principle [4] implies that the amplitudes of stable modes are enslaved to the slow coordinate and decay rapidly. Consequently, the macroscopic temperature profile can be parameterised as $\mathbf{T}_{\mathcal{M}}(z)$, and the pair $(z,\tilde{\mathbf{T}})$ provides a natural decomposition into slow and fast components.

The spatial coordinate $\rho\in[0,1]$ is discretised into $N$ nodes, yielding a method-of-lines formulation in $\mathbb{R}^N$. The resulting semi-discrete temperature dynamics are integrated using adaptive, differentiable ODE solvers (such as Tsit5 or Kvaerno) within the JAX Diffrax framework [5, 6]. Unlike the native TORAX solver, which is primarily engineered for forward prediction stability, production robustness, and operator-split stepping, this formulation exposes the entire time-evolution map as a differentiable computational graph. In the machine learning setting, this is essential: differentiable solvers support adjoint-based backpropagation through the ODE, enabling efficient end-to-end optimisation of the neural source $\mathbf{S}_{\text{net}}$ and of the manifold representation $\mathbf{T}_{\mathcal{M}}$.

From the perspective of dynamical-systems analysis, the differentiable semi-discrete formulation provides a mathematically explicit ODE on $\mathbb{R}^N$ that is well suited as a substrate for continuation and stability studies. Because the right-hand side is available in closed form and is differentiable, one can apply automatic differentiation to obtain Jacobians of the vector field and thereby construct the operators required for dynamic-equilibrium or periodic-orbit continuation. This contrasts with finite-difference or operator-split implementations, as used in TORAX, where the update rule is not naturally expressed as an explicit ODE map and exact Jacobian construction becomes difficult and often incompatible with Newton–Krylov methods. The differentiable ODE formulation does not by itself perform continuation, but it supplies the complete mathematical ingredients—explicit vector field, differentiable right-hand side, and accessible Jacobians—needed by modern continuation and stability algorithms, while remaining compatible with gradient-based training of the embedded neural components.


## 2. Governing Transport Model

### 2.1 Hybrid Transport-Latent Model

The system is modelled as a hybrid dynamical system coupling an infinite-dimensional PDE for the profile evolution with a low-dimensional ODE for the latent order parameter. The continuous electron temperature $T_e(\rho,t)$ and the latent coordinate $z(t)$ evolve according to:
$$
\begin{aligned}
\frac{\partial T_e}{\partial t}
&= \frac{1}{V'(\rho)}\frac{\partial}{\partial \rho}\left(V'(\rho)\,\chi(\rho)\,\frac{\partial T_e}{\partial \rho}\right)
+ \mathcal{S}_{\text{net}}(\rho,T_e,n_e,\mathbf{u}, z), \\
\frac{dz}{dt} &= f_{\text{latent}}(z, \mathbf{u}),
\end{aligned}
$$
where $\rho\in[0,1]$ is the normalised poloidal flux coordinate, defined in terms of the equilibrium poloidal flux by
$$
\rho(R,Z)
= \sqrt{\frac{\psi(R,Z)-\psi_{\text{axis}}}{\psi_{\text{sep}}-\psi_{\text{axis}}}},
$$
with $\psi_{\text{axis}}$ and $\psi_{\text{sep}}$ denoting the flux values at the magnetic axis and separatrix, respectively. The quantity $V'(\rho)=dV/d\rho$ is the differential volume enclosed by the flux surface labelled by $\rho$, supplied by the equilibrium reconstruction. The scalar function $\chi(\rho)$ denotes an effective thermal diffusivity, and the term $\mathcal{S}_{\text{net}}(\rho,T_e,n_e,\mathbf{u}, z)$ aggregates non-diffusive contributions including auxiliary heating, radiative losses, convective transport, and model mismatch.

Formally, let $\mathcal{H} = L^2([0,1]; V' d\rho)$ denote the weighted Hilbert space of square-integrable functions on the radial domain. The source term is defined as a nonlinear operator between function spaces:
$$
\mathcal{S}_{\text{net}}: \mathcal{H} \times \mathcal{H} \times \mathcal{H} \times \mathbb{R}^M \times \mathbb{R} \to \mathcal{H},
$$
which maps the spatial coordinate field $\rho \in \mathcal{H}$ (where $\rho(x)=x$), the full temperature profile $T_e \in \mathcal{H}$, and density profile $n_e \in \mathcal{H}$, along with the global control vector $\mathbf{u} \in \mathbb{R}^M$ and latent coordinate $z \in \mathbb{R}$, to a source profile in $\mathcal{H}$. The control vector $\mathbf{u}$ resides in $\mathbb{R}^M$ because it consists of global scalar actuators (e.g., total heating power, plasma current).

### 2.2 Geometry and transport coefficients

Flux surfaces corresponding to constant $\psi$ or constant $\rho$ induce a one-dimensional reduction of the geometry, mapping the three-dimensional plasma domain to the interval $\rho\in[0,1]$. The Jacobian factor $V'(\rho)$ acts as the volume element in this reduced description and enforces energy conservation within the finite-volume discretisation.  

The thermal diffusivity is modelled as a smooth pedestal-like profile, capturing the formation of transport barriers:
$$
\chi(\rho)
= \chi_{\text{core}} + \big(\chi_{\text{edge}}(z)-\chi_{\text{core}}\big)\,\sigma\left(\frac{\rho-\rho_{\text{ped}}}{w_{\text{ped}}}\right),
$$
where $\chi_{\text{core}}$ is the core diffusivity, $\rho_{\text{ped}}$ and $w_{\text{ped}}$ define the pedestal location and width, and $\sigma$ is a logistic (sigmoid) function. The edge diffusivity is modulated by the latent coordinate via
$$
\chi_{\text{edge}}(z)
= \chi_{\text{edge,base}} - \chi_{\text{edge,drop}}\,\sigma(k z),
$$
with $\chi_{\text{edge,base}}$ and $\chi_{\text{edge,drop}}$ specifying the baseline and the latent-controlled reduction of edge transport. This functional form constitutes a differentiable ansatz for the transport barrier. While standard pedestal analyses [7] typically fit the temperature profile $T_e(\rho)$ directly using a modified hyperbolic tangent (mtanh) function, our approach imposes this structure on the diffusivity $\chi(\rho)$. This is physically motivated by the fact that the steep temperature gradient arises from a localized reduction in transport. The logistic function $\sigma(\cdot)$ provides a $C^\infty$-smooth approximation to the step-change in confinement, ensuring compatibility with the differentiable ODE solver. The modulation of $\chi_{\text{edge}}$ by $z$ effectively implements a Landau-type phase transition [8], where the order parameter $z$ continuously drives the system between high-transport (L-mode) and low-transport (H-mode) branches, with the sigmoid ensuring physical saturation of the transport coefficients.

### 2.3 Spatial discretisation

The radial interval $[0,1]$ is discretised at nodes $\{\rho_i\}_{i=1}^N$ with associated cell widths $\Delta\rho_i$ and interface widths $\Delta\rho_{i\pm1/2}$. The discrete heat flux through the interface between nodes $i$ and $i+1$ is
$$
\Gamma_{i+1/2} = -\chi_{i+1/2}\frac{T_{i+1}-T_i}{\Delta\rho_{i+1/2}},
$$
and the discrete divergence operator is defined by
$$
(\mathbf{D}\mathbf{T})_i
= \frac{1}{V'_i\Delta\rho_i}\left(V'_{i+1/2}\Gamma_{i+1/2}-V'_{i-1/2}\Gamma_{i-1/2}\right),
$$
where $V'_i$ and $V'_{i\pm1/2}$ denote the cell-centred and interface values of the equilibrium Jacobian $V'(\rho)$, respectively.

**Alignment with TORAX Numerics:**
This specific finite-volume scheme is adopted directly from TORAX. By sharing the exact numerical stencil, grid staggering, and geometric coefficients ($V', g$) with the legacy solver, the hybrid model ensures that any deviation from a baseline simulation is attributable strictly to the learned physics $\mathbf{S}_{\text{net}}$ and not to numerical discrepancies.

We define the geometric conductance coefficients $g_{i+1/2} = \frac{V'_{i+1/2}\chi_{i+1/2}}{\Delta\rho_{i+1/2}}$ and the inverse volume factor $\alpha_i = \frac{1}{V'_i\Delta\rho_i}$. The discrete divergence operator $\mathbf{D}$ is linear and acts on the state vector $\mathbf{T}$ via matrix multiplication. The $i$-th component $(\mathbf{D}\mathbf{T})_i$ corresponds to the discrete flux balance at node $i$. The operator is represented by a tridiagonal matrix $\mathbf{D} \in \mathbb{R}^{N \times N}$ with entries:

**Interior nodes ($1 < i < N$):**
$$
\begin{aligned}
D_{i,i-1} &= \alpha_i g_{i-1/2} \\
D_{i,i}   &= -\alpha_i (g_{i-1/2} + g_{i+1/2}) \\
D_{i,i+1} &= \alpha_i g_{i+1/2}
\end{aligned}
$$

**Boundary conditions:**
1.  **Magnetic Axis ($i=1$):** The Neumann condition $\partial T_e/\partial\rho|_{\rho=0}=0$ implies zero flux at the inner boundary ($\Gamma_{1/2}=0$), equivalent to setting $g_{1/2}=0$. The first row becomes:
    $$
    D_{1,1} = -\alpha_1 g_{3/2}, \quad D_{1,2} = \alpha_1 g_{3/2}.
    $$
2.  **Plasma Edge ($i=N$):** The Dirichlet condition $T(\rho=1) = T_{\text{edge}}$ introduces a dependency on the external boundary value $T_{N+1} \equiv T_{\text{edge}}$. This modifies the last diagonal element and generates the affine source vector $\mathbf{b}_{\text{edge}}$:
    $$
    D_{N,N-1} = \alpha_N g_{N-1/2}, \quad D_{N,N} = -\alpha_N (g_{N-1/2} + g_{N+1/2}).
    $$
    The contribution from the fixed boundary value is separated into the vector $\mathbf{b}_{\text{edge}} \in \mathbb{R}^N$:
    $$
    \mathbf{b}_{\text{edge}} = \begin{bmatrix} 0 \\ \vdots \\ 0 \\ \alpha_N g_{N+1/2} T_{\text{edge}} \end{bmatrix}.
    $$

To obtain the semi-discrete system, we employ a nodal collocation approximation. The operator $\mathbf{S}_{\text{net}}$ is parameterised as a **Nemytskii (superposition) operator** induced by a local residual function $R_{\theta}$. This means the operator acts pointwise on the function values.

For any spatial point $x \in [0,1]$, the local source intensity is given by:
$$
[\mathbf{S}_{\text{net}}(\rho, T_e, n_e, \mathbf{u}, z)](x) = R_{\theta}(\rho(x), T_e(x), n_e(x), \mathbf{u}, z).
$$
Here, $R_{\theta}: [0,1] \times \mathbb{R} \times \mathbb{R} \times \mathbb{R}^M \times \mathbb{R} \to \mathbb{R}$ is a universal function approximator (implemented as a neural network) parameterized by a set of weights $\theta_{\text{source}}$.

For each node $i$ in the discrete model, the source term is evaluated as:
$$
(\mathbf{S}_{\text{net}})_i = R_{\theta}(\rho_i, T_i, n_{e,i}, \mathbf{u}, z).
$$

**Mathematical Note on Residual Physics:**
The inclusion of $\rho$ allows the approximator to learn spatially localized corrections. The term $\mathbf{S}_{\text{net}}$ acts as a closure for all physics not captured by the simplified diffusion operator $\mathbf{D}$. If the "true" evolution is governed by an operator $\mathcal{F}_{\text{true}}$, then $\mathbf{S}_{\text{net}}$ learns the residual:
$$
\mathbf{S}_{\text{net}} \approx \mathcal{F}_{\text{true}}(\mathbf{T}) - \mathbf{D}\mathbf{T}.
$$
This residual includes:
*   **Unmodelled Transport:** Convective terms ($\mathbf{v} \cdot \nabla T$) and radiative losses not present in the diffusion model.
*   **Diffusive Errors:** Corrections for the mismatch between the assumed fixed diffusivity $\chi(\rho)$ and the true turbulent diffusivity $\chi_{\text{true}}(\rho)$.
*   **External Sources:** The spatial deposition profiles of auxiliary heating (NBI, ECRH) which depend strongly on $\rho$.

The complete semi-discrete evolution equation for the temperature state $\mathbf{T}$ combines the diffusive transport, the neural source, and the coherent motion of the slow manifold:
$$
\frac{d\mathbf{T}}{dt}
= \mathbf{D}\mathbf{T} + \mathbf{b}_{\text{edge}}
+ \mathbf{S}_{\text{net}}(\boldsymbol{\rho}, \mathbf{T},\mathbf{n}_e,\mathbf{u}, z)
+ \frac{\partial \mathbf{T}_{\mathcal{M}}}{\partial z}\frac{dz}{dt}.
$$
The last term represents the explicit chain-rule contribution from the evolution of the latent coordinate $z$. The decomposition implies that $\mathbf{S}_{\text{net}}$ accounts for the residual physics required to match the observed dynamics, given the prescribed manifold motion and diffusive transport.

This formulation assumes a **quasi-steady** approximation for the manifold: the training objective (Section 5) encourages the total tendency to be close to zero on the manifold. This implies that the manifold consists of dynamical equilibria (fixed points) of the transport system for a given $z$ and $\mathbf{u}$. The actual evolution along the manifold is driven by the slow dynamics of $z(t)$ and the external changes in controls $\mathbf{u}(t)$, which shift the equilibrium location.

## 3. Equilibrium Manifold Construction

### 3.1 Parameterisation

The slow manifold $\mathcal{M}$ is parameterised as a one-parameter family of profiles embedded in $\mathbb{R}^N$. We employ a linear basis expansion with a nonlinear coefficient modulation:
$$
\mathcal{M}
= \left\{\mathbf{T}_{\mathcal{M}}(z)
= \Phi(\boldsymbol{\rho})\big(\mathbf{c}_{\text{base}} + \sigma(kz)\mathbf{c}_{\text{latent}}\big)
\;\middle|\; z\in\mathbb{R}\right\},
$$
where:
*   $\Phi(\boldsymbol{\rho})\in\mathbb{R}^{N\times K}$ is a fixed matrix of B-spline basis functions evaluated on the grid.
*   $\mathbf{c}_{\text{base}}\in\mathbb{R}^K$ is a **fixed** coefficient vector representing the reference (L-mode) profile shape.
*   $\mathbf{c}_{\text{latent}}\in\mathbb{R}^K$ is a **trainable** parameter vector ($\theta_{\text{manifold}}$) that encodes the deformation of the profile shape associated with the transition to H-mode.
*   $\sigma(\cdot)$ is the sigmoid function and $k$ is a fixed gain parameter.

The map $\mathbf{T}_{\mathcal{M}}:\mathbb{R}\to\mathbb{R}^N$ is smooth and globally defined. Its derivative with respect to the latent coordinate is:
$$
\frac{\partial \mathbf{T}_{\mathcal{M}}}{\partial z}
= \Phi(\boldsymbol{\rho})\mathbf{c}_{\text{latent}}\sigma'(kz)\,k.
$$

### 3.2 Justification of the Functional Form

The functional form $\mathbf{c}(z) = \mathbf{c}_{\text{base}} + \sigma(kz)\mathbf{c}_{\text{latent}}$ is chosen to represent a smooth transition between two distinct topological states of the plasma:
1.  **L-mode limit ($z \to -\infty$):** $\sigma(kz) \to 0$, so $\mathbf{T} \to \Phi \mathbf{c}_{\text{base}}$. This corresponds to the baseline low-confinement profile.
2.  **H-mode limit ($z \to +\infty$):** $\sigma(kz) \to 1$, so $\mathbf{T} \to \Phi (\mathbf{c}_{\text{base}} + \mathbf{c}_{\text{latent}})$. This corresponds to the high-confinement profile with a developed pedestal.

The sigmoid modulation acts as a soft switch, allowing the latent variable $z$ to drive the system continuously between these regimes. This is consistent with the physical observation of the L-H transition as a bifurcation where the profile shape stiffens and develops a transport barrier. The use of B-splines ensures that the resulting spatial profiles are smooth ($C^2$) and physically realistic, avoiding the oscillatory artifacts common in global polynomial approximations.

### 3.3 Latent Dynamics

The latent coordinate $z(t)$ evolves according to a cubic normal form that captures codimension-one bifurcation behaviour:
$$
\frac{dz}{dt}
= \alpha\big(\mu(\mathbf{u})-\mu_{\text{ref}}\big) - \beta z - \gamma z^3,
$$
where $\theta_{\text{latent}} = \{\alpha, \beta, \gamma, \mathbf{w}_{\mu}, b_{\mu}\}$ is the set of **trainable** parameters governing the bifurcation. Here, $\mu(\mathbf{u}) = \mathbf{w}_{\mu}^T \mathbf{u}_{\text{sub}} + b_{\mu}$ is a learned affine combination of a subset of actuators (e.g., power, density). The linear and cubic terms generate bistability and hysteresis in $z$, consistent with the Landau theory of phase transitions.

### 3.4 Identifiability notes: How $z$ captures the transition?
A common concern in latent variable modelling is non-identifiability, where the powerful neural source $\mathbf{S}_{\text{net}}$ overfits the data, rendering the latent variable $z$ redundant or meaningless. In this framework, $z$ is forced to capture the physical L-H transition by two competing constraints:

1.  **The Transport Cost (Physics Loss):**
    In H-mode, the temperature gradient $\nabla T$ at the edge is steep. If the model fails to transition $z$ (i.e., $z$ remains in the "L-mode" state), the diffusivity $\chi_{\text{edge}}(z)$ remains high. The resulting diffusive flux $\Gamma \propto \chi_{\text{high}} \nabla T_{\text{steep}}$ becomes unphysically large. To satisfy the transport equation, the neural source $\mathbf{S}_{\text{net}}$ would have to learn a massive, artificial counter-flux to balance this term.
    Conversely, if $z$ correctly transitions to the "H-mode" state, $\chi_{\text{edge}}(z)$ drops. The flux $\Gamma \propto \chi_{\text{low}} \nabla T_{\text{steep}}$ remains moderate, requiring only a small correction from $\mathbf{S}_{\text{net}}$. Since the training objective minimizes the magnitude of the residual (Physics Loss), the system is energetically driven to switch $z$ to match the transport regime.

2.  **The Complexity Cost (Information Bottleneck):**
    The manifold $\mathbf{T}_{\mathcal{M}}(z)$ captures global, coherent profile changes (pedestal formation) using a single scalar degree of freedom. For $\mathbf{S}_{\text{net}}$ to mimic this transition without $z$, it would need to output a precise, spatially correlated correction across all $N$ grid points. The optimisation landscape favours the low-dimensional explanation ($z$) over the high-dimensional one ($\mathbf{S}_{\text{net}}$) for capturing the dominant structural changes.

## 4. Optimization and Trainable Parameters

The model is trained by minimizing a composite loss function over a dataset of tokamak discharges. The optimization problem is defined with respect to the following disjoint parameter sets:

1.  **Source Parameters ($\theta_{\text{source}}$):** Weights and biases of the neural network $R_{\theta}$. These parameters learn the residual physics and corrections to the transport model.
2.  **Manifold Parameters ($\theta_{\text{manifold}}$):** The coefficient vector $\mathbf{c}_{\text{latent}}$. This learns the shape of the H-mode pedestal structure relative to the baseline.
3.  **Latent Dynamics Parameters ($\theta_{\text{latent}}$):** The scalars $\alpha, \beta, \gamma$ and the affine mapping weights for $\mu(\mathbf{u})$. These learn the timescale and threshold of the L-H transition.

**Fixed Parameters:**
*   **Geometry:** The grid $\rho$, volume elements $V'$, and basis matrix $\Phi$ are fixed.
*   **Transport Coefficients:** The diffusivity profile parameters ($\chi_{\text{core}}, \chi_{\text{edge}}, w_{\text{ped}}$, etc.) are **fixed** during this training phase. The model assumes a static background diffusivity and learns to correct it via $\mathbf{S}_{\text{net}}$.
*   **Reference Profile:** The base coefficients $\mathbf{c}_{\text{base}}$ are fixed to the initial condition of the training set.

### 5. Training Objective

The training objective couples data fidelity with a physics-informed slow-manifold regularisation. The total loss is
$$
\mathcal{L}_{\text{total}}
= \mathcal{L}_{\text{data}} + \lambda_{\text{phy}}\mathcal{L}_{\text{slow}},
$$
where $\lambda_{\text{phy}}>0$ weights the relative contribution of the physics-consistency term.

### 5.1 Data reconstruction

Let $\mathbf{T}_{\text{model}}(t)$ denote the model output and $\mathbf{T}_{\text{obs}}(t)$ the observed temperature, both restricted to the Thomson grid and masked by a binary operator $\mathbf{M}$ that excludes invalid points (NaNs, unphysical values, etc.). The data loss is
$$
\mathcal{L}_{\text{data}}
= \frac{1}{N_t}\sum_{t}\left\|\mathbf{M}\big(\mathbf{T}_{\text{model}}(t) - \mathbf{T}_{\text{obs}}(t)\big)\right\|_2^2,
$$
where $N_t$ is the number of time samples and the masking is derived from `Te_mask` in the data.

### 5.2 Slow-manifold penalty

To encourage the learned manifold to approximate transport equilibria, we define the residual operator
$$
\mathbf{F}(\mathbf{T},\mathbf{u}, z)
= \mathbf{D}\mathbf{T} + \mathbf{b}_{\text{edge}} + \mathbf{S}_{\text{net}}(\boldsymbol{\rho}, \mathbf{T},\mathbf{n}_e,\mathbf{u}, z),
$$
evaluated on manifold states $\mathbf{T}(z)=\mathbf{T}_{\mathcal{M}}(z)$, with $(z,\mathbf{u})$ sampled over relevant ranges. The slow-manifold loss is
$$
\mathcal{L}_{\text{slow}}
= \mathbb{E}_{z,\mathbf{u}}\left[\big\|\mathbf{F}(\mathbf{T}(z),\mathbf{u}, z)\big\|_2^2\right],
$$
which penalises deviations from steady-state diffusion balance on the manifold and constrains $\mathbf{c}_{\text{latent}}$ to span shapes that are approximately stationary under the learned closure and transport operator.

## 6. Numerical Stability and Gradient Propagation

Training neural ODEs embedded with transport physics presents unique challenges in numerical analysis, particularly regarding the stiffness of the diffusion operator and the sensitivity of gradients to geometric factors.

### 6.1 Stiffness and Solver Choice
The semi-discrete transport equation is stiff due to the diffusive term $\mathbf{D}\mathbf{T}$, where the spectral radius scales as $O(N^2)$. Explicit solvers require prohibitively small time steps. We employ L-stable or A-stable implicit solvers (such as Kvaerno3/4/5 or semi-implicit Rosenbrock methods) or stiff-aware explicit methods (Tsit5 with PID control) within the JAX Diffrax framework.

### 6.2 The Challenge of Differentiable Geometry ($V'(\rho)$)
A critical issue arises from the coupling of the transport equation with the magnetic geometry via the specific volume term $V'(\rho)$. In a fully consistent simulation, $V'(\rho)$ evolves with the magnetic equilibrium. However, including $V'(\rho)$ as a time-varying input during training introduces significant numerical instability.

The transport operator $\mathcal{L}_{\text{diff}}[T] = \frac{1}{V'}\nabla \cdot (V' \chi \nabla T)$ depends on the metric factor $V'(\rho)$. In a rigorous MHD formulation, $V'$ is time-dependent, satisfying $\partial_t V' = \dots$. However, experimental reconstructions $\hat{V}'(t)$ contain high-frequency stochastic noise $\eta(t)$.
1.  **Noise Amplification:** Injecting $\hat{V}'(t)$ directly introduces a stochastic source term $\propto \frac{\delta \hat{V}'}{\hat{V}'} T$, which dominates the physical residual $\mathbf{S}_{\text{net}}$.
2.  **Gradient Variance:** The loss gradient $\nabla_{V'} \mathcal{L}$ exhibits high variance due to the divergence structure of the operator. The Hessian spectrum of the loss with respect to geometric parameters possesses eigenvalues significantly larger than those associated with transport coefficients $\chi$, leading to stiff optimization dynamics where geometric noise creates "cliffs" in the loss landscape.

To mitigate this, we impose a stationarity constraint $V'(\rho, t) \approx \langle V'(\rho, t) \rangle_{\text{shot}}$, treating the geometry as a fixed metric background. This regularises the optimisation landscape and forces the neural source $\mathbf{S}_{\text{net}}$ to learn the transport physics rather than overfitting to geometric noise.

### 6.3 Backpropagation vs. Forward Sensitivity
Standard transport codes solve the discretized system $\mathbf{A}(\mathbf{T}^{n+1}) \mathbf{T}^{n+1} = \mathbf{B}(\mathbf{T}^n) \mathbf{T}^n + \mathbf{S}$. Gradient evaluation via finite differences scales as $O(N_{params})$ and suffers from subtractive cancellation.
Our approach utilizes Reverse-Mode Automatic Differentiation (AD) through the integrator. The gradient of the loss $\mathcal{L}$ with respect to parameters $\theta$ is obtained by solving the adjoint ODE:
$$ \frac{d\boldsymbol{\lambda}}{dt} = - \left(\frac{\partial \mathbf{F}}{\partial \mathbf{T}}\right)^T \boldsymbol{\lambda} - \frac{\partial \mathcal{L}}{\partial \mathbf{T}}, \quad \boldsymbol{\lambda}(T) = 0 $$
where $\boldsymbol{\lambda}$ is the adjoint state (costate).
Stability requires the Jacobian $J = \partial \mathbf{F}/\partial \mathbf{T}$ to be well-conditioned and Lipschitz continuous. Discontinuous transitions in $\chi(\rho)$ (e.g., step functions) introduce Dirac delta distributions in $J$, rendering the adjoint system ill-posed. We ensure regularity by enforcing $\chi \in C^1([0,1])$ via sigmoid mollification.

### 6.4 Implementation Details
- **Framework:** JAX + Equinox + Diffrax (`Tsit5`, `rtol=atol=1e-3`, `dt0=5e-4`, `max_steps=1e5`).
- **TORAX Surrogate Architecture:** The JAX implementation is designed as a 1:1 differentiable port of the TORAX core transport solver. The spatial grid, divergence operators, and boundary condition implementations are ported directly from the TORAX codebase. This ensures that in the absence of the neural source $\mathbf{S}_{\text{net}}$, the model reproduces the baseline TORAX transport solution, validating the "physics-consistent" claim.
- **Synthetic Initialization:** $\mathbf{T}(t_0) \leftarrow \mathbf{T}_{\text{syn}}$ if $\mathbf{T}_{\text{obs}}(t_0) \notin \mathcal{D}_{\text{valid}}$. If the initial state is invalid (NaN or unphysical), the ODE solver is initialised with a synthetic parabolic profile ($100\,$eV core, $10\,$eV edge).
- **Observable Clipping:** To prevent outliers and non-physical values in the input data from destabilising the neural closure, we apply strict clipping functions to the observables before they enter the network:
    *   **Density:** Inputs are projected onto compact sets: $n_e \leftarrow \mathcal{C}_{[10^{17}, 10^{21}]}(n_e)$.
    *   **Temperature:** Input $T_e$ to the network is normalised and clipped to avoid extreme gradients during transients.
    *   **Control Signals:** Actuators are normalized via $\hat{\mathbf{u}} = (\mathbf{u} - \mu_{1/2}) / (Q_3 - Q_1)$, where $\mu_{1/2}$ is the median and $Q_i$ are quartiles, minimizing the influence of outliers in the distribution $P(\mathbf{u})$.
- **Derivative clipping:** To ensure stability of the adjoint integration, we enforce $\|\frac{d\mathbf{T}}{dt}\|_\infty \le 10^4$.
- **Ghost nodes:** Boundary conditions and ghost nodes are implemented explicitly inside the JAX step, preserving Neumann symmetry at the core and conservation at the discrete level.

## 7. Data Inventory

| Quantity | Source | Usage |
| --- | --- | --- |
| `Te`, `Te_mask` | Thomson trajectory | Supervised loss + masking |
| `ne` | Thomson density | Input feature (replayed) |
| `P_nbi`, `Ip`, `nebar` | Actuator logs | Define $\mu(\mathbf{u})$ |
| `S_gas`, `S_rec`, `S_nbi` | Fueling / recycling | Provide context to $\mathbf{S}_{\text{net}}$ |
| `rho`, `Vprime` | Equilibrium files | Geometry for diffusion |

Multiple packs (`data/*_torax_training*.npz`) share the solver grid and model weights but retain shot-specific masks and controls. Clean synthetic targets in `simulate_outputs.npz` provide regression baselines.

## 8. HPC Optimization Strategies for Avoiding Local Minima

Training physics-consistent manifold models involves navigating a highly non-convex optimisation landscape generated by the interaction of ODE solvers, PDE discretisations, and neural networks. The system also contains intrinsic multiscale dynamics: fast diffusion, slow L–H transitions, and stiff profile relaxation. These factors interact with the two competing loss terms — the data fidelity term and the slow-manifold physics term — creating multiple local minima and flat regions. Effective optimisation therefore requires careful control of learning rates, gradient magnitudes, architecture capacity, and training schedules.

### 9.1 Advanced optimisers

Several modern gradient-based optimisers provide robustness against non-convexity:

- Adam serves as a strong default because it adapts per-parameter learning rates and performs reliably across a variety of shots and geometries.  
- AdamW decouples weight decay from the gradient update, improving long-training stability and helping avoid overfitting.  
- LAMB is better suited to very large batch sizes or when many shots are used simultaneously in training; it maintains stable updates even when the effective batch size varies.  
- Lion is more memory-efficient and tends to escape sharp minima more effectively than Adam-type methods.

Choosing among these depends on dataset size, batch structure, and memory constraints.

### 9.2 Learning rate schedules

Learning rate schedules are essential for preventing the optimiser from becoming trapped in flat regions or oscillating near sharp minima. Cosine annealing provides a smooth decay curve that reliably improves convergence. Introducing a warmup stage before cosine decay stabilises training when gradients are initially noisy, especially when geometry varies between shots. Exponential decay is simpler and works well for shorter runs. In all cases, a decreasing learning rate helps refine solutions during later iterations without destabilising the solver-network coupling.

### 9.3 Gradient clipping

The stiffness of the ODE system can generate unusually large gradients during backpropagation. Gradient clipping limits the magnitude of updates and prevents both divergence and excessively conservative step sizes. Lower clipping thresholds improve robustness when training on noisy or heterogeneous datasets; larger thresholds may encourage faster convergence on smoother datasets.

### 9.4 Random restarts (ensemble strategy)

Because distinct initialisations can converge to qualitatively different basins, training with multiple random seeds significantly increases the chance of locating high-quality solutions. Each run explores a different region of parameter space, and the resulting models can be compared using validation metrics. In practice, selecting the best model — or even averaging predictions from top performers — yields more reliable and physically consistent results.

### 9.5 Hyperparameter tuning recommendations

When training on many shots, particularly on HPC systems, learning rate reductions, warmup schedules, moderate physics-loss weighting, and careful regularisation are essential. Warmup stabilises the interaction between data and physics terms when geometry or noise varies across shots. Lower learning rates improve long-term stability. Frequent checkpointing guards against divergence and allows recovery from solver failures. The physics weight should not overpower the data term early in training; increasing it gradually yields better manifold structure.

### 9.6 Monitoring and diagnostics

Several diagnostic signals should be tracked to assess training quality:

- The balance between data loss and slow-manifold loss reveals whether the model overfits or drifts away from physics.  
- Gradient norms identify exploding or vanishing gradients early.  
- The latent coordinate trajectory reflects whether the learned dynamics remain within a reasonable range.  
- Manifold profiles evaluated at representative latent values allow visual inspection of whether the learned shapes remain physically plausible.

Monitoring these quantities helps detect instability before it propagates into the ODE solver.

### 9.7 Architecture adjustments

If the optimiser becomes trapped in poor minima, architectural changes can improve expressivity and gradient flow. Increasing the width or depth of the closure network enhances its ability to represent unmodelled physics. Residual connections mitigate vanishing gradients. Enlarging the B-spline basis used for the manifold increases the capacity of the low-dimensional representation. Curriculum learning — training first on simpler or cleaner shots — can guide the model toward better regions of parameter space before introducing more complex data.

### 9.8 Physics loss annealing

The physics-based term in the loss should not dominate early in training, because the model must first learn to reproduce data sufficiently well. A smooth increase of the physics weight shifts the training regime gradually from data-driven learning to physics-constrained refinement. This approach avoids unstable early interactions between the learned manifold, the closure network, and the diffusion operator.

### 9.9 HPC execution strategy

On HPC systems, parallel multi-seed execution is recommended to efficiently explore different initialisations. Large-time-budget runs benefit from reduced learning rates, warmup schedules, and robust optimisers. Automatic selection of the best checkpoint across seeds or training trajectories maximises stability and predictive accuracy while minimising manual tuning effort.

## References

[1] H. Mori. Transport, collective motion, and brownian motion. *Progress of Theoretical Physics*, 33(3):423–455, 1965.

[2] R. Zwanzig. Memory effects in irreversible thermodynamics. *Physical Review*, 124(4):983, 1961.

[3] J. Carr. *Applications of Centre Manifold Theory*. Applied Mathematical Sciences. Springer-Verlag, New York, 1981.

[4] H. Haken. *Synergetics: An Introduction*. Springer-Verlag, Berlin, 3rd edition, 1983.

[5] J. Bradbury et al. JAX: composable transformations of Python+NumPy programs. 2018. URL: http://github.com/google/jax.

[6] P. Kidger. On Neural Differential Equations. PhD thesis, University of Oxford, 2021.

[7] T. H. Osborne, K. H. Burrell, and R. J. Groebner. H-mode pedestal characteristics in DIII-D. *Plasma Physics and Controlled Fusion*, 40(5):845, 1998.

[8] L. D. Landau. On the theory of phase transitions. *Zh. Eksp. Teor. Fiz.*, 7:19–32, 1937.

[9] B. Coppi. Non-classical transport and the "principle of profile consistency". *Comments on Plasma Physics and Controlled Fusion*, 5(6):261–270, 1980.

[10] F. Felici et al. TORAX: A code for simulating tokamak plasma transport. *Nuclear Fusion*, (In preparation/Internal Report), 2023.
