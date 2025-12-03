# Physics-Consistent Manifold Learning

**Status:** Active – implemented in `train_ode_physics_manifold.py`

## 1. Motivation

Conventional manifold learning minimizes a purely data-driven loss such as $\|\mathbf{T}_{model}-\mathbf{T}_{data}\|^2$. The resulting manifold may interpolate the measurements yet fails to respect the diffusion physics that governs the plasma. In practice the learned states can sit in regions where the conductive flux is enormous, forcing the neural source term to counteract the deterministic physics instead of capturing the genuinely missing phenomena.

**Physics-consistent goal.** We learn an equilibrium manifold $\mathcal{M}$ that resides inside the slow (approximately null) space of the transport operator. Profiles restricted to $\mathcal{M}$ satisfy $\tfrac{d\mathbf{T}}{dt}\approx 0$ under the diffusion dynamics, so any residual neural source focuses on true exogenous effects rather than canceling large unbalanced fluxes.

## 2. Governing Transport Model

### 2.1 Radial transport PDE

The electron temperature $T_e(\rho,t)$ evolves through a one-dimensional flux-surface–averaged diffusion equation
$$
\frac{\partial T_e}{\partial t} = \frac{1}{V'(\rho)}\frac{\partial}{\partial \rho}\left(V'(\rho)\,\chi(\rho)\,\frac{\partial T_e}{\partial \rho}\right) + S_{\text{net}}(\rho,T_e,n_e,u),
$$
where
- $\rho\in[0,1]$ is the normalized poloidal flux coordinate
  $$
  \rho(\mathbf{x}) = \sqrt{\frac{\psi(R,Z)-\psi_{\text{axis}}}{\psi_{\text{sep}}-\psi_{\text{axis}}}};
  $$
- $V'(\rho)=\tfrac{dV}{d\rho}$ is the differential volume enclosed by the flux surface;
- $\chi(\rho)$ is the effective thermal diffusivity;
- $S_{\text{net}}$ aggregates non-diffusive physics (auxiliary heating, radiation, convection, model mismatch).

### 2.2 Geometry and transport coefficients

- **Flux surfaces.** Constant-$\psi$ toroidal shells given by the equilibrium reconstruction. Mapping the plasma to $\rho$ collapses the 3D structure into a tractable 1D problem while preserving flux-surface averages.
- **Differential volume $V'(\rho)$.** Acts as the Jacobian of the coordinate transform and enforces energy conservation in the finite-volume scheme.
- **Thermal diffusivity $\chi(\rho)$.** Modeled as a smooth pedestal that captures L/H transport barriers:
  $$
  \chi(\rho) = \chi_{\text{core}} + \big(\chi_{\text{edge}}(z)-\chi_{\text{core}}\big) \sigma\left(\frac{\rho-\rho_{\text{ped}}}{w_{\text{ped}}}\right),
  $$
  with parameters $\chi_{\text{core}}=0.6$, $\rho_{\text{ped}}=0.85$, $w_{\text{ped}}=0.08$, and a latent-controlled edge diffusivity
  $$ \chi_{\text{edge}}(z) = \chi_{\text{edge,base}} - \chi_{\text{edge,drop}}\,\sigma(k z), \qquad \chi_{\text{edge,base}}=2.0,\;\chi_{\text{edge,drop}}=1.0. $$
  The shared logistic gate $\sigma$ implements a differentiable L/H transition inspired by Groebner-style pedestal fits and Landau phase-transition theory.

### 2.3 Spatial discretization

The radial interval is discretized onto $N=65$ nodes $\rho_i$. Discrete heat fluxes and their divergence are
$$
\Gamma_{i+1/2} = -\chi_{i+1/2}\frac{T_{i+1}-T_i}{\Delta\rho_{i+1/2}}, \qquad
(\mathcal{L}\mathbf{T})_i = \frac{1}{V'_i\Delta\rho_i}\left(V'_{i+1/2}\Gamma_{i+1/2}-V'_{i-1/2}\Gamma_{i-1/2}\right).
$$
Boundary conditions:
- **Axis symmetry ($\rho=0$):** Neumann enforced by a ghost node so that $\Gamma_{-1/2}=0$ and the first row of $\mathbf{D}$ becomes $(-\alpha,\alpha,0,\dots)$.
- **Edge Dirichlet ($\rho=1$):** $T_N=T_{\text{edge}}$ from the pack; its effect is isolated into a boundary source vector $\mathbf{b}_{\text{edge}}$.

Overall semi-discrete system:
$$
\frac{d\mathbf{T}}{dt} = \mathbf{D}\mathbf{T} + \mathbf{b}_{\text{edge}}(T_{\text{edge}}) + \mathbf{S}_{\text{net}}.
$$

## 3. Hybrid Source Decomposition

The net source on the grid is split into
$$
\mathbf{S}_{\text{net}} = S_{NN}(\boldsymbol{\rho},\mathbf{T},\mathbf{n}_e,\mathbf{u}) + \frac{\partial \mathbf{T}_{\text{manifold}}}{\partial z}\frac{dz}{dt}.
$$
- $S_{NN}$ (MLP) captures unmodeled heating, radiative losses, and geometry errors from $(\mathbf{T},\mathbf{n}_e,\mathbf{u})$.
- $\partial \mathbf{T}_{\text{manifold}}/\partial z\,dz/dt$ explicitly represents motion along the slow manifold instead of forcing $S_{NN}$ to mimic structural profile changes.

## 4. Equilibrium Manifold Construction

### 4.1 Parameterization

$$
\mathcal{M} = \{\mathbf{T}(z) = \Phi(\boldsymbol{\rho})\big(\mathbf{c}_{\text{base}} + \sigma(k z)\mathbf{c}_{\text{latent}}\big) \mid z\in\mathbb{R}\},
$$
where $\Phi$ is a fixed B-spline basis, $\mathbf{c}_{\text{base}}$ projects the reference initial condition, and the trainable $\mathbf{c}_{\text{latent}}$ defines the H-mode shape. The shared sigmoid keeps the interpolation bounded and differentiable.

### 4.2 Latent dynamics

A cubic normal form captures the L/H bifurcation:
$$
\frac{dz}{dt} = \alpha\big(\mu(\mathbf{u})-\mu_{\text{ref}}\big) - \beta z - \gamma z^3,
$$
with $\mu(\mathbf{u})$ a learned linear combination of $(P_{\text{NBI}}, I_p, \bar{n}_e)$. The cubic term introduces the observed hysteresis loop.

### 4.3 Justification

- **Profile stiffness:** Experiments (Coppi 1980) show $T_e(\rho)$ rapidly collapses to a canonical shape; the dominant degree of freedom is pedestal strength captured by $z$.
- **Center-manifold theory:** Near bifurcation only one slow eigenmode remains; fast modes are slaved to it, validating a 1D manifold description.
- **Interpretability:** $z(t)$ is bounded by the sigmoid gate and directly indicates regime transitions.

## 5. State, Observations, and Chain Rule

State vector: $(\mathbf{T},\mathbf{n}_e,z)$; controls: $(P_{\text{NBI}}, I_p, \bar{n}_e, S_{\text{gas}}, S_{\text{rec}}, S_{\text{nbi}})$.

Temperature dynamics:
$$
\frac{d\mathbf{T}}{dt} = \mathcal{L}(\mathbf{T}) + S_{NN}(\boldsymbol{\rho},\mathbf{T},\mathbf{n}_e,\mathbf{u}) + \Phi(\boldsymbol{\rho})\mathbf{c}_{\text{latent}}\sigma'(kz)k\frac{dz}{dt}.
$$
Density is replayed from measured time series ($d\mathbf{n}_e/dt=0$ after interpolation). The latent ODE follows Section 4.2.

Decomposing $T_e = T_{\text{manifold}} + \tilde{T}_e$ ensures diffusion acts on the full profile while the chain-rule term explicitly accounts for structural motion along $\mathcal{M}$.

## 6. Training Objective

Total loss:
$$ \mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phy}}\mathcal{L}_{\text{slow}}. $$


### 6.1 Data reconstruction
$$
\mathcal{L}_{\text{data}} = \frac{1}{N_t}\sum_t \big\|\mathbf{M}\big(\mathbf{T}_{model}(t)-\mathbf{T}_{obs}(t)\big)\big\|_2^2,
$$
where $\mathbf{M}$ masks invalid Thomson points using `Te_mask`.

### 6.2 Slow-manifold penalty

Define
$$
\mathbf{F}(\mathbf{T},\mathbf{u}) = \mathbf{D}\mathbf{T} + \mathbf{b}_{\text{edge}} + S_{NN}(\mathbf{T},\mathbf{u}).
$$
Sampling $(z,\mathbf{u})$ across the manifold yields
$$
\mathcal{L}_{\text{slow}} = \mathbb{E}_{z,\mathbf{u}} \left[ \|\mathbf{F}(\mathbf{T}(z),\mathbf{u})\|_2^2 \right],
$$
forcing $\mathbf{c}_{\text{latent}}$ to span shapes that satisfy the steady-state diffusion balance.

## 7. Numerical Stability and Implementation

- **Framework:** JAX + Equinox + Diffrax (`Tsit5`, `rtol=atol=1e-3`, `dt0=5e-4`, `max_steps=1e5`).
- **Fixed geometry:** $V'$, $\Delta\rho$, and $\chi(\rho)$ are static; only $S_{NN}$, $\mathbf{c}_{\text{latent}}$, and $(\alpha,\beta,\gamma)$ learn.
- **Data gating:** Bad temperature slices (NaNs or <10 eV) are replaced by smooth parabolic profiles; masks keep them out of the loss.
- **Derivative clipping:** $dT/dt$ limited to $\pm 10^4$ eV/s to keep the stiff solver stable.
- **Ghost nodes:** Implemented explicitly inside the JAX step to maintain Neumann symmetry and conservation.

## 8. Data Inventory

| Quantity | Source | Usage |
| --- | --- | --- |
| `Te`, `Te_mask` | Thomson trajectory | Supervised loss + masking |
| `ne` | Thomson density | Input feature (replayed) |
| `P_nbi`, `Ip`, `nebar` | Actuator logs | Define $\mu(\mathbf{u})$ |
| `S_gas`, `S_rec`, `S_nbi` | Fueling / recycling | Provide context to $S_{NN}$ |
| `rho`, `Vprime` | Equilibrium files | Geometry for diffusion |

Multiple packs (`data/*_torax_training*.npz`) share the solver grid and model weights but retain shot-specific masks and controls. Clean synthetic targets in `simulate_outputs.npz` provide regression baselines.
