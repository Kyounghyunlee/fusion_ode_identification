# Physics-Consistent Manifold Learning

**Status:** Active – implemented in `train_ode_physics_manifold.py`

## 1. Motivation

**Physics-consistent goal.** We model the system state as a superposition 
 

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

## 9. HPC Optimization Strategies for Avoiding Local Minima

Training physics-consistent manifolds is challenging due to:
1. **Non-convex loss landscape** from the coupled ODE-PDE system
2. **Multiple time scales** (fast diffusion vs. slow L/H dynamics)
3. **Interacting loss terms** ($\mathcal{L}_{\text{data}}$ vs. $\mathcal{L}_{\text{slow}}$)

### 9.1 Advanced Optimizers

**Recommendations:**

- **Adam** (default): Good baseline with adaptive learning rates
  - `--lr 1e-3` typically works well
  - Consider `--weight_decay 1e-5` for regularization

- **AdamW**: Better weight decay handling than Adam
  - Decouples weight decay from gradient updates
  - More stable for long training runs
  - `--optimizer adamw --weight_decay 1e-4`

- **LAMB** (Layer-wise Adaptive Moments optimizer for Batch training)
  - Better for very large datasets with many shots
  - Adapts learning rate per-layer
  - More robust to batch size changes
  - `--optimizer lamb --lr 1e-3`

- **Lion** (EvoLved Sign Momentum)
  - More memory efficient than Adam
  - Often escapes sharp minima better
  - Try lower learning rates: `--optimizer lion --lr 3e-4`

### 9.2 Learning Rate Schedules

**Critical for avoiding plateaus:**

- **Cosine Annealing**: Gradually reduces LR with periodic "warm restarts"
  ```bash
  --lr_schedule cosine --lr 1e-3 --lr_min 1e-6 --steps 5000
  ```
  - Smooth decay helps fine-tuning
  - Prevents oscillations at convergence

- **Warmup + Cosine**: Best for large-scale training
  ```bash
  --lr_schedule warmup_cosine --warmup_steps 200 --lr 1e-3 --lr_min 1e-6
  ```
  - Warmup stabilizes early training (critical when $\mathcal{L}_{\text{slow}}$ is noisy)
  - Cosine decay refines solution

- **Exponential Decay**: Simple but effective
  ```bash
  --lr_schedule exponential --lr 1e-3 --lr_min 1e-6 --steps 3000
  ```

### 9.3 Gradient Clipping

Stiff ODE solvers can produce explosive gradients:
- **Default**: `--grad_clip 1.0` (gradient norm clipping)
- **For unstable datasets**: Try `--grad_clip 0.5`
- **For smooth convergence**: Increase to `--grad_clip 2.0`

### 9.4 Random Restarts (Ensemble Strategy)

Run multiple training sessions with different seeds:
```bash
for seed in {0..4}; do
  python -m scripts.train_ode_physics_manifold \
    --seed $seed \
    --steps 3000 \
    --out outputs/physics_manifold_seed${seed} \
    data/*_torax_training.npz
done
```
- Select best model based on validation loss
- Average predictions from top-3 models for robustness
- Different initializations explore different basins

### 9.5 Hyperparameter Tuning Recommendations

**For HPC with many shots (10+ packs):**

```bash
python -m scripts.train_ode_physics_manifold \
  --steps 5000 \
  --optimizer adamw \
  --lr 5e-4 \
  --lr_schedule warmup_cosine \
  --warmup_steps 300 \
  --lr_min 1e-7 \
  --weight_decay 1e-4 \
  --grad_clip 1.0 \
  --lambda_phy 0.01 \
  --checkpoint_every 250 \
  --early_stop_patience 500 \
  --num_restarts 3 \
  data/*_torax_training.npz
```

**Key principles:**
- Start with moderate `lambda_phy` (0.01-0.1), increase if profiles drift from physics
- Use warmup when adding new shots (geometry variations need adaptation)
- Lower learning rates (1e-4 to 5e-4) for stable long runs
- Checkpoint frequently to recover from divergence

### 9.6 Monitoring and Diagnostics

**Track these metrics** (add to training loop):
1. **Component losses**: Separate $\mathcal{L}_{\text{data}}$ and $\mathcal{L}_{\text{slow}}$
2. **Gradient norms**: Detect vanishing/exploding gradients
3. **Latent trajectory**: Monitor $z(t)$ range (should stay ~[-3, 3])
4. **Manifold shape**: Visualize $\mathbf{T}(z)$ for $z\in\{-2,0,2\}$

### 9.7 Architecture Adjustments

If still stuck in local minima after optimization tuning:
- **Increase MLP capacity**: Change `SourceNN` from 64→128 nodes or depth 3→4
- **Add residual connections**: Help gradient flow through deep networks
- **Increase manifold basis**: More B-spline centers (currently 5, try 7-9)
- **Curriculum learning**: Train on easy shots first, gradually add complex ones

### 9.8 Physics Loss Annealing

Gradually increase $\lambda_{\text{phy}}$ during training:
```python
# In training loop
lambda_current = lambda_phy * min(1.0, step / warmup_steps)
```
- Start data-fitting focused, then enforce physics constraints
- Prevents early training collapse from conflicting gradients

### 9.9 Example HPC SLURM Script

```bash
#!/bin/bash
#SBATCH --job-name=fusion_manifold
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G

module load cuda/12.0
source venv/bin/activate

# Multi-seed ensemble
for seed in {0..4}; do
  python -m scripts.train_ode_physics_manifold \
    --seed $seed \
    --steps 10000 \
    --optimizer adamw \
    --lr 5e-4 \
    --lr_schedule warmup_cosine \
    --warmup_steps 500 \
    --lambda_phy 0.02 \
    --out outputs/manifold_s${seed} \
    data/*_torax_training.npz &
done
wait

# Select best model
python -m scripts.evaluate_models outputs/manifold_s*
```

**Recommendations summary:**
- **Short test runs**: Adam + constant LR to verify setup
- **Production HPC**: AdamW + warmup_cosine + multiple seeds
- **Large datasets (>20 shots)**: LAMB optimizer with batch processing
- **Stuck in plateau**: Lower LR by 10x or switch to Lion optimizer
