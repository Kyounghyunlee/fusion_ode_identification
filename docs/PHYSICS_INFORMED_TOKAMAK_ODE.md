# Physics-Consistent Transport ROM (Intersection Grid)

**Status:** Active – implemented in `train_ode_physics_manifold_hpc.py`; compatible with transport analysis in TORAX [4].

We no longer use an equilibrium slow manifold. The model is a stiff transport ODE on a non-uniform reduced grid with a learned residual source and a scalar latent $z$ that modulates edge transport.

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

## 1. Overview of the current pipeline

1) Build an observed intersection grid $\boldsymbol{\rho}_{\cap}$: keep columns whose mean Thomson mask coverage is above `data.intersection_rho_threshold` across all training shots. If too few, fall back to top-$k$ coverage columns.

2) Build a non-uniform ROM grid $\boldsymbol{\rho}_{\text{rom}}$: take $\boldsymbol{\rho}_{\cap}$, add Chebyshev-clustered interior nodes in $[0,\min \boldsymbol{\rho}_{\cap}]$, and include $0$ and $1$. This preserves edge resolution and adds coarse interior support without changing the observed set.

3) Regrid per-shot profiles/masks/geometry to $\boldsymbol{\rho}_{\text{rom}}$; carry $\mathcal{I}_{\cap}$ indices for the data term. Edge BC uses a mask-weighted mean of the observed edge column.

4) Evolve a stiff transport ODE on $\boldsymbol{\rho}_{\text{rom}}$ with a neural residual source and a latent $z$ that modulates edge diffusivity via a sigmoid drop.
   - **Safe Core (Toroidal Geometry)**: The core volume element $V'[0]$ is clamped to a small finite value (derived from the first neighbor $V'[1]$) to avoid division by zero while preserving the toroidal geometry ($V' \propto \rho$) elsewhere. This avoids the need for a slab approximation ($V'=1$) and allows for physically consistent transport coefficients.

5) Train with a robust composite loss: pseudo-Huber data term on $\boldsymbol{\rho}_{\cap}$, source magnitude penalty, weak-constraint model-error penalty, optional regime supervision on $z$, and small $z$ regularisation. No manifold or latent subspace penalties remain.

---

## 1.1 Robustness to experimental artifacts and sparse observations

Experimental tokamak data is sparse, noisy, and occasionally corrupted (Thomson dropouts, bad equilibria). The data term is applied only on the observed intersection grid $\boldsymbol{\rho}_{\cap}$ with masks $m_i(t)$ and per-column coverage weights $c_i$. Invalid points (NaNs/outliers) are removed before interpolation; the solver is free to coast through temporal gaps rather than fitting artifacts.

### Why an intersection grid (and no latent subspace)

- A pure mask on the full grid leaves interior nodes weakly constrained and harms identifiability. The common intersection $\boldsymbol{\rho}_{\cap}$ keeps only columns observed across shots, giving a well-conditioned data term.
- We still evolve the full non-uniform ROM grid $\boldsymbol{\rho}_{\text{rom}}$ that contains $\boldsymbol{\rho}_{\cap}$ plus added interior nodes; unobserved nodes are regularised indirectly via diffusion and the source prior—no low-rank projector is used.

### Practical note on incomplete equilibria

- If psi-based $\rho$ or $V'$ are missing, we fall back to linear-in-$R$ $\rho\in[0,1]$; for $V'$, we detect slab-like $V'$ and use $2\rho$ (cylindrical/toroidal approximation). Flags (`rho_fallback_used`, `psi_axis`, `psi_edge`) remain in packs for audit.
- When better equilibria become available, re-run the packer to restore flux-aligned $\rho$ and $V'$; the ROM/intersection construction remains unchanged.

### Data sparsity and observability (all channels)

- Profiles: `Te`, `ne` with masks on the Thomson grid; edge-biased coverage. The data loss uses only $\mathcal{I}_{\cap}$ with mask/coverage weights.
- Controls: `P_nbi`, `Ip`, `nebar`, `S_gas`, `S_rec`, `S_nbi` are dense 1D; z-scored and clipped. Note: `P_rad` is not used in the current training script.
- Optional scalars: `W_tot`, `P_ohm`, `P_tot`, `H98`, `beta_n`, `B_t0`, `q95`, `li` included only when finite and aligned.
- Geometry: equilibrium $\rho$, $V'$ when present; otherwise fallbacks as above.

### Expected vs. effective dimensions (current packs)

- Nominal radial nodes $N\approx65$ on the reference grid; $|\mathcal{I}_{\cap}|$ typically 12–20 edge-biased columns. ROM grid size $N_{\text{rom}} = |\boldsymbol{\rho}_{\text{rom}}|$ adds $m$ interior Chebyshev nodes plus boundaries.
- Thomson time samples ~90–120; controls ~1.7–2.3k, interpolated to the profile grid.

---

## 2. Data, geometry, and grids

### 2.1 Equilibrium and fallbacks
- If equilibrium $\rho, V'$ are present, they are regridded to the common reference grid and then to $\boldsymbol{\rho}_{\text{rom}}$.
- If missing, $\rho$ falls back to linear-in-$R$ on $[0,1]$; for $V'$, we apply a toroidal fallback $V'(\rho)=2\rho$ when slab-like constants are detected, and clamp the core ($V'[0]$) to a small positive floor for stability.

### 2.2 Intersection observed set
Let $c_i$ be the mean mask coverage of column $i$ for shot $s$, and $\bar{c}_i$ the minimum across shots. With threshold $\tau_{\cap}=\texttt{data.intersection\_rho\_threshold}$,
$$
\mathcal{I}_{\cap}=\{ i \mid \bar{c}_i \ge \tau_{\cap}\}.
$$
If $|\mathcal{I}_{\cap}|$ is small, choose the top-$k$ columns by $\bar{c}_i$ (sorted) to ensure a usable observed grid.

### 2.3 ROM grid construction
- Interior nodes: $m=\texttt{data.rom\_n\_interior}$ Chebyshev-like points in $(0,\rho_{\cap,\min})$:
  $$
  x_k = 0.5\,(1-\cos(\pi (k+1)/(m+1))),\quad \rho_k = \rho_{\cap,\min}\, x_k,
  $$
  for $k=0,\dots,m-1$.
- ROM grid:
  $$
  \boldsymbol{\rho}_{\text{rom}} = \text{unique\_sort}\big(\{0\}\cup\{\rho_k\}\cup\boldsymbol{\rho}_{\cap}\cup\{1\}\big).
  $$
- Observed indices $\mathcal{I}_{\cap}$ are mapped into $\boldsymbol{\rho}_{\text{rom}}$ by nearest match (exact grid points after construction).

### 2.4 Regridding profiles and masks
For each shot, interpolate $T_e$, $n_e$, and masks from the reference grid to $\boldsymbol{\rho}_{\text{rom}}$ using only finite points per time slice; masks are interpolated as floats then thresholded $>0.5$. The observed view is a slice on $\mathcal{I}_{\cap}$.

---

## 3. Governing model

We evolve temperature and a scalar latent:
$$
\frac{d \mathbf{T}}{dt} = \mathbf{D}(\boldsymbol{\rho}_{\text{rom}}, V')\,\mathbf{T} + \mathbf{S}_{\text{net}}(\boldsymbol{\rho}_{\text{rom}}, \mathbf{T}, \mathbf{n}_e, \mathbf{u}, z),
$$
$$
\frac{dz}{dt} = \alpha(\mu(\mathbf{u})-\mu_{\text{ref}}) - \beta z - \gamma z^3,
$$
with $\alpha,\beta,\gamma>0$ enforced by softplus. In the current implementation, $\mu(\mathbf{u})$ depends on the first three controls (`P_nbi`, `Ip`, `nebar`). The source network is a node-wise MLP on $(\rho, T, n_e, \mathbf{u}, z)$ with tanh activations and zero-initialised output layer for stability. Edge diffusivity drops with $z$ via $\chi_{\text{edge}}(z) = \chi_{\text{edge,base}} - \chi_{\text{edge,drop}}\,\sigma(k z)$; $\chi$ is blended across $\rho$ by a sigmoid pedestal [1].

The evolved state in the solver is the interior temperature nodes (scaled) plus $z$; the edge value $T_{\text{edge}}$ is fixed from masked data and appended during reconstructions.

### 3.1 Continuous framing

The continuous model is diffusion with a latent-modulated diffusivity and a neural residual source:
$$
\frac{\partial T_e}{\partial t} = \frac{1}{V'(\rho)}\frac{\partial}{\partial \rho}\left(V'(\rho)\,\chi(\rho,z)\,\frac{\partial T_e}{\partial \rho}\right) + \mathcal{S}_{\text{net}}(\rho, T_e, n_e, \mathbf{u}, z),
$$
where $\chi$ has a sigmoid pedestal and edge drop $\chi_{\text{edge}}(z) = \chi_{\text{edge,base}} - \chi_{\text{edge,drop}}\,\sigma(k z)$. The latent ODE is $\dot z = f_z(\mathbf{u}, z)$ as above. This is discretised to the non-uniform FVM operator $\mathbf{D}$ plus the Nemytskii source $\mathbf{S}_{\text{net}}$.

### 3.2 Nemytskii source operator

The source acts pointwise:
$$
(\mathbf{S}_{\text{net}})_i = R_\theta(\rho_i, T_i, n_{e,i}, \mathbf{u}, z),
$$
with a small MLP $R_\theta$ (tanh, zero init on output) to ensure the residual starts near zero and learns spatially varying corrections.

### 3.3 Semi-discrete hybrid ODE

Combining diffusion and source yields the ODE used in training and inference:
$$
\frac{d\mathbf{T}}{dt} = \mathbf{D}\mathbf{T} + \mathbf{b}_{\text{edge}} + \mathbf{S}_{\text{net}}(\boldsymbol{\rho}_{\text{rom}}, \mathbf{T}, \mathbf{n}_e, \mathbf{u}, z),
$$
with fixed boundary contribution $\mathbf{b}_{\text{edge}}$ derived from the masked $T_{\text{edge}}$. No equilibrium slow manifold term is present; the latent influences only $\chi$ and the source. The solver integrates up to a dynamic end time (last valid profile) and evaluates on the full padded time grid using a time mask.

---

## 4. Spatial discretisation (non-uniform FVM)

- Faces: $\rho_{i+1/2} = (\rho_i + \rho_{i+1})/2$.
- Edge spacings: $\Delta\rho_{i+1/2} = \rho_{i+1} - \rho_i$; gradients $\partial T/\partial\rho \approx (T_{i+1}-T_i)/\Delta\rho_{i+1/2}$.
- Diffusivity on faces: arithmetic mean $\chi_{i+1/2} = 0.5(\chi_i+\chi_{i+1})$.
- Flux: $\Gamma_{i+1/2} = -\chi_{i+1/2}\,(T_{i+1}-T_i)/\Delta\rho_{i+1/2}$.
- Volume factors: $V'_{i+1/2}=0.5(V'_i+V'_{i+1})$; cell width $\Delta\rho_i = \rho_{i+1/2}-\rho_{i-1/2}$ with Neumann at the axis ($\rho_{-1/2}=\rho_0$) and Dirichlet at the edge handled via the fixed boundary value.
- Divergence (interior nodes, with cell-averaged volume element):
$$
(\mathbf{D}\mathbf{T})_i = \frac{V'_{i+1/2}\Gamma_{i+1/2}-V'_{i-1/2}\Gamma_{i-1/2}}{\bar V'_i\,\Delta\rho_i},\quad i=0,\dots,N_{\text{rom}}-2,
$$
where $\bar V'_i = 0.5\,(V'_i + V'_{i+1})$.
- Boundary conditions: zero-flux at the axis (left flux set to zero); fixed $T_{\text{edge}}$ at $\rho=1$ supplied from masked data.

---

## 5. Loss terms (no manifold)

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

## 6. Data handling and preprocessing

- Masks: drop NaNs; masks follow Thomson grid, then regridded to $\boldsymbol{\rho}_{\text{rom}}$.
- Edge BC: masked mean on the edge column; fallback to 50 eV if undefined. Boundary values are linearly interpolated over time gaps on the shot’s time grid.
- Initial condition: first valid profile on $\boldsymbol{\rho}_{\text{rom}}$; fallback synthetic $T(\rho)=100(1-\rho^2)+10$ when missing.
- Clipping: $n_e\in[10^{17},10^{21}]$ before scaling; $T_e$ and controls are normalised/clipped to avoid outliers.

---

## 7. Configuration knobs

- `data.intersection_rho_threshold`: coverage threshold for the observed intersection.
- `data.rom_n_interior`, `data.rom_interior_strategy`: interior node count/placement.
- `data.min_rho_coverage`: optional global prune before intersection (rarely needed now).
- `loss.huber_delta`, `loss.model_error_delta`, `training.lambda_src`, `training.lambda_w`, `training.lambda_z`.
- `model.latent_gain`, `model.chi_*`: latent-edge coupling and diffusivity envelope.

---

## 8. Logging and diagnostics

- Report $|\boldsymbol{\rho}_{\text{rom}}|$, $|\boldsymbol{\rho}_{\cap}|$, their min/max, and the actual values at startup.
- Per-shot $\text{mask\_mean}_{\cap}$ to verify intersection coverage.
- Recommended debug mode: integrate one shot, dump $T_{\text{model}}(t,\boldsymbol{\rho}_{\cap})$ vs. data and $z(t)$.

---

## 9. Numerical stability

- Solver: `Kvaerno5` with PID control (`rtol=atol=1e-3`, `dt0=1e-4`, `max_steps=2e5`) via Diffrax [3], 64-bit JAX enabled [2]. The solver evaluates on the full (padded) time grid and uses a dynamic end time from the last valid profile.
- Derivative clipping on $dT/dt$ to $10^4$.
- Non-uniform FVM widths to avoid negative cells on stretched grids; Neumann at the axis enforced via zero left flux.
- **Safe Core**: The core volume element $V'[0]$ is clamped to avoid division by zero, preserving toroidal geometry elsewhere.

---

## 10. Data inventory (unchanged sources)

| Quantity | Source | Usage |
| --- | --- | --- |
| `Te`, `Te_mask` | Thomson trajectory | Supervised loss + masking |
| `ne` | Thomson density | Input feature (replayed) |
| `P_nbi`, `Ip`, `nebar`, `S_gas`, `S_rec`, `S_nbi` | Actuator logs | Inputs to $\mu(\mathbf{u})$ (first three) and $\mathbf{S}_{\text{net}}$ |
| `rho`, `Vprime` | Equilibrium files (fallbacks if missing) | Geometry for diffusion |

Optional scalars (`W_tot`, `P_ohm`, `P_tot`, `H98`, `beta_n`, `B_t0`, `q95`, `li`, `P_rad`) are ingested only when finite and aligned; they are not used by default in the current training script.

---

## 11. HPC optimisation quick notes

- Use warmup + cosine decay, gradient clipping, and multi-seed restarts.
- Keep physics weights modest early; tune `lambda_w` and `lambda_src` to balance data fit vs. regularity.
- Monitor loss components and $z(t)$ to detect solver or identifiability issues.

---

## 12. HPC optimisation strategies (expanded)

- **Optimisers:** Adam/AdamW as defaults; LAMB for very large batches; Lion when memory is tight.
- **Schedules:** warmup $\rightarrow$ cosine decay or exponential; keep physics weights modest early, raise $\lambda_w$ and $\lambda_{\text{src}}$ gradually.
- **Gradient control:** clip grads and keep derivative clipping at $10^4$ on $d\mathbf{T}/dt$ for solver stability.
- **Restarts:** run multi-seed jobs and keep best checkpoints; divergence often resolves by restarting with a lower LR or slightly smaller physics weights.
- **Diagnostics:** track $|\boldsymbol{\rho}_{\text{rom}}|$, $|\boldsymbol{\rho}_{\cap}|$, loss components, $z(t)$ traces, and one-shot overlays $T_{\text{model}}(t,\boldsymbol{\rho}_{\cap})$ vs. data.
- **Architecture levers:** widen the source MLP or adjust pedestal sharpness only after stabilising training; keep the source output layer zero-initialised to avoid large initial residuals.

---

---

## 13. Training Mechanics (Batches and Restarts)

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
  	heta^{(r)}_\ast = \operatorname*{argmin}_{\theta} J^{(r)}(\theta),\quad r=1,\dots,R,
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

## 14. ODE Solver Choice and Rationale

We integrate a stiff semi-discrete transport ODE:
$$
\frac{d\mathbf{y}}{dt} = f(t, \mathbf{y}; \theta), \quad \mathbf{y} = [\mathbf{T}_{\text{int}}, z].
$$
Stiffness arises from diffusion on a non-uniform grid: the discrete operator has large negative eigenvalues with magnitude scaling like $\|\lambda\| \sim \chi / \Delta\rho^2$, creating fast transients alongside slower profile evolution.

- **Solver:** We use `Kvaerno5`, a stiff, high-order Runge–Kutta method with adaptive step size.
- **Adaptivity (PID control):** The controller adjusts $\Delta t$ to keep the local truncation error near a target. Conceptually,
  $$
  \Delta t_{k+1} = \Delta t_k\,\left(\frac{\mathrm{tol}}{\mathrm{err}_k}\right)^{1/(p+1)}\cdot \text{(PI corrections)},
  $$
  where $p$ is the method order and $\mathrm{err}_k$ is the estimated error at step $k$.
- **Why not an explicit method?** For stiff problems, explicit methods would require prohibitively small $\Delta t$ for stability ($\Delta t \lesssim 2/\|\lambda\|$), making training impractically slow. Stiff solvers remain stable at larger $\Delta t$, reducing total steps.
- **Evaluation grid:** We save solutions at the Thomson time stamps via `SaveAt(ts=...)`; the final integration time is the last valid profile (dynamic end time), and padded time points are masked in the loss.
- **Safety limits:** We set `rtol=atol=1e-3`, initial `dt0=1e-4`, and cap the number of steps (`max_steps=2\times10^5`) to avoid runaway integrations in degenerate cases; derivatives are softly clipped for additional stability.

Overall, `Kvaerno5` balances accuracy and robustness for our stiff transport dynamics, enabling training to proceed without the extreme step sizes an explicit solver would require.

### What is a stiff ODE (mathematically) and why ours is stiff
- A linear ODE $\dot{\mathbf{y}} = A\mathbf{y}$ is stiff when its eigenvalues have widely separated scales: the stiffness ratio $R = \max_i |\lambda_i| / \min_{i: \Re \lambda_i < 0} |\lambda_i|$ satisfies $R \gg 1$. This causes very fast decays (large $|\lambda|$) coexisting with slow modes, forcing explicit methods to take $\Delta t$ constrained by the fastest mode.
- Our semi-discrete diffusion operator on a non-uniform grid has eigenvalues scaling like $\lambda \sim -\chi / \Delta\rho^2$. Where $\Delta\rho$ is small (near clustered interior/edge nodes), $|\lambda|$ is large, so $R$ becomes large. The sigmoid pedestal [1] and strong edge drop in $\chi(\rho,z)$ further amplify gradients, increasing stiffness.
- Implicit/stiffly-stable Runge–Kutta methods (like `Kvaerno5` via Diffrax [3]) permit larger $\Delta t$ while maintaining stability, making them suitable for our transport ROM.

## References

[1] T. H. Osborne, K. H. Burrell, and R. J. Groebner. H-mode pedestal characteristics in DIII-D. *Plasma Physics and Controlled Fusion*, 40(5):845, 1998.

[2] J. Bradbury et al. JAX: composable transformations of Python+NumPy programs. 2018. URL: http://github.com/google/jax.

[3] P. Kidger. On Neural Differential Equations. PhD thesis, University of Oxford, 2021.

[4] F. Felici et al. TORAX: A code for simulating tokamak plasma transport. *Nuclear Fusion*, (In preparation/Internal Report), 2023.

[5] B. Coppi. Non-classical transport and the "principle of profile consistency". *Comments on Plasma Physics and Controlled Fusion*, 5(6):261–270, 1980.

[6] H. Haken. *Synergetics: An Introduction*. Springer-Verlag, Berlin, 3rd edition, 1983.

[7] H. Mori. Transport, collective motion, and brownian motion. *Progress of Theoretical Physics*, 33(3):423–455, 1965.

[8] R. Zwanzig. Memory effects in irreversible thermodynamics. *Physical Review*, 124(4):983, 1961.

[9] L. D. Landau. On the theory of phase transitions. *Zh. Eksp. Teor. Fiz.*, 7:19–32, 1937.

[10] J. Carr. *Applications of Centre Manifold Theory*. Applied Mathematical Sciences. Springer-Verlag, New York, 1981.
