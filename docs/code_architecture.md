# Code Architecture and HPC Overview

## Big Picture
- Goal: learn a reduced plasma transport dynamics model (1D heat equation + latent source) from many discharges ("shots").
- Pipeline: download → pack build (NetCDF → NPZ) → train JAX model → evaluate.
- Design themes: physics-structured model, JAX for autodiff + accelerator execution, data-parallel training for throughput.

## Core Packages and What They Do (plain language)
- **JAX**: NumPy-like arrays that can be *compiled* for GPU/TPU. It gives:
	- `jit`: Just-In-Time compile a Python function into a fast GPU kernel. First call traces, later calls are fast.
	- `vmap`: Vectorize a function over a batch dimension (no Python loops). Think “apply over many shots/rows in one go.”
	- `pmap`: Parallel map across multiple devices (GPUs). Each GPU gets a slice of the batch; gradients are averaged.
	- Autodiff: reverse-mode differentiation (backprop) through regular code and even through the ODE solver (via diffrax).
- **equinox (eqx)**: Minimal neural-net module system for JAX. Lets us define `HybridField`/`SourceNN`, split trainable parameters from static fields, and save/load checkpoints.
- **diffrax**: JAX-native ODE solvers. Runs inside JIT, supports adaptive timesteps and backprop through the solve.
- **optax**: Optimizers (AdamW), gradient clipping, learning-rate schedules, EMA of weights.
- **NumPy/xarray/zarr/fsspec**: Load and reshape data; zarr+xarray read S3-stored diagnostics; NumPy writes packs.
- **Matplotlib**: Make evaluation plots.

## Model Architecture (training script)
- `HybridField`: PDE-inspired RHS combining diffusion (chi profile), finite-volume divergence, and a learned source NN. Latent scalar `z` evolves via a low-order ODE (`LatentDynamics`).
- `SourceNN`: small MLP taking `(rho, Te, ne, controls, z)`; final layer zero-initialized for stability, scaled by `source_scale`.
- `ShotBundle`: batched, padded shot data (times, profiles, masks, controls, boundary Te, geometry); built in `train_ode_physics_manifold_hpc.load_data`.
- Loss: weighted Huber data term, source magnitude regularizer, weak-constraint model error, regime supervision on `z`, latent regularization.

## Parallelism and Performance (why GPU, what the terms mean)
- **Why GPU helps now**: We have many shots and many rho points. We batch them so the GPU crunches large matrices instead of tiny loops. More arithmetic per batch → GPU wins.
- **Batching**: We stack shots into uniform shapes (padding). One forward/backward pass handles a whole batch.
- **`vmap` (vectorize)**: Replaces Python loops over batch/shot with a single fused kernel. Used inside loss to evaluate multiple shots on one device.
- **`pmap` (parallel map)**: Splits the batch across GPUs. Each GPU runs the same compiled code on its chunk. Gradients are averaged (all-reduce) so the update is as if it saw the whole batch.
- **`jit` (just-in-time compile)**: First call traces the function (slow); the compiled version then runs fast. We JIT the training step so the ODE solve, loss, and optimizer are fused.
- **Adaptive ODE on GPU**: diffrax solvers (Kvaerno5 + PID controller) run inside JIT. Control flow is staged; math kernels still land on GPU. NaN guards keep the solver stable.
- **EMA (exponential moving average)**: A smoothed copy of weights that often generalizes better. Optax updates it cheaply each step.
- **Gradient clipping**: Limit global gradient norm to avoid exploding updates.
- **L-BFGS finetune**: After AdamW, an optional quasi-Newton refinement on a small batch to squeeze a bit more accuracy.

## Autodiff and Backprop (what actually happens)
- We run the model forward: solve the ODE for each shot → predicted temperatures + latent `z`.
- Loss is computed (Huber data loss, model-error penalty, source/latent regs, regime supervision).
- JAX records the operations; reverse-mode autodiff replays them backward to get gradients w.r.t. every parameter (MLP weights, chi params, latent params).
- diffrax provides backprop through the ODE solve, so sensitivities account for solver steps.
- Equinox “filters” params: only arrays marked as trainable get gradients/updates; static config stays untouched.
- Gradient clipping caps the norm before the optimizer step.

## What “jit/compile” means in practice
- JIT = “trace once, then run fast.” On the first call, JAX traces the Python/JAX ops, hands the graph to XLA, and XLA emits GPU code. Subsequent calls reuse that executable.
- Benefits: fused kernels (fewer memory trips), GPU-friendly tiling, elimination of Python overhead in the training loop.

## How this becomes big linear algebra on GPU
Let $B$ be batch size (shots), $N_\rho$ radial points.

1) **State per batch:** $T \in \mathbb{R}^{B \times N_\rho}$, $z \in \mathbb{R}^B$.
2) **Finite-volume flux/divergence** (schematic):
$$
	ext{grad}_T = D T,\quad \text{flux} = -\tfrac12 (P\chi) \odot \text{grad}_T,\quad \text{div} = A\,\text{flux},
$$
with $D$ a banded difference matrix, $P$ a face-averaging matrix, $A$ an accumulation matrix. Batched over $B$, these are matrix–matrix ops that GPUs accelerate.
3) **Source MLP:** For each $(\rho, t)$ row, $S_\theta([\rho, T, n_e, u, z])$ is an MLP layer $Y = XW + b$ (GEMM). Batched over $(B \times N_\rho)$ rows → large GEMM on GPU.
4) **Loss/reductions:** Weighted Huber, norms, and clipping are vectorized reductions over $(B, N_\rho)$.
5) **`vmap` and batching:** Replace Python loops over shots with one batched kernel. Shapes become $(B, ...)$, turning many small vector ops into big matrix ops.
6) **`pmap` across GPUs:** Split batch across devices; each GPU runs the same compiled program on its slice. Gradients are averaged via all-reduce.
7) **Backprop:** Autodiff builds backward passes that mirror the same linear-algebra ops (Jacobian–vector products), so gradients are also batched GEMMs/reductions.

## Why GPU wins (even for ODEs)
- We integrate many shots × many rho points at once → high arithmetic intensity.
- XLA fuses flux/divergence + MLP + loss into a few kernels → less memory traffic.
- `vmap` + `pmap` keep GPUs busy with dense math; CPU would run many small loops.
- All-reduce once per step is cheap relative to compute; most time is spent in fused kernels.

## Data Flow
- Packs (`*_torax_training.npz`) contain time bases (`t`, `t_ts`), profiles (Te, ne), masks, geometry (`rho`, `Vprime`), controls, regimes.
- `load_data` loads packs, aligns grids, builds ROM rho grid, computes intersection observation indices, constructs `ShotBundle` with interpolants for controls, ne, and edge Te.
- Training loop samples batches of `ShotBundle`s, runs `diffeqsolve` over each shot, computes losses, and updates parameters.

## Why This Is Faster Than CPU (even for ODEs)
- We aren’t integrating one tiny time series; we integrate many shots × many rho points per batch. That’s a lot of math to fuse.
- XLA fuses the flux/divergence math, MLP calls, and loss into big kernels → fewer memory trips.
- `vmap` + `pmap` keep GPUs fed with large dense ops; CPU would run many small loops instead.
- All-reduce once per step is cheap relative to the compute; GPUs stay busy most of the time.

## Practical Notes for New GPU Users
- First step may be slower (JIT compile); subsequent steps are fast.
- Batch sizes must divide number of devices for `pmap`.
- Watch memory: padded batches increase footprint; adjust batch size if OOM.
- Ensure CUDA/JAX wheels match driver/toolkit; use the provided `run_training_gpu.sh` wrapper.

## Recent Optimization and Solver Fixes (stability + perf)
- **Static solver selection:** `solver_name` is threaded as a static arg through `pmap/jit/vmap` so XLA sees one solver specialization (KenCarp3 IMEX vs Kvaerno5/Tsit5). Avoids recompiles and shape polymorphism errors.
- **LossCfg static in validation JIT:** `eval_loss_on_indices` now treats `loss_cfg` as static, preventing tracer-to-float concretization when building `PIDController` tolerances.
- **IMEX decomposition for KenCarp3:** Split RHS into explicit (NN source + latent) and implicit (diffusion) via `diffrax.MultiTerm`, matching solver expectations and eliminating stiffness-related instability.
- **Deterministic batching + clamping:** When shot count < requested batch, we clamp to `n_shots * n_devices` and permute per device, ensuring every shot appears twice in the first step and avoiding uneven replica loads.
- **Safe dt0 estimation:** Initial step size derived from masked mean `dt` over valid windows (with clip), replacing brittle dynamic slicing that sometimes produced zero/negative `dt0`.
- **Strictly increasing time padding:** `pad_time_to_max_strict` pads with positive eps steps, guaranteeing monotone time for diffrax interpolants/SaveAt.
- **Dtype-stable norms:** Grad and update norms accumulate in float32 to avoid bfloat16 drift and are averaged across devices with `pmean`.
- **No silent repartitioning:** `make_step` works directly on partitioned params; optimizer state shape is validated to match params before replication.
- **NaN/Inf guards:** Gradients are scanned for non-finite values; if found, the update is skipped and norm logged zero, preventing optimizer corruption.
- **Validation-based checkpointing:** Fixed train/val split per restart; best-on-val checkpoints (and EMA snapshots when enabled) are saved to reduce overfitting/regressions.
- **Divergence/core safeguards:** Core volume floor and divergence floor prevent division blow-ups; boundary and source values are clipped to physical ranges before loss.

## References (package docs)
- JAX: https://jax.readthedocs.io
- Equinox: https://docs.kidger.site/equinox/
- Diffrax: https://docs.kidger.site/diffrax/
- Optax: https://optax.readthedocs.io
