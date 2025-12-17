# Code Architecture and HPC Overview

## Big Picture
- Goal: learn a reduced plasma transport dynamics model (1D heat equation + latent source) from many discharges ("shots").
- Pipeline: download → pack build (NetCDF → NPZ) → train JAX model → evaluate.
- Design themes: physics-structured model, JAX for autodiff + accelerator execution, data-parallel training for throughput.

## Core Packages and Roles
- **JAX**: array library with XLA compilation; provides `jit`, `vmap`, `pmap`, and automatic differentiation (reverse-mode/backprop). It lets us stage computations to GPU/TPU and fuse operations.
- **equinox (eqx)**: lightweight PyTorch-like module system on top of JAX; handles parameter trees, filtering params vs. static fields, serialization (`tree_serialise_leaves`).
- **diffrax**: differentiable ODE solvers in JAX; integrates the hybrid field with adaptive steppers (Kvaerno5/PID) and supports autodiff through the solver path.
- **optax**: optimizer library (AdamW, schedules, clipping, EMA) with JAX primitives for parallel-safe updates.
- **NumPy/xarray/zarr/fsspec**: data handling; xarray/zarr for reading remote S3-stored diagnostics, NumPy for pack creation.
- **Matplotlib**: plotting for evaluation outputs.

## Model Architecture (training script)
- `HybridField`: PDE-inspired RHS combining diffusion (chi profile), finite-volume divergence, and a learned source NN. Latent scalar `z` evolves via a low-order ODE (`LatentDynamics`).
- `SourceNN`: small MLP taking `(rho, Te, ne, controls, z)`; final layer zero-initialized for stability, scaled by `source_scale`.
- `ShotBundle`: batched, padded shot data (times, profiles, masks, controls, boundary Te, geometry); built in `train_ode_physics_manifold_hpc.load_data`.
- Loss: weighted Huber data term, source magnitude regularizer, weak-constraint model error, regime supervision on `z`, latent regularization.

## Parallelism and Performance
- **Why GPU helps here**: Although individual time series are modest, we train over many shots and multiple rho points. Vectorized operations (over time, rho, batch) let XLA fuse large kernels; GPU excels at dense FLOPs and parallel reductions. More shots → higher arithmetic intensity.
- **Batching**: random batches of shots; arrays are stacked/padded to uniform shapes so kernels are dense and GPU-friendly.
- **`vmap`**: applies model and loss over shots within a device batch without Python loops, generating single fused kernels.
- **`pmap` / data parallelism**: training script shards batches across available GPUs; gradients are averaged with all-reduce (via `jax.lax.pmean`). Each device runs the same compiled program on its slice.
- **`jit`**: JIT-compiles solver + loss; after warmup, subsequent steps reuse the compiled executable, eliminating Python overhead.
- **Adaptive ODE solve on GPU**: diffrax runs inside JIT; adaptive stepper still benefits because the control flow is staged; masked/NaN-safe ops keep kernels stable.
- **EMA and L-BFGS**: optax EMA smooths parameters; optional single-device L-BFGS finetune refines the best checkpoint.

## Autodiff and Backprop
- **Reverse-mode autodiff**: JAX traces the forward pass, builds a computation graph, then applies reverse-mode (backprop) to compute gradients of the loss w.r.t. parameters, even through the ODE solve (diffrax supplies sensitivity handling/backprop-compatible adjoints).
- **Filter-based grads**: equinox filters parameters (floating leaves) from static fields so only trainable tensors receive gradients/updates.
- **Stability aids**: gradient clipping in optax (`clip_by_global_norm`), Huber loss to reduce sensitivity to outliers, source and latent regularizers to avoid exploding dynamics.

## Data Flow
- Packs (`*_torax_training.npz`) contain time bases (`t`, `t_ts`), profiles (Te, ne), masks, geometry (`rho`, `Vprime`), controls, regimes.
- `load_data` loads packs, aligns grids, builds ROM rho grid, computes intersection observation indices, constructs `ShotBundle` with interpolants for controls, ne, and edge Te.
- Training loop samples batches of `ShotBundle`s, runs `diffeqsolve` over each shot, computes losses, and updates parameters.

## Why This Is Faster Than CPU
- GPU benefits from large, fused linear algebra kernels; `vmap`+`pmap` turn many small per-shot operations into big batched kernels.
- XLA fuses elementwise ops (e.g., flux/divergence, MLP, loss) reducing memory traffic.
- Data-parallel shards keep all GPUs busy; communication cost is limited to gradient all-reduce once per step.
- Even with adaptive solvers, per-step work is vectorized over rho/time and batched over shots, making good use of GPU SIMD units.

## Practical Notes for New GPU Users
- First step may be slower (JIT compile); subsequent steps are fast.
- Batch sizes must divide number of devices for `pmap`.
- Watch memory: padded batches increase footprint; adjust batch size if OOM.
- Ensure CUDA/JAX wheels match driver/toolkit; use the provided `run_training_gpu.sh` wrapper.

## References (package docs)
- JAX: https://jax.readthedocs.io
- Equinox: https://docs.kidger.site/equinox/
- Diffrax: https://docs.kidger.site/diffrax/
- Optax: https://optax.readthedocs.io
