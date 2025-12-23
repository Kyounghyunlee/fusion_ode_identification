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
	- Autodiff: reverse-mode differentiation (backprop) through regular code and through the fixed-step IMEX integrator.
- **equinox (eqx)**: Minimal neural-net module system for JAX. Lets us define `HybridField`/`SourceNN`, split trainable parameters from static fields, and save/load checkpoints.
- **IMEX integrator**: Custom fixed-step theta-method (theta=0.7, substeps=5 by default) for implicit diffusion + explicit source/latent updates. This branch is **IMEX-only**; all Diffrax-based integration paths have been removed for simplicity and reverse-mode autodiff compatibility (see `fusion_ode_identification.imex_solver`).
- **optax**: Optimizers (AdamW), gradient clipping, learning-rate schedules, EMA of weights.
- **NumPy/xarray/zarr/fsspec**: Load and reshape data; zarr+xarray read S3-stored diagnostics; NumPy writes packs.
- **Matplotlib**: Make evaluation plots.

## Model Architecture (training script)
- `HybridField`: PDE-inspired RHS combining diffusion (chi profile), finite-volume divergence, and a learned source NN. Latent scalar `z` evolves via a low-order ODE (`LatentDynamics`). State bounds (Te ∈ [0, 5000], z ∈ [-10, 10]) are enforced via **smooth clamps** (softplus-based) rather than hard clips to avoid zero-gradient saturation.
- `SourceNN`: small MLP taking `(rho, Te, ne, controls, z)`; final layer zero-initialized for stability, scaled by `source_scale`.
- `ShotBundle`: batched, padded shot data (times, profiles, masks, controls, boundary Te, geometry); built in `fusion_ode_identification.data.load_data` and consumed by `train_tokamak_ode_hpc.py`.
- Loss: weighted Huber data term on **all interior masked radii** (weighted by per-column coverage), source magnitude regularizer, weak-constraint model error, regime supervision on `z`, latent regularization. Training now supervises all available radii, not just a small intersection subset.

## Parallelism and Performance (why GPU, what the terms mean)
- **Why GPU helps now**: We have many shots and many rho points. We batch them so the GPU crunches large matrices instead of tiny loops. More arithmetic per batch → GPU wins.
- **Batching**: We stack shots into uniform shapes (padding). One forward/backward pass handles a whole batch.
- **`vmap` (vectorize)**: Replaces Python loops over batch/shot with a single fused kernel. Used inside loss to evaluate multiple shots on one device.
- **`pmap` (parallel map)**: Splits the batch across GPUs. Each GPU runs the same compiled code on its chunk. Gradients are averaged (all-reduce) so the update is as if it saw the whole batch.
- **`jit` (just-in-time compile)**: First call traces the function (slow); the compiled version then runs fast. We JIT the training step so the ODE solve, loss, and optimizer are fused.
- **IMEX on GPU**: fixed-step IMEX updates run inside JIT; diffusion is a small banded linear solve per step, source/latent are explicit. The first step includes a one-time warmup compile so the initial "quiet period" is clearly attributed to XLA compilation. Current defaults: theta=0.7, substeps=5 per observation interval.
- **EMA (exponential moving average)**: A smoothed copy of weights that often generalizes better. Optax updates it cheaply each step. Checkpoint selection prefers `_best_ema.eqx` first.
- **Gradient clipping**: Limit global gradient norm to avoid exploding updates.
- **L-BFGS finetune**: After AdamW, an optional quasi-Newton refinement on a small batch to squeeze a bit more accuracy. Only enabled via `training.lbfgs_finetune: true`.
- **Diagnostics**: Training logs now include mean absolute diffusion `|diff|`, mean absolute source `|src|`, and their ratio `src/diff` alongside standard metrics.

## Autodiff and Backprop (what actually happens)
- We run the model forward: solve the ODE for each shot → predicted temperatures + latent `z`.
- Loss is computed (Huber data loss, model-error penalty, source/latent regs, regime supervision).
- JAX records the operations; reverse-mode autodiff replays them backward to get gradients w.r.t. every parameter (MLP weights, chi params, latent params).
- The IMEX integrator is written in JAX (`lax.scan`/`fori_loop`), so sensitivities account for every fixed step.
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
	ext{grad}_T = D T,\quad \text{flux} = -(P\,V'\chi) \odot \text{grad}_T,\quad \text{div} = A\,\text{flux},
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
- `fusion_ode_identification.data.load_data` loads packs, aligns per-shot time windows, builds an intersection observed set, builds a ROM grid, constructs edge boundary-condition traces, and stacks everything into a padded `ShotBundle`.
- Training loop samples batches of `ShotBundle`s, runs the IMEX integrator over each shot, computes losses, and updates parameters.

## Why This Is Faster Than CPU (even for ODEs)
- We aren’t integrating one tiny time series; we integrate many shots × many rho points per batch. That’s a lot of math to fuse.
- XLA fuses the flux/divergence math, MLP calls, and loss into big kernels → fewer memory trips.
- `vmap` + `pmap` keep GPUs fed with large dense ops; CPU would run many small loops instead.
- All-reduce once per step is cheap relative to the compute; GPUs stay busy most of the time.

## Practical Notes for New GPU Users
- First step may be slower (JIT compile); subsequent steps are fast.
- Batch sizes must divide number of devices for `pmap`.
- Watch memory: padded batches increase footprint; adjust batch size if OOM.
- Ensure CUDA/JAX wheels match driver/toolkit; on SDCC/HPC use the canonical wrapper `scripts/run_training_gpu.sh` for all operations (training, debug, smoke checks).
- Run smoke checks before/after training: `./scripts/run_training_gpu.sh --python scripts/smoke_diffusion_sanity.py --config <cfg> --shot <id>` and similar for time-padding and BC checks.

## Recent Optimization and Solver Fixes (stability + perf)
- **IMEX-only branch:** All Diffrax and non-IMEX solver paths removed for simplicity. Implicit diffusion operator now exactly matches explicit discretization (conservative flux-form).
- **Static IMEX/loss config in compiled paths:** In `train_tokamak_ode_hpc.py`, `make_step` uses `pmap(..., static_broadcasted_argnums=(3, 4))`, and `eval_loss_on_indices` uses `jit(..., static_argnums=(2, 3))`, so `LossCfg` and `IMEXConfig` are compile-time constants.
- **Padded-time solves with valid-window masking:** Training solves over the padded time array (to keep `SaveAt(ts=...)` valid/strictly increasing) and applies a `time_mask` derived from `t_len` for every loss term. IMEX integrator uses `active_mask` to freeze dynamics in padded regions.
- **Strictly increasing padding:** `fusion_ode_identification.data.pad_time_to_max_strict` appends steps guaranteed to remain strictly increasing even after float32 downcast, using `max(eps, ulp64, ulp32)` at the last time value.
- **Smooth clamps replace hard clips:** State bounds (Te, z) enforced via softplus-based smooth clamps to preserve nonzero gradients near saturation, reducing gradient-killing.
- **All-radii supervision with inverse-coverage weighting:** Training loss now supervises all interior masked radii with inverse-coverage normalized weights (each radius gets comparable total weight regardless of observation density), eliminating "coverage² punishment" of sparse radii and improving data utilization.
- **Lambda_z smoothness regularization:** `lambda_z` is implemented as a latent smoothness penalty (mean squared Δz across time) rather than a no-op, encouraging stable latent trajectories.
- **EMA validation and independent best tracking:** When `training.ema_decay > 0`, the training script evaluates both raw and EMA parameters on validation at log intervals, tracks separate global bests (`_best.eqx` for raw, `_best_ema.eqx` for EMA), and logs both metrics for comparison.
- **Geometry precomputation (P1.2):** Per-shot geometry arrays (`dr`, `Vprime_face`, `Vprime_cell`, `denom`) are computed once and passed through IMEX args, eliminating repeated recomputation in every substep and significantly reducing diffusion operator overhead.
- **Interpolant-free IMEX substeps (P1.1):** Controls and edge BC are evaluated once at save points and linearly blended across substeps, removing per-substep searchsorted interpolation overhead.
- **IMEX lax.cond branch fix:** `interval_scan` active/inactive branches now return identical carry tuple structures `(y, t, Te_edge, ctrl, ne)` to satisfy JAX's type-structure invariant.
- **Numerical guards:** diffusion denominator floors near the core, "skip update on NaN/Inf grads", and finiteness checks on solver success.
- **L-BFGS finetune correctness:** Uses shared `build_model_template` for deserialization (matching trained model statics), selects hardest train-only shots (never validation), and finetunes the full inexact parameter tree rather than last-layer-only.
- **Multi-GPU one-shot debug:** Device list is reduced to `min(n_devices, n_shots)` when `--debug_one_shot` is used on multi-GPU allocations, preventing batch-sizing failures.
- **X64 enforcement:** `JAX_ENABLE_X64=1` exported in the canonical GPU wrapper for consistent dtype behavior across training/eval/smoke checks.
- **Smoke checks:** Lightweight sanity checks added: `scripts/smoke_diffusion_sanity.py` (const profile → div≈0, BC coupling sign), `scripts/smoke_time_padding_strict.py` (padding strictness under downcast), `scripts/check_bc.py` (edge BC sanity).

## References (package docs)
- JAX: https://jax.readthedocs.io
- Equinox: https://docs.kidger.site/equinox/
- Optax: https://optax.readthedocs.io
