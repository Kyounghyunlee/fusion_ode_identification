# Fusion ODE Identification

## Development Workflow

To work on the files directly on the SDCC cluster from your local machine (if configured) or to access the storage:

```bash
sdcc-mount
code ~/mnt/sdcc      # edits ITER files directly
sdcc-umount
```

## HPC Environment Setup (GPU)

This project uses a specific JAX + CUDA setup. We use a virtual environment with specific NVIDIA libraries linked.

### 1. Initial Setup (Run Once)

```bash
cd ~/research/fusion_ode_identification

# Load base Python
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.3.0
module load cuDNN/8.9.7.29-CUDA-12.3.0

# Create venv
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
# Note: We use specific JAX/CUDA versions. See requirements.txt or setup scripts.
pip install -r requirements.txt
```

## Everyday use

```bash
cd ~/research/fusion_ode_identification
module load Python/3.11.5-GCCcore-13.2.0
module load CUDA/12.3.0
module load cuDNN/8.9.7.29-CUDA-12.3.0
source venv/bin/activate

```

## Data Workflow

1) Download raw NetCDFs from S3:
```bash
python preprocessing/download_data.py --shots 27567 27568 --overwrite
```
2) Build TORAX training packs (.npz):
```bash
python preprocessing/build_training_pack.py --shots 27567 27568
# or discover any shot folders already under data/
python preprocessing/build_training_pack.py --discover
```
3) Inspect packs (optional):
```bash
python scripts/inspect_data.py
```

## Configuration

Edit [config/config.yaml](config/config.yaml) to set data paths, shots, and outputs. Minimal example:
```yaml
data:
	data_dir: "data"
	shots: [27567, 27568]
	intersection_rho_threshold: 0.05
	rom_n_interior: 16
output:
	model_id: "production_run_v1"
	save_dir: "models"
	log_dir: "logs"
	model_name: "physics_manifold_model"
training:
	batch_size: 64
	total_steps: 1000
	learning_rate: 3.0e-4
	ema_decay: 0.999  # Enable EMA for better generalization
	lambda_z: 1.0e-4  # Latent smoothness penalty
	imex:
		theta: 0.7
		substeps: 5
model:
	latent_gain: 1.0
	source_scale: 3.0e5
```
Adjust shots as needed; `shots: "all"` will load every `*_torax_training.npz` in `data_dir`.

**Recent optimizations:**
- Inverse-coverage weighting ensures all radii (dense or sparse) are supervised fairly.
- Geometry precomputation (P1.2) eliminates per-substep recomputation overhead.
- EMA validation tracking saves both raw and EMA best checkpoints independently.
- Lambda_z smoothness penalty stabilizes latent trajectories.

## Connect to Compute Node

**Important:** Training must run on the compute node, not the login node.

### Step 1: Find Available Compute Nodes

```bash
# List all partitions and nodes
sinfo

# Or search for GPU partitions specifically
sinfo | grep gpu
```

Look for the `titan` partition with node `98dci4-gpu-0002`.

### Step 2: Check Node Resources

```bash
# See detailed node information
scontrol show node 98dci4-gpu-0002

# Quick check: just show key resources
scontrol show node 98dci4-gpu-0002 | grep -E "State|CPUTot|RealMemory|Gres"
```

Expected output shows:
- `State=IDLE` or `ALLOCATED` (availability)
- `CPUTot=20` (20 CPU cores available)
- `RealMemory=256000` (256 GB RAM)
- `Gres=gpu:8` (8 GPUs available)

### Step 3: Connect to Compute Node

**Option A: Maximum resources (recommended for training)**
```bash
srun --partition=titan --gres=gpu:8 --cpus-per-task=20 --mem=200G --pty bash
```
This allocates all 8 GPUs, 20 CPUs, and 200GB RAM.

**Option B: Minimal resources (for testing)**
```bash
srun --partition=titan --gres=gpu:1 --cpus-per-task=4 --mem=32G --pty bash
```
This allocates 1 GPU, 4 CPUs, and 32GB RAM for quick tests.

### Step 4: Verify You're on Compute Node

```bash
# Check hostname (should show 98dci4-gpu-0002)
hostname

# Check available GPUs (should show 8 × Tesla P100)
nvidia-smi -L

# Check CPUs
nproc

# Check memory
free -h
```

### Step 5: Run Training

Once on the compute node:
```bash
# Standard training run
./scripts/run_training_gpu.sh --config config/config.yaml

# Or with tmux (recommended for long runs)
tmux new -s train
./scripts/run_training_gpu.sh --config config/config.yaml
# Detach: Ctrl+b then d
# Reattach later: tmux attach -t train
```

### Step 6: Return to Login Node

When training is complete or you want to disconnect:
```bash
exit
```

Your session on the compute node ends and you return to the login node.

### Notes on Resource Allocation

- **Automatic scaling**: Training code uses `jax.pmap` to automatically distribute across all allocated GPUs
- **Batch size**: Will be adjusted to be divisible by number of GPUs
- **Each GPU processes**: `batch_size / n_gpus` shots in parallel
- **Node specifications**: 8 × Tesla P100 (16GB each), 20 CPUs, 256GB RAM total

## Running Training on GPU

This repo supports both “auto” device selection (whatever JAX sees) and explicit forcing of CPU/GPU.

Important:
- Run commands from the repo root.
- If you run ad-hoc one-liners (`python -c ...`), export `PYTHONPATH="$PWD"` so imports work.

### Auto (CPU or GPU)

If you are on a GPU node and your JAX install supports CUDA, this will use GPU automatically; otherwise it will run on CPU.

```bash
export PYTHONPATH="$PWD"
python train_tokamak_ode_hpc.py --config config/config.yaml
```

### Force CPU

```bash
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cpu
python train_tokamak_ode_hpc.py --config config/config.yaml
```

### Force GPU (generic)

On a GPU node, you can force CUDA backend like this:

```bash
export PYTHONPATH="$PWD"
export JAX_PLATFORMS=cuda
python train_tokamak_ode_hpc.py --config config/config.yaml
```

### Force GPU (HPC wrapper)

We provide a **canonical GPU wrapper** (`scripts/run_training_gpu.sh`) that loads modules, exports CUDA/NCCL paths, and enforces `JAX_ENABLE_X64=1` for the SDCC environment. This is the **only GPU entrypoint**; use it for training, debugging, and smoke checks.

- Standard run:
```bash
./scripts/run_training_gpu.sh --config config/config.yaml
```
- With tmux (recommended):
```bash
tmux new -s train
./scripts/run_training_gpu.sh --config config/config.yaml
# detach: Ctrl+b then d; reattach: tmux attach -t train
```
- Run arbitrary Python scripts via `--python`:
```bash
./scripts/run_training_gpu.sh --python scripts/check_bc.py --config config/config_debug.yaml --shot 27567
```

### What to expect in logs

The first step typically triggers a large JAX/XLA compile. The training script prints a warmup/compile timing line before the main loop so a “quiet period” is clearly identified as compile time.

## Debugging (single-shot)

### Quick debug-eval via training entrypoint

This loads the dataset, loads a checkpoint (prefers `_best_ema.eqx` first, then `_best.eqx`, then `_finetuned.eqx` only if `training.lbfgs_finetune: true`), runs evaluation for one shot, and writes PNG/NPZ into `out/`:

```bash
./scripts/run_training_gpu.sh --config config/config.yaml --debug_eval_only --debug_eval_shot 27567
```

### Smoke Checks (Sanity Tests)

Run lightweight regression checks before/after training:

```bash
# BC regression check (edge Te peak-to-peak > threshold)
./scripts/run_training_gpu.sh --python scripts/check_bc.py --config config/config_debug.yaml --shot 27567

# Diffusion operator sanity (const profile → div≈0, BC coupling sign)
./scripts/run_training_gpu.sh --python scripts/smoke_diffusion_sanity.py --config config/config_debug.yaml --shot 27567

# Time padding strictness (survives float32 downcast)
./scripts/run_training_gpu.sh --python scripts/smoke_time_padding_strict.py
```

## Evaluate a Trained Model

Generate evaluation plots and metrics for the saved checkpoint (prefers `_best_ema.eqx` if available):
```bash
python scripts/evaluate_model.py --config config/config.yaml --model-id production_run_v1 --data-check
```
Outputs go to `logs/<model_id>/evaluation/` (JSON report + PNG plots).

## Dependency Management

- Install pinned deps (after creating/activating the venv):
```bash
pip install -r requirements.txt
```
- Add or update deps: edit `requirements.in`, then regenerate and install:
```bash
pip-compile --output-file=requirements.txt requirements.in
pip install -r requirements.txt
```
