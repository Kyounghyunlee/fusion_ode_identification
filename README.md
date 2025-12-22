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
model:
	latent_gain: 1.0
	source_scale: 3.0e5
```
Adjust shots as needed; `shots: "all"` will load every `*_torax_training.npz` in `data_dir`.

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

We provide a wrapper that loads modules and exports CUDA/NCCL paths for the SDCC environment, then runs training.

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

### What to expect in logs

The first step typically triggers a large JAX/XLA compile. The training script prints a warmup/compile timing line before the main loop so a “quiet period” is clearly identified as compile time.

## Debugging (single-shot)

There are two convenient options:

### Option A: Standalone debug runner (recommended)

Runs a single-shot forward solve using the configured checkpoint selection, then writes:
- `out/debug_shot_<SHOT>.png` (Te traces + z(t) + BC + div/src diagnostics)
- `out/debug_shot_<SHOT>.npz` (arrays used in the plot + diagnostics)

```bash
export PYTHONPATH="$PWD"
python -c "from fusion_ode_identification.debug import run_debug_shot; run_debug_shot('config/config.yaml', 27567, out_dir='out', solver_throw=False)"
ls -lh out/debug_shot_27567.png out/debug_shot_27567.npz
```

On SDCC/HPC GPU nodes, you can run the same command through the GPU wrapper (so you inherit the module loads + CUDA/NCCL library paths):

```bash
./scripts/run_training_gpu.sh --python -c "from fusion_ode_identification.debug import run_debug_shot; run_debug_shot('config/config.yaml', 27567, out_dir='out', solver_throw=False)"
```

### Option B: Debug-eval via the training entrypoint

This loads the dataset, loads a checkpoint, runs evaluation for one shot, and writes PNG/NPZ into `out/`:

```bash
export PYTHONPATH="$PWD"
python train_tokamak_ode_hpc.py --config config/config.yaml --debug_eval_only --debug_eval_shot 27567
```

On SDCC/HPC GPU nodes, the equivalent wrapper run is:

```bash
./scripts/run_training_gpu.sh --config config/config.yaml --debug_eval_only --debug_eval_shot 27567
```

## Evaluate a Trained Model

Generate evaluation plots and metrics for the saved checkpoint:
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
