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

## Running Training on GPU

We have a wrapper that exports CUDA paths and runs the training script with the current config.

- Standard run:
```bash
./scripts/run_training_gpu.sh
```
- With tmux (recommended):
```bash
tmux new -s train
./scripts/run_training_gpu.sh
# detach: Ctrl+b then d; reattach: tmux attach -t train
```

## Configuration

Training configuration lives in [config/config.yaml](config/config.yaml). Key knobs:
- `model.source_scale`: Source NN scale (≈3e5 for physical units).
- `training.grad_clip`: Gradient clipping value.
- `training.batch_size`: Batch size.
