#!/bin/bash -l
#SBATCH --job-name=ngis_train_4gpu
#SBATCH --account=plghack2025bioart-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/ngis_train_4gpu_%j.out
#SBATCH --error=logs/ngis_train_4gpu_%j.err

# Exit on error
set -e

# Create logs directory if it doesn't exist
mkdir -p logs

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Account: plghack2025bioart-gpu-gh200"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_PER_TASK"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 200GB"
echo "Start time: $(date)"
echo "=========================================="

# Change to ngis directory (relative to script location)
cd "$(dirname "$0")/.." || { echo "Failed to cd to ngis directory"; exit 1; }

# Load ML bundle module
echo "Loading ML-bundle module..."
module add ML-bundle/25.04

# Create and activate virtual environment with system site packages
if [ ! -d "venv/pytorch" ]; then
    echo "Creating new virtual environment at venv/pytorch..."
    python3 -m venv --system-site-packages venv/pytorch
fi

echo "Activating virtual environment..."
source venv/pytorch/bin/activate

# Install specific PyTorch version if not already installed
echo "Python: $(which python)"
if ! python -c "import torch; assert torch.__version__.startswith('2.8')" 2>/dev/null; then
    echo "Installing PyTorch 2.8.0+cu128..."
    pip3 install --no-cache-dir torch==2.8.0+cu128 torchvision==0.23.0
fi

# Install project dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing project dependencies from requirements.txt..."
    pip3 install -r requirements.txt
else
    echo "WARNING: requirements.txt not found"
fi

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

# Verify MNE installation
echo "Verifying MNE installation..."
python -c "import mne; print(f'MNE version: {mne.__version__}')"

# Remove old checkpoints if requested (comment out to keep them)
# rm -rf checkpoints/*.pth

echo "=========================================="
echo "Starting distributed training on 4× GH200 GPUs"
echo "Config: configs/cluster_full.yaml"
echo "Data sharding: Each GPU loads ~8 subjects (31 total / 4 GPUs)"
echo "=========================================="

# Run distributed training using torchrun
# Using standalone mode since we're on a single node
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    main.py \
    --config configs/cluster_full.yaml \
    --epochs 50 \
    --world_size 4

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
