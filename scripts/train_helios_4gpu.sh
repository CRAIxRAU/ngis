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

# Change to ngis directory
cd $HOME/ngis/ngis || { echo "Failed to cd to ngis directory"; exit 1; }

# Activate existing virtual environment
if [ -d "$HOME/venv/pytorch" ]; then
    echo "Activating existing virtual environment..."
    source $HOME/venv/pytorch/bin/activate
else
    echo "ERROR: Virtual environment not found at $HOME/venv/pytorch"
    echo "Please create it first with: python3 -m venv $HOME/venv/pytorch"
    exit 1
fi

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" || { echo "PyTorch not properly installed"; exit 1; }

# Remove old checkpoints if requested (comment out to keep them)
# rm -rf checkpoints/*.pth

echo "=========================================="
echo "Starting distributed training on 4× GH200 GPUs"
echo "Config: configs/cluster_full.yaml"
echo "Max duration per subject: 240 seconds"
echo "=========================================="

# Run distributed training using torchrun
# Using standalone mode since we're on a single node
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    main.py \
    --config configs/cluster_full.yaml \
    --epochs 50

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
