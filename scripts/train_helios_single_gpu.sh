#!/bin/bash
#SBATCH --job-name=ngis_train_1gpu
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/ngis_train_1gpu_%j.out
#SBATCH --error=logs/ngis_train_1gpu_%j.err

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

# Load modules
module purge
module load ML-bundle

# Activate virtual environment
source ~/venvs/ngis/bin/activate

# Environment variables for optimal performance
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0

# Navigate to project directory
cd $HOME/ngis

# Check GPU availability
nvidia-smi

# Run training
python main.py \
    --config configs/ds003766.yaml \
    --epochs 10

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
