#!/bin/bash
#SBATCH --job-name=ngis_train_4gpu
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=48:00:00
#SBATCH --output=logs/ngis_train_4gpu_%j.out
#SBATCH --error=logs/ngis_train_4gpu_%j.err

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: 256GB"
echo "Start time: $(date)"
echo "=========================================="


module add ML-bundle/25.04

# Activate virtual environment
python3 -m venv --system-site-packages venv/pytorch
. venv/pytorch/bin/activate

pip3 install --no-cache-dir torch==2.8.0+cu128 torchvision==0.23.0


# Environment variables for distributed training
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500
export WORLD_SIZE=4
export NCCL_DEBUG=INFO

# Navigate to project directory
cd ../


echo "=========================================="
echo "Starting distributed training on 4× GH200 GPUs"
echo "=========================================="

# Run distributed training using torchrun
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=4 \
    main.py \
    --config configs/cluster.yaml \
    --epochs 50 \
    --distributed

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
