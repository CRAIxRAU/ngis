#!/bin/bash -l
#SBATCH --job-name=ngis_train_4gpu
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=logs/ngis_train_4gpu_%j.out
#SBATCH --error=logs/ngis_train_4gpu_%j.err

# Job info
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_PER_TASK"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "=========================================="

# Load ML bundle
module add ML-bundle/25.04

# Create and activate virtual environment
python3 -m venv --system-site-packages venv/pytorch
. venv/pytorch/bin/activate

# Install PyTorch
pip3 install --no-cache-dir torch==2.8.0+cu128 torchvision==0.23.0

# Set up distributed training environment
nodes_array=( $( scontrol show hostname $SLURM_NODELIST ) )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
export LOGLEVEL=INFO

echo "=========================================="
echo "Starting distributed training on 4× GH200 GPUs"
echo "Head node: $head_node_ip"
echo "=========================================="

# Run distributed training using torchrun
srun torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_id=$RANDOM \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$head_node_ip:12345 \
    main.py \
    --config configs/cluster.yaml \
    --epochs 50 \
    --distributed

echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
