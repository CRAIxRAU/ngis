#!/bin/bash
# Benchmark script for cluster

set -e

echo "Loading modules..."
# Adjust these based on your cluster's available modules
module purge
module load Python/3.11.5-GCCcore-13.2.0
# Add other required modules (PyTorch, CUDA, etc.)

echo "Python version:"
python --version

echo "PyTorch check:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

echo ""
echo "Running benchmark..."
python benchmark_lif.py
