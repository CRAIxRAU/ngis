#!/bin/bash
#SBATCH --job-name=ngis_test
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plgghack2025bioart-gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/ngis_test_%j.out
#SBATCH --error=logs/ngis_test_%j.err

# Helios cluster testing script for NGIS model
# Uses single GPU for inference

echo "=========================================="
echo "NGIS Model Testing"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Started at: $(date)"
echo "=========================================="

# Load modules
module purge
module load CUDA/12.8
module load PyTorch/2.8.0-CUDA-12.8

# Activate virtual environment
source /net/storage/pr3/plgrid/plgghack2025bioart/venv/pytorch/bin/activate

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Print environment info
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "CUDA device: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"

# Find the best checkpoint (or use specified one)
CHECKPOINT=${1:-checkpoints/best_model.pth}
CONFIG=${2:-configs/cluster_full.yaml}
OUTPUT=${3:-test_results_$(date +%Y%m%d_%H%M%S).json}

echo ""
echo "Configuration:"
echo "  Checkpoint: $CHECKPOINT"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT"
echo ""

# Run testing
python test.py \
    --checkpoint "$CHECKPOINT" \
    --config "$CONFIG" \
    --output "$OUTPUT" \
    --device cuda \
    --verbose

# Print results
echo ""
echo "=========================================="
echo "Testing completed!"
if [ -f "$OUTPUT" ]; then
    echo "Results saved to: $OUTPUT"
    echo ""
    echo "Key metrics:"
    python -c "
import json
with open('$OUTPUT', 'r') as f:
    data = json.load(f)
    results = data['test_results']
    print(f\"  Mean Correlation: {results.get('mean_correlation_mean', 0):.4f} ± {results.get('mean_correlation_std', 0):.4f}\")
    print(f\"  MSE: {results.get('mse_mean', 0):.6f}\")
    print(f\"  RMSE: {results.get('rmse_mean', 0):.6f}\")
    print(f\"  NRMSE: {results.get('nrmse_mean', 0):.6f}\")
    print(f\"  PSD Similarity: {results.get('psd_similarity_mean', 0):.4f}\")
    print(f\"  Total Samples: {results.get('total_samples', 0)}\")
"
fi
echo "=========================================="
echo "Job finished at: $(date)"
echo "=========================================="
