#!/bin/bash
# Run training with logs saved to file

set -e

# Create outputs directory
mkdir -p outputs/logs

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="outputs/logs/training_${TIMESTAMP}.log"

echo "Running training..."
echo "Logs will be saved to: $LOG_FILE"
echo ""

# Activate venv and run training
source .venv/bin/activate

# Run with both terminal output AND file logging
python test_quick_training.py 2>&1 | tee "$LOG_FILE"

echo ""
echo "Training complete!"
echo "Logs saved to: $LOG_FILE"
echo "Model saved to: outputs/quick_model.pt"
