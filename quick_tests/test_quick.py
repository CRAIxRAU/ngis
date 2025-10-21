#!/usr/bin/env python3
"""
SUPER QUICK TEST - Uses tiny synthetic data
Tests if the model works in <10 seconds
"""

import torch
from utils.config import Config
from training.trainer import NGISTrainer

print("="*60)
print("QUICK TEST - Synthetic Data")
print("="*60)

# Load config
config = Config.from_yaml('../configs/ds003766.yaml')
print(f"[1/5] Config loaded: {config.data.channels} channels")

# Create trainer
trainer = NGISTrainer(config, is_distributed=False)
print(f"[2/5] Trainer initialized on {trainer.device}")

# Create FAKE batch (no data loading needed!)
fake_batch = {
    'eeg': torch.randn(2, 128, 1000).to(trainer.device),  # 2 samples, 128 channels, 1 second
    'info': [{}, {}]
}
print(f"[3/5] Created synthetic batch: {fake_batch['eeg'].shape}")

# Test forward pass
print("[4/5] Testing forward pass...")
try:
    with torch.no_grad():
        outputs = trainer.model(
            eeg_input=fake_batch['eeg'],
            return_spikes=True,
            return_graph=True
        )

    print(f"[5/5] SUCCESS!")
    print(f"  Input shape:  {fake_batch['eeg'].shape}")
    print(f"  Output shape: {outputs['eeg_output'].shape}")
    print(f"  Match: {outputs['eeg_output'].shape == fake_batch['eeg'].shape}")

    print("\n" + "="*60)
    print("MODEL WORKS! Ready for real training.")
    print("="*60)

except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
