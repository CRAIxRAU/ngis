#!/usr/bin/env python3
"""
Quick test: Does one forward pass work?
Tests channel selection + model architecture fixes.
"""

import torch
from utils.config import Config
from training.trainer import NGISTrainer
from data.dataloader import create_training_dataloader

def test_single_batch():
    """Test one forward pass through the model."""

    print("="*60)
    print("Testing NGIS Forward Pass")
    print("="*60)

    # Load config
    config = Config.from_yaml('configs/ds003766.yaml')
    print(f"[OK] Config loaded: {config.data.channels} channels")

    # Create trainer
    trainer = NGISTrainer(config, is_distributed=False)
    print(f"[OK] Trainer initialized")
    print(f"  - Model device: {trainer.device}")
    print(f"  - Model parameters: {sum(p.numel() for p in trainer.model.parameters()):,}")

    # Create dataloader (just need 1 batch)
    dataloader = create_training_dataloader(
        data_path=config.data.data_path,
        batch_size=2,
        segment_length=config.data.segment_length,
        overlap=config.data.overlap,
        augment=False,
        num_workers=0,  # Single-threaded for testing
        channel_selection_strategy=config.data.channel_selection_strategy,
        target_channels=config.data.channels
    )
    print(f"[OK] Dataloader created: {len(dataloader)} batches")

    # Get one batch
    batch = next(iter(dataloader))
    print(f"[OK] Loaded batch:")
    print(f"  - EEG shape: {batch['eeg'].shape}")
    print(f"  - EEG device: {batch['eeg'].device}")
    print(f"  - Has NaN: {torch.isnan(batch['eeg']).any().item()}")

    # Move to GPU
    batch['eeg'] = batch['eeg'].to(trainer.device)
    print(f"[OK] Moved to device: {trainer.device}")

    # Forward pass
    print("\nAttempting forward pass...")
    try:
        with torch.no_grad():
            outputs = trainer.model(
                eeg_input=batch['eeg'],
                return_spikes=True,
                return_graph=True
            )

        print("[OK] Forward pass SUCCESS!")
        print(f"\nOutputs:")
        print(f"  - EEG output shape: {outputs['eeg_output'].shape}")
        print(f"  - Spikes shape: {outputs['spikes'].shape if 'spikes' in outputs else 'N/A'}")
        print(f"  - Graph nodes: {outputs['graph'].num_nodes if 'graph' in outputs else 'N/A'}")
        print(f"  - Output has NaN: {torch.isnan(outputs['eeg_output']).any().item()}")

        # Test loss computation
        print("\nAttempting loss computation...")
        loss = trainer.criterion(
            outputs['eeg_output'],
            batch['eeg'],
            outputs.get('spikes'),
            outputs.get('graph')
        )
        print(f"[OK] Loss computed: {loss.item():.4f}")

        print("\n" + "="*60)
        print("[OK][OK][OK] ALL TESTS PASSED [OK][OK][OK]")
        print("="*60)
        print("\nThe model is working! Ready for full training.")
        return True

    except Exception as e:
        print(f"\n[FAIL] Forward pass FAILED!")
        print(f"Error: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("Need to fix architecture bugs before training.")
        print("="*60)
        return False

if __name__ == "__main__":
    success = test_single_batch()
    exit(0 if success else 1)
