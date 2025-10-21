#!/usr/bin/env python3
"""
Test data loading for ds003766 dataset
"""

import logging
from pathlib import Path

import torch
from utils.config import Config
from utils.logging import setup_logging
from data.dataloader import create_training_dataloader

def main():
    """Test loading EEG data."""

    # Setup logging
    setup_logging(level=logging.INFO)
    logger = logging.getLogger("ngis")

    logger.info("="*60)
    logger.info("Testing ds003766 Data Loading")
    logger.info("="*60)

    # Load config
    config = Config.from_yaml("configs/ds003766.yaml")
    logger.info(f"Loaded config from configs/ds003766.yaml")
    logger.info(f"Data path: {config.data.data_path}")
    logger.info(f"Channels: {config.data.channels}")
    logger.info(f"Batch size: {config.data.batch_size}")
    logger.info(f"Segment length: {config.data.segment_length}")

    # Create dataloader
    try:
        logger.info("\nCreating dataloader...")
        dataloader = create_training_dataloader(
            data_path=config.data.data_path,
            batch_size=config.data.batch_size,
            segment_length=config.data.segment_length,
            overlap=config.data.overlap,
            augment=config.data.augment,
            num_workers=config.data.num_workers
        )
        logger.info(f"✓ Dataloader created: {len(dataloader)} batches")
    except Exception as e:
        logger.error(f"✗ Failed to create dataloader: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test loading one batch
    try:
        logger.info("\nLoading first batch...")
        batch = next(iter(dataloader))
        logger.info(f"✓ Batch loaded successfully!")
        logger.info(f"  EEG shape: {batch['eeg'].shape}")
        logger.info(f"  EEG dtype: {batch['eeg'].dtype}")
        logger.info(f"  EEG min/max: {batch['eeg'].min():.3f} / {batch['eeg'].max():.3f}")
        logger.info(f"  Number of samples in batch: {len(batch['info'])}")

        expected_shape = (config.data.batch_size, config.data.channels, config.data.segment_length)
        if batch['eeg'].shape == expected_shape:
            logger.info(f"✓ Shape matches expected: {expected_shape}")
        else:
            logger.warning(f"✗ Shape mismatch! Expected {expected_shape}, got {batch['eeg'].shape}")

    except Exception as e:
        logger.error(f"✗ Failed to load batch: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test loading multiple batches
    try:
        logger.info("\nTesting multiple batches...")
        n_test_batches = min(3, len(dataloader))
        for i, batch in enumerate(dataloader):
            if i >= n_test_batches:
                break
            logger.info(f"  Batch {i+1}/{n_test_batches}: shape={batch['eeg'].shape}")
        logger.info(f"✓ Successfully loaded {n_test_batches} batches")
    except Exception as e:
        logger.error(f"✗ Failed during multi-batch test: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("\n" + "="*60)
    logger.info("✓ All tests passed! Data loading works correctly.")
    logger.info("="*60)

    logger.info("\nNext steps:")
    logger.info("1. Fix model architecture issues (see ISSUES.md)")
    logger.info("2. Run: python main.py --config configs/ds003766.yaml --dry_run")
    logger.info("3. After fixes: python main.py --config configs/ds003766.yaml")

if __name__ == "__main__":
    main()
