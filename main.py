#!/usr/bin/env python3
"""
NGIS: Neural Graph Inverse Simulator
Main entry point for training G-SNN models on EEG data.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from rich.console import Console
from rich.logging import RichHandler

from data.dataloader import create_training_dataloader
from training.trainer import NGISTrainer
from utils.config import Config
from utils.logging import setup_logging

console = Console()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="NGIS: Neural Graph Inverse Simulator Training"
    )
    
    # Configuration
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=None,
        help="Path to EEG data directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs",
        help="Output directory for checkpoints and logs"
    )
    
    # Training parameters
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=None,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=None,
        help="Learning rate"
    )
    
    # Distributed training
    parser.add_argument(
        "--world_size", 
        type=int, 
        default=1,
        help="Number of processes for distributed training"
    )
    parser.add_argument(
        "--rank", 
        type=int, 
        default=0,
        help="Rank of current process"
    )
    parser.add_argument(
        "--dist_url", 
        type=str, 
        default="tcp://localhost:10001",
        help="URL for distributed training"
    )
    
    # Misc
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Dry run without actual training"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    return Config.from_yaml(config_path)


def setup_distributed(args):
    """Setup distributed training if world_size > 1."""
    # When using torchrun, get rank from environment variable
    if 'RANK' in os.environ:
        args.rank = int(os.environ['RANK'])
    if 'LOCAL_RANK' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    else:
        args.local_rank = args.rank

    if args.world_size > 1:
        # When using torchrun, it sets all env vars automatically
        # Just call init_process_group without manual init_method
        torch.distributed.init_process_group(backend='nccl')
        # Set device to local rank
        torch.cuda.set_device(args.local_rank)
        return True
    return False


def main():
    """Main training function."""
    args = parse_args()
    
    # Setup logging
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("ngis")
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Override config with command line arguments
    if args.data_path:
        config.data.data_path = args.data_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
    
    # Setup distributed training
    is_distributed = setup_distributed(args)
    if is_distributed:
        logger.info(f"Initialized distributed training with {args.world_size} processes")
    
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Create output directory
    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Log system information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA devices: {torch.cuda.device_count()}")
        logger.info(f"Current device: {torch.cuda.current_device()}")
    
    # Initialize trainer
    try:
        trainer = NGISTrainer(
            config,
            is_distributed=is_distributed,
            rank=args.rank,
            world_size=args.world_size
        )
        logger.info("Initialized NGIS trainer successfully")
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        sys.exit(1)

    # Create dataloaders with proper train/val/test splits
    try:
        logger.info("Creating dataloaders with subject-level splits...")

        # Get optional max_duration for fast testing
        max_duration = getattr(config.data, 'max_duration', None)
        
        # Get splits configuration path
        splits_config = getattr(config.data, 'splits_config', 'configs/splits.yaml')

        # Create TRAIN dataloader - uses subjects from 'train' split
        train_dataloader = create_training_dataloader(
            data_path=config.data.data_path,
            batch_size=config.data.batch_size,
            segment_length=config.data.segment_length,
            overlap=config.data.overlap,
            augment=config.data.augment,  # Augment training data
            num_workers=config.data.num_workers,
            distributed=is_distributed,
            rank=args.rank,
            world_size=args.world_size,
            channel_selection_strategy=config.data.channel_selection_strategy,
            target_channels=config.data.channels,
            max_duration=max_duration,
            split='train',  # FIXED: Use 'train' split
            splits_config_path=splits_config,
            shard_across_ranks=False  # Use DistributedSampler with drop_last=True for even distribution
        )

        # Create VALIDATION dataloader - uses subjects from 'validation' split
        # FIXED: No more data leakage - validation uses completely different subjects
        val_dataloader = create_training_dataloader(
            data_path=config.data.data_path,
            batch_size=config.data.batch_size,
            segment_length=config.data.segment_length,
            overlap=config.data.overlap,
            augment=False,  # No augmentation for validation
            num_workers=config.data.num_workers,
            distributed=is_distributed,
            rank=args.rank,
            world_size=args.world_size,
            channel_selection_strategy=config.data.channel_selection_strategy,
            target_channels=config.data.channels,
            max_duration=max_duration,
            split='validation',  # FIXED: Use 'validation' split
            splits_config_path=splits_config,
            shard_across_ranks=False  # Use DistributedSampler with drop_last=True for even distribution
        )

        # Set dataloaders on trainer
        trainer.set_dataloaders(train_dataloader, val_dataloader)
        logger.info(f"Created dataloaders: {len(train_dataloader)} train batches, {len(val_dataloader)} val batches")
        logger.info("FIXED: Train and validation now use different subjects - proper generalization assessment!")
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}")
        sys.exit(1)

    # Dry run check
    if args.dry_run:
        logger.info("Dry run mode - exiting without training")
        return
    
    # Start training
    try:
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training completed successfully")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 