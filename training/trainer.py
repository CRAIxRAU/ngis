"""
Main trainer for NGIS.

Handles training loop, distributed training, and model optimization
for the graph-structured spiking neural network.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.gsnn import GSNN
from training.loss_functions import CombinedLoss
from training.optimizer import NGISOptimizer
from training.scheduler import NGISScheduler
from utils.checkpointing import CheckpointManager
from utils.logging import setup_logging

logger = logging.getLogger(__name__)


class NGISTrainer:
    """
    Main trainer for NGIS.
    
    Handles training loop, distributed training, and model optimization
    for the graph-structured spiking neural network.
    """
    
    def __init__(
        self,
        config,
        is_distributed: bool = False,
        rank: int = 0,
        world_size: int = 1
    ):
        """
        Initialize NGIS trainer.
        
        Args:
            config: Configuration object.
            is_distributed: Whether using distributed training.
            rank: Rank of current process.
            world_size: Total number of processes.
        """
        self.config = config
        self.is_distributed = is_distributed
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self._init_model()
        self._init_loss_function()
        self._init_optimizer()
        self._init_scheduler()
        self._init_checkpointing()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        logger.info(f"Initialized NGIS trainer on {self.device}")
    
    def _init_model(self):
        """Initialize G-SNN model."""
        model_config = self.config.model
        
        self.model = GSNN(
            n_channels=model_config.n_channels,
            n_neurons=model_config.n_neurons,
            n_layers=model_config.n_layers,
            hidden_dim=model_config.hidden_dim,
            graph_type=model_config.graph_type,
            connectivity_threshold=model_config.connectivity_threshold,
            dropout=model_config.dropout
        ).to(self.device)
        
        # Wrap with DDP if distributed
        if self.is_distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
                find_unused_parameters=True
            )
        
        logger.info(f"Initialized model with {self.model.count_parameters()['trainable_parameters']} parameters")
    
    def _init_loss_function(self):
        """Initialize loss function."""
        loss_config = self.config.loss
        
        self.loss_function = CombinedLoss(
            eeg_weight=loss_config.eeg_weight,
            spiking_weight=loss_config.spiking_weight,
            biological_weight=loss_config.biological_weight,
            regularization_weight=loss_config.regularization_weight
        ).to(self.device)
    
    def _init_optimizer(self):
        """Initialize optimizer."""
        opt_config = self.config.optimizer
        
        self.optimizer = NGISOptimizer(
            model=self.model,
            learning_rate=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
            optimizer_type=opt_config.type
        )
    
    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        scheduler_config = self.config.scheduler

        self.scheduler = NGISScheduler(
            optimizer=self.optimizer.optimizer,  # Pass the actual PyTorch optimizer
            scheduler_type=scheduler_config.type,
            **scheduler_config.params
        )
    
    def _init_checkpointing(self):
        """Initialize checkpointing."""
        checkpoint_config = self.config.checkpointing
        
        self.checkpoint_manager = CheckpointManager(
            save_dir=checkpoint_config.save_dir,
            save_frequency=checkpoint_config.save_frequency,
            max_checkpoints=checkpoint_config.max_checkpoints
        )
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Load checkpoint if exists
        self._load_checkpoint()
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self._train_epoch()
            
            # Validate
            val_loss = self._validate_epoch()
            
            # Update scheduler
            self.scheduler.step(val_loss)
            
            # Log progress
            self._log_epoch(epoch, train_loss, val_loss)
            
            # Save checkpoint
            if self.rank == 0:  # Only save on main process
                self._save_checkpoint(val_loss)
            
            # Early stopping check
            if self._should_stop_early(val_loss):
                logger.info("Early stopping triggered")
                break
        
        logger.info("Training completed")
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        # Create progress bar
        if self.rank == 0:
            pbar = tqdm(
                desc=f"Epoch {self.current_epoch}",
                total=len(self.train_dataloader),
                leave=False
            )
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(
                eeg_input=batch['eeg'],
                return_spikes=True,
                return_graph=True
            )
            
            # Calculate loss
            loss = self.loss_function(
                real_eeg=batch['eeg'],
                simulated_eeg=outputs['eeg_output'],
                spike_trains=outputs['spike_trains'],
                graph_data=outputs['graph_data']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            if self.rank == 0:
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
                pbar.update()
            
            # Log batch metrics
            if self.global_step % self.config.logging.log_frequency == 0:
                self._log_batch_metrics(loss.item())
        
        if self.rank == 0:
            pbar.close()
        
        return total_loss / num_batches
    
    def _validate_epoch(self) -> float:
        """Validate for one epoch."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                outputs = self.model(
                    eeg_input=batch['eeg'],
                    return_spikes=True,
                    return_graph=True
                )
                
                # Calculate loss
                loss = self.loss_function(
                    real_eeg=batch['eeg'],
                    simulated_eeg=outputs['eeg_output'],
                    spike_trains=outputs['spike_trains'],
                    graph_data=outputs['graph_data']
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float):
        """Log epoch metrics."""
        if self.rank == 0:
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )
    
    def _log_batch_metrics(self, loss: float):
        """Log batch metrics."""
        if self.rank == 0:
            logger.info(f"Step {self.global_step}: Loss: {loss:.4f}")
    
    def _save_checkpoint(self, val_loss: float):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        self.checkpoint_manager.save_checkpoint(checkpoint, val_loss)
        
        # Update best loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss
    
    def _load_checkpoint(self):
        """Load training checkpoint."""
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()
        
        if checkpoint is not None:
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_loss = checkpoint['best_loss']
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def _should_stop_early(self, val_loss: float) -> bool:
        """Check if training should stop early."""
        if hasattr(self.config.training, 'early_stopping_patience'):
            patience = self.config.training.early_stopping_patience
            if val_loss >= self.best_loss:
                self.early_stopping_counter += 1
            else:
                self.early_stopping_counter = 0
            
            return self.early_stopping_counter >= patience
        
        return False
    
    def set_dataloaders(self, train_dataloader: DataLoader, val_dataloader: DataLoader):
        """Set training and validation dataloaders."""
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
    
    def get_model(self) -> nn.Module:
        """Get the trained model."""
        if self.is_distributed:
            return self.model.module
        else:
            return self.model
    
    def get_training_info(self) -> Dict:
        """Get comprehensive training information."""
        return {
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_loss': self.best_loss,
            'device': str(self.device),
            'is_distributed': self.is_distributed,
            'rank': self.rank,
            'world_size': self.world_size,
            'model_info': self.model.get_model_info(),
            'optimizer_info': self.optimizer.get_optimizer_info(),
            'scheduler_info': self.scheduler.get_scheduler_info()
        } 