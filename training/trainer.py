"""
Main trainer for NGIS.

Handles training loop, distributed training, and model optimization
for the graph-structured spiking neural network.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.nvtx import range_push, range_pop
from tqdm import tqdm

from models.gsnn import GSNN
from training.loss_functions import CombinedLoss
from training.optimizer import NGISOptimizer
from training.scheduler import NGISScheduler
from utils.checkpointing import CheckpointManager
from utils.logging import setup_logging
from utils.metrics import compute_all_metrics, format_metrics_for_logging

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
        world_size: int = 1,
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
        # Set device to specific GPU based on rank
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")

        # Initialize components
        self._init_model()
        self._init_loss_function()
        self._init_optimizer()
        self._init_scheduler()
        self._init_checkpointing()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.early_stopping_counter = 0

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
            dropout=model_config.dropout,
        ).to(self.device)

        # Wrap with DDP if distributed
        if self.is_distributed:
            # Use current device instead of rank for device_ids
            current_device = torch.cuda.current_device()
            self.model = DDP(
                self.model,
                device_ids=[current_device],
                output_device=current_device,
                find_unused_parameters=True,
            )

        # Access the underlying module if wrapped in DDP
        model_to_count = self.model.module if hasattr(self.model, 'module') else self.model
        logger.info(
            f"Initialized model with {model_to_count.count_parameters()['trainable_parameters']} parameters"
        )

    def _init_loss_function(self):
        """Initialize loss function."""
        loss_config = self.config.loss

        self.loss_function = CombinedLoss(
            eeg_weight=loss_config.eeg_weight,
            spiking_weight=loss_config.spiking_weight,
            biological_weight=loss_config.biological_weight,
            regularization_weight=loss_config.regularization_weight,
        ).to(self.device)

    def _init_optimizer(self):
        """Initialize optimizer."""
        opt_config = self.config.optimizer

        self.optimizer = NGISOptimizer(
            model=self.model,
            learning_rate=opt_config.learning_rate,
            weight_decay=opt_config.weight_decay,
            optimizer_type=opt_config.type,
        )

    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        scheduler_config = self.config.scheduler

        self.scheduler = NGISScheduler(
            optimizer=self.optimizer.optimizer,  # Pass the actual PyTorch optimizer
            scheduler_type=scheduler_config.type,
            **scheduler_config.params,
        )

    def _init_checkpointing(self):
        """Initialize checkpointing."""
        checkpoint_config = self.config.checkpointing

        self.checkpoint_manager = CheckpointManager(
            save_dir=checkpoint_config.save_dir,
            save_frequency=checkpoint_config.save_frequency,
            max_checkpoints=checkpoint_config.max_checkpoints,
        )

    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        range_push("Training Loop start")
        # Load checkpoint if exists
        self._load_checkpoint()

        # Training loop
        for epoch in range(self.current_epoch, self.config.training.epochs):
            self.current_epoch = epoch

            # Train for one epoch
            train_loss = self._train_epoch()

            # Validate and compute metrics
            val_loss, val_metrics = self._validate_epoch()
            
            # CRITICAL: Synchronize validation loss across all ranks
            # All ranks must see the same loss for scheduler/early stopping consistency
            if self.is_distributed and dist.is_initialized():
                loss_tensor = torch.tensor([val_loss], device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                val_loss = loss_tensor.item()

            # Update scheduler (all ranks must see same loss)
            self.scheduler.step(val_loss)

            # Log progress with metrics
            self._log_epoch(epoch, train_loss, val_loss, val_metrics)

            # Save checkpoint
            if self.rank == 0:  # Only save on main process
                self._save_checkpoint(val_loss)

            # Early stopping check (must be synchronized across all ranks)
            should_stop = self._should_stop_early(val_loss)
            
            # CRITICAL: Synchronize early stopping decision across all ranks
            # If we don't sync, ranks may disagree and cause NCCL hangs
            if self.is_distributed and dist.is_initialized():
                # Convert to tensor for all_reduce
                stop_tensor = torch.tensor([1 if should_stop else 0], device=self.device)
                # All ranks get the maximum (if any rank wants to stop, all stop)
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                should_stop = stop_tensor.item() == 1
            
            if should_stop:
                if self.rank == 0:
                    logger.info("Early stopping triggered")
                break
        range_pop()  # End of training loop

        logger.info("Training completed")

    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()

        total_loss = 0.0
        num_batches = 0
        range_push(f"Train Epoch {self.current_epoch} start")
        # Create progress bar
        if self.rank == 0:
            pbar = tqdm(
                desc=f"Epoch {self.current_epoch}",
                total=len(self.train_dataloader),
                leave=False,
            )

        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            range_push(f"Batch {batch_idx} processing")
            logger.debug(f"Processing batch {batch_idx}")
            batch = self._move_batch_to_device(batch)
            logger.debug(f"Batch moved to device, shape: {batch['eeg'].shape}")
            range_pop()  # End of batch processing
            # Forward pass
            logger.debug("Starting forward pass")
            range_push(f"Batch {batch_idx} forward pass")
            outputs = self.model(
                eeg_input=batch["eeg"], return_spikes=True, return_graph=True
            )
            logger.debug("Forward pass completed")
            range_pop()  # End of forward pass

            # Sanity check model outputs
            simulated_eeg = outputs["eeg_output"]
            spike_trains = outputs["spike_trains"]

            # Check for NaN/Inf in outputs
            if torch.isnan(simulated_eeg).any() or torch.isinf(simulated_eeg).any():
                logger.error(f"🔴 Model output contains NaN/Inf! Batch {batch_idx}")
                logger.error(f"   Output range: [{simulated_eeg.min():.4f}, {simulated_eeg.max():.4f}]")
                logger.error(f"   Spike rate: {spike_trains.mean().item():.4f}")
                continue  # Skip this batch

            # Warn on extreme outputs
            if simulated_eeg.abs().max() > 10.0:
                logger.warning(f"⚠️  Large model output: max={simulated_eeg.max():.4f}, min={simulated_eeg.min():.4f}")

            # Calculate loss
            logger.debug("Calculating loss")
            range_push(f"Batch {batch_idx} loss calculation")
            loss = self.loss_function(
                real_eeg=batch["eeg"],
                simulated_eeg=simulated_eeg,
                spike_trains=spike_trains,
                graph_data=outputs["graph_data"],
            )

            # FIXED: Clamp loss to prevent training instability
            # If loss > 100, likely numerical issue - clamp and warn
            if loss.item() > 100.0:
                logger.warning(f"⚠️  Extreme loss detected: {loss.item():.2f}, clamping to 100.0")
                loss = torch.clamp(loss, max=100.0)

            logger.debug(f"Loss calculated: {loss.item()}")
            range_pop()  # End of loss calculation
            # Backward pass
            range_push(f"Batch {batch_idx} backward pass")
            self.optimizer.zero_grad()
            loss.backward()

            # Check for NaN/extreme gradients BEFORE clipping
            max_grad = 0.0
            nan_grads = False
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    max_grad = max(max_grad, grad_norm)
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        logger.error(f"🔴 NaN/Inf gradient in {name}! grad_norm: {grad_norm:.4f}")
                        nan_grads = True

            # Warn on extreme gradients
            if max_grad > 10.0:
                logger.warning(f"🔶 Large gradient detected: {max_grad:.2f} (will be clipped to 1.0)")
            if nan_grads:
                logger.error(f"🔴 NaN gradients detected! Skipping optimizer step.")
                continue  # Skip this batch

            # Gradient clipping
            grad_norm_before = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0
            )
            if grad_norm_before > 5.0:
                logger.warning(f"⚠️  Clipped gradient norm: {grad_norm_before:.2f} → 1.0")

            # Optimizer step
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            range_pop()  # End of backward pass
            # Update progress bar
            if self.rank == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                pbar.update()

            # Log batch metrics
            if self.global_step % self.config.logging.log_frequency == 0:
                self._log_batch_metrics(loss.item())
        
        if self.rank == 0:
            pbar.close()

        # Safety guard: handle case where all batches were skipped
        if num_batches == 0:
            logger.error("All training batches were skipped! This indicates a critical issue.")
            logger.error("Possible causes: NaN gradients in all batches, data loading failure, etc.")
            raise RuntimeError("Training epoch failed: all batches were skipped (num_batches=0)")

        return float(total_loss / num_batches)

    def _validate_epoch(self) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch and compute comprehensive metrics."""
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        
        # Accumulate metrics across batches
        accumulated_metrics = {}

        # Create progress bar for validation
        if self.rank == 0:
            pbar = tqdm(
                desc=f"Validation {self.current_epoch}",
                total=len(self.val_dataloader),
                leave=False,
            )

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_dataloader):
                # Move batch to device
                range_push(f"Val Batch {batch_idx} processing")
                batch = self._move_batch_to_device(batch)
                range_pop()  # End of batch processing

                
                
                # Forward pass
                range_push(f"Val Batch {batch_idx} forward pass")
                outputs = self.model(
                    eeg_input=batch["eeg"],
                    return_spikes=True,
                    return_graph=True,
                )
                range_pop()  # End of forward pass
                
                # Calculate loss
                range_push(f"Val Batch {batch_idx} loss calculation")
                loss = self.loss_function(
                    real_eeg=batch["eeg"],
                    simulated_eeg=outputs["eeg_output"],
                    spike_trains=outputs["spike_trains"],
                    graph_data=outputs["graph_data"],
                )
                range_pop()  # End of loss calculation

                # Compute validation metrics
                range_push(f"Val Batch {batch_idx} metrics computation")
                batch_metrics = compute_all_metrics(
                    real_eeg=batch["eeg"],
                    simulated_eeg=outputs["eeg_output"],
                    spike_trains=outputs["spike_trains"],
                    graph_data=outputs["graph_data"]
                )
                
                # Accumulate metrics
                for key, value in batch_metrics.items():
                    if key not in accumulated_metrics:
                        accumulated_metrics[key] = []
                    accumulated_metrics[key].append(value)
                range_pop()  # End of metrics computation

                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                if self.rank == 0:
                    pbar.update(1)
                    pbar.set_postfix({
                        "val_loss": loss.item(),
                        "corr": batch_metrics.get('mean_correlation', 0.0)
                    })
        
        # Close progress bar
        if self.rank == 0:
            pbar.close()

        # Safety guard: handle case where all batches were skipped
        if num_batches == 0:
            logger.error("All validation batches were skipped! This indicates a critical issue.")
            logger.error("Possible causes: NaN gradients in all batches, data loading failure, etc.")
            raise RuntimeError("Validation epoch failed: all batches were skipped (num_batches=0)")

        # Average accumulated metrics
        avg_metrics = {k: float(np.mean(v)) for k, v in accumulated_metrics.items()}

        return float(total_loss / num_batches), avg_metrics

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float, 
                   val_metrics: Optional[Dict[str, float]] = None):
        """Log epoch metrics including validation metrics."""
        if self.rank == 0:
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )
            
            # Log detailed validation metrics
            if val_metrics:
                logger.info("Validation Metrics:")
                logger.info(format_metrics_for_logging(val_metrics, prefix="  "))

    def _log_batch_metrics(self, loss: float):
        """Log batch metrics."""
        if self.rank == 0:
            logger.info(f"Step {self.global_step}: Loss: {loss:.4f}")

    def _save_checkpoint(self, val_loss: float):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_loss": val_loss,
            "best_loss": self.best_loss,
            "config": self.config,
        }

        # Check if this is the best model
        is_best = val_loss < self.best_loss

        self.checkpoint_manager.save_checkpoint(
            checkpoint=checkpoint,
            metric=val_loss,
            epoch=self.current_epoch,
            is_best=is_best,
        )

        # Update best loss
        if val_loss < self.best_loss:
            self.best_loss = val_loss

    def _load_checkpoint(self):
        """Load training checkpoint."""
        checkpoint = self.checkpoint_manager.load_latest_checkpoint()

        if checkpoint is not None:
            self.current_epoch = checkpoint["epoch"]
            self.global_step = checkpoint["global_step"]
            self.best_loss = float(checkpoint["best_loss"])

            # Load model state
            self.model.load_state_dict(checkpoint["model_state_dict"])

            # Load optimizer state
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load scheduler state
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def _should_stop_early(self, val_loss: float) -> bool:
        """Check if training should stop early."""
        if hasattr(self.config.training, "early_stopping_patience"):
            patience = self.config.training.early_stopping_patience
            if val_loss >= self.best_loss:
                self.early_stopping_counter += 1
            else:
                self.early_stopping_counter = 0

            return self.early_stopping_counter >= patience

        return False

    def set_dataloaders(
        self, train_dataloader: DataLoader, val_dataloader: DataLoader
    ):
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
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
            "device": str(self.device),
            "is_distributed": self.is_distributed,
            "rank": self.rank,
            "world_size": self.world_size,
            "model_info": self.model.get_model_info(),
            "optimizer_info": self.optimizer.get_optimizer_info(),
            "scheduler_info": self.scheduler.get_scheduler_info(),
        }
