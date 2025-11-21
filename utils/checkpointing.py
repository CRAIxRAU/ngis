"""
Checkpointing utilities for NGIS.

Handles saving and loading model checkpoints during training
with automatic cleanup and versioning.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Optional, Any

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Handles saving, loading, and cleanup of model checkpoints
    with automatic versioning and best model tracking.
    """
    
    def __init__(
        self,
        save_dir: str = "checkpoints",
        save_frequency: int = 10,
        max_checkpoints: int = 5,
        save_best_only: bool = False
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            save_dir: Directory to save checkpoints.
            save_frequency: How often to save checkpoints (epochs).
            max_checkpoints: Maximum number of checkpoints to keep.
            save_best_only: Whether to save only the best model.
        """
        self.save_dir = Path(save_dir)
        self.save_frequency = save_frequency
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Track best model
        self.best_metric = float('inf')
        self.best_checkpoint_path = None
        
        logger.info(f"Initialized checkpoint manager: {save_dir}")
    
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        metric: float,
        epoch: int,
        is_best: bool = False
    ) -> str:
        """
        Save a checkpoint.
        
        Args:
            checkpoint: Checkpoint dictionary.
            metric: Metric value (e.g., validation loss).
            epoch: Current epoch.
            is_best: Whether this is the best model so far.
            
        Returns:
            Path to saved checkpoint.
        """
        # Determine checkpoint filename
        if is_best:
            filename = "best_model.pth"
        else:
            filename = f"checkpoint_epoch_{epoch:03d}.pth"
        
        checkpoint_path = self.save_dir / filename

        # Save checkpoint (convert Path to string for torch.save)
        torch.save(checkpoint, str(checkpoint_path))
        
        # Update best model tracking
        if is_best or metric < self.best_metric:
            self.best_metric = metric
            self.best_checkpoint_path = checkpoint_path

            # Copy to best model (only if not already saved as best_model.pth)
            if not is_best:
                best_link = self.save_dir / "best_model.pth"
                if best_link.exists():
                    best_link.unlink()
                # Use copy instead of symlink for Windows compatibility
                import shutil
                shutil.copy2(str(checkpoint_path), str(best_link))
        
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Load a checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file.
            
        Returns:
            Loaded checkpoint dictionary or None if failed.
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the latest checkpoint.
        
        Returns:
            Latest checkpoint dictionary or None if no checkpoint found.
        """
        # Try to load best model first
        best_path = self.save_dir / "best_model.pth"
        if best_path.exists():
            return self.load_checkpoint(str(best_path))
        
        # Look for latest checkpoint
        checkpoint_files = list(self.save_dir.glob("checkpoint_epoch_*.pth"))
        if not checkpoint_files:
            return None
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        latest_checkpoint = checkpoint_files[-1]
        
        return self.load_checkpoint(str(latest_checkpoint))
    
    def load_best_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Load the best checkpoint.
        
        Returns:
            Best checkpoint dictionary or None if no checkpoint found.
        """
        if self.best_checkpoint_path is None:
            return None
        
        return self.load_checkpoint(str(self.best_checkpoint_path))
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to stay within max_checkpoints limit."""
        if self.save_best_only:
            return
        
        checkpoint_files = list(self.save_dir.glob("checkpoint_epoch_*.pth"))
        
        if len(checkpoint_files) <= self.max_checkpoints:
            return
        
        # Sort by epoch number
        checkpoint_files.sort(key=lambda x: int(x.stem.split('_')[-1]))
        
        # Remove oldest checkpoints
        files_to_remove = checkpoint_files[:-self.max_checkpoints]
        
        for file_path in files_to_remove:
            try:
                file_path.unlink()
                logger.debug(f"Removed old checkpoint: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {file_path}: {e}")
    
    def get_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about available checkpoints.
        
        Returns:
            Dictionary with checkpoint information.
        """
        checkpoint_files = list(self.save_dir.glob("*.pth"))
        
        info = {
            'save_dir': str(self.save_dir),
            'total_checkpoints': len(checkpoint_files),
            'best_metric': self.best_metric,
            'best_checkpoint': str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            'checkpoints': []
        }
        
        for file_path in checkpoint_files:
            try:
                checkpoint = torch.load(file_path, map_location='cpu')
                checkpoint_info = {
                    'path': str(file_path),
                    'epoch': checkpoint.get('epoch', 'unknown'),
                    'metric': checkpoint.get('val_loss', 'unknown'),
                    'size_mb': file_path.stat().st_size / (1024 * 1024)
                }
                info['checkpoints'].append(checkpoint_info)
            except Exception as e:
                logger.warning(f"Failed to load checkpoint info for {file_path}: {e}")
        
        return info
    
    def cleanup_all(self):
        """Remove all checkpoints."""
        checkpoint_files = list(self.save_dir.glob("*.pth"))
        
        for file_path in checkpoint_files:
            try:
                file_path.unlink()
                logger.info(f"Removed checkpoint: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove checkpoint {file_path}: {e}")
    
    def copy_checkpoint(self, source_path: str, dest_path: str):
        """
        Copy a checkpoint to a new location.
        
        Args:
            source_path: Source checkpoint path.
            dest_path: Destination checkpoint path.
        """
        try:
            shutil.copy2(source_path, dest_path)
            logger.info(f"Copied checkpoint: {source_path} -> {dest_path}")
        except Exception as e:
            logger.error(f"Failed to copy checkpoint: {e}")


class ModelCheckpoint:
    """
    Lightweight checkpoint for model state only.
    
    Used for saving/loading just the model state without
    training state information.
    """
    
    def __init__(self, save_dir: str = "models"):
        """
        Initialize model checkpoint.
        
        Args:
            save_dir: Directory to save model checkpoints.
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def save_model(self, model, filename: str):
        """
        Save model state.
        
        Args:
            model: PyTorch model.
            filename: Filename for the model.
        """
        model_path = self.save_dir / filename
        
        # Save model state
        torch.save(model.state_dict(), model_path)
        
        logger.info(f"Saved model: {model_path}")
    
    def load_model(self, model, filename: str):
        """
        Load model state.
        
        Args:
            model: PyTorch model.
            filename: Filename of the model to load.
        """
        model_path = self.save_dir / filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load model state
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        
        logger.info(f"Loaded model: {model_path}")
    
    def list_models(self) -> list:
        """
        List available model files.
        
        Returns:
            List of model filenames.
        """
        model_files = list(self.save_dir.glob("*.pth"))
        return [f.name for f in model_files] 