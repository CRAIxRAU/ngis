"""
Learning rate scheduler for NGIS.

Implements learning rate scheduling with support for different
scheduling strategies.
"""

import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, Any


class NGISScheduler:
    """Learning rate scheduler for NGIS training."""
    
    def __init__(
        self,
        optimizer,
        scheduler_type: str = "reduce_lr_on_plateau",
        **kwargs
    ):
        """
        Initialize scheduler.
        
        Args:
            optimizer: PyTorch optimizer.
            scheduler_type: Type of scheduler.
            **kwargs: Additional scheduler parameters.
        """
        self.scheduler_type = scheduler_type
        self.scheduler = self._create_scheduler(optimizer, **kwargs)
    
    def _create_scheduler(self, optimizer, **kwargs):
        """Create scheduler based on type."""
        if self.scheduler_type == "reduce_lr_on_plateau":
            # Set defaults, allow kwargs to override
            params = {
                'mode': 'min',
                'factor': 0.5,
                'patience': 5,
                'min_lr': 1e-6
            }
            params.update(kwargs)
            return lr_scheduler.ReduceLROnPlateau(optimizer, **params)
        elif self.scheduler_type == "cosine":
            return lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100),
                eta_min=kwargs.get('eta_min', 1e-6)
            )
        elif self.scheduler_type == "step":
            return lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif self.scheduler_type == "exponential":
            return lr_scheduler.ExponentialLR(
                optimizer,
                gamma=kwargs.get('gamma', 0.95)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
    
    def step(self, metrics=None):
        """Step the scheduler."""
        if self.scheduler_type == "reduce_lr_on_plateau":
            self.scheduler.step(metrics)
        else:
            self.scheduler.step()
    
    def get_last_lr(self):
        """Get current learning rates."""
        return self.scheduler.get_last_lr()
    
    def state_dict(self):
        """Get scheduler state."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load scheduler state."""
        self.scheduler.load_state_dict(state_dict)
    
    def get_scheduler_info(self) -> Dict[str, Any]:
        """Get scheduler information."""
        return {
            'type': self.scheduler_type,
            'current_lr': self.get_last_lr()
        } 