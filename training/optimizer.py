"""
Optimizer for NGIS.

Implements custom optimizer with support for different optimization
algorithms and learning rate scheduling.
"""

import torch
import torch.optim as optim
from typing import Dict, Any


class NGISOptimizer:
    """Custom optimizer for NGIS training."""
    
    def __init__(
        self,
        model,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_type: str = "adam",
        **kwargs
    ):
        """
        Initialize optimizer.
        
        Args:
            model: PyTorch model.
            learning_rate: Learning rate.
            weight_decay: Weight decay.
            optimizer_type: Type of optimizer.
            **kwargs: Additional optimizer parameters.
        """
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_type = optimizer_type
        
        # Create optimizer
        self.optimizer = self._create_optimizer(model, **kwargs)
    
    def _create_optimizer(self, model, **kwargs):
        """Create optimizer based on type."""
        if self.optimizer_type == "adam":
            return optim.Adam(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                **kwargs
            )
        elif self.optimizer_type == "sgd":
            return optim.SGD(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                **kwargs
            )
        elif self.optimizer_type == "adamw":
            return optim.AdamW(
                model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
    
    def zero_grad(self):
        """Zero gradients."""
        self.optimizer.zero_grad()
    
    def step(self):
        """Perform optimization step."""
        self.optimizer.step()
    
    def state_dict(self):
        """Get optimizer state."""
        return self.optimizer.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load optimizer state."""
        self.optimizer.load_state_dict(state_dict)
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """Get optimizer information."""
        return {
            'type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'param_groups': len(self.optimizer.param_groups)
        } 