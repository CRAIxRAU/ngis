"""
Loss functions for NGIS.

Implements loss functions for EEG reconstruction, spiking dynamics,
and biological constraints for the G-SNN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGLoss(nn.Module):
    """Loss function for EEG reconstruction."""
    
    def __init__(self, loss_type: str = "mse"):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, real_eeg: torch.Tensor, simulated_eeg: torch.Tensor) -> torch.Tensor:
        """Calculate EEG reconstruction loss."""
        if self.loss_type == "mse":
            return F.mse_loss(simulated_eeg, real_eeg)
        elif self.loss_type == "mae":
            return F.l1_loss(simulated_eeg, real_eeg)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class SpikingLoss(nn.Module):
    """Loss function for spiking dynamics."""
    
    def __init__(self, loss_type: str = "rate"):
        super().__init__()
        self.loss_type = loss_type
    
    def forward(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """Calculate spiking loss."""
        if self.loss_type == "rate":
            # Encourage reasonable firing rates
            firing_rates = spike_trains.mean(dim=-1)
            target_rate = 0.1  # 10% firing rate
            return F.mse_loss(firing_rates, torch.full_like(firing_rates, target_rate))
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class BiologicalLoss(nn.Module):
    """Loss function for biological constraints."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, model) -> torch.Tensor:
        """Calculate biological constraint loss."""
        # Placeholder for biological constraints
        return torch.tensor(0.0, device=next(model.parameters()).device)


class CombinedLoss(nn.Module):
    """Combined loss function for NGIS training."""
    
    def __init__(
        self,
        eeg_weight: float = 1.0,
        spiking_weight: float = 0.1,
        biological_weight: float = 0.01,
        regularization_weight: float = 0.001
    ):
        super().__init__()
        self.eeg_weight = eeg_weight
        self.spiking_weight = spiking_weight
        self.biological_weight = biological_weight
        self.regularization_weight = regularization_weight
        
        self.eeg_loss = EEGLoss()
        self.spiking_loss = SpikingLoss()
        self.biological_loss = BiologicalLoss()
    
    def forward(
        self,
        real_eeg: torch.Tensor,
        simulated_eeg: torch.Tensor,
        spike_trains: torch.Tensor,
        graph_data,
        model = None
    ) -> torch.Tensor:
        """Calculate combined loss."""
        # EEG reconstruction loss
        eeg_loss = self.eeg_loss(real_eeg, simulated_eeg)
        
        # Spiking dynamics loss
        spiking_loss = self.spiking_loss(spike_trains)
        
        # Biological constraints loss
        biological_loss = self.biological_loss(model) if model is not None else torch.tensor(0.0, device=real_eeg.device)
        
        # Regularization loss
        regularization_loss = self._calculate_regularization(model, real_eeg.device)
        
        # Combined loss
        total_loss = (
            self.eeg_weight * eeg_loss +
            self.spiking_weight * spiking_loss +
            self.biological_weight * biological_loss +
            self.regularization_weight * regularization_loss
        )
        
        return total_loss
    
    def _calculate_regularization(self, model, device: torch.device) -> torch.Tensor:
        """Calculate regularization loss."""
        if model is None:
            return torch.tensor(0.0, device=device)
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += torch.norm(param, p=2)
        return l2_loss 
