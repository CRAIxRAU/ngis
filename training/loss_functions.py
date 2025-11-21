"""
Loss functions for NGIS.

Implements loss functions for EEG reconstruction, spiking dynamics,
and biological constraints for the G-SNN training.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


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
            firing_rates = spike_trains.mean(dim=-1)  # (batch, neurons)
            target_rate = 0.1  # 10% firing rate

            # FIXED: Use mean over neurons to prevent explosion with large neuron counts
            # Old: (256 neurons) × (0.9)² = ~200 loss when all spike
            # New: mean((0.9)²) = 0.81 loss regardless of neuron count
            loss = F.mse_loss(firing_rates, torch.full_like(firing_rates, target_rate), reduction='mean')

            return loss
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")


class CorrelationLoss(nn.Module):
    """Loss function for temporal correlation between predicted and target EEG."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate correlation loss (1 - mean correlation).

        Args:
            pred: Predicted EEG (batch, channels, time)
            target: Target EEG (batch, channels, time)

        Returns:
            Correlation loss (lower is better)
        """
        # Debug: Check for NaN/Inf in inputs
        if torch.isnan(pred).any() or torch.isinf(pred).any():
            logger.error(f"🔴 CorrelationLoss: pred contains NaN/Inf! pred range: [{pred.min():.4f}, {pred.max():.4f}]")
        if torch.isnan(target).any() or torch.isinf(target).any():
            logger.error(f"🔴 CorrelationLoss: target contains NaN/Inf! target range: [{target.min():.4f}, {target.max():.4f}]")

        # Center the signals (remove mean)
        pred_centered = pred - pred.mean(dim=-1, keepdim=True)
        target_centered = target - target.mean(dim=-1, keepdim=True)

        # Calculate correlation
        numerator = (pred_centered * target_centered).sum(dim=-1)
        pred_std = torch.sqrt((pred_centered ** 2).sum(dim=-1))
        target_std = torch.sqrt((target_centered ** 2).sum(dim=-1))

        # Debug: Check for zero std (constant signals)
        zero_pred_std = (pred_std < 1e-8).sum().item()
        zero_target_std = (target_std < 1e-8).sum().item()
        if zero_pred_std > 0:
            logger.warning(f"⚠️  CorrelationLoss: {zero_pred_std} predictions have zero std (constant output)")
        if zero_target_std > 0:
            logger.warning(f"⚠️  CorrelationLoss: {zero_target_std} targets have zero std (constant input)")

        # Add eps to denominator for numerical stability
        correlation = numerator / (pred_std * target_std + 1e-8)

        # Clamp correlation to valid range [-1, 1] to prevent numerical issues
        correlation = torch.clamp(correlation, -1.0, 1.0)

        # Calculate loss
        loss = 1.0 - correlation.mean()

        # Debug: Check for NaN/extreme loss
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(f"🔴 CorrelationLoss: Loss is NaN/Inf! correlation range: [{correlation.min():.4f}, {correlation.max():.4f}]")
            logger.error(f"   pred_std range: [{pred_std.min():.4f}, {pred_std.max():.4f}]")
            logger.error(f"   target_std range: [{target_std.min():.4f}, {target_std.max():.4f}]")
            return torch.tensor(1.0, device=pred.device, requires_grad=True)  # Return safe fallback

        if loss.item() > 2.0:
            logger.warning(f"⚠️  CorrelationLoss: High loss {loss.item():.4f} (correlation mean: {correlation.mean().item():.4f})")

        # Return 1 - mean correlation (so minimizing loss maximizes correlation)
        return loss


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
        correlation_weight: float = 1.0,
        spiking_weight: float = 0.1,
        biological_weight: float = 0.01,
        regularization_weight: float = 0.001
    ):
        super().__init__()
        self.eeg_weight = eeg_weight
        self.correlation_weight = correlation_weight
        self.spiking_weight = spiking_weight
        self.biological_weight = biological_weight
        self.regularization_weight = regularization_weight

        self.eeg_loss = EEGLoss()
        self.correlation_loss = CorrelationLoss()
        self.spiking_loss = SpikingLoss()
        self.biological_loss = BiologicalLoss()

        # Diagnostic counters
        self.step_count = 0
        self.log_every = 100  # Log every N steps
    
    def forward(
        self,
        real_eeg: torch.Tensor,
        simulated_eeg: torch.Tensor,
        spike_trains: torch.Tensor,
        graph_data,
        model = None
    ) -> torch.Tensor:
        """Calculate combined loss."""
        # EEG reconstruction loss (MSE)
        eeg_loss = self.eeg_loss(real_eeg, simulated_eeg)

        # Correlation loss (temporal similarity)
        correlation_loss = self.correlation_loss(simulated_eeg, real_eeg)

        # Spiking dynamics loss
        spiking_loss = self.spiking_loss(spike_trains)

        # Biological constraints loss
        biological_loss = self.biological_loss(model) if model is not None else torch.tensor(0.0, device=real_eeg.device)

        # Regularization loss
        regularization_loss = self._calculate_regularization(model, real_eeg.device)

        # Combined loss
        total_loss = (
            self.eeg_weight * eeg_loss +
            self.correlation_weight * correlation_loss +
            self.spiking_weight * spiking_loss +
            self.biological_weight * biological_loss +
            self.regularization_weight * regularization_loss
        )

        # Diagnostic logging every N steps
        self.step_count += 1

        # Check for NaN in total loss
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.error(f"🔴 TOTAL LOSS IS NaN/Inf!")
            logger.error(f"   EEG loss: {eeg_loss.item():.4f}")
            logger.error(f"   Correlation loss: {correlation_loss.item():.4f}")
            logger.error(f"   Spiking loss: {spiking_loss.item():.4f}")
            logger.error(f"   Bio loss: {biological_loss.item():.4f}")
            logger.error(f"   Reg loss: {regularization_loss.item():.4f}")
            logger.error(f"   Simulated EEG range: [{simulated_eeg.min():.4f}, {simulated_eeg.max():.4f}]")
            logger.error(f"   Real EEG range: [{real_eeg.min():.4f}, {real_eeg.max():.4f}]")
            logger.error(f"   Spike rate: {spike_trains.mean().item():.4f}")

        # Warn on extreme loss values
        if total_loss.item() > 5.0:
            spike_rate = spike_trains.mean().item()
            logger.warning(
                f"🔶 HIGH LOSS DETECTED: {total_loss.item():.2f} | "
                f"EEG={eeg_loss.item():.4f}, "
                f"Corr={correlation_loss.item():.4f}, "
                f"Spiking={spiking_loss.item():.4f}, "
                f"Reg={regularization_loss.item():.4f}, "
                f"Spikes={spike_rate:.4f}"
            )

        if self.step_count % self.log_every == 0:
            # Calculate spike statistics
            spike_rate = spike_trains.mean().item()
            spike_std = spike_trains.std().item()

            logger.debug(
                f"Loss components: "
                f"EEG={eeg_loss.item():.4f}, "
                f"Corr={correlation_loss.item():.4f}, "
                f"Spiking={spiking_loss.item():.4f}, "
                f"Bio={biological_loss.item():.4f}, "
                f"Reg={regularization_loss.item():.4f}, "
                f"Total={total_loss.item():.4f} | "
                f"Spike rate: {spike_rate:.3f}±{spike_std:.3f}"
            )
            if spike_rate < 0.001:
                logger.warning(f"⚠️  Neurons not spiking (rate={spike_rate:.4f})")
            if spike_rate > 0.5:
                logger.warning(f"⚠️  Excessive spiking (rate={spike_rate:.4f})")

        return total_loss
    
    def _calculate_regularization(self, model, device: torch.device) -> torch.Tensor:
        """Calculate regularization loss (proper L2)."""
        if model is None:
            return torch.tensor(0.0, device=device)

        # FIXED: Proper L2 regularization = sum of squared parameters
        # Old: sum of L2 norms (unbounded, wrong)
        # New: sum of squared weights (standard L2 regularization)
        l2_loss = 0.0
        for param in model.parameters():
            if param.requires_grad:
                l2_loss += param.pow(2).sum()

        return l2_loss 
