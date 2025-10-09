"""
Simple EEG reconstruction model for testing training pipeline.
Bypasses complex graph/spiking architecture to verify training works.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEEGModel(nn.Module):
    """
    Simple CNN-based EEG reconstruction model.
    Used for testing training pipeline without graph/spiking complexity.
    """

    def __init__(
        self,
        n_channels: int = 128,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize simple EEG model.

        Args:
            n_channels: Number of EEG channels.
            hidden_dim: Hidden dimension size.
            dropout: Dropout probability.
        """
        super().__init__()

        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Encoder: EEG -> latent representation
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder: latent -> EEG reconstruction
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, n_channels, kernel_size=3, padding=1)
        )

    def forward(
        self,
        eeg_input: torch.Tensor,
        return_spikes: bool = False,
        return_graph: bool = False
    ):
        """
        Forward pass.

        Args:
            eeg_input: Input EEG (batch, channels, time).
            return_spikes: Whether to return spike trains (ignored, for compatibility).
            return_graph: Whether to return graph data (ignored, for compatibility).

        Returns:
            Dictionary with 'eeg_output' and dummy 'spike_trains', 'graph_data'.
        """
        # Encode
        latent = self.encoder(eeg_input)

        # Decode
        eeg_output = self.decoder(latent)

        # Create dummy outputs for compatibility with loss function
        batch_size, n_channels, seq_len = eeg_input.shape

        # Dummy spike trains (zeros)
        spike_trains = torch.zeros(batch_size, 256, seq_len, device=eeg_input.device)

        # Dummy graph data (None, loss function should handle this)
        graph_data = None

        return {
            'eeg_output': eeg_output,
            'spike_trains': spike_trains,
            'graph_data': graph_data
        }

    def reset_state(self):
        """Reset model state (no-op for this model)."""
        pass

    def count_parameters(self):
        """Count model parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {
            'trainable_parameters': trainable,
            'total_parameters': total
        }

    def get_model_info(self):
        """Get model information."""
        return {
            'type': 'SimpleEEGModel',
            'n_channels': self.n_channels,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'parameters': self.count_parameters()
        }
