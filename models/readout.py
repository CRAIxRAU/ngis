"""
EEG readout layer for NGIS.

Converts spike trains from the G-SNN back to EEG signals
for comparison with real EEG data.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EEGReadout(nn.Module):
    """
    EEG readout layer.
    
    Converts spike trains from the G-SNN back to EEG signals
    for comparison with real EEG data.
    """
    
    def __init__(
        self,
        n_neurons: int,
        n_channels: int = 128,
        readout_type: str = "linear",
        activation: str = "tanh",
        hidden_dim: int = 64,
        dropout: float = 0.1,
        learnable_readout: bool = True
    ):
        """
        Initialize EEG readout layer.
        
        Args:
            n_neurons: Number of neurons in the network.
            n_channels: Number of EEG channels to output.
            readout_type: Type of readout ("linear", "mlp", "attention").
            activation: Activation function ("tanh", "relu", "sigmoid").
            hidden_dim: Hidden dimension for MLP readout.
            dropout: Dropout rate.
            learnable_readout: Whether readout parameters are learnable.
        """
        super().__init__()
        
        self.n_neurons = n_neurons
        self.n_channels = n_channels
        self.readout_type = readout_type
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learnable_readout = learnable_readout
        
        # Initialize readout layers
        self._init_readout_layers()
        
        logger.info(f"Initialized EEG readout: {readout_type} type, {n_channels} channels")
    
    def _init_readout_layers(self):
        """Initialize readout layers based on type."""
        if self.readout_type == "linear":
            self._init_linear_readout()
        elif self.readout_type == "mlp":
            self._init_mlp_readout()
        elif self.readout_type == "attention":
            self._init_attention_readout()
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")
    
    def _init_linear_readout(self):
        """Initialize linear readout layer."""
        self.readout_layer = nn.Linear(self.n_neurons, self.n_channels)
        
        if not self.learnable_readout:
            # Initialize with identity-like mapping
            with torch.no_grad():
                # Create a sparse mapping from neurons to channels
                for i in range(self.n_channels):
                    start_neuron = (i * self.n_neurons) // self.n_channels
                    end_neuron = ((i + 1) * self.n_neurons) // self.n_channels
                    self.readout_layer.weight[i, start_neuron:end_neuron] = 1.0
                    self.readout_layer.bias[i] = 0.0
    
    def _init_mlp_readout(self):
        """Initialize MLP readout layer."""
        self.readout_layers = nn.Sequential(
            nn.Linear(self.n_neurons, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.n_channels)
        )
    
    def _init_attention_readout(self):
        """Initialize attention-based readout layer."""
        # Query, key, value projections
        self.query_proj = nn.Linear(self.n_neurons, self.hidden_dim)
        self.key_proj = nn.Linear(self.n_neurons, self.hidden_dim)
        self.value_proj = nn.Linear(self.n_neurons, self.hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.n_channels)
        
        # Attention parameters
        self.attention_heads = 4
        self.attention_dim = self.hidden_dim // self.attention_heads
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.attention_heads,
            dropout=self.dropout,
            batch_first=True
        )
    
    def forward(
        self,
        spike_trains: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through EEG readout layer.
        
        Args:
            spike_trains: Spike trains (batch_size, n_neurons, seq_len).
            return_attention: Whether to return attention weights (for attention readout).
            
        Returns:
            Dictionary containing EEG output and optional attention weights.
        """
        batch_size, n_neurons, seq_len = spike_trains.shape
        
        # Aggregate spike trains over time
        spike_features = self._aggregate_spikes(spike_trains)
        
        # Apply readout
        if self.readout_type == "linear":
            eeg_output = self._linear_readout(spike_features)
        elif self.readout_type == "mlp":
            eeg_output = self._mlp_readout(spike_features)
        elif self.readout_type == "attention":
            eeg_output, attention_weights = self._attention_readout(spike_features)
        else:
            raise ValueError(f"Unknown readout type: {self.readout_type}")
        
        # Apply activation function
        eeg_output = self._apply_activation(eeg_output)
        
        # Prepare output
        output = {'eeg_output': eeg_output}
        
        if self.readout_type == "attention" and return_attention:
            output['attention_weights'] = attention_weights
        
        return output
    
    def _aggregate_spikes(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """
        Aggregate spike trains into features.
        
        Args:
            spike_trains: Spike trains (batch_size, n_neurons, seq_len).
            
        Returns:
            Aggregated spike features (batch_size, n_neurons).
        """
        # Calculate firing rates
        firing_rates = spike_trains.mean(dim=2)  # Average over time
        
        # Alternative: use spike count
        # spike_counts = spike_trains.sum(dim=2)
        
        # Alternative: use temporal features
        # temporal_features = torch.cat([
        #     spike_trains.mean(dim=2),  # Mean firing rate
        #     spike_trains.std(dim=2),   # Firing rate variability
        #     spike_trains.max(dim=2)[0] # Peak firing rate
        # ], dim=1)
        
        return firing_rates
    
    def _linear_readout(self, spike_features: torch.Tensor) -> torch.Tensor:
        """Apply linear readout."""
        return self.readout_layer(spike_features)
    
    def _mlp_readout(self, spike_features: torch.Tensor) -> torch.Tensor:
        """Apply MLP readout."""
        return self.readout_layers(spike_features)
    
    def _attention_readout(
        self,
        spike_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply attention-based readout."""
        batch_size = spike_features.shape[0]
        
        # Create learnable channel embeddings
        channel_embeddings = nn.Parameter(
            torch.randn(self.n_channels, self.hidden_dim)
        ).expand(batch_size, -1, -1)
        
        # Project spike features
        query = self.query_proj(spike_features).unsqueeze(1)  # (batch, 1, hidden_dim)
        key = self.key_proj(spike_features).unsqueeze(1)      # (batch, 1, hidden_dim)
        value = self.value_proj(spike_features).unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Apply multi-head attention
        attended_features, attention_weights = self.multihead_attn(
            query=query,
            key=key,
            value=value
        )
        
        # Project to EEG channels
        eeg_output = self.output_proj(attended_features.squeeze(1))
        
        return eeg_output, attention_weights
    
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function."""
        if self.activation == "tanh":
            return torch.tanh(x)
        elif self.activation == "relu":
            return F.relu(x)
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.activation == "none":
            return x
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def get_readout_weights(self) -> torch.Tensor:
        """Get readout weights."""
        if self.readout_type == "linear":
            return self.readout_layer.weight.clone()
        elif self.readout_type == "mlp":
            # Return weights from first layer
            return self.readout_layers[0].weight.clone()
        else:
            return torch.zeros(self.n_channels, self.n_neurons)
    
    def set_readout_weights(self, weights: torch.Tensor):
        """Set readout weights."""
        if self.readout_type == "linear":
            if weights.shape == self.readout_layer.weight.shape:
                self.readout_layer.weight.data = weights.clone()
            else:
                raise ValueError(f"Weight shape mismatch: {weights.shape}")
        elif self.readout_type == "mlp":
            if weights.shape == self.readout_layers[0].weight.shape:
                self.readout_layers[0].weight.data = weights.clone()
            else:
                raise ValueError(f"Weight shape mismatch: {weights.shape}")
    
    def get_readout_info(self) -> Dict:
        """Get comprehensive readout information."""
        return {
            'n_neurons': self.n_neurons,
            'n_channels': self.n_channels,
            'readout_type': self.readout_type,
            'activation': self.activation,
            'hidden_dim': self.hidden_dim,
            'dropout': self.dropout,
            'learnable_readout': self.learnable_readout
        }


class TemporalEEGReadout(EEGReadout):
    """
    Temporal EEG readout with temporal dynamics.
    
    Extends basic EEG readout with temporal processing
    to capture time-varying EEG patterns.
    """
    
    def __init__(
        self,
        n_neurons: int,
        n_channels: int = 128,
        readout_type: str = "linear",
        activation: str = "tanh",
        hidden_dim: int = 64,
        dropout: float = 0.1,
        learnable_readout: bool = True,
        temporal_window: int = 10,
        temporal_stride: int = 1
    ):
        """
        Initialize temporal EEG readout.
        
        Args:
            n_neurons: Number of neurons in the network.
            n_channels: Number of EEG channels to output.
            readout_type: Type of readout ("linear", "mlp", "attention").
            activation: Activation function ("tanh", "relu", "sigmoid").
            hidden_dim: Hidden dimension for MLP readout.
            dropout: Dropout rate.
            learnable_readout: Whether readout parameters are learnable.
            temporal_window: Size of temporal window for processing.
            temporal_stride: Stride for temporal processing.
        """
        super().__init__(
            n_neurons=n_neurons,
            n_channels=n_channels,
            readout_type=readout_type,
            activation=activation,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learnable_readout=learnable_readout
        )
        
        self.temporal_window = temporal_window
        self.temporal_stride = temporal_stride
        
        # Temporal processing layers
        self.temporal_conv = nn.Conv1d(
            in_channels=n_neurons,
            out_channels=hidden_dim,
            kernel_size=temporal_window,
            stride=temporal_stride,
            padding=temporal_window // 2
        )
        
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
    
    def _aggregate_spikes(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """
        Aggregate spike trains with temporal processing.
        
        Args:
            spike_trains: Spike trains (batch_size, n_neurons, seq_len).
            
        Returns:
            Temporally processed features (batch_size, hidden_dim).
        """
        batch_size, n_neurons, seq_len = spike_trains.shape
        
        # Apply temporal convolution
        temporal_features = self.temporal_conv(spike_trains)  # (batch, hidden_dim, seq_len)
        
        # Global temporal pooling
        pooled_features = self.temporal_pool(temporal_features)  # (batch, hidden_dim, 1)
        
        return pooled_features.squeeze(-1)  # (batch, hidden_dim)
    
    def get_readout_info(self) -> Dict:
        """Get comprehensive readout information including temporal info."""
        info = super().get_readout_info()
        info.update({
            'temporal_window': self.temporal_window,
            'temporal_stride': self.temporal_stride
        })
        return info 