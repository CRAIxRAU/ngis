"""
Synaptic connection model for NGIS.

Implements synaptic dynamics, plasticity, and weight management
for the graph-structured spiking neural network.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class Synapse(nn.Module):
    """
    Synaptic connection model.
    
    Implements synaptic dynamics, plasticity, and weight management
    for connections between neurons in the G-SNN.
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_s: float = 5.0,  # Synaptic time constant (ms)
        weight_scale: float = 1.0,  # Weight scaling factor
        plasticity: bool = True,  # Enable synaptic plasticity
        learning_rate: float = 0.01,  # Learning rate for plasticity
        weight_decay: float = 0.0,  # Weight decay rate
        max_weight: float = 10.0,  # Maximum synaptic weight
        min_weight: float = 0.0,  # Minimum synaptic weight
        dt: float = 1.0,  # Time step (ms)
        learnable_weights: bool = True
    ):
        """
        Initialize synapse model.
        
        Args:
            n_neurons: Number of neurons.
            tau_s: Synaptic time constant in milliseconds.
            weight_scale: Weight scaling factor.
            plasticity: Whether to enable synaptic plasticity.
            learning_rate: Learning rate for weight updates.
            weight_decay: Weight decay rate.
            max_weight: Maximum synaptic weight.
            min_weight: Minimum synaptic weight.
            dt: Time step in milliseconds.
            learnable_weights: Whether weights are learnable parameters.
        """
        super().__init__()
        
        self.n_neurons = n_neurons
        self.tau_s = tau_s
        self.weight_scale = weight_scale
        self.plasticity = plasticity
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.dt = dt
        
        # Initialize synaptic weights
        if learnable_weights:
            self.weights = nn.Parameter(
                torch.randn(n_neurons, n_neurons) * 0.1
            )
        else:
            self.register_buffer(
                'weights',
                torch.randn(n_neurons, n_neurons) * 0.1
            )
        
        # Initialize synaptic state
        self.reset_state()
        
        logger.info(f"Initialized synapses: {n_neurons}x{n_neurons} connections")
    
    def reset_state(self):
        """Reset synaptic state variables."""
        # Synaptic current
        self.synaptic_current = torch.zeros(self.n_neurons)
        
        # Spike history for plasticity
        self.spike_history = torch.zeros(self.n_neurons)
        
        # Weight update history
        self.weight_updates = torch.zeros(self.n_neurons, self.n_neurons)
    
    def forward(
        self,
        spikes: torch.Tensor,
        return_current: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through synapses.
        
        Args:
            spikes: Input spikes (batch_size, n_neurons).
            return_current: Whether to return synaptic currents.
            
        Returns:
            Tuple of (filtered_spikes, synaptic_currents).
        """
        batch_size = spikes.shape[0]
        
        # Initialize output tensors
        filtered_spikes = torch.zeros_like(spikes)
        synaptic_currents = torch.zeros_like(spikes)
        
        # Process each batch element
        for b in range(batch_size):
            # Update synaptic dynamics
            batch_filtered, batch_current = self._update_synapses(spikes[b])
            
            filtered_spikes[b] = batch_filtered
            synaptic_currents[b] = batch_current
        
        if return_current:
            return filtered_spikes, synaptic_currents
        else:
            return filtered_spikes, None
    
    def _update_synapses(
        self,
        spikes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update synaptic dynamics for a single time step.
        
        Args:
            spikes: Input spikes (n_neurons).
            
        Returns:
            Tuple of (filtered_spikes, synaptic_current).
        """
        # Update synaptic current
        # Synaptic current equation: ds/dt = -s / tau_s + spikes
        # Discretized: s(t+1) = s(t) + dt * (-s(t) / tau_s + spikes)
        
        synaptic_decay = self.dt / self.tau_s
        self.synaptic_current = self.synaptic_current * (1 - synaptic_decay) + spikes
        
        # Apply synaptic weights
        weighted_current = torch.matmul(self.weights, self.synaptic_current)
        
        # Apply weight constraints
        weighted_current = torch.clamp(weighted_current, self.min_weight, self.max_weight)
        
        # Update spike history for plasticity
        if self.plasticity:
            self._update_plasticity(spikes)
        
        return spikes, weighted_current
    
    def _update_plasticity(self, spikes: torch.Tensor):
        """
        Update synaptic weights based on spike-timing dependent plasticity (STDP).
        
        Args:
            spikes: Current spikes (n_neurons).
        """
        # Simple STDP rule: weight change depends on spike timing
        # This is a simplified implementation
        
        # Calculate weight updates based on spike correlations
        spike_correlations = torch.outer(spikes, self.spike_history)
        
        # STDP rule: LTP for positive correlations, LTD for negative
        weight_updates = self.learning_rate * spike_correlations
        
        # Apply weight updates
        self.weights = self.weights + weight_updates
        
        # Apply weight constraints
        self.weights = torch.clamp(self.weights, self.min_weight, self.max_weight)
        
        # Apply weight decay
        if self.weight_decay > 0:
            self.weights = self.weights * (1 - self.weight_decay)
        
        # Update spike history
        self.spike_history = spikes.clone()
        
        # Store weight updates for monitoring
        self.weight_updates = weight_updates
    
    def get_weights(self) -> torch.Tensor:
        """Get current synaptic weights."""
        return self.weights.clone()
    
    def set_weights(self, weights: torch.Tensor):
        """Set synaptic weights."""
        if weights.shape == self.weights.shape:
            self.weights.data = weights.clone()
        else:
            raise ValueError(f"Weight shape mismatch: {weights.shape} vs {self.weights.shape}")
    
    def get_weight_statistics(self) -> Dict[str, float]:
        """Get statistics about synaptic weights."""
        weights = self.weights.detach().cpu().numpy()
        
        return {
            'mean_weight': float(weights.mean()),
            'std_weight': float(weights.std()),
            'min_weight': float(weights.min()),
            'max_weight': float(weights.max()),
            'sparsity': float((weights == 0).sum() / weights.size),
            'total_connections': int(weights.size),
            'active_connections': int((weights != 0).sum())
        }
    
    def get_synapse_info(self) -> Dict:
        """Get comprehensive synapse information."""
        return {
            'n_neurons': self.n_neurons,
            'tau_s': self.tau_s,
            'weight_scale': self.weight_scale,
            'plasticity': self.plasticity,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'max_weight': self.max_weight,
            'min_weight': self.min_weight,
            'dt': self.dt,
            'learnable_weights': isinstance(self.weights, nn.Parameter),
            'weight_statistics': self.get_weight_statistics()
        }


class PlasticSynapse(Synapse):
    """
    Enhanced plastic synapse with more sophisticated plasticity rules.
    
    Extends basic synapse with advanced plasticity mechanisms
    including homeostatic plasticity and metaplasticity.
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_s: float = 5.0,
        weight_scale: float = 1.0,
        plasticity: bool = True,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        max_weight: float = 10.0,
        min_weight: float = 0.0,
        dt: float = 1.0,
        learnable_weights: bool = True,
        homeostatic_plasticity: bool = True,
        target_firing_rate: float = 0.1,
        homeostatic_strength: float = 0.1
    ):
        """
        Initialize plastic synapse.
        
        Args:
            n_neurons: Number of neurons.
            tau_s: Synaptic time constant in milliseconds.
            weight_scale: Weight scaling factor.
            plasticity: Whether to enable synaptic plasticity.
            learning_rate: Learning rate for weight updates.
            weight_decay: Weight decay rate.
            max_weight: Maximum synaptic weight.
            min_weight: Minimum synaptic weight.
            dt: Time step in milliseconds.
            learnable_weights: Whether weights are learnable parameters.
            homeostatic_plasticity: Whether to enable homeostatic plasticity.
            target_firing_rate: Target firing rate for homeostasis.
            homeostatic_strength: Strength of homeostatic plasticity.
        """
        super().__init__(
            n_neurons=n_neurons,
            tau_s=tau_s,
            weight_scale=weight_scale,
            plasticity=plasticity,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_weight=max_weight,
            min_weight=min_weight,
            dt=dt,
            learnable_weights=learnable_weights
        )
        
        self.homeostatic_plasticity = homeostatic_plasticity
        self.target_firing_rate = target_firing_rate
        self.homeostatic_strength = homeostatic_strength
        
        # Homeostatic plasticity state
        self.firing_rate_history = torch.zeros(n_neurons)
        self.homeostatic_scaling = torch.ones(n_neurons)
    
    def reset_state(self):
        """Reset synaptic state variables."""
        super().reset_state()
        self.firing_rate_history = torch.zeros(self.n_neurons)
        self.homeostatic_scaling = torch.ones(self.n_neurons)
    
    def _update_plasticity(self, spikes: torch.Tensor):
        """
        Update synaptic weights with enhanced plasticity rules.
        
        Args:
            spikes: Current spikes (n_neurons).
        """
        # Update firing rate history
        decay_rate = 0.95
        self.firing_rate_history = (
            decay_rate * self.firing_rate_history + 
            (1 - decay_rate) * spikes
        )
        
        # Calculate homeostatic scaling
        if self.homeostatic_plasticity:
            firing_rate_error = self.firing_rate_history - self.target_firing_rate
            self.homeostatic_scaling = torch.clamp(
                1.0 + self.homeostatic_strength * firing_rate_error,
                min=0.1,
                max=2.0
            )
        
        # Calculate STDP weight updates
        spike_correlations = torch.outer(spikes, self.spike_history)
        stdp_updates = self.learning_rate * spike_correlations
        
        # Apply homeostatic scaling to weight updates
        if self.homeostatic_plasticity:
            homeostatic_factor = torch.outer(self.homeostatic_scaling, self.homeostatic_scaling)
            stdp_updates = stdp_updates * homeostatic_factor
        
        # Apply weight updates
        self.weights = self.weights + stdp_updates
        
        # Apply weight constraints
        self.weights = torch.clamp(self.weights, self.min_weight, self.max_weight)
        
        # Apply weight decay
        if self.weight_decay > 0:
            self.weights = self.weights * (1 - self.weight_decay)
        
        # Update spike history
        self.spike_history = spikes.clone()
        
        # Store weight updates for monitoring
        self.weight_updates = stdp_updates
    
    def get_synapse_info(self) -> Dict:
        """Get comprehensive synapse information including plasticity info."""
        info = super().get_synapse_info()
        info.update({
            'homeostatic_plasticity': self.homeostatic_plasticity,
            'target_firing_rate': self.target_firing_rate,
            'homeostatic_strength': self.homeostatic_strength,
            'mean_firing_rate': float(self.firing_rate_history.mean()),
            'mean_homeostatic_scaling': float(self.homeostatic_scaling.mean())
        })
        return info 