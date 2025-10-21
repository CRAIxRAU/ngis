"""
LIF (Leaky Integrate-and-Fire) neuron model for NGIS.

Implements biological neuron dynamics using LIF model with
configurable parameters and state management.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron model.
    
    Implements biological neuron dynamics with configurable parameters
    for membrane potential, spike threshold, and refractory period.
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_m: float = 20.0,  # Membrane time constant (ms)
        v_rest: float = -65.0,  # Resting potential (mV)
        v_threshold: float = -55.0,  # Spike threshold (mV)
        v_reset: float = -65.0,  # Reset potential (mV)
        refractory_period: float = 2.0,  # Refractory period (ms)
        dt: float = 1.0,  # Time step (ms)
        noise_std: float = 0.0,  # Noise standard deviation
        learnable_params: bool = False
    ):
        """
        Initialize LIF neuron model.
        
        Args:
            n_neurons: Number of neurons.
            tau_m: Membrane time constant in milliseconds.
            v_rest: Resting membrane potential in millivolts.
            v_threshold: Spike threshold in millivolts.
            v_reset: Reset potential in millivolts.
            refractory_period: Refractory period in milliseconds.
            dt: Time step in milliseconds.
            noise_std: Standard deviation of noise.
            learnable_params: Whether neuron parameters are learnable.
        """
        super().__init__()
        
        self.n_neurons = n_neurons
        self.dt = dt
        self.noise_std = noise_std
        
        # Initialize parameters
        if learnable_params:
            self.tau_m = nn.Parameter(torch.tensor(tau_m))
            self.v_rest = nn.Parameter(torch.tensor(v_rest))
            self.v_threshold = nn.Parameter(torch.tensor(v_threshold))
            self.v_reset = nn.Parameter(torch.tensor(v_reset))
            self.refractory_period = nn.Parameter(torch.tensor(refractory_period))
        else:
            self.register_buffer('tau_m', torch.tensor(tau_m))
            self.register_buffer('v_rest', torch.tensor(v_rest))
            self.register_buffer('v_threshold', torch.tensor(v_threshold))
            self.register_buffer('v_reset', torch.tensor(v_reset))
            self.register_buffer('refractory_period', torch.tensor(refractory_period))
        
        # Initialize state variables
        self.reset_state()
        
        logger.info(f"Initialized LIF neurons: {n_neurons} neurons")
    
    def reset_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ):
        """Reset neuron state variables."""
        target_device = device or self.v_rest.device

        # Expand scalar buffers to match the working batch
        v_rest = self.v_rest.detach().clone().to(target_device)
        self.v = v_rest.expand(batch_size, self.n_neurons).clone()

        self.refractory_counter = torch.zeros(
            batch_size, self.n_neurons, device=target_device
        )

        self.last_spike_time = torch.full(
            (batch_size, self.n_neurons),
            -float("inf"),
            device=target_device,
        )
    
    def forward(
        self,
        input_current: torch.Tensor,
        return_membrane: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through LIF neurons.
        
        Args:
            input_current: Input current to neurons (batch_size, n_neurons).
            return_membrane: Whether to return membrane potentials.
            
        Returns:
            Tuple of (spikes, membrane_potentials).
        """
        if input_current.dim() != 2 or input_current.size(1) != self.n_neurons:
            raise ValueError(
                "Expected input_current with shape (batch_size, n_neurons)"
            )

        batch_size = input_current.shape[0]

        # Lazily resize the internal state if the batch changes
        if not hasattr(self, "v") or self.v.size(0) != batch_size:
            self.reset_state(batch_size=batch_size, device=input_current.device)

        spikes, membrane_potentials = self._update_neurons(input_current)

        if return_membrane:
            return spikes, membrane_potentials
        else:
            return spikes, None
    
    def _update_neurons(
        self,
        input_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update neuron states for a single time step.

        Args:
            input_current: Input current to neurons (batch_size, n_neurons).

        Returns:
            Tuple of (spikes, membrane_potential).
        """
        if input_current.dim() != 2:
            raise ValueError("Expected batched input_current")

        # Add noise if specified
        if self.noise_std > 0:
            noise = torch.randn_like(input_current) * self.noise_std
            input_current = input_current + noise

        # Update refractory counter for all neurons in parallel
        self.refractory_counter = torch.clamp(
            self.refractory_counter - self.dt, min=0
        )

        refractory_mask = self.refractory_counter > 0

        # Discretized LIF update
        membrane_update = self.dt * (
            (self.v_rest - self.v) / self.tau_m + input_current
        )

        self.v = torch.where(
            refractory_mask,
            self.v,
            self.v + membrane_update,
        )

        # Spike generation and reset
        spike_mask = (self.v >= self.v_threshold) & ~refractory_mask
        spikes = spike_mask.float()

        self.v = torch.where(spike_mask, self.v_reset, self.v)

        self.refractory_counter = torch.where(
            spike_mask,
            self.refractory_period,
            self.refractory_counter,
        )

        self.last_spike_time = torch.where(
            spike_mask,
            torch.zeros_like(self.last_spike_time),
            self.last_spike_time,
        )

        return spikes, self.v.clone()

    def forward_vectorized(
        self,
        input_currents: torch.Tensor,
        return_membrane: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Vectorized forward pass for entire sequence (all timesteps at once).
        Uses exponential Euler integration for better stability.

        Args:
            input_currents: Input currents (batch_size, n_neurons, seq_len).
            return_membrane: Whether to return membrane potentials.

        Returns:
            Tuple of (spike_trains, membrane_potentials).
            - spike_trains: (batch_size, n_neurons, seq_len)
            - membrane_potentials: (batch_size, n_neurons, seq_len) or None
        """
        if input_currents.dim() != 3 or input_currents.size(1) != self.n_neurons:
            raise ValueError(
                "Expected input_currents with shape (batch_size, n_neurons, seq_len)"
            )

        batch_size, n_neurons, seq_len = input_currents.shape
        device = input_currents.device

        # Exponential Euler coefficients (more accurate than forward Euler)
        alpha = torch.exp(-self.dt / self.tau_m)  # e^(-dt/tau_m)
        beta = self.tau_m * (1 - alpha)  # tau_m * (1 - e^(-dt/tau_m))
        gamma = (1 - alpha) * self.v_rest  # (1 - alpha) * v_rest

        # Initialize state
        v = torch.full((batch_size, n_neurons), self.v_rest.item(), device=device, dtype=input_currents.dtype)
        refractory_counter = torch.zeros(batch_size, n_neurons, device=device, dtype=input_currents.dtype)

        # Output tensors
        spike_trains = torch.zeros_like(input_currents)
        membrane_potentials = torch.zeros_like(input_currents) if return_membrane else None

        # Vectorized time loop (still sequential but can be optimized by compiler)
        for t in range(seq_len):
            # Current input at time t
            I_t = input_currents[:, :, t]

            # Add noise if specified
            if self.noise_std > 0:
                I_t = I_t + torch.randn_like(I_t) * self.noise_std

            # Update refractory counter
            refractory_counter = torch.clamp(refractory_counter - self.dt, min=0.0)
            refractory_mask = refractory_counter > 0

            # Exponential Euler update for membrane potential
            v_next = alpha * v + beta * I_t + gamma

            # Apply refractory mask (keep v unchanged if in refractory period)
            v = torch.where(refractory_mask, v, v_next)

            # Spike detection
            spike_mask = (v >= self.v_threshold) & ~refractory_mask
            spikes = spike_mask.float()

            # Reset membrane potential for spiking neurons
            v = torch.where(spike_mask, self.v_reset, v)

            # Set refractory counter for spiking neurons
            refractory_counter = torch.where(
                spike_mask,
                self.refractory_period,
                refractory_counter
            )

            # Store outputs
            spike_trains[:, :, t] = spikes
            if return_membrane:
                membrane_potentials[:, :, t] = v

        return spike_trains, membrane_potentials

    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current neuron parameters."""
        return {
            'tau_m': self.tau_m,
            'v_rest': self.v_rest,
            'v_threshold': self.v_threshold,
            'v_reset': self.v_reset,
            'refractory_period': self.refractory_period,
            'dt': torch.tensor(self.dt),
            'noise_std': torch.tensor(self.noise_std)
        }
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set neuron parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                if isinstance(getattr(self, key), nn.Parameter):
                    getattr(self, key).data = value
                else:
                    setattr(self, key, value)
    
    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get current neuron state."""
        return {
            'membrane_potential': self.v.clone(),
            'refractory_counter': self.refractory_counter.clone(),
            'last_spike_time': self.last_spike_time.clone()
        }
    
    def set_state(self, state: Dict[str, torch.Tensor]):
        """Set neuron state."""
        if 'membrane_potential' in state:
            self.v = state['membrane_potential'].clone()
        if 'refractory_counter' in state:
            self.refractory_counter = state['refractory_counter'].clone()
        if 'last_spike_time' in state:
            self.last_spike_time = state['last_spike_time'].clone()
    
    def get_firing_rate(self, time_window: float = 1000.0) -> torch.Tensor:
        """
        Calculate firing rate over a time window.
        
        Args:
            time_window: Time window in milliseconds.
            
        Returns:
            Firing rate for each neuron (spikes per second).
        """
        # This is a simplified calculation
        # In practice, you'd track spike times and calculate rate
        return torch.zeros(self.n_neurons)
    
    def get_neuron_info(self) -> Dict:
        """Get comprehensive neuron information."""
        return {
            'n_neurons': self.n_neurons,
            'parameters': self.get_parameters(),
            'learnable_params': any(isinstance(p, nn.Parameter) for p in self.parameters()),
            'device': next(self.parameters()).device if list(self.parameters()) else 'cpu'
        }


class AdaptiveLIFNeuron(LIFNeuron):
    """
    Adaptive LIF neuron with adaptive threshold.
    
    Extends LIF neuron with adaptive threshold that increases
    after each spike and decays back to baseline.
    """
    
    def __init__(
        self,
        n_neurons: int,
        tau_m: float = 20.0,
        v_rest: float = -65.0,
        v_threshold: float = -55.0,
        v_reset: float = -65.0,
        refractory_period: float = 2.0,
        dt: float = 1.0,
        noise_std: float = 0.0,
        tau_threshold: float = 100.0,  # Threshold adaptation time constant
        threshold_adaptation: float = 2.0,  # Threshold increase per spike
        learnable_params: bool = False
    ):
        """
        Initialize adaptive LIF neuron.
        
        Args:
            n_neurons: Number of neurons.
            tau_m: Membrane time constant in milliseconds.
            v_rest: Resting membrane potential in millivolts.
            v_threshold: Baseline spike threshold in millivolts.
            v_reset: Reset potential in millivolts.
            refractory_period: Refractory period in milliseconds.
            dt: Time step in milliseconds.
            noise_std: Standard deviation of noise.
            tau_threshold: Threshold adaptation time constant.
            threshold_adaptation: Threshold increase per spike.
            learnable_params: Whether neuron parameters are learnable.
        """
        super().__init__(
            n_neurons=n_neurons,
            tau_m=tau_m,
            v_rest=v_rest,
            v_threshold=v_threshold,
            v_reset=v_reset,
            refractory_period=refractory_period,
            dt=dt,
            noise_std=noise_std,
            learnable_params=learnable_params
        )
        
        # Adaptive threshold parameters
        if learnable_params:
            self.tau_threshold = nn.Parameter(torch.tensor(tau_threshold))
            self.threshold_adaptation = nn.Parameter(torch.tensor(threshold_adaptation))
        else:
            self.register_buffer('tau_threshold', torch.tensor(tau_threshold))
            self.register_buffer('threshold_adaptation', torch.tensor(threshold_adaptation))
        
        # Adaptive threshold state
        self.adaptive_threshold = self.v_threshold.clone()
    
    def reset_state(self):
        """Reset neuron state variables."""
        super().reset_state()
        self.adaptive_threshold = self.v_threshold.clone()
    
    def _update_neurons(
        self,
        input_current: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update adaptive LIF neuron states.
        
        Args:
            input_current: Input current to neurons (n_neurons).
            
        Returns:
            Tuple of (spikes, membrane_potential).
        """
        # Update adaptive threshold
        threshold_decay = self.dt / self.tau_threshold
        self.adaptive_threshold = self.adaptive_threshold + threshold_decay * (
            self.v_threshold - self.adaptive_threshold
        )
        
        # Store original threshold for spike detection
        original_threshold = self.v_threshold.clone()
        self.v_threshold = self.adaptive_threshold
        
        # Call parent update
        spikes, membrane = super()._update_neurons(input_current)
        
        # Restore original threshold
        self.v_threshold = original_threshold
        
        # Update adaptive threshold based on spikes
        spike_mask = spikes > 0
        self.adaptive_threshold = torch.where(
            spike_mask,
            self.adaptive_threshold + self.threshold_adaptation,
            self.adaptive_threshold
        )
        
        return spikes, membrane
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get current neuron parameters including adaptive parameters."""
        params = super().get_parameters()
        params.update({
            'tau_threshold': self.tau_threshold,
            'threshold_adaptation': self.threshold_adaptation,
            'adaptive_threshold': self.adaptive_threshold
        })
        return params 
