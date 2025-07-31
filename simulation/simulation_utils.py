"""
Simulation utilities for NGIS.

Provides utility functions for simulation tasks and analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional


class SimulationUtils:
    """Utility functions for simulation tasks."""
    
    @staticmethod
    def calculate_firing_rate(spike_trains: torch.Tensor) -> torch.Tensor:
        """Calculate firing rate from spike trains."""
        return spike_trains.mean(dim=-1)
    
    @staticmethod
    def calculate_spike_timing(spike_trains: torch.Tensor) -> List[List[float]]:
        """Calculate spike timing from spike trains."""
        # Placeholder implementation
        return []
    
    @staticmethod
    def analyze_network_activity(spike_trains: torch.Tensor) -> Dict:
        """Analyze network activity patterns."""
        firing_rates = SimulationUtils.calculate_firing_rate(spike_trains)
        
        return {
            'mean_firing_rate': float(firing_rates.mean()),
            'std_firing_rate': float(firing_rates.std()),
            'total_spikes': int(spike_trains.sum()),
            'spike_rate': float(spike_trains.mean())
        } 