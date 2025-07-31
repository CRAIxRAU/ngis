"""
Simulation module for NGIS.

This module handles Brian2 integration and forward simulation
for the graph-structured spiking neural network.
"""

from .brian2_simulator import Brian2Simulator
from .forward_simulator import ForwardSimulator
from .simulation_utils import SimulationUtils

__all__ = [
    "Brian2Simulator",
    "ForwardSimulator", 
    "SimulationUtils"
] 