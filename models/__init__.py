"""
Models module for NGIS.

This module contains the graph-structured spiking neural network (G-SNN)
architecture and related components for the Neural Graph Inverse Simulator.
"""

from .gsnn import GSNN
from .graph_constructor import GraphConstructor
from .lif_neuron import LIFNeuron
from .synapse import Synapse
from .readout import EEGReadout

__all__ = [
    "GSNN",
    "GraphConstructor", 
    "LIFNeuron",
    "Synapse",
    "EEGReadout"
] 