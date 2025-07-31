"""
Training module for NGIS.

This module handles loss functions, optimization, and distributed
training for the graph-structured spiking neural network.
"""

from .trainer import NGISTrainer
from .loss_functions import EEGLoss, SpikingLoss, CombinedLoss
from .optimizer import NGISOptimizer
from .scheduler import NGISScheduler

__all__ = [
    "NGISTrainer",
    "EEGLoss",
    "SpikingLoss", 
    "CombinedLoss",
    "NGISOptimizer",
    "NGISScheduler"
] 