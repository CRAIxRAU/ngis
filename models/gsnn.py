"""
Graph-Structured Spiking Neural Network (G-SNN) for NGIS.

Main model class that integrates graph structure, LIF neurons, and EEG readout
for reconstructing functional brain networks from EEG data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch.cuda.nvtx import range_push, range_pop

from .graph_constructor import GraphConstructor
from .lif_neuron import LIFNeuron
from .synapse import Synapse
from .readout import EEGReadout

logger = logging.getLogger(__name__)


class GSNN(nn.Module):
    """
    Graph-Structured Spiking Neural Network (G-SNN).

    Main model for reconstructing functional brain networks from EEG data.
    Integrates graph structure, LIF neurons, and EEG readout layers.
    """

    def __init__(
        self,
        n_channels: int = 128,
        n_neurons: int = 256,
        n_layers: int = 3,
        hidden_dim: int = 64,
        graph_type: str = "functional",
        connectivity_threshold: float = 0.1,
        lif_params: Optional[Dict] = None,
        synapse_params: Optional[Dict] = None,
        readout_params: Optional[Dict] = None,
        dropout: float = 0.1,
    ):
        """
        Initialize G-SNN model.

        Args:
            n_channels: Number of EEG channels (input).
            n_neurons: Number of neurons in the network.
            n_layers: Number of graph convolution layers.
            hidden_dim: Hidden dimension for graph convolutions.
            graph_type: Type of graph construction ("functional", "anatomical", "learned").
            connectivity_threshold: Threshold for graph connectivity.
            lif_params: Parameters for LIF neurons.
            synapse_params: Parameters for synaptic connections.
            readout_params: Parameters for EEG readout layer.
            dropout: Dropout rate.
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_neurons = n_neurons
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.graph_type = graph_type
        self.connectivity_threshold = connectivity_threshold
        self.dropout = dropout

        # Initialize components
        self._init_graph_constructor()
        self._init_lif_neurons(lif_params)
        self._init_synapses(synapse_params)
        self._init_graph_layers()
        self._init_readout(readout_params)

        self.node_to_neuron = nn.Linear(self.n_channels, self.n_neurons)

        logger.info(
            f"Initialized G-SNN with {n_neurons} neurons and {n_layers} layers"
        )

    def _init_graph_constructor(self):
        """Initialize graph constructor."""
        self.graph_constructor = GraphConstructor(
            n_channels=self.n_channels,
            n_neurons=self.n_neurons,
            graph_type=self.graph_type,
            connectivity_threshold=self.connectivity_threshold,
        )

    def _init_lif_neurons(self, lif_params: Optional[Dict] = None):
        """Initialize LIF neurons."""
        if lif_params is None:
            lif_params = {
                "tau_m": 20.0,  # Membrane time constant (ms)
                "v_rest": -65.0,  # Resting potential (mV)
                "v_threshold": -55.0,  # Spike threshold (mV)
                "v_reset": -65.0,  # Reset potential (mV)
                "refractory_period": 2.0,  # Refractory period (ms)
            }

        self.lif_neurons = LIFNeuron(n_neurons=self.n_neurons, **lif_params)

    def _init_synapses(self, synapse_params: Optional[Dict] = None):
        """Initialize synaptic connections."""
        if synapse_params is None:
            synapse_params = {
                "tau_s": 5.0,  # Synaptic time constant (ms)
                "weight_scale": 1.0,  # Weight scaling factor
                "plasticity": True,  # Enable synaptic plasticity
            }

        self.synapses = Synapse(n_neurons=self.n_neurons, **synapse_params)

    def _init_graph_layers(self):
        """Initialize graph convolution layers."""
        self.graph_layers = nn.ModuleList()

        # Input projection (graph constructor outputs 64-dim node features)
        self.input_projection = nn.Linear(64, self.hidden_dim)

        # Graph convolution layers
        for i in range(self.n_layers):
            if i == 0:
                in_dim = self.hidden_dim
            else:
                in_dim = self.hidden_dim

            # Use GAT for better attention mechanism
            conv_layer = GATConv(
                in_channels=in_dim,
                out_channels=self.hidden_dim,
                heads=4,
                dropout=self.dropout,
                concat=False,
            )
            self.graph_layers.append(conv_layer)

        # Output projection
        self.output_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def _init_readout(self, readout_params: Optional[Dict] = None):
        """Initialize EEG readout layer."""
        if readout_params is None:
            readout_params = {"readout_type": "linear", "activation": "tanh"}

        self.readout = EEGReadout(
            n_neurons=self.n_neurons,
            n_channels=self.n_channels,
            **readout_params,
        )

    def forward(
        self,
        eeg_input: torch.Tensor,
        graph_data: Optional[Data] = None,
        return_spikes: bool = False,
        return_graph: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through G-SNN.

        Args:
            eeg_input: Input EEG data of shape (batch_size, n_channels, seq_len).
            graph_data: Optional pre-computed graph data.
            return_spikes: Whether to return spike trains.
            return_graph: Whether to return graph structure.

        Returns:
            Dictionary containing model outputs.
        """
        batch_size, n_channels, seq_len = eeg_input.shape

        # Construct or use provided graph
        if graph_data is None:
            graph_data = self.graph_constructor(eeg_input)

        # Process through graph layers
        range_push("Process Graph Layers")
        neuron_currents = self._process_graph_layers(graph_data, seq_len)
        range_pop()

        # Simulate spiking dynamics
        range_push("Simulate Spiking Dynamics")
        spike_trains, membrane_potentials = self._simulate_spiking(
            neuron_currents
        )
        range_pop()

        # Readout to EEG
        range_push("Readout to EEG")
        readout_outputs = self.readout(spike_trains)
        eeg_output = readout_outputs["eeg_output"]
        range_pop()

        # Prepare output
        output = {
            "eeg_output": eeg_output,
            "spike_trains": spike_trains if return_spikes else None,
            "membrane_potentials": (
                membrane_potentials if return_spikes else None
            ),
            "graph_data": graph_data if return_graph else None,
        }

        # Include any additional readout artifacts (e.g., attention weights)
        for key, value in readout_outputs.items():
            if key != "eeg_output":
                output[key] = value

        return output

    def _process_graph_layers(
        self, graph_data: Union[Data, Batch], seq_len: int
    ) -> torch.Tensor:
        """Process data through graph convolution layers and prepare neuron currents."""
        x = graph_data.x
        edge_index = graph_data.edge_index
        batch = getattr(graph_data, "batch", None)

        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # Input projection
        x = self.input_projection(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for i, conv_layer in enumerate(self.graph_layers):
            x = conv_layer(x, edge_index)
            if i < len(self.graph_layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.output_projection(x)

        batch_size = int(batch.max().item()) + 1
        node_embeddings = x.view(batch_size, self.n_channels, self.hidden_dim)

        # Map channel-wise embeddings to neuron space
        neuron_embeddings = self.node_to_neuron(
            node_embeddings.permute(0, 2, 1)
        )  # (batch, hidden_dim, n_neurons)
        neuron_embeddings = neuron_embeddings.permute(0, 2, 1)

        # Interpolate embeddings across time to create input currents
        neuron_currents = torch.nn.functional.interpolate(
            neuron_embeddings, size=seq_len, mode="linear", align_corners=False
        )

        return neuron_currents

    def _simulate_spiking(
        self, neuron_currents: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate spiking dynamics using LIF neurons."""

        batch_size, n_neurons, seq_len = neuron_currents.shape
        device = neuron_currents.device

        # Use vectorized forward pass (processes all timesteps efficiently)
        range_push("LIF Vectorized Forward")
        spike_trains, membrane_potentials = self.lif_neurons.forward_vectorized(
            neuron_currents, return_membrane=True
        )
        range_pop()

        # Apply synaptic filtering if needed (only during training)
        use_synapses = hasattr(self, "synapses") and self.training
        if use_synapses:
            range_push("Synapse Filtering")
            self.synapses.reset_state(batch_size=batch_size, device=device)

            filtered_trains = torch.zeros_like(spike_trains)
            for t in range(seq_len):
                filtered_spikes, _ = self.synapses(spike_trains[:, :, t])
                filtered_trains[:, :, t] = filtered_spikes

            spike_trains = filtered_trains
            range_pop()

        return spike_trains, membrane_potentials

    def get_graph_structure(self) -> Data:
        """Get the current graph structure."""
        return self.graph_constructor.get_graph()

    def update_graph_structure(self, eeg_data: torch.Tensor):
        """Update graph structure based on new EEG data."""
        self.graph_constructor.update_graph(eeg_data)

    def get_connectivity_matrix(self) -> torch.Tensor:
        """Get the connectivity matrix between neurons."""
        return self.graph_constructor.get_connectivity_matrix()

    def get_neuron_parameters(self) -> Dict[str, torch.Tensor]:
        """Get LIF neuron parameters."""
        return self.lif_neurons.get_parameters()

    def set_neuron_parameters(self, params: Dict[str, torch.Tensor]):
        """Set LIF neuron parameters."""
        self.lif_neurons.set_parameters(params)

    def get_synapse_weights(self) -> torch.Tensor:
        """Get synaptic weights."""
        return self.synapses.get_weights()

    def set_synapse_weights(self, weights: torch.Tensor):
        """Set synaptic weights."""
        self.synapses.set_weights(weights)

    def reset_state(self):
        """Reset neuron and synapse states."""
        self.lif_neurons.reset_state()
        if hasattr(self, "synapses"):
            self.synapses.reset_state()

    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters."""
        total_params = 0
        trainable_params = 0

        for name, param in self.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params

        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    def get_model_info(self) -> Dict:
        """Get comprehensive model information."""
        param_counts = self.count_parameters()

        info = {
            "model_type": "G-SNN",
            "n_channels": self.n_channels,
            "n_neurons": self.n_neurons,
            "n_layers": self.n_layers,
            "hidden_dim": self.hidden_dim,
            "graph_type": self.graph_type,
            "connectivity_threshold": self.connectivity_threshold,
            "dropout": self.dropout,
            "parameters": param_counts,
        }

        return info
