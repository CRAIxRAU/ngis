"""
Graph construction utilities for NGIS.

Handles construction of functional brain networks from EEG data
using various connectivity measures and graph types.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from torch_geometric.data import Data, Batch

logger = logging.getLogger(__name__)


class GraphConstructor(nn.Module):
    """
    Graph constructor for functional brain networks.
    
    Builds graph structures from EEG data using various connectivity
    measures and graph construction methods.
    """
    
    def __init__(
        self,
        n_channels: int = 128,
        n_neurons: int = 256,
        graph_type: str = "functional",
        connectivity_threshold: float = 0.1,
        connectivity_measure: str = "correlation",
        learnable: bool = True,
        sparsity: float = 0.1
    ):
        """
        Initialize graph constructor.
        
        Args:
            n_channels: Number of EEG channels.
            n_neurons: Number of neurons in the network.
            graph_type: Type of graph construction ("functional", "anatomical", "learned").
            connectivity_threshold: Threshold for edge creation.
            connectivity_measure: Measure for connectivity ("correlation", "coherence", "mutual_info").
            learnable: Whether graph structure is learnable.
            sparsity: Target sparsity of the graph.
        """
        super().__init__()
        
        self.n_channels = n_channels
        self.n_neurons = n_neurons
        self.graph_type = graph_type
        self.connectivity_threshold = connectivity_threshold
        self.connectivity_measure = connectivity_measure
        self.learnable = learnable
        self.sparsity = sparsity
        self.feature_dim = 64

        # Initialize learnable parameters if needed
        if self.learnable and self.graph_type == "learned":
            self._init_learnable_parameters()
        
        # Store current graph
        self.current_graph = None
        
        logger.info(f"Initialized graph constructor: {graph_type} type")
    
    def _init_learnable_parameters(self):
        """Initialize learnable graph parameters."""
        # Learnable adjacency matrix
        self.adjacency_matrix = nn.Parameter(
            torch.randn(self.n_neurons, self.n_neurons) * 0.1
        )
        
        # Learnable node features
        self.node_features = nn.Parameter(
            torch.randn(self.n_neurons, 64) * 0.1
        )
    
    def forward(self, eeg_data: torch.Tensor) -> Union[Data, Batch]:
        """
        Construct graph from EEG data.

        Args:
            eeg_data: EEG data of shape (batch_size, n_channels, seq_len).
            
        Returns:
            PyTorch Geometric Data object.
        """
        if eeg_data.dim() != 3:
            raise ValueError(
                "Expected eeg_data with shape (batch, channels, seq_len)"
            )

        if self.graph_type == "functional":
            return self._construct_functional_graph(eeg_data)
        elif self.graph_type == "anatomical":
            return self._construct_anatomical_graph(eeg_data)
        elif self.graph_type == "learned":
            return self._construct_learned_graph(eeg_data)
        else:
            raise ValueError(f"Unknown graph type: {self.graph_type}")

    def _construct_functional_graph(self, eeg_data: torch.Tensor) -> Union[Data, Batch]:
        """Construct functional graph based on EEG connectivity."""
        batch_size, n_channels, _ = eeg_data.shape

        graphs = []
        for sample_idx in range(batch_size):
            sample = eeg_data[sample_idx]

            connectivity_matrix = self._compute_connectivity_matrix(sample)
            adjacency_matrix = self._threshold_connectivity(connectivity_matrix)
            edge_index = self._adjacency_to_edge_index(adjacency_matrix)

            if edge_index.numel() == 0:
                # Create fully connected fallback to avoid empty graphs
                edge_index = self._fully_connected_edge_index(n_channels).to(
                    eeg_data.device
                )
                edge_weight = torch.ones(edge_index.shape[1], device=eeg_data.device)
            else:
                edge_index = edge_index.to(eeg_data.device)
                edge_weight = connectivity_matrix[edge_index[0], edge_index[1]]

            node_features = self._create_node_features(sample)

            graph = Data(
                x=node_features,
                edge_index=edge_index.long(),
                edge_weight=edge_weight
            )
            graphs.append(graph)

        if len(graphs) == 1:
            self.current_graph = graphs[0]
            return graphs[0]

        batch = Batch.from_data_list(graphs)
        self.current_graph = batch
        return batch
    
    def _construct_anatomical_graph(self, eeg_data: torch.Tensor) -> Union[Data, Batch]:
        """Construct anatomical graph based on spatial proximity."""
        batch_size, n_channels, _ = eeg_data.shape

        spatial_coords = self._get_spatial_coordinates()
        distance_matrix = self._compute_distance_matrix(spatial_coords)
        adjacency_matrix = self._proximity_to_adjacency(distance_matrix)
        edge_index = self._adjacency_to_edge_index(adjacency_matrix)

        edge_index = edge_index.to(eeg_data.device).long()
        graphs = []
        for sample_idx in range(batch_size):
            sample = eeg_data[sample_idx]
            node_features = self._create_node_features(sample)

            graph = Data(
                x=node_features,
                edge_index=edge_index,
                edge_weight=torch.ones(edge_index.shape[1], device=eeg_data.device)
            )
            graphs.append(graph)

        if len(graphs) == 1:
            self.current_graph = graphs[0]
            return graphs[0]

        batch = Batch.from_data_list(graphs)
        self.current_graph = batch
        return batch
    
    def _construct_learned_graph(self, eeg_data: torch.Tensor) -> Data:
        """Construct learnable graph structure."""
        # Use learnable adjacency matrix
        adjacency_matrix = torch.sigmoid(self.adjacency_matrix)
        
        # Apply sparsity constraint
        adjacency_matrix = self._apply_sparsity(adjacency_matrix)
        
        # Create edge index
        edge_index = self._adjacency_to_edge_index(adjacency_matrix)
        
        # Use learnable node features
        node_features = self.node_features
        
        # Create PyTorch Geometric Data object
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_weight=torch.ones(edge_index.shape[1])
        )
        
        self.current_graph = graph_data
        return graph_data
    
    def _compute_connectivity_matrix(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Compute connectivity matrix from EEG data."""
        if eeg_data.dim() == 3:
            eeg_mean = eeg_data.mean(dim=0)
        elif eeg_data.dim() == 2:
            eeg_mean = eeg_data
        else:
            raise ValueError("EEG data must have 2 or 3 dimensions")

        if self.connectivity_measure == "correlation":
            return self._compute_correlation_matrix(eeg_mean)
        elif self.connectivity_measure == "coherence":
            return self._compute_coherence_matrix(eeg_mean)
        elif self.connectivity_measure == "mutual_info":
            return self._compute_mutual_info_matrix(eeg_mean)
        else:
            raise ValueError(f"Unknown connectivity measure: {self.connectivity_measure}")
    
    def _compute_correlation_matrix(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Compute correlation matrix."""
        # Convert to numpy for correlation computation
        eeg_np = eeg_data.detach().cpu().numpy()
        
        # Compute correlation matrix
        correlation_matrix = np.corrcoef(eeg_np)
        
        # Convert back to torch
        return torch.from_numpy(correlation_matrix).float().to(eeg_data.device)
    
    def _compute_coherence_matrix(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Compute coherence matrix."""
        eeg_np = eeg_data.detach().cpu().numpy()
        n_channels = eeg_np.shape[0]
        
        coherence_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                freqs, coh = signal.coherence(eeg_np[i, :], eeg_np[j, :])
                coherence_matrix[i, j] = np.mean(coh)
                coherence_matrix[j, i] = coherence_matrix[i, j]
        
        return torch.from_numpy(coherence_matrix).float().to(eeg_data.device)
    
    def _compute_mutual_info_matrix(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """Compute mutual information matrix (simplified)."""
        # Simplified mutual information using correlation
        return self._compute_correlation_matrix(eeg_data)
    
    def _threshold_connectivity(self, connectivity_matrix: torch.Tensor) -> torch.Tensor:
        """Apply thresholding to connectivity matrix."""
        # Apply threshold
        adjacency_matrix = (connectivity_matrix > self.connectivity_threshold).float()
        
        # Remove self-loops
        adjacency_matrix.fill_diagonal_(0)
        
        return adjacency_matrix
    
    def _adjacency_to_edge_index(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Convert adjacency matrix to edge index format."""
        # Find non-zero elements
        edge_indices = torch.nonzero(adjacency_matrix, as_tuple=False)
        
        # Transpose to get (2, num_edges) format
        edge_index = edge_indices.t()
        
        return edge_index
    
    def _create_node_features(self, eeg_sample: torch.Tensor) -> torch.Tensor:
        """Create node features from a single EEG sample."""
        if eeg_sample.dim() != 2:
            raise ValueError("Expected eeg_sample with shape (channels, seq_len)")

        # Down-sample the temporal dimension to a fixed feature size
        channel_features = torch.nn.functional.interpolate(
            eeg_sample.unsqueeze(0),
            size=self.feature_dim,
            mode="linear",
            align_corners=False
        ).squeeze(0)

        return channel_features

    def _fully_connected_edge_index(self, n_nodes: int) -> torch.Tensor:
        """Create fully connected edge index (without self-loops)."""
        rows, cols = torch.meshgrid(
            torch.arange(n_nodes),
            torch.arange(n_nodes),
            indexing="ij"
        )
        mask = rows != cols
        edge_index = torch.stack([rows[mask], cols[mask]])
        return edge_index
    
    def _get_spatial_coordinates(self) -> np.ndarray:
        """Get spatial coordinates for EEG channels."""
        # Simplified 2D coordinates for EEG channels
        # In practice, this would use actual electrode positions
        n_channels = self.n_channels
        
        # Create circular arrangement
        angles = np.linspace(0, 2 * np.pi, n_channels, endpoint=False)
        radius = 1.0
        
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        return np.column_stack([x, y])
    
    def _compute_distance_matrix(self, spatial_coords: np.ndarray) -> np.ndarray:
        """Compute distance matrix from spatial coordinates."""
        return squareform(pdist(spatial_coords))
    
    def _proximity_to_adjacency(self, distance_matrix: np.ndarray) -> torch.Tensor:
        """Convert distance matrix to adjacency matrix based on proximity."""
        # Create adjacency based on nearest neighbors
        n_channels = distance_matrix.shape[0]
        k_neighbors = max(1, int(n_channels * self.sparsity))
        
        adjacency_matrix = np.zeros((n_channels, n_channels))
        
        for i in range(n_channels):
            # Find k nearest neighbors
            distances = distance_matrix[i, :]
            nearest_indices = np.argsort(distances)[1:k_neighbors + 1]  # Exclude self
            
            adjacency_matrix[i, nearest_indices] = 1
            adjacency_matrix[nearest_indices, i] = 1  # Undirected graph
        
        return torch.from_numpy(adjacency_matrix).float()
    
    def _apply_sparsity(self, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """Apply sparsity constraint to adjacency matrix."""
        # Keep top k% of connections
        n_edges = int(self.n_neurons * self.n_neurons * self.sparsity)
        
        # Flatten and get top k values
        flat_adj = adjacency_matrix.flatten()
        top_k_values, _ = torch.topk(flat_adj, n_edges)
        threshold = top_k_values[-1]
        
        # Apply threshold
        sparse_adj = (adjacency_matrix >= threshold).float()
        
        # Remove self-loops
        sparse_adj.fill_diagonal_(0)
        
        return sparse_adj
    
    def get_graph(self) -> Optional[Data]:
        """Get the current graph structure."""
        return self.current_graph
    
    def update_graph(self, eeg_data: torch.Tensor):
        """Update graph structure based on new EEG data."""
        self.forward(eeg_data)
    
    def get_connectivity_matrix(self) -> torch.Tensor:
        """Get the current connectivity matrix."""
        if self.current_graph is None:
            return torch.zeros(self.n_neurons, self.n_neurons)
        
        # Convert edge index back to adjacency matrix
        edge_index = self.current_graph.edge_index
        adjacency_matrix = torch.zeros(self.n_neurons, self.n_neurons)
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[:, i]
            adjacency_matrix[src, dst] = 1
        
        return adjacency_matrix
    
    def get_graph_statistics(self) -> Dict:
        """Get statistics about the current graph."""
        if self.current_graph is None:
            return {}
        
        edge_index = self.current_graph.edge_index
        n_edges = edge_index.shape[1]
        n_nodes = self.current_graph.x.shape[0]
        
        # Calculate density
        max_edges = n_nodes * (n_nodes - 1)  # No self-loops
        density = n_edges / max_edges if max_edges > 0 else 0
        
        return {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': density,
            'graph_type': self.graph_type,
            'connectivity_measure': self.connectivity_measure
        } 
