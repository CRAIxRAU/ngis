"""
Evaluation metrics for NGIS.

Provides comprehensive metrics for assessing EEG reconstruction quality,
graph structure, and spiking dynamics.
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


def compute_eeg_metrics(real_eeg: torch.Tensor, simulated_eeg: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive EEG reconstruction metrics.
    
    Args:
        real_eeg: Real EEG tensor (batch, channels, time) or (channels, time)
        simulated_eeg: Simulated EEG tensor (same shape as real_eeg)
        
    Returns:
        Dictionary containing:
        - mean_correlation: Average Pearson correlation across channels
        - min_correlation: Minimum correlation
        - max_correlation: Maximum correlation
        - mse: Mean squared error
        - mae: Mean absolute error
        - rmse: Root mean squared error
        - nrmse: Normalized RMSE
        - psd_similarity: Power spectral density similarity
    """
    # Handle batch dimension
    if real_eeg.dim() == 3:
        # Average over batch
        real_eeg = real_eeg.mean(dim=0)
        simulated_eeg = simulated_eeg.mean(dim=0)
    
    # Move to CPU and convert to numpy for correlation computation
    real_np = real_eeg.detach().cpu().numpy()
    sim_np = simulated_eeg.detach().cpu().numpy()
    
    n_channels = real_np.shape[0]
    
    # Compute per-channel Pearson correlation
    correlations = []
    for ch in range(n_channels):
        try:
            corr, _ = pearsonr(real_np[ch], sim_np[ch])
            if not np.isnan(corr):
                correlations.append(corr)
        except Exception as e:
            logger.debug(f"Failed to compute correlation for channel {ch}: {e}")
            continue
    
    correlations = np.array(correlations)
    
    # Compute error metrics
    mse = F.mse_loss(simulated_eeg, real_eeg).item()
    mae = F.l1_loss(simulated_eeg, real_eeg).item()
    rmse = np.sqrt(mse)
    
    # Normalized RMSE (by std of real signal)
    real_std = real_eeg.std().item()
    nrmse = rmse / (real_std + 1e-8)
    
    # Power spectral density similarity
    psd_sim = compute_psd_similarity(real_np, sim_np)
    
    metrics = {
        'mean_correlation': float(correlations.mean()) if len(correlations) > 0 else 0.0,
        'min_correlation': float(correlations.min()) if len(correlations) > 0 else 0.0,
        'max_correlation': float(correlations.max()) if len(correlations) > 0 else 0.0,
        'std_correlation': float(correlations.std()) if len(correlations) > 0 else 0.0,
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'nrmse': nrmse,
        'psd_similarity': psd_sim
    }
    
    return metrics


def compute_psd_similarity(real_eeg: np.ndarray, simulated_eeg: np.ndarray, 
                          fs: float = 1000.0) -> float:
    """
    Compute power spectral density similarity between real and simulated EEG.
    
    Args:
        real_eeg: Real EEG numpy array (channels, time)
        simulated_eeg: Simulated EEG numpy array (channels, time)
        fs: Sampling frequency in Hz
        
    Returns:
        PSD similarity score (0-1, higher is better)
    """
    try:
        # Compute PSD for each channel
        n_channels = real_eeg.shape[0]
        psd_similarities = []
        
        for ch in range(n_channels):
            # Compute PSD using Welch's method
            freqs_real, psd_real = signal.welch(real_eeg[ch], fs=fs, nperseg=min(256, real_eeg.shape[1]))
            freqs_sim, psd_sim = signal.welch(simulated_eeg[ch], fs=fs, nperseg=min(256, simulated_eeg.shape[1]))
            
            # Focus on 1-40 Hz range (typical EEG frequencies)
            freq_mask = (freqs_real >= 1) & (freqs_real <= 40)
            psd_real_band = psd_real[freq_mask]
            psd_sim_band = psd_sim[freq_mask]
            
            # Normalize PSDs
            psd_real_norm = psd_real_band / (np.sum(psd_real_band) + 1e-10)
            psd_sim_norm = psd_sim_band / (np.sum(psd_sim_band) + 1e-10)
            
            # Compute correlation between PSDs
            corr, _ = pearsonr(psd_real_norm, psd_sim_norm)
            if not np.isnan(corr):
                psd_similarities.append(corr)
        
        return float(np.mean(psd_similarities)) if psd_similarities else 0.0
    
    except Exception as e:
        logger.warning(f"Failed to compute PSD similarity: {e}")
        return 0.0


def compute_band_power_metrics(real_eeg: torch.Tensor, simulated_eeg: torch.Tensor,
                               fs: float = 1000.0) -> Dict[str, float]:
    """
    Compute band power correlation for standard EEG frequency bands.
    
    Args:
        real_eeg: Real EEG tensor (batch, channels, time) or (channels, time)
        simulated_eeg: Simulated EEG tensor (same shape)
        fs: Sampling frequency in Hz
        
    Returns:
        Dictionary with band power correlations for delta, theta, alpha, beta, gamma
    """
    # Handle batch dimension
    if real_eeg.dim() == 3:
        real_eeg = real_eeg.mean(dim=0)
        simulated_eeg = simulated_eeg.mean(dim=0)
    
    real_np = real_eeg.detach().cpu().numpy()
    sim_np = simulated_eeg.detach().cpu().numpy()
    
    # Define frequency bands
    bands = {
        'delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 40)
    }
    
    band_metrics = {}
    
    for band_name, (low_freq, high_freq) in bands.items():
        try:
            # Extract band power for each channel
            real_band_power = []
            sim_band_power = []
            
            for ch in range(real_np.shape[0]):
                # Compute PSD
                freqs, psd_real = signal.welch(real_np[ch], fs=fs, nperseg=min(256, real_np.shape[1]))
                _, psd_sim = signal.welch(sim_np[ch], fs=fs, nperseg=min(256, sim_np.shape[1]))
                
                # Extract band power
                band_mask = (freqs >= low_freq) & (freqs <= high_freq)
                real_band_power.append(np.mean(psd_real[band_mask]))
                sim_band_power.append(np.mean(psd_sim[band_mask]))
            
            # Compute correlation between real and simulated band powers
            corr, _ = pearsonr(real_band_power, sim_band_power)
            band_metrics[f'{band_name}_power_corr'] = float(corr) if not np.isnan(corr) else 0.0
        
        except Exception as e:
            logger.debug(f"Failed to compute {band_name} band power: {e}")
            band_metrics[f'{band_name}_power_corr'] = 0.0
    
    return band_metrics


def compute_spike_metrics(spike_trains: torch.Tensor) -> Dict[str, float]:
    """
    Compute spiking dynamics metrics.
    
    Args:
        spike_trains: Spike trains tensor (batch, neurons, time)
        
    Returns:
        Dictionary with spike metrics:
        - mean_firing_rate: Average firing rate across neurons
        - std_firing_rate: Standard deviation of firing rates
        - population_synchrony: Measure of population synchrony
        - sparsity: Fraction of neurons that never fire
    """
    # Handle batch dimension
    if spike_trains.dim() == 3:
        spike_trains = spike_trains.mean(dim=0)  # Average over batch
    
    # Compute firing rates (Hz)
    # Assuming dt = 1ms, so firing rate = spikes per ms * 1000
    firing_rates = spike_trains.mean(dim=1) * 1000  # (neurons,)
    
    # Compute metrics
    mean_fr = firing_rates.mean().item()
    std_fr = firing_rates.std().item()
    
    # Population synchrony: correlation between neuron pairs
    # Simplified: variance of population firing rate
    pop_activity = spike_trains.mean(dim=0)  # Average over neurons at each time
    pop_synchrony = pop_activity.std().item()
    
    # Sparsity: fraction of neurons that never fire
    never_fire = (spike_trains.sum(dim=1) == 0).float().mean().item()
    
    # Active neurons: fraction that fire at least once
    active_fraction = 1.0 - never_fire
    
    metrics = {
        'mean_firing_rate': mean_fr,
        'std_firing_rate': std_fr,
        'population_synchrony': pop_synchrony,
        'sparsity': never_fire,
        'active_fraction': active_fraction
    }
    
    return metrics


def compute_graph_metrics(graph_data, reference_graph=None) -> Dict[str, float]:
    """
    Compute graph structure metrics.
    
    Args:
        graph_data: PyTorch Geometric Data object with current graph
        reference_graph: Optional reference graph for comparison
        
    Returns:
        Dictionary with graph metrics:
        - n_nodes: Number of nodes
        - n_edges: Number of edges
        - density: Graph density
        - avg_degree: Average node degree
    """
    if graph_data is None:
        return {}
    
    n_nodes = graph_data.x.shape[0]
    n_edges = graph_data.edge_index.shape[1]
    
    # Compute density
    max_edges = n_nodes * (n_nodes - 1)  # Assuming directed graph
    density = n_edges / max_edges if max_edges > 0 else 0.0
    
    # Compute average degree
    avg_degree = n_edges / n_nodes if n_nodes > 0 else 0.0
    
    metrics = {
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'density': density,
        'avg_degree': avg_degree
    }
    
    # If reference graph provided, compute comparison metrics
    if reference_graph is not None:
        # Compute edge overlap / precision / recall
        # This would require converting edge_index to sets and comparing
        # Placeholder for now
        metrics['graph_similarity'] = 0.0
    
    return metrics


def compute_all_metrics(real_eeg: torch.Tensor, 
                       simulated_eeg: torch.Tensor,
                       spike_trains: Optional[torch.Tensor] = None,
                       graph_data = None) -> Dict[str, float]:
    """
    Compute all available metrics for a validation batch.
    
    Args:
        real_eeg: Real EEG tensor (batch, channels, time)
        simulated_eeg: Simulated EEG tensor (batch, channels, time)
        spike_trains: Optional spike trains tensor (batch, neurons, time)
        graph_data: Optional graph data structure
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # EEG reconstruction metrics
    eeg_metrics = compute_eeg_metrics(real_eeg, simulated_eeg)
    metrics.update(eeg_metrics)
    
    # Band power metrics
    band_metrics = compute_band_power_metrics(real_eeg, simulated_eeg)
    metrics.update(band_metrics)
    
    # Spike metrics
    if spike_trains is not None:
        spike_metrics = compute_spike_metrics(spike_trains)
        metrics.update(spike_metrics)
    
    # Graph metrics
    if graph_data is not None:
        graph_metrics = compute_graph_metrics(graph_data)
        metrics.update(graph_metrics)
    
    return metrics


def format_metrics_for_logging(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics dictionary as a readable string for logging.
    
    Args:
        metrics: Dictionary of metrics
        prefix: Optional prefix for metric names
        
    Returns:
        Formatted string
    """
    lines = []
    
    # Group metrics by category
    eeg_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['correlation', 'mse', 'mae', 'rmse', 'psd'])}
    spike_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['firing', 'synchrony', 'sparsity', 'active'])}
    band_metrics = {k: v for k, v in metrics.items() if 'power_corr' in k}
    graph_metrics = {k: v for k, v in metrics.items() if any(x in k for x in ['nodes', 'edges', 'density', 'degree'])}
    
    if eeg_metrics:
        lines.append(f"{prefix}EEG Reconstruction:")
        for k, v in eeg_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
    
    if band_metrics:
        lines.append(f"{prefix}Band Power:")
        for k, v in band_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
    
    if spike_metrics:
        lines.append(f"{prefix}Spiking Dynamics:")
        for k, v in spike_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
    
    if graph_metrics:
        lines.append(f"{prefix}Graph Structure:")
        for k, v in graph_metrics.items():
            lines.append(f"  {k}: {v:.4f}")
    
    return "\n".join(lines)
