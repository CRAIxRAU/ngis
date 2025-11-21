"""
Visualization utilities for NGIS.

Provides visualization tools for EEG data and network analysis.
"""

from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


class EEGVisualizer:
    """Visualization tools for EEG data."""
    
    @staticmethod
    def plot_eeg_comparison(
        real_eeg: np.ndarray,
        simulated_eeg: np.ndarray,
        channels: Optional[List[str]] = None,
        max_channels: int = 4,
        time_range: Optional[Tuple[int, int]] = None,
        title: str = "Real vs Simulated EEG",
    ) -> plt.Figure:
        """Plot side-by-side EEG traces for a subset of channels."""

        if real_eeg.shape != simulated_eeg.shape:
            raise ValueError("Real and simulated EEG must have the same shape")

        n_channels, seq_len = real_eeg.shape
        idx = list(range(min(n_channels, max_channels)))

        if channels is not None:
            idx = [i for i, ch in enumerate(channels) if i < len(channels)][:max_channels]

        start, end = (0, seq_len) if time_range is None else time_range
        start = max(start, 0)
        end = min(end, seq_len)

        fig, axes = plt.subplots(len(idx), 1, figsize=(10, 2.5 * len(idx)), sharex=True)
        if len(idx) == 1:
            axes = [axes]

        time_axis = np.arange(start, end)

        for ax, channel_idx in zip(axes, idx):
            ax.plot(time_axis, real_eeg[channel_idx, start:end], label="Real", linewidth=1.5)
            ax.plot(
                time_axis,
                simulated_eeg[channel_idx, start:end],
                label="Simulated",
                linewidth=1.2,
                alpha=0.8,
            )
            ch_label = channels[channel_idx] if channels and channel_idx < len(channels) else f"Ch {channel_idx+1}"
            ax.set_ylabel(ch_label)
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Time (samples)")
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    @staticmethod
    def plot_eeg_timeseries(
        eeg_data: np.ndarray,
        channels: Optional[List[str]] = None,
        max_channels: int = 8,
        title: str = "EEG Time Series",
    ) -> plt.Figure:
        """Plot raw EEG traces for a subset of channels."""
        n_channels, seq_len = eeg_data.shape
        idx = list(range(min(n_channels, max_channels)))

        fig, axes = plt.subplots(len(idx), 1, figsize=(10, 2 * len(idx)), sharex=True)
        if len(idx) == 1:
            axes = [axes]

        time_axis = np.arange(seq_len)

        for ax, channel_idx in zip(axes, idx):
            ax.plot(time_axis, eeg_data[channel_idx], linewidth=1.2)
            ch_label = channels[channel_idx] if channels and channel_idx < len(channels) else f"Ch {channel_idx+1}"
            ax.set_ylabel(ch_label)
            ax.grid(alpha=0.3)

        axes[-1].set_xlabel("Time (samples)")
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig


class NetworkVisualizer:
    """Visualization tools for network analysis."""
    
    @staticmethod
    def plot_connectivity_matrix(
        connectivity: np.ndarray,
        title: str = "Connectivity Matrix",
        cmap: str = "viridis",
    ) -> plt.Figure:
        """Plot a heatmap of the connectivity matrix."""
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(connectivity, cmap=cmap, aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Target Neuron")
        ax.set_ylabel("Source Neuron")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Weight")
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_spike_raster(
        spike_trains: np.ndarray,
        title: str = "Spike Raster",
        max_neurons: int = 128,
    ) -> plt.Figure:
        """Plot a spike raster diagram."""
        n_neurons, seq_len = spike_trains.shape
        neuron_indices = np.arange(min(n_neurons, max_neurons))
        spike_events = [
            np.where(spike_trains[i] > 0)[0]
            for i in neuron_indices
        ]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.eventplot(spike_events, colors="black", lineoffsets=neuron_indices, linelengths=0.8)
        ax.set_ylabel("Neuron Index")
        ax.set_xlabel("Time (samples)")
        ax.set_title(title)
        ax.set_ylim(-1, len(neuron_indices))
        ax.grid(alpha=0.2)
        fig.tight_layout()
        return fig
