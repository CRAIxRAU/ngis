"""
EEG preprocessing utilities for NGIS.

Handles filtering, artifact removal, and signal processing for EEG data.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
from scipy import signal
from scipy.stats import zscore

logger = logging.getLogger(__name__)


class EEGPreprocessor:
    """
    Preprocessor for EEG data.
    
    Handles filtering, artifact removal, normalization, and other
    preprocessing steps required for G-SNN training.
    """
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        notch_freq: Optional[float] = 50.0,  # Power line frequency
        bandpass_freq: Tuple[float, float] = (1.0, 40.0),
        normalize: bool = True,
        remove_artifacts: bool = True,
        artifact_threshold: float = 3.0
    ):
        """
        Initialize EEG preprocessor.
        
        Args:
            sampling_rate: Sampling rate of the EEG data.
            notch_freq: Frequency for notch filter (usually 50 or 60 Hz).
            bandpass_freq: (low, high) frequency range for bandpass filter.
            normalize: Whether to normalize the data.
            remove_artifacts: Whether to remove artifacts.
            artifact_threshold: Z-score threshold for artifact detection.
        """
        self.sampling_rate = sampling_rate
        self.notch_freq = notch_freq
        self.bandpass_freq = bandpass_freq
        self.normalize = normalize
        self.remove_artifacts = remove_artifacts
        self.artifact_threshold = artifact_threshold
    
    def preprocess_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """
        Preprocess MNE Raw object.
        
        Args:
            raw: MNE Raw object.
            
        Returns:
            Preprocessed MNE Raw object.
        """
        logger.info("Starting EEG preprocessing...")
        
        # Create a copy to avoid modifying original
        raw_processed = raw.copy()
        
        # Apply notch filter
        if self.notch_freq is not None:
            raw_processed = self._apply_notch_filter(raw_processed)
        
        # Apply bandpass filter
        raw_processed = self._apply_bandpass_filter(raw_processed)
        
        # Remove artifacts
        if self.remove_artifacts:
            raw_processed = self._remove_artifacts(raw_processed)
        
        # Normalize
        if self.normalize:
            raw_processed = self._normalize_data(raw_processed)
        
        logger.info("EEG preprocessing completed")
        return raw_processed
    
    def preprocess_array(
        self, 
        data: np.ndarray,
        channel_names: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Preprocess numpy array of EEG data.
        
        Args:
            data: EEG data array of shape (n_channels, n_samples).
            channel_names: List of channel names.
            
        Returns:
            Preprocessed EEG data array.
        """
        logger.info("Starting array-based EEG preprocessing...")
        
        # Create temporary MNE Raw object
        if channel_names is None:
            channel_names = [f"EEG{i:03d}" for i in range(data.shape[0])]
        
        info = mne.create_info(
            ch_names=channel_names,
            sfreq=self.sampling_rate,
            ch_types=['eeg'] * len(channel_names)
        )
        
        raw = mne.io.RawArray(data, info)
        raw_processed = self.preprocess_raw(raw)
        
        return raw_processed.get_data()
    
    def _apply_notch_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply notch filter to remove power line interference."""
        logger.info(f"Applying notch filter at {self.notch_freq}Hz")
        
        # Apply notch filter
        raw.notch_filter(
            freqs=self.notch_freq,
            picks='eeg',
            method='fir',
            phase='zero'
        )
        
        return raw
    
    def _apply_bandpass_filter(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply bandpass filter to focus on relevant frequency range."""
        logger.info(f"Applying bandpass filter: {self.bandpass_freq[0]}-{self.bandpass_freq[1]}Hz")
        
        # Apply bandpass filter
        raw.filter(
            l_freq=self.bandpass_freq[0],
            h_freq=self.bandpass_freq[1],
            picks='eeg',
            method='fir',
            phase='zero'
        )
        
        return raw
    
    def _remove_artifacts(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Remove artifacts using statistical methods."""
        logger.info("Removing artifacts...")

        data = raw.get_data()
        n_channels, n_samples = data.shape

        # Calculate z-scores for each channel (handle constant channels)
        z_scores = np.abs(zscore(data, axis=1, nan_policy='propagate'))
        z_scores = np.nan_to_num(z_scores, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Find samples that exceed threshold in any channel
        artifact_mask = np.any(z_scores > self.artifact_threshold, axis=0)
        
        # Interpolate artifact segments
        for ch_idx in range(n_channels):
            channel_data = data[ch_idx, :]
            artifact_indices = np.where(z_scores[ch_idx, :] > self.artifact_threshold)[0]
            
            if len(artifact_indices) > 0:
                # Simple interpolation for artifacts
                for idx in artifact_indices:
                    if idx > 0 and idx < n_samples - 1:
                        # Linear interpolation
                        data[ch_idx, idx] = (
                            data[ch_idx, idx - 1] + data[ch_idx, idx + 1]
                        ) / 2
        
        # Update the raw object
        raw._data = data
        
        logger.info(f"Removed artifacts from {np.sum(artifact_mask)} samples")
        return raw
    
    def _normalize_data(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Normalize EEG data."""
        logger.info("Normalizing EEG data...")

        data = raw.get_data()

        # Z-score normalization per channel (handle constant channels)
        data_normalized = zscore(data, axis=1, nan_policy='propagate')
        data_normalized = np.nan_to_num(data_normalized, nan=0.0, posinf=0.0, neginf=0.0)

        # Update the raw object
        raw._data = data_normalized

        return raw
    
    def segment_data(
        self,
        data: np.ndarray,
        segment_length: int,
        overlap: float = 0.0
    ) -> np.ndarray:
        """
        Segment EEG data into fixed-length windows.
        
        Args:
            data: EEG data array of shape (n_channels, n_samples).
            segment_length: Length of each segment in samples.
            overlap: Overlap between segments (0.0 to 1.0).
            
        Returns:
            Segmented data array of shape (n_segments, n_channels, segment_length).
        """
        n_channels, n_samples = data.shape
        
        # Calculate step size
        step_size = int(segment_length * (1 - overlap))
        
        # Calculate number of segments
        n_segments = max(1, (n_samples - segment_length) // step_size + 1)
        
        segments = []
        for i in range(n_segments):
            start_idx = i * step_size
            end_idx = start_idx + segment_length
            
            if end_idx <= n_samples:
                segment = data[:, start_idx:end_idx]
                segments.append(segment)
        
        # Return compact float32 to reduce memory footprint
        return np.array(segments, dtype=np.float32)
    
    def extract_features(
        self,
        data: np.ndarray,
        feature_types: List[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        Extract features from EEG data.
        
        Args:
            data: EEG data array of shape (n_channels, n_samples).
            feature_types: List of feature types to extract.
            
        Returns:
            Dictionary of extracted features.
        """
        if feature_types is None:
            feature_types = ['power', 'phase', 'connectivity']
        
        features = {}
        
        for feature_type in feature_types:
            if feature_type == 'power':
                features['power'] = self._extract_power_features(data)
            elif feature_type == 'phase':
                features['phase'] = self._extract_phase_features(data)
            elif feature_type == 'connectivity':
                features['connectivity'] = self._extract_connectivity_features(data)
        
        return features
    
    def _extract_power_features(self, data: np.ndarray) -> np.ndarray:
        """Extract power spectral density features."""
        # Calculate power spectral density
        freqs, psd = signal.welch(data, fs=self.sampling_rate, axis=1)
        
        # Extract power in different frequency bands
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        power_features = {}
        for band_name, (low_freq, high_freq) in bands.items():
            # Find frequency indices
            freq_mask = (freqs >= low_freq) & (freqs <= high_freq)
            power_features[band_name] = np.mean(psd[:, freq_mask], axis=1)
        
        return power_features
    
    def _extract_phase_features(self, data: np.ndarray) -> np.ndarray:
        """Extract phase features using Hilbert transform."""
        # Apply Hilbert transform to get analytic signal
        analytic_signal = signal.hilbert(data, axis=1)
        
        # Extract phase
        phase = np.angle(analytic_signal)
        
        return phase
    
    def _extract_connectivity_features(self, data: np.ndarray) -> np.ndarray:
        """Extract connectivity features between channels."""
        n_channels = data.shape[0]
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(data)
        
        # Calculate coherence (simplified)
        coherence_matrix = np.zeros((n_channels, n_channels))
        for i in range(n_channels):
            for j in range(i + 1, n_channels):
                freqs, coh = signal.coherence(
                    data[i, :], 
                    data[j, :], 
                    fs=self.sampling_rate
                )
                coherence_matrix[i, j] = np.mean(coh)
                coherence_matrix[j, i] = coherence_matrix[i, j]
        
        return {
            'correlation': correlation_matrix,
            'coherence': coherence_matrix
        } 
