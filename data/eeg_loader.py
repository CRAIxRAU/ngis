"""
EEG data loading utilities for NGIS.

Handles loading of raw EEG data from various formats including EDF, BDF, 
and other common EEG file formats.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
from mne.io import read_raw_edf, read_raw_bdf, read_raw_fif

logger = logging.getLogger(__name__)


class EEGLoader:
    """
    Loader for EEG data from various file formats.
    
    Supports EDF, BDF, FIF, and other common EEG formats.
    Handles channel selection, resampling, and basic preprocessing.
    """
    
    def __init__(
        self,
        channels: Optional[List[str]] = None,
        sampling_rate: Optional[int] = None,
        preload: bool = True
    ):
        """
        Initialize EEG loader.
        
        Args:
            channels: List of channel names to load. If None, loads all channels.
            sampling_rate: Target sampling rate. If None, keeps original rate.
            preload: Whether to preload data into memory.
        """
        self.channels = channels
        self.sampling_rate = sampling_rate
        self.preload = preload
        self.supported_formats = ['.edf', '.bdf', '.fif', '.set', '.cnt']
    
    def load_file(self, file_path: Union[str, Path]) -> mne.io.Raw:
        """
        Load EEG data from file.
        
        Args:
            file_path: Path to EEG file.
            
        Returns:
            MNE Raw object containing EEG data.
            
        Raises:
            ValueError: If file format is not supported.
            FileNotFoundError: If file does not exist.
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"EEG file not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        logger.info(f"Loading EEG data from: {file_path}")
        
        try:
            if file_ext == '.edf':
                raw = read_raw_edf(file_path, preload=self.preload)
            elif file_ext == '.bdf':
                raw = read_raw_bdf(file_path, preload=self.preload)
            elif file_ext == '.fif':
                raw = read_raw_fif(file_path, preload=self.preload)
            else:
                # For other formats, try MNE's generic reader
                raw = mne.io.read_raw(file_path, preload=self.preload)
        except Exception as e:
            logger.error(f"Failed to load EEG file {file_path}: {e}")
            raise
        
        # Select channels if specified
        if self.channels is not None:
            available_channels = raw.ch_names
            missing_channels = set(self.channels) - set(available_channels)
            if missing_channels:
                logger.warning(f"Missing channels: {missing_channels}")
                # Use only available channels
                self.channels = [ch for ch in self.channels if ch in available_channels]
            raw.pick_channels(self.channels)
        
        # Resample if specified
        if self.sampling_rate is not None and self.sampling_rate != raw.info['sfreq']:
            logger.info(f"Resampling from {raw.info['sfreq']}Hz to {self.sampling_rate}Hz")
            raw.resample(self.sampling_rate)
        
        logger.info(f"Loaded EEG data: {raw.n_times} samples, {raw.n_channels} channels")
        return raw
    
    def load_directory(
        self, 
        data_dir: Union[str, Path],
        file_pattern: str = "*.edf"
    ) -> Dict[str, mne.io.Raw]:
        """
        Load all EEG files from a directory.
        
        Args:
            data_dir: Directory containing EEG files.
            file_pattern: Glob pattern for file selection.
            
        Returns:
            Dictionary mapping filenames to Raw objects.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        files = list(data_dir.glob(file_pattern))
        if not files:
            raise ValueError(f"No files matching pattern '{file_pattern}' found in {data_dir}")
        
        logger.info(f"Found {len(files)} EEG files in {data_dir}")
        
        raw_data = {}
        for file_path in files:
            try:
                raw = self.load_file(file_path)
                raw_data[file_path.stem] = raw
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(raw_data)} files")
        return raw_data
    
    def get_data_array(self, raw: mne.io.Raw) -> np.ndarray:
        """
        Extract data array from MNE Raw object.
        
        Args:
            raw: MNE Raw object.
            
        Returns:
            Numpy array of shape (n_channels, n_samples).
        """
        return raw.get_data()
    
    def get_info(self, raw: mne.io.Raw) -> Dict:
        """
        Extract metadata from MNE Raw object.
        
        Args:
            raw: MNE Raw object.
            
        Returns:
            Dictionary containing EEG metadata.
        """
        info = {
            'n_channels': raw.n_channels,
            'n_samples': raw.n_times,
            'sampling_rate': raw.info['sfreq'],
            'duration': raw.n_times / raw.info['sfreq'],
            'channel_names': raw.ch_names,
            'bad_channels': raw.info['bads']
        }
        return info
    
    def validate_data(self, raw: mne.io.Raw) -> bool:
        """
        Validate EEG data quality.
        
        Args:
            raw: MNE Raw object.
            
        Returns:
            True if data passes validation checks.
        """
        # Check for reasonable sampling rate
        if raw.info['sfreq'] < 100 or raw.info['sfreq'] > 10000:
            logger.warning(f"Unusual sampling rate: {raw.info['sfreq']}Hz")
            return False
        
        # Check for reasonable number of channels
        if raw.n_channels < 16 or raw.n_channels > 256:
            logger.warning(f"Unusual number of channels: {raw.n_channels}")
            return False
        
        # Check for reasonable duration
        duration = raw.n_times / raw.info['sfreq']
        if duration < 1 or duration > 3600:  # 1 second to 1 hour
            logger.warning(f"Unusual duration: {duration}s")
            return False
        
        return True 