"""
EEG dataset implementation for NGIS.

Provides PyTorch Dataset classes for EEG data with segmentation,
augmentation, and preprocessing capabilities.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from .eeg_loader import EEGLoader
from .preprocessor import EEGPreprocessor

logger = logging.getLogger(__name__)


class EEGDataset(Dataset):
    """
    PyTorch Dataset for EEG data.
    
    Handles loading, preprocessing, and segmentation of EEG data
    for training G-SNN models.
    """
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        segment_length: int = 1000,  # 1 second at 1000Hz
        overlap: float = 0.0,
        preprocess: bool = True,
        augment: bool = False,
        channels: Optional[List[str]] = None,
        sampling_rate: float = 1000.0,
        channel_selection_strategy: Optional[str] = None,
        target_channels: int = 128,
        max_duration: Optional[float] = None,
        rank: int = 0,
        world_size: int = 1,
        split_subjects: Optional[set] = None,
        **preprocessor_kwargs
    ):
        """
        Initialize EEG dataset.

        Args:
            data_path: Path to EEG file(s) or directory.
            segment_length: Length of each segment in samples.
            overlap: Overlap between segments (0.0 to 1.0).
            preprocess: Whether to apply preprocessing.
            augment: Whether to apply data augmentation.
            channels: List of channel names to use.
            sampling_rate: Target sampling rate.
            max_duration: Maximum duration in seconds to load (for fast testing).
            rank: Process rank for distributed training (0 to world_size-1).
            world_size: Total number of processes in distributed training.
            **preprocessor_kwargs: Additional arguments for preprocessor.
        """
        self.segment_length = segment_length
        self.overlap = overlap
        self.preprocess = preprocess
        self.augment = augment
        self.sampling_rate = sampling_rate
        self.rank = rank
        self.world_size = world_size
        self.split_subjects = split_subjects
        self.training = True  # Default to training mode

        # Initialize loader and preprocessor
        # Extract preload parameter from kwargs (default True if not specified)
        preload = preprocessor_kwargs.pop('preload', True)

        self.loader = EEGLoader(
            channels=channels,
            sampling_rate=sampling_rate,
            channel_selection_strategy=channel_selection_strategy,
            target_channels=target_channels,
            max_duration=max_duration,
            preload=preload  # Pass preload to avoid OOM
        )
        
        self.preprocessor = EEGPreprocessor(
            sampling_rate=sampling_rate,
            **preprocessor_kwargs
        )
        
        # Load and process data
        self._load_data(data_path)
        self._segment_data()
        
        logger.info(f"Created EEG dataset with {len(self.segments)} segments")
    
    def _load_data(self, data_path: Union[str, List[str]]):
        """Load EEG data from file(s). All ranks load all files - DistributedSampler handles segment sharding."""
        if isinstance(data_path, str):
            # Single file or directory
            path = data_path
            if path.endswith(('.edf', '.bdf', '.fif', '.set', '.cnt')):
                # Single file - all ranks load it
                raw = self.loader.load_file(path)
                self.raw_data = {'single_file': raw}
            else:
                # Directory - ALL ranks load ALL files (no file-level sharding)
                # DistributedSampler handles segment-level sharding for balanced batches
                self.raw_data = self.loader.load_directory(
                    path,
                    rank=0,  # Dummy value - all ranks load all files
                    world_size=1,  # Dummy value - no file sharding
                    split_subjects=self.split_subjects
                )
        else:
            # List of files - ALL ranks load ALL files (no file-level sharding)
            self.raw_data = {}
            for file_path in data_path:
                try:
                    raw = self.loader.load_file(file_path)
                    self.raw_data[Path(file_path).stem] = raw
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        if not self.raw_data:
            raise ValueError(f"No valid EEG data found in {data_path}")
    
    def _segment_data(self):
        """Segment EEG data into training segments."""
        self.segments = []
        self.segment_info = []
        
        for name, raw in self.raw_data.items():
            # Preprocess if requested
            if self.preprocess:
                raw = self.preprocessor.preprocess_raw(raw)
            
            # Get data array
            data = self.loader.get_data_array(raw)
            
            # Segment data
            segments = self.preprocessor.segment_data(
                data, 
                self.segment_length, 
                self.overlap
            )
            
            # Store segments
            for i, segment in enumerate(segments):
                self.segments.append(segment)
                self.segment_info.append({
                    'source': name,
                    'segment_idx': i,
                    'start_sample': i * int(self.segment_length * (1 - self.overlap)),
                    'end_sample': i * int(self.segment_length * (1 - self.overlap)) + self.segment_length
                })
    
    def __len__(self) -> int:
        """Return number of segments."""
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get EEG segment at index."""
        segment = self.segments[idx]
        info = self.segment_info[idx]

        # Convert to torch tensor
        segment_tensor = torch.from_numpy(segment).float()

        # Handle NaN values from preprocessing (filter edge effects)
        if torch.isnan(segment_tensor).any():
            segment_tensor = torch.nan_to_num(segment_tensor, nan=0.0)
            logger.debug(f"Replaced NaN values in segment {idx}")

        # Apply augmentation if requested
        if self.augment and self.training:
            segment_tensor = self._augment_segment(segment_tensor)

        return {
            'eeg': segment_tensor,
            'info': info
        }
    
    def _augment_segment(self, segment: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation to segment."""
        # Add random noise
        if np.random.random() < 0.3:
            noise = torch.randn_like(segment) * 0.01
            segment = segment + noise
        
        # Random time shift
        if np.random.random() < 0.3:
            shift = np.random.randint(-10, 11)
            if shift > 0:
                segment = torch.cat([segment[:, shift:], segment[:, :shift]], dim=1)
            elif shift < 0:
                segment = torch.cat([segment[:, shift:], segment[:, :shift]], dim=1)
        
        # Random amplitude scaling
        if np.random.random() < 0.3:
            scale = 0.8 + 0.4 * np.random.random()  # 0.8 to 1.2
            segment = segment * scale
        
        return segment

    def train(self):
        """Set dataset to training mode (enables augmentation)."""
        self.training = True

    def eval(self):
        """Set dataset to evaluation mode (disables augmentation)."""
        self.training = False

    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        if not self.segments:
            return {}
        
        # Calculate statistics across all segments
        all_data = np.concatenate(self.segments, axis=1)
        
        stats = {
            'n_segments': len(self.segments),
            'n_channels': self.segments[0].shape[0],
            'segment_length': self.segment_length,
            'total_samples': all_data.shape[1],
            'mean': float(np.mean(all_data)),
            'std': float(np.std(all_data)),
            'min': float(np.min(all_data)),
            'max': float(np.max(all_data))
        }
        
        return stats


class EEGInferenceDataset(Dataset):
    """
    Dataset for EEG inference/prediction.
    
    Similar to EEGDataset but optimized for inference without
    augmentation and with different preprocessing options.
    """
    
    def __init__(
        self,
        data_path: Union[str, List[str]],
        segment_length: int = 1000,
        overlap: float = 0.5,  # Higher overlap for inference
        channels: Optional[List[str]] = None,
        sampling_rate: float = 1000.0,
        **preprocessor_kwargs
    ):
        """
        Initialize EEG inference dataset.
        
        Args:
            data_path: Path to EEG file(s).
            segment_length: Length of each segment in samples.
            overlap: Overlap between segments.
            channels: List of channel names to use.
            sampling_rate: Target sampling rate.
            **preprocessor_kwargs: Additional arguments for preprocessor.
        """
        super().__init__()
        
        # Create regular dataset without augmentation
        self.dataset = EEGDataset(
            data_path=data_path,
            segment_length=segment_length,
            overlap=overlap,
            preprocess=True,
            augment=False,  # No augmentation for inference
            channels=channels,
            sampling_rate=sampling_rate,
            **preprocessor_kwargs
        )
    
    def __len__(self) -> int:
        """Return number of segments."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get EEG segment at index."""
        return self.dataset[idx]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        return self.dataset.get_statistics()


class EEGPairedDataset(Dataset):
    """
    Dataset for paired EEG data (real vs simulated).
    
    Used for training G-SNN models where we have both real EEG
    and simulated EEG from the model.
    """
    
    def __init__(
        self,
        real_data_path: Union[str, List[str]],
        simulated_data_path: Union[str, List[str]],
        segment_length: int = 1000,
        overlap: float = 0.0,
        **kwargs
    ):
        """
        Initialize paired EEG dataset.
        
        Args:
            real_data_path: Path to real EEG data.
            simulated_data_path: Path to simulated EEG data.
            segment_length: Length of each segment in samples.
            overlap: Overlap between segments.
            **kwargs: Additional arguments for EEGDataset.
        """
        self.real_dataset = EEGDataset(
            data_path=real_data_path,
            segment_length=segment_length,
            overlap=overlap,
            **kwargs
        )
        
        self.simulated_dataset = EEGDataset(
            data_path=simulated_data_path,
            segment_length=segment_length,
            overlap=overlap,
            **kwargs
        )
        
        # Ensure datasets have same length
        if len(self.real_dataset) != len(self.simulated_dataset):
            raise ValueError(
                f"Real and simulated datasets have different lengths: "
                f"{len(self.real_dataset)} vs {len(self.simulated_dataset)}"
            )
    
    def __len__(self) -> int:
        """Return number of paired segments."""
        return len(self.real_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get paired EEG segments at index."""
        real_sample = self.real_dataset[idx]
        simulated_sample = self.simulated_dataset[idx]
        
        return {
            'real_eeg': real_sample['eeg'],
            'simulated_eeg': simulated_sample['eeg'],
            'real_info': real_sample['info'],
            'simulated_info': simulated_sample['info']
        }
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics for both real and simulated data."""
        real_stats = self.real_dataset.get_statistics()
        simulated_stats = self.simulated_dataset.get_statistics()
        
        return {
            'real': real_stats,
            'simulated': simulated_stats
        } 