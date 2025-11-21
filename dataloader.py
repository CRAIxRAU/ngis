"""
DataLoader utilities for NGIS.

Provides functions for creating PyTorch DataLoaders with proper
collation, distributed training support, and batch processing.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate

from .dataset import EEGDataset, EEGInferenceDataset, EEGPairedDataset

logger = logging.getLogger(__name__)


def create_dataloader(
    dataset: EEGDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1
) -> DataLoader:
    """
    Create DataLoader for EEG dataset.
    
    Args:
        dataset: EEG dataset.
        batch_size: Batch size.
        shuffle: Whether to shuffle data.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory for faster GPU transfer.
        drop_last: Whether to drop incomplete batches.
        distributed: Whether using distributed training.
        rank: Rank of current process.
        world_size: Total number of processes.
        
    Returns:
        PyTorch DataLoader.
    """
    # Create sampler for distributed training
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            drop_last=True  # Ensure all ranks have exactly same number of samples
        )
        shuffle = False  # Sampler handles shuffling
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=collate_eeg_batch
    )
    
    logger.info(
        f"Created DataLoader with {len(dataloader)} batches "
        f"(batch_size={batch_size}, num_workers={num_workers})"
    )
    
    return dataloader


def collate_eeg_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for EEG batches.
    
    Args:
        batch: List of EEG samples.
        
    Returns:
        Collated batch dictionary.
    """
    # Separate EEG data and info
    eeg_data = [sample['eeg'] for sample in batch]
    info_data = [sample['info'] for sample in batch]
    
    # Stack EEG tensors
    eeg_batch = torch.stack(eeg_data, dim=0)
    
    return {
        'eeg': eeg_batch,
        'info': info_data  # Keep as list since it's not tensor
    }


def collate_paired_eeg_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for paired EEG batches.
    
    Args:
        batch: List of paired EEG samples.
        
    Returns:
        Collated batch dictionary.
    """
    # Separate real and simulated EEG data
    real_eeg = [sample['real_eeg'] for sample in batch]
    simulated_eeg = [sample['simulated_eeg'] for sample in batch]
    real_info = [sample['real_info'] for sample in batch]
    simulated_info = [sample['simulated_info'] for sample in batch]
    
    # Stack tensors
    real_eeg_batch = torch.stack(real_eeg, dim=0)
    simulated_eeg_batch = torch.stack(simulated_eeg, dim=0)
    
    return {
        'real_eeg': real_eeg_batch,
        'simulated_eeg': simulated_eeg_batch,
        'real_info': real_info,
        'simulated_info': simulated_info
    }


def create_training_dataloader(
    data_path: Union[str, List[str]],
    batch_size: int = 32,
    segment_length: int = 1000,
    overlap: float = 0.0,
    augment: bool = True,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **dataset_kwargs
) -> DataLoader:
    """
    Create DataLoader for training.
    
    Args:
        data_path: Path to EEG data.
        batch_size: Batch size.
        segment_length: Length of EEG segments.
        overlap: Overlap between segments.
        augment: Whether to apply data augmentation.
        num_workers: Number of worker processes.
        distributed: Whether using distributed training.
        rank: Rank of current process.
        world_size: Total number of processes.
        **dataset_kwargs: Additional arguments for EEGDataset.
        
    Returns:
        DataLoader for training.
    """
    # Create dataset - when using DistributedSampler, don't pass rank/world_size
    # The sampler will handle data sharding, not the dataset
    dataset = EEGDataset(
        data_path=data_path,
        segment_length=segment_length,
        overlap=overlap,
        augment=augment,
        rank=0 if distributed else rank,  # Always 0 for distributed (sampler handles sharding)
        world_size=1 if distributed else world_size,  # Always 1 for distributed
        **dataset_kwargs
    )
    
    # Create DataLoader
    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        drop_last=distributed  # Drop last batch in distributed mode to avoid NCCL timeout
    )


def create_inference_dataloader(
    data_path: Union[str, List[str]],
    batch_size: int = 16,  # Smaller batch size for inference
    segment_length: int = 1000,
    overlap: float = 0.5,  # Higher overlap for inference
    num_workers: int = 2,  # Fewer workers for inference
    **dataset_kwargs
) -> DataLoader:
    """
    Create DataLoader for inference.
    
    Args:
        data_path: Path to EEG data.
        batch_size: Batch size.
        segment_length: Length of EEG segments.
        overlap: Overlap between segments.
        num_workers: Number of worker processes.
        **dataset_kwargs: Additional arguments for EEGInferenceDataset.
        
    Returns:
        DataLoader for inference.
    """
    # Create dataset
    dataset = EEGInferenceDataset(
        data_path=data_path,
        segment_length=segment_length,
        overlap=overlap,
        **dataset_kwargs
    )
    
    # Create DataLoader
    return create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,  # No shuffling for inference
        num_workers=num_workers,
        distributed=False  # No distributed training for inference
    )


def create_paired_dataloader(
    real_data_path: Union[str, List[str]],
    simulated_data_path: Union[str, List[str]],
    batch_size: int = 32,
    segment_length: int = 1000,
    overlap: float = 0.0,
    num_workers: int = 4,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
    **dataset_kwargs
) -> DataLoader:
    """
    Create DataLoader for paired EEG data.
    
    Args:
        real_data_path: Path to real EEG data.
        simulated_data_path: Path to simulated EEG data.
        batch_size: Batch size.
        segment_length: Length of EEG segments.
        overlap: Overlap between segments.
        num_workers: Number of worker processes.
        distributed: Whether using distributed training.
        rank: Rank of current process.
        world_size: Total number of processes.
        **dataset_kwargs: Additional arguments for EEGPairedDataset.
        
    Returns:
        DataLoader for paired data.
    """
    # Create dataset
    dataset = EEGPairedDataset(
        real_data_path=real_data_path,
        simulated_data_path=simulated_data_path,
        segment_length=segment_length,
        overlap=overlap,
        **dataset_kwargs
    )
    
    # Create DataLoader with custom collate function
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True  # Ensure all ranks have exactly same number of samples
        )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=not distributed,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=distributed,  # Drop last batch in distributed mode to avoid NCCL timeout
        sampler=sampler,
        collate_fn=collate_paired_eeg_batch
    )
    
    logger.info(
        f"Created paired DataLoader with {len(dataloader)} batches "
        f"(batch_size={batch_size}, num_workers={num_workers})"
    )
    
    return dataloader


def get_dataloader_stats(dataloader: DataLoader) -> Dict:
    """
    Get statistics about DataLoader.
    
    Args:
        dataloader: PyTorch DataLoader.
        
    Returns:
        Dictionary with DataLoader statistics.
    """
    dataset = dataloader.dataset
    
    if hasattr(dataset, 'get_statistics'):
        dataset_stats = dataset.get_statistics()
    else:
        dataset_stats = {}
    
    stats = {
        'num_batches': len(dataloader),
        'batch_size': dataloader.batch_size,
        'num_workers': dataloader.num_workers,
        'dataset_size': len(dataset),
        'dataset_stats': dataset_stats
    }
    
    return stats 