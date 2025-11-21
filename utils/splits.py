"""
Data split utilities for NGIS.

Handles train/validation/test splits at the subject level to ensure
proper generalization assessment.
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Set

import yaml

logger = logging.getLogger(__name__)


def load_splits(config_path: str = "configs/splits.yaml") -> Dict[str, List[int]]:
    """
    Load subject splits from configuration file.
    
    Args:
        config_path: Path to splits configuration file.
        
    Returns:
        Dictionary with 'train', 'validation', and 'test' keys containing
        lists of subject IDs.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Splits config not found at {config_path}, using default single-subject split")
        # Default: use subject 1 for train, no val/test
        return {
            'train': [1],
            'validation': [],
            'test': []
        }
    
    with open(config_path, 'r') as f:
        splits = yaml.safe_load(f)
    
    # Ensure all required keys exist
    for key in ['train', 'validation', 'test']:
        if key not in splits:
            splits[key] = []
    
    logger.info(f"Loaded splits: {len(splits['train'])} train, "
                f"{len(splits['validation'])} val, {len(splits['test'])} test subjects")
    
    return splits


def get_subject_id_from_filename(filename: str) -> int:
    """
    Extract subject ID from filename.
    
    Examples:
        'sub-01_task-resting_eeg.set' -> 1
        'sub-15_task-motor_eeg.set' -> 15
        'subject_003.edf' -> 3
    
    Args:
        filename: EEG filename.
        
    Returns:
        Subject ID as integer, or 0 if not found.
    """
    # Try standard BIDS format: sub-XX
    match = re.search(r'sub-(\d+)', filename)
    if match:
        return int(match.group(1))
    
    # Try subject_XXX format
    match = re.search(r'subject[_-](\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    # Try S followed by digits
    match = re.search(r's(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    
    logger.warning(f"Could not extract subject ID from filename: {filename}")
    return 0


def filter_files_by_split(files: List[Path], split: str, 
                          splits_config: Dict[str, List[int]]) -> List[Path]:
    """
    Filter list of files to only include those from specified split.
    
    Args:
        files: List of file paths.
        split: 'train', 'validation', or 'test'.
        splits_config: Dictionary of splits from load_splits().
        
    Returns:
        Filtered list of file paths.
    """
    if split not in splits_config:
        logger.error(f"Unknown split: {split}. Valid splits: {list(splits_config.keys())}")
        return []
    
    allowed_subjects = set(splits_config[split])
    
    if not allowed_subjects:
        logger.warning(f"No subjects defined for split '{split}'")
        return []
    
    filtered_files = []
    for file_path in files:
        subject_id = get_subject_id_from_filename(file_path.name)
        if subject_id in allowed_subjects:
            filtered_files.append(file_path)
    
    logger.info(f"Split '{split}': filtered {len(files)} files -> {len(filtered_files)} files "
                f"(subjects: {sorted(allowed_subjects)})")
    
    return filtered_files


def get_split_subjects(data_path: str, split: str, 
                       splits_config_path: str = "configs/splits.yaml") -> Set[int]:
    """
    Get set of subject IDs that should be included in a split.
    
    Args:
        data_path: Path to data directory.
        split: 'train', 'validation', or 'test'.
        splits_config_path: Path to splits configuration.
        
    Returns:
        Set of subject IDs for this split.
    """
    splits = load_splits(splits_config_path)
    return set(splits.get(split, []))


def validate_splits(data_path: str, splits_config_path: str = "configs/splits.yaml") -> Dict:
    """
    Validate that splits configuration matches available data.
    
    Args:
        data_path: Path to data directory.
        splits_config_path: Path to splits configuration.
        
    Returns:
        Dictionary with validation results and statistics.
    """
    data_path = Path(data_path)
    splits = load_splits(splits_config_path)
    
    # Find all available subjects
    available_subjects = set()
    if data_path.exists():
        for file_path in data_path.glob("**/*.set"):
            subject_id = get_subject_id_from_filename(file_path.name)
            if subject_id > 0:
                available_subjects.add(subject_id)
        
        # Also check other formats
        for pattern in ["**/*.edf", "**/*.fif", "**/*.bdf"]:
            for file_path in data_path.glob(pattern):
                subject_id = get_subject_id_from_filename(file_path.name)
                if subject_id > 0:
                    available_subjects.add(subject_id)
    
    # Check each split
    results = {
        'available_subjects': sorted(available_subjects),
        'total_available': len(available_subjects),
        'splits': {}
    }
    
    for split_name, subject_ids in splits.items():
        split_subjects = set(subject_ids)
        available_in_split = split_subjects & available_subjects
        missing_in_split = split_subjects - available_subjects
        
        results['splits'][split_name] = {
            'configured': sorted(split_subjects),
            'available': sorted(available_in_split),
            'missing': sorted(missing_in_split),
            'count_configured': len(split_subjects),
            'count_available': len(available_in_split),
            'count_missing': len(missing_in_split)
        }
        
        if missing_in_split:
            logger.warning(f"Split '{split_name}': {len(missing_in_split)} subjects missing from data: {sorted(missing_in_split)}")
    
    # Check for subjects not in any split
    all_split_subjects = set()
    for subject_ids in splits.values():
        all_split_subjects.update(subject_ids)
    
    unassigned = available_subjects - all_split_subjects
    if unassigned:
        logger.warning(f"Found {len(unassigned)} subjects not assigned to any split: {sorted(unassigned)}")
        results['unassigned_subjects'] = sorted(unassigned)
    
    # Check for overlaps between splits
    train_subjects = set(splits.get('train', []))
    val_subjects = set(splits.get('validation', []))
    test_subjects = set(splits.get('test', []))
    
    overlaps = []
    if train_subjects & val_subjects:
        overlaps.append(('train', 'validation', sorted(train_subjects & val_subjects)))
    if train_subjects & test_subjects:
        overlaps.append(('train', 'test', sorted(train_subjects & test_subjects)))
    if val_subjects & test_subjects:
        overlaps.append(('validation', 'test', sorted(val_subjects & test_subjects)))
    
    if overlaps:
        logger.error(f"Found overlapping subjects between splits: {overlaps}")
        results['overlaps'] = overlaps
    
    return results

