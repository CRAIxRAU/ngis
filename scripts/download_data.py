#!/usr/bin/env python3
"""
Download HD-EEG datasets from OpenNeuro.

Usage:
    python scripts/download_data.py --sample  # Download 2 subjects for testing
    python scripts/download_data.py           # Download all subjects
"""

import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Dataset configurations
DATASETS = {
    "ds003420": "https://github.com/OpenNeuroDatasets/ds003420",  # naming/spelling
    "ds003421": "https://github.com/OpenNeuroDatasets/ds003421",  # auditory/memory/rest
}

def check_datalad() -> bool:
    """Check if DataLad is installed."""
    try:
        subprocess.run(["datalad", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("DataLad not found. Install with: pip install datalad")
        return False

def download_dataset(dataset_id: str, subjects_limit: int = None) -> bool:
    """Download OpenNeuro dataset using DataLad."""
    url = DATASETS[dataset_id]
    data_dir = Path("data/raw")
    dataset_path = data_dir / dataset_id
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Installing {dataset_id}...")
    try:
        # Install dataset
        subprocess.run([
            "datalad", "install", "-s", url, str(dataset_path)
        ], check=True)
        
        # Get list of subjects
        subjects = sorted([d.name for d in dataset_path.glob("sub-*") if d.is_dir()])
        
        if subjects_limit:
            subjects = subjects[:subjects_limit]
            
        logger.info(f"Downloading data for {len(subjects)} subjects...")
        
        # Download EEG data for each subject
        for subject in subjects:
            logger.info(f"Downloading {subject}...")
            subprocess.run([
                "datalad", "get", f"{subject}/eeg/*.bdf", f"{subject}/eeg/*.json"
            ], cwd=dataset_path, check=True)
            
        logger.info(f"✓ Downloaded {dataset_id}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Failed to download {dataset_id}: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download HD-EEG datasets")
    parser.add_argument("--sample", action="store_true", help="Download sample (2 subjects only)")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if not check_datalad():
        sys.exit(1)
        
    subjects_limit = 2 if args.sample else None
    
    success = True
    for dataset_id in DATASETS:
        if not download_dataset(dataset_id, subjects_limit):
            success = False
            
    if success:
        logger.info("✓ All datasets downloaded successfully")
        logger.info("Data location: data/raw/")
    else:
        logger.error("✗ Some downloads failed")
        sys.exit(1)

if __name__ == "__main__":
    main()