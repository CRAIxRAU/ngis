"""Test script to verify all ranks see same number of batches."""
from data.dataset import EEGDataset
from data.dataloader import create_training_dataloader
import yaml
import gc

with open('configs/cluster_test_10epochs.yaml') as f:
    config = yaml.safe_load(f)

print("Testing dataloader batch counts for all ranks:")
print("=" * 60)

for rank in range(4):
    dl = create_training_dataloader(
        data_path=config['data']['data_path'],
        batch_size=4,
        segment_length=1000,
        overlap=0.0,
        augment=False,
        num_workers=0,  # Use 0 workers to avoid memory issues
        distributed=True,
        rank=rank,
        world_size=4,
        channel_selection_strategy='uniform_spatial',
        target_channels=128
    )
    dataset_size = len(dl.dataset)
    num_batches = len(dl)

    print(f'Rank {rank}: {num_batches} batches, {dataset_size} samples in dataset')

    # Clean up to free memory
    del dl
    gc.collect()

print("=" * 60)
print("\nIf all ranks show the same number of batches, drop_last is working!")
print("If they differ, we have a problem with data distribution.")