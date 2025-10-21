"""
Quick real-data training smoke test for NGIS.

Loads a subset of the OpenNeuro ds003766 dataset, runs the SimpleEEGModel
for a few epochs, and writes a checkpoint. This exercises the data pipeline,
preprocessing, and training loop end to end with real EEG recordings.
"""

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from data.dataset import EEGDataset
from data.dataloader import collate_eeg_batch
from models.simple_eeg_model import SimpleEEGModel

# Path to the resting-state EEG recording for subject 01
DATA_PATH = Path(
    "data/raw/ds003766/sub-01/eeg/sub-01_task-resting_eeg.set"
)

# Default list of the 128 BioSemi channel names used in ds003766
CHANNELS_128: List[str] = [f"E{i}" for i in range(1, 129)]


def select_device() -> torch.device:
    """Pick the best available device in CUDA -> MPS -> CPU order."""
    if torch.cuda.is_available():
        print("Using: CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using: MPS (Apple Silicon)")
        return torch.device("mps")
    print("Using: CPU")
    return torch.device("cpu")


def build_dataloader(
    data_path: Path,
    batch_size: int,
    overlap: float,
    subset_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a dataloader over a small subset of real EEG segments."""
    dataset = EEGDataset(
        data_path=str(data_path),
        segment_length=1000,
        overlap=overlap,
        preprocess=True,
        augment=False,
        channels=CHANNELS_128,
        sampling_rate=1000,
        bandpass_freq=(1.0, 40.0),
        normalize=True,
    )

    if subset_size < len(dataset):
        indices = list(range(subset_size))
        dataset = Subset(dataset, indices)
        print(f"Using subset of {subset_size} segments from {len(indices)} available")
    else:
        print(f"Using all {len(dataset)} segments")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_eeg_batch,
    )

    print(f"Created dataloader with {len(dataloader)} batches")
    return dataloader


def main() -> None:
    print("=" * 60)
    print("Quick Training Test (Real EEG Data)")
    print("=" * 60)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"EEG file not found at {DATA_PATH}. Make sure the dataset is downloaded."
        )

    device = select_device()

    batch_size = 2
    epochs = 50
    train_subset = 32
    val_subset = 8

    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Training subset segments: {train_subset}")
    print(f"Validation subset segments: {val_subset}")

    print("\nPreparing dataloaders...")
    train_loader = build_dataloader(
        DATA_PATH, batch_size=batch_size, overlap=0.5, subset_size=train_subset, shuffle=True
    )
    val_loader = build_dataloader(
        DATA_PATH, batch_size=batch_size, overlap=0.0, subset_size=val_subset, shuffle=False
    )

    print("\nInstantiating model...")
    model = SimpleEEGModel(n_channels=128, hidden_dim=64, dropout=0.1).to(device)
    param_count = model.count_parameters()
    print(f"Trainable parameters: {param_count['trainable_parameters']:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            eeg = batch["eeg"].to(device)
            optimizer.zero_grad()
            outputs = model(eeg)
            loss = criterion(outputs["eeg_output"], eeg)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 5 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, "
                    f"Loss: {loss.item():.4f}"
                )

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch["eeg"].to(device)
                outputs = model(eeg)
                loss = criterion(outputs["eeg_output"], eeg)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print("=" * 60)
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print("=" * 60)

    print("\nTraining completed successfully! Saving checkpoint...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "quick_model_real.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
