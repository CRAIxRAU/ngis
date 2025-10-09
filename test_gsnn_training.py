"""
Quick training harness for the full G-SNN model on real EEG data.

This script mirrors `test_quick_training.py` but uses the graph-structured
spiking neural network (GSNN) to exercise the end-to-end pipeline with
real recordings from OpenNeuro ds003766.
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from data.dataset import EEGDataset
from data.dataloader import collate_eeg_batch
from models.gsnn import GSNN
from training.loss_functions import CombinedLoss
from utils.visualization import EEGVisualizer, NetworkVisualizer

REAL_DATA_PATH = Path(
    "data/raw/ds003766/sub-01/eeg/sub-01_task-resting_eeg.set"
)
CHANNELS_128: List[str] = [f"E{i}" for i in range(1, 129)]


def select_device() -> torch.device:
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
    print("GSNN Training Test (Real EEG Data)")
    print("=" * 60)

    if not REAL_DATA_PATH.exists():
        raise FileNotFoundError(
            f"EEG file not found at {REAL_DATA_PATH}. Ensure the dataset is available."
        )

    device = select_device()

    batch_size = 2
    epochs = 5
    train_subset = 32
    val_subset = 8

    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"Training subset segments: {train_subset}")
    print(f"Validation subset segments: {val_subset}")

    print("\nPreparing dataloaders...")
    train_loader = build_dataloader(
        REAL_DATA_PATH, batch_size=batch_size, overlap=0.5, subset_size=train_subset, shuffle=True
    )
    val_loader = build_dataloader(
        REAL_DATA_PATH, batch_size=batch_size, overlap=0.0, subset_size=val_subset, shuffle=False
    )

    print("\nInstantiating GSNN model...")
    model = GSNN(
        n_channels=128,
        n_neurons=256,
        n_layers=2,
        hidden_dim=64,
        graph_type="functional",
        connectivity_threshold=0.1,
        dropout=0.1,
    ).to(device)

    param_count = model.count_parameters()
    print(f"Trainable parameters: {param_count['trainable_parameters']:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = CombinedLoss(
        eeg_weight=1.0,
        spiking_weight=0.1,
        biological_weight=0.0,
        regularization_weight=1e-4
    )

    print("\n" + "=" * 60)
    print("Training...")
    print("=" * 60)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            eeg = batch["eeg"].to(device)

            optimizer.zero_grad()
            outputs = model(eeg, return_spikes=True, return_graph=False)
            loss = criterion(
                real_eeg=eeg,
                simulated_eeg=outputs["eeg_output"],
                spike_trains=outputs["spike_trains"],
                graph_data=outputs.get("graph_data"),
                model=model
            )
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            if batch_idx % 5 == 0:
                print(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}"
                )

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch["eeg"].to(device)
                outputs = model(eeg, return_spikes=True, return_graph=False)
                loss = criterion(
                    real_eeg=eeg,
                    simulated_eeg=outputs["eeg_output"],
                    spike_trains=outputs["spike_trains"],
                    graph_data=outputs.get("graph_data"),
                    model=model
                )
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print("=" * 60)
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        print("=" * 60)

    print("\nTraining completed successfully! Saving checkpoint and visualizations...")
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    checkpoint_path = output_dir / "gsnn_quick_model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model saved to: {checkpoint_path}")

    # Generate qualitative visualizations using one validation batch
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        eeg = sample_batch["eeg"].to(device)
        outputs = model(eeg, return_spikes=True, return_graph=True)

    real_eeg = eeg[0].cpu().numpy()
    simulated_eeg = outputs["eeg_output"][0].cpu().numpy()
    spike_trains = outputs["spike_trains"][0].cpu().numpy()
    connectivity = model.get_connectivity_matrix().detach().cpu().numpy()

    # Save figures
    comparison_fig = EEGVisualizer.plot_eeg_comparison(real_eeg, simulated_eeg, channels=CHANNELS_128)
    comparison_path = output_dir / "gsnn_eeg_comparison.png"
    comparison_fig.savefig(comparison_path, dpi=200)
    plt.close(comparison_fig)

    raster_fig = NetworkVisualizer.plot_spike_raster(spike_trains)
    raster_path = output_dir / "gsnn_spike_raster.png"
    raster_fig.savefig(raster_path, dpi=200)
    plt.close(raster_fig)

    connectivity_fig = NetworkVisualizer.plot_connectivity_matrix(connectivity)
    connectivity_path = output_dir / "gsnn_connectivity.png"
    connectivity_fig.savefig(connectivity_path, dpi=200)
    plt.close(connectivity_fig)

    print(f"Saved comparison plot to: {comparison_path}")
    print(f"Saved spike raster to: {raster_path}")
    print(f"Saved connectivity heatmap to: {connectivity_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
