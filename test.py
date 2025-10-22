#!/usr/bin/env python3
"""
NGIS Testing Script
Evaluates trained G-SNN models on test set with comprehensive metrics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table
from torch.utils.data import DataLoader

from data.dataloader import create_training_dataloader
from models.gsnn import GSNN
from utils.config import Config
from utils.metrics import compute_eeg_metrics, compute_all_metrics

console = Console()
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="NGIS Model Testing")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pth file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/cluster_full.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="test_results.json",
        help="Output file for test results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run testing on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for testing (default: use config)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed metrics"
    )

    return parser.parse_args()


def load_model(checkpoint_path: str, config: Config, device: str) -> GSNN:
    """Load trained model from checkpoint."""
    console.print(f"[cyan]Loading checkpoint from {checkpoint_path}...[/cyan]")

    # Initialize model
    model_config = config.model
    model = GSNN(
        n_channels=model_config.n_channels,
        n_neurons=model_config.n_neurons,
        n_layers=model_config.n_layers,
        hidden_dim=model_config.hidden_dim,
        graph_type=model_config.graph_type,
        connectivity_threshold=model_config.connectivity_threshold,
        dropout=model_config.dropout,
    ).to(device)

    # Load checkpoint (allow Config class for backward compatibility)
    torch.serialization.add_safe_globals([Config])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Handle DDP checkpoints (state_dict might have 'module.' prefix)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()

    # Print checkpoint info
    if 'epoch' in checkpoint:
        console.print(f"[green]✓[/green] Loaded model from epoch {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        console.print(f"[green]✓[/green] Validation loss: {checkpoint['val_loss']:.4f}")

    return model


def evaluate_model(
    model: GSNN,
    dataloader: DataLoader,
    device: str,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate model on test set.

    Returns:
        Dictionary of aggregated metrics across all test samples.
    """
    model.eval()

    all_metrics = []
    total_samples = 0

    console.print("\n[cyan]Evaluating on test set...[/cyan]")

    with torch.no_grad():
        for batch_idx, batch in enumerate(track(
            dataloader,
            description="Testing",
            total=len(dataloader)
        )):
            # Move to device
            eeg_input = batch["eeg"].to(device)
            batch_size = eeg_input.shape[0]
            total_samples += batch_size

            # Forward pass
            outputs = model(
                eeg_input=eeg_input,
                return_spikes=True,
                return_graph=True
            )

            # Compute metrics for this batch
            batch_metrics = compute_all_metrics(
                real_eeg=eeg_input,
                simulated_eeg=outputs["eeg_output"],
                spike_trains=outputs["spike_trains"],
                graph_data=outputs.get("graph_data")
            )

            all_metrics.append(batch_metrics)

            # Print verbose batch info
            if verbose and batch_idx % 10 == 0:
                console.print(
                    f"  Batch {batch_idx}/{len(dataloader)}: "
                    f"Corr={batch_metrics.get('mean_correlation', 0):.3f}, "
                    f"MSE={batch_metrics.get('mse', 0):.4f}"
                )

    # Aggregate metrics across all batches
    aggregated = {}
    metric_keys = all_metrics[0].keys()

    for key in metric_keys:
        values = [m[key] for m in all_metrics if key in m and not np.isnan(m[key])]
        if values:
            aggregated[f"{key}_mean"] = float(np.mean(values))
            aggregated[f"{key}_std"] = float(np.std(values))
            aggregated[f"{key}_min"] = float(np.min(values))
            aggregated[f"{key}_max"] = float(np.max(values))

    aggregated['total_samples'] = total_samples
    aggregated['num_batches'] = len(dataloader)

    return aggregated


def print_results_table(results: Dict[str, float]):
    """Print results in a formatted table."""
    table = Table(title="Test Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=30)
    table.add_column("Value", justify="right", style="green")

    # Key metrics to highlight
    key_metrics = [
        ('mean_correlation_mean', 'Mean Correlation'),
        ('mean_correlation_std', 'Correlation Std'),
        ('mse_mean', 'MSE'),
        ('rmse_mean', 'RMSE'),
        ('nrmse_mean', 'Normalized RMSE'),
        ('psd_similarity_mean', 'PSD Similarity'),
        ('total_samples', 'Total Samples'),
    ]

    for key, display_name in key_metrics:
        if key in results:
            value = results[key]
            if isinstance(value, float):
                if 'correlation' in key or 'similarity' in key:
                    table.add_row(display_name, f"{value:.4f}")
                elif 'std' in key:
                    table.add_row(display_name, f"{value:.4f}")
                else:
                    table.add_row(display_name, f"{value:.6f}")
            else:
                table.add_row(display_name, str(value))

    console.print("\n")
    console.print(table)
    console.print("\n")


def save_results(results: Dict, output_path: str, args):
    """Save results to JSON file."""
    output_data = {
        'test_results': results,
        'config': {
            'checkpoint': args.checkpoint,
            'config_file': args.config,
            'device': args.device,
            'batch_size': args.batch_size,
        }
    }

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    console.print(f"[green]✓[/green] Results saved to {output_path}")


def main():
    """Main testing function."""
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    console.print("\n[bold blue]NGIS Model Testing[/bold blue]\n")

    # Load configuration
    console.print(f"[cyan]Loading configuration from {args.config}...[/cyan]")
    config = Config.from_yaml(args.config)

    # Override batch size if specified
    if args.batch_size:
        config.data.batch_size = args.batch_size

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        console.print(f"[red]✗[/red] Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Load model
    device = torch.device(args.device)
    console.print(f"[cyan]Using device: {device}[/cyan]")

    model = load_model(args.checkpoint, config, device)

    # Create test dataloader
    console.print("\n[cyan]Creating test dataloader...[/cyan]")

    # Get splits configuration
    splits_config = config.data.get('splits_config', 'configs/splits.yaml')

    test_dataloader = create_training_dataloader(
        data_path=config.data.data_path,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        segment_length=config.data.segment_length,
        overlap=config.data.overlap,
        sampling_rate=config.data.sampling_rate,
        target_channels=config.data.channels,
        max_duration=config.data.get('max_duration', None),
        split='test',  # Use test split
        splits_config_path=splits_config
    )

    console.print(f"[green]✓[/green] Test set: {len(test_dataloader)} batches")

    # Set dataset to eval mode (disable augmentation)
    test_dataloader.dataset.eval()

    # Run evaluation
    results = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        verbose=args.verbose
    )

    # Print results
    print_results_table(results)

    # Save results
    save_results(results, args.output, args)

    # Summary
    console.print("\n[bold green]Testing completed successfully![/bold green]\n")

    # Print key takeaways
    corr = results.get('mean_correlation_mean', 0)
    if corr > 0.7:
        console.print("🎉 [green]Excellent correlation (>0.7)![/green]")
    elif corr > 0.5:
        console.print("✓ [yellow]Good correlation (>0.5)[/yellow]")
    elif corr > 0.3:
        console.print("⚠️  [yellow]Moderate correlation (>0.3)[/yellow]")
    else:
        console.print("❌ [red]Low correlation (<0.3) - model needs improvement[/red]")


if __name__ == "__main__":
    main()
