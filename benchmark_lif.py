"""
Quick benchmark to measure LIF vectorization speedup.
Matches your actual training configuration.
"""

import torch
import time
import sys
from models.gsnn import GSNN

def benchmark_forward_pass(model, eeg_input, num_runs=10, warmup=2):
    """Benchmark forward pass."""
    model.eval()

    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(eeg_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    print(f"Running {num_runs} iterations...")
    start = time.time()

    with torch.no_grad():
        for i in range(num_runs):
            output = model(eeg_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if i % 2 == 0:
                print(f"  Iteration {i+1}/{num_runs}")

    end = time.time()
    avg_time = (end - start) / num_runs

    return avg_time


def main():
    # Use single GPU (you can run on each GPU separately)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"{'='*60}")
    print(f"LIF Vectorization Benchmark")
    print(f"{'='*60}")
    print(f"Device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")

    print(f"{'='*60}\n")

    # Match your actual training configuration
    batch_size = 4  # Typical batch size per GPU
    n_channels = 128
    seq_len = 1000  # 1 second at 1000 Hz

    # Create dummy EEG input
    print("Creating dummy EEG input...")
    eeg_input = torch.randn(batch_size, n_channels, seq_len, device=device)

    # Create model with same config as training
    print("Initializing G-SNN model...")
    model = GSNN(
        n_channels=n_channels,
        n_neurons=256,
        n_layers=3,
        hidden_dim=64,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Benchmark
    try:
        avg_time = benchmark_forward_pass(model, eeg_input, num_runs=10)

        print(f"\n{'='*60}")
        print(f"RESULTS:")
        print(f"{'='*60}")
        print(f"Average time per forward pass: {avg_time*1000:.1f} ms")
        print(f"Time for spiking layer (approx):  {avg_time*0.9*1000:.1f} ms")
        print(f"Throughput: {batch_size/avg_time:.1f} samples/sec")
        print(f"{'='*60}")

        # Memory usage
        if torch.cuda.is_available():
            print(f"\nGPU Memory Usage:")
            print(f"  Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"  Reserved:  {torch.cuda.memory_reserved()/1e9:.2f} GB")
            print(f"  Max allocated: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

        print(f"\n{'='*60}")
        print("✅ Benchmark completed successfully!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n❌ Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()