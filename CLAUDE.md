# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NGIS (Neural Graph Inverse Simulator) is a scientific ML framework for reconstructing subject-specific functional brain networks from raw 128-channel EEG time series using graph-structured spiking neural networks (G-SNN). The system combines PyTorch, PyTorch Geometric, Brian2, and custom CUDA kernels for efficient training and simulation.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Using Docker
docker build -t ngis .
docker run --gpus all -v $(pwd)/data:/app/data ngis

# Using Singularity
singularity build ngis.sif Singularity
singularity run --nv ngis.sif
```

### Training Commands
```bash
# Basic training
python main.py --config configs/default.yaml

# Multi-GPU distributed training
python main.py --config configs/ddp.yaml --world_size 4

# Custom configuration with overrides
python main.py --config configs/default.yaml --data_path /path/to/eeg/data --epochs 200 --lr 0.0005
```

### Testing and Code Quality
```bash
# Run tests
python -m pytest tests/

# Code formatting and linting
black .
isort .
flake8 .
mypy .
```

## Architecture Overview

### Core Components

**Data Pipeline (`data/`)**
- `eeg_loader.py`: Loads 128-channel EEG data in various formats (.edf, .bdf, .fif, .set, .cnt)
- `preprocessor.py`: EEG preprocessing with configurable filtering and artifact removal
- `dataset.py`: PyTorch Dataset implementation with segment-based processing
- `dataloader.py`: DataLoader with distributed training support

**Graph-Structured Spiking Neural Network (`models/`)**
- `gsnn.py`: Main G-SNN model integrating graph structure with LIF neurons
- `graph_constructor.py`: Constructs functional brain graphs from EEG data
- `lif_neuron.py`: Leaky Integrate-and-Fire neuron implementation with Brian2 integration
- `synapse.py`: Synaptic connectivity and plasticity mechanisms
- `readout.py`: EEG signal reconstruction from spiking activity

**Training System (`training/`)**
- `trainer.py`: Main trainer with distributed training support (DDP + NCCL)
- `loss_functions.py`: Combined loss functions (EEG reconstruction, biological constraints)
- `optimizer.py`: Custom optimizer with gradient clipping and mixed precision
- `scheduler.py`: Learning rate scheduling strategies

**Simulation Engine (`simulation/`)**
- `brian2_simulator.py`: Brian2 integration for biologically accurate spiking dynamics
- `forward_simulator.py`: Forward pass simulation for EEG signal generation
- `simulation_utils.py`: Utilities for spike processing and biological constraint validation

**Utilities (`utils/`)**
- `config.py`: Configuration management with dataclasses and OmegaConf
- `checkpointing.py`: Model checkpointing and resume functionality
- `logging.py`: Structured logging with wandb/tensorboard integration
- `metrics.py`: EEG reconstruction metrics and biological plausibility measures
- `visualization.py`: Plotting utilities for EEG signals and network connectivity

### Configuration System

Configuration uses YAML files with hierarchical structure:
- `configs/default.yaml`: Standard single-GPU training configuration
- `configs/ddp.yaml`: Multi-GPU distributed training setup

Key configuration sections:
- `data`: EEG data loading parameters (sampling rate, channels, preprocessing)
- `model`: G-SNN architecture (neurons, layers, LIF parameters, graph construction)
- `training`: Training hyperparameters (epochs, learning rate, batch size)
- `loss`: Loss function weighting (EEG reconstruction vs biological constraints)
- `optimizer`: Optimization settings (AdamW, gradient clipping, mixed precision)

### Distributed Training Architecture

The system supports multi-GPU training using:
- **PyTorch DDP**: Distributed Data Parallel for model synchronization
- **NCCL**: NVIDIA Collective Communication Library for efficient GPU communication
- **Custom CUDA kernels**: Optimized LIF neuron updates and synaptic transmission

Launch distributed training with:
```bash
python main.py --config configs/ddp.yaml --world_size 4 --rank 0 --dist_url tcp://localhost:10001
```

### Data Flow

1. **EEG Loading**: Raw 128-channel time series → preprocessing → segmentation
2. **Graph Construction**: EEG segments → functional connectivity → graph structure
3. **G-SNN Forward**: Graph + spikes → LIF dynamics → simulated EEG
4. **Loss Computation**: Real vs simulated EEG + biological constraints
5. **Optimization**: Gradient descent on graph weights and neuron parameters

## Development Guidelines

### Code Organization
- Follow existing module structure and import patterns
- Use dataclasses for configuration objects
- Implement proper error handling with module-level loggers
- Add comprehensive docstrings in Google style
- Include type annotations for all function signatures

### PyTorch Patterns
- Inherit from `nn.Module` for all neural components
- Support both CPU and CUDA devices
- Use batch-first dimension convention: `(batch, channels, time)`
- Implement proper `forward()` methods with train/eval differences
- Clean up GPU memory explicitly when needed

### EEG Data Handling
- Default sampling rate: 1000 Hz
- Default channels: 128 (full montage)
- Segment length: 1000 samples (1 second)
- Support overlapping segments for data augmentation
- Preserve metadata throughout processing pipeline

### Biological Constraints
- Maintain biologically plausible LIF parameters (tau_m=20ms, v_threshold=-55mV)
- Enforce realistic spike rates and refractory periods
- Validate graph connectivity patterns against known neuroanatomy
- Use Brian2 units system for time constants and voltages

### Performance Optimization
- Use mixed precision training (FP16/BF16) for memory efficiency
- Implement gradient checkpointing for large models
- Profile GPU memory usage during development
- Use `torch.no_grad()` for inference and validation
- Batch graph operations when possible

## Important File Locations

- Main entry point: `main.py`
- Configuration files: `configs/*.yaml`
- Model definitions: `models/gsnn.py`
- Training logic: `training/trainer.py`
- Data loading: `data/eeg_loader.py`
- Utilities: `utils/config.py`, `utils/logging.py`

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size or enable gradient checkpointing
- **Brian2 simulation errors**: Check LIF parameter ranges and time step consistency
- **Distributed training hangs**: Verify NCCL backend and network connectivity
- **Configuration errors**: Validate YAML syntax and parameter types
- **EEG loading failures**: Check file format support and channel count

### Performance Tuning
- Monitor GPU utilization with `nvidia-smi`
- Profile with PyTorch profiler: `torch.profiler.profile()`
- Use wandb or tensorboard for training visualization
- Check distributed training efficiency with communication profiling

Follow the comprehensive coding standards in `.cursor/rules.md` for detailed guidelines on documentation, error handling, and API design patterns.