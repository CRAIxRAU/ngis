# NGIS: Neural Graph Inverse Simulator

A scientific ML framework for reconstructing subject-specific functional brain networks from raw 128-channel EEG time series using graph-structured spiking neural networks (G-SNN).

## Overview

NGIS reconstructs functional brain networks by training a graph-structured spiking neural network whose simulated EEG output matches real EEG recordings. The framework uses:

- **PyTorch**: Training and optimization
- **PyTorch Geometric**: Graph neural network structure
- **Brian2**: Leaky Integrate-and-Fire (LIF) neuron dynamics
- **CUDA**: Custom spiking kernels for acceleration
- **DDP + NCCL**: Multi-GPU distributed training

## Architecture

```
Input: Raw EEG (128 channels) → G-SNN Training → Output: Functional Brain Network
```

- **Nodes**: Individual neurons with LIF dynamics
- **Edges**: Synaptic weights between neurons (learned)
- **Optimization**: Minimize error between simulated and real EEG under biological constraints

## Project Structure

```
ngis/
├── data/           # EEG loading and preprocessing
├── models/         # G-SNN architecture and graph construction
├── simulation/     # Brian2 integration and forward simulation
├── training/       # Loss functions, optimization, DDP integration
├── utils/          # Logging, visualization, checkpointing
├── configs/        # Configuration files
├── scripts/        # Utility scripts
├── tests/          # Unit tests
├── main.py         # Entry point for training
├── requirements.txt # Python dependencies
├── Dockerfile      # Docker container
├── Singularity     # Singularity container
└── README.md       # This file
```

## Installation

### Using Docker
```bash
docker build -t ngis .
docker run --gpus all -v $(pwd)/data:/app/data ngis
```

### Using Singularity
```bash
singularity build ngis.sif Singularity
singularity run --nv ngis.sif
```

### Local Development
```bash
pip install -r requirements.txt
```

## Usage

### Basic Training
```bash
python main.py --config configs/default.yaml
```

### Multi-GPU Training
```bash
python main.py --config configs/ddp.yaml --world_size 4
```

### Custom Configuration
```bash
python main.py --config configs/custom.yaml --data_path /path/to/eeg/data
```

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
black .
isort .
flake8 .
```

## License

[Add your license information here]

## Citation

[Add citation information when published] 

## Used Datasets

https://openneuro.org/datasets/ds003766/versions/2.0.3