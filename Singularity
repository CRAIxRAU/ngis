Bootstrap: docker
From: nvidia/cuda:11.8-devel-ubuntu20.04

%labels
    Author NGIS Team
    Version v1.0.0
    Description NGIS: Neural Graph Inverse Simulator

%environment
    export DEBIAN_FRONTEND=noninteractive
    export PYTHONUNBUFFERED=1
    export PYTHONPATH=/app
    export CUDA_VISIBLE_DEVICES=0
    export OMP_NUM_THREADS=4

%post
    # Update package lists
    apt-get update
    
    # Install system dependencies
    apt-get install -y \
        python3.9 \
        python3.9-dev \
        python3-pip \
        git \
        wget \
        curl \
        build-essential \
        cmake \
        pkg-config \
        libblas-dev \
        liblapack-dev \
        libatlas-base-dev \
        gfortran \
        libhdf5-dev \
        libhdf5-serial-dev \
        libhdf5-103 \
        libqtgui4 \
        libqtwebkit4 \
        libqt4-test \
        python3-pyqt5 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libgtk-3-0 \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        gfortran \
        libopenblas-dev \
        liblapack-dev \
        libhdf5-serial-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # Create app directory
    mkdir -p /app
    cd /app
    
    # Copy requirements and install Python dependencies
    # Note: In actual usage, requirements.txt should be copied from host
    pip3 install --no-cache-dir --upgrade pip setuptools wheel
    
    # Install PyTorch with CUDA support
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install PyTorch Geometric with CUDA support
    pip3 install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
    
    # Install other dependencies
    pip3 install --no-cache-dir \
        numpy>=1.21.0 \
        scipy>=1.9.0 \
        scikit-learn>=1.1.0 \
        brian2>=2.5.0 \
        pandas>=1.5.0 \
        h5py>=3.8.0 \
        mne>=1.4.0 \
        matplotlib>=3.6.0 \
        seaborn>=0.12.0 \
        plotly>=5.14.0 \
        pyyaml>=6.0 \
        hydra-core>=1.3.0 \
        wandb>=0.15.0 \
        tensorboard>=2.12.0 \
        tqdm>=4.64.0 \
        click>=8.1.0 \
        rich>=13.0.0 \
        colorama>=0.4.6 \
        pytest>=7.3.0 \
        pytest-cov>=4.0.0 \
        black>=23.0.0 \
        isort>=5.12.0 \
        flake8>=6.0.0 \
        mypy>=1.3.0
    
    # Create necessary directories
    mkdir -p /app/data /app/outputs /app/checkpoints /app/logs
    
    # Clean up
    apt-get clean
    rm -rf /var/lib/apt/lists/*

%files
    # Copy application files (will be copied from host during build)
    # main.py /app/
    # data/ /app/data/
    # models/ /app/models/
    # training/ /app/training/
    # simulation/ /app/simulation/
    # utils/ /app/utils/
    # configs/ /app/configs/
    # requirements.txt /app/

%runscript
    # Default command when container is run
    python3 /app/main.py --config /app/configs/default.yaml "$@"

%startscript
    # Command when container is started
    python3 /app/main.py --config /app/configs/default.yaml "$@"

%test
    # Test script to verify installation
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}')"
    python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python3 -c "import brian2; print(f'Brian2 version: {brian2.__version__}')"
    python3 -c "import torch_geometric; print('PyTorch Geometric imported successfully')"
    python3 -c "import mne; print(f'MNE version: {mne.__version__}')" 