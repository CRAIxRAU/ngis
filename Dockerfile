# NGIS Dockerfile
# Multi-stage build for optimized container

# Base image with CUDA support
FROM nvidia/cuda:11.8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
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
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric with CUDA support
RUN pip3 install --no-cache-dir torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Install Brian2
RUN pip3 install --no-cache-dir brian2

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data outputs checkpoints logs

# Set permissions
RUN chmod +x main.py

# Expose port for distributed training
EXPOSE 10001

# Set default command
CMD ["python3", "main.py", "--config", "configs/default.yaml"] 