# NGIS Installation Guide

Complete installation instructions for local development and cluster deployment.

---

## Prerequisites

- Python 3.10 or 3.11
- CUDA 11.8+ (for GPU support)
- 8GB+ RAM
- 10GB+ free disk space

---

## Installation Methods

### Method 1: Local Development (Laptop/Workstation)

#### Step 1: Create Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

#### Step 2: Install PyTorch with CUDA

Check your CUDA version first:
```bash
nvidia-smi  # Look at "CUDA Version" in top-right
```

**For CUDA 12.1:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CUDA 11.8:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**For CPU-only (not recommended):**
```bash
pip install torch torchvision torchaudio
```

#### Step 3: Install Core Dependencies

```bash
pip install -r requirements.txt
```

#### Step 4: Install PyTorch Geometric CUDA Extensions

**Automatic (recommended):**
```bash
bash install_pyg_extensions.sh
```

**Manual:**
```bash
# Replace cu121 with your CUDA version (cu118, cu121, etc.)
pip install torch-scatter torch-cluster torch-spline-conv torch-sparse \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

#### Step 5: Verify Installation

```bash
python -c "
import torch
import torch_geometric
import torch_scatter
import torch_sparse

print('✓ PyTorch:', torch.__version__)
print('✓ CUDA available:', torch.cuda.is_available())
print('✓ PyG:', torch_geometric.__version__)
print('✓ Extensions: OK')
"
```

---

### Method 2: HPC Cluster (SLURM)

#### Step 1: Load Modules

```bash
# Adjust module names for your cluster
module load cuda/12.1
module load python/3.10
module load gcc/11.2.0
```

#### Step 2: Create Virtual Environment

```bash
cd $HOME/ngis
python -m venv venv
source venv/bin/activate
```

#### Step 3: Install Dependencies

```bash
# Install PyTorch for your cluster's CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install -r requirements.txt

# Install PyG extensions
bash install_pyg_extensions.sh
```

#### Step 4: Test Installation

```bash
# Quick test
python test_forward_pass.py

# Full dry run
python main.py --config configs/ds003766.yaml --dry_run
```

---

### Method 3: Docker Container

```bash
# Build container
docker build -t ngis:latest .

# Run with GPU
docker run --gpus all -v $(pwd)/data:/workspace/data ngis:latest \
    python main.py --config configs/default.yaml
```

---

## Cluster-Specific Instructions

### OpenAAC Cluster (4× A100)

```bash
# SSH to cluster
ssh user@cluster.openaacc.org

# Clone repository
git clone https://github.com/your-org/ngis.git
cd ngis

# Setup
module load cuda/12.1 python/3.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash install_pyg_extensions.sh

# Submit job
sbatch slurm_train.sh
```

### Other Clusters

Check with your cluster documentation for:
- Available CUDA versions
- Python versions
- Job submission system (SLURM/PBS/LSF)
- GPU node specifications

Adjust module loads and CUDA versions accordingly.

---

## Troubleshooting

### Issue: "torch-scatter not found"

**Cause:** CUDA extension version mismatch

**Fix:**
```bash
# Uninstall old versions
pip uninstall torch-scatter torch-sparse torch-cluster torch-spline-conv

# Reinstall with correct version
bash install_pyg_extensions.sh
```

### Issue: "CUDA out of memory"

**Fix:**
- Reduce batch size in config: `batch_size: 16` → `batch_size: 8`
- Enable gradient checkpointing (add to config)
- Use mixed precision training

### Issue: "numpy version conflict"

**Fix:**
```bash
pip install numpy==1.24.3  # Compatible version
```

### Issue: Import errors on cluster

**Fix:**
```bash
# Ensure modules are loaded
module list

# Check Python path
which python

# Verify CUDA
nvidia-smi
```

---

## Version Compatibility

### Tested Configurations

| PyTorch | CUDA | Python | torch-geometric | Status |
|---------|------|--------|-----------------|--------|
| 2.5.1   | 12.1 | 3.10   | 2.3.0          | ✓ Tested |
| 2.5.1   | 12.1 | 3.11   | 2.3.0          | ✓ Tested |
| 2.4.0   | 11.8 | 3.10   | 2.3.0          | Should work |
| 2.3.0   | 11.8 | 3.9    | 2.3.0          | Should work |

### Finding Compatible Versions

Check available wheels at: https://data.pyg.org/whl/

Format: `torch-{TORCH_VERSION}+{CUDA_TAG}.html`

Examples:
- `torch-2.5.1+cu121.html` - PyTorch 2.5.1 + CUDA 12.1
- `torch-2.4.0+cu118.html` - PyTorch 2.4.0 + CUDA 11.8

---

## Next Steps

After installation:

1. **Download data:** See [data/README.md](data/README.md)
2. **Test setup:** `python test_forward_pass.py`
3. **Configure:** Edit `configs/ds003766.yaml`
4. **Train:** `python main.py --config configs/ds003766.yaml`

---

## Support

For issues:
1. Check [ISSUES.md](ISSUES.md) for known problems
2. Run diagnostics: `python -m torch.utils.collect_env`
3. Open GitHub issue with full traceback

---

## Quick Reference

```bash
# Install everything (Linux/Mac)
python -m venv venv && source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
bash install_pyg_extensions.sh

# Verify
python test_forward_pass.py

# Train
python main.py --config configs/ds003766.yaml
```
