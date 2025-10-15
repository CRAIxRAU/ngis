# NGIS Deployment Guide for Helios Cluster

## Prerequisites

1. ✅ PLGrid account with Helios access
2. ✅ SSH key configured
3. ✅ Computing grant allocated (plghack2025spectro-gpu-gh200)

---

## 🚀 Quick Start (Run Training)

### Step 0: SSH into Cluster

```bash
ssh plgsorinturculet@helios.cyfronet.pl
```

### Step 1: Clone Repository and Switch Branch

```bash
cd ~
git clone <your-repo-url> ngis
cd ngis
git checkout dev/local-prototype
```

### Step 2: Allocate GPU Resources

```bash
srun -p plgrid-gpu-gh200 --ntasks=1 --cpus-per-task 72 --gres gpu:4 --time 08:20:00 -A plghack2025bioart-gpu-gh200 --pty bash -l
```

Your prompt will change to show the compute node: `[helios][plgsorinturculet@x1002c4s2b0n0 ~]$`

### Step 3: Load ML Bundle

```bash
module add ML-bundle/25.04
```

### Step 4: Create and Activate Python Environment

```bash
python3 -m venv --system-site-packages venv/pytorch
. venv/pytorch/bin/activate

and 

pip3 install --no-cache-dir torch==2.8.0+cu128 torchvision==0.23.0
```

### Step 5: Navigate to Project and Download Data (First Time Only)

```bash
cd ngis
python download_data.py
```

---
### Step 6: Install requirements
```bash
pip install -r requirements.txt
```
 
Here we need to figure out how to use venvs, but my concern is that we already use the pytorch venv, can we use 2?

### Step 6: Install requirements
```bash
python main.py --config configs/ds003766.yaml --epochs 1
```
 
Here we need to figure out how to use venvs, but my concern is that we already use the pytorch venv, can we use 2?
### Step 7: Run it
```bash
python main.py --config configs/ds003766.yaml --epochs 1
```