# NGIS: NeuroGraph Inverse Solver

## Team Snapshot
- Team *Biologically Artificial*: three undergraduate builders plus a university lecturer mentor spanning Romanian-American University, University of Staffordshire, and Babeș-Bolyai University.
- Skills matrix covers UI/UX, neuroscience, ML/AI research, Python/C++, CUDA, and distributed systems.
- Backed by CRAI compute resources and eager to exploit the hackathon GPU cluster documentation and mentorship.

## Hackathon Pitch
- Reconstruct subject-specific functional brain networks from raw 128-channel EEG via graph-structured spiking neural networks (G-SNN).
- Deliver interpretable, biologically grounded connectivity maps that clinicians, BCI teams, and researchers can interrogate.
- Package a reproducible, containerised pipeline ready to scale from laptops to multi-GPU A100 nodes.

## Problem
- High-density EEG is affordable and portable, yet translating recordings into subject-specific connectivity remains labour-intensive and noisy.
- Existing EEG deep-learning work focuses on black-box classification and rarely exposes generative, biologically meaningful structure.
- Researchers lack an open, physics-informed toolkit that can iterate rapidly on personalised brain network hypotheses.

## Solution Overview
- Functional graph constructor converts EEG segments into per-subject connectivity graphs.
- G-SNN core fuses PyTorch Geometric message-passing (GraphSAGE/GAT layers) with Brian2 LIF and Izhikevich neuron dynamics.
- Diffusion-style generative prior regularises latent connectivity to remain plausible.
- Custom CUDA kernels, mixed precision, and PyTorch DDP/NCCL keep training efficient on multi-GPU hardware.
- Rich OmegaConf-driven configs, logging, and checkpointing ensure experiments are reproducible end to end.

## Architecture, Frameworks, and Tooling
- **Data pipeline**: MNE-Python ingestion across .edf/.bdf/.fif/.set/.cnt formats, configurable preprocessing, augmentation, and segmenting.
- **Graph + spiking stack**: Connectivity graph constructor, message-passing GNN, plastic synapses, and Brian2-based spike simulation.
- **Training utilities**: AdamW optimiser with cyclical LR, gradient clipping at 1.0, mixed precision (FP16/BF16), checkpoint rotation, TensorBoard/W&B hooks.
- **Codebase**: Python-first with performance-sensitive components in C++/CUDA; targeting Apache 2.0 licensing with public repo release by 5 Aug 2025.

## Target Hardware and Performance Plan
- Baseline runs on developer RTX 3080 desktops and CRAI resources.
- Hackathon target: 4 × NVIDIA A100 (80 GB) nodes interconnected via NVLink; x86_64 CPUs handle preprocessing.
- Immediate tasks: profile CUDA-based LIF updates and graph ops, port hotspots for multi-GPU execution, establish performance baselines.

## Data Sources
- Temple University Hospital EEG Corpus (~1.5 TB, CC BY-NC).
- BCI Competition IV-2a (128-channel motor imagery dataset, CC BY-NC).
- ~30 hours of new 128-channel EEG to be collected with a Brain Products actiChamp system.

## Differentiators
- Marries interpretable graph structure with biologically realistic spiking dynamics and generative priors.
- Produces subject-specific functional networks instead of opaque feature vectors.
- Designed for distributed GPU scaling from the outset with containerised deployment.
- Anchored to open datasets for transparent benchmarking and shared reproducibility.

## Current Progress
- Config loading, logging, scheduler, and optimizer stack stabilised; experiments now start cleanly.
- EEG loader widened to cover EEGLAB and other 128-channel formats with consistent metadata handling.
- `SimpleEEGModel` baseline validates the end-to-end training loop, logging, and checkpointing on synthetic and sample data.
- `ISSUES.md` tracks architectural blockers (batch handling, node/neuron alignment, PyG batching) with clear remediation plan.

## Hackathon Goals
- Containerise the full workflow with Docker/Singularity for shareable, reproducible experiments.
- Draft a short paper/poster summarising approach, preliminary metrics, and roadmap for neuroinformatics venues.
- Land architectural fixes that reintroduce the full G-SNN while preserving biological loss terms.
- Stand up visual analytics (TensorBoard/W&B, graph diagnostics) for reconstructed connectivity.

## Potential Applications
- Brain tumour detection/localisation via simulated lesions and EEG comparisons.
- Patient-specific seizure focus mapping to guide surgical interventions.
- Rapid BCI calibration by generating labelled synthetic EEG in minutes.
- Closed-loop neurostimulation prototyping for TMS/tDCS/DBS regimes.
- Drug discovery and neurotoxicity screening using digital twins before animal studies.

## Quick Demo (<= 5 minutes)
1. Bootstrap an environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Run the synthetic training smoke test:
   ```bash
   python test_quick_training.py
   ```
   - Shows device discovery, training/validation losses, and stores `outputs/quick_model.pt`.
   - Demonstrates the training loop, logging, and checkpointing without real EEG files.
3. Highlight the config-driven launcher without full training:
   ```bash
   python main.py --config configs/default.yaml --dry_run
   ```
   - Confirms configuration parsing, trainer initialisation, and logging setup.
4. Close with a terminal capture or screenshot of the training logs to anchor the narrative and segue into the roadmap.

## Team Ask
- Mentorship or pair-programming support for the PyG batching + spiking integration redesign.
- Early access to cluster documentation (node specs, interconnect, scheduler) to plan distributed experiments.
- Feedback on onboarding docs and demo narrative so new collaborators can ramp quickly.


## Visual Diagnostics
- `gsnn_eeg_comparison.png`: overlays a few EEG channels from the validation batch (real in blue, simulated in orange) so we can judge waveform fidelity and phase alignment at a glance.
- `gsnn_spike_raster.png`: shows which neurons fired over time during that batch; densely packed dots mean high firing rates, sparse bands highlight quiescent populations.
- `gsnn_connectivity.png`: heatmap of the current functional connectivity matrix (post-threshold), revealing which neuron pairs the graph constructor treats as strongly coupled.
