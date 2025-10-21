### NGIS runtime behavior with torchrun (4 processes) and ds003766 config

Command:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 main.py --config configs/ds003766.yaml --epochs 1
```

### High-level summary
- **What actually happens:** The script launches 4 independent processes, but the current code does not initialize Distributed Data Parallel (DDP). Each process runs a full, standalone training job.
- **Critical implications:**
  - Each process treats `world_size=1` and `rank=0` (defaults), so no distributed sampler, no gradient sync, no device assignment per local rank.
  - All processes attempt to use the same default CUDA device (GPU 0), causing contention and unstable performance.
  - All processes think they are the "main" rank and will save checkpoints to the same paths, risking race conditions and file clobbering.

### Process lifecycle (per spawned process)
1) Argument parsing and config load
   - Parses `--config`, `--epochs`, etc. No special handling for `RANK`, `WORLD_SIZE`, `LOCAL_RANK` env vars from `torchrun`.
   - Loads YAML into a `Config` object and optionally overrides training fields with CLI flags.

```131:158:ngis/main.py
def main():
    args = parse_args()
    setup_logging(level=logging.DEBUG if args.debug else logging.INFO)
    logger = logging.getLogger("ngis")
    # Load configuration
    config = load_config(args.config)
    # Override config with command line arguments
    if args.data_path:
        config.data.data_path = args.data_path
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.learning_rate = args.lr
```

2) Distributed setup (not activated)
   - `setup_distributed(args)` only initializes when `args.world_size > 1`.
   - With `torchrun`, the code does not map env vars to `args`, so `world_size` remains `1` and DDP is not initialized.

```118:128:ngis/main.py
def setup_distributed(args):
    if args.world_size > 1:
        torch.distributed.init_process_group(
            backend='nccl',
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank
        )
        return True
    return False
```

3) Seeding, output dir, and system logs
   - Seeds CPU and CUDA, creates `outputs/`, logs PyTorch/CUDA info.

4) Trainer initialization
   - Creates `NGISTrainer(config, is_distributed=False)` because DDP was not enabled.
   - Selects device as `cuda` if available, but does not call `torch.cuda.set_device(local_rank)`; all processes default to GPU 0.
   - Builds model (`GSNN`), loss (`CombinedLoss`), optimizer (`NGISOptimizer`), scheduler (`NGISScheduler`), and checkpointing (`CheckpointManager`).
   - Since `is_distributed=False`, the model is NOT wrapped in DDP.

```78:104:ngis/training/trainer.py
def _init_model(self):
    model_config = self.config.model
    self.model = GSNN(...).to(self.device)
    if self.is_distributed:
        self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank, find_unused_parameters=True)
    logger.info(
        f"Initialized model with {self.model.count_parameters()['trainable_parameters']} parameters"
    )
```

5) Dataloaders (no distributed sampling)
   - Creates `EEGDataset` from `data/raw/sub-01_task-resting_eeg.set` with segmentation: 128 channels, 1s windows (`segment_length=1000`), no overlap.
   - Uses regular `DataLoader` without `DistributedSampler` (since `distributed=False`). Each process will load/iterate the entire dataset independently.
   - Validation duration is set to ~10% (95s) if `max_duration` is not provided.

```130:178:ngis/data/dataloader.py
def create_training_dataloader(..., distributed: bool = False, rank: int = 0, world_size: int = 1, ...):
    dataset = EEGDataset(...)
    return create_dataloader(dataset=dataset, shuffle=True, num_workers=num_workers, distributed=distributed, rank=rank, world_size=world_size)
```

6) Training loop (single-process semantics)
   - Epochs: overridden to `--epochs 1`.
   - For each batch:
     - Move `eeg` to device.
     - Forward pass through `GSNN`:
       - If no `graph_data` provided, constructs a functional connectivity graph from the input.
       - Projects node features, applies `n_layers` of `GATConv`, outputs node embeddings.
       - Maps channel embeddings to `n_neurons` (256) via a linear layer, then interpolates to time length (1000) to form neuron currents.
       - Runs a time-step loop over `seq_len` to simulate LIF spiking and synaptic filtering; produces spike trains and membrane potentials.
       - Readout converts spikes back to reconstructed EEG of shape `(batch, channels, seq_len)`.
     - Loss (`CombinedLoss`): MSE EEG reconstruction + spiking rate regularization + optional biological + L2 regularization.
     - Backward: zero_grad, backward, gradient clipping (`max_norm=1.0`), optimizer step.
     - Logging: tqdm progress bar and optional per-step logging.
   - Validation each epoch on a reduced subset.
   - Scheduler `ReduceLROnPlateau` steps with validation loss.
   - Checkpoint saved each epoch.

```197:235:ngis/training/trainer.py
for batch_idx, batch in enumerate(self.train_dataloader):
    batch = self._move_batch_to_device(batch)
    outputs = self.model(eeg_input=batch["eeg"], return_spikes=True, return_graph=True)
    loss = self.loss_function(
        real_eeg=batch["eeg"],
        simulated_eeg=outputs["eeg_output"],
        spike_trains=outputs["spike_trains"],
        graph_data=outputs["graph_data"],
    )
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.optimizer.step()
```

7) Checkpointing behavior under torchrun
   - The save is guarded by `if self.rank == 0`, but `self.rank` is the constructor arg (default `0` because DDP not used). Therefore, all 4 processes satisfy the guard and attempt to save.
   - All processes write to the same `checkpoints/` directory (`best_model.pth` and `checkpoint_epoch_001.pth`), risking corruption or last-writer-wins.

```343:369:ngis/training/trainer.py
def _save_checkpoint(self, val_loss: float):
    checkpoint = {...}
    is_best = val_loss < self.best_loss
    self.checkpoint_manager.save_checkpoint(checkpoint=checkpoint, metric=val_loss, epoch=self.current_epoch, is_best=is_best)
    if val_loss < self.best_loss:
        self.best_loss = val_loss
```

### Data pipeline details (ds003766)
- Config specifies:
  - `data_path: data/raw/sub-01_task-resting_eeg.set`
  - `batch_size: 2`, `num_workers: 2`, `segment_length: 1000`, `channels: 128` (from 129 original).
  - `augment: false`, `validation_split: 0.1`.
- `EEGDataset` flow:
  - Loads raw MNE data, optional preprocessing via `EEGPreprocessor`.
  - Segments into windows of 1s (1000 samples @ 1000 Hz), no overlap.
  - `__getitem__` returns `{ 'eeg': FloatTensor [n_channels, segment_length], 'info': dict }`.
  - Collate stacks into batch shape `(batch, n_channels, segment_length)`.

```85:101:ngis/data/dataset.py
def _load_data(self, data_path: Union[str, List[str]]):
    if isinstance(data_path, str):
        if path.endswith(('.edf', '.bdf', '.fif', '.set', '.cnt')):
            raw = self.loader.load_file(path)
            self.raw_data = {'single_file': raw}
        else:
            self.raw_data = self.loader.load_directory(path)
    ...
```

### Model compute hotspots
- Graph construction per batch from EEG.
- GAT graph layers over channel graph (`n_layers=3`, `hidden_dim=64`, 4 heads, dropout).
- Time-step spiking simulation loop over `seq_len=1000` and `n_neurons=256` per sample is typically the dominant cost.
- Readout back to EEG.

```212:251:ngis/models/gsnn.py
def _process_graph_layers(...):
    x = self.input_projection(x)
    for i, conv_layer in enumerate(self.graph_layers):
        x = conv_layer(x, edge_index)
        if i < len(self.graph_layers) - 1:
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
    ...
    neuron_embeddings = self.node_to_neuron(node_embeddings.permute(0, 2, 1)).permute(0, 2, 1)
    neuron_currents = torch.nn.functional.interpolate(neuron_embeddings, size=seq_len, mode="linear", align_corners=False)
```

```253:280:ngis/models/gsnn.py
def _simulate_spiking(self, neuron_currents: torch.Tensor):
    for b in range(batch_size):
        self.lif_neurons.reset_state(...)
        if hasattr(self, "synapses"):
            self.synapses.reset_state(...)
        for t in range(seq_len):
            current_input = neuron_currents[b:b + 1, :, t]
            spikes, membrane = self.lif_neurons(current_input)
            if hasattr(self, "synapses"):
                filtered_spikes, _ = self.synapses(spikes)
            else:
                filtered_spikes = spikes
            spike_trains[b, :, t] = filtered_spikes.squeeze(0)
            membrane_potentials[b, :, t] = membrane.squeeze(0)
    return spike_trains, membrane_potentials
```

### What you will see in logs
- Each process logs like a standalone run, e.g.:
  - "Created dataloaders: X train batches, Y val batches"
  - "Starting training..."
  - tqdm progress bar per process
  - Epoch summary with loss and LR
  - Checkpoint save messages (possibly interleaved from multiple processes)

### Consequences of current torchrun usage
- **No multi-GPU speedup**: Gradients are not synchronized; you are running 4 duplicate trainings, not data-parallel training.
- **Device contention**: All processes default to GPU 0; performance degradation or CUDA OOM is likely.
- **Checkpoint conflicts**: Multiple writers to `checkpoints/` can corrupt or overwrite files.
- **Data duplication**: Every process iterates the full dataset (no `DistributedSampler`).

### How to achieve correct multi-GPU behavior (guidance)
If/when you want true multi-GPU training with the existing structure, the following are required:
- Read `LOCAL_RANK`, `RANK`, and `WORLD_SIZE` from env when launched with `torchrun`, e.g. `int(os.environ["LOCAL_RANK"])`.
- Call `torch.cuda.set_device(local_rank)` and pass `rank`/`world_size` into `NGISTrainer(..., is_distributed=True, rank=rank, world_size=world_size)`.
- Initialize process group unconditionally under `torchrun` (do not rely on CLI flags for world_size).
- Use `DistributedSampler` in dataloaders (`distributed=True`) so each rank gets a unique shard.
- Only let `rank==0` write checkpoints/logs.

This document focuses on current behavior; the above is provided to clarify why the present `torchrun` invocation does not engage DDP.


