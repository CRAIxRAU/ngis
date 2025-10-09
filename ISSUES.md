# NGIS Training Pipeline Issues

This document tracks critical issues found during pipeline testing that prevent training from running.

## Summary

The NGIS codebase has **never been successfully run end-to-end**. Multiple fundamental shape mismatches and architectural inconsistencies suggest the code was written without integration testing. The issues below must be fixed before training can proceed.

---

## Fixed Issues ✅

### 1. Config Loading (FIXED)
**File:** `main.py:112-116`
**Issue:** Used broken `Config(config_dict)` instead of `Config.from_yaml()`
**Fix:** Changed to use `Config.from_yaml()` which properly creates OmegaConf structure
**Status:** ✅ Fixed

### 2. Logging Type Mismatch (FIXED)
**File:** `utils/logging.py:44-49`
**Issue:** `setup_logging()` expected string but received `logging.DEBUG` int constant
**Fix:** Added isinstance check to handle both int and string log levels
**Status:** ✅ Fixed

### 3. Scheduler Duplicate Arguments (FIXED)
**File:** `training/scheduler.py:32-43`
**Issue:** Passed `mode='min'` both as hardcoded arg and in `**kwargs`, causing duplicate keyword argument error
**Fix:** Merged defaults with kwargs using dict.update() pattern
**Status:** ✅ Fixed

### 4. Optimizer Wrapper Compatibility (FIXED)
**File:** `training/trainer.py:121-129`
**Issue:** Scheduler expects `torch.optim.Optimizer` but received `NGISOptimizer` wrapper
**Fix:** Pass inner optimizer (`self.optimizer.optimizer`) to scheduler
**Added:** `param_groups` property to NGISOptimizer for better compatibility
**Status:** ✅ Fixed (pragmatic solution, could be improved)

### 5. EEGLAB Format Support (FIXED)
**File:** `data/eeg_loader.py`
**Issue:** EEGLAB format not supported, `n_channels` property missing on RawEEGLAB objects
**Fix:**
- Added `read_raw_eeglab` import
- Added `.set` file handling
- Changed `raw.n_channels` to `len(raw.ch_names)` for compatibility
**Status:** ✅ Fixed

---

## Critical Unfixed Issues ❌

# Check shapes in the codebase, place them in a config so we can update it!



### 6. Graph Constructor Node Features Shape Mismatch (PARTIALLY FIXED)
**File:** `models/graph_constructor.py:245-259`
**Original Issue:** Tried to project `(128, 1000)` tensor but transpose logic was wrong
**Line:** 256
**Original Code:**
```python
projection = nn.Linear(seq_len, 64).to(eeg_data.device)
node_features = projection(node_features.t()).t()  # WRONG
```
**Attempted Fix:**
```python
projection = nn.Linear(seq_len, 64).to(eeg_data.device)
node_features = projection(node_features)  # (n_channels, 64)
```
**Status:** ⚠️ Partially fixed, but led to downstream issues

Add in a config file and update the graph constructor to use it:

```python
        # Learnable node features
        self.node_features = nn.Parameter(
            torch.randn(self.n_neurons, 64) * 0.1
        )
```


### 7. Input Projection Dimension Mismatch (PARTIALLY FIXED)
**File:** `models/gsnn.py:123-124`
**Issue:** `input_projection` expects `n_channels` (128) but graph constructor outputs 64-dim features
**Original Code:**
```python
self.input_projection = nn.Linear(self.n_channels, self.hidden_dim)  # 128 -> 64
```
**Attempted Fix:**
```python
self.input_projection = nn.Linear(64, self.hidden_dim)  # 64 -> 64
```
**Status:** ⚠️ Fixed this specific error, but exposed deeper architectural issues

### 8. Fundamental Architecture Mismatch: Nodes vs Neurons vs Batch (CRITICAL) ❌
**Files:** `models/gsnn.py:225-234`, entire G-SNN architecture
**Issue:** The code conflates graph nodes, spiking neurons, and batch dimensions in incompatible ways

**Error Message:**
```
RuntimeError: shape '[0, 256, -1]' is invalid for input of size 32768
```

**Root Cause Analysis:**

The `_simulate_spiking` method (line 231) tries to compute batch size:
```python
batch_size = x.shape[0] // self.n_neurons  # 128 // 256 = 0 ❌
```

**The Flow:**
1. **Input:** EEG batch `(batch=2, channels=128, time=1000)`
2. **Graph Constructor:** Creates graph with `n_nodes=128` (one per EEG channel)
   - Output: `node_features` of shape `(128, 64)`
3. **Graph Layers:** Process nodes through GCN
   - Output: `x` of shape `(128, hidden_dim)` → `(128, 256)` after output_projection
4. **Spiking Simulation:** Expects `x` to be `(batch * n_neurons, features)`
   - Tries: `batch_size = 128 // 256 = 0` ❌
   - Fails: Cannot reshape to `(0, 256, -1)`

**Fundamental Design Confusion:**
- Graph has 128 nodes (one per EEG channel)
- Model has 256 neurons (spiking units)
- **These should be different!** Neurons process graph node features, not replace them
- Current code assumes nodes ARE neurons (expects `n_nodes == n_neurons`)
- Batch dimension is lost during graph construction

**Status:** ❌ **CRITICAL - Requires architectural redesign**

---

## Architectural Problems

### Problem 1: Batch Processing in Graph Constructor
**File:** `models/graph_constructor.py:245-250`

```python
def _create_node_features(self, eeg_data: torch.Tensor) -> torch.Tensor:
    batch_size, n_channels, seq_len = eeg_data.shape
    node_features = eeg_data.mean(dim=0)  # (n_channels, seq_len)
```

**Issue:** Takes mean over batch dimension, **losing batch information entirely**. Graph construction should preserve batch or create separate graphs per batch sample.

**Impact:** Cannot process batches properly through the model.

### Problem 2: Node/Neuron Conflation
The architecture conflates:
- **Graph nodes** (representations of EEG channels/brain regions)
- **Spiking neurons** (computational units that fire spikes)

**Expected Architecture:**
```
EEG Input (batch, channels, time)
    ↓
Graph Construction (per batch sample)
    ↓
Graph Nodes (batch, n_nodes, node_features)
    ↓
Map to Spiking Neurons (batch, n_neurons, neuron_features)
    ↓
LIF Dynamics over time
    ↓
Readout to EEG (batch, channels, time)
```

**Current Architecture:**
```
EEG Input (batch, channels, time)
    ↓
Graph Construction (LOSES BATCH!)
    ↓
Graph Nodes (n_nodes, features) ❌ No batch dimension
    ↓
??? Tries to treat nodes as neurons ???
    ↓
Fails
```

### Problem 3: Missing Batch Handling in PyTorch Geometric
**File:** `models/graph_constructor.py:_construct_functional_graph`

PyTorch Geometric supports batching via `torch_geometric.data.Batch`, but the code doesn't use it. Each batch sample should create its own graph, then batch them together.

---

## Required Fixes

### Priority 1: Fix Batch Processing in Graph Construction

**File:** `models/graph_constructor.py`

**Required Changes:**
1. Create separate graph for each batch sample
2. Use `torch_geometric.data.Batch` to combine graphs
3. Preserve batch dimension through graph processing

**Example Fix:**
```python
def forward(self, eeg_data: torch.Tensor) -> Batch:
    """
    Args:
        eeg_data: (batch_size, n_channels, seq_len)
    Returns:
        Batched graphs
    """
    batch_size = eeg_data.shape[0]
    graphs = []

    for i in range(batch_size):
        # Create graph for each sample
        graph = self._construct_single_graph(eeg_data[i])
        graphs.append(graph)

    # Batch graphs together
    return Batch.from_data_list(graphs)
```

### Priority 2: Clarify Node-to-Neuron Mapping

**File:** `models/gsnn.py`

**Decision Needed:**
- **Option A:** Nodes ARE neurons (`n_nodes == n_neurons`)
  - Simplest, but limits flexibility
  - Graph has 256 nodes, one per neuron

- **Option B:** Separate nodes and neurons
  - More complex, more flexible
  - Graph has 128 nodes (brain regions/channels)
  - Each node maps to multiple neurons (e.g., 128 nodes × 2 neurons/node = 256 neurons)

**Requires:** Architectural decision and complete refactor of `_simulate_spiking`

### Priority 3: Fix Spiking Simulation Shape Handling

**File:** `models/gsnn.py:225-260`

After batch processing is fixed, update `_simulate_spiking` to properly handle:
```python
def _simulate_spiking(self, x: torch.Tensor, seq_len: int):
    """
    Args:
        x: Node features from graph layers
           Shape depends on architecture choice:
           - Option A: (batch * n_neurons, features)
           - Option B: (batch * n_nodes, features) → needs mapping to neurons
    """
    # Need to properly extract batch_size based on chosen architecture
    # Current code assumes batch_size = x.shape[0] // n_neurons (WRONG)
```

---

## Testing Gaps

The following were never tested:
1. ❌ End-to-end forward pass with real data
2. ❌ Batch processing through the model
3. ❌ Shape compatibility between components
4. ❌ Loss function with model outputs
5. ❌ Backward pass and gradient flow

**Recommendation:** Add integration tests that verify:
- Model can process a single sample
- Model can process a batch
- Loss can be computed
- Gradients flow correctly
- Full training loop runs for 1 iteration

---

## Environment Notes

### Current Setup
- **Mac M-series**: MPS available but not used (trainer only checks CUDA)
- **Training device**: Falls back to CPU (slow)
- **Windows RTX 4090**: Will need device selection fix and possibly backend fix (NCCL vs gloo)

### Device Selection Fix Needed
**File:** `training/trainer.py:58`

**Current:**
```python
self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**Should be:**
```python
if torch.cuda.is_available():
    self.device = torch.device('cuda')
elif torch.backends.mps.is_available():
    self.device = torch.device('mps')
else:
    self.device = torch.device('cpu')
```

---

## Data Pipeline Status

✅ **Working:**
- EEG file loading (.set format)
- Preprocessing (filtering, normalization)
- Segmentation into 1-second chunks
- DataLoader creation
- Batch collation

The data pipeline is fully functional. All issues are in the model architecture.

---

## Recommendations

### Immediate Actions
1. **Do NOT attempt to train** until architectural issues are resolved
2. **Create unit tests** for each model component with known input/output shapes
3. **Document expected tensor shapes** at each step in the forward pass
4. **Decide on architecture:** Nodes == Neurons or separate?

### Refactoring Strategy
1. Fix graph constructor to handle batches properly
2. Add shape assertions throughout forward pass for debugging
3. Create simple end-to-end test with single sample
4. Verify shapes at every step
5. Only then attempt batch processing

### Alternative Approach
If timeline is critical, consider:
1. **Simplify architecture** temporarily (remove graph components, direct EEG → neurons)
2. **Verify training loop** works with simplified model
3. **Add back complexity** incrementally with tests at each step

---

## Files Modified

### Successfully Fixed
- `main.py` - Config loading
- `utils/logging.py` - Log level handling
- `training/scheduler.py` - Duplicate kwargs
- `training/trainer.py` - Optimizer wrapper
- `training/optimizer.py` - Added param_groups property
- `data/eeg_loader.py` - EEGLAB support
- `CHANGELOG.md` - Documented all changes

### Attempted Fixes (Incomplete)
- `models/graph_constructor.py` - Node features projection
- `models/gsnn.py` - Input projection dimensions

### Needs Refactoring
- `models/graph_constructor.py` - Batch handling
- `models/gsnn.py` - Entire spiking simulation
- `models/lif_neuron.py` - May need updates depending on architecture decision
- `models/readout.py` - Likely needs shape fixes

---

## Next Steps

1. **Architectural Decision Meeting**
   - Decide: Nodes == Neurons or separate?
   - Define exact tensor shapes at each step
   - Document architecture clearly

2. **Create Shape Specification Document**
   - Input: `(batch, 128, 1000)`
   - Graph: `(batch, n_nodes, node_dim)` → exact dimensions
   - Neurons: `(batch, n_neurons, time)` → exact dimensions
   - Output: `(batch, 128, 1000)` → matches input

3. **Implement Unit Tests First**
   - Test each component in isolation
   - Mock inputs with known shapes
   - Verify outputs match expectations

4. **Refactor Components**
   - Start with graph constructor
   - Then GSNN main model
   - Then spiking simulation
   - Test after each change

5. **Integration Testing**
   - Single sample forward pass
   - Batch forward pass
   - Loss computation
   - Single training iteration
   - Full epoch

---

## Conclusion

The NGIS codebase requires significant architectural fixes before training can proceed. The good news: **the data pipeline works perfectly**. The issues are entirely in the model architecture, specifically around batch handling and the node/neuron relationship.

**Estimated effort to fix:** 4-8 hours for someone familiar with PyTorch Geometric and spiking neural networks.

**Current status:** Pipeline verified up to model forward pass. Cannot proceed without architecture refactor.
