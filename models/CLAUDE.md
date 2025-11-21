# models/CLAUDE.md

Guidelines for neural network architecture components in the `models/` directory.

## Overview

The `models/` directory contains Graph-Structured Spiking Neural Network (G-SNN) architectures. Focus on graph construction, spiking neuron dynamics, synaptic connections, and EEG readout layers.

## G-SNN Architecture Standards

### Main Model Structure Rules
- Inherit from nn.Module for all neural components
- Initialize components in required order: graph constructor → LIF neurons → synapses → readout
- Implement proper state management with reset_state() method
- Return standardized dictionaries with keys: 'eeg_reconstruction', 'spike_trains', 'graph_weights', 'neuron_states'
- Support both training and inference modes
- Handle device placement explicitly (CPU/GPU)

### Model Registration Requirements
- Register all models in __init__.py with descriptive names
- Implement factory function for model creation
- Support model configuration through Config objects
- Validate model compatibility with configuration
- Provide clear error messages for unsupported configurations

## Graph Constructor Standards

### Graph Construction Rules
- Use PyTorch Geometric Data objects for graph representation
- Support multiple graph types: functional, structural, hybrid
- Implement learnable graph parameters as nn.Parameter
- Apply biological constraints during graph construction
- Validate connectivity patterns against neuroanatomical knowledge
- Support batched graph processing

### Connectivity Requirements
- Compute functional connectivity from EEG correlations
- Apply connectivity thresholds to create sparse graphs
- Maintain edge_index format as (2, num_edges)
- Include edge attributes for synaptic weights
- Support dynamic connectivity updates during training
- Handle disconnected nodes gracefully

### Biological Constraints
- Enforce Dale's principle: separate excitatory/inhibitory neurons
- Apply distance-dependent connectivity decay when spatial info available
- Limit connection probabilities to realistic ranges
- Validate graph topology against biological networks
- Log constraint violations for debugging

## LIF Neuron Implementation Standards

### Brian2 Integration Rules
- Use Brian2 backend for biological accuracy
- Validate all LIF parameters are within biological ranges
- Include proper time constants with units (ms, mV)
- Implement refractory periods correctly
- Support both excitatory and inhibitory neurons
- Handle batch processing efficiently

### Parameter Validation
- tau_m: 1.0-100.0 ms (membrane time constant)
- v_rest: -80.0 to -60.0 mV (resting potential)
- v_threshold: -60.0 to -45.0 mV (spike threshold)
- v_reset: -80.0 to -60.0 mV (reset potential)
- refractory_period: 0.5-10.0 ms
- Raise ValueError for parameters outside valid ranges

### State Management Requirements
- Reset neuron states between sequences
- Initialize membrane potentials near resting potential
- Clear spike history and synaptic currents
- Support state checkpointing for training resumption
- Handle state loading with validation
- Monitor state evolution during training

## Synapse Implementation Standards

### Synaptic Dynamics Rules
- Implement exponential decay filters for spike trains
- Apply synaptic weights through matrix multiplication
- Support both static and plastic synapses
- Handle pre-synaptic and post-synaptic filtering
- Maintain causal temporal relationships
- Validate synaptic weight bounds

### STDP Plasticity Requirements
- Implement spike-timing dependent plasticity when enabled
- Use biologically realistic STDP parameters
- Update plasticity traces with proper time constants
- Apply weight bounds to prevent runaway dynamics
- Support both LTP and LTD mechanisms
- Log plasticity changes for monitoring

### Dale's Principle Enforcement
- Excitatory neurons: only positive outgoing weights
- Inhibitory neurons: only negative outgoing weights
- Validate weight signs during training
- Apply constraints during weight updates
- Handle violations with appropriate corrections

## EEG Readout Standards

### Signal Reconstruction Rules
- Convert spiking activity to continuous EEG signals
- Use spatial filtering to map neurons to EEG channels
- Apply temporal filtering for realistic signal generation
- Support both linear and nonlinear readout methods
- Maintain proper signal amplitude ranges
- Validate output signal characteristics

### Readout Types
- Linear: direct spatial projection with temporal convolution
- Nonlinear: multi-layer approach with activation functions
- Configurable kernel sizes for temporal filtering
- Support different readout methods per application
- Validate readout dimensionality matches EEG channels

## Device and Memory Management

### CUDA Optimization Rules
- Enable mixed precision training when available
- Use tensor cores on compatible hardware
- Optimize memory usage with gradient checkpointing
- Handle large graph operations efficiently
- Clear GPU memory explicitly when needed
- Profile memory usage during development

### Memory Management
- Implement gradient checkpointing for large models
- Use efficient data structures for sparse graphs
- Minimize CPU-GPU data transfers
- Handle out-of-memory errors gracefully
- Monitor memory usage in training loops
- Clear intermediate variables when possible

## Model Validation Standards

### Architecture Validation Requirements
- Check parameter counts against expected ranges
- Validate output shapes match configuration
- Test forward pass with dummy inputs
- Verify gradient flow through all components
- Check model compatibility with loss functions
- Validate device handling across components

### Biological Constraint Testing
- Test Dale's principle enforcement
- Validate LIF parameter ranges
- Check synaptic weight bounds
- Verify spike timing constraints
- Test membrane potential bounds
- Validate connectivity patterns

### Performance Testing
- Profile forward and backward pass timing
- Test memory usage with different batch sizes
- Validate distributed training compatibility
- Check gradient computation efficiency
- Test checkpointing and resumption
- Monitor training stability

## Model Serialization Standards

### Checkpoint Requirements
- Save complete model state including configuration
- Include optimizer state for training resumption
- Store Brian2 simulation state when applicable
- Validate architecture compatibility on load
- Support version compatibility checks
- Handle missing or corrupted checkpoints

### Model Export Rules
- Support standard PyTorch serialization formats
- Include model metadata and version info
- Validate model integrity after loading
- Handle backward compatibility issues
- Provide clear migration paths for updates

## Error Handling and Debugging

### Model Debugging Requirements
- Monitor gradient flow and detect vanishing/exploding gradients
- Validate model outputs for NaN/Inf values
- Check reasonable value ranges for each component
- Log model statistics during training
- Implement debugging hooks for internal states
- Provide diagnostic functions for troubleshooting

### Error Recovery
- Handle CUDA out-of-memory errors
- Recover from Brian2 simulation failures
- Validate inputs before processing
- Provide meaningful error messages
- Log error context for debugging
- Support graceful degradation when possible

## Performance Guidelines

### Optimization Best Practices
- Design for batched input processing
- Use gradient checkpointing for memory efficiency
- Leverage PyTorch Geometric optimizations
- Minimize state copying between components
- Profile and optimize bottleneck operations
- Use efficient graph data structures

### Monitoring Requirements
- Track model performance metrics
- Monitor memory usage patterns
- Log timing information for bottlenecks
- Record biological constraint violations
- Track gradient statistics
- Monitor training stability indicators

## Testing Requirements

### Mandatory Tests
- Unit tests for each model component
- Integration tests for full G-SNN pipeline
- Biological constraint validation tests
- Performance and memory benchmarks
- Gradient flow verification tests
- Device compatibility tests (CPU/GPU)

### Test Coverage Requirements
- Test all supported model configurations
- Validate edge cases and error conditions
- Test with different batch sizes and sequence lengths
- Verify checkpoint saving and loading
- Test distributed training compatibility
- Validate biological parameter ranges