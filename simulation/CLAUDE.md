# simulation/CLAUDE.md

Guidelines for spiking neural network simulation components in the `simulation/` directory.

## Overview

The `simulation/` directory handles biologically realistic neural simulation using Brian2. Focus on LIF neuron dynamics, synaptic transmission, spike processing, and integration with PyTorch training loops.

## Brian2 Integration Standards

### Setup and Configuration Rules
- Use C++ standalone mode for performance (set_device('cpp_standalone'))
- Enable Cython code generation for speed (prefs.codegen.target = 'cython')
- Set appropriate time step (default 0.1 ms)
- Configure OpenMP threads for parallel execution
- Initialize default clock settings globally
- Clean output directory between simulation runs

### Brian2 Optimization Requirements
- Use compiled simulation for large-scale networks
- Enable compiler optimizations (-O3, -march=native)
- Set appropriate number of OpenMP threads
- Use loop invariant optimizations
- Profile simulation bottlenecks regularly
- Handle memory allocation efficiently

## LIF Neuron Implementation Standards

### Biological Parameter Validation
- Membrane time constant (tau_m): 1-100 ms
- Resting potential (v_rest): -80 to -60 mV
- Spike threshold (v_threshold): -60 to -45 mV
- Reset potential (v_reset): -80 to -60 mV
- Refractory period: 0.5-10 ms
- Synaptic time constant (tau_syn): 1-20 ms

### Neuron Group Setup Rules
- Define LIF equations with proper units
- Set threshold and reset conditions correctly
- Initialize membrane potentials near resting potential
- Separate excitatory (80%) and inhibitory (20%) populations
- Use different parameters for E/I neuron types
- Add small random variation to initial conditions

### State Management Requirements
- Reset neuron states between simulation runs
- Clear spike history and synaptic currents
- Initialize all state variables properly
- Support state checkpointing for training resumption
- Monitor state evolution during long simulations
- Validate state bounds throughout simulation

## Synaptic Connection Standards

### Connection Topology Rules
- Implement realistic connection probabilities
- Separate excitatory-excitatory, excitatory-inhibitory, etc. connections
- Use sparse connectivity matrices for efficiency
- Avoid self-connections unless biologically justified
- Apply distance-dependent connection rules when spatial info available
- Validate connectivity patterns against biological data

### STDP Plasticity Requirements
- Use biologically realistic STDP time constants (10-40 ms)
- Implement both LTP and LTD mechanisms
- Set appropriate learning rates (typically 0.01-0.1)
- Apply synaptic weight bounds to prevent instability
- Update plasticity traces with event-driven dynamics
- Monitor plasticity changes during training

### Dale's Principle Enforcement
- Excitatory neurons: only positive outgoing weights
- Inhibitory neurons: only negative outgoing weights
- Validate weight signs during initialization
- Enforce constraints during weight updates
- Handle violations with appropriate corrections
- Log constraint violations for debugging

## Forward Simulation Standards

### PyTorch Integration Rules
- Convert PyTorch tensors to numpy for Brian2 processing
- Process batches sequentially to avoid Brian2 conflicts
- Convert simulation results back to PyTorch tensors
- Maintain gradient flow where possible
- Handle device placement correctly (CPU for Brian2)
- Clear Brian2 state between batch elements

### Simulation Execution Requirements
- Update synaptic weights from graph parameters
- Set external input currents appropriately
- Reset simulation state between runs
- Execute simulation for specified duration
- Extract spike trains and membrane potentials
- Validate simulation results before returning

### Biological Constraint Validation
- Check firing rates are within biological range (0.1-100 Hz)
- Validate membrane potential bounds (-80 to -40 mV)
- Ensure inter-spike intervals respect refractory periods
- Monitor population synchrony levels
- Verify Dale's principle compliance
- Log constraint violations with context

## Spike Processing Standards

### Spike Train Analysis Rules
- Compute firing rates over appropriate time windows
- Calculate inter-spike interval statistics
- Measure coefficient of variation for spike timing
- Compute Fano factor for spike count variability
- Analyze population correlation patterns
- Generate spike raster plots for visualization

### Statistical Validation Requirements
- Validate spike timing distributions
- Check for excessive synchronization
- Monitor spike rate stability over time
- Analyze cross-correlation patterns
- Detect pathological activity patterns
- Compare against experimental data when available

### Performance Optimization
- Use efficient spike detection algorithms
- Minimize memory allocation during processing
- Cache frequently computed statistics
- Parallel process independent analyses
- Profile spike analysis bottlenecks
- Optimize data structures for sparse spike trains

## Simulation Utilities

### Batch Processing Rules
- Process simulation batches in chunks to manage memory
- Clear Brian2 device state between chunks
- Handle batch size optimization automatically
- Monitor memory usage during batch processing
- Implement retry logic for failed simulations
- Log batch processing statistics

### Memory Management Standards
- Monitor Brian2 memory usage
- Clear simulation objects between runs
- Use memory mapping for large spike datasets
- Implement garbage collection hooks
- Handle out-of-memory errors gracefully
- Profile memory allocation patterns

### Error Handling Requirements
- Catch and handle Brian2-specific exceptions
- Implement retry logic for simulation failures
- Validate simulation inputs before execution
- Handle dimension mismatch errors
- Provide meaningful error messages
- Log error context for debugging

## Performance Guidelines

### Optimization Best Practices
- Use standalone mode for large simulations
- Enable compiler optimizations appropriately
- Set optimal number of simulation threads
- Profile and eliminate bottlenecks
- Use efficient data structures
- Minimize Python-Brian2 interface overhead

### Scalability Considerations
- Design for batch processing efficiency
- Handle memory constraints gracefully
- Implement chunked processing for large datasets
- Monitor computational resource usage
- Scale simulation parameters appropriately
- Balance accuracy with computational cost

### Monitoring and Profiling
- Track simulation execution times
- Monitor memory usage patterns
- Profile Brian2 code generation
- Identify computational bottlenecks
- Log performance metrics
- Compare performance across configurations

## Integration with Training

### Gradient Flow Management
- Identify differentiable vs non-differentiable operations
- Handle simulation boundaries appropriately
- Minimize gradient computation overhead
- Validate gradient flow through simulation boundaries
- Implement surrogate gradients where needed
- Monitor gradient quality during training

### Training Loop Integration
- Reset simulation state between training batches
- Handle variable batch sizes efficiently
- Coordinate with PyTorch training loops
- Manage device placement consistently
- Log simulation metrics during training
- Handle training interruptions gracefully

## Testing Requirements

### Mandatory Tests
- Unit tests for Brian2 integration components
- Biological constraint validation tests
- Performance benchmarks for different configurations
- Memory usage tests with various batch sizes
- Gradient flow verification tests
- Error handling and recovery tests

### Biological Validation Tests
- Test LIF parameter bounds enforcement
- Validate Dale's principle compliance
- Check spike rate constraint satisfaction
- Test membrane potential bound validation
- Verify refractory period enforcement
- Validate connectivity pattern generation

### Performance Testing Requirements
- Benchmark simulation speed vs batch size
- Test memory usage scaling
- Validate chunked processing efficiency
- Test error recovery mechanisms
- Benchmark Brian2 code generation
- Profile integration overhead with PyTorch

### Integration Testing
- Test complete simulation pipeline
- Validate PyTorch tensor conversions
- Test distributed training compatibility
- Verify checkpoint saving/loading
- Test simulation resumption from saved states
- Validate batch processing consistency