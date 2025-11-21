# training/CLAUDE.md

Guidelines for training system components in the `training/` directory.

## Overview

The `training/` directory handles model training including training loops, distributed training, loss functions, optimizers, and scheduling. This orchestrates model updates and optimization.

## Trainer Architecture Standards

### Main Trainer Requirements
- Initialize components in order: device → model → data → optimization → logging → checkpointing
- Implement comprehensive error handling for training interruptions
- Support both single-GPU and distributed training modes
- Provide clear training progress reporting
- Handle graceful shutdown and state saving
- Validate all configurations before starting training

### Training Loop Rules
- Separate training and validation phases clearly
- Update learning rate after validation phase
- Save checkpoints at regular intervals
- Check early stopping conditions after each epoch
- Log metrics with appropriate frequency
- Handle batch failures gracefully (skip corrupted batches)
- Clear gradients before each optimization step

### Distributed Training Standards
- Use NCCL backend for GPU communication
- Initialize process groups with proper timeout settings
- Wrap models with DistributedDataParallel correctly
- Set find_unused_parameters=False for performance
- Aggregate metrics across all processes
- Handle process cleanup on training completion
- Support training resumption from checkpoints

## Loss Function Standards

### Combined Loss Architecture Rules
- Weight individual loss components appropriately
- Return both total loss and component breakdown
- Validate loss values for NaN/Inf before backpropagation
- Log individual loss components for monitoring
- Support dynamic loss weighting during training
- Handle missing components gracefully

### EEG Reconstruction Loss Requirements
- Support multiple loss types: MSE, correlation-based, spectral
- Compute channel-wise correlations for detailed analysis
- Handle frequency domain losses when specified
- Validate temporal consistency in reconstructed signals
- Scale losses appropriately for different signal magnitudes
- Report correlation statistics for model evaluation

### Biological Constraint Loss Rules
- Enforce Dale's principle with appropriate penalties
- Constrain spike rates to biological ranges (0.1-100 Hz)
- Validate membrane potential bounds during training
- Penalize unnatural connectivity patterns
- Log constraint violations for debugging
- Balance biological realism with training objectives

## Optimizer Standards

### Custom Optimizer Requirements
- Support multiple optimizer types: AdamW, Adam, SGD
- Use different learning rates for different parameter groups
- Apply appropriate weight decay to prevent overfitting
- Set reasonable default hyperparameters
- Validate optimizer configuration before training
- Support learning rate scheduling integration

### Parameter Group Management
- Separate graph, neuron, and readout parameters
- Apply different learning rates based on component sensitivity
- Use lower learning rates for graph parameters
- Higher learning rates for readout layers
- Document learning rate rationales
- Monitor parameter-specific gradient statistics

## Learning Rate Scheduling

### Scheduler Configuration Rules
- Support multiple scheduler types: ReduceLROnPlateau, CosineAnnealing, StepLR
- Implement warmup period for training stability
- Use appropriate metrics for plateau detection
- Set reasonable minimum learning rates
- Log learning rate changes
- Handle scheduler state in checkpoints

### Adaptive Scheduling Requirements
- Monitor validation loss for plateau detection
- Reduce learning rate when training stagnates
- Support custom scheduling policies
- Validate scheduler compatibility with optimizer
- Handle learning rate bounds properly
- Provide scheduler status in logs

## Checkpointing Standards

### Comprehensive State Saving
- Save complete training state: model, optimizer, scheduler, epoch, metrics
- Include configuration and random states
- Store best model separately from regular checkpoints
- Implement automatic cleanup of old checkpoints
- Validate checkpoint integrity on save
- Support distributed training state preservation

### Checkpoint Management Rules
- Maintain configurable number of recent checkpoints
- Keep best performing checkpoints permanently
- Sort checkpoints by validation performance
- Handle disk space management
- Provide checkpoint loading validation
- Support training resumption from any checkpoint

## Validation and Early Stopping

### Validation Implementation Requirements
- Run validation after each training epoch
- Compute comprehensive metrics: MSE, MAE, correlation, spectral similarity
- Disable gradient computation during validation
- Handle distributed validation aggregation
- Log validation progress appropriately
- Clear model state between validation runs

### Early Stopping Rules
- Monitor primary metric (typically validation loss)
- Use configurable patience for stopping decisions
- Require minimum improvement delta
- Support both minimization and maximization metrics
- Log early stopping decisions
- Save best model before stopping

### Validation Metrics Standards
- EEG reconstruction quality (MSE, MAE)
- Channel-wise correlation analysis
- Spectral similarity measurements
- Biological constraint satisfaction
- Training stability indicators
- Memory usage monitoring

## Performance Monitoring

### Training Performance Rules
- Profile memory usage throughout training
- Monitor GPU utilization and efficiency
- Track timing for bottleneck identification
- Record gradient statistics
- Log hardware performance metrics
- Detect and report performance anomalies

### Profiling Requirements
- Use PyTorch profiler for detailed analysis
- Monitor CUDA memory allocation patterns
- Track data loading bottlenecks
- Profile distributed communication overhead
- Report performance summaries
- Identify optimization opportunities

## Error Handling and Recovery

### Robust Training Rules
- Implement retry logic for recoverable errors
- Handle out-of-memory errors gracefully
- Recover from NCCL communication failures
- Save interruption checkpoints on user termination
- Provide meaningful error diagnostics
- Support automatic batch size reduction

### Error Recovery Strategies
- Reduce batch size on OOM errors
- Enable gradient checkpointing for memory issues
- Reinitialize distributed training on communication failures
- Skip corrupted batches and continue training
- Clear GPU cache on memory errors
- Log all recovery actions for debugging

### Training Resumption
- Automatically detect and load latest checkpoint
- Restore exact training state including random seeds
- Validate configuration compatibility on resume
- Handle distributed training resumption properly
- Log resumption information clearly
- Support manual checkpoint selection

## Performance Guidelines

### Optimization Best Practices
- Use mixed precision training for memory efficiency
- Apply gradient clipping to prevent exploding gradients
- Scale batch size appropriately for available hardware
- Monitor and optimize data loading pipelines
- Use efficient tensor operations
- Profile and eliminate bottlenecks

### Distributed Training Optimization
- Minimize communication overhead between processes
- Use appropriate backend settings for hardware
- Balance computation and communication
- Optimize batch sizes for multi-GPU efficiency
- Handle load balancing across processes
- Monitor distributed training efficiency

### Memory Management
- Monitor GPU memory usage throughout training
- Clear unused variables explicitly
- Use gradient checkpointing for large models
- Handle memory fragmentation issues
- Implement memory profiling hooks
- Optimize tensor storage and operations

## Testing Requirements

### Mandatory Tests
- Unit tests for individual training components
- Integration tests for complete training loops
- Distributed training functionality tests
- Error recovery mechanism tests
- Checkpoint saving and loading tests
- Performance regression tests

### Test Coverage Requirements
- Test all supported optimizer configurations
- Validate loss function combinations
- Test scheduler behavior under various conditions
- Verify error handling for common failure modes
- Test distributed training with multiple processes
- Validate training resumption from checkpoints

### Performance Testing
- Benchmark training speed with different configurations
- Test memory usage patterns and limits
- Validate distributed training scaling
- Test error recovery mechanisms
- Benchmark checkpoint operations
- Verify profiling tool integration