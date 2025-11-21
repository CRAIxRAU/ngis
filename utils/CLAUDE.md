# utils/CLAUDE.md

Infrastructure utilities guidelines for the `utils/` directory.

## Configuration Management (`config.py`)

### Standard Requirements
- Use dataclasses for configuration structures
- Support loading from YAML files with validation
- Implement parameter range validation in __post_init__
- Support configuration inheritance and overrides
- Use OmegaConf for complex nested configurations
- Provide clear error messages for invalid configurations

### Validation Rules
- Always validate parameter ranges and types
- Check file paths exist when specified
- Validate biological parameter bounds
- Ensure configuration compatibility between components
- Log validation warnings and errors appropriately
- Support configuration schema validation

## Logging (`logging.py`)

### Setup Standards
- Use Rich console handlers for colored output
- Support both console and file logging
- Include timestamps and module names in log format
- Enable rich tracebacks for debugging
- Set appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Support distributed logging for multi-process training

### Usage Rules
- Use module-level loggers (logger = logging.getLogger(__name__))
- Use f-strings for variable formatting in log messages
- Log at appropriate levels based on message importance
- Include context information in error messages
- Never use print() statements in production code
- Log performance metrics and system information

## Checkpointing (`checkpointing.py`)

### Essential Components
- Save complete training state: model, optimizer, scheduler, epoch, metrics
- Implement automatic cleanup of old checkpoints
- Support training resumption from interruptions
- Include configuration validation on checkpoint load
- Store best model separately from regular checkpoints
- Validate checkpoint integrity before saving

### Management Rules
- Maintain configurable number of recent checkpoints
- Sort checkpoints by validation performance
- Handle disk space management appropriately
- Provide checkpoint loading validation
- Support distributed training state preservation
- Include metadata and version information

## Metrics (`metrics.py`)

### EEG-Specific Requirements
- Compute reconstruction quality metrics (MSE, MAE, correlation)
- Calculate channel-wise correlation statistics
- Measure spectral similarity between signals
- Track biological constraint satisfaction
- Monitor training stability indicators
- Support both scalar and tensor metric computations

### Computation Rules
- Handle batch processing efficiently
- Validate input tensor shapes and ranges
- Return meaningful metric names and values
- Support distributed metric aggregation
- Cache expensive computations when appropriate
- Provide metric summaries and statistics

## Visualization (`visualization.py`)

### Standard Plot Types
- EEG timeseries with channel overlays
- Connectivity matrices as heatmaps
- Spike raster plots for neural activity
- Training curves with multiple metrics
- Loss component breakdowns
- Biological constraint validation plots

### Visualization Standards
- Use consistent color schemes across plots
- Maintain readable figure sizes and fonts
- Include appropriate axis labels and titles
- Support both interactive and static plotting
- Handle large datasets efficiently
- Save plots in appropriate formats (PNG, PDF, SVG)

## Error Handling Standards

### Robust Function Design
- Implement retry logic for operations that may fail
- Use appropriate exception types for different error conditions
- Log errors with sufficient context for debugging
- Provide meaningful error messages to users
- Handle resource cleanup on failures
- Support graceful degradation when possible

### Input Validation Rules
- Validate tensor shapes and dimensions
- Check parameter ranges and types
- Verify file paths and permissions
- Handle missing or corrupted data
- Validate configuration compatibility
- Provide clear validation error messages

### Recovery Mechanisms
- Continue processing when individual items fail
- Offer alternative methods when primary fails
- Maintain partial results when possible
- Clear resources properly on failures
- Log recovery actions for monitoring
- Support manual error recovery options

## Performance Guidelines

### Optimization Standards
- Profile utilities used in training loops
- Cache expensive computations appropriately
- Use efficient data structures and algorithms
- Minimize memory allocation in hot paths
- Leverage vectorized operations when possible
- Monitor resource usage patterns

### Memory Management
- Clear large objects when no longer needed
- Use generators for large dataset processing
- Implement lazy loading for optional features
- Monitor memory usage in utility functions
- Handle memory fragmentation issues
- Optimize tensor operations for efficiency

## Testing Requirements

### Mandatory Tests
- Unit tests for all utility functions
- Integration tests with main components
- Performance benchmarks for critical utilities
- Memory usage tests for large operations
- Error handling and recovery tests
- Configuration validation tests

### Test Coverage Standards
- Test all supported input types and ranges
- Validate edge cases and error conditions
- Test with different hardware configurations
- Verify distributed processing compatibility
- Test checkpoint saving and loading
- Validate metric computation accuracy

## Key Guidelines

- **Modularity**: Each utility should be independent and reusable
- **Error Handling**: Always include proper exception handling
- **Documentation**: Use clear docstrings with type hints
- **Testing**: Include comprehensive unit tests
- **Performance**: Profile and optimize critical utilities
- **Consistency**: Follow project-wide coding standards