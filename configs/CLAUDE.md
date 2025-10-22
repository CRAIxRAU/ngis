# configs/CLAUDE.md

Configuration management guidelines for the `configs/` directory.

## File Organization Standards

### Required Configuration Files
- `default.yaml`: Standard single-GPU configuration
- `ddp.yaml`: Multi-GPU distributed training setup
- `debug.yaml`: Fast debugging configuration with small datasets
- Environment-specific configs: `dev.yaml`, `prod.yaml`
- Experiment-specific configs for different model variants

### Configuration Structure Requirements
- Group related parameters under clear sections: data, model, training, loss
- Use consistent parameter naming across all configuration files
- Include version information for backward compatibility
- Document complex or non-obvious parameters with comments
- Maintain hierarchy that reflects code organization

## Parameter Validation Rules

### Mandatory Validations
- Biological constraints: LIF parameters within realistic ranges
- Hardware limits: Batch size fits available GPU memory
- File paths: Data directories exist and are accessible
- Compatibility: Model and data dimensions match
- Type checking: All parameters have correct types
- Range validation: Numerical parameters within acceptable bounds

### Biological Parameter Bounds
- tau_m: 1.0-100.0 ms (membrane time constant)
- v_rest: -80.0 to -60.0 mV (resting potential)
- v_threshold: -60.0 to -45.0 mV (spike threshold)
- v_reset: -80.0 to -60.0 mV (reset potential)
- refractory_period: 0.5-10.0 ms
- Connection probabilities: 0.01-0.5 (realistic ranges)

### Hardware Validation
- Batch size must fit in GPU memory
- Number of workers should not exceed CPU cores
- Model size must fit in available memory
- Data path must be accessible from all processes
- Output directories must be writable

## Parameter Naming Standards

### Naming Conventions
- Use snake_case for all parameter names
- Group related parameters under logical sections
- Include units in parameter names where ambiguous
- Use descriptive names over abbreviations
- Maintain consistency across configuration files
- Avoid special characters and spaces

### Section Organization
- `data`: Data loading, preprocessing, and dataset configuration
- `model`: Neural network architecture and parameters
- `training`: Training loop, optimization, and scheduling
- `loss`: Loss function weights and configuration
- `logging`: Logging levels and output destinations
- `checkpointing`: Checkpoint saving and loading settings

## Default Value Strategy

### Default Value Requirements
- Provide sensible defaults for all parameters
- Use values from established literature for biological parameters
- Scale defaults based on typical hardware capabilities
- Ensure defaults produce working configurations
- Document the source or reasoning for each default
- Test defaults across different hardware configurations

### Literature-Based Defaults
- Use published values for LIF neuron parameters
- Reference connectivity patterns from neuroscience literature
- Base EEG processing parameters on standard practices
- Validate defaults against experimental data when available

## Environment Override Support

### Command Line Integration
- Support CLI overrides for commonly changed parameters
- Use clear parameter substitution syntax
- Provide default values when overrides not specified
- Validate overridden parameters appropriately
- Log when parameters are overridden from command line

### Environment Variable Support
- Enable environment variable substitution for deployment
- Support different configurations for development vs production
- Handle missing environment variables gracefully
- Validate environment variable values
- Document required environment variables

## Distributed Training Configuration

### Multi-GPU Requirements
- Configure appropriate backend (NCCL for GPUs)
- Set per-GPU batch sizes correctly
- Adjust number of workers per process
- Enable synchronized batch normalization
- Configure communication timeouts
- Handle process group initialization

### Distributed Settings
- World size and rank configuration
- Backend selection and optimization
- Communication URL and port settings
- Timeout configuration for robustness
- Load balancing considerations
- Error handling for process failures

## Configuration Inheritance

### Inheritance Rules
- Support base configuration with overrides
- Allow composition of configuration modules
- Enable experiment-specific parameter overrides
- Maintain clear inheritance hierarchy
- Validate inheritance compatibility
- Document inheritance relationships

### Override Priorities
- Command line arguments (highest priority)
- Environment variables
- Configuration file overrides
- Base configuration defaults (lowest priority)

## Version Management

### Backward Compatibility Requirements
- Include config_version field in all configurations
- Support loading older configuration versions
- Provide migration warnings for deprecated parameters
- Maintain changelog for configuration format changes
- Test compatibility across versions
- Handle missing parameters gracefully

### Migration Support
- Automatic parameter name migration
- Default value updates for new parameters
- Deprecation warnings for old parameters
- Clear migration documentation
- Validation of migrated configurations

## Validation and Error Handling

### Configuration Validation
- Validate all parameters on load
- Check parameter compatibility between sections
- Verify file paths and permissions
- Validate hardware requirements
- Check biological parameter constraints
- Provide clear error messages for violations

### Error Reporting
- Include parameter names and values in error messages
- Suggest corrections for common mistakes
- Provide context for validation failures
- Log validation warnings appropriately
- Support dry-run validation mode

## Key Guidelines

- **Validation**: Always validate configurations thoroughly on load
- **Documentation**: Comment complex parameters and provide examples
- **Defaults**: Ensure all defaults produce working configurations
- **Flexibility**: Support overrides via CLI and environment variables
- **Compatibility**: Maintain backward compatibility across versions
- **Testing**: Test configurations across different hardware setups