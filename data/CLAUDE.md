# data/CLAUDE.md

Guidelines for EEG data pipeline components in the `data/` directory.

## Overview

The `data/` directory handles EEG data processing from raw file loading to PyTorch-ready datasets. Focus on format support, preprocessing, segmentation, and efficient batch loading.

## EEG File Format Standards

### Supported Formats
- Always support: .edf, .bdf, .fif, .set, .cnt
- Use MNE-Python for consistent file loading
- Auto-detect format from file extension
- Preserve all metadata during loading
- Handle corrupted files gracefully with retry logic

### File Loading Rules
- Return data as (channels, time) numpy arrays
- Extract and preserve sampling rate, channel names, duration
- Validate file exists before attempting to load
- Use appropriate error handling for each format
- Log loading progress for large files

## EEG Data Standards

### Channel Configuration
- Default: 128 channels (high-density EEG)
- Channel naming: Use standardized 10-20 system names
- Maintain consistent channel ordering across datasets
- Handle missing channels with interpolation or masking
- Validate against expected channel prefixes (Fp, F, C, P, O, T, FC, CP, FT, TP)

### Sampling Rate Standards
- Default: 1000 Hz for consistency
- Acceptable range: 250-4000 Hz
- Always resample to standard rate
- Log resampling operations
- Validate final sampling rate matches expectations

## Preprocessing Pipeline Standards

### Required Preprocessing Steps
1. Bad channel detection and marking
2. Bandpass filtering (0.1-100 Hz default)
3. Artifact removal (eye blinks, muscle artifacts)
4. Amplitude normalization
5. Common average referencing

### Preprocessing Rules
- Apply steps in the specified order
- Validate each step's output before proceeding
- Log all preprocessing operations
- Check for NaN values after each step
- Warn about unusual amplitude ranges
- Identify and log flat channels

### Quality Control
- Reject recordings shorter than minimum duration (1 second)
- Flag unusually high amplitudes (>5000 µV)
- Check for missing or infinite values
- Compute and log signal quality metrics
- Validate channel count is reasonable (64, 128, or 256)

## Segmentation Standards

### Segment Configuration
- Default length: 1000 samples (1 second at 1000 Hz)
- Support configurable overlap (0-50% recommended)
- Minimum segment length: 250 samples
- Handle incomplete segments by padding or discarding
- Log total number of segments created

### Segmentation Rules
- Validate segment length doesn't exceed data length
- Use consistent step size calculation
- Maintain temporal continuity in overlapping segments
- Preserve metadata for each segment
- Support both fixed and variable length segments

## Dataset Implementation Standards

### PyTorch Dataset Requirements
- Inherit from torch.utils.data.Dataset
- Implement __len__ and __getitem__ methods
- Return dictionaries with clear keys ('eeg', 'metadata')
- Support data augmentation with training flags
- Handle transforms consistently
- Validate tensor shapes and dtypes

### DataLoader Configuration
- Use DistributedSampler for multi-GPU training
- Enable pin_memory for GPU efficiency
- Set appropriate num_workers based on system
- Use persistent_workers when num_workers > 0
- Implement custom collate function if needed

## Data Validation Rules

### Input Validation Requirements
- Ensure data is 2D (channels, time)
- Validate channel count against expected values
- Check recording duration meets minimum requirements
- Verify amplitude ranges are reasonable
- Reject data with NaN or infinite values
- Log warnings for unusual characteristics

### Quality Metrics
- Compute signal-to-noise ratio
- Calculate amplitude statistics (range, std)
- Count flat and noisy channels
- Measure recording completeness
- Track temporal quality metrics

## Memory Management Guidelines

### Efficient Loading Strategies
- Use lazy loading for large datasets
- Implement memory mapping for very large files
- Load only required time segments when possible
- Cache frequently accessed data
- Monitor memory usage during processing

### Performance Optimization
- Use multiprocessing for dataset preparation
- Optimize batch sizes for available GPU memory
- Cache preprocessed segments to disk
- Profile memory usage in training loops
- Clear unused data from memory explicitly

## Error Handling Rules

### Robust Operations
- Implement retry logic for file I/O operations
- Handle format-specific loading errors
- Gracefully degrade when files are corrupted
- Log all errors with sufficient context
- Provide meaningful error messages to users

### Recovery Strategies
- Continue processing other files if one fails
- Offer alternative loading methods
- Implement fallback preprocessing options
- Maintain partial results when possible
- Clear resources properly on failures

## Testing Requirements

### Mandatory Tests
- Unit tests for each supported file format
- Integration tests with full preprocessing pipeline
- Memory leak detection for large datasets
- Performance benchmarks for loading operations
- Edge case handling (corrupted files, missing channels)
- Validation of data shapes and types

### Test Data Requirements
- Sample files for each supported format
- Files with various channel counts
- Short and long duration recordings
- Files with artifacts and noise
- Corrupted or incomplete files for error testing