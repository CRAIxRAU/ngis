# EEG Processing Tests

This directory contains comprehensive tests for the EEG data processing pipeline, including channel selection algorithms and data validation.

## Test Files

### Core Tests
- **`test_eeg_loader.py`** - Main test suite for EEG file loading and channel selection
- **`run_all_tests.py`** - Test runner script that executes all tests

### Channel Selection Tests
- **`test_university_standard.py`** - Tests the university-standard channel selection strategy
- **`validate_channel_selection.py`** - Comprehensive validation of spatial accuracy and coverage
- **`analyze_electrode_layout.py`** - Analysis comparing our selection vs real university EEG systems

## Running Tests

### Run All Tests
```bash
cd tests
python run_all_tests.py
```

### Run Individual Tests
```bash
cd tests
python test_eeg_loader.py                    # Core EEG loader tests
python test_university_standard.py          # University standard selection
python validate_channel_selection.py        # Channel selection validation
python analyze_electrode_layout.py          # Electrode layout analysis
```

## Test Coverage

### EEG Loading (`test_eeg_loader.py`)
- ✅ BrainVision format loading (.vhdr, .eeg, .vmrk)
- ✅ Channel selection integration (256→128 channels)
- ✅ Data integrity validation
- ✅ Multiple selection strategies

### Channel Selection (`validate_channel_selection.py`)
- ✅ Spatial coverage validation (>99% coverage)
- ✅ Channel name consistency
- ✅ Regional distribution analysis
- ✅ Visual verification (generates PNG plots)

### University Standard (`test_university_standard.py`)
- ✅ Scalp-only electrode selection
- ✅ Temporal electrode inclusion
- ✅ Face/neck electrode exclusion
- ✅ Realistic brain region distribution

### Electrode Analysis (`analyze_electrode_layout.py`)
- ✅ Comparison with typical 128-channel EEG systems
- ✅ Face/neck electrode identification
- ✅ Regional distribution recommendations

## Expected Results

When all tests pass, you should see:

```
🏁 Test Summary:
   EEG Loader and Channel Selection          ✅ PASSED
   University Standard Channel Selection     ✅ PASSED  
   Channel Selection Validation              ✅ PASSED
   Electrode Layout Analysis                 ✅ PASSED

🎉 All tests passed! EEG processing pipeline is ready.
```

## Key Validation Metrics

### Channel Selection Quality
- **Spatial Coverage**: >99% across X, Y, Z dimensions
- **Channel Matching**: 128/128 channels with valid positions
- **Face/Neck Exclusion**: 0 electrodes below scalp line
- **Temporal Coverage**: >10% for language/auditory processing

### University Standard Distribution
- **Frontal**: ~20% (F, FC regions)
- **Central**: ~25% (C, CP regions - motor/sensory)
- **Parietal**: ~25% (P, CP regions - cognitive)
- **Occipital**: ~20% (O, PO regions - visual)
- **Temporal**: ~10% (T, TP regions - language/auditory)

## Troubleshooting

### Common Issues
1. **Missing data files**: Ensure dataset is downloaded with `python scripts/download_data.py --sample`
2. **Import errors**: Tests expect to run from the `tests/` directory
3. **Missing dependencies**: Install with `pip install matplotlib pandas numpy`

### File Requirements
- EEG data: `data/raw/*/eeg/*.vhdr` files
- Electrode positions: `data/raw/*/eeg/*electrodes.tsv` files
- MNE-Python and dependencies installed