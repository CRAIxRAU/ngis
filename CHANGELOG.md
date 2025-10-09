# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Added
- Support for EEGLAB format (.set/.fdt files) in EEG loader
- Compatible with ds003766 OpenNeuro dataset (128-channel EGI G128 at 1000Hz)

### Changed
- Updated `data/eeg_loader.py` to use `len(raw.ch_names)` instead of `raw.n_channels` for better compatibility across all MNE file formats
- EEGLAB format does not have `n_channels` property, so channel count is now calculated from channel names list
- Fixed `main.py` config loading to use `Config.from_yaml()` instead of broken `Config(config_dict)` approach

### Fixed
- Fixed `utils/logging.py` to accept both int and string log levels (handles `logging.DEBUG` int constant and string "DEBUG")
- Fixed `training/scheduler.py` to avoid duplicate keyword arguments when creating ReduceLROnPlateau scheduler
- Fixed `training/trainer.py` to pass inner PyTorch optimizer to scheduler (NGISOptimizer wrapper compatibility)
- Added `param_groups` property to NGISOptimizer for better compatibility

### Technical Details
- Added `read_raw_eeglab` import from `mne.io`
- Added `.set` file handling in `load_file()` method
- Replaced `raw.n_channels` with `len(raw.ch_names)` in 3 locations: logging (line 104), get_info (line 167), and validate_data (line 192)
- This change is backward compatible with all existing formats (EDF, BDF, FIF)
- `Config.from_yaml()` uses OmegaConf to properly create nested configuration structure with attribute access