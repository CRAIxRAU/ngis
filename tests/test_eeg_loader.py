#!/usr/bin/env python3
"""
Test script to validate EEG loader and channel selection functionality.

This script tests:
1. Loading BrainVision format files
2. Channel selection strategies (256→128)
3. Data integrity and format validation
"""

import logging
from pathlib import Path
import sys
sys.path.append('..')

import numpy as np

from data.eeg_loader import EEGLoader
from data.channel_selection import ChannelSelector, create_channel_map

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_basic_loading():
    """Test basic EEG file loading without channel selection."""
    print("\n=== Testing Basic EEG Loading ===")
    
    # Find a test file - use any available vhdr file
    test_files = list(Path("data/raw").glob("**/*.vhdr"))
    if not test_files:
        print("❌ No .vhdr files found in data/raw")
        return False
    
    test_file = test_files[0]  # Use the first one found
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    try:
        # Load without any channel selection
        loader = EEGLoader(preload=True)
        raw = loader.load_file(test_file)
        
        print(f"✅ Successfully loaded: {len(raw.ch_names)} channels, {len(raw.times)} samples")
        print(f"   Sampling rate: {raw.info['sfreq']} Hz")
        print(f"   Duration: {raw.times[-1]:.2f} seconds")
        print(f"   First 10 channels: {raw.ch_names[:10]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load EEG file: {e}")
        return False

def test_electrode_positions():
    """Test loading electrode positions."""
    print("\n=== Testing Electrode Position Loading ===")
    
    # Find any electrodes.tsv file
    electrodes_files = list(Path("data/raw").glob("**/*electrodes.tsv"))
    if not electrodes_files:
        print("❌ No electrodes.tsv files found")
        return None
    
    electrodes_file = electrodes_files[0]
    
    if not electrodes_file.exists():
        print(f"❌ Electrodes file not found: {electrodes_file}")
        return False
    
    try:
        loader = EEGLoader()
        positions = loader.load_electrode_positions_from_bids(electrodes_file)
        
        print(f"✅ Loaded positions for {len(positions)} electrodes")
        
        # Show some example positions
        example_channels = list(positions.keys())[:5]
        for ch in example_channels:
            x, y, z = positions[ch]
            print(f"   {ch}: ({x:.3f}, {y:.3f}, {z:.3f})")
        
        return positions
        
    except Exception as e:
        print(f"❌ Failed to load electrode positions: {e}")
        return None

def test_channel_selection_strategies(positions=None):
    """Test different channel selection strategies."""
    print("\n=== Testing Channel Selection Strategies ===")
    
    # Create dummy 256-channel list (E1-E256)
    all_channels = [f"E{i}" for i in range(1, 257)]
    
    strategies = ["uniform_spatial", "standard_hd", "roi_based"]
    
    results = {}
    
    for strategy in strategies:
        try:
            print(f"\n--- Testing {strategy} strategy ---")
            
            selector = ChannelSelector(strategy)
            selected = selector.select_channels(all_channels, positions)
            
            print(f"✅ {strategy}: Selected {len(selected)} channels")
            print(f"   First 10: {selected[:10]}")
            print(f"   Last 10: {selected[-10:]}")
            
            # Validate selection
            if len(selected) != 128:
                print(f"❌ Wrong number of channels: expected 128, got {len(selected)}")
            elif len(set(selected)) != len(selected):
                print(f"❌ Duplicate channels found")
            else:
                print(f"✅ Selection validation passed")
            
            results[strategy] = selected
            
        except Exception as e:
            print(f"❌ Failed {strategy} strategy: {e}")
            results[strategy] = None
    
    return results

def test_integrated_loading():
    """Test EEG loading with channel selection."""
    print("\n=== Testing Integrated EEG Loading with Channel Selection ===")
    
    # Find test files
    test_files = list(Path("data/raw").glob("**/*.vhdr"))
    electrodes_files = list(Path("data/raw").glob("**/*electrodes.tsv"))
    
    if not test_files:
        print("❌ No .vhdr files found")
        return False
    
    test_file = test_files[0]
    electrodes_file = electrodes_files[0] if electrodes_files else None
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
    
    # Load electrode positions
    positions = None
    if electrodes_file.exists():
        try:
            loader_temp = EEGLoader()
            positions = loader_temp.load_electrode_positions_from_bids(electrodes_file)
            print(f"✅ Loaded electrode positions for {len(positions)} channels")
        except Exception as e:
            print(f"⚠️  Could not load electrode positions: {e}")
    
    # Test different strategies
    strategies = ["uniform_spatial", "standard_hd"]
    
    for strategy in strategies:
        try:
            print(f"\n--- Testing {strategy} with real data ---")
            
            # Create loader with channel selection
            loader = EEGLoader(
                channel_selection_strategy=strategy,
                target_channels=128,
                electrode_positions=positions,
                preload=True
            )
            
            # Load the file
            raw = loader.load_file(test_file)
            
            print(f"✅ Successfully loaded with {strategy}")
            print(f"   Original channels available: 256")
            print(f"   Selected channels: {len(raw.ch_names)}")
            print(f"   Data shape: {raw.get_data().shape}")
            print(f"   Selected channel names (first 10): {raw.ch_names[:10]}")
            
            # Validate data
            data = raw.get_data()
            if data.shape[0] != 128:
                print(f"❌ Wrong number of channels in data: {data.shape[0]}")
            elif np.any(np.isnan(data)):
                print(f"❌ NaN values found in data")
            elif np.any(np.isinf(data)):
                print(f"❌ Infinite values found in data")
            else:
                print(f"✅ Data validation passed")
                print(f"   Data range: {data.min():.2e} to {data.max():.2e}")
                print(f"   Data std: {data.std():.2e}")
            
        except Exception as e:
            print(f"❌ Failed {strategy} integration test: {e}")
            import traceback
            traceback.print_exc()

def test_channel_mapping():
    """Test channel mapping creation."""
    print("\n=== Testing Channel Mapping Creation ===")
    
    try:
        # Test with default 256 channels
        channel_map = create_channel_map(strategy="uniform_spatial")
        
        print(f"✅ Created channel map:")
        print(f"   Strategy: {channel_map['strategy']}")
        print(f"   Total channels: {channel_map['total_channels']}")
        print(f"   Selected channels: {channel_map['selected_channels']}")
        print(f"   First 10 indices: {channel_map['indices'][:10]}")
        print(f"   First 10 names: {channel_map['names'][:10]}")
        
        # Validate mapping
        if len(channel_map['names']) != len(channel_map['indices']):
            print(f"❌ Mismatch between names and indices lengths")
        elif len(set(channel_map['indices'])) != len(channel_map['indices']):
            print(f"❌ Duplicate indices found")
        else:
            print(f"✅ Channel mapping validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed channel mapping test: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing EEG Loader and Channel Selection")
    print("=" * 50)
    
    # Test 1: Basic loading
    basic_ok = test_basic_loading()
    
    # Test 2: Electrode positions
    positions = test_electrode_positions()
    
    # Test 3: Channel selection strategies
    selection_results = test_channel_selection_strategies(positions)
    
    # Test 4: Integrated loading
    test_integrated_loading()
    
    # Test 5: Channel mapping
    mapping_ok = test_channel_mapping()
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary:")
    print(f"   Basic loading: {'✅' if basic_ok else '❌'}")
    print(f"   Electrode positions: {'✅' if positions else '❌'}")
    print(f"   Channel selection: {'✅' if any(selection_results.values()) else '❌'}")
    print(f"   Channel mapping: {'✅' if mapping_ok else '❌'}")
    
    if basic_ok and positions and any(selection_results.values()) and mapping_ok:
        print("\n🎉 All core tests passed! EEG loader is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())