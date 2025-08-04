#!/usr/bin/env python3
"""
Analyze the electrode layout to understand what we're actually selecting.

Compare our selection against typical 128-channel EEG systems used in universities.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_electrode_data():
    """Load the actual electrode positions and names."""
    electrodes_files = list(Path("data/raw").glob("**/*electrodes.tsv"))
    electrodes_file = electrodes_files[0]
    
    df = pd.read_csv(electrodes_file, sep='\t')
    print(f"Dataset has {len(df)} electrodes total")
    print(f"First 20 electrode names: {df['name'].tolist()[:20]}")
    
    return df

def analyze_electrode_positions(df):
    """Analyze the 3D positions to understand the layout."""
    print(f"\n=== Electrode Position Analysis ===")
    
    # Look at the coordinate ranges
    x_range = df['x'].max() - df['x'].min()
    y_range = df['y'].max() - df['y'].min() 
    z_range = df['z'].max() - df['z'].min()
    
    print(f"X range (left-right): {df['x'].min():.1f} to {df['x'].max():.1f} ({x_range:.1f})")
    print(f"Y range (anterior-posterior): {df['y'].min():.1f} to {df['y'].max():.1f} ({y_range:.1f})")
    print(f"Z range (inferior-superior): {df['z'].min():.1f} to {df['z'].max():.1f} ({z_range:.1f})")
    
    # Find electrodes that might be on the face/neck (very anterior Y or very low Z)
    face_electrodes = df[(df['y'] > 8) | (df['z'] < -8)]
    if len(face_electrodes) > 0:
        print(f"\n⚠️  Found {len(face_electrodes)} electrodes that might be on face/neck:")
        for _, row in face_electrodes.head(10).iterrows():
            print(f"   {row['name']}: ({row['x']:.1f}, {row['y']:.1f}, {row['z']:.1f})")
    
    # Find electrodes on scalp (typical EEG locations)
    scalp_electrodes = df[(df['y'] <= 8) & (df['z'] >= -6)]
    print(f"\n✅ {len(scalp_electrodes)} electrodes appear to be on scalp")
    
    return face_electrodes, scalp_electrodes

def compare_with_standard_128():
    """Compare with typical 128-channel EEG layouts."""
    print(f"\n=== Comparison with Standard 128-Channel EEG ===")
    
    # Typical 128-channel layouts (like EGI, Biosemi, etc.)
    typical_128_regions = {
        'frontal': 25,      # ~20% - F, AF, Fp regions
        'central': 35,      # ~27% - C, FC, CP regions  
        'parietal': 30,     # ~23% - P, CP regions
        'occipital': 25,    # ~20% - O, PO regions
        'temporal': 13      # ~10% - T, FT, TP regions
    }
    
    print("Typical 128-channel EEG distribution:")
    for region, count in typical_128_regions.items():
        percentage = (count / 128) * 100
        print(f"   {region:>10}: {count:>3} electrodes ({percentage:>5.1f}%)")
    
    print(f"\n🎓 University EEG systems typically:")
    print(f"   - Cover scalp only (no face/neck electrodes)")
    print(f"   - Focus on brain activity regions")
    print(f"   - Avoid eye movement artifacts (minimal frontal)")
    print(f"   - Emphasize motor/sensory areas (more central/parietal)")
    
    return typical_128_regions

def analyze_our_selection_strategy():
    """Analyze what our current selection is actually doing."""
    print(f"\n=== Our Selection Strategy Analysis ===")
    
    df = load_electrode_data()
    
    # Show where our electrodes actually are
    print(f"\nOur current results show:")
    print(f"   - 35.2% occipital (too much - should be ~20%)")
    print(f"   - 26.6% frontal (might include face electrodes)")
    print(f"   - 0% temporal (missing completely!)")
    
    # Check if we're including face/neck electrodes
    face_electrodes, scalp_electrodes = analyze_electrode_positions(df)
    
    if len(face_electrodes) > 0:
        print(f"\n❌ PROBLEM: We might be including face/neck electrodes")
        print(f"   This is NOT what university EEG systems do!")
    
    # Show channel naming pattern
    channel_names = df['name'].tolist()
    print(f"\nChannel naming analysis:")
    
    # Count different prefixes
    prefixes = {}
    for name in channel_names:
        # Extract prefix (letters before numbers)
        prefix = ''.join(c for c in name if c.isalpha())
        if prefix:
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    print(f"Channel prefixes found:")
    for prefix, count in sorted(prefixes.items()):
        print(f"   {prefix}: {count} channels")

def create_proper_128_selection(df):
    """Create a proper 128-channel selection that matches university EEG systems."""
    print(f"\n=== Creating Proper 128-Channel Selection ===")
    
    # Filter out face/neck electrodes - focus on scalp only
    scalp_df = df[(df['y'] <= 8) & (df['z'] >= -6)]  # Remove face/neck
    print(f"Scalp electrodes available: {len(scalp_df)}")
    
    # Define regions more carefully for scalp-only electrodes
    regions = {
        'frontal': [],
        'central': [], 
        'parietal': [],
        'occipital': [],
        'temporal': []
    }
    
    for _, row in scalp_df.iterrows():
        name = row['name']
        x, y, z = row['x'], row['y'], row['z']
        
        # Better region classification for scalp electrodes
        if y > 2 and abs(x) < 6:  # Frontal: anterior but not too lateral
            regions['frontal'].append(name)
        elif abs(y) <= 2:  # Central: around the central line
            regions['central'].append(name)
        elif y < -2 and y > -6:  # Parietal: posterior but not too posterior
            regions['parietal'].append(name)
        elif y <= -6:  # Occipital: very posterior
            regions['occipital'].append(name)
        elif abs(x) >= 6:  # Temporal: very lateral
            regions['temporal'].append(name)
        else:
            regions['central'].append(name)  # Default to central
    
    print(f"\nScalp electrode distribution:")
    for region, channels in regions.items():
        print(f"   {region:>10}: {len(channels):>3} available")
    
    # Select channels to match typical university EEG distribution
    target_distribution = {
        'frontal': 20,    # Reduced to avoid artifacts
        'central': 35,    # Increased for motor/sensory
        'parietal': 30,   # Good for cognitive tasks
        'occipital': 25,  # Visual processing
        'temporal': 18    # Language/auditory
    }
    
    selected_channels = []
    for region, target_count in target_distribution.items():
        available = regions[region]
        if len(available) >= target_count:
            # Select evenly spaced channels from this region
            step = len(available) // target_count
            selected = available[::step][:target_count]
        else:
            # Take all available if not enough
            selected = available
        
        selected_channels.extend(selected)
        print(f"   {region:>10}: selected {len(selected):>2}/{target_count:>2}")
    
    print(f"\nTotal selected: {len(selected_channels)}/128")
    
    if len(selected_channels) < 128:
        # Fill remaining slots from central region
        remaining_central = [ch for ch in regions['central'] if ch not in selected_channels]
        needed = 128 - len(selected_channels)
        selected_channels.extend(remaining_central[:needed])
        print(f"Added {min(needed, len(remaining_central))} more from central region")
    
    return selected_channels[:128]

def main():
    """Analyze electrode layout and create proper selection."""
    print("🔬 Analyzing Electrode Layout vs University EEG Systems")
    print("=" * 60)
    
    df = load_electrode_data()
    
    # Analyze the positions
    face_electrodes, scalp_electrodes = analyze_electrode_positions(df)
    
    # Compare with standard
    typical_128 = compare_with_standard_128()
    
    # Analyze our current approach
    analyze_our_selection_strategy()
    
    # Create a proper selection
    proper_selection = create_proper_128_selection(df)
    
    print(f"\n🎯 RECOMMENDATION:")
    print(f"   Current selection includes face/neck electrodes (not typical)")
    print(f"   University EEG systems focus on scalp-only electrodes")
    print(f"   We should modify our selection strategy to:")
    print(f"   1. Exclude face/neck electrodes (Y > 8 or Z < -6)")
    print(f"   2. Focus on brain activity regions")
    print(f"   3. Include temporal electrodes (currently missing)")
    print(f"   4. Reduce occipital over-representation")

if __name__ == "__main__":
    main()