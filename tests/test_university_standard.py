#!/usr/bin/env python3
"""
Test the university_standard channel selection strategy.
"""

import sys
sys.path.append('..')
from data.channel_selection import ChannelSelector
import pandas as pd
from pathlib import Path

def test_university_standard():
    """Test the new university standard selection."""
    print("🎓 Testing University Standard Channel Selection")
    print("=" * 50)
    
    # Load electrode positions
    electrodes_files = list(Path("data/raw").glob("**/*electrodes.tsv"))
    electrodes_file = electrodes_files[0]
    df = pd.read_csv(electrodes_file, sep='\t')
    
    positions = {}
    for _, row in df.iterrows():
        name = row['name']
        x, y, z = row['x'], row['y'], row['z']
        positions[name] = (x, y, z)
    
    all_channels = list(positions.keys())
    
    # Test university standard selection
    selector = ChannelSelector("university_standard")
    selected = selector.select_channels(all_channels, positions)
    
    print(f"\nSelected {len(selected)} channels")
    print(f"First 20: {selected[:20]}")
    
    # Analyze the distribution
    regions = {
        'frontal': [],
        'central': [], 
        'parietal': [],
        'occipital': [],
        'temporal': []
    }
    
    for channel in selected:
        if channel not in positions:
            continue
        x, y, z = positions[channel]
        
        # Same classification as in the algorithm
        if 1 < y <= 6 and abs(x) < 5:
            regions['frontal'].append(channel)
        elif -1 <= y <= 1:
            regions['central'].append(channel)
        elif -5 < y < -1 and abs(x) < 6:
            regions['parietal'].append(channel) 
        elif y <= -5:
            regions['occipital'].append(channel)
        elif abs(x) >= 5:
            regions['temporal'].append(channel)
        else:
            if abs(x) > abs(y):
                regions['temporal'].append(channel)
            else:
                regions['central'].append(channel)
    
    print(f"\nRegional distribution:")
    total = len(selected)
    for region, channels in regions.items():
        count = len(channels)
        percentage = (count / total) * 100
        print(f"   {region:>10}: {count:>3} channels ({percentage:>5.1f}%)")
        if region == 'temporal' and count > 0:
            print(f"              {channels[:10]}")  # Show temporal channels
    
    # Check for face/neck electrodes
    face_count = 0
    for channel in selected:
        if channel in positions:
            x, y, z = positions[channel]
            if y > 6 or z < -4:  # Face/neck criteria
                face_count += 1
    
    print(f"\nValidation:")
    print(f"   Face/neck electrodes: {face_count}")
    print(f"   Temporal coverage: {'✅' if len(regions['temporal']) > 0 else '❌'}")
    print(f"   Total channels: {len(selected)}")

if __name__ == "__main__":
    test_university_standard()