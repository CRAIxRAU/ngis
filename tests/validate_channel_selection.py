#!/usr/bin/env python3
"""
Validate channel selection spatial accuracy and position mapping.

This script verifies:
1. Selected channels have correct spatial positions
2. Good spatial coverage across scalp regions
3. Channel names match between selection and position data
4. Visual verification of selected electrode positions
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

import sys
sys.path.append('..')
from data.channel_selection import ChannelSelector


def load_real_electrode_positions():
    """Load electrode positions from the actual dataset."""
    # Find any electrodes.tsv file from the downloaded data
    electrodes_files = list(Path("data/raw").glob("**/*electrodes.tsv"))
    
    if not electrodes_files:
        raise FileNotFoundError("No electrodes.tsv files found in downloaded data")
    
    electrodes_file = electrodes_files[0]
    print(f"Loading electrode positions from: {electrodes_file}")
    
    df = pd.read_csv(electrodes_file, sep='\t')
    
    positions = {}
    for _, row in df.iterrows():
        name = row['name']
        x, y, z = row['x'], row['y'], row['z']
        positions[name] = (x, y, z)
    
    print(f"✅ Loaded {len(positions)} electrode positions")
    return positions, df


def validate_channel_selection_coverage(positions, selected_channels):
    """Validate spatial coverage of selected channels."""
    print(f"\n=== Validating Spatial Coverage ===")
    
    # Get positions for selected channels
    selected_positions = []
    missing_positions = []
    
    for ch in selected_channels:
        if ch in positions:
            selected_positions.append(positions[ch])
        else:
            missing_positions.append(ch)
    
    if missing_positions:
        print(f"⚠️  {len(missing_positions)} selected channels have no position data: {missing_positions[:10]}")
    
    selected_positions = np.array(selected_positions)
    all_positions = np.array(list(positions.values()))
    
    print(f"✅ Position data available for {len(selected_positions)}/{len(selected_channels)} selected channels")
    
    # Analyze spatial coverage
    if len(selected_positions) > 0:
        # Calculate coverage statistics
        selected_x_range = selected_positions[:, 0].max() - selected_positions[:, 0].min()
        selected_y_range = selected_positions[:, 1].max() - selected_positions[:, 1].min()
        selected_z_range = selected_positions[:, 2].max() - selected_positions[:, 2].min()
        
        all_x_range = all_positions[:, 0].max() - all_positions[:, 0].min()
        all_y_range = all_positions[:, 1].max() - all_positions[:, 1].min()
        all_z_range = all_positions[:, 2].max() - all_positions[:, 2].min()
        
        x_coverage = selected_x_range / all_x_range
        y_coverage = selected_y_range / all_y_range
        z_coverage = selected_z_range / all_z_range
        
        print(f"   X-axis coverage: {x_coverage:.1%} ({selected_x_range:.1f}/{all_x_range:.1f})")
        print(f"   Y-axis coverage: {y_coverage:.1%} ({selected_y_range:.1f}/{all_y_range:.1f})")
        print(f"   Z-axis coverage: {z_coverage:.1%} ({selected_z_range:.1f}/{all_z_range:.1f})")
        
        # Check if coverage is reasonable (should be > 80% for good spatial sampling)
        avg_coverage = (x_coverage + y_coverage + z_coverage) / 3
        if avg_coverage > 0.8:
            print(f"✅ Good spatial coverage: {avg_coverage:.1%}")
        else:
            print(f"⚠️  Limited spatial coverage: {avg_coverage:.1%}")
    
    return selected_positions, all_positions


def analyze_channel_distribution(positions, selected_channels):
    """Analyze distribution of selected channels across brain regions."""
    print(f"\n=== Analyzing Regional Distribution ===")
    
    # Define regions based on electrode positions
    regions = {
        'frontal': [],
        'central': [], 
        'parietal': [],
        'occipital': [],
        'temporal': [],
        'unknown': []
    }
    
    for ch in selected_channels:
        if ch not in positions:
            regions['unknown'].append(ch)
            continue
            
        x, y, z = positions[ch]
        
        # Simple region classification based on y-coordinate (anterior-posterior)
        if y > 4:  # Anterior
            regions['frontal'].append(ch)
        elif y > 0:  # Central
            regions['central'].append(ch)
        elif y > -4:  # Posterior
            regions['parietal'].append(ch)
        elif y > -7:  # Very posterior
            regions['occipital'].append(ch)
        else:
            # Use x-coordinate for temporal classification
            if abs(x) > 6:
                regions['temporal'].append(ch)
            else:
                regions['occipital'].append(ch)
    
    print("Regional distribution:")
    total_selected = len(selected_channels)
    for region, channels in regions.items():
        count = len(channels)
        percentage = (count / total_selected) * 100
        print(f"   {region:>10}: {count:>3} channels ({percentage:>5.1f}%)")
        if count > 0 and count <= 5:  # Show channel names for small regions
            print(f"              {channels}")
    
    return regions


def visualize_channel_selection(all_positions, selected_positions, selected_channels, strategy):
    """Create 3D visualization of selected vs all channels."""
    print(f"\n=== Creating Visualization ===")
    
    fig = plt.figure(figsize=(15, 5))
    
    # Top view (X-Y plane)
    ax1 = fig.add_subplot(131)
    ax1.scatter(all_positions[:, 0], all_positions[:, 1], 
               c='lightgray', alpha=0.6, s=20, label='All channels')
    ax1.scatter(selected_positions[:, 0], selected_positions[:, 1], 
               c='red', alpha=0.8, s=40, label='Selected channels')
    ax1.set_xlabel('X (Left-Right)')
    ax1.set_ylabel('Y (Anterior-Posterior)')
    ax1.set_title(f'Top View - {strategy}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Side view (Y-Z plane)
    ax2 = fig.add_subplot(132)
    ax2.scatter(all_positions[:, 1], all_positions[:, 2], 
               c='lightgray', alpha=0.6, s=20, label='All channels')
    ax2.scatter(selected_positions[:, 1], selected_positions[:, 2], 
               c='red', alpha=0.8, s=40, label='Selected channels')
    ax2.set_xlabel('Y (Anterior-Posterior)')
    ax2.set_ylabel('Z (Inferior-Superior)')
    ax2.set_title(f'Side View - {strategy}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Front view (X-Z plane)
    ax3 = fig.add_subplot(133)
    ax3.scatter(all_positions[:, 0], all_positions[:, 2], 
               c='lightgray', alpha=0.6, s=20, label='All channels')
    ax3.scatter(selected_positions[:, 0], selected_positions[:, 2], 
               c='red', alpha=0.8, s=40, label='Selected channels')
    ax3.set_xlabel('X (Left-Right)')
    ax3.set_ylabel('Z (Inferior-Superior)')
    ax3.set_title(f'Front View - {strategy}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    plt.tight_layout()
    
    # Save the plot
    output_file = f"channel_selection_{strategy}_validation.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Saved visualization: {output_file}")
    
    return fig


def validate_channel_name_consistency(electrodes_df, selected_channels):
    """Validate that selected channel names exist in the electrode data."""
    print(f"\n=== Validating Channel Name Consistency ===")
    
    available_channels = set(electrodes_df['name'].tolist())
    selected_set = set(selected_channels)
    
    # Check for exact matches
    exact_matches = selected_set.intersection(available_channels)
    missing_channels = selected_set - available_channels
    
    print(f"✅ Exact matches: {len(exact_matches)}/{len(selected_channels)} channels")
    
    if missing_channels:
        print(f"⚠️  Missing channels: {len(missing_channels)}")
        print(f"   Examples: {list(missing_channels)[:10]}")
        
        # Try to find similar channel names
        print(f"\n   Available channel name patterns:")
        available_list = sorted(available_channels)
        print(f"   First 20: {available_list[:20]}")
        print(f"   Last 20: {available_list[-20:]}")
    
    return len(exact_matches), len(missing_channels)


def main():
    """Run comprehensive channel selection validation."""
    print("🔍 Validating Channel Selection Accuracy")
    print("=" * 50)
    
    try:
        # Load real electrode positions
        positions, electrodes_df = load_real_electrode_positions()
        
        # Test different selection strategies
        strategies = ["uniform_spatial", "standard_hd", "roi_based"]
        all_channels = list(positions.keys())  # Use real channel names
        
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*20} Testing {strategy} {'='*20}")
            
            # Create selector and select channels
            selector = ChannelSelector(strategy)
            selected_channels = selector.select_channels(
                all_channels, positions, None
            )
            
            print(f"Selected {len(selected_channels)} channels using {strategy}")
            
            # Validate channel name consistency
            exact_matches, missing_channels = validate_channel_name_consistency(
                electrodes_df, selected_channels
            )
            
            # Validate spatial coverage
            selected_positions, all_positions = validate_channel_selection_coverage(
                positions, selected_channels
            )
            
            # Analyze regional distribution
            regions = analyze_channel_distribution(positions, selected_channels)
            
            # Create visualization
            if len(selected_positions) > 0:
                fig = visualize_channel_selection(
                    all_positions, selected_positions, selected_channels, strategy
                )
                plt.close(fig)  # Close to save memory
            
            # Store results
            results[strategy] = {
                'selected_channels': selected_channels,
                'exact_matches': exact_matches,
                'missing_channels': missing_channels,
                'regions': regions,
                'spatial_coverage': len(selected_positions) / len(selected_channels) if selected_channels else 0
            }
        
        # Summary
        print(f"\n{'='*50}")
        print("🏁 Validation Summary:")
        print(f"{'='*50}")
        
        for strategy, result in results.items():
            coverage = result['spatial_coverage']
            matches = result['exact_matches']
            total = len(result['selected_channels'])
            
            print(f"{strategy:>15}: {matches}/{total} channels with positions ({coverage:.1%} coverage)")
            
            # Check if validation passed
            if matches >= 120 and coverage >= 0.9:  # Allow some tolerance
                print(f"               ✅ PASSED - Good spatial coverage and channel matching")
            elif matches >= 100 and coverage >= 0.8:
                print(f"               ⚠️  PARTIAL - Acceptable but could be improved")
            else:
                print(f"               ❌ FAILED - Poor spatial coverage or channel matching")
        
        print(f"\n📊 Visualization files saved for manual inspection")
        print(f"🎯 Next steps: Review the generated .png files to visually verify spatial distribution")
        
        return True
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)