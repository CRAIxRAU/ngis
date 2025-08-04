"""
Channel selection algorithms for 256→128 channel EEG data reduction.

Implements multiple strategies for intelligent channel subsampling while
maintaining good spatial coverage across the scalp.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ChannelSelector:
    """
    Select 128 channels from 256-channel EEG data using various strategies.
    
    Supports multiple selection strategies:
    - uniform_spatial: Every 2nd electrode for uniform coverage
    - standard_hd: 64 standard 10-20 + 64 high-density positions
    - roi_based: Balanced coverage across brain regions
    - custom: User-defined channel list
    """
    
    def __init__(self, strategy: str = "uniform_spatial"):
        """
        Initialize channel selector.
        
        Args:
            strategy: Channel selection strategy to use
            
        Raises:
            ValueError: If strategy is not supported
        """
        self.strategy = strategy
        self.supported_strategies = [
            "uniform_spatial", "standard_hd", "roi_based", "university_standard", "custom"
        ]
        
        if strategy not in self.supported_strategies:
            raise ValueError(f"Unsupported strategy: {strategy}. "
                           f"Available: {self.supported_strategies}")
        
        logger.info(f"Initialized ChannelSelector with strategy: {strategy}")
    
    def select_channels(
        self, 
        all_channels: List[str], 
        electrode_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
        custom_channels: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select 128 channels from 256-channel list.
        
        Args:
            all_channels: List of all available channel names
            electrode_positions: Dictionary of channel positions (x, y, z)
            custom_channels: Custom channel list (only used with custom strategy)
            
        Returns:
            List of 128 selected channel names
            
        Raises:
            ValueError: If input validation fails
        """
        if len(all_channels) < 128:
            raise ValueError(f"Not enough channels for selection: got {len(all_channels)}, need at least 128")
        
        # Handle cases where we have more or fewer than 256 channels
        if len(all_channels) != 256:
            logger.warning(f"Expected 256 channels, got {len(all_channels)}. Adapting selection strategy.")
        
        logger.info(f"Selecting 128 channels from {len(all_channels)} using {self.strategy}")
        
        if self.strategy == "uniform_spatial":
            return self._uniform_spatial_selection(all_channels)
        elif self.strategy == "standard_hd":
            return self._standard_hd_selection(all_channels, electrode_positions)
        elif self.strategy == "roi_based":
            return self._roi_based_selection(all_channels, electrode_positions)
        elif self.strategy == "university_standard":
            return self._university_standard_selection(all_channels, electrode_positions)
        elif self.strategy == "custom":
            return self._custom_selection(all_channels, custom_channels)
        
    def _uniform_spatial_selection(self, all_channels: List[str]) -> List[str]:
        """Select every 2nd electrode for uniform spatial coverage."""
        # Simple stride-based selection
        selected = all_channels[::2]  # Every 2nd channel
        
        if len(selected) != 128:
            # Adjust if needed (should be exactly 128 from 256)
            if len(selected) > 128:
                selected = selected[:128]
            else:
                # Add remaining channels if somehow we have fewer
                remaining = [ch for ch in all_channels if ch not in selected]
                selected.extend(remaining[:128 - len(selected)])
        
        logger.info(f"Selected {len(selected)} channels using uniform spatial strategy")
        return selected
    
    def _standard_hd_selection(
        self, 
        all_channels: List[str], 
        electrode_positions: Optional[Dict[str, Tuple[float, float, float]]]
    ) -> List[str]:
        """
        Select 64 standard 10-20 positions + 64 high-density intermediate positions.
        
        This strategy prioritizes standard EEG positions first, then fills in
        with high-density electrodes for better spatial resolution.
        """
        # Standard 10-20 system electrode patterns (simplified mapping)
        standard_patterns = [
            'Fp', 'F', 'C', 'P', 'O', 'T', 'Fz', 'Cz', 'Pz', 'Oz',
            'FC', 'CP', 'FT', 'TP', 'AF', 'PO'
        ]
        
        selected = []
        
        # First, select channels matching standard 10-20 patterns
        for pattern in standard_patterns:
            matching = [ch for ch in all_channels if ch.startswith(pattern) and ch not in selected]
            # Take a subset from each pattern
            pattern_count = min(len(matching), 8)  # Limit per pattern
            selected.extend(matching[:pattern_count])
            
            if len(selected) >= 64:
                break
        
        # Ensure we have exactly 64 standard channels
        selected = selected[:64]
        
        # Fill remaining 64 slots with other channels (high-density)
        remaining_channels = [ch for ch in all_channels if ch not in selected]
        needed = 128 - len(selected)  # How many more channels we need
        
        if electrode_positions:
            # If we have positions, select based on spatial distribution
            remaining_selected = self._spatial_distribution_selection(
                remaining_channels, electrode_positions, needed
            )
        else:
            # Otherwise, just take the first N remaining
            remaining_selected = remaining_channels[:needed]
        
        selected.extend(remaining_selected)
        
        logger.info(f"Selected {len(selected)} channels: {len(selected)-len(remaining_selected)} standard + {len(remaining_selected)} high-density")
        return selected[:128]  # Ensure exactly 128
    
    def _roi_based_selection(
        self, 
        all_channels: List[str], 
        electrode_positions: Optional[Dict[str, Tuple[float, float, float]]]
    ) -> List[str]:
        """
        Select channels to match typical university 128-channel EEG systems.
        
        Focuses on scalp electrodes only, excludes face/neck electrodes.
        Distribution matches standard EEG systems: 20% frontal, 27% central, 
        23% parietal, 20% occipital, 10% temporal.
        """
        if not electrode_positions:
            logger.warning("No electrode positions provided, falling back to uniform selection")
            return self._uniform_spatial_selection(all_channels)
        
        # First, filter out face/neck electrodes - focus on scalp only
        scalp_channels = []
        for channel in all_channels:
            if channel not in electrode_positions:
                continue
            x, y, z = electrode_positions[channel]
            # Exclude face/neck electrodes (very anterior or very inferior)
            if y <= 8 and z >= -6:  # Scalp only
                scalp_channels.append(channel)
        
        logger.info(f"Filtered to {len(scalp_channels)} scalp electrodes (excluding face/neck)")
        
        # Define brain regions based on electrode positions (scalp only)
        regions = {
            'frontal': [],
            'central': [], 
            'parietal': [],
            'occipital': [],
            'temporal': []
        }
        
        # Classify scalp channels into regions
        for channel in scalp_channels:
            x, y, z = electrode_positions[channel]
            
            # Better region classification for scalp electrodes
            if y > 2 and abs(x) < 6:  # Frontal: anterior but not too lateral
                regions['frontal'].append(channel)
            elif abs(y) <= 2:  # Central: around the central line
                regions['central'].append(channel)
            elif y < -2 and y > -6:  # Parietal: posterior but not too posterior
                regions['parietal'].append(channel)
            elif y <= -6:  # Occipital: very posterior
                regions['occipital'].append(channel)
            elif abs(x) >= 6:  # Temporal: very lateral
                regions['temporal'].append(channel)
            else:
                regions['central'].append(channel)  # Default to central
        
        # Target distribution matching university EEG systems
        target_distribution = {
            'frontal': 25,    # 19.5% - reduced to avoid artifacts
            'central': 35,    # 27.3% - motor/sensory areas
            'parietal': 30,   # 23.4% - cognitive processing
            'occipital': 25,  # 19.5% - visual processing
            'temporal': 13    # 10.2% - language/auditory
        }
        
        selected = []
        for region_name, target_count in target_distribution.items():
            region_channels = regions[region_name]
            
            if not region_channels:
                logger.warning(f"No channels available in {region_name} region")
                continue
                
            if len(region_channels) >= target_count:
                # Select evenly spaced channels from this region
                step = max(1, len(region_channels) // target_count)
                region_selected = region_channels[::step][:target_count]
            else:
                # Take all available if not enough
                region_selected = region_channels
            
            selected.extend(region_selected)
            logger.debug(f"Selected {len(region_selected)}/{target_count} channels from {region_name} region")
        
        # Fill remaining slots from central region if needed
        if len(selected) < 128:
            remaining_central = [ch for ch in regions['central'] if ch not in selected]
            needed = 128 - len(selected)
            selected.extend(remaining_central[:needed])
            logger.info(f"Added {min(needed, len(remaining_central))} additional channels from central region")
        
        logger.info(f"Selected {len(selected)} channels with university-standard ROI distribution")
        return selected[:128]
    
    def _university_standard_selection(
        self, 
        all_channels: List[str], 
        electrode_positions: Optional[Dict[str, Tuple[float, float, float]]]
    ) -> List[str]:
        """
        Select channels exactly like university 128-channel EEG systems.
        
        This is the most realistic selection for research use:
        - Excludes face/neck electrodes completely
        - Focuses on brain activity regions
        - Includes temporal coverage
        - Matches standard EEG cap distributions
        """
        if not electrode_positions:
            logger.warning("No electrode positions provided, falling back to uniform selection")
            return self._uniform_spatial_selection(all_channels)
        
        # More restrictive filtering for university-standard selection
        scalp_channels = []
        for channel in all_channels:
            if channel not in electrode_positions:
                continue
            x, y, z = electrode_positions[channel]
            # Very strict scalp-only criteria (exclude more face/neck electrodes)
            if y <= 6 and z >= -4:  # More conservative scalp boundaries
                scalp_channels.append(channel)
        
        logger.info(f"University-standard: filtered to {len(scalp_channels)} scalp electrodes")
        
        # More precise region classification
        regions = {
            'frontal': [],
            'central': [], 
            'parietal': [],
            'occipital': [],
            'temporal': []
        }
        
        for channel in scalp_channels:
            x, y, z = electrode_positions[channel]
            
            # More precise region boundaries
            if 1 < y <= 6 and abs(x) < 5:  # Frontal: more restricted
                regions['frontal'].append(channel)
            elif -1 <= y <= 1:  # Central: narrow band around y=0
                regions['central'].append(channel)
            elif -5 < y < -1 and abs(x) < 6:  # Parietal: mid-posterior
                regions['parietal'].append(channel)
            elif y <= -5:  # Occipital: far posterior
                regions['occipital'].append(channel)
            elif abs(x) >= 5:  # Temporal: lateral regions
                regions['temporal'].append(channel)
            else:
                # Default assignment based on strongest coordinate
                if abs(x) > abs(y):
                    regions['temporal'].append(channel)
                else:
                    regions['central'].append(channel)
        
        # Log region availability
        for region_name, region_channels in regions.items():
            logger.info(f"University-standard: {len(region_channels)} channels available in {region_name}")
        
        # University-standard target distribution
        target_distribution = {
            'frontal': 25,    # F, FC regions
            'central': 35,    # C, CP regions (motor/sensory)
            'parietal': 30,   # P, CP regions (cognitive)
            'occipital': 25,  # O, PO regions (visual)
            'temporal': 13    # T, TP regions (language/auditory)
        }
        
        selected = []
        for region_name, target_count in target_distribution.items():
            region_channels = regions[region_name]
            
            if len(region_channels) == 0:
                logger.warning(f"No channels available in {region_name} region")
                continue
                
            if len(region_channels) >= target_count:
                # Select evenly distributed channels
                indices = np.linspace(0, len(region_channels)-1, target_count, dtype=int)
                region_selected = [region_channels[i] for i in indices]
            else:
                # Take all available
                region_selected = region_channels
            
            selected.extend(region_selected)
            logger.info(f"University-standard: selected {len(region_selected)}/{target_count} from {region_name}")
        
        # Ensure we have exactly 128 channels
        if len(selected) < 128:
            # Fill from central and parietal (most important for brain activity)
            remaining_channels = [ch for ch in scalp_channels if ch not in selected]
            needed = 128 - len(selected)
            selected.extend(remaining_channels[:needed])
            logger.info(f"University-standard: added {min(needed, len(remaining_channels))} additional channels")
        
        final_selection = selected[:128]
        logger.info(f"University-standard selection complete: {len(final_selection)} channels")
        return final_selection
    
    def _spatial_distribution_selection(
        self,
        channels: List[str],
        electrode_positions: Dict[str, Tuple[float, float, float]],
        n_select: int
    ) -> List[str]:
        """
        Select channels to maximize spatial distribution.
        
        Uses a greedy algorithm to select channels that are maximally
        separated in space.
        """
        if n_select >= len(channels):
            return channels
        
        available_channels = [ch for ch in channels if ch in electrode_positions]
        
        if not available_channels:
            return channels[:n_select]
        
        selected = []
        remaining = available_channels.copy()
        
        # Start with the channel closest to the center
        center_distances = []
        for ch in remaining:
            x, y, z = electrode_positions[ch]
            dist = np.sqrt(x**2 + y**2 + z**2)
            center_distances.append((dist, ch))
        
        # Start with a central channel
        center_distances.sort()
        selected.append(center_distances[0][1])
        remaining.remove(center_distances[0][1])
        
        # Greedily select channels that are farthest from already selected
        while len(selected) < n_select and remaining:
            max_min_distance = -1
            best_channel = None
            
            for candidate in remaining:
                candidate_pos = np.array(electrode_positions[candidate])
                
                # Find minimum distance to any selected channel
                min_distance = float('inf')
                for selected_ch in selected:
                    selected_pos = np.array(electrode_positions[selected_ch])
                    distance = np.linalg.norm(candidate_pos - selected_pos)
                    min_distance = min(min_distance, distance)
                
                # Select the candidate with maximum minimum distance
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_channel = candidate
            
            if best_channel:
                selected.append(best_channel)
                remaining.remove(best_channel)
        
        return selected
    
    def _custom_selection(
        self, 
        all_channels: List[str], 
        custom_channels: Optional[List[str]]
    ) -> List[str]:
        """Use user-provided custom channel list."""
        if not custom_channels:
            raise ValueError("Custom channel list required for custom strategy")
        
        if len(custom_channels) != 128:
            raise ValueError(f"Custom list must contain exactly 128 channels, got {len(custom_channels)}")
        
        # Validate that all custom channels exist
        missing = set(custom_channels) - set(all_channels)
        if missing:
            raise ValueError(f"Custom channels not found in dataset: {missing}")
        
        logger.info(f"Using custom channel selection with {len(custom_channels)} channels")
        return custom_channels
    
    def get_channel_indices(self, all_channels: List[str], selected_channels: List[str]) -> List[int]:
        """
        Get indices of selected channels in the original channel list.
        
        Args:
            all_channels: Complete list of channel names
            selected_channels: Selected channel names
            
        Returns:
            List of indices for selected channels
        """
        indices = []
        for selected_ch in selected_channels:
            try:
                idx = all_channels.index(selected_ch)
                indices.append(idx)
            except ValueError:
                logger.warning(f"Selected channel {selected_ch} not found in channel list")
        
        return indices
    
    def load_electrode_positions(self, electrodes_file: Union[str, Path]) -> Dict[str, Tuple[float, float, float]]:
        """
        Load electrode positions from TSV file.
        
        Args:
            electrodes_file: Path to electrodes.tsv file
            
        Returns:
            Dictionary mapping channel names to (x, y, z) coordinates
        """
        electrodes_file = Path(electrodes_file)
        
        if not electrodes_file.exists():
            raise FileNotFoundError(f"Electrodes file not found: {electrodes_file}")
        
        df = pd.read_csv(electrodes_file, sep='\t')
        
        positions = {}
        for _, row in df.iterrows():
            name = row['name']
            x, y, z = row['x'], row['y'], row['z']
            positions[name] = (x, y, z)
        
        logger.info(f"Loaded positions for {len(positions)} electrodes")
        return positions


def create_channel_map(
    strategy: str = "uniform_spatial",
    all_channels: Optional[List[str]] = None,
    electrode_positions: Optional[Dict[str, Tuple[float, float, float]]] = None,
    custom_channels: Optional[List[str]] = None
) -> Dict[str, Union[List[str], List[int]]]:
    """
    Create a channel mapping for 256→128 selection.
    
    Args:
        strategy: Selection strategy to use
        all_channels: List of all 256 channel names
        electrode_positions: Electrode position dictionary
        custom_channels: Custom channel list (for custom strategy)
        
    Returns:
        Dictionary with 'names' and 'indices' keys
    """
    if all_channels is None:
        # Default 256-channel naming (E1-E256)
        all_channels = [f"E{i}" for i in range(1, 257)]
    
    selector = ChannelSelector(strategy)
    selected_channels = selector.select_channels(
        all_channels, electrode_positions, custom_channels
    )
    selected_indices = selector.get_channel_indices(all_channels, selected_channels)
    
    return {
        'names': selected_channels,
        'indices': selected_indices,
        'strategy': strategy,
        'total_channels': len(all_channels),
        'selected_channels': len(selected_channels)
    }