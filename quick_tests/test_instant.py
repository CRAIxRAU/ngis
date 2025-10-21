#!/usr/bin/env python3
"""INSTANT TEST - Just check configuration"""
import sys
sys.path.insert(0, '.')

from utils.config import Config

config = Config.from_yaml('configs/ds003766.yaml')

print("Configuration Check:")
print(f"  Data channels: {config.data.channels}")
print(f"  Model channels: {config.model.n_channels}")
print(f"  Model neurons: {config.model.n_neurons}")

if config.data.channels == config.model.n_channels == 128:
    print("\n[SUCCESS] All set to 128 channels!")
    sys.exit(0)
else:
    print(f"\n[ERROR] Mismatch!")
    sys.exit(1)
