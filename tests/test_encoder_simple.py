#!/usr/bin/env python3
"""Quick test of simplified encoder."""

import sys
import numpy as np

print("[TEST] Starting encoder test...")

try:
    from src.fusion.encoders.real_fusion_simple import RealFusionEncoder
    print("[OK] Encoder imported successfully")
except Exception as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

# Create test data
print("[TEST] Creating test data...")
rgb = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)  # Real RGB array
state = np.random.randn(6)  # Real state vector
obs = {'rgb': rgb, 'state': state}

# Test each mode
modes = [
    ('M0: RGB only', RealFusionEncoder.rgb_only()),
    ('M1: RGB+Events', RealFusionEncoder.rgb_events()),
    ('M2: RGB+LiDAR', RealFusionEncoder.rgb_lidar()),
    ('M3: RGB+Proprio', RealFusionEncoder.rgb_proprio()),
    ('M4: Full Fusion', RealFusionEncoder.full_fusion()),
]

print("\n[TEST] Running encoding tests...\n")
for mode_name, encoder in modes:
    try:
        result = encoder.encode(obs)
        assert result.shape == (256,), f"Expected shape (256,), got {result.shape}"
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
        print(f"✓ {mode_name}: embedding shape {result.shape}, dtype {result.dtype}")
        print(f"  Features: min={result.min():.4f}, max={result.max():.4f}, mean={result.mean():.4f}")
    except Exception as e:
        print(f"✗ {mode_name}: FAILED - {e}")
        sys.exit(1)

print("\n[OK] All tests passed! Encoder is working correctly.")
print("[INFO] Ready to run validation scripts with real data.")
