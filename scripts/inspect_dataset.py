#!/usr/bin/env python3
"""Inspect actual lerobot dataset structure"""

import sys
sys.path.insert(0, 'src')

try:
    import lerobot
    print(f"✓ LeRobot module found")
except ImportError as e:
    print(f"✗ LeRobot not found: {e}")
    sys.exit(1)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    print("\n📦 Loading lerobot/utokyo_xarm_pick_and_place...")
    ds = LeRobotDataset(
        "lerobot/utokyo_xarm_pick_and_place",
        root="data/cache"
    )
    
    print(f"✓ Dataset loaded!")
    print(f"  Episodes: {ds.num_episodes}")
    print(f"  Total frames: {ds.num_frames}")
    
    # Get first sample
    sample = ds[0]
    all_keys = sorted(sample.keys())
    
    print(f"\n📋 All keys in first sample:")
    for key in all_keys:
        value = sample[key]
        if hasattr(value, 'shape'):
            print(f"   {key:45s} shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"   {key:45s} type={type(value).__name__}")
    
    # Identify critical keys
    print(f"\n🖼️  Image keys:")
    for key in all_keys:
        if 'image' in key.lower() or 'rgb' in key.lower():
            value = sample[key]
            if hasattr(value, 'shape'):
                print(f"   {key:45s} shape={value.shape}")
    
    print(f"\n🎮 State/Action keys:")
    for key in all_keys:
        if any(x in key.lower() for x in ['state', 'action', 'joint', 'observation.pos', 'observation.vel']):
            value = sample[key]
            if hasattr(value, 'shape'):
                print(f"   {key:45s} shape={value.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
