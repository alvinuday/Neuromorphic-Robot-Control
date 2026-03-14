#!/usr/bin/env python3
"""
Quick check: Load one sample from dataset and print all keys
This avoids torch import issues by using pandas directly
"""

from pathlib import Path
import json

# Check if the dataset has metadata
cache_dir = Path("data/cache/lerobot/utokyo_xarm_pick_and_place")

if cache_dir.exists():
    print(f"Dataset cache found at: {cache_dir}")
    
    # Check for metadata files
    metadata_files = list(cache_dir.glob("*.json"))
    print(f"\nMetadata files found:")
    for mf in metadata_files:
        print(f"  - {mf.name}")
        if "features" in mf.name.lower() or "meta" in mf.name.lower():
            try:
                with open(mf) as f:
                    data = json.load(f)
                    print(f"\n    Content: {json.dumps(data, indent=2)[:500]}...")
            except Exception as e:
                print(f"    Could not read: {e}")
    
    # List all files in directory
    print(f"\nAll files in cache directory:")
    all_files = list(cache_dir.iterdir())[:20]
    for f in all_files:
        print(f"  - {f.name}")
        
else:
    print(f"❌ Dataset cache not found at: {cache_dir}")
    print(f"   Checking alternate locations...")
    
    # Try to find any lerobot cache
    potential_paths = [
        Path("data/cache"),
        Path(".cache/lerobot"),
        Path("~/.cache/lerobot").expanduser(),
    ]
    
    for path in potential_paths:
        if path.exists():
            print(f"\n   Found cache at: {path}")
            dirs = list(path.glob("*/"))[:5]
            for d in dirs:
                print(f"     - {d.name}/")
