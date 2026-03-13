#!/usr/bin/env python3
"""Download REAL robot manipulation data from OpenX-Embodiment"""

import json
import time
import numpy as np
from pathlib import Path

print("="*80)
print("LOADING REAL ROBOT DATA FROM OPENX-EMBODIMENT")
print("="*80)

# Create output directory
output_dir = Path('data/real_robot_datasets')
output_dir.mkdir(parents=True, exist_ok=True)

# Try to import datasets
try:
    from datasets import load_dataset
    print("\n✓ datasets library available")
except ImportError:
    print("\nInstalling datasets library...")
    import subprocess
    subprocess.run(["pip", "install", "datasets", "-q"], check=False)
    from datasets import load_dataset

print("\n" + "="*80)
print("Connecting to OpenX-Embodiment (150K real robot episodes)...")
print("="*80 + "\n")

success = False

# Try OpenX-Embodiment
try:
    print("Streaming OpenX-Embodiment dataset...")
    
    ds = load_dataset(
        "openx-embodiment",
        split="train",
        streaming=True,
        trust_remote_code=True,
        cache_dir="/tmp/openx_cache"
    )
    
    print(f"✓ Successfully connected to OpenX-Embodiment!")
    print(f"  Loading real robot episodes...")
    
    # Collect real episodes
    real_episodes = []
    
    for i, example in enumerate(ds.take(10)):  # Take first 10 real episodes
        try:
            # Extract episode info
            ep_info = {
                'episode_id': i,
                'dataset_name': example.get('dataset_name', 'unknown'),
                'robot_type': example.get('robot_type', 'unknown'),
                'keys': list(example.keys()),
            }
            
            # Check for data
            if 'steps' in example:
                steps = example['steps']
                ep_info['num_steps'] = len(steps) if hasattr(steps, '__len__') else '?'
            
            if 'observation' in example:
                obs = example['observation']
                if hasattr(obs, 'keys'):
                    ep_info['observation_keys'] = list(obs.keys())
            
            real_episodes.append(ep_info)
            
            robot = ep_info.get('robot_type', '?')
            dataset = ep_info.get('dataset_name', '?')
            
            print(f"  [{i+1}] {robot:15} - {dataset:20} ({ep_info.get('num_steps', '?')} steps)")
            
        except Exception as e:
            print(f"  [{i+1}] Error extracting: {str(e)[:50]}")
    
    # Save real data metadata
    metadata = {
        'source': 'OpenX-Embodiment (Google)',
        'real_data': True,
        'url': 'https://huggingface.co/datasets/openx-embodiment',
        'description': '150K+ real robot manipulation episodes from multiple labs',
        'episodes_loaded': len(real_episodes),
        'episodes': real_episodes,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(output_dir / 'openx_real_data.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Saved {len(real_episodes)} real episodes to {output_dir / 'openx_real_data.json'}")
    success = True
    
except Exception as e:
    print(f"✗ OpenX-Embodiment failed: {str(e)[:150]}")
    print("\nTrying Bridge Dataset (UC Berkeley)...")

if not success:
    # Try BRIDGE as fallback
    try:
        print("\nConnecting to BRIDGE dataset...")
        
        ds = load_dataset(
            "rail-berkeley/bridge_dataset",
            split="train",
            streaming=True,
            trust_remote_code=True,
            cache_dir="/tmp/bridge_cache"
        )
        
        print(f"✓ Successfully connected to BRIDGE (UR5)!")
        print(f"  Loading real UR5 manipulation episodes...")
        
        real_episodes = []
        
        for i, example in enumerate(ds.take(10)):
            try:
                ep_info = {
                    'episode_id': i,
                    'robot': 'UR5',
                    'keys': list(example.keys()),
                }
                
                if 'trajectory' in example:
                    ep_info['has_trajectory'] = True
                if 'language_instruction' in example:
                    ep_info['instruction'] = example.get('language_instruction', '')[:100]
                
                real_episodes.append(ep_info)
                print(f"  [{i+1}] Real UR5 episode - {ep_info.get('instruction', 'n/a')[:60]}")
                
            except Exception as e:
                print(f"  [{i+1}] Error: {str(e)[:50]}")
        
        metadata = {
            'source': 'BRIDGE - UC Berkeley RAIL Lab',
            'real_data': True,
            'robot': 'UR5',
            'url': 'https://huggingface.co/datasets/rail-berkeley/bridge_dataset',
            'episodes_loaded': len(real_episodes),
            'episodes': real_episodes,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        with open(output_dir / 'bridge_real_data.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Saved {len(real_episodes)} real BRIDGE episodes")
        success = True
        
    except Exception as e:
        print(f"✗ BRIDGE failed: {str(e)[:150]}")

if not success:
    print("\n⚠ Could not connect to any real datasets")
    print("\nAvailable datasets to try manually:")
    print("  1. OpenX-Embodiment: https://huggingface.co/datasets/openx-embodiment")
    print("  2. Bridge: https://huggingface.co/datasets/rail-berkeley/bridge_dataset")
    print("  3. RT-1: https://huggingface.co/datasets/google/rt-1")
    print("  4. RoboNet: https://huggingface.co/datasets/robonet/robonet")

print("\n" + "="*80)
print("REAL DATASET CHECK COMPLETE")
print("="*80)

# List what we have
if success:
    files = list(output_dir.glob('*.json'))
    if files:
        print(f"\nReal data files created:")
        for f in files:
            size = f.stat().st_size
            print(f"  ✓ {f.name} ({size} bytes)")
