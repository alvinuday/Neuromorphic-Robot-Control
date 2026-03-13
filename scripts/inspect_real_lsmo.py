#!/usr/bin/env python3
"""
Real LSMO Dataset Inspector
===========================

Load real LSMO dataset, inspect structure (language, RGB inputs),
and prepare for SmolVLA + MPC integration testing.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("="*80)
print("REAL LSMO DATASET INSPECTOR")
print("="*80)

# ============================================================================
# PHASE 1: Load real LSMO dataset
# ============================================================================

print("\n[PHASE 1] Loading real LSMO dataset from TensorFlow Datasets...")

try:
    import tensorflow_datasets as tfds
    
    # Try to load the real dataset
    print("   Attempting to load: tokyo_u_lsmo_converted_externally_to_rlds")
    
    dataset = tfds.load(
        'tokyo_u_lsmo_converted_externally_to_rlds',
        split='train',
        data_dir=None,  # Uses default TensorFlow cache
        download=True,  # Download if not cached
        as_supervised=False
    )
    
    print("✅ Real LSMO dataset loaded!")
    
    # Get dataset info
    dataset_builder = tfds.builder('tokyo_u_lsmo_converted_externally_to_rlds')
    info = dataset_builder.info
    
    print(f"\n📊 Dataset Information:")
    print(f"   Name: {info.name}")
    print(f"   Description: {info.description[:100]}...")
    print(f"   Features: {list(info.features.keys())}")
    print(f"   Supervised Keys: {info.supervised_keys}")
    
except Exception as e:
    print(f"⚠️  Could not load from TFDS: {type(e).__name__}: {str(e)[:150]}")
    dataset = None

# ============================================================================
# PHASE 2: Inspect real episodes
# ============================================================================

print("\n[PHASE 2] Inspecting real LSMO episodes...")

episodes_data = []
episode_count = 0

if dataset is not None:
    print("   Loading episodes...")
    
    for episode_idx, episode in enumerate(dataset.take(5)):  # First 5 episodes
        print(f"\n   ── Episode {episode_idx + 1} ──")
        
        episode_info = {
            'episode_id': episode_idx,
            'keys': list(episode.keys()) if isinstance(episode, dict) else [],
        }
        
        # Inspect structure
        if isinstance(episode, dict):
            for key, value in episode.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in value.items():
                        if hasattr(v, 'shape'):
                            print(f"      {k}: shape={v.shape}, dtype={v.dtype}")
                        else:
                            print(f"      {k}: {type(v).__name__}")
                        
                        if k not in episode_info:
                            episode_info[k] = {}
                        if hasattr(v, 'shape'):
                            episode_info[k] = {
                                'shape': str(v.shape),
                                'dtype': str(v.dtype)
                            }
                elif hasattr(value, 'shape'):
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    episode_info[key] = {
                        'shape': str(value.shape),
                        'dtype': str(value.dtype)
                    }
                else:
                    print(f"   {key}: {type(value).__name__}")
        
        episodes_data.append(episode_info)
        episode_count += 1
    
    print(f"\n✅ Loaded {episode_count} real episodes")

# ============================================================================
# PHASE 3: Examine data modalities
# ============================================================================

print("\n[PHASE 3] Examining data modalities...")

modalities = {
    'observations': False,
    'rgb_images': False,
    'language_instructions': False,
    'actions': False,
    'trajectories': False,
}

episode_sample = None

if dataset is not None:
    for episode in dataset.take(1):
        episode_sample = episode
        
        if isinstance(episode, dict):
            keys_lower = [k.lower() for k in episode.keys()]
            
            # Check for images
            if any('image' in k or 'rgb' in k or 'observation' in k for k in keys_lower):
                modalities['rgb_images'] = True
                print("   ✅ RGB/Image observations found")
            
            # Check for language
            if any('text' in k or 'language' in k or 'instruction' in k for k in keys_lower):
                modalities['language_instructions'] = True
                print("   ✅ Language instructions found")
            
            # Check for actions
            if any('action' in k or 'control' in k for k in keys_lower):
                modalities['actions'] = True
                print("   ✅ Action data found")
            
            # Check for trajectory/steps
            if any('step' in k or 'trajectory' in k or 'episode' in k for k in keys_lower):
                modalities['trajectories'] = True
                print("   ✅ Trajectory/step data found")
            
            # List all keys
            print(f"\n   Available keys: {list(episode.keys())}")

# ============================================================================
# PHASE 4: Sample inspection
# ============================================================================

print("\n[PHASE 4] Detailed sample inspection...")

if episode_sample is not None:
    if isinstance(episode_sample, dict):
        for key, value in episode_sample.items():
            if isinstance(value, dict):
                print(f"\n   {key}:")
                for k, v in list(value.items())[:3]:  # First 3 items
                    if hasattr(v, 'shape'):
                        print(f"      {k}: {v.shape} {v.dtype}")
                        if 'image' in k.lower() or 'rgb' in k.lower():
                            print(f"         └─ First pixel: {v[0, 0, 0, :] if len(v.shape) > 2 else v[0, 0]}")
                    else:
                        val_str = str(v)[:50]
                        print(f"      {k}: {val_str}")
            else:
                if hasattr(value, 'shape'):
                    print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
                    print(f"      First value: {value[0] if hasattr(value, '__getitem__') else value}")
                else:
                    print(f"   {key}: {type(value).__name__}")

# ============================================================================
# PHASE 5: Save dataset schema
# ============================================================================

print("\n[PHASE 5] Saving dataset schema...")

schema = {
    'dataset_name': 'tokyo_u_lsmo_converted_externally_to_rlds',
    'episodes_inspected': episode_count,
    'modalities': modalities,
    'episodes': episodes_data,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}

results_dir = Path('results/lsmo_dataset_inspection')
results_dir.mkdir(parents=True, exist_ok=True)

with open(results_dir / 'lsmo_dataset_schema.json', 'w') as f:
    json.dump(schema, f, indent=2)

print(f"✅ Saved: results/lsmo_dataset_inspection/lsmo_dataset_schema.json")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "="*80)
print("DATASET STRUCTURE SUMMARY")
print("="*80)
print(f"""
✅ Real LSMO Dataset Loaded
   Episodes Inspected: {episode_count}
   
Modalities Present:
   └─ RGB Images:             {'✅ YES' if modalities['rgb_images'] else '❌ NO'}
   └─ Language Instructions:  {'✅ YES' if modalities['language_instructions'] else '❌ NO'}
   └─ Actions/Controls:       {'✅ YES' if modalities['actions'] else '❌ NO'}
   └─ Trajectory Data:        {'✅ YES' if modalities['trajectories'] else '❌ NO'}

Next Steps:
   1. For each episode: Extract RGB frames
   2. Query SmolVLA with: image + language instruction
   3. Run SL MPC solver with: state/action from LSMO
   4. Benchmark performance on real trajectories
   5. Visualize results

Schema saved to: results/lsmo_dataset_inspection/lsmo_dataset_schema.json
""")
print("="*80)
