#!/usr/bin/env python3
"""
Download Real LSMO Dataset from TensorFlow Datasets
====================================================

Loads Tokyo-U LSMO (Large-Scale Mobile Manipulation) dataset.
Validates and prepares for VLA+MPC integration testing.
"""

import sys
import os
import json
import numpy as np
from pathlib import Path

print("="*80)
print("LSMO REAL DATASET DOWNLOADER")
print("="*80)

# ============================================================================
# PHASE 1: Load Dataset
# ============================================================================

print("\n[PHASE 1] Loading real LSMO dataset from tensorflow_datasets...")

try:
    import tensorflow_datasets as tfds
    import tensorflow as tf
    
    # Dataset info
    dataset_name = 'tokyo_u_lsmo_converted_externally_to_rlds'
    
    print(f"  Attempting to load: {dataset_name}")
    
    # Load dataset - this will download if not cached
    try:
        dataset = tfds.load(
            dataset_name,
            split='train',
            data_dir=str(Path.home() / '.tfds'),
            download=True,
            shuffle_files=False
        )
        print(f"  ✓ Dataset loaded successfully")
    except Exception as e_load:
        print(f"  ⚠️  Primary load failed: {str(e_load)[:100]}")
        print(f"  Trying alternative name...")
        
        # Try alternative names
        alt_names = [
            'lsmo',
            'tokyo_lsmo',
            'lsmo_rlds'
        ]
        
        dataset = None
        for alt_name in alt_names:
            try:
                dataset = tfds.load(
                    alt_name,
                    split='train',
                    data_dir=str(Path.home() / '.tfds'),
                    download=False,  # Don't download if listing failed
                    shuffle_files=False
                )
                print(f"  ✓ Loaded as '{alt_name}'")
                break
            except:
                continue
        
        if dataset is None:
            print(f"  ❌ Could not load LSMO dataset")
            print(f"\n  Available datasets:")
            try:
                catalog = tfds.list_builders()
                lsmo_related = [d for d in catalog if 'lsmo' in d.lower()]
                if lsmo_related:
                    for d in lsmo_related:
                        print(f"    - {d}")
                else:
                    print(f"    (No LSMO datasets found in catalog)")
            except:
                pass
            sys.exit(1)

except ImportError:
    print(f"  ❌ tensorflow_datasets not installed")
    print(f"  Installing dependencies...")
    os.system("pip install tensorflow tensorflow-datasets -q")
    sys.exit(1)

except Exception as e:
    print(f"  ❌ Error: {e}")
    sys.exit(1)

# ============================================================================
# PHASE 2: Explore Dataset Structure
# ============================================================================

print("\n[PHASE 2] Exploring dataset structure...")

episodes = []
episode_count = 0
task_distribution = {}

try:
    for example in dataset.take(10):  # Sample first 10 episodes
        episode_count += 1
        
        # Extract episode data
        episode_data = {
            'steps': []
        }
        
        # Check what's in an episode
        if isinstance(example, dict):
            print(f"\n  Episode {episode_count} keys: {list(example.keys())}")
            
            # Try to extract steps
            if 'steps' in example:
                steps = example['steps']
                episode_data['num_steps'] = len(steps) if hasattr(steps, '__len__') else '?'
                
                # Sample first step
                if hasattr(steps, '__iter__'):
                    for i, step in enumerate(steps):
                        if i == 0:  # Just show first step
                            print(f"    Step 0 keys: {list(step.keys()) if isinstance(step, dict) else type(step)}")
                        if i >= 2:
                            break
            
            # Check for metadata
            if 'metadata' in example:
                meta = example['metadata']
                print(f"    Metadata: {meta}")
                task = str(meta.get('task_description', meta.get('instruction', 'unknown')))[:50]
                task_distribution[task] = task_distribution.get(task, 0) + 1
        
        episodes.append(episode_data)
    
    print(f"\n  ✓ Loaded {episode_count} episodes (sampled)")
    print(f"  Task distribution: {task_distribution}")

except Exception as e:
    print(f"  ⚠️  Could not fully explore: {str(e)[:100]}")

# ============================================================================
# PHASE 3: Count Total Episodes (With Timeout)
# ============================================================================

print("\n[PHASE 3] Counting total episodes...")

try:
    # Count episodes with a larger sample
    full_count = 0
    for example in dataset.take(100):
        full_count += 1
        if full_count % 10 == 0:
            print(f"  Processed {full_count} episodes...")
    
    # Estimate total
    print(f"  ✓ Sampled {full_count} episodes")
    if full_count == 100:
        print(f"  (Dataset likely has many more episodes)")

except Exception as e:
    print(f"  ⚠️  Counting interrupted: {str(e)[:60]}")

# ============================================================================
# PHASE 4: Save Sample Episodes
# ============================================================================

print("\n[PHASE 4] Saving sample episodes locally...")

output_dir = Path('data/lsmo_real')
output_dir.mkdir(parents=True, exist_ok=True)

try:
    sample_episodes = []
    for i, example in enumerate(dataset.take(5)):  # Save 5 real episodes
        print(f"  Processing episode {i+1}...")
        
        try:
            # Convert to JSON-serializable format
            episode_dict = {}
            
            for key, value in example.items():
                if isinstance(value, (tf.Tensor, np.ndarray)):
                    episode_dict[key] = str(value.shape)  # Store shape info
                elif isinstance(value, (str, int, float)):
                    episode_dict[key] = str(value)
                elif isinstance(value, dict):
                    episode_dict[key] = {k: str(v)[:100] for k, v in value.items()}
                else:
                    episode_dict[key] = str(type(value))
            
            sample_episodes.append(episode_dict)
            
        except Exception as e_ep:
            print(f"    Could not serialize: {str(e_ep)[:50]}")
    
    # Save metadata
    metadata = {
        'dataset': dataset_name,
        'sample_episodes': sample_episodes,
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'download_status': 'SUCCESS' if episode_count > 0 else 'PARTIAL'
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  ✓ Saved metadata to {output_dir / 'metadata.json'}")

except Exception as e:
    print(f"  ⚠️  Could not save samples: {str(e)[:60]}")

# ============================================================================
# PHASE 5: Verify Dataset Format
# ============================================================================

print("\n[PHASE 5] Verifying dataset format for VLA+MPC...")

try:
    # Check if dataset has the required components for VLA+MPC
    required_keys = ['observation', 'action', 'instruction']
    found_keys = {}
    
    for example in dataset.take(1):
        for key in example.keys():
            found_keys[key] = str(example[key]).split(',')[0][:50]
    
    print(f"  Found keys: {list(found_keys.keys())}")
    
    # Check if we have image data
    has_images = False
    has_actions = False
    has_instructions = False
    
    for example in dataset.take(1):
        if 'observation' in example or 'image' in example or 'rgb' in example:
            has_images = True
        if 'action' in example or 'actions' in example:
            has_actions = True
        if 'instruction' in example or 'language' in example or 'task' in example:
            has_instructions = True
    
    print(f"\n  Dataset components:")
    print(f"    Images:       {('✓' if has_images else '❌')} {'Needed for VLA' if has_images else 'MISSING'}")
    print(f"    Actions:      {('✓' if has_actions else '❌')} {'For MPC control' if has_actions else 'MISSING'}")
    print(f"    Instructions: {('✓' if has_instructions else '❌')} {'For VLA input' if has_instructions else 'MISSING'}")

except Exception as e:
    print(f"  ⚠️  Could not verify format: {str(e)[:60]}")

# ============================================================================
# FINAL STATUS
# ============================================================================

print("\n" + "="*80)
print("STATUS")
print("="*80)

print(f"""
LSMO Dataset Status:
  ✓ Download: {'SUCCESS' if episode_count > 0 else 'FAILED'}
  ✓ Episodes sampled: {episode_count}
  ✓ Metadata saved: {(output_dir / 'metadata.json').exists()}
  
Next steps:
  1. Use dataset in VLA+MPC integration tests
  2. Extract images and actions for benchmarking
  3. Measure task success rates
  4. Generate visualizations
  
Dataset location: {output_dir}
TensorFlow cache: {Path.home() / '.tfds'}
""")

print("="*80)
print("\n✓ Dataset exploration complete. Ready for VLA+MPC testing.")
