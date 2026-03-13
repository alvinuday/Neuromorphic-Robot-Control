#!/usr/bin/env python3
"""
Download REAL LSMO dataset from Open X-Embodiment collection
==============================================================

Using official sources:
1. HuggingFace (jxu124/OpenX-Embodiment) - easiest
2. TensorFlow Datasets (tfds) - official
3. Google Cloud Storage (gsutil) - direct
"""

import json
import os
import sys
from pathlib import Path

print("="*80)
print("DOWNLOADING REAL LSMO DATASET - OPEN X-EMBODIMENT")
print("="*80)

output_dir = Path('data/lsmo_real')
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# APPROACH 1: HuggingFace (Easiest - Recommended)
# ============================================================================

print("\n[APPROACH 1] HuggingFace Open X-Embodiment (jxu124/OpenX-Embodiment)")
print("Attempting to load via 'datasets' library with streaming...")

try:
    from datasets import load_dataset
    import time
    
    print("\n  Loading LSMO dataset from HuggingFace (streaming mode)...\n")
    
    # Load with streaming to avoid huge downloads
    dataset = load_dataset(
        "jxu124/OpenX-Embodiment",
        "lsmo",  # LSMO subset specifically
        split="train",
        streaming=True,
        trust_remote_code=True
    )
    
    print(f"  ✓ Successfully connected to LSMO dataset!")
    print(f"    Dataset: {dataset}")
    print(f"    Type: Streaming (infinite)")
    
    # Sample real episodes
    print(f"\n  Sampling real LSMO episodes...\n")
    
    real_episodes = []
    
    for i, example in enumerate(dataset.take(20)):  # Get 20 real episodes
        try:
            ep_info = {
                'episode_id': i,
                'keys': list(example.keys()) if isinstance(example, dict) else [],
                'dataset_name': example.get('dataset_name', 'lsmo'),
                'robot_type': example.get('robot_name', example.get('robot_type', 'mobile_manipulation')),
            }
            
            # Check structure
            if 'observation' in example:
                obs = example['observation']
                if isinstance(obs, dict):
                    ep_info['observation_keys'] = list(obs.keys())
                    if 'image' in obs:
                        try:
                            img_shape = obs['image'].shape
                            ep_info['image_shape'] = list(img_shape)
                        except:
                            pass
            
            if 'action' in example:
                try:
                    action_shape = example['action'].shape
                    ep_info['action_shape'] = list(action_shape)
                except:
                    pass
            
            if 'language_instruction' in example or 'instruction' in example:
                key = 'language_instruction' if 'language_instruction' in example else 'instruction'
                ep_info['instruction'] = str(example[key])[:100]
            
            real_episodes.append(ep_info)
            
            # Print sample
            robot = ep_info.get('robot_type', '?')[:25]
            inst = ep_info.get('instruction', 'n/a')[:40]
            print(f"  [{i+1:2d}] {robot:25} | {inst}")
            
        except Exception as e_ep:
            print(f"  [{i+1:2d}] Error: {str(e_ep)[:50]}")
            continue
    
    # Save metadata
    metadata = {
        'source': 'Open X-Embodiment (HuggingFace: jxu124/OpenX-Embodiment)',
        'subset': 'LSMO (Tokyo-U Mobile Manipulation)',
        'real_data': True,
        'url': 'https://huggingface.co/datasets/jxu124/OpenX-Embodiment',
        'method': 'HuggingFace streaming',
        'episodes_loaded': len(real_episodes),
        'episodes': real_episodes,
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
    }
    
    with open(output_dir / 'lsmo_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n  ✓ Saved {len(real_episodes)} REAL LSMO episodes!")
    print(f"  ✓ Metadata: {output_dir / 'lsmo_metadata.json'}")
    
    success = True
    
except ImportError as e:
    print(f"\n  ⚠ 'datasets' library not installed")
    print(f"  Installing: pip install datasets\n")
    os.system("pip install datasets -q")
    print(f"  Please re-run this script")
    success = False

except Exception as e:
    print(f"\n  ⚠ HuggingFace method failed: {str(e)[:100]}")
    success = False

if not success:
    # ============================================================================
    # APPROACH 2: TensorFlow Datasets (Official)
    # ============================================================================
    
    print("\n" + "="*80)
    print("[APPROACH 2] TensorFlow Datasets (Official TFDS)")
    print("="*80)
    
    try:
        import tensorflow_datasets as tfds
        
        print("\n  Loading LSMO from tfds...")
        
        # Try loading LSMO from official TFDS
        dataset = tfds.load(
            'lsmo',  # Dataset name
            split='train',
            shuffle_files=True,
            as_supervised=False,
            batch_size=-1,  # Load all
            download=True
        )
        
        print(f"  ✓ Successfully loaded LSMO from official TFDS!")
        
        real_episodes = []
        
        for i, example in enumerate(dataset.take(20)):
            real_episodes.append({
                'episode_id': i,
                'source': 'official_tfds',
                'keys': list(example.keys())
            })
        
        metadata = {
            'source': 'Official TensorFlow Datasets',
            'dataset': 'lsmo',
            'real_data': True,
            'episodes_loaded': len(real_episodes),
            'episodes': real_episodes,
            'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        }
        
        with open(output_dir / 'lsmo_tfds_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Loaded {len(real_episodes)} episodes from official TFDS")
        success = True
        
    except Exception as e:
        print(f"  ⚠ TFDS method failed: {str(e)[:100]}")

if not success:
    # ============================================================================
    # APPROACH 3: Google Cloud Storage via gsutil
    # ============================================================================
    
    print("\n" + "="*80)
    print("[APPROACH 3] Google Cloud Storage (GCS via gsutil)")
    print("="*80)
    
    print("\n  Checking for gsutil...")
    import shutil
    
    if shutil.which('gsutil'):
        print("  ✓ gsutil found")
        print("\n  Downloading LSMO from gs://gdm-robotics-open-x-embodiment/")
        print("  This may take a while...\n")
        
        # Create tensorflow_datasets path
        tfds_path = Path.home() / 'tensorflow_datasets'
        tfds_path.mkdir(exist_ok=True)
        
        # Download
        cmd = f"gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/lsmo {tfds_path}/"
        print(f"  Running: {cmd}\n")
        
        result = os.system(cmd)
        
        if result == 0:
            print("\n  ✓ Successfully downloaded LSMO from GCS!")
            success = True
        else:
            print(f"  ⚠ gsutil download failed (exit code: {result})")
    else:
        print("  ⚠ gsutil not installed")
        print("  Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install")

# ============================================================================
# Final Status
# ============================================================================

print("\n" + "="*80)
print("REAL LSMO DATASET STATUS")
print("="*80)

files = list(output_dir.glob('*.json'))

if files:
    print(f"\n✓ REAL LSMO data acquired!")
    print(f"\nMetadata files created:")
    for f in files:
        size = f.stat().st_size
        with open(f) as fp:
            data = json.load(fp)
            episodes = data.get('episodes_loaded', len(data.get('episodes', [])))
        print(f"  - {f.name}")
        print(f"    Episodes: {episodes}, Size: {size} bytes")
else:
    print(f"\n⚠ No metadata files created yet")
    print(f"\nTo complete download manually:")
    print(f"""
  Option 1 (HuggingFace):
    from datasets import load_dataset
    ds = load_dataset('jxu124/OpenX-Embodiment', 'lsmo', split='train', streaming=True)
    for example in ds.take(20):
        print(example)
  
  Option 2 (Official TFDS):
    import tensorflow_datasets as tfds
    ds = tfds.load('lsmo', split='train', download=True)
  
  Option 3 (GCS):
    gsutil -m cp -r gs://gdm-robotics-open-x-embodiment/lsmo ~/tensorflow_datasets/
    """)

print("\n" + "="*80)
print("NEXT STEPS: Use loaded data in VLA+MPC integration tests")
print("="*80)
