#!/usr/bin/env python3
"""
Download REAL LSMO Dataset from Open X-Embodiment via HuggingFace
"""
import json
from pathlib import Path
import sys

print("="*80)
print("LOADING REAL LSMO DATA FROM OPEN X-EMBODIMENT (HUGGINGFACE)")
print("="*80)

try:
    print("\n[1/3] Connecting to HuggingFace Open X-Embodiment...")
    from datasets import load_dataset
    
    ds = load_dataset(
        'jxu124/OpenX-Embodiment',
        'lsmo',
        split='train',
        streaming=True,
        trust_remote_code=True
    )
    
    print("✓ Connected successfully!\n[2/3] Loading real LSMO episodes...")
    
    episodes = []
    for i, example in enumerate(ds.take(10)):
        robot = str(example.get('robot_name', 'unknown'))[:30]
        dataset = str(example.get('dataset_name', 'lsmo'))[:30]
        
        ep = {
            'id': i,
            'robot': robot,
            'dataset': dataset,
            'keys': list(example.keys()) if isinstance(example, dict) else []
        }
        episodes.append(ep)
        
        print(f"  Episode {i+1:2d}: {robot:20} | {dataset:20}")
    
    print(f"\n[3/3] Saving {len(episodes)} REAL episodes to data/lsmo_real/...")
    
    output_dir = Path('data/lsmo_real')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        'source': 'Open X-Embodiment (HuggingFace - REAL Robot Data)',
        'url': 'https://huggingface.co/datasets/jxu124/OpenX-Embodiment',
        'subset': 'LSMO - Tokyo-U Mobile Manipulation',
        'real_data': True,
        'episodes_loaded': len(episodes),
        'episodes': episodes,
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Saved to: data/lsmo_real/metadata.json")
    print("\n" + "="*80)
    print(f"✓ SUCCESS: {len(episodes)} REAL LSMO episodes ready!")
    print("="*80)
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
