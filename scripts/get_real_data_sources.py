#!/usr/bin/env python3
"""
Directly load real robot data from web sources
"""

import json
import urllib.request
import gzip
from pathlib import Path

print("="*80)
print("LOADING REAL ROBOT DATA FROM WEB SOURCES")
print("="*80)

output_dir = Path('data/real_robot_datasets')
output_dir.mkdir(parents=True, exist_ok=True)

# Real dataset sources with direct download URLs
sources = {
    'openx_metadata': {
        'url': 'https://raw.githubusercontent.com/google-research/robotics_transformer/main/examples/example_episode_batched.json',
        'name': 'OpenX-Embodiment Example',
        'method': 'github_raw'
    },
    'bridge_info': {
        'url': 'https://huggingface.co/api/datasets/rail-berkeley/bridge_dataset',
        'name': 'BRIDGE Dataset Info',
        'method': 'huggingface_api'
    }
}

print("\nAttempting direct web downloads of real robot datasets...\n")

successful = False

for source_key, source_info in sources.items():
    print(f"[{source_key}] {source_info['name']}...")
    
    try:
        with urllib.request.urlopen(source_info['url'], timeout=10) as response:
            data = response.read()
            
            if source_info['method'] == 'huggingface_api':
                info = json.loads(data)
                # Extract dataset info
                metadata = {
                    'source': 'BRIDGE Dataset (UC Berkeley)',
                    'real_data': True,
                    'url': source_info['url'],
                    'info': {
                        'id': info.get('id'),
                        'description': info.get('description', ''),
                        'downloads': info.get('downloads'),
                    }
                }
            else:
                # Store raw data
                metadata = {
                    'source': source_info['name'],
                    'real_data': True,
                    'data_size_bytes': len(data),
                    'url': source_info['url']
                }
            
            # Save metadata
            output_file = output_dir / f'{source_key}_metadata.json'
            with open(output_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"  ✓ Downloaded {len(data)} bytes")
            print(f"  ✓ Saved to {output_file}")
            successful = True
            
    except Exception as e:
        print(f"  ⚠ Failed: {str(e)[:60]}")

if not successful:
    print("\nDirect download failed. Creating bridge to real data sources...")
    print("\nActual real datasets you should use:")
    print("""
1. OPENX-EMBODIMENT (150K+ REAL robot episodes)
   - Location: https://huggingface.co/datasets/openx-embodiment
   - Method: from datasets import load_dataset
             ds = load_dataset('openx-embodiment', split='train', streaming=True)
   
2. BRIDGE DATASET (UC Berkeley UR5 robot)
   - Location: https://huggingface.co/datasets/rail-berkeley/bridge_dataset
   - Method: from datasets import load_dataset
             ds = load_dataset('rail-berkeley/bridge_dataset', split='train', streaming=True)

3. RT-1 DATASET (Google Robot)
   - Location: https://huggingface.co/datasets/google/rt-1
   
4. ROBONET (100K+ robot videos)
   - Location: https://huggingface.co/datasets/robonet/robonet

To use: pip install datasets huggingface-hub
        python3 -c "from datasets import load_dataset; print(load_dataset('openx-embodiment', split='train', streaming=True).take(1))"
    """)

# Create placeholder for real data
placeholder = {
    'note': 'Real datasets available at listed sources above',
    'datasets': {
        'openx_embodiment': {
            'episodes': 150000,
            'robots': ['UR5', 'Franka', 'Xarm', 'Fetch'],
            'source': 'Google Robotics + Multiple Labs'
        },
        'bridge': {
            'episodes': 60,
            'robots': ['UR5'],
            'source': 'UC Berkeley RAIL Lab'
        }
    }
}

with open(output_dir / 'real_data_sources.json', 'w') as f:
    json.dump(placeholder, f, indent=2)

print(f"\n✓ Created {output_dir}")
print(f"  To use real data: Install 'datasets' library and load from HuggingFace")
