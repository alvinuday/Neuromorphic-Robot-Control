import sys
from datasets import load_dataset
import json
from pathlib import Path

print('Connecting to Open X-Embodiment LSMO dataset...')

ds = load_dataset(
    'jxu124/OpenX-Embodiment',
    'lsmo',
    split='train',
    streaming=True,
    trust_remote_code=True
)

print('✓ Connected!')
print('Loading 10 real LSMO episodes...\n')

episodes = []
for i, ex in enumerate(ds.take(10)):
    dataset_name = ex.get('dataset_name', 'lsmo')
    robot_name = ex.get('robot_name', '?')
    print(f'Episode {i+1}: {dataset_name} - {robot_name}')
    episodes.append({
        'id': i,
        'robot': robot_name,
        'dataset': dataset_name,
        'keys': list(ex.keys())
    })

# Save metadata
output_dir = Path('data/lsmo_real')
output_dir.mkdir(parents=True, exist_ok=True)

metadata = {
    'source': 'Open X-Embodiment (HuggingFace: jxu124/OpenX-Embodiment)',
    'real_data': True,
    'subset': 'LSMO (Tokyo-U Mobile Manipulation)',
    'episodes': episodes,
    'total_loaded': len(episodes)
}

with open(output_dir / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print(f'\n✓ Saved {len(episodes)} REAL LSMO episodes')
print(f'✓ Metadata file: data/lsmo_real/metadata.json')
print(f'\nReady for VLA+MPC integration tests!')
