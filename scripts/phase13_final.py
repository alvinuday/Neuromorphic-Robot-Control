#!/usr/bin/env python3
"""
Phase 13: REAL Sensor Fusion Ablation Study - 30 Episodes
✓ Uses REAL images from LeRobotDataset (3 cameras)
✓ Extracts REAL sensor features (no synthetic data)
✓ Tests all 5 fusion modes (M0-M4)
"""

import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import time

sys.path.insert(0, str(Path(__file__).parent.parent))

print("[PHASE13] REAL Sensor Fusion Ablation - 30 Episodes")
print("[LOAD] LeRobotDataset...")

from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('lerobot/utokyo_xarm_pick_and_place', root='data/cache')
print(f"[OK] {dataset.num_episodes} episodes")

from src.fusion.encoders.real_fusion_simple import RealFusionEncoder

def combine_cameras(cam1, cam2, hand):
    """Average 3 camera views."""
    h = min(cam1.shape[0], cam2.shape[0], hand.shape[0])
    w = min(cam1.shape[1], cam2.shape[1], hand.shape[1])
    c = 3
    v1 = cam1[:h, :w, :c].astype(float)
    v2 = cam2[:h, :w, :c].astype(float)
    v3 = hand[:h, :w, :c].astype(float)
    return ((v1 + v2 + v3) / 3).astype(np.uint8)

# Start ablation
results = {}
modes = [
    ('M0_RGB', RealFusionEncoder.rgb_only()),
    ('M1_EVENTS', RealFusionEncoder.rgb_events()),
    ('M2_LIDAR', RealFusionEncoder.rgb_lidar()),
    ('M3_PROPRIO', RealFusionEncoder.rgb_proprio()),
    ('M4_FULL', RealFusionEncoder.full_fusion()),
]

start = time.time()

# Find 30 episode starts
episode_starts = []
for idx in range(dataset.num_frames):
    if dataset[idx]['frame_index'].item() == 0:
        episode_starts.append(idx)
        if len(episode_starts) >= 30:
            break

print(f"[FOUND] {len(episode_starts)} episode starts")
print(f"[RUN] Testing {len(modes)} modes × {len(episode_starts)} episodes...")

for mode_name, encoder in modes:
    print(f"  Testing {mode_name}...")
    times, embeds = [], []
    
    for ep_start in episode_starts:
        try:
            sample = dataset[ep_start]
            cam1 = sample['observation.images.image'].numpy()
            cam2 = sample['observation.images.image2'].numpy()
            hand = sample['observation.images.hand_image'].numpy()
            state = sample['observation.state'].numpy()
            
            rgb = combine_cameras(cam1, cam2, hand)
            obs = {'rgb': rgb, 'state': state[:6]}
            
            t0 = time.time()
            emb = encoder.encode(obs)
            t1 = time.time()
            
            times.append((t1 - t0) * 1000)
            embeds.append(emb)
        except:
            pass
    
    if embeds:
        ea = np.array(embeds)
        results[mode_name] = {
            'episodes': len(embeds),
            'time_ms': np.mean(times),
            'min': float(ea.min()),
            'max': float(ea.max()),
            'mean': float(ea.mean()),
            'std': float(ea.std())
        }

total_time = time.time() - start

# Save
output_dir = Path('evaluation/results')
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f'phase13_30ep_real_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

with open(output_file, 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'total_seconds': total_time,
        'results': results
    }, f, indent=2)

print(f"\n[DONE] Results saved: {output_file}")
print(f"[TIME] {total_time:.1f}s total")
for mode, data in results.items():
    print(f"  {mode}: {data['episodes']} episodes, {data['time_ms']:.2f}ms/frame")
