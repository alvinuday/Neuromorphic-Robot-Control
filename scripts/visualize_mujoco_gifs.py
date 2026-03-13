#!/usr/bin/env python3
"""
MuJoCo 3D Visualization and GIF Generation
===========================================

Creates 3D GIF visualizations of robot tasks using MuJoCo simulator.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import List
import logging

try:
    from PIL import Image
    import imageio
except ImportError:
    print("Installing visualization dependencies...")
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pillow", "imageio", "-q"])
    from PIL import Image
    import imageio

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environments.mujoco_3dof_env import MuJoCo3DOFEnv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("="*80)
print("MUJOCO 3D VISUALIZATION - GIF GENERATION")
print("="*80)

# ============================================================================
# LOAD REAL ROBOT DATA
# ============================================================================

print("\n[PHASE 1] Loading real robot data...")

episodes = []
data_file = Path('data/real_robot_datasets/openx_real_data.json')

if data_file.exists():
    with open(data_file, 'r') as f:
        data = json.load(f)
    episodes = data.get('episodes', [])[:3]  # Use first 3 episodes
    print(f"  ✓ Loaded {len(episodes)} episodes from OpenX-Embodiment")
else:
    print(f"  ⚠ No real data file at {data_file}")
    # Create synthetic episodes
    episodes = [
        {
            'episode_id': i,
            'robot_type': 'mobile_arm',
            'task': ['pick_place', 'pushing', 'reaching'][i % 3],
            'num_steps': 50,
        }
        for i in range(3)
    ]
    print(f"  ✓ Created {len(episodes)} synthetic episodes")

# ============================================================================
# PHASE 2: INITIALIZE MUJOCO ENVIRONMENT
# ============================================================================

print("\n[PHASE 2] Initializing MuJoCo environment...")

try:
    env = MuJoCo3DOFEnv(
        render_mode='rgb_array',
        camera_height=480,
        camera_width=640,
        headless=True
    )
    print(f"  ✓ Environment initialized")
    print(f"    Resolution: {env.camera_width}x{env.camera_height}")
except Exception as e:
    print(f"  ⚠ Could not initialize MuJoCo: {str(e)[:50]}")
    print(f"    Proceeding with synthetic frames instead...")
    env = None

# ============================================================================
# PHASE 3: GENERATE VISUALIZATIONS
# ============================================================================

print("\n[PHASE 3] Generating 3D visualizations...")

output_dir = Path('results/mujoco_visualizations')
output_dir.mkdir(parents=True, exist_ok=True)

generated_count = 0

for ep_idx, episode in enumerate(episodes):
    ep_id = episode.get('episode_id', ep_idx)
    task = episode.get('task', 'unknown')
    num_steps = episode.get('num_steps', 50)
    
    gif_path = output_dir / f"episode_{ep_id:02d}_{task}.gif"
    
    print(f"\n  Episode {ep_id+1}/{len(episodes)}: {task}")
    
    if env:
        # Reset environment
        env.reset()
        
        # Generate frames
        frames = []
        
        # Create random trajectory
        for step in range(min(num_steps, 30)):  # Limit to 30 steps for GIF
            # Random action
            action = np.random.uniform(-1, 1, size=3) * 0.5
            
            # Step simulation
            env.step(action)
            
            # Render frame
            rgb_frame = env.render()
            if rgb_frame is not None:
                frames.append(rgb_frame)
        
        # Save as GIF
        if frames:
            try:
                imageio.mimsave(
                    str(gif_path),
                    frames,
                    fps=30,
                    quality=7
                )
                print(f"    ✓ Saved {len(frames)} frames → {gif_path.name}")
                generated_count += 1
            except Exception as e:
                print(f"    ⚠ Could not save GIF: {str(e)[:40]}")
    
    else:
        # Create synthetic frames (colored rectangles)
        print(f"    Creating synthetic frames...")
        frames = []
        for step in range(20):
            # Create a simple 480x640 RGB frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Color based on task
            colors = {
                'pick_place': (255, 100, 100),  # Red
                'pushing': (100, 255, 100),      # Green
                'reaching': (100, 100, 255),     # Blue
            }
            color = colors.get(task, (128, 128, 128))
            
            # Fill frame with task color
            frame[:, :] = color
            
            # Add text overlay
            import cv2
            text = f"Episode {ep_id}: {task} (Step {step+1}/20)"
            cv2.putText(frame, text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (255, 255, 255), 2)
            
            frames.append(frame)
        
        # Save as GIF
        try:
            imageio.mimsave(str(gif_path), frames, fps=10)
            print(f"    ✓ Saved {len(frames)} synthetic frames → {gif_path.name}")
            generated_count += 1
        except Exception as e:
            print(f"    ⚠ Could not save GIF: {str(e)[:40]}")

# ============================================================================
# PHASE 4: SUMMARY
# ============================================================================

print("\n" + "="*80)
print("VISUALIZATION SUMMARY")
print("="*80)
print(f"\n✓ Generated {generated_count} GIFs")
print(f"✓ Output directory: {output_dir}")
print(f"\nGIF Examples:")

for gif_file in sorted(output_dir.glob("*.gif"))[:5]:
    file_size_mb = gif_file.stat().st_size / (1024*1024)
    print(f"  • {gif_file.name} ({file_size_mb:.1f} MB)")

print("\n✓ Visualization complete!")
print("="*80)

# Close environment
if env:
    env.close()
