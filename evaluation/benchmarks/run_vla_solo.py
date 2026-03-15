"""
B2 — VLA Solo Benchmark

Setup: XArmEnv + MockVLA only (execute VLA actions without MPC).
Episodes: 5, Steps: 100 each
Metrics: joint range covered, action smoothness (diff between consecutive actions)
Note: All results labeled "MOCK" in JSON since we use MockVLA
"""

import sys
from pathlib import Path

# Ensure src module is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import os
# Do NOT set MUJOCO_GL here - let user control it via environment

import json
import time
import datetime
import platform
import numpy as np

from src.simulation.envs.xarm_env import XArmEnv
from src.smolvla.mock_vla import MockVLAServer


def _mujoco_version():
    """Get MuJoCo version string."""
    try:
        import mujoco
        return mujoco.__version__
    except Exception:
        return "unknown"


def make_result_header(benchmark_id: str, config: dict) -> dict:
    """Create benchmark result header with environment info."""
    return {
        "benchmark_id": benchmark_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": config,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "mujoco": _mujoco_version(),
            "vla_mode": "MOCK",
        }
    }


def run_episode_vla_only(env, vla_client, episode_idx: int, n_steps: int = 100):
    """
    Run single VLA-only episode and collect metrics.
    
    Args:
        env: XArmEnv instance
        vla_client: MockVLAServer instance
        episode_idx: Episode index
        n_steps: Number of steps to run
        
    Returns:
        dict with episode metrics
    """
    obs = env.reset()
    actions = []
    joint_ranges = [[] for _ in range(6)]
    rgb_history = [obs['rgb']]
    
    for step in range(n_steps):
        # Get VLA action
        try:
            result = vla_client.predict(obs['rgb'], obs['q'], "pick up object")
            action_7d = np.array(result.get('action', np.zeros(7)), dtype=np.float32)
            action = action_7d[:6]  # Take arm joints only, drop gripper
        except Exception as e:
            # Fallback to zero action on error
            print(f"    Warning: VLA query failed at step {step}: {e}")
            action = np.zeros(6, dtype=np.float32)
        
        actions.append(action)
        
        # Add action zeros for gripper (we control arm joints only)
        tau = np.concatenate([action, np.zeros(2)])  # [6] + [2] = [8]
        
        # Environment step
        obs, reward, done, info = env.step(tau)
        
        # Track joint ranges
        for j in range(6):
            joint_ranges[j].append(obs['q'][j])
        
        rgb_history.append(obs['rgb'].copy())
        
        if done:
            break
    
    # Compute metrics
    actions = np.array(actions)  # [n_steps, 6]
    
    # Action smoothness: mean absolute difference between consecutive actions
    if len(actions) > 1:
        diffs = np.abs(np.diff(actions, axis=0))
        smoothness = float(np.mean(diffs))
    else:
        smoothness = 0.0
    
    # Joint range: max - min for each joint
    ranges = []
    for j in range(6):
        jr = np.array(joint_ranges[j])
        ranges.append(float(np.max(jr) - np.min(jr)))
    
    return {
        'action_smoothness': smoothness,
        'joint_range_mean': float(np.mean(ranges)),
        'joint_ranges': ranges,
        'steps_completed': step + 1,
    }


def main():
    """Run B2 benchmark."""
    print("=" * 80)
    print("B2 — VLA Solo Benchmark (MockVLA)")
    print("=" * 80)
    
    config = {
        "controller": "vla_only",
        "vla_mode": "MOCK",
        "n_episodes": 5,
        "steps_per_episode": 100,
    }
    
    env = XArmEnv(render_mode='offscreen')
    vla = MockVLAServer()
    
    print("\nRunning 5 episodes with VLA-only control...")
    episodes = []
    for i in range(5):
        print(f"  Episode {i+1}/5...", end=" ", flush=True)
        ep_result = run_episode_vla_only(env, vla, i, n_steps=100)
        episodes.append(ep_result)
        print(f"Smoothness={ep_result['action_smoothness']:.4f}, Joint range={ep_result['joint_range_mean']:.4f}")
    
    env.close()
    
    # Aggregate results
    smoothness = [e['action_smoothness'] for e in episodes]
    ranges = [e['joint_range_mean'] for e in episodes]
    
    result = {
        **make_result_header("B2_vla_solo_mock", config),
        "results": {
            "episodes": episodes,
            "mean_action_smoothness": float(np.mean(smoothness)),
            "std_action_smoothness": float(np.std(smoothness)),
            "mean_joint_range": float(np.mean(ranges)),
            "std_joint_range": float(np.std(ranges)),
        }
    }
    
    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    path = f"evaluation/results/B2_vla_solo_mock_{int(time.time())}.json"
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"B2 Complete")
    print(f"{'='*80}")
    print(f"Mean Action Smoothness: {result['results']['mean_action_smoothness']:.4f}")
    print(f"Mean Joint Range: {result['results']['mean_joint_range']:.4f} rad")
    print(f"Saved: {path}")
    
    return result


if __name__ == "__main__":
    main()
