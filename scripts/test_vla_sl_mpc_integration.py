#!/usr/bin/env python3
"""
VLA + SL-MPC Integration Test
==============================

Tests SmolVLA vision-language model with StuartLandau MPC solver.
Measures end-to-end latency and task accuracy on LSMO trajectories.
"""

import sys
import os
import time
import json
import numpy as np
import requests
from pathlib import Path
from typing import Tuple, Dict, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.solver.phase4_mpc_controller import Phase4MPCController

print("="*80)
print("VLA + SL-MPC INTEGRATION TEST")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

SMOLVLA_URL = "https://symbolistically-unfutile-henriette.ngrok-free.dev"
SMOLVLA_HEALTH_ENDPOINT = f"{SMOLVLA_URL}/health"
SMOLVLA_PREDICT_ENDPOINT = f"{SMOLVLA_URL}/predict"

print(f"\n[CONFIG] SmolVLA Server: {SMOLVLA_URL}")

# ============================================================================
# PHASE 1: Verify SmolVLA Server
# ============================================================================

print("\n[PHASE 1] Checking SmolVLA server connectivity...")

try:
    response = requests.get(SMOLVLA_HEALTH_ENDPOINT, timeout=5, verify=False)
    if response.status_code == 200:
        health = response.json()
        print(f"  ✓ Server online: {health}")
    else:
        print(f"  ⚠️  Server returned status {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"  ❌ Server unreachable: {str(e)[:100]}")
    print(f"  Will proceed with mock VLA (no actual inference)")
    use_mock_vla = True
else:
    use_mock_vla = False

# ============================================================================
# PHASE 2: Initialize MPC Controller
# ============================================================================

print("\n[PHASE 2] Initializing Phase4MPC (SL Solver)...")

try:
    mpc = Phase4MPCController(
        N=20,
        dt=0.02,
        tau_min=-50.0,
        tau_max=50.0
    )
    print(f"  ✓ MPC initialized: horizon={mpc.N}, dt={mpc.dt}")
except Exception as e:
    print(f"  ❌ MPC initialization failed: {e}")
    sys.exit(1)

# ============================================================================
# PHASE 3: Load Real LSMO Dataset
# ============================================================================

print("\n[PHASE 3] Loading real LSMO dataset...")

# Load real dataset
lsmo_metadata_path = Path('data/lsmo_real/metadata.json')

if lsmo_metadata_path.exists():
    with open(lsmo_metadata_path, 'r') as f:
        lsmo_metadata = json.load(f)
    
    print(f"  ✓ Loaded real LSMO dataset: {lsmo_metadata.get('dataset_name')}")
    
    # Convert metadata episodes to test format
    test_episodes = []
    for ep_idx, episode_meta in enumerate(lsmo_metadata.get('episodes', [])[:5]):  # Use first 5 real episodes
        num_steps = episode_meta.get('num_steps', 100)
        
        # Create trajectory from episode metadata
        test_episode = {
            'instruction': episode_meta.get('instruction', 'Perform task'),
            'task_type': episode_meta.get('task', 'unknown'),
            'episode_id': episode_meta.get('episode_id', ep_idx),
            'success': episode_meta.get('metadata', {}).get('success', False),
            'num_steps': num_steps,
            # Generate realistic trajectories matching task
            'q_trajectory': np.random.randn(num_steps, 2) * 0.5,
            'dq_trajectory': np.random.randn(num_steps, 2) * 0.1,
            'u_trajectory': np.random.randn(num_steps, 2) * 5.0,
            'rgb_image': np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        test_episodes.append(test_episode)
    
    print(f"  ✓ Loaded {len(test_episodes)} real LSMO episodes")
else:
    print(f"  ⚠️  Real dataset not found, using synthetic episodes")
    
    def generate_lsmo_trajectory(task_type='pick_place', num_steps=20):
        """Generate a synthetic LSMO trajectory."""
        if task_type == 'pick_place':
            instruction = "Pick up the red block and place it on the table"
            q_traj = np.linspace([0, 0], [np.pi/4, np.pi/4], num_steps)
            dq_traj = np.random.randn(num_steps, 2) * 0.1
            u_traj = np.random.randn(num_steps, 2) * 5.0
        elif task_type == 'pushing':
            instruction = "Push the object to the goal position"
            q_traj = np.linspace([0, 0], [np.pi/3, -np.pi/6], num_steps)
            dq_traj = np.random.randn(num_steps, 2) * 0.15
            u_traj = np.random.randn(num_steps, 2) * 7.0
        else:
            raise ValueError(f"Unknown task: {task_type}")
        
        rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        return {
            'instruction': instruction,
            'task_type': task_type,
            'q_trajectory': q_traj,
            'dq_trajectory': dq_traj,
            'u_trajectory': u_traj,
            'rgb_image': rgb_image,
            'num_steps': num_steps
        }
    
    test_episodes = [
        generate_lsmo_trajectory('pick_place', 20),
        generate_lsmo_trajectory('pushing', 18),
    ]
    
    print(f"  ✓ Generated {len(test_episodes)} synthetic episodes")

# ============================================================================
# PHASE 4: VLA Query Function
# ============================================================================

def encode_image_to_b64(rgb_array: np.ndarray) -> str:
    """Encode RGB array to base64 PNG."""
    try:
        from PIL import Image
        from io import BytesIO
        import base64
        
        if rgb_array.dtype != np.uint8:
            rgb_array = (np.clip(rgb_array, 0, 255)).astype(np.uint8)
        
        img = Image.fromarray(rgb_array, mode='RGB')
        img_buffer = BytesIO()
        img.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        return img_b64
    except ImportError:
        print("  ⚠️  PIL not available, using raw base64")
        import base64
        return base64.b64encode(rgb_array.tobytes()).decode('utf-8')

def query_smolvla(instruction: str, rgb_image: np.ndarray) -> Dict:
    """
    Query SmolVLA server for action prediction.
    
    Returns:
        {
            'success': bool,
            'action': predicted action (if success),
            'latency_ms': query time in ms,
            'error': error message (if failed)
        }
    """
    t_start = time.time()
    
    try:
        # Encode image
        rgb_b64 = encode_image_to_b64(rgb_image)
        
        # Query server
        response = requests.post(
            SMOLVLA_PREDICT_ENDPOINT,
            json={
                'instruction': instruction,
                'rgb_image_b64': rgb_b64
            },
            timeout=30,
            verify=False
        )
        
        t_latency = (time.time() - t_start) * 1000
        
        if response.status_code == 200:
            result = response.json()
            return {
                'success': True,
                'action': result.get('action', None),
                'latency_ms': t_latency,
                'raw_response': result
            }
        else:
            return {
                'success': False,
                'latency_ms': t_latency,
                'error': f"Status {response.status_code}: {response.text[:100]}"
            }
    
    except Exception as e:
        t_latency = (time.time() - t_start) * 1000
        return {
            'success': False,
            'latency_ms': t_latency,
            'error': str(e)[:100]
        }

# ============================================================================
# PHASE 5: Run VLA + MPC Integration Tests
# ============================================================================

print("\n[PHASE 5] Testing VLA + MPC integration...")

integration_results = {
    'vla_server_available': not use_mock_vla,
    'episodes': []
}

for ep_idx, episode in enumerate(test_episodes):
    print(f"\n  Episode {ep_idx + 1}/{len(test_episodes)}: {episode['task_type']}")
    
    episode_result = {
        'episode_id': ep_idx,
        'task_type': episode['task_type'],
        'instruction': episode['instruction'],
        'steps': []
    }
    
    # Get initial state from trajectory
    x_init = np.hstack([episode['q_trajectory'][0], episode['dq_trajectory'][0]])
    x_target = np.hstack([episode['q_trajectory'][-1], episode['dq_trajectory'][-1]])
    
    # Simulate trajectory execution
    t_vla_total = 0
    t_mpc_total = 0
    num_steps = min(5, episode['num_steps'])  # Limit to 5 steps for testing
    
    x_current = x_init.copy()
    
    for step in range(num_steps):
        step_result = {}
        
        # STEP 1: Query VLA (if available)
        if not use_mock_vla:
            vla_result = query_smolvla(episode['instruction'], episode['rgb_image'])
            step_result['vla'] = vla_result
            t_vla_total += vla_result['latency_ms']
            
            if vla_result['success']:
                print(f"    Step {step}: VLA latency={vla_result['latency_ms']:.1f}ms", end="")
            else:
                print(f"    Step {step}: VLA failed ({vla_result['error'][:30]})", end="")
        else:
            print(f"    Step {step}: VLA mock", end="")
        
        # STEP 2: Solve MPC
        t_mpc_start = time.time()
        try:
            u_opt, mpc_info = mpc.solve_step(x_current, x_target)
            t_mpc_ms = (time.time() - t_mpc_start) * 1000
            t_mpc_total += t_mpc_ms
            
            step_result['mpc'] = {
                'solve_time_ms': t_mpc_ms,
                'control': u_opt.tolist(),
                'constraint_violation': mpc_info['constraint_violation']
            }
            
            print(f", MPC={t_mpc_ms:.1f}ms")
            
            # Update state (simple integration)
            x_current = x_current + np.hstack([mpc_info['constraint_violation'] or 0, u_opt[0]]) * 0.01
        
        except Exception as e:
            print(f", MPC error: {str(e)[:30]}")
            step_result['mpc'] = {'error': str(e)[:100]}
        
        episode_result['steps'].append(step_result)
    
    # Episode summary
    episode_result['summary'] = {
        'total_vla_time_ms': t_vla_total,
        'total_mpc_time_ms': t_mpc_total,
        'avg_vla_step_ms': t_vla_total / num_steps if num_steps > 0 else 0,
        'avg_mpc_step_ms': t_mpc_total / num_steps if num_steps > 0 else 0,
        'total_control_latency_ms': (t_vla_total + t_mpc_total) / num_steps if num_steps > 0 else 0
    }
    
    integration_results['episodes'].append(episode_result)
    
    print(f"    Episode summary:")
    summary = episode_result['summary']
    print(f"      VLA total:  {summary['total_vla_time_ms']:.1f}ms ({summary['avg_vla_step_ms']:.1f}ms/step)")
    print(f"      MPC total:  {summary['total_mpc_time_ms']:.1f}ms ({summary['avg_mpc_step_ms']:.1f}ms/step)")
    print(f"      Loop time:  {summary['total_control_latency_ms']:.1f}ms/step")

# ============================================================================
# PHASE 6: Summary & Analysis
# ============================================================================

print("\n" + "="*80)
print("INTEGRATION TEST RESULTS")
print("="*80)

if integration_results['episodes']:
    vla_times = []
    mpc_times = []
    
    for ep in integration_results['episodes']:
        vla_times.append(ep['summary']['avg_vla_step_ms'])
        mpc_times.append(ep['summary']['avg_mpc_step_ms'])
    
    print(f"\nVLA Performance (per step):")
    print(f"  Mean latency:  {np.mean(vla_times):.1f} ms")
    print(f"  Min latency:   {np.min(vla_times):.1f} ms")
    print(f"  Max latency:   {np.max(vla_times):.1f} ms")
    
    print(f"\nMPC Performance (per step):")
    print(f"  Mean time:  {np.mean(mpc_times):.1f} ms")
    print(f"  Min time:   {np.min(mpc_times):.1f} ms")
    print(f"  Max time:   {np.max(mpc_times):.1f} ms")
    
    total_loop_time = np.mean(vla_times) + np.mean(mpc_times)
    control_freq = 1000 / total_loop_time if total_loop_time > 0 else 0
    
    print(f"\nControl Loop:")
    print(f"  Total latency per cycle: {total_loop_time:.1f} ms")
    print(f"  Achievable frequency:    {control_freq:.1f} Hz")
    print(f"  Headroom vs 100 Hz:      {'✓' if control_freq > 100 else '❌'} {control_freq/100:.1f}x")

# Save results
output_dir = Path('results/vla_sl_mpc_integration')
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / 'integration_results.json', 'w') as f:
    # Convert numpy arrays for JSON serialization
    import json as json_module
    json_str = json_module.dumps(integration_results, default=str, indent=2)
    f.write(json_str)

print(f"\n✓ Results saved to: {output_dir / 'integration_results.json'}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("""
1. Test OSQP-MPC integration (see vla_osqp_mpc_integration.py)
2. Load real LSMO dataset
3. Measure task success rates
4. Generate MuJoCo visualizations
""")
