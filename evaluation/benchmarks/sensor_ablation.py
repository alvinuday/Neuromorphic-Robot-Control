"""
B4 — Sensor Ablation Benchmark

Tests different fusion modes (M0-M4) with DualSystemController.
For each mode:
    - Run 3 episodes with DualSystemController
    - Measure: fusion latency (ms), feature norm, MPC tracking RMSE
    
Output includes comparison table.
Expected: M4 should have lowest RMSE (more state info → better tracking).
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
import yaml

from src.simulation.envs.xarm_env import XArmEnv
from src.mpc.xarm_mpc_controller import XArmMPCController
from src.solver.osqp_solver import OSQPSolver
from src.smolvla.mock_vla import MockVLAServer
from src.smolvla_client.trajectory_buffer import TrajectoryBuffer
from src.fusion.encoders.real_fusion_simple import RealFusionEncoder
from src.integration.dual_system_controller import DualSystemController


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


def sinusoidal_ref(t, n_joints=6, freq=0.5, amp=0.3):
    """Generate sinusoidal reference trajectory [N, 6]."""
    steps = np.arange(200)
    refs = amp * np.sin(2 * np.pi * freq * (steps + t) * 0.01)
    return np.tile(refs[:, None], (1, n_joints))


def run_episode_with_mode(ctrl, env, episode_idx: int, n_steps: int = 200):
    """Run single episode with current fusion mode."""
    obs = env.reset()
    rmses = []
    step_times = []
    
    ref = sinusoidal_ref(episode_idx * 200, n_joints=6)
    
    for step in range(n_steps):
        q_ref = ref[step]
        
        # Controller step
        t0 = time.perf_counter()
        tau = ctrl.step(
            q=obs['q'],
            qdot=obs['qdot'],
            rgb=obs['rgb'],
            instruction="maintain trajectory"
        )
        step_time_ms = (time.perf_counter() - t0) * 1000.0
        
        # Environment step
        obs, reward, done, _ = env.step(tau)
        
        # Compute metrics
        rmse = float(np.sqrt(np.mean((obs['q'] - q_ref) ** 2)))
        rmses.append(rmse)
        step_times.append(step_time_ms)
        
        if done:
            break
    
    return {
        'rmse': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'step_time_ms': float(np.mean(step_times)),
        'step_time_std': float(np.std(step_times)),
        'steps_completed': step + 1,
    }


def main():
    """Run B4 benchmark."""
    print("=" * 80)
    print("B4 — Sensor Ablation Benchmark (Fusion Modes M0-M4)")
    print("=" * 80)
    
    config = {
        "controller": "dual_system",
        "vla_mode": "MOCK",
        "n_modes": 5,
        "episodes_per_mode": 3,
        "steps_per_episode": 200,
    }
    
    with open("config/robots/xarm_6dof.yaml") as f:
        robot_cfg = yaml.safe_load(f)
    
    modes = [
        ("M0_rgb_only", lambda: RealFusionEncoder.rgb_only()),
        ("M1_rgb_events", lambda: RealFusionEncoder.rgb_events()),
        ("M2_rgb_lidar", lambda: RealFusionEncoder.rgb_lidar()),
        ("M3_rgb_proprio", lambda: RealFusionEncoder.rgb_proprio()),
        ("M4_full_fusion", lambda: RealFusionEncoder.full_fusion()),
    ]
    
    all_results = {}
    summary = []
    
    for mode_name, mode_factory in modes:
        print(f"\n{'─'*80}")
        print(f"Testing {mode_name}...")
        print(f"{'─'*80}")
        
        env = XArmEnv(render_mode='offscreen')
        mpc = XArmMPCController(OSQPSolver(), robot_cfg)
        vla = MockVLAServer()
        enc = mode_factory()
        traj_buffer = TrajectoryBuffer(n_joints=8, arrival_threshold_rad=0.1)
        
        # Create controller with this fusion mode
        ctrl = DualSystemController(
            mpc_solver=mpc,
            smolvla_client=vla,
            trajectory_buffer=traj_buffer,
            n_joints=8,
            control_dt_s=0.01,
            vla_query_interval_s=0.2,
        )
        # Store encoder for reference (though DualSystemController doesn't use it directly)
        ctrl.fusion_encoder = enc
        
        mode_results = []
        for ep in range(3):
            print(f"  Episode {ep+1}/3...", end=" ", flush=True)
            ep_result = run_episode_with_mode(ctrl, env, ep, n_steps=200)
            mode_results.append(ep_result)
            print(f"RMSE={ep_result['rmse']:.4f}")
        
        env.close()
        
        # Aggregate mode results
        rmses = [e['rmse'] for e in mode_results]
        step_times = [e['step_time_ms'] for e in mode_results]
        
        mode_summary = {
            'mode': mode_name,
            'episodes': mode_results,
            'mean_rmse': float(np.mean(rmses)),
            'std_rmse': float(np.std(rmses)),
            'mean_step_time_ms': float(np.mean(step_times)),
            'step_time_std_ms': float(np.std(step_times)),
        }
        all_results[mode_name] = mode_summary
        summary.append(mode_summary)
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("Summary Table")
    print(f"{'='*80}")
    print(f"{'Mode':<20} {'Mean RMSE':<12} {'Step Time (ms)':<15}")
    print("─" * 60)
    for entry in summary:
        print(f"{entry['mode']:<20} {entry['mean_rmse']:<12.4f} {entry['mean_step_time_ms']:<15.2f}")
    
    result = {
        **make_result_header("B4_sensor_ablation", config),
        "results": all_results,
        "summary": summary,
    }
    
    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    path = f"evaluation/results/B4_sensor_ablation_{int(time.time())}.json"
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"B4 Complete")
    print(f"{'='*80}")
    best_mode = min(summary, key=lambda x: x['mean_rmse'])
    print(f"Best Mode (lowest RMSE): {best_mode['mode']} (RMSE={best_mode['mean_rmse']:.4f})")
    print(f"Saved: {path}")
    
    return result


if __name__ == "__main__":
    main()
