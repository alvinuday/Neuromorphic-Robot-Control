"""
B3 — Dual System Benchmark

Setup: XArmEnv + DualSystemController (OSQP-MPC + MockVLA)
Episodes: 5, Steps: 200 each
Metrics: MPC tracking RMSE, VLA query rate achieved, buffer staleness stats
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


def run_episode_dual(ctrl, env, episode_idx: int, n_steps: int = 200):
    """
    Run single dual-system episode.
    
    Args:
        ctrl: DualSystemController instance
        env: XArmEnv instance
        episode_idx: Episode index
        n_steps: Number of steps
        
    Returns:
        dict with episode metrics
    """
    obs = env.reset()
    rmses = []
    mpc_times = []
    vla_query_times = []
    
    ref = sinusoidal_ref(episode_idx * 200, n_joints=6)
    
    for step in range(n_steps):
        q_ref = ref[step]
        
        # Controller step (synchronous - non-blocking)
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
        
        # Compute RMSE
        rmse = float(np.sqrt(np.mean((obs['q'] - q_ref) ** 2)))
        rmses.append(rmse)
        mpc_times.append(step_time_ms)
        
        if done:
            break
    
    return {
        'rmse': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'step_time_ms': float(np.mean(mpc_times)),
        'step_time_ms_std': float(np.std(mpc_times)),
        'steps_completed': step + 1,
    }


def main():
    """Run B3 benchmark."""
    print("=" * 80)
    print("B3 — Dual System Benchmark (MPC + MockVLA)")
    print("=" * 80)
    
    config = {
        "mpc_solver": "osqp",
        "vla_mode": "MOCK",
        "n_episodes": 5,
        "steps_per_episode": 200,
        "reference": "sinusoidal",
    }
    
    with open("config/robots/xarm_6dof.yaml") as f:
        robot_cfg = yaml.safe_load(f)
    
    # Create components
    env = XArmEnv(render_mode='offscreen')
    mpc = XArmMPCController(OSQPSolver(), robot_cfg)
    vla = MockVLAServer()
    traj_buffer = TrajectoryBuffer(n_joints=8, arrival_threshold_rad=0.1)
    
    # Create dual controller
    ctrl = DualSystemController(
        mpc_solver=mpc,
        smolvla_client=vla,
        trajectory_buffer=traj_buffer,
        n_joints=8,
        control_dt_s=0.01,
        vla_query_interval_s=0.2,
    )
    
    print("\nRunning 5 episodes with dual-system control...")
    episodes = []
    for i in range(5):
        print(f"  Episode {i+1}/5...", end=" ", flush=True)
        ep_result = run_episode_dual(ctrl, env, i, n_steps=200)
        episodes.append(ep_result)
        print(f"RMSE={ep_result['rmse']:.4f}, Step={ep_result['step_time_ms']:.1f}ms")
    
    env.close()
    
    # Get controller stats
    ctrl_stats = {
        'state': str(ctrl.state),
        'step_count': ctrl.step_count,
        'avg_step_time_ms': float(np.mean(ctrl.step_times_ms)) if ctrl.step_times_ms else 0.0,
    }
    
    # Aggregate results
    rmses = [e['rmse'] for e in episodes]
    step_times = [e['step_time_ms'] for e in episodes]
    
    result = {
        **make_result_header("B3_dual_system_mock", config),
        "results": {
            "episodes": episodes,
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_step_time_ms": float(np.mean(step_times)),
            "std_step_time_ms": float(np.std(step_times)),
            "controller_stats": ctrl_stats,
        }
    }
    
    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    path = f"evaluation/results/B3_dual_system_mock_{int(time.time())}.json"
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"B3 Complete")
    print(f"{'='*80}")
    print(f"Mean RMSE: {result['results']['mean_rmse']:.4f} rad")
    print(f"Mean Step Time: {result['results']['mean_step_time_ms']:.1f} ms")
    print(f"Saved: {path}")
    
    return result


if __name__ == "__main__":
    main()
