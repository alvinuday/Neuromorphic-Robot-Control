"""
B1 — MPC Solo Benchmark

Setup: XArmEnv + MPC (no VLA). Reference = sinusoidal trajectory.
Episodes: 5, Steps: 200 each
Metrics: per-joint tracking RMSE, mean/std solve time
Solvers: run once with OSQP, once with SL (separately) - currently OSQP only
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
            "vla_mode": config.get("vla_mode", "NONE"),
        }
    }


def sinusoidal_ref(t, n_joints=6, freq=0.5, amp=0.3):
    """
    Generate sinusoidal reference trajectory [N, 6].
    
    Args:
        t: Episode time offset
        n_joints: Number of joints (6 for arm only)
        freq: Frequency in Hz
        amp: Amplitude in radians
        
    Returns:
        [200, 6] reference trajectory array
    """
    steps = np.arange(200)
    refs = amp * np.sin(2 * np.pi * freq * (steps + t) * 0.01)
    return np.tile(refs[:, None], (1, n_joints))   # [200, 6]


def run_episode(env, mpc, episode_idx: int, n_steps: int = 200):
    """
    Run single MPC-only episode and collect metrics.
    
    Args:
        env: XArmEnv instance
        mpc: XArmMPCController instance
        episode_idx: Episode index (for reference trajectory offset)
        n_steps: Number of steps to run
        
    Returns:
        dict with episode metrics (rmse, solve_ms, steps_completed)
    """
    obs = env.reset()
    rmses = []
    solve_times = []
    ref = sinusoidal_ref(episode_idx * 200, n_joints=6)   # [200, 6]
    
    for step in range(n_steps):
        q_ref = ref[step]
        
        # MPC step
        t_start = time.perf_counter()
        tau = mpc.step((obs['q'], obs['qdot']), ref[step:])
        solve_ms = (time.perf_counter() - t_start) * 1000.0
        
        # Environment step
        obs, reward, done, info = env.step(tau)
        
        # Compute tracking RMSE
        rmse = float(np.sqrt(np.mean((obs['q'] - q_ref) ** 2)))
        rmses.append(rmse)
        solve_times.append(solve_ms)
        
        if done:
            break
    
    return {
        'rmse': float(np.mean(rmses)),
        'rmse_std': float(np.std(rmses)),
        'solve_ms': float(np.mean(solve_times)),
        'solve_ms_std': float(np.std(solve_times)),
        'steps_completed': step + 1,
    }


def main():
    """Run B1 benchmark."""
    print("=" * 80)
    print("B1 — MPC Solo Benchmark (OSQP Solver)")
    print("=" * 80)
    
    config = {
        "solver": "osqp",
        "n_episodes": 5,
        "steps_per_episode": 200,
        "reference": "sinusoidal",
    }
    
    with open("config/robots/xarm_6dof.yaml") as f:
        robot_cfg = yaml.safe_load(f)
    
    env = XArmEnv(render_mode='offscreen')
    mpc = XArmMPCController(OSQPSolver(), robot_cfg)
    
    print("\nRunning 5 episodes with sinusoidal reference...")
    episodes = []
    for i in range(5):
        print(f"  Episode {i+1}/5...", end=" ", flush=True)
        ep_result = run_episode(env, mpc, i, n_steps=200)
        episodes.append(ep_result)
        print(f"RMSE={ep_result['rmse']:.4f}, Solve={ep_result['solve_ms']:.1f}ms")
    
    env.close()
    
    # Aggregate results
    rmses = [e['rmse'] for e in episodes]
    solve_times = [e['solve_ms'] for e in episodes]
    
    result = {
        **make_result_header("B1_mpc_solo_osqp", config),
        "results": {
            "episodes": episodes,
            "mean_rmse": float(np.mean(rmses)),
            "std_rmse": float(np.std(rmses)),
            "mean_solve_ms": float(np.mean(solve_times)),
            "std_solve_ms": float(np.std(solve_times)),
        }
    }
    
    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    path = f"evaluation/results/B1_mpc_solo_osqp_{int(time.time())}.json"
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"B1 Complete")
    print(f"{'='*80}")
    print(f"Mean RMSE: {result['results']['mean_rmse']:.4f} rad")
    print(f"Mean Solve Time: {result['results']['mean_solve_ms']:.1f} ms")
    print(f"Saved: {path}")
    
    return result


if __name__ == "__main__":
    main()
