#!/usr/bin/env python3
"""
End-to-end pick-and-place demonstration.

Usage:
    python scripts/run_demo.py --solver osqp --steps 200 --gif outputs/demo_osqp.gif
    python scripts/run_demo.py --solver sl   --steps 100 --gif outputs/demo_sl.gif
"""
import argparse
import json
import time
import numpy as np
import yaml
import os

os.environ.setdefault('MUJOCO_GL', 'osmesa')

from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="End-to-end robot control demo")
    parser.add_argument('--solver', choices=['osqp', 'sl'], default='osqp',
                        help='QP solver to use')
    parser.add_argument('--steps', type=int, default=200,
                        help='Number of simulation steps')
    parser.add_argument('--gif', default='outputs/demo.gif',
                        help='Output GIF path')
    parser.add_argument('--vla', choices=['mock'], default='mock',
                        help='VLA mode (mock for demo, real would require GPU)')
    args = parser.parse_args()

    # Load robot configuration
    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)

    # Import required modules
    from src.simulation.envs.xarm_env import XArmEnv
    from src.mpc.xarm_mpc_controller import XArmMPCController
    from src.integration.dual_system_controller import DualSystemController
    from src.smolvla.mock_vla import MockVLAServer
    from src.smolvla_client.trajectory_buffer import TrajectoryBuffer
    from src.visualization.episode_recorder import EpisodeRecorder

    # Choose solver
    if args.solver == 'osqp':
        from src.solver.osqp_solver import OSQPSolver
        solver_cls = OSQPSolver
        solver_name = "OSQP"
    else:
        from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
        solver_cls = StuartLandauLagrangeDirect
        solver_name = "Stuart-Landau"

    # Initialize environment and controllers
    print("=" * 80)
    print(f"E2E DEMO: {args.steps} steps, {solver_name} solver, MockVLA")
    print("=" * 80)

    env = XArmEnv(render_mode='offscreen')
    mpc = XArmMPCController(solver_cls(), cfg)
    vla = MockVLAServer()
    traj_buffer = TrajectoryBuffer(n_joints=8, arrival_threshold_rad=0.1)
    
    ctrl = DualSystemController(
        mpc_solver=mpc,
        smolvla_client=vla,
        trajectory_buffer=traj_buffer,
        n_joints=8,
        control_dt_s=0.01,
        vla_query_interval_s=0.2,
    )

    # Create output directory for GIF
    gif_dir = Path(args.gif).parent
    gif_dir.mkdir(exist_ok=True, parents=True)

    # Record episode
    rec = EpisodeRecorder(fps=10)
    obs = env.reset()
    rmses = []
    step_times = []
    
    # Simple reference: joint 0 moves to 0.4 rad
    ref = np.zeros((10, 6))
    ref[:, 0] = 0.4

    print(f"Running {args.steps} steps...")
    t0 = time.perf_counter()
    
    for step in range(args.steps):
        step_t0 = time.perf_counter()
        tau = ctrl.step(q=obs['q'], qdot=obs['qdot'], rgb=obs['rgb'], 
                       instruction="maintain trajectory")
        step_time = (time.perf_counter() - step_t0) * 1000.0
        
        obs, _, done, _ = env.step(tau)
        rec.add_frame(obs['rgb'])
        
        rmse = float(np.sqrt(np.mean((obs['q'] - ref[0]) ** 2)))
        rmses.append(rmse)
        step_times.append(step_time)
        
        if step % 50 == 0:
            print(f"  step {step:3d} | q[0]={obs['q'][0]:7.3f} | RMSE={rmse:.4f} | "
                  f"step_ms={step_time:6.2f}")
        if done:
            break

    elapsed = time.perf_counter() - t0
    env.close()

    # Save GIF
    gif_path = rec.save(args.gif)
    
    # Compute statistics
    stats = {
        'solver': args.solver,
        'vla_mode': 'MOCK',
        'n_steps': len(rmses),
        'elapsed_s': round(elapsed, 3),
        'hz': round(len(rmses) / elapsed, 1),
        'mean_rmse': round(float(np.mean(rmses)), 4),
        'std_rmse': round(float(np.std(rmses)), 4),
        'mean_step_time_ms': round(float(np.mean(step_times)), 2),
        'max_step_time_ms': round(float(np.max(step_times)), 2),
        'final_q0': round(float(obs['q'][0]), 3),
        'target_q0': 0.4,
        'tracking_error': round(abs(float(obs['q'][0]) - 0.4), 4),
        'gif_path': str(gif_path),
    }

    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(json.dumps(stats, indent=2))
    print(f"GIF saved: {gif_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
