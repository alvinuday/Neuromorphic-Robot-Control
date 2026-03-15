"""
End-to-end tests for Phase 11.

These tests verify complete system integration:
- MPC-only control
- Dual MPC + VLA control
- GIF recording
- Benchmark JSON production
"""
import os
import sys

# Unset MUJOCO_GL for macOS compatibility
if 'MUJOCO_GL' in os.environ:
    del os.environ['MUJOCO_GL']

import numpy as np
import pytest
import yaml
from pathlib import Path


@pytest.mark.e2e
def test_mpc_osqp_episode():
    """Test 50 steps of OSQP-MPC without VLA."""
    from src.simulation.envs.xarm_env import XArmEnv
    from src.mpc.xarm_mpc_controller import XArmMPCController
    from src.solver.osqp_solver import OSQPSolver

    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)

    env = XArmEnv(render_mode='offscreen')
    mpc = XArmMPCController(OSQPSolver(), cfg)
    obs = env.reset()
    q0 = obs['q'].copy()
    ref = np.zeros((10, 6))
    ref[:, 0] = 0.3

    for step in range(50):
        tau = mpc.step((obs['q'], obs['qdot']), ref)
        obs, _, done, _ = env.step(tau)
        if done:
            break

    env.close()

    # Arm must have moved toward reference
    assert abs(obs['q'][0]) > 0.001, "Arm didn't move at all"
    assert len(obs['q']) == 6, "State vector has wrong size"


@pytest.mark.e2e
def test_mpc_stuart_landau_episode():
    """Test 50 steps of Stuart-Landau MPC without VLA."""
    from src.simulation.envs.xarm_env import XArmEnv
    from src.mpc.xarm_mpc_controller import XArmMPCController
    from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect

    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)

    env = XArmEnv(render_mode='offscreen')
    mpc = XArmMPCController(StuartLandauLagrangeDirect(T_solve=3.0), cfg)
    obs = env.reset()
    ref = np.zeros((10, 6))
    ref[:, 0] = 0.3

    for step in range(50):
        tau = mpc.step((obs['q'], obs['qdot']), ref)
        obs, _, done, _ = env.step(tau)
        if done:
            break

    env.close()

    # Arm must have moved
    assert abs(obs['q'][0]) > 0.001, "Arm didn't move at all"


@pytest.mark.e2e
def test_dual_controller_episode():
    """Test dual MPC + VLA controller with MockVLA."""
    from src.simulation.envs.xarm_env import XArmEnv
    from src.mpc.xarm_mpc_controller import XArmMPCController
    from src.solver.osqp_solver import OSQPSolver
    from src.integration.dual_system_controller import DualSystemController
    from src.smolvla.mock_vla import MockVLAServer
    from src.smolvla_client.trajectory_buffer import TrajectoryBuffer

    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)

    env = XArmEnv(render_mode='offscreen')
    mpc = XArmMPCController(OSQPSolver(), cfg)
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

    obs = env.reset()
    for step in range(30):
        tau = ctrl.step(q=obs['q'], qdot=obs['qdot'], rgb=obs['rgb'],
                       instruction="maintain trajectory")
        obs, _, done, _ = env.step(tau)
        if done:
            break

    env.close()

    assert obs is not None
    assert len(obs['q']) == 6


@pytest.mark.e2e
def test_gif_produced(tmp_path):
    """Test GIF recording from simulation episode."""
    from src.simulation.envs.xarm_env import XArmEnv
    from src.visualization.episode_recorder import EpisodeRecorder

    env = XArmEnv(render_mode='offscreen')
    obs = env.reset()
    gif_path = str(tmp_path / "test_episode.gif")
    
    rec = EpisodeRecorder(fps=10)
    
    tau = np.zeros(8)
    tau[0] = 2.0
    
    for step in range(50):
        obs, _, _, _ = env.step(tau)
        rec.add_frame(obs['rgb'])

    env.close()
    
    saved_path = rec.save(gif_path)
    assert saved_path is not None
    
    size = Path(saved_path).stat().st_size
    assert size > 1000, f"GIF too small ({size} bytes) — may have encoding issues"


@pytest.mark.e2e
def test_benchmarks_produced_output():
    """Verify that at least one benchmark JSON exists from Phase 9."""
    results = list(Path("evaluation/results").glob("B*.json"))
    assert len(results) > 0, \
        "No benchmark results found. Run Phase 9 benchmarks first."


@pytest.mark.e2e
def test_dashboard_loads_benchmarks():
    """Verify dashboard can load benchmark data."""
    results = list(Path("evaluation/results").glob("B*.json"))
    
    for json_file in results[:3]:  # Test first 3 files
        import json
        with open(json_file) as f:
            data = json.load(f)
        
        assert 'benchmark_id' in data, f"Missing benchmark_id in {json_file}"
        assert 'environment' in data, f"Missing environment in {json_file}"
        # B1-B4 have 'results', B5 has 'summary'
        assert 'results' in data or 'summary' in data, f"Missing results/summary in {json_file}"


@pytest.mark.e2e
def test_demo_script_runs():
    """Test the end-to-end demo script (fast version, 10 steps)."""
    import subprocess
    result = subprocess.run(
        ['python', 'scripts/run_demo.py', '--solver', 'osqp', '--steps', '10',
         '--gif', '/tmp/test_demo.gif'],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0, f"Demo failed:\n{result.stderr}"
    assert "E2E DEMO" in result.stdout, "Demo didn't execute properly"
    assert "DEMO COMPLETE" in result.stdout, "Demo didn't complete properly"
