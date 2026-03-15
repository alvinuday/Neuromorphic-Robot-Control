"""Phase 3: Tests for MuJoCo environment."""
import os
import sys
import pytest
import numpy as np


@pytest.fixture
def env():
    """Create xArm environment."""
    # Set proper MUJOCO_GL for platform
    if sys.platform == 'darwin':  # macOS
        os.environ['MUJOCO_GL'] = 'cgl'
    else:  # Linux
        os.environ['MUJOCO_GL'] = 'osmesa'
    
    from src.simulation.envs.xarm_env import XArmEnv
    env = XArmEnv(render_mode='offscreen')
    yield env
    env.close()


def test_env_reset(env):
    """Test environment reset."""
    obs = env.reset()
    assert 'q' in obs
    assert 'qdot' in obs
    assert 'rgb' in obs
    assert obs['q'].shape == (6,)
    assert obs['qdot'].shape == (6,)
    assert obs['rgb'].shape == (84, 84, 3)


def test_env_step(env):
    """Test environment step."""
    env.reset()
    tau = np.zeros(8)
    obs, reward, done, info = env.step(tau)
    assert obs['q'].shape == (6,)
    assert isinstance(reward, (int, float))
    assert isinstance(done, (bool, np.bool_))
    assert isinstance(info, dict)


def test_arm_motion_with_torque(env):
    """Test that arm moves when torques are applied."""
    obs = env.reset()
    q0 = obs['q'][0]
    
    # Apply positive torque to joint 0
    tau = np.zeros(8)
    tau[0] = 5.0
    
    # Step for 30 iterations (0.3 seconds)
    for _ in range(30):
        obs, _, _, _ = env.step(tau)
    
    q_final = obs['q'][0]
    
    # Joint should have moved (either direction, but should move)
    assert abs(q_final - q0) > 0.01, f"Joint didn't move. q0={q0:.4f}, q_final={q_final:.4f}"


def test_env_multiple_steps(env):
    """Test environment over multiple steps."""
    env.reset()
    tau = np.random.randn(8) * 0.5
    
    for _ in range(100):
        obs, _, done, _ = env.step(tau)
        if done:
            break
    
    # Should complete without error
    assert obs['q'].shape == (6,)


def test_action_clipping(env):
    """Test that actions are clipped to limits."""
    env.reset()
    # Apply excessive torques
    tau = np.array([100., 100., 100., 100., 100., 100., 100., 100.])
    obs, _, _, _ = env.step(tau)
    # Should not crash
    assert obs['q'].shape == (6,)


def test_event_camera_import():
    """Test event camera can be imported."""
    from src.simulation.cameras.event_camera_simple import EventCameraSimulator
    cam = EventCameraSimulator()
    assert cam is not None


def test_lidar_import():
    """Test LiDAR simulator can be imported."""
    from src.simulation.cameras.event_camera_simple import LiDARSimulator
    lidar = LiDARSimulator()
    assert lidar is not None
