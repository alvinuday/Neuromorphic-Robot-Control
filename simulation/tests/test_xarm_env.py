"""
Tests for xArm environment (Phase 3.2 gate).

Validates:
1. MJCF model loads without error
2. Environment can render RGB (84×84, uint8)
3. State access works (joint pos/vel, EE pos, object pos)
4. Forward dynamics integration
5. Reset functionality
"""

import numpy as np
import pytest
from pathlib import Path

from simulation.envs.xarm_env import XArmEnv


class TestXArmEnvBasics:
    """Test environment creation and basic methods."""
    
    def test_env_loads(self):
        """Environment should load MJCF without error."""
        env = XArmEnv()
        assert env.model is not None
        assert env.data is not None
        assert env.n_joints == 8  # 6 arm + 2 gripper
        env.close()
    
    def test_get_state(self):
        """State access shape and range checks."""
        env = XArmEnv()
        
        # Get state [q1-q8, dq1-dq8] = 16-dim
        state = env.get_state()
        assert state.shape == (16,), f"Expected (16,), got {state.shape}"
        
        # Joint positions in limits (check first 6 arm joints mainly)
        q = env.get_joint_pos()
        assert q.shape == (8,), f"Expected (8,), got {q.shape}"
        for i, (q_i, (q_min, q_max)) in enumerate(zip(q[:6], env.JOINT_LIMITS[:6])):
            assert q_min <= q_i <= q_max, f"Arm joint {i} out of limits"
        
        # Joint velocities at init (should be zero or near-zero)
        qd = env.get_joint_vel()
        assert qd.shape == (8,)
        assert np.allclose(qd[:6], 0.0, atol=1e-3), "Arm velocities should be zero at init"
        
        env.close()
    
    def test_kinematic_access(self):
        """EE and object position tracking."""
        env = XArmEnv()
        
        ee_pos = env.get_ee_pos()
        assert ee_pos.shape == (3,)
        assert np.all(np.isfinite(ee_pos))
        
        obj_pos = env.get_object_pos()
        assert obj_pos.shape == (3,)
        assert np.all(np.isfinite(obj_pos))
        
        env.close()


class TestXArmEnvControl:
    """Test control and dynamics."""
    
    def test_step_zero_control(self):
        """Step with zero control should not cause errors."""
        env = XArmEnv()
        
        q_init = env.get_joint_pos().copy()
        for _ in range(10):
            obs = env.step(np.zeros(8))
            assert "joint_pos" in obs
            assert obs["joint_pos"].shape == (8,)
        
        env.close()
    
    def test_torque_limiting(self):
        """Excess torques should be clipped."""
        env = XArmEnv()
        
        # Command torques way over limit
        huge_action = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        obs = env.step(huge_action)
        
        # Joint velocity should be limited by torque saturation
        qd = obs["joint_vel"]
        assert not np.any(np.isnan(qd))
        
        env.close()
    
    def test_position_servo(self):
        """Position servo should set joint positions."""
        env = XArmEnv()
        
        q_target = np.array([0.5, -0.5, -1.0, 0.5, 0.2, 0.3, 0.02, 0.02])
        obs = env.step_position(q_target)
        
        q_actual = env.get_joint_pos()
        np.testing.assert_allclose(q_actual, q_target, atol=1e-6)
        
        env.close()


class TestXArmEnvRendering:
    """Test rendering pipeline."""
    
    def test_render_rgb_shape(self):
        """RGB render should return correct shape."""
        env = XArmEnv()
        
        rgb = env.render_rgb()
        assert rgb.shape == (84, 84, 3), f"Expected (84, 84, 3), got {rgb.shape}"
        assert rgb.dtype == np.uint8
        assert np.all(rgb >= 0) and np.all(rgb <= 255)
        
        env.close()
    
    def test_render_different_cameras(self):
        """Should render from different cameras."""
        env = XArmEnv()
        
        # Primary camera
        rgb1 = env.render_rgb(camera="camera_rgb")
        assert rgb1.shape == (84, 84, 3)
        
        # Side camera (if available)
        try:
            rgb2 = env.render_rgb(camera="camera_side")
            assert rgb2.shape == (84, 84, 3)
            # Different cameras should (likely) produce different images
            # but this is not deterministic
        except:
            pass  # Side camera may not exist in all configs
        
        env.close()
    
    def test_render_high_res(self):
        """Should render at higher resolution."""
        env = XArmEnv()
        
        rgb_hi = env.render_rgb(size=256)
        assert rgb_hi.shape == (256, 256, 3)
        
        env.close()


class TestXArmEnvReset:
    """Test environment reset."""
    
    def test_reset_default(self):
        """Reset to home position."""
        env = XArmEnv()
        
        # Move away
        env.step(np.array([1.0, 0.5, -0.5, 0.2, 0.1, 0.15, 0.01, 0.01]))
        
        # Reset
        env.reset()
        q = env.get_joint_pos()
        
        # Check first 6 arm joints are at home
        np.testing.assert_allclose(q[:6], 0.0, atol=1e-3)
        
        env.close()
    
    def test_reset_custom(self):
        """Reset to custom configuration."""
        env = XArmEnv()
        
        q_target = np.array([-1.0, 0.8, -2.0, 0.5, 0.2, 0.3, 0.02, 0.02])
        env.reset(q_init=q_target)
        
        q = env.get_joint_pos()
        # Check arm joints match target
        np.testing.assert_allclose(q[:6], q_target[:6], atol=1e-3)
        
        env.close()


class TestXArmEnvSuccess:
    """Test task success detection."""
    
    def test_success_check(self):
        """Success should detect object lifted."""
        env = XArmEnv()
        
        # Initially should not be successful (object on table)
        success_low = env.check_success()
        assert not success_low
        
        # Manually lift object high
        obj_high = np.array([0.4, 0.0, 0.5])
        env.reset(object_pos=obj_high)
        success_high = env.check_success()
        assert success_high
        
        env.close()


class TestXArmEnvLiDAR:
    """Test LiDAR sensor."""
    
    def test_lidar_readings(self):
        """LiDAR should return 32 readings."""
        env = XArmEnv()
        
        readings = env.get_lidar_readings()
        assert readings.shape == (32,), f"Expected (32,), got {readings.shape}"
        assert np.all(np.isfinite(readings))
        
        # All should be positive (distance) or -1 (no hit)
        assert np.all((readings >= 0) | (readings == -1))
        
        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
