"""Phase 1: Tests for core abstractions."""
import pytest
import yaml
import numpy as np


def test_base_solver_import():
    """Test that BaseQPSolver can be imported."""
    from src.core.base_solver import BaseQPSolver
    assert BaseQPSolver is not None


def test_base_controller_import():
    """Test that BaseController can be imported."""
    from src.core.base_controller import BaseController
    assert BaseController is not None


def test_base_env_import():
    """Test that BaseEnv can be imported."""
    from src.core.base_env import BaseEnv
    assert BaseEnv is not None


def test_xarm_config_loads():
    """Test that xarm_6dof.yaml loads correctly."""
    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)
    
    assert cfg is not None
    assert "robot" in cfg
    robot = cfg["robot"]
    
    assert robot["name"] == "xarm_6dof"
    assert robot["n_joints"] == 6
    assert robot["n_gripper"] == 2
    assert robot["n_total_dof"] == 8
    
    assert len(robot["joint_limits"]["q_min"]) == 6
    assert len(robot["joint_limits"]["q_max"]) == 6
    assert len(robot["velocity_limits"]["qdot_max"]) == 6
    assert len(robot["torque_limits"]["tau_max"]) == 8
    
    assert len(robot["dynamics"]["link_masses"]) == 6
    assert len(robot["dynamics"]["link_lengths"]) == 6
    assert robot["dynamics"]["gravity"] == 9.81


def test_base_solver_is_abstract():
    """Test that BaseQPSolver cannot be instantiated directly."""
    from src.core.base_solver import BaseQPSolver
    with pytest.raises(TypeError):
        BaseQPSolver()


def test_base_controller_is_abstract():
    """Test that BaseController cannot be instantiated directly."""
    from src.core.base_controller import BaseController
    with pytest.raises(TypeError):
        BaseController()


def test_base_env_is_abstract():
    """Test that BaseEnv cannot be instantiated directly."""
    from src.core.base_env import BaseEnv
    with pytest.raises(TypeError):
        BaseEnv()
