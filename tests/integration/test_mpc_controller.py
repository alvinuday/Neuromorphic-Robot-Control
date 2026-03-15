"""Tests for Phase 4: MPC Controller."""
import numpy as np
import pytest
import yaml
from src.mpc.xarm_mpc_controller import XArmMPCController
from src.solver.osqp_solver import OSQPSolver


@pytest.fixture
def config():
    with open("config/robots/xarm_6dof.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def mpc(config):
    return XArmMPCController(solver=OSQPSolver(), robot_config=config, dt=0.01)


def test_output_shape(mpc):
    """Test that MPC output is [8] shape (6 arm + 2 gripper)."""
    q = np.zeros(6)
    qdot = np.zeros(6)
    ref = np.zeros((10, 6))
    tau = mpc.step((q, qdot), ref)
    assert tau.shape == (8,), f"Expected (8,), got {tau.shape}"


def test_output_within_limits(mpc, config):
    """Test that torques stay within limits."""
    q = np.zeros(6)
    qdot = np.zeros(6)
    ref = np.ones((10, 6))
    tau = mpc.step((q, qdot), ref)
    tau_max = config['robot']['torque_limits']['tau_max']
    
    for i in range(6):
        assert abs(tau[i]) <= tau_max[i] + 1e-3, \
            f"Joint {i} torque {tau[i]:.4f} exceeds limit {tau_max[i]}"


def test_positive_tracking_direction(mpc):
    """Positive reference error on joint 0 must yield positive torque on joint 0."""
    q = np.zeros(6)
    qdot = np.zeros(6)
    ref = np.zeros((10, 6))
    ref[:, 0] = 0.5    # joint 0 reference = +0.5 rad, current = 0
    tau = mpc.step((q, qdot), ref)
    assert tau[0] > 0, f"Expected positive torque toward +ref, got {tau[0]:.4f}"


def test_zero_error_near_zero_torque(mpc):
    """When reference == current, torque should be small (gravity comp only)."""
    q = np.array([0., 0., 0., 0., 0., 0.])
    ref = np.tile(q, (10, 1))
    tau = mpc.step((q, np.zeros(6)), ref)
    # Allow some gravity compensation torque
    assert np.abs(tau[:6]).max() < 15.0, \
        f"Expected small torques at zero error, got max {np.abs(tau[:6]).max():.2f}"


def test_qp_matrices_available(mpc):
    """Test that QP matrices are cached and accessible."""
    mpc.step((np.zeros(6), np.zeros(6)), np.zeros((10, 6)))
    qp = mpc.get_last_qp_matrices()
    for key in ('P', 'q', 'A', 'l', 'u', 'solution', 'info'):
        assert key in qp, f"Missing key: {key}"
        
        
def test_reference_vector_handling(mpc):
    """Test MPC handles both [6] and [N,6] reference formats."""
    q = np.zeros(6)
    qdot = np.zeros(6)
    
    # Test with [6] reference
    ref_vec = np.ones(6)
    tau1 = mpc.step((q, qdot), ref_vec)
    assert tau1.shape == (8,)
    
    # Test with [N, 6] reference
    ref_mat = np.ones((10, 6))
    tau2 = mpc.step((q, qdot), ref_mat)
    assert tau2.shape == (8,)
    
    # Both should give the same output (same first row)
    np.testing.assert_allclose(tau1, tau2, atol=1e-6, 
                               err_msg="Reference formats should produce same output")


def test_reset_clears_state(mpc):
    """Test that reset() clears cached QP data."""
    mpc.step((np.zeros(6), np.zeros(6)), np.zeros((10, 6)))
    qp_before = mpc.get_last_qp_matrices()
    assert len(qp_before) > 0
    
    mpc.reset()
    qp_after = mpc.get_last_qp_matrices()
    assert len(qp_after) == 0
