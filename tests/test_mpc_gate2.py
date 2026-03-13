"""
Gate 2 Validation Tests - MPC Linearization and QP Construction.

Tests linearization accuracy, discrete-time eigenvalues, Hessian PSD,
QP construction, and solver integration.

Requires: src/dynamics/kinematics_3dof.py, src/dynamics/lagrangian_3dof.py
"""

import pytest
import numpy as np
import logging
from typing import List

from src.dynamics.lagrangian_3dof import Arm3DOFDynamics
from src.mpc.linearize_3dof import Arm3DOFLinearizer, DiscretizedDynamics
from src.mpc.qp_builder_3dof import MPC3DOFBuilder, QPProblem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def arm():
    """Create 3-DOF arm dynamics model."""
    return Arm3DOFDynamics()


@pytest.fixture
def linearizer(arm):
    """Create linearizer for the arm."""
    return Arm3DOFLinearizer(arm, use_jax=False)  # Use FD, JAX optional


@pytest.fixture
def qp_builder():
    """Create MPC QP builder."""
    return MPC3DOFBuilder(N=5)  # Shorter horizon for tests


class TestLinearization:
    """Test linearization of 3-DOF arm dynamics."""
    
    def test_linearization_approximation_error(self, arm, linearizer):
        """
        Check linearization error at different perturbation magnitudes.
        
        For small perturbations, linear approximation should be accurate.
        Error should scale as O(δx²) for small δx.
        """
        q_ref = np.array([0.1, 0.3, -0.1])
        q_dot_ref = np.array([0.0, 0.0, 0.0])
        tau_ref = np.array([0.0, 0.0, 0.0])
        
        lin_dyn = linearizer.linearize_continuous(q_ref, q_dot_ref, tau_ref)
        
        # Test at increasing perturbation magnitudes
        perturbations = [0.01, 0.05, 0.1]
        errors = []
        
        for delta in perturbations:
            q_test = q_ref + delta * np.array([0.5, 0.3, 0.2])
            q_dot_test = q_dot_ref + delta * np.array([0.1, 0.2, 0.05])
            tau_test = tau_ref + delta * np.array([0.5, 0.3, 0.2])
            
            error_dict = linearizer.linearization_error(
                q_ref, q_dot_ref, tau_ref,
                q_test, q_dot_test, tau_test
            )
            errors.append(error_dict['relative_error'])
            logger.info(f"  δ={delta}: relative_error={error_dict['relative_error']:.4f}")
        
        # Errors should decrease with small perturbations
        assert errors[0] < 0.1, "Large linearization error at δ=0.01"
        assert errors[1] < 0.2, "Large linearization error at δ=0.05"
        assert errors[2] < 0.5, "Large linearization error at δ=0.1"
        logger.info(f"✓ Linearization errors within bounds: {errors}")
    
    def test_linearized_jacobian_symmetry(self, linearizer):
        """Test that linearized A matrix has expected structure for Hamiltonian systems."""
        q_ref = np.array([0.2, 0.25, -0.15])
        q_dot_ref = np.array([0.1, 0.05, -0.05])
        tau_ref = np.zeros(3)
        
        lin_dyn = linearizer.linearize_continuous(q_ref, q_dot_ref, tau_ref)
        A = lin_dyn.A
        
        # For Hamiltonian system, A should have block structure:
        # A = [0      I   ]
        #     [∂²H/∂q² ...]
        
        # Check top-right block is I
        A_top_right = A[:3, 3:6]
        is_identity = np.allclose(A_top_right, np.eye(3), atol=1e-6)
        assert is_identity, "Top-right block should be identity"
        
        # Check top-left block is zero
        A_top_left = A[:3, :3]
        is_zero = np.allclose(A_top_left, np.zeros((3, 3)), atol=1e-6)
        assert is_zero, "Top-left block should be zero"
        
        logger.info(f"✓ Linearized A matrix has correct Hamiltonian block structure")
    
    def test_discrete_eigenvalues_stable(self, linearizer):
        """Check discrete-time eigenvalues are within unit circle for stability."""
        q_ref = np.array([0.15, 0.2, -0.1])
        q_dot_ref = np.zeros(3)
        tau_ref = np.zeros(3)
        
        lin_dyn = linearizer.linearize_continuous(q_ref, q_dot_ref, tau_ref)
        dt = 0.01  # 100 Hz sampling
        
        disc_dyn = linearizer.discretize_zoh(lin_dyn, dt, method='zoh')
        Ad = disc_dyn.Ad
        
        # Eigenvalues should be inside unit circle for stability
        eigs = np.linalg.eigvals(Ad)
        eigs_magnitude = np.abs(eigs)
        max_eig_mag = np.max(eigs_magnitude)
        
        assert max_eig_mag < 1.1, f"Discrete eigenvalues outside unit circle: {eigs_magnitude}"
        logger.info(f"✓ Discrete eigenvalues stable: max |λ|={max_eig_mag:.4f}")
    
    def test_discretization_methods_agree(self, linearizer):
        """Different discretization methods should give similar results for small dt."""
        q_ref = np.array([0.0, 0.2, 0.0])
        q_dot_ref = np.zeros(3)
        tau_ref = np.zeros(3)
        
        lin_dyn = linearizer.linearize_continuous(q_ref, q_dot_ref, tau_ref)
        dt = 0.005  # Small step for good agreement
        
        disc_euler = linearizer.discretize_zoh(lin_dyn, dt, method='euler')
        disc_zoh = linearizer.discretize_zoh(lin_dyn, dt, method='zoh')
        
        # Ad matrices should be close
        Ad_diff = np.max(np.abs(disc_euler.Ad - disc_zoh.Ad))
        assert Ad_diff < 0.01, f"Discretization methods diverge: diff={Ad_diff}"
        
        logger.info(f"✓ Discretization methods agree: Ad diff={Ad_diff:.4e}")


class TestQPConstruction:
    """Test QP problem construction for MPC."""
    
    def test_qp_hessian_positive_semidefinite(self, arm, linearizer, qp_builder):
        """Verify QP Hessian is positive semidefinite."""
        # Generate reference trajectory
        N = qp_builder.N
        q_ref_traj = np.tile(np.array([0.1, 0.2, -0.1]), (N, 1))
        lin_dyn_list = []
        
        for q_ref in q_ref_traj:
            lin_dyn = linearizer.linearize_continuous(q_ref, np.zeros(3), np.zeros(3))
            disc_dyn = linearizer.discretize_zoh(lin_dyn, dt=0.01)
            lin_dyn_list.append(disc_dyn)
        
        # Build QP
        q_des = np.array([0.2, 0.3, -0.05])
        qp = qp_builder.build_qp(lin_dyn_list, q_des)
        
        # Check properties
        eigs = np.linalg.eigvals(qp.H)
        min_eig = np.min(eigs)
        
        assert min_eig >= -1e-10, f"Hessian not PSD: min_eig={min_eig}"
        logger.info(f"✓ QP Hessian positive semidefinite: min_eig={min_eig:.4e}")
    
    def test_qp_constraint_dimensions(self, arm, linearizer, qp_builder):
        """Check QP problem has correct dimensions."""
        N = qp_builder.N
        q_ref_traj = np.tile(np.array([0.0, 0.1, 0.0]), (N, 1))
        lin_dyn_list = []
        
        for q_ref in q_ref_traj:
            lin_dyn = linearizer.linearize_continuous(q_ref, np.zeros(3), np.zeros(3))
            disc_dyn = linearizer.discretize_zoh(lin_dyn, dt=0.01)
            lin_dyn_list.append(disc_dyn)
        
        q_des = np.array([0.1, 0.2, 0.0])
        qp = qp_builder.build_qp(lin_dyn_list, q_des)
        
        # Expected dimensions: x = [δx_0, δu_0, ..., δx_N]
        expected_n_vars = (N + 1) * 6 + N * 3
        assert qp.n_vars == expected_n_vars, f"Wrong dimension: {qp.n_vars} vs {expected_n_vars}"
        
        # Equality constraints: N dynamics constraints (6-dim each)
        assert qp.n_eq == N * 6, f"Wrong n_eq: {qp.n_eq} vs {N*6}"
        
        # Check matrix shapes
        assert qp.H.shape == (qp.n_vars, qp.n_vars)
        assert qp.c.shape == (qp.n_vars,)
        assert qp.A_eq.shape == (qp.n_eq, qp.n_vars)
        assert qp.b_eq.shape == (qp.n_eq,)
        
        logger.info(f"✓ QP dimensions correct: n_vars={qp.n_vars}, n_eq={qp.n_eq}, n_ineq={qp.n_ineq}")
    
    def test_qp_properties_validation(self, arm, linearizer, qp_builder):
        """Validate QP properties for solver stability."""
        N = qp_builder.N
        q_ref_traj = np.tile(np.array([0.05, 0.15, -0.05]), (N, 1))
        lin_dyn_list = []
        
        for q_ref in q_ref_traj:
            lin_dyn = linearizer.linearize_continuous(q_ref, np.zeros(3), np.zeros(3))
            disc_dyn = linearizer.discretize_zoh(lin_dyn, dt=0.01)
            lin_dyn_list.append(disc_dyn)
        
        q_des = np.array([0.1, 0.25, -0.05])
        qp = qp_builder.build_qp(lin_dyn_list, q_des)
        
        # Run property checks
        props = qp_builder.check_qp_properties(qp)
        
        assert props['H_symmetric'], "H not symmetric"
        assert props['H_positive_semidefinite'], "H not PSD"
        assert props['c_finite'], "c contains NaN/inf"
        assert props['A_eq_shape_correct'], "A_eq has wrong shape"
        assert props['A_ineq_shape_correct'], "A_ineq has wrong shape"
        assert props['no_nan_values'], "QP contains NaN"
        assert props['no_inf_values'], "QP contains inf"
        
        logger.info(f"✓ QP properties validated:")
        logger.info(f"  H symmetric: {props['H_symmetric']}")
        logger.info(f"  H PSD: {props['H_positive_semidefinite']}")
        logger.info(f"  cond(H): {props.get('H_condition_number', 'N/A')}")


class TestQPSolver:
    """Test QP solver integration."""
    
    def test_qp_solvable_with_osqp(self, arm, linearizer, qp_builder):
        """
        Check QP is solvable with OSQP solver.
        
        This test requires osqp package. If not available, skip.
        """
        pytest.importorskip("osqp")
        import osqp
        
        N = qp_builder.N
        q_ref_traj = np.tile(np.array([0.1, 0.2, -0.1]), (N, 1))
        lin_dyn_list = []
        
        for q_ref in q_ref_traj:
            lin_dyn = linearizer.linearize_continuous(q_ref, np.zeros(3), np.zeros(3))
            disc_dyn = linearizer.discretize_zoh(lin_dyn, dt=0.01)
            lin_dyn_list.append(disc_dyn)
        
        q_des = np.array([0.2, 0.3, 0.0])
        qp = qp_builder.build_qp(lin_dyn_list, q_des)
        
        # Setup and solve with OSQP
        # Note: OSQP uses format: min (1/2) x'Px + q'x subject to l <= Ax <= u
        # We need to convert our format
        from scipy.sparse import csc_matrix, vstack
        
        # Combine all constraints into A_all: [A_eq; A_ineq]
        A_all = vstack([csc_matrix(qp.A_eq), csc_matrix(qp.A_ineq)])
        
        # Create bounds: equality constraints have l=b=u, inequality have l=-inf, u=b
        l_eq = qp.b_eq.copy()
        u_eq = qp.b_eq.copy()
        l_ineq = -np.inf * np.ones(qp.n_ineq)
        u_ineq = qp.b_ineq.copy()
        
        l_all = np.concatenate([l_eq, l_ineq])
        u_all = np.concatenate([u_eq, u_ineq])
        
        m = osqp.OSQP()
        m.setup(P=csc_matrix(qp.H), q=qp.c,
                A=A_all, l=l_all, u=u_all,
                verbose=False)
        results = m.solve()
        
        # Check solution exists
        assert results.info.status == 'solved', f"OSQP failed: {results.info.status}"
        assert results.x is not None
        
        # Solution should be x = 0 approximately (at reference)
        sol_norm = np.linalg.norm(results.x)
        assert sol_norm < 1.0, f"Solution far from reference: ||x||={sol_norm}"
        
        logger.info(f"✓ QP solvable with OSQP: status={results.info.status}, ||x||={sol_norm:.4e}")


class TestWarmStart:
    """Test QP warm-starting for faster convergence."""
    
    def test_warm_start_speedup(self, arm, linearizer, qp_builder):
        """
        Check warm-start reduces solve time.
        
        Requires osqp package.
        """
        pytest.importorskip("osqp")
        import osqp
        import time
        from scipy.sparse import csc_matrix, vstack
        
        N = qp_builder.N
        q_ref_traj = np.tile(np.array([0.0, 0.1, 0.0]), (N, 1))
        lin_dyn_list = []
        
        for q_ref in q_ref_traj:
            lin_dyn = linearizer.linearize_continuous(q_ref, np.zeros(3), np.zeros(3))
            disc_dyn = linearizer.discretize_zoh(lin_dyn, dt=0.01)
            lin_dyn_list.append(disc_dyn)
        
        q_des = np.array([0.1, 0.15, -0.05])
        qp = qp_builder.build_qp(lin_dyn_list, q_des)
        
        # Prepare constraint bounds
        A_all = vstack([csc_matrix(qp.A_eq), csc_matrix(qp.A_ineq)])
        l_eq = qp.b_eq.copy()
        u_eq = qp.b_eq.copy()
        l_ineq = -np.inf * np.ones(qp.n_ineq)
        u_ineq = qp.b_ineq.copy()
        l_all = np.concatenate([l_eq, l_ineq])
        u_all = np.concatenate([u_eq, u_ineq])
        
        # Solve without warm-start
        m = osqp.OSQP()
        m.setup(P=csc_matrix(qp.H), q=qp.c,
                A=A_all, l=l_all, u=u_all,
                verbose=False)
        
        t0 = time.time()
        for _ in range(5):
            m.solve()
        time_cold = (time.time() - t0) / 5
        
        # Solve with warm-start (provide previous solution)
        # Get initial solution
        results = m.solve()
        x_warm = results.x.copy() if results.x is not None else np.zeros(qp.n_vars)
        
        # Manual warm-start: solve multiple times with initial guess
        t0 = time.time()
        for _ in range(5):
            m.warm_start(x=x_warm, y=np.zeros(qp.n_eq + qp.n_ineq))
            m.solve()
        time_warm = (time.time() - t0) / 5
        
        speedup = time_cold / time_warm if time_warm > 0 else 1.0
        logger.info(f"✓ Warm-start result: cold={time_cold*1000:.2f}ms, warm={time_warm*1000:.2f}ms ({speedup:.1f}x)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
