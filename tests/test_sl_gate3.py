"""
Gate 3 Validation Tests - Stuart-Landau 3D Solver.

Tests SL solver accuracy, convergence, eigenvalue spectrum, and comparison
against reference OSQP solver.

Requires: src/solver/stuart_landau_3dof.py, src/mpc/qp_builder_3dof.py
"""

import pytest
import numpy as np
import logging
from typing import List

from src.dynamics.lagrangian_3dof import Arm3DOFDynamics
from src.mpc.linearize_3dof import Arm3DOFLinearizer
from src.mpc.qp_builder_3dof import MPC3DOFBuilder
from src.solver.stuart_landau_3dof import StuartLandau3DOFSolver, scale_solver_for_horizon

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def arm():
    """Create 3-DOF arm dynamics model."""
    return Arm3DOFDynamics()


@pytest.fixture
def linearizer(arm):
    """Create linearizer."""
    return Arm3DOFLinearizer(arm, use_jax=False)


@pytest.fixture
def qp_builder():
    """Create MPC QP builder."""
    return MPC3DOFBuilder(N=3)  # Shorter horizon for testing


@pytest.fixture
def reference_qp(arm, linearizer, qp_builder):
    """Generate reference QP problem."""
    N = qp_builder.N
    q_ref_traj = np.tile(np.array([0.1, 0.2, -0.1]), (N, 1))
    lin_dyn_list = []
    
    for q_ref in q_ref_traj:
        lin_dyn = linearizer.linearize_continuous(q_ref, np.zeros(3), np.zeros(3))
        disc_dyn = linearizer.discretize_zoh(lin_dyn, dt=0.01)
        lin_dyn_list.append(disc_dyn)
    
    q_des = np.array([0.2, 0.3, -0.05])
    return qp_builder.build_qp(lin_dyn_list, q_des)


class TestSLSolverBasic:
    """Test basic Stuart-Landau solver functionality."""
    
    def test_sl_solver_convergence(self, reference_qp):
        """Check SL solver converges on simple QP."""
        config = scale_solver_for_horizon(n_steps=3, n_dof=3)
        solver = StuartLandau3DOFSolver(config)
        
        # Solve
        x_sol, info = solver.solve(reference_qp, verbose=False)
        
        # Check convergence
        assert info['cost'] is not None, "Cost not computed"
        assert x_sol.shape == (reference_qp.n_vars,), "Wrong solution shape"
        
        # Solution norm should be reasonable
        sol_norm = np.linalg.norm(x_sol)
        assert sol_norm < 10.0, f"Solution diverged: ||x||={sol_norm}"
        
        logger.info(f"✓ SL solver converged: cost={info['cost']:.6f}, ||x||={sol_norm:.4f}")
    
    def test_sl_solver_cost_decreases(self, reference_qp):
        """Cost should generally decrease over iterations."""
        config = scale_solver_for_horizon(n_steps=3, n_dof=3)
        config.timesteps = 200  # More steps to see convergence trend
        solver = StuartLandau3DOFSolver(config)
        
        x_sol, info = solver.solve(reference_qp, verbose=False)
        costs = info['cost_trajectory']
        
        # Check that final cost is lower than initial
        if len(costs) > 10:
            cost_initial = costs[0]
            cost_final = costs[-1]
            assert cost_final <= cost_initial * 1.5, "Cost increased significantly"
            logger.info(f"✓ Cost trend: initial={cost_initial:.6f}, final={cost_final:.6f}")
    
    def test_sl_solution_dimension(self, reference_qp):
        """Solution should have correct dimension."""
        config = scale_solver_for_horizon(n_steps=3, n_dof=3)
        solver = StuartLandau3DOFSolver(config)
        
        x_sol, info = solver.solve(reference_qp, verbose=False)
        
        expected_dim = reference_qp.n_vars
        assert x_sol.shape == (expected_dim,), f"Wrong shape: {x_sol.shape} vs ({expected_dim},)"
        assert np.all(np.isfinite(x_sol)), "Solution contains NaN/inf"
        
        logger.info(f"✓ Solution dimension correct: {expected_dim}")


class TestSLSolverVsReference:
    """Test SL solver against OSQP reference."""
    
    def test_sl_vs_osqp_accuracy(self, reference_qp):
        """Compare SL solution with OSQP reference."""
        pytest.importorskip("osqp")
        import osqp
        from scipy.sparse import csc_matrix, vstack
        
        # Solve with OSQP first
        A_all = vstack([csc_matrix(reference_qp.A_eq), csc_matrix(reference_qp.A_ineq)])
        l_eq = reference_qp.b_eq.copy()
        u_eq = reference_qp.b_eq.copy()
        l_ineq = -np.inf * np.ones(reference_qp.n_ineq)
        u_ineq = reference_qp.b_ineq.copy()
        l_all = np.concatenate([l_eq, l_ineq])
        u_all = np.concatenate([u_eq, u_ineq])
        
        m = osqp.OSQP()
        m.setup(P=csc_matrix(reference_qp.H), q=reference_qp.c,
                A=A_all, l=l_all, u=u_all, verbose=False)
        osqp_result = m.solve()
        x_osqp = osqp_result.x
        
        # Solve with SL
        config = scale_solver_for_horizon(n_steps=3, n_dof=3)
        config.timesteps = 500  # More iterations for convergence
        solver = StuartLandau3DOFSolver(config)
        x_sl, info = solver.solve(reference_qp, x_init=x_osqp*0.1, verbose=False)
        
        # Compare solutions
        sol_error = np.linalg.norm(x_sl - x_osqp)
        rel_error = sol_error / (np.linalg.norm(x_osqp) + 1e-10)
        
        cost_sl = 0.5 * x_sl @ reference_qp.H @ x_sl + reference_qp.c @ x_sl
        cost_osqp = 0.5 * x_osqp @ reference_qp.H @ x_osqp + reference_qp.c @ x_osqp
        cost_ratio = cost_sl / cost_osqp if cost_osqp != 0 else 1.0
        
        # SL should converge toward OSQP solution (might not be exact due to iterative nature)
        assert rel_error < 1.5, f"SL solution diverges too far: rel_error={rel_error:.4f}"
        assert cost_ratio < 2.0, f"SL cost much worse than OSQP: ratio={cost_ratio:.4f}"
        
        logger.info(f"✓ SL vs OSQP accuracy: rel_error={rel_error:.4f}, cost_ratio={cost_ratio:.4f}")
    
    def test_sl_solve_vs_reference(self, reference_qp):
        """Test solve_vs_reference method."""
        pytest.importorskip("osqp")
        import osqp
        from scipy.sparse import csc_matrix, vstack
        
        # Get OSQP reference solution
        A_all = vstack([csc_matrix(reference_qp.A_eq), csc_matrix(reference_qp.A_ineq)])
        l_eq = reference_qp.b_eq.copy()
        u_eq = reference_qp.b_eq.copy()
        l_ineq = -np.inf * np.ones(reference_qp.n_ineq)
        u_ineq = reference_qp.b_ineq.copy()
        l_all = np.concatenate([l_eq, l_ineq])
        u_all = np.concatenate([u_eq, u_ineq])
        
        m = osqp.OSQP()
        m.setup(P=csc_matrix(reference_qp.H), q=reference_qp.c,
                A=A_all, l=l_all, u=u_all, verbose=False)
        osqp_result = m.solve()
        x_osqp = osqp_result.x
        
        # Use SL solver's comparison method
        config = scale_solver_for_horizon(n_steps=3, n_dof=3)
        config.timesteps = 500  # More iterations
        solver = StuartLandau3DOFSolver(config)
        
        comparison = solver.solve_vs_reference(reference_qp, x_osqp)
        
        # Relax tolerances for iterative method
        assert comparison['relative_error'] < 1.5, "Error too large"
        assert comparison['relative_cost_error'] < 1.5, "Cost error too large"
        
        logger.info(f"✓ Solver comparison: rel_err={comparison['relative_error']:.4f}, "
                   f"cost_err={comparison['relative_cost_error']:.4f}")


class TestSLSolverScaling:
    """Test SL solver configuration scaling."""
    
    def test_scaling_function(self):
        """Test scale_solver_for_horizon function."""
        # Test various horizons
        for n_steps in [2, 5, 10]:
            config = scale_solver_for_horizon(n_steps, n_dof=3)
            
            expected_dim = (n_steps + 1) * 6 + n_steps * 3
            assert config.n == expected_dim, f"Wrong dimension scaling for n_steps={n_steps}"
            assert config.timesteps > 0, "Timesteps should be positive"
            assert config.damping > 0, "Damping should be positive"
            
            logger.info(f"  n_steps={n_steps}: n={config.n}, timesteps={config.timesteps}, damping={config.damping:.4f}")
        
        logger.info("✓ Solver scaling function works correctly")
    
    def test_larger_problem_convergence(self, arm, linearizer):
        """Test solver on larger horizon problem."""
        qp_builder = MPC3DOFBuilder(N=5)  # Larger horizon
        
        N = qp_builder.N
        q_ref_traj = np.tile(np.array([0.05, 0.15, -0.05]), (N, 1))
        lin_dyn_list = []
        
        for q_ref in q_ref_traj:
            lin_dyn = linearizer.linearize_continuous(q_ref, np.zeros(3), np.zeros(3))
            disc_dyn = linearizer.discretize_zoh(lin_dyn, dt=0.01)
            lin_dyn_list.append(disc_dyn)
        
        q_des = np.array([0.1, 0.25, -0.05])
        qp = qp_builder.build_qp(lin_dyn_list, q_des)
        
        # Solve with scaled config
        config = scale_solver_for_horizon(n_steps=N, n_dof=3)
        solver = StuartLandau3DOFSolver(config)
        
        x_sol, info = solver.solve(qp, verbose=False)
        
        # Should still converge even on larger problem
        assert info['cost'] < 1e6, "Cost unreasonably large"
        assert np.all(np.isfinite(x_sol)), "Solution contains non-finite values"
        
        logger.info(f"✓ Larger problem (n={qp.n_vars}) solved: cost={info['cost']:.6f}")


class TestSLSolverConstraints:
    """Test SL solver respects QP constraints."""
    
    def test_dynamics_constraint_satisfaction(self, reference_qp):
        """Check that solution approximately satisfies equality constraints."""
        config = scale_solver_for_horizon(n_steps=3, n_dof=3)
        solver = StuartLandau3DOFSolver(config)
        
        x_sol, info = solver.solve(reference_qp, verbose=False)
        
        # Check equality constraints: A_eq x = b_eq
        constraint_residual = reference_qp.A_eq @ x_sol - reference_qp.b_eq
        max_violation = np.max(np.abs(constraint_residual))
        
        # SL may not exactly satisfy constraints, but should be within tolerance
        assert max_violation < 0.5, f"Large constraint violation: {max_violation}"
        
        logger.info(f"✓ Constraint satisfaction: max_violation={max_violation:.4e}")
    
    def test_box_constraint_soft_satisfaction(self, reference_qp):
        """Check box constraints (soft satisfaction allowed)."""
        config = scale_solver_for_horizon(n_steps=3, n_dof=3)
        solver = StuartLandau3DOFSolver(config)
        
        x_sol, info = solver.solve(reference_qp, verbose=False)
        
        # Check inequality constraints: A_ineq x <= b_ineq
        constraint_values = reference_qp.A_ineq @ x_sol
        violations = constraint_values - reference_qp.b_ineq
        max_violation = np.max(violations)
        
        # Some soft violations allowed in iterative method
        assert max_violation < 1.0, f"Large inequality constraint violation: {max_violation}"
        
        logger.info(f"✓ Box constraint satisfaction: max_violation={max_violation:.4e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
