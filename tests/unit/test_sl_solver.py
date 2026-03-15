"""Phase 2: Tests for QP solvers (SL + OSQP)."""
import pytest
import numpy as np


# Validation QP from the plan
# minimize    x₁² + x₂²           (P = 2I, q = 0)
# subject to  x₁ + x₂ = 1         (equality)
#             -5 ≤ x₁ ≤ 5        (box)
#             -5 ≤ x₂ ≤ 5
# 
# Analytic solution: x₁ = x₂ = 0.5,  obj = 0.5

@pytest.fixture
def validation_qp():
    """The standard validation QP."""
    P = np.array([[2., 0.], [0., 2.]])
    q = np.array([0., 0.])
    A = np.array([
        [1., 1.],   # row 0: equality x1+x2=1
        [1., 0.],   # row 1: x1 box
        [0., 1.],   # row 2: x2 box
    ])
    l = np.array([ 1., -5., -5.])
    u = np.array([ 1.,  5.,  5.])
    return P, q, A, l, u


def test_osqp_validation_qp(validation_qp):
    """Test OSQP on the validation QP."""
    from src.solver.osqp_solver import OSQPSolver
    P, q, A, l, u = validation_qp
    solver = OSQPSolver()
    x, info = solver.solve(P, q, A, l, u)
    
    # Solution should be near [0.5, 0.5]
    assert abs(x[0] - 0.5) < 0.05, f"x[0]={x[0]}"
    assert abs(x[1] - 0.5) < 0.05, f"x[1]={x[1]}"
    
    # Objective should be near 0.5
    obj_expected = 0.5
    assert abs(info['obj_val'] - obj_expected) < 0.1, f"obj={info['obj_val']}"
    
    # Constraint violation should be near zero
    assert info['constraint_viol'] < 0.01, f"viol={info['constraint_viol']}"
    
    # Status should be "optimal"
    assert info['status'] == "optimal", f"status={info['status']}"
    
    print(f"OSQP: time={info['solve_time_ms']:.1f}ms, obj={info['obj_val']:.4f}, viol={info['constraint_viol']:.6f}")


def test_sl_validation_qp(validation_qp):
    """Test SL solver on the validation QP."""
    from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
    P, q, A, l, u = validation_qp
    solver = StuartLandauLagrangeDirect(T_solve=3.0)
    x, info = solver.solve(P, q, A, l, u)
    
    # Solution should be reasonably close [0.5, 0.5]
    assert abs(x[0] - 0.5) < 0.1, f"x[0]={x[0]}"
    assert abs(x[1] - 0.5) < 0.1, f"x[1]={x[1]}"
    
    # Objective should be in the right ballpark
    obj_expected = 0.5
    assert abs(info['obj_val'] - obj_expected) < 0.5, f"obj={info['obj_val']}"
    
    # Constraint violation should be reasonable (SL is approximate)
    assert info['constraint_viol'] < 0.5, f"viol={info['constraint_viol']}"
    
    print(f"SL: time={info['solve_time_ms']:.1f}ms, obj={info['obj_val']:.4f}, viol={info['constraint_viol']:.6f}")


def test_osqp_name():
    """Test solver name."""
    from src.solver.osqp_solver import OSQPSolver
    solver = OSQPSolver()
    assert solver.name == "OSQP"


def test_sl_name():
    """Test solver name."""
    from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
    solver = StuartLandauLagrangeDirect()
    assert solver.name == "StuartLandauLagrange"


def test_osqp_returns_tuple():
    """Test return type."""
    from src.solver.osqp_solver import OSQPSolver
    P, q, A, l, u = (
        np.eye(2), np.zeros(2),
        np.eye(2), np.array([-1., -1.]), np.array([1., 1.])
    )
    solver = OSQPSolver()
    result = solver.solve(P, q, A, l, u)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    x, info = result
    assert isinstance(x, np.ndarray)
    assert isinstance(info, dict)
    assert 'solve_time_ms' in info
    assert 'obj_val' in info
    assert 'status' in info


def test_sl_returns_tuple():
    """Test return type."""
    from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
    P, q, A, l, u = (
        np.eye(2), np.zeros(2),
        np.eye(2), np.array([-1., -1.]), np.array([1., 1.])
    )
    solver = StuartLandauLagrangeDirect()
    result = solver.solve(P, q, A, l, u)
    
    assert isinstance(result, tuple)
    assert len(result) == 2
    x, info = result
    assert isinstance(x, np.ndarray)
    assert isinstance(info, dict)
    assert 'solve_time_ms' in info
    assert 'obj_val' in info
    assert 'status' in info
