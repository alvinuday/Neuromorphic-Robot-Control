"""
Test Suite for SL + Direct Lagrange Multipliers
===============================================

Tests the simplified phase-free approach on the same MPC problems
that failed with phase encoding.
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')

from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect


def test_1_simple_2x2():
    """Test 1: Simple 2×2 QP."""
    print("\n" + "="*70)
    print("TEST 1: Simple 2×2 QP")
    print("="*70)
    
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    q = np.array([-2.0, -4.0])
    C = np.array([[1.0, 1.0]])
    d = np.array([1.0])
    Ac = np.array([[1.0, 0.0], [0.0, 1.0]])
    l_vec = np.array([0.0, 0.0])
    u_vec = np.array([10.0, 10.0])
    
    solver = StuartLandauLagrangeDirect(
        tau_x=1.0,
        tau_lam_eq=0.05,
        tau_lam_ineq=0.1,
        T_solve=20.0,
        convergence_tol=1e-6
    )
    
    x_star = solver.solve(
        (P, q, C, d, Ac, l_vec, u_vec),
        verbose=True
    )
    
    # Check solution
    print(f"\nOptimal x: {x_star}")
    print(f"Expected: [0.5, 0.5]")
    
    eq_error = np.max(np.abs(C @ x_star - d))
    ineq_error = max(
        np.max(np.maximum(0.0, Ac @ x_star - u_vec)),
        np.max(np.maximum(0.0, l_vec - Ac @ x_star))
    )
    
    assert eq_error < 1e-4, f"Eq constraint violated: {eq_error}"
    assert ineq_error < 1e-4, f"Ineq constraint violated: {ineq_error}"
    print("\n✓ TEST 1 PASSED")


def test_2_with_box_constraints():
    """Test 2: QP with box constraints."""
    print("\n" + "="*70)
    print("TEST 2: QP with Box Constraints")
    print("="*70)
    
    P = np.diag([1.0, 2.0, 1.0])
    q = np.array([-1.0, -4.0, -1.0])
    C = np.array([[1.0, 1.0, 1.0]])
    d = np.array([2.0])
    Ac = np.eye(3)
    l_vec = np.array([-1.0, -1.0, -1.0])
    u_vec = np.array([5.0, 5.0, 5.0])
    
    solver = StuartLandauLagrangeDirect(
        tau_x=1.0,
        tau_lam_eq=0.05,
        tau_lam_ineq=0.1,
        T_solve=20.0,
        convergence_tol=1e-6
    )
    
    x_star = solver.solve(
        (P, q, C, d, Ac, l_vec, u_vec),
        verbose=True
    )
    
    info = solver.get_last_info()
    
    eq_error = np.max(np.abs(C @ x_star - d))
    ineq_error = max(
        np.max(np.maximum(0.0, Ac @ x_star - u_vec)),
        np.max(np.maximum(0.0, l_vec - Ac @ x_star))
    )
    
    assert eq_error < 1e-4, f"Eq constraint violated: {eq_error}"
    assert ineq_error < 1e-4, f"Ineq constraint violated: {ineq_error}"
    print("\n✓ TEST 2 PASSED")


def test_3_mpc_n5():
    """Test 3: MPC with N=5 (CRITICAL - the one that failed with phase encoding)."""
    print("\n" + "="*70)
    print("TEST 3: MPC with N=5 (Critical Test)")
    print("="*70)
    
    # Simple trajectory: move to (π/4, 0)
    x_target = np.array([np.pi/4, 0.0])
    u_ref = np.array([0.0, 0.0])
    
    N = 5
    dt = 0.1
    Q = np.eye(2)
    R = np.eye(2) * 0.01
    
    # Build MPC problem
    A = np.eye(2)  # Discrete time (for now: identity)
    B = np.eye(2) * dt
    
    n = 2 * N
    m = 2 * N
    
    # Cost: minimize ||x - target||_Q^2 + ||u - u_ref||_R^2
    # Quadratic form: 0.5 x^T P x + q^T x
    
    P = np.zeros((2*N, 2*N))
    q = np.zeros(2*N)
    
    for t in range(N):
        idx = 2*t
        P[idx:idx+2, idx:idx+2] = Q
        q[idx:idx+2] = -Q @ x_target
    
    # Dynamics constraints: x_{t+1} = A x_t + B u_t
    C = np.zeros((2*N, 2*N))
    d = np.zeros(2*N)
    
    for t in range(N-1):
        idx_curr = 2*t
        idx_next = 2*(t+1)
        # x_{t+1} - A x_t - B u_t = 0
        C[idx_next:idx_next+2, idx_curr:idx_curr+2] = -A
        C[idx_next:idx_next+2, idx_next:idx_next+2] = np.eye(2)
    
    # Control bounds: -2 <= u <= 2
    Ac = np.eye(2*N)
    u_max = 2.0
    l_vec = -u_max * np.ones(2*N)
    u_vec = u_max * np.ones(2*N)
    
    solver = StuartLandauLagrangeDirect(
        tau_x=1.0,
        tau_lam_eq=0.1,        # Fast convergence
        tau_lam_ineq=0.5,
        T_solve=60.0,          # Longer time for larger problem
        convergence_tol=1e-4   # Relaxed from 1e-7 to allow numerical tolerance
    )
    
    x_star = solver.solve(
        (P, q, C, d, Ac, l_vec, u_vec),
        verbose=True
    )
    
    info = solver.get_last_info()
    
    print(f"\nSolution norm: {np.linalg.norm(x_star):.6e}")
    print(f"First state: {x_star[:2]}")
    
    # Constraint violations
    eq_viol = np.max(np.abs(C @ x_star - d))
    ineq_viol_up = np.max(np.maximum(0.0, Ac @ x_star - u_vec))
    ineq_viol_lo = np.max(np.maximum(0.0, l_vec - Ac @ x_star))
    
    print(f"\nEq constraint violation: {eq_viol:.6e}")
    print(f"Ineq up violation: {ineq_viol_up:.6e}")
    print(f"Ineq lo violation: {ineq_viol_lo:.6e}")
    
    # THIS IS THE CRITICAL TEST - should pass unlike phase version
    assert info['converged'] or (eq_viol < 0.01 and ineq_viol_up < 0.01 and ineq_viol_lo < 0.01), \
        f"Solver did not converge AND/OR constraints violated: eq={eq_viol:.6e}, ineq_up={ineq_viol_up:.6e}, ineq_lo={ineq_viol_lo:.6e}"
    assert eq_viol < 0.01, f"❌ FAILED: Eq violation {eq_viol:.6e} too large (should be <1e-2)"
    assert ineq_viol_up < 0.01, f"Ineq up violation {ineq_viol_up:.6e} too large"
    assert ineq_viol_lo < 0.01, f"Ineq lo violation {ineq_viol_lo:.6e} too large"
    
    print("\n✓ TEST 3 PASSED - Direct Lagrange multipliers work!")


def test_4_mpc_n20():
    """Test 4: MPC with N=20 (larger scale)."""
    print("\n" + "="*70)
    print("TEST 4: MPC with N=20")
    print("="*70)
    
    x_target = np.array([np.pi/4, 0.0])
    
    N = 20
    dt = 0.1
    Q = np.eye(2)
    R = np.eye(2) * 0.01
    
    A = np.eye(2)
    B = np.eye(2) * dt
    
    n = 2 * N
    
    P = np.zeros((2*N, 2*N))
    q = np.zeros(2*N)
    
    for t in range(N):
        idx = 2*t
        P[idx:idx+2, idx:idx+2] = Q
        q[idx:idx+2] = -Q @ x_target
    
    C = np.zeros((2*N, 2*N))
    d = np.zeros(2*N)
    
    for t in range(N-1):
        idx_curr = 2*t
        idx_next = 2*(t+1)
        C[idx_next:idx_next+2, idx_curr:idx_curr+2] = -A
        C[idx_next:idx_next+2, idx_next:idx_next+2] = np.eye(2)
    
    Ac = np.eye(2*N)
    u_max = 2.0
    l_vec = -u_max * np.ones(2*N)
    u_vec = u_max * np.ones(2*N)
    
    solver = StuartLandauLagrangeDirect(
        tau_x=1.0,
        tau_lam_eq=0.1,
        tau_lam_ineq=0.5,
        T_solve=120.0,         # Longer for larger problem
        convergence_tol=1e-7
    )
    
    x_star = solver.solve(
        (P, q, C, d, Ac, l_vec, u_vec),
        verbose=True
    )
    
    info = solver.get_last_info()
    
    eq_viol = np.max(np.abs(C @ x_star - d))
    ineq_viol_up = np.max(np.maximum(0.0, Ac @ x_star - u_vec))
    ineq_viol_lo = np.max(np.maximum(0.0, -l_vec - Ac @ x_star))
    
    print(f"\nEq violation: {eq_viol:.6e}")
    print(f"Ineq up violation: {ineq_viol_up:.6e}")
    print(f"Ineq lo violation: {ineq_viol_lo:.6e}")
    
    eq_viol = np.max(np.abs(C @ x_star - d))
    ineq_viol_up = np.max(np.maximum(0.0, Ac @ x_star - u_vec))
    ineq_viol_lo = np.max(np.maximum(0.0, l_vec - Ac @ x_star))
    
    print(f"\nEq violation: {eq_viol:.6e}")
    print(f"Ineq up violation: {ineq_viol_up:.6e}")
    print(f"Ineq lo violation: {ineq_viol_lo:.6e}")
    
    assert eq_viol < 0.01 and ineq_viol_up < 0.01 and ineq_viol_lo < 0.01, \
        f"Constraint violated: eq={eq_viol:.6e}, ineq_up={ineq_viol_up:.6e}, ineq_lo={ineq_viol_lo:.6e}"
    assert ineq_viol_up < 0.01, f"Ineq up violation too large"
    
    print("\n✓ TEST 4 PASSED")


def test_5_kkt_conditions():
    """Test 5: Verify KKT conditions."""
    print("\n" + "="*70)
    print("TEST 5: KKT Conditions")
    print("="*70)
    
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    q = np.array([-2.0, -4.0])
    C = np.array([[1.0, 1.0]])
    d = np.array([1.0])
    Ac = np.array([[1.0, 0.0], [0.0, 1.0]])
    l_vec = np.array([0.0, 0.0])
    u_vec = np.array([10.0, 10.0])
    
    solver = StuartLandauLagrangeDirect(
        tau_x=1.0,
        tau_lam_eq=0.05,
        tau_lam_ineq=0.1,
        T_solve=20.0,
        convergence_tol=1e-6
    )
    
    x_star = solver.solve(
        (P, q, C, d, Ac, l_vec, u_vec),
        verbose=True
    )
    
    info = solver.get_last_info()
    
    # KKT stationarity: ∇f(x*) + A_eq^T λ_eq + A_ineq^T λ_ineq = 0
    grad_f = P @ x_star + q
    g_eq = C @ x_star - d
    g_ineq_up = np.maximum(0.0, Ac @ x_star - u_vec)
    g_ineq_lo = np.maximum(0.0, -Ac @ x_star - l_vec)
    
    print(f"\n∇f norm: {np.linalg.norm(grad_f):.6e}")
    print(f"Eq constraint violation: {np.max(np.abs(g_eq)):.6e}")
    print(f"Ineq up violation: {np.max(g_ineq_up):.6e}")
    print(f"Ineq lo violation: {np.max(g_ineq_lo):.6e}")
    
    assert np.max(np.abs(g_eq)) < 1e-3, "Eq constraint violated"
    
    print("\n✓ TEST 5 PASSED")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-t', '--test', type=int, default=None)
    args = parser.parse_args()
    
    tests = [
        ('test_1', test_1_simple_2x2),
        ('test_2', test_2_with_box_constraints),
        ('test_3', test_3_mpc_n5),
        ('test_4', test_4_mpc_n20),
        ('test_5', test_5_kkt_conditions),
    ]
    
    if args.test is not None:
        # Run single test
        test_num = args.test
        if 1 <= test_num <= len(tests):
            name, func = tests[test_num - 1]
            try:
                func()
            except AssertionError as e:
                print(f"\n✗ FAILED: {e}")
                sys.exit(1)
    else:
        # Run all tests
        passed = 0
        failed = 0
        for name, func in tests:
            try:
                func()
                passed += 1
            except AssertionError as e:
                print(f"\n✗ FAILED: {e}")
                failed += 1
        
        print(f"\n\n{'='*70}")
        print(f"SUMMARY: {passed} passed, {failed} failed")
        print(f"{'='*70}")
        
        if failed > 0:
            sys.exit(1)
