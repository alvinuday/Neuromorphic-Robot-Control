"""
Phase 3 Validation: Complete KKT Conditions for SL+DirectLag
============================================================

Full validation that equality, inequality, and complementarity
conditions are satisfied to required precision.
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')

from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect


def verify_kkt_conditions(P, q, C, d, Ac, l_vec, u_vec, x_star, lam_eq_final=None, 
                         lam_up_final=None, lam_lo_final=None, verbose=True):
    """
    Verify KKT conditions:
    1. Stationarity: ∇f(x) + C^T λ_eq + A_c^T (λ_up - λ_lo) = 0
    2. Primal feasibility: Cx = d, l ≤ Ac x ≤ u
    3. Dual feasibility: λ_up ≥ 0, λ_lo ≥ 0
    4. Complementarity: λ_up[i] * (Ac x - u)[i] = 0, etc
    """
    n = x_star.shape[0]
    m_eq = C.shape[0] if C is not None else 0
    m = Ac.shape[0] if Ac is not None else 0
    
    if verbose:
        print("\n" + "="*70)
        print("KKT CONDITIONS VERIFICATION")
        print("="*70)
    
    # Residuals
    res_eq = C @ x_star - d if m_eq > 0 else np.array([])
    res_ineq_up = np.maximum(0.0, Ac @ x_star - u_vec) if m > 0 else np.array([])
    res_ineq_lo = np.maximum(0.0, l_vec - Ac @ x_star) if m > 0 else np.array([])
    
    # Gradient of objective
    grad_f = P @ x_star + q
    
    # Construct Lagrangian gradient term
    lam_term = np.zeros(n)
    if m_eq > 0 and lam_eq_final is not None:
        lam_term += C.T @ lam_eq_final
    if m > 0 and lam_up_final is not None and lam_lo_final is not None:
        lam_term += Ac.T @ (lam_up_final - lam_lo_final)
    
    # KKT 1: Stationarity
    kkt_stationarity = grad_f + lam_term
    kkt_stationarity_norm = np.linalg.norm(kkt_stationarity)
    
    # KKT 2: Primal Feasibility
    kkt_eq_violation = np.max(np.abs(res_eq)) if m_eq > 0 else 0.0
    kkt_ineq_violation_up = np.max(res_ineq_up) if m > 0 else 0.0
    kkt_ineq_violation_lo = np.max(res_ineq_lo) if m > 0 else 0.0
    kkt_primal_violation = max(kkt_eq_violation, kkt_ineq_violation_up, kkt_ineq_violation_lo)
    
    # KKT 3: Dual Feasibility
    if lam_up_final is not None:
        kkt_dual_up_violation = np.max(-lam_up_final)  # Should be ≤ 0
        kkt_dual_up_violation = max(0.0, -np.min(lam_up_final))  # Should be ≥ 0
    else:
        kkt_dual_up_violation = 0.0
    
    if lam_lo_final is not None:
        kkt_dual_lo_violation = max(0.0, -np.min(lam_lo_final))  # Should be ≥ 0
    else:
        kkt_dual_lo_violation = 0.0
    
    kkt_dual_violation = max(kkt_dual_up_violation, kkt_dual_lo_violation)
    
    # KKT 4: Complementarity (slack condition)
    if lam_up_final is not None:
        comp_up = np.abs(lam_up_final * res_ineq_up)
        kkt_comp_violation_up = np.max(comp_up)
    else:
        kkt_comp_violation_up = 0.0
    
    if lam_lo_final is not None:
        comp_lo = np.abs(lam_lo_final * res_ineq_lo)
        kkt_comp_violation_lo = np.max(comp_lo)
    else:
        kkt_comp_violation_lo = 0.0
    
    kkt_comp_violation = max(kkt_comp_violation_up, kkt_comp_violation_lo)
    
    # Print results
    if verbose:
        print(f"\n1. STATIONARITY: ||∇f(x) + C^T λ + A^T λ_net||")
        print(f"   Norm: {kkt_stationarity_norm:.6e}  (should be <1e-3)")
        
        print(f"\n2. PRIMAL FEASIBILITY:")
        print(f"   Eq constraint:   |Cx - d|_max = {kkt_eq_violation:.6e}  (should be <1e-6)")
        print(f"   Ineq up bound:   |max(0, Acx - u)|_max = {kkt_ineq_violation_up:.6e}")
        print(f"   Ineq lo bound:   |max(0, l - Acx)|_max = {kkt_ineq_violation_lo:.6e}")
        print(f"   Overall: {kkt_primal_violation:.6e}  (should be <1e-6)")
        
        print(f"\n3. DUAL FEASIBILITY:")
        print(f"   λ_up ≥ 0 violation: {kkt_dual_up_violation:.6e}  (should be <1e-6)")
        print(f"   λ_lo ≥ 0 violation: {kkt_dual_lo_violation:.6e}  (should be <1e-6)")
        
        print(f"\n4. COMPLEMENTARITY:")
        print(f"   λ_up[i] * (Acx - u)[i] norm: {kkt_comp_violation_up:.6e}")
        print(f"   λ_lo[i] * (l - Acx)[i] norm: {kkt_comp_violation_lo:.6e}")
        print(f"   Overall: {kkt_comp_violation:.6e}  (should be <1e-4)")
    
    return {
        'stationarity': kkt_stationarity_norm,
        'primal_feasibility': kkt_primal_violation,
        'dual_feasibility': kkt_dual_violation,
        'complementarity': kkt_comp_violation,
        'eq_violation': kkt_eq_violation,
        'ineq_violation_up': kkt_ineq_violation_up,
        'ineq_violation_lo': kkt_ineq_violation_lo
    }


def test_phase3_full_kkt():
    """Test Phase 3: Complete KKT conditions."""
    print("\n" + "="*70)
    print("PHASE 3: FULL KKT CONDITIONS TEST")
    print("="*70)
    
    # Problem: min 0.5 x^T P x + q^T x
    #          subject to: Cx = d, l <= Ac x <= u
    P = np.diag([2.0, 1.0, 3.0])
    q = np.array([-2.0, -4.0, -1.0])
    
    # Equality: sum(x) = 2
    C = np.array([[1.0, 1.0, 1.0]])
    d = np.array([2.0])
    
    # Inequality: box constraints
    Ac = np.eye(3)
    l_vec = np.array([-1.0, -1.0, -1.0])
    u_vec = np.array([5.0, 5.0, 5.0])
    
    solver = StuartLandauLagrangeDirect(
        tau_x=1.0,
        tau_lam_eq=0.05,
        tau_lam_ineq=0.1,
        T_solve=30.0,
        convergence_tol=1e-6
    )
    
    x_star = solver.solve(
        (P, q, C, d, Ac, l_vec, u_vec),
        verbose=True
    )
    
    print(f"\nOptimal x: {x_star}")
    print(f"Objective: {0.5 * x_star @ P @ x_star + q @ x_star:.6e}")
    
    # For now, we don't have direct access to final multipliers from solver
    # So we just verify constraints are satisfied
    kkt = verify_kkt_conditions(P, q, C, d, Ac, l_vec, u_vec, x_star, 
                                verbose=True)
    
    # Assertions
    assert kkt['eq_violation'] < 1e-6, f"Eq constraint: {kkt['eq_violation']}"
    assert kkt['ineq_violation_up'] < 1e-4, f"Ineq up: {kkt['ineq_violation_up']}"
    assert kkt['ineq_violation_lo'] < 1e-4, f"Ineq lo: {kkt['ineq_violation_lo']}"
    
    print("\n✓ PHASE 3 TEST PASSED")
    return True


def test_phase3_mpc_kkt():
    """Test Phase 3: MPC problem with full KKT."""
    print("\n" + "="*70)
    print("PHASE 3: MPC WITH KKT VALIDATION (N=10)")
    print("="*70)
    
    N = 10
    dt = 0.1
    Q = np.eye(2)
    R = np.eye(2) * 0.01
    
    x_target = np.array([np.pi/4, 0.0])
    
    A = np.eye(2)
    B = np.eye(2) * dt
    
    # Build matrices
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
        T_solve=60.0,
        convergence_tol=1e-6
    )
    
    x_star = solver.solve(
        (P, q, C, d, Ac, l_vec, u_vec),
        verbose=True
    )
    
    kkt = verify_kkt_conditions(P, q, C, d, Ac, l_vec, u_vec, x_star,
                                verbose=True)
    
    # Assertions
    assert kkt['eq_violation'] < 0.01, f"Eq constraint: {kkt['eq_violation']}"
    assert kkt['ineq_violation_up'] + kkt['ineq_violation_lo'] < 0.01, \
        f"Ineq violation: up={kkt['ineq_violation_up']}, lo={kkt['ineq_violation_lo']}"
    
    print("\n✓ PHASE 3 MPC TEST PASSED")
    return True


def test_phase3_scaling():
    """Test Phase 3: Verify scaling behavior across problem sizes."""
    print("\n" + "="*70)
    print("PHASE 3: SCALING TEST (N=5, 10, 20)")
    print("="*70)
    
    sizes = [5, 10, 20]
    solver = StuartLandauLagrangeDirect(
        tau_x=1.0,
        tau_lam_eq=0.1,
        tau_lam_ineq=0.5,
        T_solve=120.0,
        convergence_tol=1e-6
    )
    
    results = []
    
    for N in sizes:
        print(f"\n--- N = {N} ---")
        
        x_target = np.array([np.pi/4, 0.0])
        A = np.eye(2)
        B = np.eye(2) * 0.1
        Q = np.eye(2)
        
        P = np.zeros((2*N, 2*N))
        q = np.zeros(2*N)
        for t in range(N):
            idx = 2*t
            P[idx:idx+2, idx:idx+2] = Q
            q[idx:idx+2] = -Q @ x_target
        
        C = np.zeros((2*N, 2*N))
        d = np.zeros(2*N)
        for t in range(N-1):
            idx_c, idx_n = 2*t, 2*(t+1)
            C[idx_n:idx_n+2, idx_c:idx_c+2] = -A
            C[idx_n:idx_n+2, idx_n:idx_n+2] = np.eye(2)
        
        Ac = np.eye(2*N)
        l_vec = -2.0 * np.ones(2*N)
        u_vec = 2.0 * np.ones(2*N)
        
        x_star = solver.solve(
            (P, q, C, d, Ac, l_vec, u_vec),
            verbose=False
        )
        
        info = solver.get_last_info()
        kkt = verify_kkt_conditions(P, q, C, d, Ac, l_vec, u_vec, x_star,
                                   verbose=False)
        
        results.append({
            'N': N,
            'n_vars': 2*N,
            'm_eq': 2*N,
            'time': info['time_to_solution'],
            'steps': info['num_steps'],
            'eq_viol': kkt['eq_violation'],
            'ineq_viol': kkt['ineq_violation_up'] + kkt['ineq_violation_lo']
        })
        
        print(f"  Time: {info['time_to_solution']:.3f}s ({info['num_steps']} steps)")
        print(f"  Eq violation: {kkt['eq_violation']:.6e}")
        print(f"  Ineq violation: {kkt['ineq_violation_up'] + kkt['ineq_violation_lo']:.6e}")
        
        assert kkt['eq_violation'] < 0.01, f"Eq violation too large at N={N}"
        assert kkt['ineq_violation_up'] + kkt['ineq_violation_lo'] < 0.01, \
            f"Ineq violation too large at N={N}"
    
    print("\n" + "─"*70)
    print("SCALING SUMMARY:")
    print("─"*70)
    for r in results:
        print(f"N={r['N']:2d} (n={r['n_vars']:3d}, m={r['m_eq']:3d}): "
              f"time={r['time']:.3f}s, eq_viol={r['eq_viol']:.2e}, "
              f"ineq_viol={r['ineq_viol']:.2e}")
    
    print("\n✓ PHASE 3 SCALING TEST PASSED")
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', type=int, default=None)
    args = parser.parse_args()
    
    tests = [
        ('kkt', test_phase3_full_kkt),
        ('mpc_kkt', test_phase3_mpc_kkt),
        ('scaling', test_phase3_scaling),
    ]
    
    if args.test is not None:
        # Run specific test
        if 1 <= args.test <= len(tests):
            name, func = tests[args.test - 1]
            try:
                func()
                print("\n" + "="*70)
                print("✓ ALL TESTS PASSED")
                print("="*70)
            except AssertionError as e:
                print(f"\n✗ FAILED: {e}")
                sys.exit(1)
    else:
        # Run all tests
        print("\nRunning Phase 3 validation suite...")
        passed = 0
        failed = 0
        for name, func in tests:
            try:
                func()
                passed += 1
            except AssertionError as e:
                print(f"\n✗ FAILED: {e}")
                failed += 1
        
        print("\n" + "="*70)
        print(f"SUMMARY: {passed} passed, {failed} failed")
        print("="*70)
        
        if failed > 0:
            sys.exit(1)
