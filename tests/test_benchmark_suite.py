"""
Comprehensive Benchmark Test Suite
==================================

Tests for benchmarking framework and solver integration.
"""

import sys
import numpy as np
sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')

from src.benchmark.benchmark_solvers import create_solver, NeuromorphicSolver, ILQRSolver, OSQPSolver
from src.benchmark.metrics import BenchmarkMetrics, BenchmarkSuite, BenchmarkResult


def test_osqp_solver():
    """Test OSQP solver."""
    print("\n" + "="*70)
    print("TEST: OSQP Solver")
    print("="*70)
    
    try:
        solver = OSQPSolver(verbose=False)
        print("[OSQP] ✓ Solver created")
    except ImportError as e:
        print(f"[OSQP] ⚠ Skipped (not installed): {e}")
        print("[OSQP] Install with: pip install osqp")
        return False
    
    # Simple QP
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    q = np.array([-2.0, -4.0])
    C = np.array([[1.0, 1.0]])
    d = np.array([1.0])
    Ac = np.eye(2)
    l = np.array([0.0, 0.0])
    u = np.array([10.0, 10.0])
    
    x_opt = solver.solve(P, q, C, d, Ac, l, u)
    info = solver.get_info()
    
    print(f"[OSQP] Optimal x: {x_opt}")
    print(f"[OSQP] Objective: {info['objective']:.6f}")
    print(f"[OSQP] Eq violation: {info['eq_violation']:.6e}")
    print(f"[OSQP] Solve time: {info['solve_time']:.6f}s")
    print(f"[OSQP] Status: {info['status']}")
    
    assert info['eq_violation'] < 1e-6, f"Eq violation too large: {info['eq_violation']}"
    assert info['solve_time'] < 1.0, f"Solve time too long: {info['solve_time']}"
    
    print("[OSQP] ✓ PASSED")
    return True


def test_neuromorphic_solver():
    """Test Neuromorphic solver."""
    print("\n" + "="*70)
    print("TEST: Neuromorphic (SL+DirectLag) Solver")
    print("="*70)
    
    solver = NeuromorphicSolver()
    print("[Neuromorphic] ✓ Solver created")
    
    # Simple QP
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    q = np.array([-2.0, -4.0])
    C = np.array([[1.0, 1.0]])
    d = np.array([1.0])
    Ac = np.eye(2)
    l = np.array([0.0, 0.0])
    u = np.array([10.0, 10.0])
    
    x_opt = solver.solve(P, q, C, d, Ac, l, u)
    info = solver.get_info()
    
    print(f"[Neuromorphic] Optimal x: {x_opt}")
    print(f"[Neuromorphic] Objective: {info['objective_value']:.6f}")
    print(f"[Neuromorphic] Eq violation: {info['constraint_eq_violation']:.6e}")
    print(f"[Neuromorphic] Solve time: {info['time_to_solution']:.6f}s")
    print(f"[Neuromorphic] Converged: {info['converged']}")
    
    assert info['constraint_eq_violation'] < 0.01, f"Eq violation too large"
    
    print("[Neuromorphic] ✓ PASSED")
    return True


def test_ilqr_solver():
    """Test iLQR solver."""
    print("\n" + "="*70)
    print("TEST: iLQR Solver")
    print("="*70)
    
    solver = ILQRSolver()
    print("[iLQR] ✓ Solver created")
    
    # Simple QP
    P = np.array([[2.0, 0.0], [0.0, 2.0]])
    q = np.array([-2.0, -4.0])
    C = np.array([[1.0, 1.0]])
    d = np.array([1.0])
    Ac = np.eye(2)
    l = np.array([0.0, 0.0])
    u = np.array([10.0, 10.0])
    
    x_opt = solver.solve(P, q, C, d, Ac, l, u)
    info = solver.get_info()
    
    print(f"[iLQR] Solution x: {x_opt}")
    print(f"[iLQR] Objective: {info['objective']:.6f}")
    print(f"[iLQR] Eq violation: {info['eq_violation']:.6e}")
    print(f"[iLQR] Solve time: {info['solve_time']:.6f}s")
    print(f"[iLQR] Iterations: {info['iterations']}")
    
    print("[iLQR] ✓ PASSED")
    return True


def test_benchmark_metrics():
    """Test metrics computation."""
    print("\n" + "="*70)
    print("TEST: Benchmark Metrics")
    print("="*70)
    
    # Test optimality gap
    gap = BenchmarkMetrics.optimality_gap(1.05, 1.0)
    print(f"[Metrics] Optimality gap (1.05 vs 1.0): {gap:.2f}%")
    assert 4.9 < gap < 5.1, f"Gap incorrect: {gap}"
    
    # Test constraint violation
    viol = BenchmarkMetrics.constraint_violation(0.01, 0.001)
    print(f"[Metrics] Constraint violation (0.01, 0.001): {viol:.6f}")
    assert viol == 0.01, f"Violation incorrect: {viol}"
    
    # Test real-time satisfaction
    rt = BenchmarkMetrics.real_time_satisfaction(0.05, 0.1)
    print(f"[Metrics] Real-time satisfaction (50ms/100ms): {rt:.1%}")
    assert 0.49 < rt < 0.51, f"RT satisfaction incorrect: {rt}"
    
    # Test energy estimate
    ene_loihi = BenchmarkMetrics.energy_estimate_loihi(100)
    print(f"[Metrics] Energy (Loihi, 100 steps): {ene_loihi:.0f} pJ")
    assert ene_loihi > 0, f"Loihi energy incorrect: {ene_loihi}"
    
    ene_cpu_j, ene_cpu_mj = BenchmarkMetrics.energy_estimate_cpu(100)
    print(f"[Metrics] Energy (CPU, 100ms): {ene_cpu_mj:.3f} mJ")
    assert ene_cpu_j > 0, f"CPU energy incorrect: {ene_cpu_j}"
    
    print("[Metrics] ✓ PASSED")
    return True


def test_solver_factory():
    """Test solver factory function."""
    print("\n" + "="*70)
    print("TEST: Solver Factory")
    print("="*70)
    
    # Test creating solvers
    try:
        solver_neuro = create_solver('neuromorphic')
        print("[Factory] ✓ Created Neuromorphic solver")
    except Exception as e:
        print(f"[Factory] ✗ Failed to create Neuromorphic: {e}")
        return False
    
    try:
        solver_ilqr = create_solver('ilqr')
        print("[Factory] ✓ Created iLQR solver")
    except Exception as e:
        print(f"[Factory] ✗ Failed to create iLQR: {e}")
        return False
    
    try:
        solver_osqp = create_solver('osqp')
        print("[Factory] ✓ Created OSQP solver")
    except ImportError:
        print("[Factory] ⚠ OSQP not available (optional)")
    
    print("[Factory] ✓ PASSED")
    return True


def test_small_benchmark():
    """Run a small benchmark suite."""
    print("\n" + "="*70)
    print("TEST: Small Benchmark Suite (N=5 only, 2 trials)")
    print("="*70)
    
    suite = BenchmarkSuite(problem_sizes=[5], num_trials=2)
    
    try:
        df = suite.run_benchmark(
            solver_names=['Neuromorphic', 'iLQR'],
            verbose=True
        )
        
        print(f"\n[Benchmark] Results shape: {df.shape}")
        print(f"[Benchmark] Solvers tested: {df['solver'].unique()}")
        
        # Check results
        for solver_name in df['solver'].unique():
            solver_df = df[df['solver'] == solver_name]
            avg_time = solver_df['solve_time_ms'].mean()
            avg_violation = solver_df['constraint_violation'].mean()
            print(f"[Benchmark] {solver_name}: avg_time={avg_time:.2f}ms, violation={avg_violation:.6e}")
        
        print("[Benchmark] ✓ PASSED")
        return True
    
    except Exception as e:
        print(f"[Benchmark] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_benchmark_result():
    """Test BenchmarkResult class."""
    print("\n" + "="*70)
    print("TEST: Benchmark Result")
    print("="*70)
    
    # Create dummy result
    info = {
        'objective': 1.0,
        'solve_time': 0.05,
        'eq_violation': 1e-8,
        'ineq_violation': 0.0,
        'num_steps': 100
    }
    
    result = BenchmarkResult(
        problem_size=10,
        solver='TestSolver',
        x_opt=np.array([1.0, 2.0]),
        info=info
    )
    
    # Compute metrics
    result.compute_metrics()
    
    print(f"[BenchmarkResult] Problem size: {result.problem_size}")
    print(f"[BenchmarkResult] Solver: {result.solver}")
    print(f"[BenchmarkResult] Constraint violation: {result.metrics['constraint_violation']:.6e}")
    print(f"[BenchmarkResult] Solve time: {result.metrics['solve_time_ms']:.2f}ms")
    print(f"[BenchmarkResult] Real-time (10Hz): {result.metrics['real_time_10Hz']:.2%}")
    print(f"[BenchmarkResult] Energy (Loihi): {result.metrics['energy_loihi_pJ']:.0f} pJ")
    
    assert result.metrics['solve_time_ms'] == 50.0
    assert result.metrics['constraint_violation'] == 1e-8
    
    print("[BenchmarkResult] ✓ PASSED")
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', type=int, default=None)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    
    tests = [
        ('solver_factory', test_solver_factory),
        ('neuromorphic_solver', test_neuromorphic_solver),
        ('ilqr_solver', test_ilqr_solver),
        ('osqp_solver', test_osqp_solver),
        ('metrics', test_benchmark_metrics),
        ('result', test_benchmark_result),
        ('small_benchmark', test_small_benchmark),
    ]
    
    if args.test is not None:
        if 1 <= args.test <= len(tests):
            name, func = tests[args.test - 1]
            try:
                result = func()
                if result or result is None:
                    print(f"\n✓ Test passed")
                else:
                    print(f"\n✗ Test failed")
                    sys.exit(1)
            except Exception as e:
                print(f"\n✗ Test failed with exception: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    else:
        # Run all tests
        passed = 0
        failed = 0
        for name, func in tests:
            try:
                result = func()
                if result or result is None:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"\n✗ {name} failed: {e}")
                failed += 1
        
        print("\n" + "="*70)
        print(f"SUMMARY: {passed} passed, {failed} failed")
        print("="*70)
        
        if failed > 0:
            sys.exit(1)
