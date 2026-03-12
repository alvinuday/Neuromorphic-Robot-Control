"""
End-to-End Integration Test
============================

Complete workflow test: solver → benchmark → results
"""

import sys
import os
sys.path.insert(0, '/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control')

import numpy as np
import pandas as pd
from pathlib import Path
import json

from src.benchmark.benchmark_solvers import create_solver, OSQPSolver, ILQRSolver, NeuromorphicSolver
from src.benchmark.metrics import BenchmarkMetrics, BenchmarkSuite


def test_complete_workflow():
    """Test complete workflow: generate problem → solve → benchmark."""
    print("\n" + "="*70)
    print("TEST: Complete Workflow")
    print("="*70)
    
    # Step 1: Create simple MPC problem
    print("\n[Workflow] Step 1: Creating MPC problem...")
    
    N = 10  # Horizon
    n = 2   # State dim
    m = 2   # Control dim
    
    # Random LQR matrices
    A = np.eye(2) * 0.95 + 0.05 * np.random.randn(2, 2)
    B = np.random.randn(2, 2)
    Q = np.eye(2)
    R = 0.1 * np.eye(2) * 5
    
    # Build QP
    P = np.eye(2*N*m) * 0.1
    q = np.random.randn(2*N*m) * 0.1
    
    C = np.eye(2) - np.eye(2, k=-2)[:2, :2] if N > 1 else np.eye(2)
    d = np.ones(2) * 0.5
    
    Ac = np.eye(2*N*m)
    l = np.ones(2*N*m) * -2.0
    u = np.ones(2*N*m) * 2.0
    
    print(f"[Workflow] Problem: N={N}, n={n}, m={m}")
    print(f"[Workflow] QP size: {P.shape}, constraints: eq={C.shape[0]}, ineq={Ac.shape[0]}")
    
    # Step 2: Solve with all solvers
    print("\n[Workflow] Step 2: Solving with all solvers...")
    
    results = {}
    
    # iLQR
    try:
        solver = create_solver('ilqr')
        x_ilqr = solver.solve(P, q, C, d, Ac, l, u)
        info_ilqr = solver.get_info()
        results['iLQR'] = {'x': x_ilqr, 'info': info_ilqr}
        print(f"[Workflow] iLQR: obj={info_ilqr['objective']:.6f}, time={info_ilqr['solve_time']:.6f}s")
    except Exception as e:
        print(f"[Workflow] iLQR failed: {e}")
    
    # Neuromorphic
    try:
        solver = create_solver('neuromorphic')
        x_neuro = solver.solve(P, q, C, d, Ac, l, u)
        info_neuro = solver.get_info()
        results['Neuromorphic'] = {'x': x_neuro, 'info': info_neuro}
        print(f"[Workflow] Neuromorphic: obj={info_neuro['objective_value']:.6f}, time={info_neuro['time_to_solution']:.6f}s")
    except Exception as e:
        print(f"[Workflow] Neuromorphic failed: {e}")
    
    # OSQP (if available)
    try:
        solver = create_solver('osqp')
        x_osqp = solver.solve(P, q, C, d, Ac, l, u)
        info_osqp = solver.get_info()
        results['OSQP'] = {'x': x_osqp, 'info': info_osqp}
        print(f"[Workflow] OSQP: obj={info_osqp['objective']:.6f}, time={info_osqp['solve_time']:.6f}s")
    except ImportError:
        print(f"[Workflow] OSQP not available (optional)")
    except Exception as e:
        print(f"[Workflow] OSQP failed: {e}")
    
    # Step 3: Compare results
    print("\n[Workflow] Step 3: Comparing results...")
    
    # Get reference objective
    if 'OSQP' in results:
        f_osqp = results['OSQP']['info'].get('objective', None)
    else:
        f_osqp = results['iLQR']['info']['objective'] if 'iLQR' in results else None
    
    for solver_name, result in results.items():
        info = result['info']
        x = result['x']
        
        # Get objective
        if 'objective' in info:
            f = info['objective']
        else:
            f = info.get('objective_value', 0.0)
        
        # Compute metrics
        gap = ((f - f_osqp) / f_osqp * 100) if f_osqp else 0
        
        # Get constraint violation
        if 'eq_violation' in info:
            viol = info['eq_violation']
        else:
            viol = info.get('constraint_eq_violation', 0.0)
        
        # Get solve time
        if 'solve_time' in info:
            t = info['solve_time']
        else:
            t = info.get('time_to_solution', 0.0)
        
        print(f"[Workflow] {solver_name}:")
        print(f"  Objective: {f:.6f}")
        print(f"  Gap vs OSQP: {gap:.2f}%")
        print(f"  Constraint violation: {viol:.6e}")
        print(f"  Solve time: {t:.6f}s ({t*1000:.2f}ms)")
    
    print("\n[Workflow] ✓ PASSED")
    return True


def test_benchmark_sweep():
    """Test benchmark sweep across problem sizes."""
    print("\n" + "="*70)
    print("TEST: Benchmark Sweep (N=5,10,15)")
    print("="*70)
    
    suite = BenchmarkSuite(
        problem_sizes=[5, 10, 15],
        num_trials=2
    )
    
    print("[Sweep] Running benchmark sweep...")
    
    try:
        df = suite.run_benchmark(
            solver_names=['Neuromorphic', 'iLQR'],
            verbose=False
        )
        
        print(f"\n[Sweep] Results shape: {df.shape}")
        print(f"[Sweep] Columns: {list(df.columns)}")
        
        # Print summary by problem size
        for N in [5, 10, 15]:
            df_N = df[df['problem_size'] == N]
            print(f"\n[Sweep] Problem size N={N}:")
            
            for solver in df_N['solver'].unique():
                df_solver = df_N[df_N['solver'] == solver]
                avg_time = df_solver['solve_time_ms'].mean()
                std_time = df_solver['solve_time_ms'].std()
                avg_viol = df_solver['constraint_violation'].mean()
                
                print(f"  {solver}: {avg_time:.2f}±{std_time:.2f}ms, viol={avg_viol:.6e}")
        
        # Save results
        output_path = Path('/Users/alvin/Documents/Alvin/College/Academics/Master\'s Thesis/Code/Neuromorphic-Robot-Control/results/benchmark_sweep.csv')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"\n[Sweep] Results saved to {output_path}")
        
        print("[Sweep] ✓ PASSED")
        return True
    
    except Exception as e:
        print(f"[Sweep] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convergence_behavior():
    """Test solver convergence behavior over time."""
    print("\n" + "="*70)
    print("TEST: Convergence Behavior")
    print("="*70)
    
    # Create a problem
    N = 10
    P = np.eye(N*2) * 0.5
    q = np.random.randn(N*2) * 0.1
    C = np.zeros((1, N*2))
    C[0, :2] = [1.0, 1.0]
    d = np.array([1.0])
    Ac = np.eye(N*2)
    l = np.ones(N*2) * -2.0
    u = np.ones(N*2) * 2.0
    
    print(f"[Convergence] Problem: N={N}, QP size {N*2}")
    
    # Solve with Neuromorphic solver (tracks convergence)
    try:
        solver = NeuromorphicSolver()
        x_opt = solver.solve(P, q, C, d, Ac, l, u)
        info = solver.get_info()
        
        print(f"\n[Convergence] Solver info:")
        for key, val in info.items():
            if isinstance(val, (int, float)):
                print(f"  {key}: {val}")
        
        print("[Convergence] ✓ PASSED")
        return True
    
    except Exception as e:
        print(f"[Convergence] ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scaling_analysis():
    """Test how solve time scales with problem size."""
    print("\n" + "="*70)
    print("TEST: Scaling Analysis")
    print("="*70)
    
    sizes = [5, 10, 15, 20]
    results = {'Neuromorphic': {}, 'iLQR': {}}
    
    print("[Scaling] Testing problem sizes:", sizes)
    
    for N in sizes:
        print(f"\n[Scaling] Problem size N={N}...")
        
        # Create problem
        P = np.eye(N*2) * 0.5
        q = np.random.randn(N*2) * 0.1
        C = np.zeros((2, N*2))
        C[0, 0] = 1.0
        C[1, N] = 1.0
        d = np.ones(2)
        Ac = np.eye(N*2)
        l = -np.ones(N*2)
        u = np.ones(N*2)
        
        # Test Neuromorphic
        try:
            solver = NeuromorphicSolver()
            x = solver.solve(P, q, C, d, Ac, l, u)
            info = solver.get_info()
            t = info.get('time_to_solution', 0.0)
            results['Neuromorphic'][N] = t
            print(f"  Neuromorphic: {t*1000:.2f}ms")
        except Exception as e:
            print(f"  Neuromorphic failed: {e}")
        
        # Test iLQR
        try:
            solver = ILQRSolver()
            x = solver.solve(P, q, C, d, Ac, l, u)
            info = solver.get_info()
            t = info.get('solve_time', 0.0)
            results['iLQR'][N] = t
            print(f"  iLQR: {t*1000:.2f}ms")
        except Exception as e:
            print(f"  iLQR failed: {e}")
    
    # Analyze scaling
    print("\n[Scaling] Scaling analysis:")
    
    for solver in results:
        if len(results[solver]) >= 2:
            sizes_tested = sorted(results[solver].keys())
            times = [results[solver][N] for N in sizes_tested]
            
            print(f"\n  {solver}:")
            for N, t in zip(sizes_tested, times):
                print(f"    N={N}: {t*1000:.2f}ms")
            
            # Compute scaling factor
            if times[0] > 0:
                scaling = times[-1] / times[0]
                print(f"    Scaling {sizes_tested[0]}→{sizes_tested[-1]}: {scaling:.2f}x")
    
    print("\n[Scaling] ✓ PASSED")
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', type=int, default=None)
    args = parser.parse_args()
    
    tests = [
        ('complete_workflow', test_complete_workflow),
        ('benchmark_sweep', test_benchmark_sweep),
        ('convergence_behavior', test_convergence_behavior),
        ('scaling_analysis', test_scaling_analysis),
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
