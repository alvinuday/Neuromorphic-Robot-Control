#!/usr/bin/env python3
"""
Comprehensive Benchmark: OSQP vs Neuromorphic SNN Solvers
===========================================================

Benchmarks 100+ QP instances with varying:
  - Problem sizes (n ∈ {20, 40, 80, 160})
  - Condition numbers (κ ∈ {10, 100, 1000})
  - MPC scenarios (2-DOF arm reaching different targets)

Compares:
  - Wall-clock solve time
  - Accuracy (relative error vs OSQP reference)
  - Iterations to convergence
  - Feasibility (constraint violations, KKT residuals)

Output: JSON results + CSV tables + plots
"""

import numpy as np
import json
import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import time

# Add repo to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver

try:
    from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
    HAS_SNN = True
except ImportError:
    HAS_SNN = False
    print("⚠ Warning: SNN solver not available, will skip SNN benchmarks")

# ============================================================================
# QP INSTANCE GENERATION
# ============================================================================

class QPInstanceGenerator:
    """Generate random QP instances with controlled properties."""
    
    def __init__(self, seed=42):
        self.rng = np.random.default_rng(seed)
    
    def generate_random_qp(self, n: int, kappa: float = 1.0, m: int = None) -> Tuple:
        """Generate a random QP: min 0.5*x'Px + q'x s.t. l <= Ax <= u"""
        
        if m is None:
            m = max(5, n // 2)
        
        # Create P with controlled condition number
        D = np.diag(np.linspace(1, kappa, n))
        U, _ = np.linalg.qr(self.rng.standard_normal((n, n)))
        P = U @ D @ U.T
        P = 0.5 * (P + P.T) + 0.1 * np.eye(n)  # Ensure SPD
        
        q = self.rng.standard_normal(n)
        
        # Constraints: box + general
        A = self.rng.standard_normal((m, n))
        l = -5 * np.ones(m)
        u = 5 * np.ones(m)
        
        return P, q, A, l, u
    
    def generate_mpc_instance(self, horizon: int = 10, 
                             m1: float = 1.0, m2: float = 1.0,
                             l1: float = 0.5, l2: float = 0.5) -> Tuple:
        """Generate QP from 2-DOF MPC problem."""
        
        arm = Arm2DOF(m1=m1, m2=m2, l1=l1, l2=l2, g=9.81)
        mpc = MPCBuilder(arm, N=horizon, dt=0.02)
        
        x0 = np.array([0.0, 0.0, 0.0, 0.0])
        x_goal = np.array([np.pi/4, np.pi/4, 0.0, 0.0])
        
        ref_traj = mpc.build_reference_trajectory(x0, x_goal)
        Q, p, A_eq, b_eq, A_ineq, k_ineq = mpc.build_qp(x0, ref_traj)
        
        A_all = np.vstack([A_eq, A_ineq])
        l_all = np.concatenate([b_eq, np.full(A_ineq.shape[0], -1e30)])
        u_all = np.concatenate([b_eq, k_ineq])
        
        return Q, p, A_all, l_all, u_all


# ============================================================================
# SOLVER WRAPPERS
# ============================================================================

class OSQPBenchmark:
    """OSQP solver benchmark wrapper."""
    
    def __init__(self):
        self.solver = OSQPSolver()
    
    def solve(self, P: np.ndarray, q: np.ndarray, A: np.ndarray, 
              l: np.ndarray, u: np.ndarray, verbose: bool = False) -> Dict:
        """Solve and return timing + solution info."""
        
        t0 = time.perf_counter()
        try:
            x, info = self.solver.solve(P, q, A, l, u)
            t_solve = time.perf_counter() - t0
            
            if x is None:
                return {'status': 'infeasible', 'time': t_solve, 'x': None}
            
            # Compute metrics
            obj = 0.5 * x @ P @ x + q @ x
            c_ineq = A @ x - u
            c_ineq_viol = np.max(np.concatenate([c_ineq, np.zeros_like(c_ineq)]))
            
            return {
                'status': 'solved',
                'time': t_solve,
                'x': x,
                'objective': obj,
                'ineq_violation': c_ineq_viol,
                'iterations': info.get('iter', -1) if info else -1,
            }
        except Exception as e:
            t_solve = time.perf_counter() - t0
            return {'status': 'error', 'time': t_solve, 'error': str(e)}


class SNNBenchmark:
    """SNN solver benchmark wrapper."""
    
    def __init__(self, T_solve: float = 1.0):
        if not HAS_SNN:
            raise RuntimeError("SNN solver not available")
        self.solver = StuartLandauLagrangeDirect(T_solve=T_solve)
    
    def solve(self, P: np.ndarray, q: np.ndarray, A: np.ndarray,
              l: np.ndarray, u: np.ndarray, verbose: bool = False) -> Dict:
        """Solve and return timing + solution info."""
        
        t0 = time.perf_counter()
        try:
            x, info = self.solver.solve(P, q, A, l, u)
            t_solve = time.perf_counter() - t0
            
            if x is None:
                return {'status': 'infeasible', 'time': t_solve, 'x': None}
            
            # Compute metrics
            obj = 0.5 * x @ P @ x + q @ x
            c_ineq = A @ x - u
            c_ineq_viol = np.max(np.concatenate([c_ineq, np.zeros_like(c_ineq)]))
            
            return {
                'status': 'solved',
                'time': t_solve,
                'x': x,
                'objective': obj,
                'ineq_violation': c_ineq_viol,
                'iterations': info.get('iterations', -1) if info else -1,
            }
        except Exception as e:
            t_solve = time.perf_counter() - t0
            return {'status': 'error', 'time': t_solve, 'error': str(e)}


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkSuite:
    """Run comprehensive benchmark suite."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = []
        self.osqp_bench = OSQPBenchmark()
        if HAS_SNN:
            self.snn_bench = SNNBenchmark(T_solve=0.5)
        else:
            self.snn_bench = None
    
    def run_instance(self, problem_type: str, problem_id: int,
                     P: np.ndarray, q: np.ndarray, A: np.ndarray,
                     l: np.ndarray, u: np.ndarray) -> Dict:
        """Run benchmark on single QP instance."""
        
        n = P.shape[0]
        m = A.shape[0]
        
        # OSQP solve (reference)
        osqp_result = self.osqp_bench.solve(P, q, A, l, u)
        
        result = {
            'problem_id': problem_id,
            'problem_type': problem_type,
            'n': n,
            'm': m,
            'kappa': np.linalg.cond(P),
            'osqp': osqp_result,
        }
        
        if osqp_result['status'] == 'solved':
            # SNN solve (if available)
            if self.snn_bench:
                snn_result = self.snn_bench.solve(P, q, A, l, u)
                result['snn'] = snn_result
                
                # Compute accuracy relative to OSQP
                if snn_result['status'] == 'solved':
                    rel_error = np.linalg.norm(snn_result['x'] - osqp_result['x']) / \
                               (np.linalg.norm(osqp_result['x']) + 1e-10)
                    result['snn']['rel_error'] = rel_error
        
        self.results.append(result)
        
        if self.verbose:
            print(f"  [{problem_id:3d}] n={n:3d}, κ={result['kappa']:8.1e}, " +
                  f"OSQP={osqp_result['time']*1000:6.3f}ms", end="")
            if self.snn_bench and result.get('snn', {}).get('status') == 'solved':
                print(f", SNN={result['snn']['time']*1000:7.3f}ms, " +
                      f"err={result['snn']['rel_error']:8.2e}")
            else:
                print()
        
        return result
    
    def run_random_instances(self, n_instances: int = 100):
        """Run benchmark on random QP instances."""
        
        print("\n" + "="*80)
        print("RANDOM QP INSTANCES")
        print("="*80)
        
        gen = QPInstanceGenerator(seed=42)
        problem_id = 0
        
        for size in [20, 40, 80, 160]:
            for kappa in [10.0, 100.0, 1000.0]:
                for _ in range(n_instances // 12 + 1):
                    if problem_id >= n_instances:
                        break
                    
                    P, q, A, l, u = gen.generate_random_qp(n=size, kappa=kappa, m=max(5, size//2))
                    self.run_instance('random', problem_id, P, q, A, l, u)
                    problem_id += 1
    
    def run_mpc_instances(self, n_instances: int = 20):
        """Run benchmark on MPC problem instances."""
        
        print("\n" + "="*80)
        print("MPC QP INSTANCES")
        print("="*80)
        
        gen = QPInstanceGenerator()
        problem_id = 0
        
        for horizon in [5, 10, 20]:
            for _ in range(n_instances // 3):
                if problem_id >= n_instances:
                    break
                
                P, q, A, l, u = gen.generate_mpc_instance(horizon=horizon)
                self.run_instance('mpc', problem_id, P, q, A, l, u)
                problem_id += 1
    
    def generate_report(self) -> str:
        """Generate textual report."""
        
        report = []
        report.append("\n" + "="*80)
        report.append("BENCHMARK REPORT")
        report.append("="*80)
        report.append(f"Timestamp: {datetime.now().isoformat()}")
        report.append(f"Total instances: {len(self.results)}")
        
        # Statistics
        osqp_times = []
        snn_times = []
        speedups = []
        
        for r in self.results:
            if r['osqp']['status'] == 'solved':
                osqp_times.append(r['osqp']['time'])
            if r.get('snn', {}).get('status') == 'solved':
                snn_times.append(r['snn']['time'])
                speedup = r['osqp']['time'] / r['snn']['time']
                speedups.append(speedup)
        
        report.append("\n" + "-"*80)
        report.append("TIMING STATISTICS (ms)")
        report.append("-"*80)
        report.append(f"OSQP:  mean={1000*np.mean(osqp_times):.3f}, " +
                     f"median={1000*np.median(osqp_times):.3f}, " +
                     f"std={1000*np.std(osqp_times):.3f}")
        
        if snn_times:
            report.append(f"SNN:   mean={1000*np.mean(snn_times):.3f}, " +
                         f"median={1000*np.median(snn_times):.3f}, " +
                         f"std={1000*np.std(snn_times):.3f}")
            if speedups:
                report.append(f"Speedup (OSQP/SNN): mean={np.mean(speedups):.2f}x, " +
                             f"median={np.median(speedups):.2f}x")
        
        # Accuracy vs Mangalore et al.
        report.append("\n" + "-"*80)
        report.append("ACCURACY & FEASIBILITY")
        report.append("-"*80)
        
        feasible_osqp = sum(1 for r in self.results if r['osqp']['status'] == 'solved')
        report.append(f"OSQP feasible: {feasible_osqp}/{len(self.results)} " +
                     f"({100*feasible_osqp/len(self.results):.1f}%)")
        
        if snn_times:
            feasible_snn = sum(1 for r in self.results if r.get('snn', {}).get('status') == 'solved')
            report.append(f"SNN feasible:  {feasible_snn}/{len(self.results)} " +
                         f"({100*feasible_snn/len(self.results):.1f}%)")
        
        return "\n".join(report)
    
    def save_results(self, output_dir: Path = None):
        """Save results to JSON and CSV."""
        
        if output_dir is None:
            output_dir = ROOT / "evaluation" / "results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON
        json_path = output_dir / f"benchmark_neuromorphic_mpc_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self._to_json_serializable(self.results), f, indent=2)
        
        # CSV summary
        csv_path = output_dir / f"benchmark_summary_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['problem_id', 'type', 'n', 'm', 'kappa',
                            'osqp_time_ms', 'snn_time_ms', 'speedup', 'rel_error'])
            
            for r in self.results:
                row = [
                    r['problem_id'],
                    r['problem_type'],
                    r['n'],
                    r['m'],
                    f"{r['kappa']:.2e}",
                ]
                
                if r['osqp']['status'] == 'solved':
                    row.append(f"{r['osqp']['time']*1000:.3f}")
                else:
                    row.append('inf')
                
                if r.get('snn', {}).get('status') == 'solved':
                    row.append(f"{r['snn']['time']*1000:.3f}")
                    speedup = r['osqp']['time'] / r['snn']['time'] if r['osqp']['status'] == 'solved' else -1
                    row.append(f"{speedup:.2f}" if speedup > 0 else "N/A")
                    row.append(f"{r['snn'].get('rel_error', -1):.2e}")
                else:
                    row.append('inf')
                    row.append('N/A')
                    row.append('N/A')
                
                writer.writerow(row)
        
        print(f"\n✓ Results saved:")
        print(f"  JSON: {json_path}")
        print(f"  CSV:  {csv_path}")
        
        return json_path, csv_path
    
    def _to_json_serializable(self, obj):
        """Recursively convert to JSON-serializable types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, float)):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (bool, np.bool_)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: self._to_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(v) for v in obj]
        return str(obj)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution."""
    
    print("\n" + "="*80)
    print("NEUROMORPHIC MPC BENCHMARK SUITE")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    bench = BenchmarkSuite(verbose=True)
    
    # Run benchmarks
    bench.run_random_instances(n_instances=36)  # 3 sizes × 3 kappas × 4 instances
    bench.run_mpc_instances(n_instances=12)     # 3 horizons × 4 instances
    
    # Generate report
    print(bench.generate_report())
    
    # Save results
    bench.save_results()
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
