"""
Benchmarking Metrics and Framework
==================================

Metrics for evaluating solver performance:
- Optimality gap vs OSQP
- Constraint violation
- Solve time
- Energy projection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class BenchmarkMetrics:
    """Compute performance metrics for QP solvers."""
    
    @staticmethod
    def optimality_gap(f_solver, f_osqp, eps=1e-10):
        """
        Compute optimality gap in percentage.
        
        Gap = (f_solver - f_OSQP) / |f_OSQP| * 100%
        
        Args:
            f_solver: Objective value from solver
            f_osqp: Objective value from OSQP (optimal)
            eps: Avoid division by zero
        
        Returns:
            gap: Percentage, should be >= 0
        """
        denom = abs(f_osqp) + eps
        gap = (f_solver - f_osqp) / denom * 100
        return max(0, gap)  # No negative gaps (solver should be no worse)
    
    @staticmethod
    def constraint_violation(eq_viol, ineq_viol):
        """
        Overall constraint violation metric.
        
        Violation = max(|Cx-d|_max, |max(0, Acx-u)|_max, |max(0, l-Acx)|_max)
        
        Args:
            eq_viol: Equality constraint violation
            ineq_viol: Inequality constraint violation
        
        Returns:
            violation: Maximum across all constraints
        """
        return max(eq_viol, ineq_viol)
    
    @staticmethod
    def real_time_satisfaction(solve_time, deadline=0.1):
        """
        Fraction of time budget used.
        
        For 10Hz control: deadline = 100ms
        For 20Hz control: deadline = 50ms
        
        Args:
            solve_time: Time taken to solve (seconds)
            deadline: Time budget (seconds)
        
        Returns:
            fraction: solve_time / deadline, should be < 1.0 for real-time
        """
        return solve_time / deadline if deadline > 0 else float('inf')
    
    @staticmethod
    def energy_estimate_loihi(num_ode_steps, num_neurons=1000):
        """
        Estimate energy consumption on Loihi 2 neuromorphic chip.
        
        Simplified model based on:
        - ODE steps → neuron firing (integration steps)
        - Neurons → spikes → energy via datasheet
        - Loihi 2: ~10 pJ per spike, 64 cores, 128K neurons/core
        
        Args:
            num_ode_steps: Number of ODE integration steps
            num_neurons: Actively used neurons (default: 1000)
        
        Returns:
            energy_pJ: Energy in picojules
        """
        # Simplified: each ODE step involves ~num_neurons operations
        num_spike_events = num_ode_steps * num_neurons
        energy_per_spike_pJ = 10.0  # From Loihi datasheet
        energy_pJ = num_spike_events * energy_per_spike_pJ
        return energy_pJ
    
    @staticmethod
    def energy_estimate_cpu(solve_time_ms, power_W=10):
        """
        Estimate energy consumption on CPU.
        
        Typical values:
        - Desktop CPU: 10-100W
        - Laptop CPU: 5-50W
        - ARM (mobile): 1-10W
        
        Args:
            solve_time_ms: Time taken (milliseconds)
            power_W: Power draw (Watts, default 10W for desktop)
        
        Returns:
            energy_joules: Energy in joules
            energy_mJ: Energy in millijoules
        """
        energy_joules = power_W * (solve_time_ms / 1000)
        energy_mJ = energy_joules * 1000
        return energy_joules, energy_mJ


class BenchmarkResult:
    """Container for a single benchmark result."""
    
    def __init__(self, problem_size, solver, x_opt, info, P=None, q=None, **kwargs):
        self.problem_size = problem_size  # Number of variables
        self.solver = solver  # Solver name
        self.x_opt = x_opt
        self.info = info
        self.P = P
        self.q = q
        
        # Derived metrics
        self.objective = info.get('objective', float('inf'))
        self.solve_time = info.get('solve_time', 0)
        self.eq_violation = info.get('eq_violation', 0)
        self.ineq_violation = info.get('ineq_violation', 0)
        self.num_steps = info.get('num_steps', info.get('iterations', 0))
        
        self.metrics = {}
    
    def compute_metrics(self, reference_result=None):
        """Compute derived metrics."""
        self.metrics['constraint_violation'] = BenchmarkMetrics.constraint_violation(
            self.eq_violation, self.ineq_violation
        )
        
        self.metrics['solve_time_ms'] = self.solve_time * 1000
        
        if reference_result is not None:
            self.metrics['optimality_gap_%'] = BenchmarkMetrics.optimality_gap(
                self.objective, reference_result.objective
            )
        
        self.metrics['real_time_10Hz'] = BenchmarkMetrics.real_time_satisfaction(
            self.solve_time, deadline=0.1
        )
        
        self.metrics['real_time_20Hz'] = BenchmarkMetrics.real_time_satisfaction(
            self.solve_time, deadline=0.05
        )
        
        # Energy estimates
        energy_loihi_pJ = BenchmarkMetrics.energy_estimate_loihi(self.num_steps) if self.num_steps > 0 else 0
        energy_cpu_J, energy_cpu_mJ = BenchmarkMetrics.energy_estimate_cpu(self.metrics['solve_time_ms'])
        
        self.metrics['energy_loihi_pJ'] = energy_loihi_pJ
        self.metrics['energy_cpu_mJ'] = energy_cpu_mJ
        self.metrics['energy_ratio_neuromorphic_vs_cpu'] = energy_loihi_pJ / (energy_cpu_mJ * 1e9) if energy_cpu_mJ > 0 else 0


class BenchmarkSuite:
    """Comprehensive benchmarking suite for QP solvers."""
    
    def __init__(self, problem_sizes=[5, 10, 20], num_trials=10):
        """
        Args:
            problem_sizes: MPC horizon sizes to test [N=5, 10, 20, ...]
            num_trials: Number of random instances per size
        """
        self.problem_sizes = problem_sizes
        self.num_trials = num_trials
        self.results = []
    
    def generate_random_mpc_problem(self, N):
        """
        Generate random MPC problem instance.
        
        Args:
            N: Horizon length
        
        Returns:
            (P, q, C, d, Ac, l, u)
        """
        nx = 4  # [q1, q2, dq1, dq2]
        nu = 2  # [tau1, tau2]
        n = N * (nx + nu)
        
        # Random cost
        P = np.eye(n) + 0.1 * np.random.randn(n, n)
        P = P @ P.T  # Make positive definite
        q = np.random.randn(n)
        
        # Dynamics constraints (simplified: identity mapping)
        C = np.eye(n) * 0.9 + 0.1 * np.random.randn(n, n)
        d = np.random.randn(n) * 0.1
        
        # Box constraints
        Ac = np.eye(n)
        u = 2.0 * np.ones(n)
        l = -2.0 * np.ones(n)
        
        return P, q, C, d, Ac, l, u
    
    def run_benchmark(self, solver_names=['OSQP', 'iLQR', 'Neuromorphic'], verbose=True):
        """
        Run full benchmark suite.
        
        Args:
            solver_names: List of solver types to test
            verbose: Print progress
        
        Returns:
            results_df: Pandas dataframe with all results
        """
        from src.benchmark.benchmark_solvers import create_solver
        
        results = []
        
        for N in self.problem_sizes:
            if verbose:
                print(f"\nBenchmark: MPC with N={N} (problem size: {2*N + N*4} vars)")
            
            for trial in range(self.num_trials):
                # Generate problem
                P, q, C, d, Ac, l, u = self.generate_random_mpc_problem(N)
                
                if verbose:
                    print(f"  Trial {trial+1}/{self.num_trials}...", end='', flush=True)
                
                # Run each solver
                osqp_result = None
                for solver_name in solver_names:
                    try:
                        solver = create_solver(solver_name)
                        x_opt = solver.solve(P, q, C, d, Ac, l, u)
                        info = solver.get_info()
                        
                        result = BenchmarkResult(N, solver_name, x_opt, info, P, q)
                        
                        # Use OSQP as reference for gap
                        if solver_name == 'OSQP':
                            osqp_result = result
                        
                        if osqp_result is not None:
                            result.compute_metrics(osqp_result)
                        else:
                            result.compute_metrics()
                        
                        results.append(result)
                    except Exception as e:
                        if verbose:
                            print(f" ERROR ({solver_name}: {str(e)[:30]})")
                
                if verbose:
                    print(" ✓")
        
        self.results = results
        return self._results_to_dataframe(results)
    
    def _results_to_dataframe(self, results):
        """Convert results to pandas dataframe."""
        data = []
        for result in results:
            row = {
                'problem_size': result.problem_size,
                'solver': result.solver,
                'objective': result.objective,
                'solve_time_ms': result.metrics.get('solve_time_ms', 0),
                'eq_violation': result.eq_violation,
                'ineq_violation': result.ineq_violation,
                'constraint_violation': result.metrics.get('constraint_violation', 0),
                'optimality_gap_%': result.metrics.get('optimality_gap_%', 0),
                'real_time_10Hz': result.metrics.get('real_time_10Hz', float('inf')),
                'energy_loihi_pJ': result.metrics.get('energy_loihi_pJ', 0),
                'energy_cpu_mJ': result.metrics.get('energy_cpu_mJ', 0),
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        return df
    
    def print_summary(self):
        """Print benchmark summary statistics."""
        if not self.results:
            print("No results to summarize. Run benchmark first.")
            return
        
        df = self._results_to_dataframe(self.results)
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for solver in df['solver'].unique():
            solver_df = df[df['solver'] == solver]
            
            print(f"\n{solver.upper()}")
            print("-" * 80)
            print(f"  Avg solve time:       {solver_df['solve_time_ms'].mean():.2f} ± {solver_df['solve_time_ms'].std():.2f} ms")
            print(f"  Constraint violation: {solver_df['constraint_violation'].mean():.6e} (max: {solver_df['constraint_violation'].max():.6e})")
            print(f"  Optimality gap:       {solver_df['optimality_gap_%'].mean():.2f}% (max: {solver_df['optimality_gap_%'].max():.2f}%)")
            print(f"  Real-time (10Hz):     {(solver_df['real_time_10Hz'] < 1.0).sum()}/{len(solver_df)} ✓")
            print(f"  Energy (Loihi est):   {solver_df['energy_loihi_pJ'].mean():.0f} pJ")
    
    def save_results(self, filename='results/benchmark_results.csv'):
        """Save benchmark results to CSV."""
        import os
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        df = self._results_to_dataframe(self.results)
        df.to_csv(filename, index=False)
        print(f"✓ Saved results to {filename}")
        
        return df


if __name__ == '__main__':
    # Quick test
    print("Benchmarking Metrics Module")
    print("="*60)
    
    # Example metrics
    print("\nOptimality gap example:")
    gap = BenchmarkMetrics.optimality_gap(1.05, 1.0)
    print(f"  f_solver=1.05, f_OSQP=1.0 → gap={gap:.2f}%")
    
    print("\nEnergy estimate example:")
    ene_loihi = BenchmarkMetrics.energy_estimate_loihi(100)
    ene_cpu, _ = BenchmarkMetrics.energy_estimate_cpu(100)
    print(f"  100 ODE steps, 100ms → Loihi: {ene_loihi:.0f} pJ, CPU: {ene_cpu:.6f} J")
    
    print("\nReal-time example:")
    rt = BenchmarkMetrics.real_time_satisfaction(0.05, 0.1)
    print(f"  50ms solve, 100ms deadline → {rt:.1%} utilization")
