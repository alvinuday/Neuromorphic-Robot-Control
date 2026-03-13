"""Comprehensive performance benchmarking and profiling toolkit."""

import time
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Metrics from a single benchmark run."""
    name: str
    timestamp: str
    duration_s: float
    num_iterations: int
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_hz: float
    success_rate: float
    notes: str = ""
    
    def to_dict(self) -> dict:
        return asdict(self)


class SystemProfiler:
    """Profile system components and generate timing reports."""
    
    def __init__(self, output_dir: str = 'results/benchmarks'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmarks: List[BenchmarkMetrics] = []
        
        logger.info(f"SystemProfiler initialized: {self.output_dir}")
    
    def benchmark_function(
        self,
        func: Callable,
        name: str,
        num_iterations: int = 1000,
        *args,
        **kwargs,
    ) -> BenchmarkMetrics:
        """Benchmark a single function.
        
        Args:
            func: Function to benchmark
            name: Benchmark name
            num_iterations: Number of iterations to run
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            BenchmarkMetrics with timing data
        """
        latencies = []
        num_failed = 0
        
        # Warmup
        for _ in range(10):
            try:
                func(*args, **kwargs)
            except Exception:
                pass
        
        # Actual benchmark
        start_time = time.time()
        
        for _ in range(num_iterations):
            iter_start = time.perf_counter()
            try:
                func(*args, **kwargs)
                iter_end = time.perf_counter()
                latencies.append((iter_end - iter_start) * 1000)  # Convert to ms
            except Exception as e:
                num_failed += 1
                logger.warning(f"Iteration failed: {e}")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        if not latencies:
            logger.error(f"Benchmark {name}: all iterations failed")
            latencies = [0.0]
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        metrics = BenchmarkMetrics(
            name=name,
            timestamp=datetime.now().isoformat(),
            duration_s=total_duration,
            num_iterations=num_iterations,
            mean_latency_ms=float(np.mean(latencies)),
            std_latency_ms=float(np.std(latencies)),
            min_latency_ms=float(np.min(latencies)),
            max_latency_ms=float(np.max(latencies)),
            p50_latency_ms=float(np.percentile(latencies, 50)),
            p95_latency_ms=float(np.percentile(latencies, 95)),
            p99_latency_ms=float(np.percentile(latencies, 99)),
            throughput_hz=num_iterations / total_duration,
            success_rate=(num_iterations - num_failed) / num_iterations,
        )
        
        self.benchmarks.append(metrics)
        
        logger.info(f"Benchmark '{name}': {metrics.mean_latency_ms:.2f}ms "
                   f"(p95: {metrics.p95_latency_ms:.2f}ms, "
                   f"{metrics.throughput_hz:.0f} Hz)")
        
        return metrics
    
    def benchmark_control_loop(
        self,
        step_func: Callable,
        num_steps: int = 1000,
        name: str = "control_loop",
    ) -> BenchmarkMetrics:
        """Benchmark full control loop timing.
        
        Args:
            step_func: Function that performs one control step
            num_steps: Number of steps to run
            name: Benchmark name
            
        Returns:
            BenchmarkMetrics for control loop
        """
        latencies = []
        
        start_time = time.time()
        
        for step_idx in range(num_steps):
            iter_start = time.perf_counter()
            try:
                step_func(step_idx)
                iter_end = time.perf_counter()
                latencies.append((iter_end - iter_start) * 1000)
            except Exception as e:
                logger.warning(f"Step {step_idx} failed: {e}")
                latencies.append(0.0)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        latencies = np.array(latencies)
        
        # Calculate actual control frequency
        valid_latencies = latencies[latencies > 0]
        actual_hz = len(valid_latencies) / total_duration
        
        metrics = BenchmarkMetrics(
            name=name,
            timestamp=datetime.now().isoformat(),
            duration_s=total_duration,
            num_iterations=num_steps,
            mean_latency_ms=float(np.mean(valid_latencies)) if len(valid_latencies) > 0 else 0,
            std_latency_ms=float(np.std(valid_latencies)) if len(valid_latencies) > 0 else 0,
            min_latency_ms=float(np.min(valid_latencies)) if len(valid_latencies) > 0 else 0,
            max_latency_ms=float(np.max(valid_latencies)) if len(valid_latencies) > 0 else 0,
            p50_latency_ms=float(np.percentile(valid_latencies, 50)) if len(valid_latencies) > 0 else 0,
            p95_latency_ms=float(np.percentile(valid_latencies, 95)) if len(valid_latencies) > 0 else 0,
            p99_latency_ms=float(np.percentile(valid_latencies, 99)) if len(valid_latencies) > 0 else 0,
            throughput_hz=actual_hz,
            success_rate=len(valid_latencies) / num_steps,
        )
        
        self.benchmarks.append(metrics)
        
        return metrics
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmarks."""
        if not self.benchmarks:
            return {}
        
        return {
            'total_benchmarks': len(self.benchmarks),
            'timestamp': datetime.now().isoformat(),
            'benchmarks': [b.to_dict() for b in self.benchmarks],
        }
    
    def save_report(self) -> Path:
        """Save benchmark report to JSON file."""
        summary = self.get_summary()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"benchmark_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Benchmark report saved: {report_file}")
        return report_file
    
    def print_summary(self):
        """Print summary to console."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        for benchmark in self.benchmarks:
            print(f"\n{benchmark.name}:")
            print(f"  Duration: {benchmark.duration_s:.2f}s, "
                  f"Iterations: {benchmark.num_iterations}")
            print(f"  Latency: {benchmark.mean_latency_ms:.2f}ms "
                  f"(±{benchmark.std_latency_ms:.2f}ms)")
            print(f"  Percentiles: p50={benchmark.p50_latency_ms:.2f}ms, "
                  f"p95={benchmark.p95_latency_ms:.2f}ms, "
                  f"p99={benchmark.p99_latency_ms:.2f}ms")
            print(f"  Throughput: {benchmark.throughput_hz:.1f} Hz")
            print(f"  Success rate: {benchmark.success_rate*100:.1f}%")


class TaskPerformanceEvaluator:
    """Evaluate task-level performance metrics."""
    
    def __init__(self, output_dir: str = 'results/task_eval'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.task_results: List[Dict[str, Any]] = []
    
    def evaluate_reaching(
        self,
        start_config: np.ndarray,
        goal_config: np.ndarray,
        step_func: Callable,
        max_steps: int = 500,
        tolerance_rad: float = 0.1,
    ) -> Dict[str, float]:
        """Evaluate reaching task (point-to-point).
        
        Args:
            start_config: Initial joint configuration [DOF,]
            goal_config: Target joint configuration [DOF,]
            step_func: Function that returns current config
            max_steps: Maximum steps allowed
            tolerance_rad: Position tolerance to declare success
            
        Returns:
            Metrics dict with 'success', 'num_steps', 'final_error', etc.
        """
        start_time = time.time()
        
        for step in range(max_steps):
            current_config = step_func(step)
            
            error = np.linalg.norm(current_config - goal_config)
            
            if error < tolerance_rad:
                elapsed = time.time() - start_time
                result = {
                    'task': 'reaching',
                    'success': True,
                    'num_steps': step + 1,
                    'final_error_rad': float(error),
                    'time_s': float(elapsed),
                    'initial_distance_rad': float(np.linalg.norm(start_config - goal_config)),
                }
                self.task_results.append(result)
                return result
        
        # Failed to reach
        elapsed = time.time() - start_time
        current_config = step_func(max_steps - 1)
        error = np.linalg.norm(current_config - goal_config)
        
        result = {
            'task': 'reaching',
            'success': False,
            'num_steps': max_steps,
            'final_error_rad': float(error),
            'time_s': float(elapsed),
            'initial_distance_rad': float(np.linalg.norm(start_config - goal_config)),
        }
        self.task_results.append(result)
        return result
    
    def evaluate_tracking(
        self,
        reference_trajectory: np.ndarray,  # [T, DOF]
        step_func: Callable,
        max_tracking_error_rad: float = 0.1,
    ) -> Dict[str, float]:
        """Evaluate trajectory tracking performance.
        
        Args:
            reference_trajectory: Reference trajectory to follow
            step_func: Function that returns current config
            max_tracking_error_rad: Max allowed tracking error
            
        Returns:
            Metrics dict with tracking statistics
        """
        tracking_errors = []
        
        for step in range(len(reference_trajectory)):
            current_config = step_func(step)
            reference_config = reference_trajectory[step]
            
            error = np.linalg.norm(current_config - reference_config)
            tracking_errors.append(error)
        
        tracking_errors = np.array(tracking_errors)
        
        result = {
            'task': 'tracking',
            'success': np.mean(tracking_errors) < max_tracking_error_rad,
            'mean_error_rad': float(np.mean(tracking_errors)),
            'std_error_rad': float(np.std(tracking_errors)),
            'max_error_rad': float(np.max(tracking_errors)),
            'p95_error_rad': float(np.percentile(tracking_errors, 95)),
        }
        self.task_results.append(result)
        return result
    
    def get_summary(self) -> Dict[str, float]:
        """Get aggregate task performance."""
        if not self.task_results:
            return {}
        
        reaching_tasks = [r for r in self.task_results if r['task'] == 'reaching']
        tracking_tasks = [r for r in self.task_results if r['task'] == 'tracking']
        
        summary = {
            'total_tasks': len(self.task_results),
        }
        
        if reaching_tasks:
            success_rate = np.mean([r['success'] for r in reaching_tasks])
            mean_steps = np.mean([r['num_steps'] for r in reaching_tasks])
            mean_error = np.mean([r['final_error_rad'] for r in reaching_tasks])
            
            summary['reaching'] = {
                'success_rate': float(success_rate),
                'mean_steps': float(mean_steps),
                'mean_final_error_rad': float(mean_error),
            }
        
        if tracking_tasks:
            success_rate = np.mean([r['success'] for r in tracking_tasks])
            mean_error = np.mean([r['mean_error_rad'] for r in tracking_tasks])
            
            summary['tracking'] = {
                'success_rate': float(success_rate),
                'mean_error_rad': float(mean_error),
            }
        
        return summary
    
    def save_results(self) -> Path:
        """Save task results to JSON."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.output_dir / f"task_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'results': self.task_results,
                'summary': self.get_summary(),
            }, f, indent=2)
        
        logger.info(f"Task results saved: {results_file}")
        return results_file


class BaselineComparator:
    """Compare system against baseline controllers."""
    
    BASELINES = {
        'random': 'Random joint torque commands',
        'gravity_comp': 'Gravity compensation only (G-mode)',
        'pid_ik': 'Simple PID inverse kinematics',
        'mpc_only': 'MPC without VLA feedback',
        'vla_only': 'Raw VLA output without MPC',
    }
    
    def __init__(self, output_dir: str = 'results/comparisons'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.comparison_results: Dict[str, Dict[str, float]] = {}
    
    def compare_on_task(
        self,
        task_func: Callable,  # Returns (success, metrics)
        baseline_implementations: Dict[str, Callable],
        test_system: Callable,
        num_trials: int = 10,
    ) -> Dict[str, float]:
        """Run comparative evaluation.
        
        Args:
            task_func: Callable that generates a random task
            baseline_implementations: Dict of {name: implementation}
            test_system: Our control system to test
            num_trials: Number of random tasks to test
            
        Returns:
            Comparison results
        """
        results = {}
        
        for baseline_name, baseline_func in baseline_implementations.items():
            successes = 0
            metrics_list = []
            
            for trial in range(num_trials):
                task = task_func()
                try:
                    success, metrics = baseline_func(task)
                    if success:
                        successes += 1
                    metrics_list.append(metrics)
                except Exception as e:
                    logger.warning(f"Baseline {baseline_name} trial {trial} failed: {e}")
            
            results[baseline_name] = {
                'success_rate': successes / num_trials,
                'num_trials': num_trials,
            }
        
        # Test our system
        successes = 0
        for trial in range(num_trials):
            task = task_func()
            try:
                success, metrics = test_system(task)
                if success:
                    successes += 1
            except Exception as e:
                logger.warning(f"Test system trial {trial} failed: {e}")
        
        results['our_system'] = {
            'success_rate': successes / num_trials,
            'num_trials': num_trials,
        }
        
        self.comparison_results = results
        return results
    
    def print_comparison(self):
        """Print comparison results."""
        print("\n" + "="*80)
        print("BASELINE COMPARISON")
        print("="*80)
        
        for name, metrics in self.comparison_results.items():
            symbol = "✓" if name == 'our_system' else " "
            print(f"{symbol} {name:20s}: {metrics['success_rate']*100:6.1f}% "
                  f"({metrics['num_trials']} trials)")
