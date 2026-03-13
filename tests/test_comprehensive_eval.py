"""Comprehensive system validation and benchmarking test suite."""

import pytest
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _has_mujoco() -> bool:
    """Check if mujoco is available."""
    try:
        import mujoco
        return True
    except ImportError:
        return False


class TestDatasetIntegration:
    """Test OpenX dataset integration."""
    
    def test_openx_dataset_load_calvin(self):
        """Load synthetic CALVIN dataset."""
        from src.datasets.openx_loader import OpenXDataset
        
        dataset = OpenXDataset()
        trajectories = dataset.load_synthetic_calvin_subset(num_episodes=50)
        
        assert len(trajectories) == 50
        assert all(t.instruction for t in trajectories)
        assert all(t.frames.shape[1:] == (224, 224, 3) for t in trajectories)
    
    def test_openx_dataset_load_reaching(self):
        """Load synthetic reaching dataset."""
        from src.datasets.openx_loader import OpenXDataset
        
        dataset = OpenXDataset()
        trajectories = dataset.load_synthetic_reaching_subset(num_episodes=30)
        
        assert len(trajectories) == 30
        assert all(t.task_name == 'reaching' for t in trajectories)
    
    def test_openx_dataset_stats(self):
        """Compute dataset statistics."""
        from src.datasets.openx_loader import OpenXDataset
        
        dataset = OpenXDataset()
        trajectories = dataset.load_synthetic_calvin_subset(num_episodes=20)
        
        stats = dataset.get_dataset_stats('calvin')
        
        assert stats['num_episodes'] == 20
        assert stats['total_steps'] > 0
        assert stats['mean_length'] > 0


class TestBenchmarking:
    """Test benchmarking infrastructure."""
    
    def test_profiler_benchmark_function(self):
        """Benchmark a simple function."""
        from src.benchmarks.profiler import SystemProfiler
        
        def dummy_func():
            """Simple function that takes ~1ms."""
            x = np.sum(np.random.randn(1000))
            return x
        
        profiler = SystemProfiler()
        metrics = profiler.benchmark_function(dummy_func, "dummy", num_iterations=100)
        
        assert metrics is not None
        assert metrics.num_iterations == 100
        assert metrics.mean_latency_ms > 0
        assert metrics.success_rate > 0.8
    
    def test_profiler_control_loop_benchmark(self):
        """Benchmark control loop timing."""
        from src.benchmarks.profiler import SystemProfiler
        
        timing_data = []
        
        def control_step(step_idx):
            """Simulate control step."""
            # Simulate ~10ms of computation
            x = np.sum(np.random.randn(10000))
            timing_data.append(x)
        
        profiler = SystemProfiler()
        metrics = profiler.benchmark_control_loop(control_step, num_steps=50)
        
        assert metrics.num_iterations == 50
        assert metrics.throughput_hz > 0
    
    def test_task_evaluator_reaching(self):
        """Test reaching task evaluation."""
        from src.benchmarks.profiler import TaskPerformanceEvaluator
        
        evaluator = TaskPerformanceEvaluator()
        
        start_config = np.array([0.0, 0.0, 0.0])
        goal_config = np.array([0.1, 0.1, 0.1])
        
        current_config = [start_config]  # Use state list
        
        def step_func(step):
            # Move toward goal
            alpha = (step + 1) / 100
            new_config = start_config + alpha * (goal_config - start_config)
            current_config[0] = new_config
            return new_config
        
        result = evaluator.evaluate_reaching(
            start_config, goal_config, step_func, max_steps=100, tolerance_rad=0.05
        )
        
        assert 'success' in result
        assert 'num_steps' in result
        assert 'final_error_rad' in result
    
    def test_baseline_comparator(self):
        """Test baseline comparison."""
        from src.benchmarks.profiler import BaselineComparator
        
        comparator = BaselineComparator()
        
        # Check that baseline names are known
        for baseline_name in BaselineComparator.BASELINES:
            assert len(baseline_name) > 0


class TestVisualization:
    """Test visualization utilities."""
    
    def test_visualizer_plot_trajectories(self):
        """Test trajectory plotting."""
        from src.visualization.visualizer import SystemVisualizer
        
        visualizer = SystemVisualizer()
        
        # Create synthetic data
        q_actual = np.random.randn(100, 3)
        q_reference = q_actual + np.random.randn(100, 3) * 0.1
        tau = np.random.randn(100, 3)
        
        output_path = visualizer.plot_control_trajectories(
            q_actual, q_reference, tau, save_name="test_trajectory.png"
        )
        
        assert output_path.exists()
    
    def test_visualizer_plot_metrics(self):
        """Test metrics plotting."""
        from src.visualization.visualizer import SystemVisualizer
        
        visualizer = SystemVisualizer()
        
        mpc_costs = np.random.exponential(1.0, 100)
        mpc_times = np.random.normal(15, 3, 100)
        vla_latencies = np.random.normal(650, 50, 100)
        
        output_path = visualizer.plot_control_metrics(
            mpc_costs, mpc_times, vla_latencies, save_name="test_metrics.png"
        )
        
        assert output_path.exists()
    
    def test_visualizer_plot_comparison(self):
        """Test comparison plotting."""
        from src.visualization.visualizer import SystemVisualizer
        
        visualizer = SystemVisualizer()
        
        results = {
            'random': {'success_rate': 0.10},
            'gravity_comp': {'success_rate': 0.30},
            'pid_ik': {'success_rate': 0.50},
            'our_system': {'success_rate': 0.85},
        }
        
        output_path = visualizer.plot_comparison(
            results, 'success_rate', save_name="test_comparison.png"
        )
        
        assert output_path.exists()
    
    def test_visualizer_plot_distribution(self):
        """Test distribution plotting."""
        from src.visualization.visualizer import SystemVisualizer
        
        visualizer = SystemVisualizer()
        
        latencies = np.random.normal(20, 5, 1000)
        
        output_path = visualizer.plot_performance_distribution(
            latencies, name="MPC Timing", save_name="test_distribution.png"
        )
        
        assert output_path.exists()


class TestDataCollector:
    """Test data collection infrastructure."""
    
    def test_data_collector_recording(self):
        """Test data collection."""
        from src.utils.data_collector import DataCollector
        
        collector = DataCollector(task_name='test')
        
        # Record some steps
        for step in range(10):
            collector.record_step(
                step=step,
                q=np.array([0.1*step, 0.2*step, 0.3*step]),
                qdot=np.array([0.01, 0.02, 0.03]),
                tau=np.array([1.0, 2.0, 3.0]),
                ee_pos=np.array([0.5, 0.6, 0.7]),
                mpc_cost=1.5 * (step + 1),
                mpc_time_ms=15.0 + np.random.randn(),
            )
        
        assert len(collector.step_data) == 10
        
        # Get summary
        summary = collector.get_summary()
        assert summary['total_steps'] == 10
        assert 'mpc_timing_ms' in summary


class TestMuJoCoEnvironment:
    """Test MuJoCo environment (skip if not available)."""
    
    @pytest.mark.skipif(
        not _has_mujoco(),
        reason="mujoco not installed"
    )
    def test_mujoco_env_basic(self):
        """Test MuJoCo environment basics."""
        from src.environments.mujoco_3dof_env import MuJoCo3DOFEnv
        
        env = MuJoCo3DOFEnv(headless=True, render_mode=None)
        
        obs, info = env.reset()
        assert 'state' in obs
        
        # Take a few steps
        for _ in range(10):
            action = np.zeros(3)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs is not None
        
        env.close()


class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_eval_pipeline(self):
        """Test complete evaluation pipeline."""
        from src.datasets.openx_loader import OpenXDataset
        from src.benchmarks.profiler import SystemProfiler
        from src.visualization.visualizer import SystemVisualizer
        
        # Load dataset
        dataset = OpenXDataset()
        trajectories = dataset.load_synthetic_reaching_subset(num_episodes=10)
        
        assert len(trajectories) == 10
        
        # Get stats
        stats = dataset.get_dataset_stats('reaching')
        assert stats['num_episodes'] == 10
        
        # Create synthetic benchmarking results
        profiler = SystemProfiler()
        
        def dummy_task():
            return np.sum(np.random.randn(1000))
        
        metrics = profiler.benchmark_function(
            dummy_task, "integration_test", num_iterations=50
        )
        
        assert metrics is not None
        
        # Visualize results
        visualizer = SystemVisualizer()
        
        latencies = np.random.normal(15, 3, 50)
        output_path = visualizer.plot_performance_distribution(
            latencies, save_name="integration_test.png"
        )
        
        assert output_path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
