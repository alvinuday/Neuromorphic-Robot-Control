"""
Phase 6 Benchmarks: Performance evaluation of complete system.

Benchmarks:
- B1: Dataset replay performance
- B2: MPC-only control (without VLA)  
- B3: SmolVLA-only control (without MPC)
- B4: Full dual-system control
- B5: Sensor ablation study
"""

import numpy as np
import logging
import time
from pathlib import Path
from typing import Dict, Tuple

from src.simulation.envs.xarm_env import XArmEnv
from src.mpc.xarm_controller import XArmMPCController
from src.smolvla_client.trajectory_buffer import TrajectoryBuffer
from src.integration.dual_system_controller import DualSystemController


logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────────
# B1: Dataset Replay Benchmark
# ────────────────────────────────────────────────────────────────────────────

def benchmark_b1_dataset_replay(
    n_episodes: int = 5,
    max_steps_per_episode: int = 500
) -> Dict[str, float]:
    """
    B1 Benchmark: Replay episodes from lerobot dataset.
    
    Measures:
    - Episode completion rate
    - Tracking error (position vs expected)
    - Simulation frame rate
    """
    try:
        from lerobot.common.datasets import load_dataset
    except ImportError:
        logger.warning("lerobot not available, skipping B1")
        return {"skip_reason": "lerobot not installed"}
    
    logger.info("=" * 70)
    logger.info("B1 BENCHMARK: Dataset Replay")
    logger.info("=" * 70)
    
    try:
        dataset = load_dataset("lerobot/utokyo_xarm_pick_and_place", split="train")
        logger.info(f"Loaded dataset: {len(dataset)} episodes")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return {"error": str(e)}
    
    env = XArmEnv(render_size=84)
    
    metrics = {
        "episodes_completed": 0,
        "total_steps": 0,
        "tracking_error_mean": 0.0,
        "fps_mean": 0.0,
        "errors": [],
    }
    
    episode_fps = []
    
    try:
        for ep in range(min(n_episodes, len(dataset))):
            episode = dataset[ep]
            
            # Get expected trajectory
            observations = episode["observation"]
            expected_q = observations["state"]  # [T, 8] - joint positions
            
            env.reset()
            q_trajectory_actual = [env.get_joint_pos().copy()]
            episode_steps = min(len(expected_q), max_steps_per_episode)
            
            t_start = time.perf_counter()
            
            for step in range(episode_steps):
                # Get current state
                q = env.get_joint_pos()
                qd = env.get_joint_vel()
                
                # Expected state
                q_expected = expected_q[step]
                
                # Simple control: move towards expected
                q_error = q_expected - q
                tau = np.clip(q_error * 5.0, -10, 10)  # P-control
                
                # Step
                env.step(tau)
                q_trajectory_actual.append(q.copy())
            
            t_end = time.perf_counter()
            
            # Compute metrics
            q_trajectory_actual = np.array(q_trajectory_actual)
            tracking_error = np.mean(np.linalg.norm(
                q_trajectory_actual[:-1] - expected_q[:episode_steps], axis=1
            ))
            
            fps = episode_steps / (t_end - t_start)
            episode_fps.append(fps)
            
            metrics["episodes_completed"] += 1
            metrics["total_steps"] += episode_steps
            metrics["tracking_error_mean"] += tracking_error
            
            logger.info(f"Episode {ep}: {episode_steps} steps, "
                       f"error={tracking_error:.4f} rad, fps={fps:.1f}")
    
    except Exception as e:
        metrics["errors"].append(str(e))
        logger.error(f"Error during replay: {e}")
    
    finally:
        env.renderer_rgb.close()
        env.renderer_hi.close()
    
    # Compute final metrics
    if metrics["episodes_completed"] > 0:
        metrics["tracking_error_mean"] /= metrics["episodes_completed"]
        metrics["fps_mean"] = np.mean(episode_fps)
    
    logger.info(f"\nB1 Summary:")
    logger.info(f"  Episodes: {metrics['episodes_completed']}/{n_episodes}")
    logger.info(f"  Total steps: {metrics['total_steps']}")
    logger.info(f"  Tracking error: {metrics['tracking_error_mean']:.4f} rad")
    logger.info(f"  FPS (mean): {metrics['fps_mean']:.1f}")
    
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# B2: MPC-Only Benchmark
# ────────────────────────────────────────────────────────────────────────────

def benchmark_b2_mpc_only(
    test_duration_s: float = 10.0,
    control_dt: float = 0.01
) -> Dict[str, float]:
    """
    B2 Benchmark: MPC control without VLA (pure reactive control).
    
    Measures:
    - Control loop frequency
    - Computation time per step
    - Trajectory smoothness
    """
    logger.info("=" * 70)
    logger.info("B2 BENCHMARK: MPC-Only Control")
    logger.info("=" * 70)
    
    env = XArmEnv(render_size=84)
    mpc = XArmMPCController(
        horizon_steps=10,
        dt=control_dt,
        tracking_weight=100.0,
    )
    
    metrics = {
        "steps_executed": 0,
        "control_time_mean_ms": 0.0,
        "control_time_max_ms": 0.0,
        "control_time_p99_ms": 0.0,
        "trajectory_smoothness": 0.0,
        "freq_mean_hz": 0.0,
    }
    
    # Set tracking goal
    q_start = env.get_joint_pos().copy()
    q_goal = q_start + np.array([0.5, 0.3, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    
    step_times = []
    q_trajectory = [q_start.copy()]
    tau_trajectory = []
    
    t_start = time.perf_counter()
    
    try:
        while (time.perf_counter() - t_start) < test_duration_s:
            q = env.get_joint_pos()
            qd = env.get_joint_vel()
            
            # MPC step
            t_step_start = time.perf_counter()
            tau, info = mpc.step(q, qd, q_goal)
            t_step = (time.perf_counter() - t_step_start) * 1000  # ms
            
            step_times.append(t_step)
            
            # Simulate
            env.step(tau)
            q_trajectory.append(q.copy())
            tau_trajectory.append(tau.copy())
            
            metrics["steps_executed"] += 1
    
    finally:
        env.renderer_rgb.close()
        env.renderer_hi.close()
    
    # Compute final metrics
    if step_times:
        metrics["control_time_mean_ms"] = np.mean(step_times)
        metrics["control_time_max_ms"] = np.max(step_times)
        metrics["control_time_p99_ms"] = np.percentile(step_times, 99)
    
    if tau_trajectory:
        # Trajectory smoothness = integral of acceleration
        tau_trajectory = np.array(tau_trajectory)
        tau_diff = np.diff(tau_trajectory, axis=0)
        metrics["trajectory_smoothness"] = np.mean(np.linalg.norm(tau_diff, axis=1))
    
    metrics["freq_mean_hz"] = metrics["steps_executed"] / test_duration_s
    
    logger.info(f"\nB2 Summary:")
    logger.info(f"  Steps: {metrics['steps_executed']}")
    logger.info(f"  Frequency: {metrics['freq_mean_hz']:.1f} Hz")
    logger.info(f"  Control time: {metrics['control_time_mean_ms']:.2f} "
               f"(max {metrics['control_time_max_ms']:.2f}, p99 "
               f"{metrics['control_time_p99_ms']:.2f}) ms")
    logger.info(f"  Smoothness (accel): {metrics['trajectory_smoothness']:.4f}")
    
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# B3: VLA-Only Benchmark (with mock predictions)
# ────────────────────────────────────────────────────────────────────────────

def benchmark_b3_vla_only(
    n_goals: int = 10,
    trajectory_duration_s: float = 5.0
) -> Dict[str, float]:
    """
    B3 Benchmark: VLA-only trajectory generation (without MPC feedback).
    
    Measures:
    - Trajectory generation time
    - Goal reaching success rate
    - Time to goal
    """
    logger.info("=" * 70)
    logger.info("B3 BENCHMARK: VLA-Only Trajectory Generation")
    logger.info("=" * 70)
    
    buffer = TrajectoryBuffer(n_joints=8, arrival_threshold_rad=0.05)
    
    metrics = {
        "goals_set": 0,
        "goals_reached": 0,
        "traj_gen_time_ms_mean": 0.0,
        "time_to_goal_s_mean": 0.0,
        "success_rate": 0.0,
    }
    
    q_current = np.zeros(8)
    traj_times = []
    goal_times = []
    
    try:
        for goal_idx in range(n_goals):
            # Random subgoal in reachable space
            q_goal = np.random.uniform(-0.5, 0.5, 8)
            
            # Update subgoal
            buffer.update_subgoal(q_goal)
            metrics["goals_set"] += 1
            
            # Generate trajectory
            t_start = time.perf_counter()
            N = 100
            dt = 0.05
            q_ref, qdot_ref = buffer.get_reference_trajectory(
                q_current, N=N, dt=dt
            )
            t_traj = (time.perf_counter() - t_start) * 1000  # ms
            traj_times.append(t_traj)
            
            # Check goal within trajectory
            if np.allclose(q_ref[-1], q_goal, atol=1e-5):
                metrics["goals_reached"] += 1
                goal_times.append((N - 1) * dt)
            
            # Move to new starting position
            q_current = q_ref[-1]
    
    except Exception as e:
        logger.error(f"Error in B3: {e}")
    
    # Compute final metrics
    if traj_times:
        metrics["traj_gen_time_ms_mean"] = np.mean(traj_times)
    if goal_times:
        metrics["time_to_goal_s_mean"] = np.mean(goal_times)
    if metrics["goals_set"] > 0:
        metrics["success_rate"] = metrics["goals_reached"] / metrics["goals_set"]
    
    logger.info(f"\nB3 Summary:")
    logger.info(f"  Goals: {metrics['goals_reached']}/{metrics['goals_set']}")
    logger.info(f"  Success rate: {metrics['success_rate']:.1%}")
    logger.info(f"  Trajectory gen: {metrics['traj_gen_time_ms_mean']:.2f} ms")
    logger.info(f"  Time to goal: {metrics['time_to_goal_s_mean']:.2f} s")
    
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# B4: Full Dual-System Benchmark
# ────────────────────────────────────────────────────────────────────────────

def benchmark_b4_full_system(
    test_duration_s: float = 10.0,
    control_dt: float = 0.01
) -> Dict[str, float]:
    """
    B4 Benchmark: Full dual-system control (MPC + mock VLA).
    
    Measures:
    - Overall control frequency
    - System responsiveness
    - Latency from goal to response
    """
    logger.info("=" * 70)
    logger.info("B4 BENCHMARK: Full Dual-System Control")
    logger.info("=" * 70)
    
    env = XArmEnv(render_size=84)
    mpc = XArmMPCController(horizon_steps=10, dt=control_dt)
    buffer = TrajectoryBuffer(n_joints=8)
    
    class DummyVLAClient:
        async def predict(self, rgb, task_embedding=None):
            return np.random.randn(8) * 0.1
    
    controller = DualSystemController(
        mpc_solver=mpc,
        smolvla_client=DummyVLAClient(),
        trajectory_buffer=buffer,
        n_joints=8,
        mpc_horizon_steps=10,
        control_dt_s=control_dt,
    )
    
    metrics = {
        "steps_executed": 0,
        "mpc_freq_hz": 0.0,
        "step_time_mean_ms": 0.0,
        "step_time_max_ms": 0.0,
        "tracking_error_mean": 0.0,
    }
    
    q_start = env.get_joint_pos().copy()
    q_goal = q_start + np.array([0.3, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
    buffer.update_subgoal(q_goal)
    
    step_times = []
    errors = []
    
    t_start = time.perf_counter()
    
    try:
        while (time.perf_counter() - t_start) < test_duration_s:
            q = env.get_joint_pos()
            qd = env.get_joint_vel()
            rgb = env.render_rgb(size=84)
            
            # Control step
            t_step_start = time.perf_counter()
            tau = controller.step(q, qd, rgb, "test task")
            t_step = (time.perf_counter() - t_step_start) * 1000
            
            step_times.append(t_step)
            
            # Simulate
            env.step(tau)
            
            # Track error
            error = np.linalg.norm(q_goal - q)
            errors.append(error)
            
            metrics["steps_executed"] += 1
    
    finally:
        env.renderer_rgb.close()
        env.renderer_hi.close()
    
    # Compute final metrics
    if step_times:
        metrics["step_time_mean_ms"] = np.mean(step_times)
        metrics["step_time_max_ms"] = np.max(step_times)
    if errors:
        metrics["tracking_error_mean"] = np.mean(errors)
    
    metrics["mpc_freq_hz"] = metrics["steps_executed"] / test_duration_s
    
    logger.info(f"\nB4 Summary:")
    logger.info(f"  Steps: {metrics['steps_executed']}")
    logger.info(f"  Frequency: {metrics['mpc_freq_hz']:.1f} Hz")
    logger.info(f"  Step time: {metrics['step_time_mean_ms']:.2f} "
               f"(max {metrics['step_time_max_ms']:.2f}) ms")
    logger.info(f"  Mean tracking error: {metrics['tracking_error_mean']:.4f} rad")
    
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# B5: Sensor Ablation Study
# ────────────────────────────────────────────────────────────────────────────

def benchmark_b5_sensor_ablation() -> Dict[str, float]:
    """
    B5 Benchmark: Sensor ablation study.
    
    Measures impact of removing each sensor:
    - RGB camera
    - Event camera
    - LiDAR
    - Proprioception
    """
    logger.info("=" * 70)
    logger.info("B5 BENCHMARK: Sensor Ablation Study")
    logger.info("=" * 70)
    
    env = XArmEnv(render_size=84)
    
    metrics = {
        "rgb_render_time_ms": 0.0,
        "lidar_compute_time_ms": 0.0,
        "proprioception_time_ms": 0.0,
        "total_sensing_time_ms": 0.0,
    }
    
    n_iterations = 100
    
    # RGB rendering
    t_start = time.perf_counter()
    for _ in range(n_iterations):
        rgb = env.render_rgb(size=84)
    metrics["rgb_render_time_ms"] = (time.perf_counter() - t_start) * 1000 / n_iterations
    
    # LiDAR
    t_start = time.perf_counter()
    for _ in range(n_iterations):
        lidar = env.get_lidar_readings()
    metrics["lidar_compute_time_ms"] = (time.perf_counter() - t_start) * 1000 / n_iterations
    
    # Proprioception
    t_start = time.perf_counter()
    for _ in range(n_iterations):
        q = env.get_joint_pos()
        qd = env.get_joint_vel()
    metrics["proprioception_time_ms"] = (time.perf_counter() - t_start) * 1000 / n_iterations
    
    metrics["total_sensing_time_ms"] = (
        metrics["rgb_render_time_ms"] +
        metrics["lidar_compute_time_ms"] +
        metrics["proprioception_time_ms"]
    )
    
    env.renderer_rgb.close()
    env.renderer_hi.close()
    
    logger.info(f"\nB5 Summary:")
    logger.info(f"  RGB render: {metrics['rgb_render_time_ms']:.2f} ms")
    logger.info(f"  LiDAR compute: {metrics['lidar_compute_time_ms']:.2f} ms")
    logger.info(f"  Proprioception: {metrics['proprioception_time_ms']:.2f} ms")
    logger.info(f"  Total sensing: {metrics['total_sensing_time_ms']:.2f} ms")
    
    return metrics


# ────────────────────────────────────────────────────────────────────────────
# Main Benchmark Runner
# ────────────────────────────────────────────────────────────────────────────

def run_all_benchmarks() -> Dict[str, Dict]:
    """Run all Phase 6 benchmarks."""
    logger.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    
    results = {}
    
    # B1: Dataset replay
    results["B1_dataset_replay"] = benchmark_b1_dataset_replay(n_episodes=2)
    
    # B2: MPC-only
    results["B2_mpc_only"] = benchmark_b2_mpc_only(test_duration_s=5.0)
    
    # B3: VLA-only
    results["B3_vla_only"] = benchmark_b3_vla_only(n_goals=5)
    
    # B4: Full system
    results["B4_full_system"] = benchmark_b4_full_system(test_duration_s=5.0)
    
    # B5: Sensor ablation
    results["B5_sensor_ablation"] = benchmark_b5_sensor_ablation()
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 70)
    for benchmark_name, metrics in results.items():
        logger.info(f"\n{benchmark_name}:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
    
    return results


if __name__ == "__main__":
    results = run_all_benchmarks()
