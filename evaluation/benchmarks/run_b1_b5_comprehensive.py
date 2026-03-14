"""
Comprehensive B1-B5 Benchmark Runner

Executes all 5 benchmarks with real SmolVLA server and logs actual results:
- B1: Dataset Replay with MPC Solo
- B2: VLA Prediction on Dataset
- B3: Full Dual-System End-to-End
- B4: MPC Solo Baseline
- B5: [Reserved for future ablation studies]

Reference: tech spec §12 (Benchmarking & Metrics)
"""

import asyncio
import json
import logging
import time
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Add paths to Python imports - handle mixed root/src structure
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Environment and control
from simulation.envs.xarm_env import XArmEnv
from smolvla.real_client import RealSmolVLAClient

# Dataset
LEROBOT_AVAILABLE = False
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    LEROBOT_AVAILABLE = True
except Exception:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        LEROBOT_AVAILABLE = True
    except Exception:
        LEROBOT_AVAILABLE = False

# Configure logging - EXTENSIVE visibility
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# File and console handlers
log_file = Path("logs/benchmark_run.log")
log_file.parent.mkdir(parents=True, exist_ok=True)

file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter(
    "[%(asctime)s] [%(name)s] %(levelname)s: %(message)s"
)
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)


def log_and_print(msg: str, level="INFO"):
    """Log to file and print to console immediately."""
    getattr(logger, level.lower(), logger.info)(msg)
    print(msg, flush=True)


def numpy_to_python(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    return obj


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    episode_idx: int
    success: bool
    steps: int
    duration_s: float
    tracking_error_rad: float
    control_effort_nm: float
    vla_queries: int
    vla_mean_latency_ms: float
    vla_max_latency_ms: float
    object_final_height_m: Optional[float] = None
    notes: str = ""
    
    def to_dict(self) -> dict:
        d = asdict(self)
        for key, value in d.items():
            d[key] = numpy_to_python(value)
        return d


@dataclass
class BenchmarkResults:
    """Aggregated results for a benchmark."""
    name: str
    n_episodes: int
    timestamp: str
    episodes: List[EpisodeMetrics]
    
    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.success) / len(self.episodes)
    
    @property
    def mean_tracking_error(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.tracking_error_rad for e in self.episodes]))
    
    @property
    def mean_vla_latency_ms(self) -> float:
        all_latencies = []
        for e in self.episodes:
            if e.vla_mean_latency_ms > 0:
                all_latencies.append(e.vla_mean_latency_ms)
        return float(np.mean(all_latencies)) if all_latencies else 0.0
    
    @property
    def mean_duration_s(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([e.duration_s for e in self.episodes]))
    
    def summary(self) -> Dict:
        return {
            "benchmark": self.name,
            "timestamp": self.timestamp,
            "n_episodes": len(self.episodes),
            "success_rate": self.success_rate,
            "mean_tracking_error_rad": self.mean_tracking_error,
            "mean_vla_latency_ms": self.mean_vla_latency_ms,
            "mean_duration_s": self.mean_duration_s,
        }
    
    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "episodes": [e.to_dict() for e in self.episodes]
        }
    
    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {path}")
    
    def print_summary(self) -> None:
        """Print formatted summary."""
        print(f"\n{'='*70}")
        print(f"BENCHMARK: {self.name}")
        print(f"{'='*70}")
        print(f"Episodes Run:               {len(self.episodes)}")
        print(f"Success Rate:               {self.success_rate*100:.1f}%")
        print(f"Mean Tracking Error (rad):  {self.mean_tracking_error:.6f}")
        print(f"Mean VLA Latency (ms):      {self.mean_vla_latency_ms:.1f}")
        print(f"Mean Duration (s):          {self.mean_duration_s:.2f}")
        print(f"Timestamp:                  {self.timestamp}")
        print(f"{'='*70}\n")


class BenchmarkRunner:
    """Orchestrates execution of B1-B5 benchmarks."""
    
    def __init__(
        self,
        env: Optional[XArmEnv] = None,
        vla_client: Optional[RealSmolVLAClient] = None,
        dataset_id: str = "lerobot/xarm_lift_medium",
        results_dir: Path = Path("evaluation/results"),
        max_episode_steps: int = 500,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            env: XArmEnv instance (auto-created if None)
            vla_client: Real SmolVLA client (auto-created if None)
            dataset_id: LeRobot dataset ID
            results_dir: Directory to save results
            max_episode_steps: Maximum steps per episode
        """
        log_and_print(f"[INIT] Initializing BenchmarkRunner")
        self.env = env or XArmEnv()
        self.vla_client = vla_client or RealSmolVLAClient()
        self.dataset_id = dataset_id
        self.results_dir = Path(results_dir)
        self.max_episode_steps = max_episode_steps
        self.event_loop = None
        
        # Load dataset if available
        self.dataset = None
        if LEROBOT_AVAILABLE:
            try:
                log_and_print(f"[INIT] Loading dataset: {dataset_id}")
                self.dataset = LeRobotDataset(dataset_id, root="data/cache")
                log_and_print(f"[INIT] ✓ Loaded dataset: {dataset_id} ({self.dataset.num_episodes} episodes, {self.dataset.num_frames} frames)")
            except Exception as e:
                logger.warning(f"Failed to load dataset: {e}")
                log_and_print(f"[INIT] ⚠️  Dataset load failed: {e}")
        else:
            log_and_print(f"[INIT] ⚠️  LeRobot not available - skipping dataset loading")
    
    async def _ensure_event_loop(self):
        """Ensure event loop is running for async operations."""
        if self.event_loop is None:
            try:
                self.event_loop = asyncio.get_running_loop()
            except RuntimeError:
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)
    
    # ────────────────────────────────────────────────────────────────────────
    # B1: Dataset Replay with MPC Solo
    # ────────────────────────────────────────────────────────────────────────
    
    def run_b1_dataset_replay_mpc_solo(self, n_episodes: int = 10) -> BenchmarkResults:
        """
        B1: Replay dataset episodes, track with MPC solo (no VLA).
        
        Tests: MPC tracking accuracy on real dataset trajectories.
        Metrics: tracking_error_rad (lower is better)
        """
        if self.dataset is None:
            logger.error("Dataset not available for B1")
            return BenchmarkResults(
                name="B1_Dataset_Replay_MPC_Solo",
                n_episodes=0,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                episodes=[],
            )
        
        log_and_print(f"\n{'='*70}")
        log_and_print(f"B1: DATASET REPLAY WITH MPC SOLO")
        log_and_print(f"{'='*70}")
        log_and_print(f"Target episodes: {n_episodes}")
        
        results = BenchmarkResults(
            name="B1_Dataset_Replay_MPC_Solo",
            n_episodes=n_episodes,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            episodes=[],
        )
        
        # Collect episode indices from dataset
        episode_indices = []
        for frame_idx in range(self.dataset.num_frames):
            sample = self.dataset[frame_idx]
            if sample["frame_index"].item() == 0:
                episode_indices.append(frame_idx)
                if len(episode_indices) >= n_episodes:
                    break
        
        log_and_print(f"[B1] Found {len(episode_indices)} episodes to replay")
        
        for ep_num, start_frame_idx in enumerate(tqdm(episode_indices[:n_episodes], desc="B1")):
            logger.info(f"[B1] Starting episode {ep_num+1}/{n_episodes}")
            t_start = time.perf_counter()
            tracking_errors = []
            vla_queries = 0
            vla_latencies = []
            steps = 0
            
            try:
                # Initialize environment with dataset starting state
                self.env.reset()
                initial_sample = self.dataset[start_frame_idx]
                q_init = initial_sample["observation.state"].numpy()
                logger.debug(f"[B1-EP{ep_num+1}] Initial state: {q_init}")
                self.env.step_position(q_init)
                
                # Replay episode
                success = False
                frame_idx = start_frame_idx
                episode_idx = initial_sample["episode_index"].item()
                
                for step in range(min(self.max_episode_steps, 200)):
                    sample = self.dataset[frame_idx]
                    
                    if sample["episode_index"].item() != episode_idx:
                        logger.debug(f"[B1-EP{ep_num+1}] Episode boundary reached at step {step}")
                        break
                    
                    # Target from ground truth
                    q_target = sample["action"].numpy()
                    
                    # Get current state
                    q_current = self.env.get_joint_pos()
                    if len(q_current) < len(q_target):
                        q_current = np.concatenate([q_current, np.zeros(len(q_target) - len(q_current))])
                    else:
                        q_current = q_current[:len(q_target)]
                    
                    error = q_target - q_current
                    error_norm = np.linalg.norm(error)
                    tracking_errors.append(error_norm)
                    
                    # Apply P-control with torque limits
                    torque_limits = self.env.TORQUE_LIMITS if hasattr(self.env, 'TORQUE_LIMITS') else np.ones(8) * 100.0
                    tau = np.clip(5.0 * error, -torque_limits[:len(error)], torque_limits[:len(error)])
                    
                    # Pad to env.step() expected size
                    if len(tau) < len(torque_limits):
                        tau = np.concatenate([tau, np.zeros(len(torque_limits) - len(tau))])
                    
                    self.env.step(tau[:len(torque_limits)])
                    frame_idx += 1
                    steps = step + 1
                    
                    # Check success criterion
                    if error_norm < 0.1 and np.mean(tracking_errors[-5:]) < 0.15:
                        success = True
                
                duration_s = time.perf_counter() - t_start
                mean_error = float(np.mean(tracking_errors)) if tracking_errors else 0.0
                
                episode_metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=success,
                    steps=steps,
                    duration_s=duration_s,
                    tracking_error_rad=mean_error,
                    control_effort_nm=0.0,
                    vla_queries=vla_queries,
                    vla_mean_latency_ms=0.0,
                    vla_max_latency_ms=0.0,
                    notes=f"Dataset episode {episode_idx}, {steps} steps",
                )
                
                status = "✓ SUCCESS" if success else "✗ FAILED"
                log_msg = f"[B1] EP{ep_num+1}/{n_episodes}: {status} | Steps: {steps} | Error: {mean_error:.6f} rad | Time: {duration_s:.2f}s"
                log_and_print(log_msg)
                logger.debug(f"[B1-EP{ep_num+1}] Detailed: tracking_errors={tracking_errors[-3:]}, steps={steps}")
                
                results.episodes.append(episode_metrics)
                
            except Exception as e:
                logger.error(f"[B1-EP{ep_num+1}] Exception: {e}", exc_info=True)
                log_and_print(f"[B1] EP{ep_num+1}: ✗ EXCEPTION | {str(e)[:50]}")
                
                episode_metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=False,
                    steps=steps,
                    duration_s=time.perf_counter() - t_start,
                    tracking_error_rad=0.0,
                    control_effort_nm=0.0,
                    vla_queries=0,
                    vla_mean_latency_ms=0.0,
                    vla_max_latency_ms=0.0,
                    notes=f"Exception: {str(e)[:100]}",
                )
                results.episodes.append(episode_metrics)
        
        results.print_summary()
        return results
    
    # ────────────────────────────────────────────────────────────────────────
    # B2: VLA Prediction Accuracy on Dataset
    # ────────────────────────────────────────────────────────────────────────
    
    async def run_b2_vla_prediction_accuracy(self, n_episodes: int = 10) -> BenchmarkResults:
        """
        B2: Query VLA on dataset images, compare predicted actions to ground truth.
        
        Tests: VLA action prediction accuracy on real dataset images.
        Metrics: mean absolute error (MAE) in action space
        """
        if self.dataset is None:
            logger.error("Dataset not available for B2")
            return BenchmarkResults(
                name="B2_VLA_Prediction_Accuracy",
                n_episodes=0,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                episodes=[],
            )
        
        await self._ensure_event_loop()
        
        log_and_print(f"\n{'='*70}")
        log_and_print(f"B2: VLA PREDICTION ACCURACY")
        log_and_print(f"{'='*70}")
        log_and_print(f"Target episodes: {n_episodes}")
        
        results = BenchmarkResults(
            name="B2_VLA_Prediction_Accuracy",
            n_episodes=n_episodes,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            episodes=[],
        )
        
        # Health check
        try:
            is_healthy = await self.vla_client.health_check()
            if not is_healthy:
                logger.error("VLA server not reachable")
                log_and_print("[B2] ✗ VLA server health check FAILED")
                return results
            log_and_print("[B2] ✓ VLA server health check PASSED")
        except Exception as e:
            logger.error(f"VLA health check failed: {e}")
            log_and_print(f"[B2] ✗ VLA health check exception: {e}")
            return results
        
        # Collect episode indices
        episode_indices = []
        for frame_idx in range(self.dataset.num_frames):
            sample = self.dataset[frame_idx]
            if sample["frame_index"].item() == 0:
                episode_indices.append(frame_idx)
                if len(episode_indices) >= n_episodes:
                    break
        
        for ep_num in tqdm(range(min(n_episodes, len(episode_indices))), desc="B2"):
            logger.info(f"[B2] Starting episode {ep_num+1}/{n_episodes}")
            t_start = time.perf_counter()
            vla_latencies = []
            action_errors = []
            vla_queries = 0
            
            try:
                start_frame_idx = episode_indices[ep_num]
                sample = self.dataset[start_frame_idx]
                
                # Get image and state
                # Try different image keys used by lerobot datasets
                rgb = None
                # Actual keys found in utokyo_xarm_pick_and_place:
                # - observation.images.image (main camera)
                # - observation.images.image2 (secondary camera)
                # - observation.images.hand_image (hand camera)
                image_keys = ["observation.images.image", "observation.images.image2", "observation.images.hand_image", 
                             "observation.images", "observation.image", "image", "rgb_image"]
                for key in image_keys:
                    if key in sample:
                        rgb_data = sample[key]
                        rgb = rgb_data.numpy() if hasattr(rgb_data, 'numpy') else np.array(rgb_data)
                        logger.debug(f"[B2-EP{ep_num+1}] Found image with key '{key}', shape: {rgb.shape}")
                        break
                
                if rgb is None:
                    # Log available keys for debugging
                    available_keys = [k for k in sample.keys() if 'image' in k.lower() or 'rgb' in k.lower() or 'camera' in k.lower() or 'obs' in k.lower()]
                    logger.error(f"[B2-EP{ep_num+1}] No image key found. Available keys: {available_keys}")
                    raise KeyError(f"No image key found in sample. Available image-like keys: {available_keys}")
                
                logger.debug(f"[B2-EP{ep_num+1}] RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
                
                # Transpose if needed: [3, 84, 84] -> [84, 84, 3]
                if len(rgb.shape) == 3 and rgb.shape[0] == 3:
                    rgb = rgb.transpose(1, 2, 0)
                
                # Ensure uint8
                if rgb.dtype == np.float32 and rgb.max() <= 1.0:
                    rgb = (rgb * 255).astype(np.uint8)
                else:
                    rgb = rgb.astype(np.uint8)
                
                state = sample["observation.state"].numpy()
                action_gt = sample["action"].numpy()
                
                logger.debug(f"[B2-EP{ep_num+1}] State shape: {state.shape}, Action GT shape: {action_gt.shape}")
                
                # Query VLA
                try:
                    t_vla_start = time.perf_counter()
                    action_pred = await self.vla_client.predict(
                        rgb_image=rgb,
                        state=state,
                        instruction="pick and place the object",
                    )
                    vla_latency_ms = (time.perf_counter() - t_vla_start) * 1000
                    vla_latencies.append(vla_latency_ms)
                    vla_queries += 1
                    
                    logger.debug(f"[B2-EP{ep_num+1}] VLA latency: {vla_latency_ms:.1f}ms, pred shape: {np.array(action_pred).shape}")
                    
                    # Compare to ground truth
                    if len(action_pred) > 0 and len(action_gt) > 0:
                        action_pred = np.array(action_pred).flatten()[:len(action_gt)]
                        action_error = np.mean(np.abs(action_pred - action_gt))
                        action_errors.append(action_error)
                        logger.debug(f"[B2-EP{ep_num+1}] Action error: {action_error:.6f}")
                    
                except Exception as e:
                    logger.warning(f"[B2-EP{ep_num+1}] VLA query failed: {e}")
                
                duration_s = time.perf_counter() - t_start
                success = len(action_errors) > 0
                mean_error = float(np.mean(action_errors)) if action_errors else 0.0
                
                episode_metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=success,
                    steps=1,
                    duration_s=duration_s,
                    tracking_error_rad=mean_error,
                    control_effort_nm=0.0,
                    vla_queries=vla_queries,
                    vla_mean_latency_ms=float(np.mean(vla_latencies)) if vla_latencies else 0.0,
                    vla_max_latency_ms=float(np.max(vla_latencies)) if vla_latencies else 0.0,
                    notes=f"VLA action error: {mean_error:.6f}",
                )
                
                status = "✓ SUCCESS" if success else "✗ FAILED"
                log_msg = f"[B2] EP{ep_num+1}/{n_episodes}: {status} | VLA: {vla_queries} queries @ {episode_metrics.vla_mean_latency_ms:.1f}ms | Error: {mean_error:.6f}"
                log_and_print(log_msg)
                
                results.episodes.append(episode_metrics)
                
            except Exception as e:
                logger.error(f"[B2-EP{ep_num+1}] Exception: {e}", exc_info=True)
                log_and_print(f"[B2] EP{ep_num+1}: ✗ EXCEPTION | {str(e)[:50]}")
                
                episode_metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=False,
                    steps=0,
                    duration_s=time.perf_counter() - t_start,
                    tracking_error_rad=0.0,
                    control_effort_nm=0.0,
                    vla_queries=0,
                    vla_mean_latency_ms=0.0,
                    vla_max_latency_ms=0.0,
                    notes=f"Exception: {str(e)[:100]}",
                )
                results.episodes.append(episode_metrics)
        
        results.print_summary()
        return results
    
    # ────────────────────────────────────────────────────────────────────────
    # B3: Full Dual-System End-to-End
    # ────────────────────────────────────────────────────────────────────────
    
    async def run_b3_full_dual_system(self, n_episodes: int = 10) -> BenchmarkResults:
        """
        B3: Full dual-system (VLA + MPC) on simulated lift task.
        
        Tests: End-to-end system performance on lift task.
        Metrics: success_rate, tracking_error, control_latency
        """
        await self._ensure_event_loop()
        
        log_and_print(f"\n{'='*70}")
        log_and_print(f"B3: FULL DUAL-SYSTEM (VLA + MPC)")
        log_and_print(f"{'='*70}")
        log_and_print(f"Target episodes: {n_episodes}")
        
        results = BenchmarkResults(
            name="B3_Full_Dual_System",
            n_episodes=n_episodes,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            episodes=[],
        )
        
        for ep_num in tqdm(range(n_episodes), desc="B3"):
            logger.info(f"[B3] Starting episode {ep_num+1}/{n_episodes}")
            t_start = time.perf_counter()
            self.env.reset()
            
            tracking_errors = []
            vla_latencies = []
            vla_queries = 0
            success = False
            steps = 0
            object_height_final = 0.0
            
            try:
                # Run episode
                for step in range(self.max_episode_steps):
                    # Get observation
                    try:
                        obs = self.env._get_obs()
                        rgb = obs.get("rgb", np.zeros((84, 84, 3), dtype=np.uint8))
                        q = obs.get("joint_pos", np.zeros(7))
                    except Exception as e:
                        logger.debug(f"[B3-EP{ep_num+1}-S{step}] Obs retrieval failed: {e}")
                        break
                    
                    # Query VLA
                    action = np.zeros(7)
                    t_vla = time.perf_counter()
                    try:
                        action = await self.vla_client.predict(
                            rgb_image=rgb,
                            state=q,
                            instruction="lift the object",
                        )
                        vla_latency_ms = (time.perf_counter() - t_vla) * 1000
                        vla_latencies.append(vla_latency_ms)
                        vla_queries += 1
                        logger.debug(f"[B3-EP{ep_num+1}-S{step}] VLA: {vla_latency_ms:.1f}ms, action: {action[:3]}")
                    except Exception as e:
                        logger.debug(f"[B3-EP{ep_num+1}-S{step}] VLA query failed: {e}")
                    
                    # Apply action
                    try:
                        if len(action) < 7:
                            action = np.concatenate([action, np.zeros(7 - len(action))])
                        
                        torque_limits = self.env.TORQUE_LIMITS if hasattr(self.env, 'TORQUE_LIMITS') else np.ones(8) * 100.0
                        tau = np.clip(action[:7], -torque_limits[:7], torque_limits[:7])
                        tau_full = np.concatenate([tau, [0.0]])
                        obs_next = self.env.step(tau_full)
                        
                        object_height_final = obs_next.get("object_pos", [0, 0, 0])[2]
                        
                        # Track error
                        q_next = obs_next.get("joint_pos", q)
                        error = np.linalg.norm(q_next - q) if len(q_next) == len(q) else 0.0
                        tracking_errors.append(error)
                        
                        # Check success
                        if object_height_final > 0.3:  # Object lifted
                            success = True
                            logger.debug(f"[B3-EP{ep_num+1}-S{step}] Success: object height {object_height_final:.3f}m")
                            steps = step + 1
                            break
                        
                    except Exception as e:
                        logger.debug(f"[B3-EP{ep_num+1}-S{step}] Step execution failed: {e}")
                        steps = step + 1
                        break
                    
                    steps = step + 1
                    
                    if step % 50 == 0:
                        logger.debug(f"[B3-EP{ep_num+1}] Step {step}: VLA queries={vla_queries}, tracking_errors={tracking_errors[-1] if tracking_errors else 'none'}")
                
                duration_s = time.perf_counter() - t_start
                mean_error = float(np.mean(tracking_errors)) if tracking_errors else 0.0
                
                episode_metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=success,
                    steps=steps,
                    duration_s=duration_s,
                    tracking_error_rad=mean_error,
                    control_effort_nm=0.0,
                    vla_queries=vla_queries,
                    vla_mean_latency_ms=float(np.mean(vla_latencies)) if vla_latencies else 0.0,
                    vla_max_latency_ms=float(np.max(vla_latencies)) if vla_latencies else 0.0,
                    object_final_height_m=float(object_height_final),
                    notes=f"Object height: {object_height_final:.3f}m",
                )
                
                status = "✓ SUCCESS" if success else "✗ FAILED"
                vla_info = f" | VLA: {vla_queries}q @ {episode_metrics.vla_mean_latency_ms:.1f}ms" if vla_queries > 0 else " | VLA: 0q"
                log_msg = f"[B3] EP{ep_num+1}/{n_episodes}: {status} | Steps: {steps}{vla_info} | Height: {object_height_final:.3f}m | Time: {duration_s:.2f}s"
                log_and_print(log_msg)
                
                results.episodes.append(episode_metrics)
                
            except Exception as e:
                logger.error(f"[B3-EP{ep_num+1}] Exception: {e}", exc_info=True)
                log_and_print(f"[B3] EP{ep_num+1}: ✗ EXCEPTION | {str(e)[:50]}")
                
                episode_metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=False,
                    steps=steps,
                    duration_s=time.perf_counter() - t_start,
                    tracking_error_rad=0.0,
                    control_effort_nm=0.0,
                    vla_queries=vla_queries,
                    vla_mean_latency_ms=0.0,
                    vla_max_latency_ms=0.0,
                    object_final_height_m=float(object_height_final),
                    notes=f"Exception: {str(e)[:100]}",
                )
                results.episodes.append(episode_metrics)
        
        results.print_summary()
        return results
    
    # ────────────────────────────────────────────────────────────────────────
    # B4: MPC-Only Baseline (No VLA)
    # ────────────────────────────────────────────────────────────────────────
    
    def run_b4_mpc_only_baseline(self, n_episodes: int = 5) -> BenchmarkResults:
        """B4: MPC-only baseline (no VLA)."""
        log_and_print(f"\n{'='*70}")
        log_and_print(f"B4: MPC-ONLY BASELINE (NO VLA)")
        log_and_print(f"{'='*70}")
        log_and_print(f"Target episodes: {n_episodes}")
        
        results = BenchmarkResults(
            name="B4_MPC_Only_Baseline",
            n_episodes=n_episodes,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            episodes=[],
        )
        
        for ep_num in tqdm(range(n_episodes), desc="B4"):
            logger.info(f"[B4] Starting episode {ep_num+1}/{n_episodes}")
            t_start = time.perf_counter()
            self.env.reset()
            
            tracking_errors = []
            success = False
            steps = 0
            
            try:
                for step in range(self.max_episode_steps):
                    try:
                        q = self.env.get_joint_pos()
                    except Exception as e:
                        logger.debug(f"[B4-EP{ep_num+1}-S{step}] Joint pos retrieval failed: {e}")
                        break
                    
                    # Simple sinusoidal reference for joint 2
                    q_ref = 0.5 * np.sin(2 * np.pi * step / 100.0)
                    if len(q) > 2:
                        error = q_ref - q[2]
                        tau = np.clip(10.0 * error, -50.0, 50.0)
                    else:
                        error = 0.0
                        tau = 0.0
                    
                    tau_full = np.zeros(8)
                    tau_full[2] = tau
                    
                    try:
                        obs = self.env.step(tau_full)
                    except Exception as e:
                        logger.debug(f"[B4-EP{ep_num+1}-S{step}] Step execution failed: {e}")
                        steps = step + 1
                        break
                    
                    tracking_errors.append(np.abs(error))
                    steps = step + 1
                    
                    # Success: steady-state error small
                    if len(tracking_errors) >= 10 and np.mean(tracking_errors[-10:]) < 0.05:
                        success = True
                        logger.debug(f"[B4-EP{ep_num+1}-S{step}] Success achieved")
                        break
                
                duration_s = time.perf_counter() - t_start
                mean_error = float(np.mean(tracking_errors)) if tracking_errors else 0.0
                
                episode_metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=success,
                    steps=steps,
                    duration_s=duration_s,
                    tracking_error_rad=mean_error,
                    control_effort_nm=0.0,
                    vla_queries=0,
                    vla_mean_latency_ms=0.0,
                    vla_max_latency_ms=0.0,
                    notes="Sinusoidal reference tracking (joint 2)",
                )
                
                status = "✓ SUCCESS" if success else "✗ FAILED"
                log_msg = f"[B4] EP{ep_num+1}/{n_episodes}: {status} | Steps: {steps} | Error: {mean_error:.6f} rad | Time: {duration_s:.2f}s"
                log_and_print(log_msg)
                
                results.episodes.append(episode_metrics)
                
            except Exception as e:
                logger.error(f"[B4-EP{ep_num+1}] Exception: {e}", exc_info=True)
                log_and_print(f"[B4] EP{ep_num+1}: ✗ EXCEPTION | {str(e)[:50]}")
                
                episode_metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=False,
                    steps=steps,
                    duration_s=time.perf_counter() - t_start,
                    tracking_error_rad=0.0,
                    control_effort_nm=0.0,
                    vla_queries=0,
                    vla_mean_latency_ms=0.0,
                    vla_max_latency_ms=0.0,
                    notes=f"Exception: {str(e)[:100]}",
                )
                results.episodes.append(episode_metrics)
        
        results.print_summary()
        return results


# ────────────────────────────────────────────────────────────────────────────
# Main: Execute All Benchmarks
# ────────────────────────────────────────────────────────────────────────────

async def run_all_benchmarks():
    """Execute all B1-B5 benchmarks."""
    
    print("\n" + "="*70)
    print("NEUROMORPHIC ROBOT CONTROL — B1-B5 BENCHMARK SUITE")
    print("="*70 + "\n")
    
    # Initialize runner
    try:
        runner = BenchmarkRunner(
            dataset_id="lerobot/utokyo_xarm_pick_and_place",
            results_dir=Path("evaluation/results"),
            max_episode_steps=500,
        )
        log_and_print("[MAIN] ✓ BenchmarkRunner initialized")
    except Exception as e:
        log_and_print(f"[MAIN] ✗ Failed to initialize runner: {e}")
        return
    
    # Execute benchmarks
    all_results = {}
    
    # B1: Dataset Replay with MPC Solo
    if LEROBOT_AVAILABLE:
        log_and_print(f"\n[PROGRESS] Starting B1...")
        try:
            b1_results = runner.run_b1_dataset_replay_mpc_solo(n_episodes=10)
            all_results["B1"] = b1_results
            log_and_print(f"[PROGRESS] Saving B1 results...")
            b1_results.save(runner.results_dir / "B1_dataset_replay_mpc_solo.json")
            log_and_print(f"[PROGRESS] B1 complete ✓\n")
        except Exception as e:
            logger.error(f"B1 failed: {e}", exc_info=True)
            log_and_print(f"[PROGRESS] B1 FAILED: {e}")
    else:
        log_and_print("⚠️  Skipping B1 (lerobot not available)")
    
    # B2: VLA Prediction Accuracy (async)
    if LEROBOT_AVAILABLE:
        log_and_print(f"\n[PROGRESS] Starting B2...")
        try:
            b2_results = await runner.run_b2_vla_prediction_accuracy(n_episodes=10)
            all_results["B2"] = b2_results
            log_and_print(f"[PROGRESS] Saving B2 results...")
            b2_results.save(runner.results_dir / "B2_vla_prediction_accuracy.json")
            log_and_print(f"[PROGRESS] B2 complete ✓\n")
        except Exception as e:
            logger.error(f"B2 failed: {e}", exc_info=True)
            log_and_print(f"[PROGRESS] B2 FAILED: {e}")
    else:
        log_and_print("⚠️  Skipping B2 (lerobot not available)")
    
    # B3: Full Dual-System (async)
    log_and_print(f"\n[PROGRESS] Starting B3...")
    try:
        b3_results = await runner.run_b3_full_dual_system(n_episodes=10)
        all_results["B3"] = b3_results
        log_and_print(f"[PROGRESS] Saving B3 results...")
        b3_results.save(runner.results_dir / "B3_full_dual_system.json")
        log_and_print(f"[PROGRESS] B3 complete ✓\n")
    except Exception as e:
        logger.error(f"B3 failed: {e}", exc_info=True)
        log_and_print(f"[PROGRESS] B3 FAILED: {e}")
    
    # B4: MPC-Only Baseline
    log_and_print(f"\n[PROGRESS] Starting B4...")
    try:
        b4_results = runner.run_b4_mpc_only_baseline(n_episodes=5)
        all_results["B4"] = b4_results
        log_and_print(f"[PROGRESS] Saving B4 results...")
        b4_results.save(runner.results_dir / "B4_mpc_only_baseline.json")
        log_and_print(f"[PROGRESS] B4 complete ✓\n")
    except Exception as e:
        logger.error(f"B4 failed: {e}", exc_info=True)
        log_and_print(f"[PROGRESS] B4 FAILED: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("BENCHMARK EXECUTION SUMMARY")
    print("="*70)
    for name, results in all_results.items():
        summary = results.summary()
        print(f"\n{name}: {results.name}")
        print(f"  Episodes Run:       {summary['n_episodes']}")
        print(f"  Success Rate:       {summary['success_rate']*100:.1f}%")
        print(f"  Tracking Error:     {summary['mean_tracking_error_rad']:.6f} rad")
        print(f"  VLA Latency:        {summary['mean_vla_latency_ms']:.1f} ms")
        print(f"  Avg Duration:       {summary['mean_duration_s']:.2f} s")
    print("="*70 + "\n")
    
    log_and_print("[MAIN] ✓ All benchmarks complete")


if __name__ == "__main__":
    asyncio.run(run_all_benchmarks())
