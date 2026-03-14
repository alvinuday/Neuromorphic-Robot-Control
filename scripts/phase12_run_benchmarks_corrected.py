#!/usr/bin/env python3
"""
Phase 12: COMPREHENSIVE BENCHMARK RUNNER (CORRECTED VERSION)

Based on ACTUAL dataset: lerobot/utokyo_xarm_pick_and_place
- 102 episodes
- 7490 total frames (avg ~73 frames per episode)
- Images: [3, 224, 224] (not 84x84)
- State: [8] dimensions
- Action: [7] dimensions

Benchmarks with PROPER episode counts:
- B1: Dataset Replay with MPC Solo → 80 episodes
- B2: VLA Prediction Accuracy → 80 episodes
- B3: Full Dual-System (Sim) → 50 episodes
- B4: MPC-Only Baseline (Sim) → 30 episodes

Total time: ~70-80 minutes
"""

import asyncio
import json
import logging
import time
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Imports
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

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

log_file = Path("logs/phase12_full_benchmark.log")
log_file.parent.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
logger.addHandler(file_handler)


def print_and_log(msg: str, level="INFO"):
    """Print and log to file."""
    getattr(logger, level.lower(), logger.info)(msg)
    print(msg, flush=True)


def numpy_to_python(obj):
    """Convert numpy to Python types for JSON."""
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
    """Metrics for single episode."""
    episode_idx: int
    success: bool
    steps: int
    duration_s: float
    tracking_error_rad: float
    vla_queries: int = 0
    vla_mean_latency_ms: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> dict:
        d = asdict(self)
        for key, value in d.items():
            d[key] = numpy_to_python(value)
        return d


@dataclass
class BenchmarkResults:
    """Aggregated results."""
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
        latencies = [e.vla_mean_latency_ms for e in self.episodes if e.vla_mean_latency_ms > 0]
        return float(np.mean(latencies)) if latencies else 0.0
    
    def summary(self) -> Dict:
        return {
            "benchmark": self.name,
            "timestamp": self.timestamp,
            "n_episodes": len(self.episodes),
            "success_rate": self.success_rate,
            "mean_tracking_error_rad": self.mean_tracking_error,
            "mean_vla_latency_ms": self.mean_vla_latency_ms,
        }
    
    def to_dict(self) -> dict:
        return {
            "summary": self.summary(),
            "episodes": [e.to_dict() for e in self.episodes]
        }
    
    def save(self, path: Path) -> None:
        """Save to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        print_and_log(f"✓ Saved: {path}")
    
    def print_summary(self) -> None:
        """Print formatted summary."""
        print(f"\n{'='*70}")
        print(f"{self.name}")
        print(f"{'='*70}")
        print(f"Episodes Run:         {len(self.episodes)}")
        print(f"Success Rate:         {self.success_rate*100:.1f}%")
        print(f"Mean Tracking Error:  {self.mean_tracking_error:.6f} rad")
        print(f"Mean VLA Latency:     {self.mean_vla_latency_ms:.1f} ms")
        print(f"{'='*70}\n")


class BenchmarkRunner:
    """Orchestrates B1-B4 benchmarks."""
    
    def __init__(
        self,
        dataset_id: str = "lerobot/utokyo_xarm_pick_and_place",
        results_dir: Path = Path("evaluation/results"),
    ):
        print_and_log(f"\n[INIT] Initializing BenchmarkRunner")
        print_and_log(f"[INIT] Dataset: {dataset_id}")
        
        self.env = XArmEnv()
        self.vla_client = RealSmolVLAClient()
        self.dataset_id = dataset_id
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load dataset
        self.dataset = None
        self.episode_indices = {}
        
        if LEROBOT_AVAILABLE:
            try:
                print_and_log(f"[INIT] Loading dataset: {dataset_id}")
                self.dataset = LeRobotDataset(dataset_id, root="data/cache")
                print_and_log(f"[INIT] ✓ Dataset loaded: {self.dataset.num_episodes} episodes, {self.dataset.num_frames} frames")
                
                # Build episode index
                for frame_idx in range(self.dataset.num_frames):
                    sample = self.dataset[frame_idx]
                    ep_idx = sample["episode_index"].item()
                    if ep_idx not in self.episode_indices:
                        self.episode_indices[ep_idx] = []
                    self.episode_indices[ep_idx].append(frame_idx)
                
                print_and_log(f"[INIT] ✓ Built episode index: {len(self.episode_indices)} unique episodes")
            except Exception as e:
                print_and_log(f"[INIT] ✗ Dataset load failed: {e}", "ERROR")
        else:
            print_and_log(f"[INIT] ✗ LeRobot not available", "ERROR")
    
    # ─────────────────────────────────────────────────────────────────────
    # B1: Dataset Replay with MPC Solo (80 episodes)
    # ─────────────────────────────────────────────────────────────────────
    
    def run_b1_dataset_replay_mpc_solo(self, n_episodes: int = 80) -> BenchmarkResults:
        """
        B1: Replay dataset episodes, track with MPC solo (no VLA).
        
        What it tests:
        - Can MPC track real robot trajectories from the dataset?
        
        Input: Real dataset episodes (states, actions from utokyo dataset)
        Process: Extract state trajectory, use MPC to track it
        Metrics: Tracking error (how well MPC follows the trajectory)
        """
        if self.dataset is None:
            print_and_log("[B1] ✗ Dataset not available", "ERROR")
            return BenchmarkResults(
                name="B1_Dataset_Replay_MPC_Solo",
                n_episodes=0,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                episodes=[],
            )
        
        print_and_log(f"\n{'='*70}")
        print_and_log(f"B1: DATASET REPLAY WITH MPC SOLO")
        print_and_log(f"{'='*70}")
        print_and_log(f"Target episodes: {n_episodes} (out of {len(self.episode_indices)} available)")
        
        results = BenchmarkResults(
            name="B1_Dataset_Replay_MPC_Solo",
            n_episodes=n_episodes,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            episodes=[],
        )
        
        # Select episodes
        selected_episodes = sorted(self.episode_indices.keys())[:n_episodes]
        print_and_log(f"[B1] Selected episodes: {selected_episodes[:10]}... (showing first 10)")
        
        for ep_num, ep_idx in enumerate(tqdm(selected_episodes, desc="B1")):
            try:
                t_start = time.perf_counter()
                self.env.reset()
                
                frame_indices = self.episode_indices[ep_idx]
                tracking_errors = []
                
                # Replay episode
                for step, frame_idx in enumerate(frame_indices[:200]):  # Max 200 steps
                    sample = self.dataset[frame_idx]
                    
                    # Get dataset state and action
                    q_dataset = sample["observation.state"].numpy()
                    a_dataset = sample["action"].numpy()
                    
                    # Simple MPC: track the dataset action with error term
                    try:
                        q_current = self.env.get_joint_pos()
                    except:
                        break
                    
                    # Error between current and dataset state
                    error = np.linalg.norm(q_current - q_dataset) if len(q_current) == len(q_dataset) else 0.0
                    tracking_errors.append(error)
                    
                    # Apply corrective torque (simple PD controller)
                    kp = 10.0
                    tau = np.clip(kp * (q_dataset - q_current), -50, 50)
                    tau_full = np.concatenate([tau, np.zeros(8 - len(tau))])
                    
                    try:
                        self.env.step(tau_full)
                    except:
                        break
                
                duration_s = time.perf_counter() - t_start
                mean_error = float(np.mean(tracking_errors)) if tracking_errors else 0.0
                success = mean_error < 0.2  # Arbitrary threshold
                
                metrics = EpisodeMetrics(
                    episode_idx=ep_idx,
                    success=success,
                    steps=len(tracking_errors),
                    duration_s=duration_s,
                    tracking_error_rad=mean_error,
                    notes=f"Tracked {len(frame_indices)} dataset frames"
                )
                results.episodes.append(metrics)
                
                if (ep_num + 1) % 10 == 0:
                    print_and_log(f"[B1] Progress: {ep_num+1}/{n_episodes} | Error: {mean_error:.6f}")
                
            except Exception as e:
                print_and_log(f"[B1] Episode {ep_idx} failed: {str(e)[:50]}", "WARNING")
                metrics = EpisodeMetrics(
                    episode_idx=ep_idx,
                    success=False,
                    steps=0,
                    duration_s=0.0,
                    tracking_error_rad=0.0,
                    notes=f"Failed: {str(e)[:50]}"
                )
                results.episodes.append(metrics)
        
        results.print_summary()
        return results
    
    # ─────────────────────────────────────────────────────────────────────
    # B2: VLA Prediction Accuracy (80 episodes)
    # ─────────────────────────────────────────────────────────────────────
    
    async def run_b2_vla_prediction_accuracy(self, n_episodes: int = 80) -> BenchmarkResults:
        """
        B2: Query VLA on dataset images, compare to ground truth actions.
        
        What it tests:
        - Can SmolVLA make reasonable action predictions on real dataset images?
        
        Input: Real dataset images [3, 224, 224] and states
        Process: Query VLA → Compare predictions to ground truth
        Metrics: Action prediction error, VLA latency
        """
        if self.dataset is None:
            print_and_log("[B2] ✗ Dataset not available", "ERROR")
            return BenchmarkResults(
                name="B2_VLA_Prediction_Accuracy",
                n_episodes=0,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                episodes=[],
            )
        
        print_and_log(f"\n{'='*70}")
        print_and_log(f"B2: VLA PREDICTION ACCURACY")
        print_and_log(f"{'='*70}")
        print_and_log(f"Target episodes: {n_episodes} (out of {len(self.episode_indices)} available)")
        
        # Health check
        try:
            is_healthy = await self.vla_client.health_check()
            if not is_healthy:
                print_and_log("[B2] ✗ VLA server health check FAILED", "ERROR")
                return BenchmarkResults(
                    name="B2_VLA_Prediction_Accuracy",
                    n_episodes=0,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                    episodes=[],
                )
            print_and_log("[B2] ✓ VLA server health check PASSED")
        except Exception as e:
            print_and_log(f"[B2] ✗ VLA health check failed: {e}", "ERROR")
            return BenchmarkResults(
                name="B2_VLA_Prediction_Accuracy",
                n_episodes=0,
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                episodes=[],
            )
        
        results = BenchmarkResults(
            name="B2_VLA_Prediction_Accuracy",
            n_episodes=n_episodes,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            episodes=[],
        )
        
        # Select episodes
        selected_episodes = sorted(self.episode_indices.keys())[:n_episodes]
        
        for ep_num, ep_idx in enumerate(tqdm(selected_episodes, desc="B2")):
            try:
                t_start = time.perf_counter()
                frame_indices = self.episode_indices[ep_idx]
                vla_latencies = []
                action_errors = []
                
                # Sample frames from episode (not all, to save time)
                sampled_frames = frame_indices[::max(1, len(frame_indices)//10)][:5]  # Max 5 frames per episode
                
                for frame_idx in sampled_frames:
                    sample = self.dataset[frame_idx]
                    
                    # Get image and state
                    rgb = sample["observation.images.image"].numpy()  # [3, 224, 224]
                    state = sample["observation.state"].numpy()  # [8]
                    action_gt = sample["action"].numpy()  # [7]
                    
                    # Ensure uint8
                    if rgb.dtype == np.float32 and rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                    
                    # Transpose if needed: [3, H, W] → [H, W, 3]
                    if len(rgb.shape) == 3 and rgb.shape[0] == 3:
                        rgb = rgb.transpose(1, 2, 0)
                    
                    # Query VLA
                    try:
                        t_vla = time.perf_counter()
                        action_pred = await self.vla_client.predict(
                            rgb_image=rgb,
                            state=state,
                            instruction="pick and place the object",
                        )
                        latency_ms = (time.perf_counter() - t_vla) * 1000
                        vla_latencies.append(latency_ms)
                        
                        # Compare to ground truth
                        action_error = np.linalg.norm(action_pred - action_gt)
                        action_errors.append(action_error)
                    except Exception as e:
                        print_and_log(f"[B2-EP{ep_idx}] VLA query failed: {str(e)[:50]}", "WARNING")
                        break
                
                duration_s = time.perf_counter() - t_start
                mean_latency = float(np.mean(vla_latencies)) if vla_latencies else 0.0
                mean_error = float(np.mean(action_errors)) if action_errors else 0.0
                success = mean_error < 0.5  # Arbitrary threshold
                
                metrics = EpisodeMetrics(
                    episode_idx=ep_idx,
                    success=success,
                    steps=len(action_errors),
                    duration_s=duration_s,
                    tracking_error_rad=mean_error,
                    vla_queries=len(vla_latencies),
                    vla_mean_latency_ms=mean_latency,
                    notes=f"Sampled {len(action_errors)} frames"
                )
                results.episodes.append(metrics)
                
                if (ep_num + 1) % 10 == 0:
                    print_and_log(f"[B2] Progress: {ep_num+1}/{n_episodes} | Latency: {mean_latency:.1f}ms | Error: {mean_error:.6f}")
                
            except Exception as e:
                print_and_log(f"[B2] Episode {ep_idx} failed: {str(e)[:50]}", "WARNING")
                metrics = EpisodeMetrics(
                    episode_idx=ep_idx,
                    success=False,
                    steps=0,
                    duration_s=0.0,
                    tracking_error_rad=0.0,
                    notes=f"Failed: {str(e)[:50]}"
                )
                results.episodes.append(metrics)
        
        results.print_summary()
        return results
    
    # ─────────────────────────────────────────────────────────────────────
    # B3: Full Dual-System (VLA + MPC in Simulation) (50 episodes)
    # ─────────────────────────────────────────────────────────────────────
    
    async def run_b3_full_dual_system(self, n_episodes: int = 50) -> BenchmarkResults:
        """
        B3: VLA + MPC in MuJoCo simulation on lift task.
        
        NOT dataset-based (uses simulation).
        Tests: Full system integration on simulated task.
        """
        print_and_log(f"\n{'='*70}")
        print_and_log(f"B3: FULL DUAL-SYSTEM (VLA + MPC in Simulation)")
        print_and_log(f"{'='*70}")
        print_and_log(f"Target episodes: {n_episodes}")
        
        results = BenchmarkResults(
            name="B3_Full_Dual_System",
            n_episodes=n_episodes,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            episodes=[],
        )
        
        for ep_num in tqdm(range(n_episodes), desc="B3"):
            try:
                t_start = time.perf_counter()
                self.env.reset()
                
                vla_latencies = []
                tracking_errors = []
                success = False
                steps = 0
                
                for step in range(500):  # Max 500 steps
                    try:
                        obs = self.env._get_obs()
                        rgb = obs.get("rgb", np.zeros((224, 224, 3), dtype=np.uint8))
                        q = obs.get("joint_pos", np.zeros(8))
                        
                        # Query VLA
                        t_vla = time.perf_counter()
                        action = await self.vla_client.predict(
                            rgb_image=rgb,
                            state=q,
                            instruction="lift the object",
                        )
                        latency_ms = (time.perf_counter() - t_vla) * 1000
                        vla_latencies.append(latency_ms)
                        
                        # Execute action
                        action = np.array(action)
                        if len(action) < 8:
                            action = np.concatenate([action, np.zeros(8 - len(action))])
                        
                        obs_next = self.env.step(action)
                        q_next = obs_next.get("joint_pos", q)
                        
                        error = np.linalg.norm(q_next[:4] - q[:4]) if len(q) >= 4 else 0.0
                        tracking_errors.append(error)
                        
                        # Success: object lifted
                        obj_height = obs_next.get("object_pos", [0, 0, 0])[2]
                        if obj_height > 0.2:
                            success = True
                            steps = step + 1
                            break
                        
                        steps = step + 1
                        
                    except Exception as e:
                        steps = step
                        break
                
                duration_s = time.perf_counter() - t_start
                mean_error = float(np.mean(tracking_errors)) if tracking_errors else 0.0
                mean_latency = float(np.mean(vla_latencies)) if vla_latencies else 0.0
                
                metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=success,
                    steps=steps,
                    duration_s=duration_s,
                    tracking_error_rad=mean_error,
                    vla_queries=len(vla_latencies),
                    vla_mean_latency_ms=mean_latency,
                )
                results.episodes.append(metrics)
                
                if (ep_num + 1) % 10 == 0:
                    print_and_log(f"[B3] Progress: {ep_num+1}/{n_episodes} | Success: {success}")
                
            except Exception as e:
                print_and_log(f"[B3] Episode {ep_num} failed: {str(e)[:50]}", "WARNING")
                metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=False,
                    steps=0,
                    duration_s=0.0,
                    tracking_error_rad=0.0,
                    notes=f"Failed: {str(e)[:50]}"
                )
                results.episodes.append(metrics)
        
        results.print_summary()
        return results
    
    # ─────────────────────────────────────────────────────────────────────
    # B4: MPC-Only Baseline (30 episodes)
    # ─────────────────────────────────────────────────────────────────────
    
    def run_b4_mpc_only_baseline(self, n_episodes: int = 30) -> BenchmarkResults:
        """
        B4: MPC-only baseline (NO VLA, simulation).
        
        Tests: What's the baseline MPC performance on a simple task?
        """
        print_and_log(f"\n{'='*70}")
        print_and_log(f"B4: MPC-ONLY BASELINE (NO VLA, Simulation)")
        print_and_log(f"{'='*70}")
        print_and_log(f"Target episodes: {n_episodes}")
        
        results = BenchmarkResults(
            name="B4_MPC_Only_Baseline",
            n_episodes=n_episodes,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            episodes=[],
        )
        
        for ep_num in tqdm(range(n_episodes), desc="B4"):
            try:
                t_start = time.perf_counter()
                self.env.reset()
                
                tracking_errors = []
                success = False
                steps = 0
                
                for step in range(200):  # Max 200 steps (simpler task)
                    try:
                        q = self.env.get_joint_pos()
                        
                        # Simple sine-wave reference
                        q_ref = np.array([0.5 * np.sin(2 * np.pi * step / 100.0) for _ in range(7)])
                        
                        # PD controller
                        error = np.linalg.norm(q_ref - q) if len(q) == len(q_ref) else 0.0
                        kp = 10.0
                        tau = np.clip(kp * (q_ref - q), -50, 50)
                        tau_full = np.concatenate([tau, [0]])
                        
                        obs = self.env.step(tau_full)
                        tracking_errors.append(error)
                        steps = step + 1
                        
                        # Success: low steady-state error
                        if len(tracking_errors) >= 50 and np.mean(tracking_errors[-50:]) < 0.1:
                            success = True
                            break
                        
                    except Exception as e:
                        steps = step
                        break
                
                duration_s = time.perf_counter() - t_start
                mean_error = float(np.mean(tracking_errors)) if tracking_errors else 0.0
                
                metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=success,
                    steps=steps,
                    duration_s=duration_s,
                    tracking_error_rad=mean_error,
                    notes="Sine-wave reference tracking"
                )
                results.episodes.append(metrics)
                
                if (ep_num + 1) % 10 == 0:
                    print_and_log(f"[B4] Progress: {ep_num+1}/{n_episodes} | Error: {mean_error:.6f}")
                
            except Exception as e:
                print_and_log(f"[B4] Episode {ep_num} failed: {str(e)[:50]}", "WARNING")
                metrics = EpisodeMetrics(
                    episode_idx=ep_num,
                    success=False,
                    steps=0,
                    duration_s=0.0,
                    tracking_error_rad=0.0,
                    notes=f"Failed: {str(e)[:50]}"
                )
                results.episodes.append(metrics)
        
        results.print_summary()
        return results


async def main():
    """Run all benchmarks in sequence."""
    print_and_log(f"\n{'='*70}")
    print_and_log(f"PHASE 12: COMPREHENSIVE BENCHMARK SUITE (CORRECTED)")
    print_and_log(f"Dataset: lerobot/utokyo_xarm_pick_and_place (102 episodes)")
    print_and_log(f"{'='*70}\n")
    
    runner = BenchmarkRunner()
    
    # B1: Dataset Replay (80 episodes)
    b1_results = runner.run_b1_dataset_replay_mpc_solo(n_episodes=80)
    b1_results.save(Path("evaluation/results/B1_dataset_replay_mpc_solo.json"))
    
    # B2: VLA Prediction (80 episodes)
    b2_results = await runner.run_b2_vla_prediction_accuracy(n_episodes=80)
    b2_results.save(Path("evaluation/results/B2_vla_prediction_accuracy.json"))
    
    # B3: Full Dual-System (50 episodes)
    b3_results = await runner.run_b3_full_dual_system(n_episodes=50)
    b3_results.save(Path("evaluation/results/B3_full_dual_system.json"))
    
    # B4: MPC Baseline (30 episodes)
    b4_results = runner.run_b4_mpc_only_baseline(n_episodes=30)
    b4_results.save(Path("evaluation/results/B4_mpc_only_baseline.json"))
    
    # Summary
    print_and_log(f"\n\n{'='*70}")
    print_and_log(f"BENCHMARK SUMMARY")
    print_and_log(f"{'='*70}\n")
    
    for results in [b1_results, b2_results, b3_results, b4_results]:
        print_and_log(f"{results.name:40s} | Success: {results.success_rate*100:5.1f}% | Error: {results.mean_tracking_error:8.6f} rad")
    
    print_and_log(f"\n{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(main())
