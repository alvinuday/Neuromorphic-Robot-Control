#!/usr/bin/env python3
"""
Phase 12: QUICK TEST with 3 episodes per benchmark

Just test if the full pipeline works before running all 102 episodes
"""

import asyncio
import json
import logging
import time
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Imports
from simulation.envs.xarm_env import XArmEnv
from smolvla.real_client import RealSmolVLAClient

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
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def log_info(msg: str):
    logger.info(msg)
    print(msg, flush=True)

def log_success(msg: str):
    print(f"✅ {msg}", flush=True)

def log_error(msg: str):
    print(f"❌ {msg}", flush=True)

@dataclass
class EpisodeMetrics:
    episode_idx: int
    success: bool
    steps: int
    duration_s: float
    tracking_error_rad: float
    vla_queries: int = 0
    vla_mean_latency_ms: float = 0.0
    notes: str = ""

def numpy_to_python(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.signedinteger, np.unsignedinteger)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    return obj

class QuickTestRunner:
    def __init__(self):
        log_info("\n" + "="*80)
        log_info("PHASE 12: QUICK TEST (5 episodes per benchmark)")
        log_info("="*80)
        
        self.env = XArmEnv()
        self.vla_client = RealSmolVLAClient()
        self.dataset = None
        self.episode_indices = {}
        
        if LEROBOT_AVAILABLE:
            try:
                log_info("Loading dataset...")
                self.dataset = LeRobotDataset("lerobot/utokyo_xarm_pick_and_place", root="data/cache")
                log_success(f"Dataset loaded: {self.dataset.num_episodes} episodes, {self.dataset.num_frames} frames")
                
                # Build episode index - ONLY for first 3 episodes
                log_info("Building episode index (for first 3 episodes)...")
                frame_count = 0
                for frame_idx in range(self.dataset.num_frames):
                    sample = self.dataset[frame_idx]
                    ep_idx = sample["episode_index"].item()
                    
                    if ep_idx not in self.episode_indices:
                        self.episode_indices[ep_idx] = []
                    self.episode_indices[ep_idx].append(frame_idx)
                    
                    # Print progress every 1000 frames
                    frame_count += 1
                    if frame_count % 1000 == 0:
                        log_info(f"  Processed {frame_count}/{self.dataset.num_frames} frames ({100*frame_count/self.dataset.num_frames:.0f}%)")
                    
                    # Stop after we have first 3 episodes
                    if len(self.episode_indices) >= 3:
                        break
                
                log_success(f"Built index for {len(self.episode_indices)} episodes")
            except Exception as e:
                log_error(f"Setup failed: {e}")
    
    async def run_b1_quick(self):
        """B1: 10 episodes MPC solo"""
        log_info("\n" + "="*80)
        log_info("B1: MPC SOLO TRACKING (10 episodes, QUICK TEST)")
        log_info("="*80)
        
        if self.dataset is None:
            log_error("Dataset not available")
            return
        
        episodes = sorted(self.episode_indices.keys())[:3]
        log_info(f"Episodes: {episodes}")
        
        for ep_num, ep_idx in enumerate(episodes):
            try:
                t_start = time.perf_counter()
                log_info(f"\n[B1] Episode {ep_num+1}/3 (idx={ep_idx})...")
                
                self.env.reset()
                frame_indices = self.episode_indices[ep_idx]
                
                log_info(f"  Frames in episode: {len(frame_indices)}")
                tracking_errors = []
                steps = 0
                
                # Just do first 10 frames for quick test
                for step, frame_idx in enumerate(frame_indices[:10]):
                    sample = self.dataset[frame_idx]
                    q_dataset = sample["observation.state"].numpy()
                    
                    try:
                        q_current = self.env.get_joint_pos()
                    except:
                        break
                    
                    error = np.linalg.norm(q_current[:len(q_dataset)] - q_dataset) if len(q_current) >= len(q_dataset) else 0.0
                    tracking_errors.append(error)
                    
                    kp = 10.0
                    tau = np.clip(kp * (q_dataset - q_current[:len(q_dataset)]), -50, 50)
                    tau_full = np.concatenate([tau, np.zeros(8 - len(tau))])
                    
                    try:
                        self.env.step(tau_full)
                        steps += 1
                    except:
                        break
                
                duration = time.perf_counter() - t_start
                mean_error = float(np.mean(tracking_errors)) if tracking_errors else 0.0
                
                log_success(f"  ✓ Complete | Steps: {steps} | Error: {mean_error:.6f} | Time: {duration:.2f}s")
                
            except Exception as e:
                log_error(f"  Episode {ep_idx} failed: {str(e)[:60]}")
    
    async def run_b2_quick(self):
        """B2: 3 episodes VLA prediction"""
        log_info("\n" + "="*80)
        log_info("B2: VLA PREDICTION (3 episodes, QUICK TEST)")
        log_info("="*80)
        
        if self.dataset is None:
            log_error("Dataset not available")
            return
        
        # Health check
        try:
            is_healthy = await self.vla_client.health_check()
            if not is_healthy:
                log_error("VLA server not healthy")
                return
            log_success("VLA server healthy")
        except Exception as e:
            log_error(f"VLA health check failed: {e}")
            return
        
        episodes = sorted(self.episode_indices.keys())[:3]
        log_info(f"Episodes: {episodes}")
        
        for ep_num, ep_idx in enumerate(episodes):
            try:
                t_start = time.perf_counter()
                log_info(f"\n[B2] Episode {ep_num+1}/3 (idx={ep_idx})...")
                
                frame_indices = self.episode_indices[ep_idx]
                vla_latencies = []
                action_errors = []
                
                log_info(f"  Frames in episode: {len(frame_indices)}")
                
                # Sample just 2 frames for quick test
                sampled_frames = frame_indices[::max(1, len(frame_indices)//10)][:2]
                log_info(f"  Sampling {len(sampled_frames)} frames for VLA")
                
                for frame_idx in sampled_frames:
                    sample = self.dataset[frame_idx]
                    rgb = sample["observation.images.image"].numpy()
                    state = sample["observation.state"].numpy()
                    action_gt = sample["action"].numpy()
                    
                    # Ensure uint8
                    if rgb.dtype == np.float32 and rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                    
                    if len(rgb.shape) == 3 and rgb.shape[0] == 3:
                        rgb = rgb.transpose(1, 2, 0)
                    
                    try:
                        t_vla = time.perf_counter()
                        action_pred = await self.vla_client.predict(
                            rgb_image=rgb,
                            state=state,
                            instruction="pick and place the object",
                        )
                        latency_ms = (time.perf_counter() - t_vla) * 1000
                        vla_latencies.append(latency_ms)
                        
                        action_error = np.linalg.norm(np.array(action_pred) - action_gt)
                        action_errors.append(action_error)
                        
                        log_info(f"    VLA latency: {latency_ms:.1f}ms | Error: {action_error:.6f}")
                    except Exception as e:
                        log_error(f"    VLA query failed: {str(e)[:40]}")
                
                duration = time.perf_counter() - t_start
                mean_latency = float(np.mean(vla_latencies)) if vla_latencies else 0.0
                mean_error = float(np.mean(action_errors)) if action_errors else 0.0
                
                log_success(f"  ✓ Complete | Latency: {mean_latency:.1f}ms | Error: {mean_error:.6f} | Time: {duration:.2f}s")
                
            except Exception as e:
                log_error(f"  Episode {ep_idx} failed: {str(e)[:60]}")
    
    async def run_b4_quick(self):
        """B4: 3 episodes VLA+MPC on real data"""
        log_info("\n" + "="*80)
        log_info("B4: VLA+MPC REAL DATA (3 episodes, QUICK TEST)")
        log_info("="*80)
        
        if self.dataset is None:
            log_error("Dataset not available")
            return
        
        # Health check
        try:
            is_healthy = await self.vla_client.health_check()
            if not is_healthy:
                log_error("VLA server not healthy")
                return
            log_success("VLA server healthy")
        except Exception as e:
            log_error(f"VLA health check failed: {e}")
            return
        
        episodes = sorted(self.episode_indices.keys())[:3]
        log_info(f"Episodes: {episodes}")
        
        for ep_num, ep_idx in enumerate(episodes):
            try:
                t_start = time.perf_counter()
                log_info(f"\n[B4] Episode {ep_num+1}/3 (idx={ep_idx})...")
                
                self.env.reset()
                frame_indices = self.episode_indices[ep_idx]
                vla_latencies = []
                tracking_errors = []
                
                log_info(f"  Frames in episode: {len(frame_indices)}")
                
                # Just do first 10 frames for quick test
                for step, frame_idx in enumerate(frame_indices[:10]):
                    sample = self.dataset[frame_idx]
                    rgb = sample["observation.images.image"].numpy()
                    q_dataset = sample["observation.state"].numpy()
                    
                    if rgb.dtype == np.float32 and rgb.max() <= 1.0:
                        rgb = (rgb * 255).astype(np.uint8)
                    else:
                        rgb = rgb.astype(np.uint8)
                    
                    if len(rgb.shape) == 3 and rgb.shape[0] == 3:
                        rgb = rgb.transpose(1, 2, 0)
                    
                    try:
                        q_current = self.env.get_joint_pos()
                    except:
                        break
                    
                    try:
                        t_vla = time.perf_counter()
                        action_vla = await self.vla_client.predict(
                            rgb_image=rgb,
                            state=q_current,
                            instruction="pick and place the object",
                        )
                        latency_ms = (time.perf_counter() - t_vla) * 1000
                        vla_latencies.append(latency_ms)
                        
                        action_vla = np.array(action_vla)
                        if len(action_vla) < 8:
                            action_vla = np.concatenate([action_vla, np.zeros(8 - len(action_vla))])
                        
                        obs_next = self.env.step(action_vla)
                        q_next = obs_next.get("joint_pos", q_current)
                        
                        error = np.linalg.norm(q_next[:len(q_dataset)] - q_dataset) if len(q_next) >= len(q_dataset) else 0.0
                        tracking_errors.append(error)
                        
                        log_info(f"    Step {step+1}: latency={latency_ms:.1f}ms, error={error:.6f}")
                    except Exception as e:
                        log_error(f"    Step {step} failed: {str(e)[:40]}")
                        break
                
                duration = time.perf_counter() - t_start
                mean_latency = float(np.mean(vla_latencies)) if vla_latencies else 0.0
                mean_error = float(np.mean(tracking_errors)) if tracking_errors else 0.0
                
                log_success(f"  ✓ Complete | Latency: {mean_latency:.1f}ms | Error: {mean_error:.6f} | Time: {duration:.2f}s")
                
            except Exception as e:
                log_error(f"  Episode {ep_idx} failed: {str(e)[:60]}")

    async def run_b3_quick(self):
        """B3: 3 episodes VLA+MPC multi-task simulation"""
        log_info("\n" + "="*80)
        log_info("B3: VLA+MPC MULTI-TASK SIMULATION (3 episodes, QUICK TEST)")
        log_info("="*80)
        
        task_configs = [
            {"task": "pick", "language": "pick up the object", "target_height": 0.2},
            {"task": "place", "language": "place the object on the table", "target_height": 0.1},
            {"task": "grasp", "language": "grasp the cube firmly", "target_height": 0.15},
            {"task": "lift", "language": "lift the item above the surface", "target_height": 0.25},
        ]
        log_info(f"Task types: {[cfg['task'] for cfg in task_configs]}")
        log_info(f"Episodes: [0-2] mapped to tasks ")
        
        for ep_num in range(3):
            try:
                t_start = time.perf_counter()
                self.env.reset()
                
                task_config = task_configs[ep_num % len(task_configs)]
                language = task_config["language"]
                target_height = task_config["target_height"]
                
                log_info(f"\n[B3] Episode {ep_num+1}/3 - Task: {task_config['task']}")
                
                vla_latencies = []
                tracking_errors = []
                success = False
                steps = 0
                
                # Max 50 steps for quick test
                for step in range(40):
                    try:
                        obs = self.env._get_obs()
                        rgb = obs.get("rgb", np.zeros((224, 224, 3), dtype=np.uint8))
                        q = obs.get("joint_pos", np.zeros(8))
                        
                        # Query VLA WITH LANGUAGE
                        t_vla = time.perf_counter()
                        action = await self.vla_client.predict(
                            rgb_image=rgb,
                            state=q,
                            instruction=language,
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
                        
                        # Check task success (simple heuristic)
                        obj_height = obs_next.get("object_pos", [0, 0, 0])[2]
                        if obj_height > target_height:
                            success = True
                            steps = step + 1
                            break
                        
                        steps = step + 1
                        
                    except Exception as e:
                        log_warn(f"  Step {step} failed: {str(e)[:30]}")
                        steps = step
                        break
                
                duration = time.perf_counter() - t_start
                mean_error = float(np.mean(tracking_errors)) if tracking_errors else 0.0
                mean_latency = float(np.mean(vla_latencies)) if vla_latencies else 0.0
                
                success_str = "✓ SUCCESS" if success else "✗ INCOMPLETE"
                log_success(f"  {success_str} | Steps: {steps} | Latency: {mean_latency:.1f}ms | Error: {mean_error:.6f} | Time: {duration:.2f}s")
                
            except Exception as e:
                log_error(f"  Episode {ep_num} failed: {str(e)[:60]}")

async def main():
    runner = QuickTestRunner()
    
    log_info("\n" + "="*80)
    log_info("RUNNING QUICK BENCHMARKS (5 episodes each)")
    log_info("="*80)
    
    # B1: MPC solo
    await runner.run_b1_quick()
    
    # B2: VLA prediction
    await runner.run_b2_quick()
    
    # B3: Multi-task simulation
    await runner.run_b3_quick()
    
    # B4: VLA+MPC
    await runner.run_b4_quick()
    
    log_info("\n" + "="*80)
    log_info("QUICK TEST COMPLETE ✅")
    log_info("="*80)
    log_info("\nIf all 4 benchmarks completed successfully:")
    log_info("  → You can run the full 102-episode version")
    log_info("  → Command: python3 scripts/phase12_run_benchmarks_v2.py")

if __name__ == "__main__":
    asyncio.run(main())
