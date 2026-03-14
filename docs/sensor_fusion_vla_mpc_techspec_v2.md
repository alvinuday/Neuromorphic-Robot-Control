# NEUROMORPHIC DUAL-SYSTEM ROBOTIC CONTROLLER
## Complete System Technical Specification v2.0
## Sensor Fusion · Event Cameras · LiDAR · Real Dataset Validation · MuJoCo 3D
### Alvin — March 2026

---

> **This document is the single source of truth for the coding agent.**
> Every question about "what to build", "which dataset to use", "how to validate", or "where to put files" is answered here. If it is not in this document, ask before implementing.

---

## ⚠️ AGENT OPERATING RULES — READ BEFORE TOUCHING ANY FILE

These rules exist because the previous implementation was chaotic. Violating them restarts a phase.

### File Discipline
```
RULE 1:  NEVER create files in the project root. Every file belongs in a module subfolder.
RULE 2:  NEVER leave temporary scripts (test_xyz.py, debug_abc.py) in any folder. Use tests/.
RULE 3:  NEVER hardcode paths. All paths go through config/paths.yaml or project constants.
RULE 4:  NEVER print results and call them validated. Write them to logs/ with timestamps.
RULE 5:  DELETE any file you created for debugging before marking a task done.
RULE 6:  NEVER claim a dataset "downloaded" unless you ran the download script and logged the file hash.
```

### Memory Protocol (MANDATORY)
The agent MUST maintain three files in `docs/agent/`:
```
AGENT_STATE.md   — current phase, last completed task, blockers
TODO.md          — remaining tasks for current phase (checked off as done)
PROGRESS.md      — log of every completed task with timestamp, result, and file path
```

**Before starting any task:** Update `AGENT_STATE.md` with what you are about to do.
**After completing any task:** Update `TODO.md` (check it off) and `PROGRESS.md` (add entry).
**If blocked:** Write the blocker to `AGENT_STATE.md` and STOP. Do not guess or hallucinate a workaround.

### Anti-Hallucination Rules
```
RULE 7:  NEVER claim success without showing the actual output (numbers, file path, plot name).
RULE 8:  NEVER generate synthetic data and label it "dataset". If download fails, write BLOCKED.
RULE 9:  NEVER fabricate benchmark numbers. All metrics come from code that actually ran.
RULE 10: If you are unsure whether code ran or what it returned, say so explicitly.
```

### Validation Protocol
Every phase ends with a **Validation Gate** — a numbered checklist. You do not proceed to the next phase until every item on the gate is checked and the results are logged in `PROGRESS.md` with actual numbers.

---

## TABLE OF CONTENTS

1. [Project Overview — What We Are Actually Building](#1-project-overview)
2. [Honest Architecture Clarification — What VLA + MPC Actually Does](#2-honest-architecture-clarification)
3. [Project Structure (Canonical, Immutable)](#3-project-structure)
4. [Real Dataset — Ground Truth](#4-real-dataset--ground-truth)
5. [MuJoCo Environment — 3D Arm Simulation](#5-mujoco-environment)
6. [Event Camera Simulation](#6-event-camera-simulation)
7. [LiDAR Simulation](#7-lidar-simulation)
8. [Multimodal Sensor Fusion](#8-multimodal-sensor-fusion)
9. [SmolVLA Integration — Corrected Architecture](#9-smolvla-integration--corrected-architecture)
10. [Stuart-Landau MPC — Trajectory Tracking](#10-stuart-landau-mpc--trajectory-tracking)
11. [Full System Integration](#11-full-system-integration)
12. [Benchmarking & Metrics](#12-benchmarking--metrics)
13. [Visualization Suite](#13-visualization-suite)
14. [Testing & Validation Protocol](#14-testing--validation-protocol)
15. [Implementation Roadmap (Phased)](#15-implementation-roadmap-phased)
16. [Agent Coding Standards](#16-agent-coding-standards)

---

## 1. PROJECT OVERVIEW

### What We Are Building

A **modular, real-dataset-validated robotic control system** with four integrated layers:

```
LAYER 4: SmolVLA (System 2 — Task Intelligence)
  Input:  Real RGB images from dataset OR MuJoCo render
  Output: Action chunk (planned EE trajectory, N=10 steps)
  Rate:   ~5 Hz (async)

LAYER 3: Action Chunk Processor
  Input:  Action chunk from SmolVLA [N, action_dim]
  Output: Joint-space reference trajectory via IK [N, n_joints]

LAYER 2: Stuart-Landau MPC (System 1 — Fast Execution)
  Input:  Reference trajectory + current state q, q̇
  Output: Optimal joint torques τ* via QP solver
  Rate:   100-300 Hz

LAYER 1: MuJoCo 3D Simulation
  Robot:  xArm (4 DOF, matching real dataset)
  Sensors: RGB camera + simulated event camera (v2e) + LiDAR rangefinders
  State feedback at control rate
```

### What We Are NOT Building (yet)
- Real hardware deployment (simulation only for now)
- Fine-tuning SmolVLA (use pretrained base checkpoint only)
- Event camera hardware (simulated from MuJoCo renders via v2e)
- LSMO / Tokyo-U datasets (unavailable, inaccessible; see §4 for what we use instead)

---

## 2. HONEST ARCHITECTURE CLARIFICATION

### How SmolVLA Actually Works (Not What You Think)

**The confusion:** SmolVLA is NOT a waypoint-planning System 2 in the GR00T/Helix sense. It is an **end-to-end flow-matching policy** that outputs action chunks — sequences of direct robot commands.

**What SmolVLA actually outputs (from the paper, arXiv:2506.01844):**
> SmolVLA uses a 10-step flow-matching expert to produce action chunks. The action chunk is a sequence of [Δx, Δy, Δz, Δroll, Δpitch, Δyaw, gripper] or joint-space deltas, depending on the training data, executed at the robot's control frequency.

**This means:**

```
SmolVLA input:  RGB frame (224x224) + language instruction + robot state [n_joints]
SmolVLA output: action_chunk [10, action_dim]  — 10 timesteps of delta-actions
```

For `lerobot/xarm_lift_medium`, action_dim = 4 (3 joint positions + gripper).

**The corrected System 1 / System 2 integration:**

```
System 2 (SmolVLA, ~5 Hz):
  Observe scene → predict action_chunk [10, 4]
  → These are PLANNED JOINT TARGETS (not waypoints in Cartesian)
  → Action chunk = reference trajectory for MPC

System 1 (SL-MPC, ~100-300 Hz):
  Take q_ref from action chunk (current chunk step)
  Build QP: minimize ||q - q_ref||² subject to torque/velocity constraints
  Solve with SL oscillators → output τ*
  Apply τ* to simulation → get next state → repeat
```

**The interface:** SmolVLA gives you planned joint positions. SL-MPC tracks them with optimal torques under physical constraints. This is architecturally clean and correct. MPC adds: constraint satisfaction, dynamic feasibility, disturbance rejection, and energy efficiency — none of which SmolVLA provides.

### Does the System Actually Work Like System 1 / System 2?

**Partially yes, partially no:**

| Claim | Reality |
|-------|---------|
| SmolVLA reasons about the task at low frequency | ✅ Yes — it runs at 5 Hz, async |
| SmolVLA generates a reference trajectory | ✅ Yes — action chunk = 10-step plan |
| MPC executes at high frequency | ✅ Yes — 100-300 Hz |
| MPC uses physics-aware trajectory tracking | ✅ Yes — dynamics model M, C, G |
| SmolVLA "reasons" in natural language | ⚠️ Partially — it conditions on text but does not plan via chain-of-thought |
| MPC replanning based on VLA subgoals | ✅ Yes — each new chunk updates the reference |

**What to claim in your thesis:** "SmolVLA acts as a visuomotor task planner operating at perception frequency, outputting action chunks that serve as reference trajectories for a high-frequency neuromorphic MPC controller based on Stuart-Landau oscillators. The dual-frequency architecture decouples task reasoning from physical execution, allowing the SL-MPC to enforce dynamics constraints and handle disturbances within each planning cycle." — This is accurate.

### Why This System Is Novel

Existing VLA deployments use simple PD controllers or just replay the action chunk open-loop. Your contribution: **replacing the open-loop execution with a constrained optimal controller (SL-MPC) that tracks the VLA's plan while respecting physical dynamics**. This has not been done with a neuromorphic oscillator-based solver.

---

## 3. PROJECT STRUCTURE

This is the canonical structure. No files go anywhere else.

```
neuromorphic_arm/
│
├── docs/
│   ├── agent/
│   │   ├── AGENT_STATE.md          ← agent updates this every task
│   │   ├── TODO.md                 ← current phase task list
│   │   └── PROGRESS.md             ← completed task log with results
│   ├── architecture.md             ← system architecture diagram (text)
│   ├── dataset_audit.md            ← verified dataset facts
│   └── results/                    ← per-benchmark result summaries
│
├── config/
│   ├── paths.yaml                  ← all data/log/model paths
│   ├── robot.yaml                  ← arm params (DOF, limits, lengths, masses)
│   ├── mpc.yaml                    ← N, Q, R, P, dt, solver params
│   ├── smolvla.yaml                ← endpoint, chunk_size, timeout
│   ├── sensors.yaml                ← event cam params, lidar params
│   └── fusion.yaml                 ← fusion encoder architecture params
│
├── data/
│   ├── download/
│   │   ├── download_dataset.py     ← ONLY real downloader; logs hash + size
│   │   └── verify_dataset.py       ← checks integrity of downloaded data
│   ├── loaders/
│   │   ├── lerobot_loader.py       ← loads lerobot/xarm_lift_medium episodes
│   │   ├── episode_player.py       ← replays episodes in MuJoCo
│   │   └── data_inspector.py       ← prints actual dataset stats
│   └── cache/                      ← gitignored, holds downloaded data
│
├── simulation/
│   ├── envs/
│   │   ├── xarm_env.py             ← MuJoCo xArm environment
│   │   └── base_env.py             ← abstract base (for modularity)
│   ├── models/
│   │   ├── xarm_4dof.xml           ← MJCF model (4-DOF, matching dataset)
│   │   └── objects/                ← MJCF for blocks, bins, etc.
│   ├── cameras/
│   │   ├── rgb_camera.py           ← MuJoCo off-screen RGB renderer
│   │   ├── event_camera.py         ← v2e wrapper for event simulation
│   │   └── lidar_sensor.py         ← rangefinder-based LiDAR sim
│   └── tests/
│       ├── test_env_loads.py
│       ├── test_camera_renders.py
│       └── test_sensor_outputs.py
│
├── sensors/
│   ├── event_processing.py         ← event voxel grid representation
│   ├── lidar_processing.py         ← point cloud processing
│   ├── proprioception.py           ← joint state processing
│   └── tests/
│       └── test_sensor_processing.py
│
├── fusion/
│   ├── encoders/
│   │   ├── rgb_encoder.py          ← SigLIP/ResNet18 RGB features
│   │   ├── event_encoder.py        ← event voxel → features
│   │   ├── lidar_encoder.py        ← point cloud → features
│   │   └── proprio_encoder.py      ← joint state → features
│   ├── fusion_model.py             ← multimodal feature concatenation/attention
│   └── tests/
│       └── test_fusion_shapes.py
│
├── dynamics/
│   ├── kinematics.py               ← FK, IK, Jacobian for xArm 4DOF
│   ├── dynamics.py                 ← M(q), C(q,qdot), G(q)
│   └── tests/
│       ├── test_kinematics.py
│       └── test_dynamics.py
│
├── mpc/
│   ├── linearize.py                ← A, B matrices, ZOH discretization
│   ├── qp_builder.py               ← H, c, A_ineq from dynamics + reference
│   ├── sl_solver.py                ← Stuart-Landau oscillator QP (EXTENDED from existing)
│   └── tests/
│       ├── test_qp_builder.py
│       └── test_sl_vs_osqp.py
│
├── smolvla/
│   ├── client.py                   ← async HTTP client (ngrok endpoint)
│   ├── action_processor.py         ← action_chunk → joint reference trajectory
│   └── tests/
│       ├── test_client_mock.py
│       └── test_action_processor.py
│
├── system/
│   ├── controller.py               ← DualSystemController (System1 + System2)
│   ├── trajectory_buffer.py        ← holds VLA action chunks, provides refs to MPC
│   └── state_machine.py            ← episode state machine
│
├── evaluation/
│   ├── metrics.py                  ← success_rate, tracking_error, latency, etc.
│   ├── benchmarks/
│   │   ├── run_dataset_replay.py   ← replay real dataset, compare MPC output
│   │   ├── run_mpc_solo.py         ← test MPC without VLA (baseline)
│   │   ├── run_smolvla_only.py     ← test SmolVLA without MPC (baseline)
│   │   └── run_full_system.py      ← full System1+System2 end-to-end
│   └── results/                    ← CSV/JSON output of benchmark runs
│
├── visualization/
│   ├── episode_viewer.py           ← replay episode with all sensor modalities
│   ├── live_dashboard.py           ← real-time matplotlib dashboard
│   ├── trajectory_plotter.py       ← planned vs actual trajectory plots
│   ├── event_visualizer.py         ← event stream visualization
│   └── report_generator.py         ← auto-generate results figures for thesis
│
├── logs/                           ← gitignored; all run logs go here
│   └── .gitkeep
│
├── notebooks/
│   ├── 00_dataset_audit.ipynb      ← first thing to run; shows real dataset stats
│   ├── 01_env_validation.ipynb     ← MuJoCo loads and renders correctly
│   ├── 02_sensor_validation.ipynb  ← event cam + LiDAR outputs
│   ├── 03_mpc_validation.ipynb     ← SL-MPC tracking test
│   ├── 04_smolvla_validation.ipynb ← SmolVLA inference on real images
│   └── 05_full_system.ipynb        ← end-to-end benchmark
│
├── scripts/
│   └── setup_env.sh                ← one-shot environment setup
│
├── requirements.txt                ← pinned dependencies
├── pyproject.toml                  ← project metadata
└── README.md                       ← setup + quickstart
```

**RULE:** If you create a file not in this tree, you must justify it in `AGENT_STATE.md` first.

---

## 4. REAL DATASET — GROUND TRUTH

### Selected Dataset: `lerobot/xarm_lift_medium`

**Why this dataset:**
- 800 total episodes, 20,000 total frames, 15 fps, 84×84 RGB images, 4-dimensional state and action vectors (3 joints + gripper)
- Total download size: 17.8 MB — fits on any laptop, downloads in seconds
- Parquet + MP4 format, loadable with one line via `LeRobotDataset`
- Real robot data (xArm manipulator in lab setting)
- Single task: "lift medium object" — clean, well-defined success criterion
- Fully open, MIT licensed

**DO NOT use any other dataset without explicit approval.** DROID is 1.7TB. Open X-Embodiment requires custom RLDS conversion. They are out of scope. `lerobot/xarm_lift_medium` is the target.

### Dataset Facts (Verified from HuggingFace)

```yaml
dataset_id: lerobot/xarm_lift_medium
total_episodes: 800
total_frames: 20000
fps: 15
image_size: [84, 84, 3]   # height, width, channels; uint8 RGB
state_dim: 4               # [q1, q2, q3, gripper_pos]
action_dim: 4              # [a1, a2, a3, gripper_cmd] — absolute joint targets
task: "lift medium object"
format: parquet (tabular) + mp4 (video)
download_size: ~17.8 MB
license: MIT
source_robot: xArm (Ufactory) — 4 DOF configuration used in dataset
```

**State/action format:**
```python
# Each frame contains:
sample = {
    "observation.image":  torch.Tensor([3, 84, 84]),     # uint8 RGB
    "observation.state":  torch.Tensor([4]),              # [q1, q2, q3, gripper]
    "action":             torch.Tensor([4]),              # next joint targets
    "timestamp":          float,
    "episode_index":      int,
    "frame_index":        int,
    "next.reward":        float,  # 1.0 if success, 0.0 otherwise
}
```

### Download Protocol

```python
# data/download/download_dataset.py
# This is the ONLY way to get the dataset. No shortcuts.

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import hashlib, json, os
from pathlib import Path

DATASET_ID = "lerobot/xarm_lift_medium"
CACHE_DIR = Path("data/cache")

def download_and_verify():
    print(f"Downloading {DATASET_ID}...")
    
    dataset = LeRobotDataset(
        DATASET_ID,
        root=str(CACHE_DIR),
        download=True
    )
    
    # Verify basic facts match expectations
    assert dataset.num_episodes == 800, f"Expected 800 episodes, got {dataset.num_episodes}"
    assert dataset.num_frames == 20000, f"Expected 20000 frames, got {dataset.num_frames}"
    assert dataset[0]["observation.image"].shape == (3, 84, 84), "Wrong image shape"
    assert dataset[0]["observation.state"].shape == (4,), "Wrong state dim"
    assert dataset[0]["action"].shape == (4,), "Wrong action dim"
    
    # Log download results
    log = {
        "dataset_id": DATASET_ID,
        "episodes": dataset.num_episodes,
        "frames": dataset.num_frames,
        "cache_dir": str(CACHE_DIR.resolve()),
        "verified": True
    }
    
    log_path = Path("logs/dataset_download.json")
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    
    print(f"✓ Dataset downloaded and verified.")
    print(f"✓ Log written to {log_path}")
    return dataset

if __name__ == "__main__":
    download_and_verify()
```

### Dataset Inspection (Run First)

```python
# data/loaders/data_inspector.py

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np, json

def inspect_dataset(dataset_id="lerobot/xarm_lift_medium"):
    ds = LeRobotDataset(dataset_id, root="data/cache")
    
    print("=" * 60)
    print(f"DATASET: {dataset_id}")
    print("=" * 60)
    print(f"Episodes:     {ds.num_episodes}")
    print(f"Total frames: {ds.num_frames}")
    print(f"FPS:          {ds.fps}")
    print()
    
    # Inspect first sample
    s = ds[0]
    print("FEATURES:")
    for k, v in s.items():
        if hasattr(v, 'shape'):
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, "
                  f"range=[{v.min():.3f}, {v.max():.3f}]")
        else:
            print(f"  {k}: {v}")
    
    # Episode length distribution
    ep_lengths = []
    for ep_idx in range(min(50, ds.num_episodes)):
        frames = [i for i in range(ds.num_frames) 
                  if ds[i]["episode_index"] == ep_idx]
        ep_lengths.append(len(frames))
    
    print(f"\nEPISODE LENGTHS (first 50):")
    print(f"  Mean:   {np.mean(ep_lengths):.1f} frames")
    print(f"  Std:    {np.std(ep_lengths):.1f} frames")
    print(f"  Min:    {np.min(ep_lengths)}")
    print(f"  Max:    {np.max(ep_lengths)}")
    
    # Success rate from rewards
    rewards = [ds[i]["next.reward"].item() for i in range(ds.num_frames)]
    success = sum(1 for r in rewards if r > 0.5) / ds.num_episodes
    print(f"\nSUCCESS RATE (from rewards): {success*100:.1f}%")
    
    return ds
```

### Why NOT to use other datasets

| Dataset | Issue |
|---------|-------|
| LSMO / Tokyo-U | Not publicly accessible without institutional access; no HuggingFace mirror |
| DROID | 1.7 TB; requires SLURM cluster to process |
| Open X-Embodiment | TFRecords format; requires complex RLDS conversion pipeline |
| BridgeData V2 | ~60 GB; too large for laptop |
| Synthetic data | **Explicitly forbidden by the agent rules** |

---

## 5. MUJOCO ENVIRONMENT

### Why xArm 4-DOF?

The dataset uses an xArm in a 4-DOF configuration (3 arm joints + gripper). Our MuJoCo model must match this exactly so that dataset states can be replayed faithfully in simulation.

### MJCF Model for xArm 4-DOF

Use the **MuJoCo Menagerie** xArm model as the base:
```bash
# Check if menagerie has xarm
pip install mujoco
python -c "import mujoco; print(mujoco.__version__)"
# Clone menagerie for xArm MJCF
git clone https://github.com/google-deepmind/mujoco_menagerie data/models/mujoco_menagerie
# xArm model at: mujoco_menagerie/ufactory_xarm7/xarm7.xml
# We will simplify to 4-DOF matching dataset
```

**If menagerie xArm is 7-DOF (likely), create a 4-DOF MJCF wrapper:**

```xml
<!-- simulation/models/xarm_4dof.xml -->
<?xml version="1.0" ?>
<mujoco model="xarm_4dof">
  
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  
  <default>
    <joint damping="0.5" armature="0.02" frictionloss="0.1"/>
    <geom contype="1" conaffinity="1" friction="0.8 0.01 0.01"/>
    <motor ctrllimited="true"/>
  </default>
  
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" pos="0 0 0" size="1 1 0.01" 
          rgba="0.7 0.7 0.7 1" contype="1" conaffinity="1"/>
    
    <!-- Table -->
    <body name="table" pos="0.4 0 0.3">
      <geom type="box" size="0.3 0.3 0.02" rgba="0.6 0.4 0.2 1"
            contype="1" conaffinity="1" mass="10"/>
    </body>
    
    <!-- xArm base (fixed) -->
    <body name="base" pos="0 0 0">
      <geom type="cylinder" fromto="0 0 0 0 0 0.05" size="0.06" 
            rgba="0.3 0.3 0.3 1" mass="2"/>
      
      <!-- Joint 1: base yaw (z-axis) -->
      <body name="link1" pos="0 0 0.05">
        <joint name="joint1" type="hinge" axis="0 0 1" 
               range="-2.618 2.618"/>
        <geom type="cylinder" fromto="0 0 0 0 0 0.08" size="0.04"
              rgba="0.2 0.4 0.8 1" mass="1.0"/>
        
        <!-- Joint 2: shoulder pitch (y-axis) -->
        <body name="link2" pos="0 0 0.08">
          <joint name="joint2" type="hinge" axis="0 1 0" 
                 range="-2.059 2.059"/>
          <geom type="capsule" fromto="0 0 0 0.265 0 0" size="0.03"
                rgba="0.2 0.4 0.8 1" mass="0.8"/>
          <inertial pos="0.1325 0 0" mass="0.8"
                    diaginertia="0.001 0.01 0.01"/>
          
          <!-- Joint 3: elbow pitch (y-axis) -->
          <body name="link3" pos="0.265 0 0">
            <joint name="joint3" type="hinge" axis="0 1 0" 
                   range="-3.927 0.193"/>
            <geom type="capsule" fromto="0 0 0 0.245 0 0" size="0.025"
                  rgba="0.2 0.4 0.8 1" mass="0.5"/>
            <inertial pos="0.1225 0 0" mass="0.5"
                      diaginertia="0.0005 0.007 0.007"/>
            
            <!-- End-effector + Gripper -->
            <body name="ee_link" pos="0.245 0 0">
              <!-- Joint 4: gripper (prismatic, simplified) -->
              <joint name="gripper" type="slide" axis="0 1 0"
                     range="0.0 0.085"/>
              <geom type="box" size="0.04 0.01 0.03" pos="0.03 0 0"
                    rgba="0.8 0.5 0.2 1" mass="0.15"/>
              
              <!-- EE site for FK validation -->
              <site name="ee_site" pos="0.08 0 0" size="0.012" 
                    rgba="1 1 0 0.8"/>
            </body>
          </body>
        </body>
      </body>
    </body>
    
    <!-- Manipulandum: medium object (matches dataset task) -->
    <body name="object" pos="0.4 0.0 0.33" mocap="false">
      <freejoint name="object_joint"/>
      <geom type="cylinder" size="0.025 0.04" rgba="1 0.3 0.1 1" 
            mass="0.08" contype="1" conaffinity="1"/>
      <site name="object_site" size="0.01" rgba="0 1 0 1"/>
    </body>
    
    <!-- Target zone marker (visual) -->
    <body name="target_zone" pos="0.4 0 0.4" mocap="true">
      <geom type="sphere" size="0.02" rgba="0 1 0 0.4" 
            contype="0" conaffinity="0"/>
    </body>
    
    <!-- Cameras -->
    <!-- Primary: matches dataset perspective (overhead/front) -->
    <camera name="primary_cam" pos="0.5 0 0.8" 
            xyaxes="0 -1 0 -0.7 0 0.7"/>
    <!-- Side view for visualization -->
    <camera name="side_cam" pos="-0.1 0.8 0.5" 
            xyaxes="-1 0 0 0 -0.5 0.866"/>
    <!-- Wrist camera (future) -->
    <camera name="wrist_cam" pos="0.08 0 0" mode="fixed"
            xyaxes="1 0 0 0 0 1" body="ee_link"/>
  </worldbody>
  
  <actuator>
    <motor name="act_j1" joint="joint1" gear="100" ctrlrange="-5 5"/>
    <motor name="act_j2" joint="joint2" gear="100" ctrlrange="-8 8"/>
    <motor name="act_j3" joint="joint3" gear="80"  ctrlrange="-5 5"/>
    <motor name="act_g"  joint="gripper" gear="20" ctrlrange="0 2"/>
  </actuator>
  
  <sensor>
    <!-- Joint positions and velocities -->
    <jointpos name="q1"    joint="joint1"/>
    <jointpos name="q2"    joint="joint2"/>
    <jointpos name="q3"    joint="joint3"/>
    <jointpos name="q_g"   joint="gripper"/>
    <jointvel name="qd1"   joint="joint1"/>
    <jointvel name="qd2"   joint="joint2"/>
    <jointvel name="qd3"   joint="joint3"/>
    <!-- EE position -->
    <framepos name="ee_pos"  objtype="site" objname="ee_site"/>
    <!-- Object position (ground truth) -->
    <framepos name="obj_pos" objtype="site" objname="object_site"/>
    <!-- LiDAR rangefinders (16-ray simulated LiDAR) -->
    <rangefinder name="lidar_0"  site="ee_site" cutoff="1.0"/>
    <rangefinder name="lidar_45" site="ee_site" cutoff="1.0"/>
    <!-- (Add 14 more with rotated directions in full implementation) -->
  </sensor>

</mujoco>
```

### MuJoCo Environment Class

```python
# simulation/envs/xarm_env.py

import mujoco
import mujoco.viewer
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import yaml

class XArmEnv:
    """
    MuJoCo environment for xArm 4-DOF matching lerobot/xarm_lift_medium.
    
    State: [q1, q2, q3, gripper_pos]  — 4 DOF
    Action: joint torques [τ1, τ2, τ3, τ_gripper]
    
    Sensors:
      - RGB camera (84x84, matches dataset)
      - Simulated event camera (via event_camera.py)
      - Simulated LiDAR (rangefinders)
      - Joint proprioception
    """
    
    JOINT_NAMES = ["joint1", "joint2", "joint3", "gripper"]
    JOINT_LIMITS = {
        "joint1":  (-2.618, 2.618),   # rad
        "joint2":  (-2.059, 2.059),   # rad
        "joint3":  (-3.927, 0.193),   # rad
        "gripper": (0.0, 0.085),      # m
    }
    TORQUE_LIMITS = [5.0, 8.0, 5.0, 2.0]  # Nm
    
    def __init__(self, config: Dict[str, Any]):
        xml_path = Path(config.get("xml_path", "simulation/models/xarm_4dof.xml"))
        assert xml_path.exists(), f"MJCF not found: {xml_path}"
        
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data  = mujoco.MjData(self.model)
        
        self.n_joints = 4
        self.dt = self.model.opt.timestep
        
        # Camera renderer for SmolVLA (matches dataset: 84x84)
        self.renderer_84  = mujoco.Renderer(self.model, height=84, width=84)
        # Camera renderer for visualization (higher res)
        self.renderer_480 = mujoco.Renderer(self.model, height=480, width=480)
        
        # Joint address lookup
        self._q_addrs  = [self.model.joint(n).qposadr[0] for n in self.JOINT_NAMES]
        self._qd_addrs = [self.model.joint(n).dofadr[0]  for n in self.JOINT_NAMES]
        
        # EE + object site IDs
        self._ee_site_id  = self.model.site("ee_site").id
        self._obj_site_id = self.model.site("object_site").id
        
        mujoco.mj_forward(self.model, self.data)
    
    # ── State Access ────────────────────────────────────────────────
    
    def get_joint_pos(self) -> np.ndarray:
        """Returns [q1, q2, q3, gripper] in rad/m."""
        return np.array([self.data.qpos[a] for a in self._q_addrs])
    
    def get_joint_vel(self) -> np.ndarray:
        """Returns [qd1, qd2, qd3, qd_gripper] in rad/s or m/s."""
        return np.array([self.data.qvel[a] for a in self._qd_addrs])
    
    def get_ee_pos(self) -> np.ndarray:
        """Returns [x, y, z] of end-effector in world frame (m)."""
        return self.data.site_xpos[self._ee_site_id].copy()
    
    def get_object_pos(self) -> np.ndarray:
        """Returns [x, y, z] of manipulated object (m)."""
        return self.data.site_xpos[self._obj_site_id].copy()
    
    def get_state_vector(self) -> np.ndarray:
        """Returns full state [q; qd] — 8-dim."""
        return np.concatenate([self.get_joint_pos(), self.get_joint_vel()])
    
    # ── Control ─────────────────────────────────────────────────────
    
    def step(self, tau: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply joint torques and advance simulation by one timestep.
        
        Args:
            tau: [4] joint torques [Nm, Nm, Nm, Nm]
        Returns:
            obs dict with sensor readings
        """
        assert tau.shape == (4,), f"Expected [4] torques, got {tau.shape}"
        
        # Clip to limits
        tau_clipped = np.clip(tau, -np.array(self.TORQUE_LIMITS), 
                                    np.array(self.TORQUE_LIMITS))
        self.data.ctrl[:4] = tau_clipped
        mujoco.mj_step(self.model, self.data)
        
        return self._get_obs()
    
    def step_position(self, q_target: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Convenience: set joint position targets directly (bypasses MPC).
        Used for dataset replay validation.
        """
        # Use MuJoCo's built-in position servo
        self.data.qpos[self._q_addrs] = q_target
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()
    
    # ── Observations ─────────────────────────────────────────────────
    
    def _get_obs(self) -> Dict[str, np.ndarray]:
        return {
            "joint_pos":  self.get_joint_pos(),     # [4]
            "joint_vel":  self.get_joint_vel(),     # [4]
            "ee_pos":     self.get_ee_pos(),         # [3]
            "object_pos": self.get_object_pos(),     # [3]
        }
    
    def render_rgb(self, camera: str = "primary_cam", 
                   size: int = 84) -> np.ndarray:
        """
        Render RGB frame.
        
        Args:
            camera: camera name in MJCF
            size:   84 (matches dataset) or 480 (visualization)
        Returns:
            [H, W, 3] uint8 RGB array
        """
        renderer = self.renderer_84 if size == 84 else self.renderer_480
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()  # [H, W, 3] uint8
    
    def get_lidar_readings(self) -> np.ndarray:
        """Returns rangefinder distances [n_rays]. -1 = no hit."""
        n_rangefinders = len([s for s in range(self.model.nsensor)
                              if self.model.sensor_type[s] == 
                              mujoco.mjtSensor.mjSENS_RANGEFINDER])
        readings = np.array([self.data.sensordata[i] 
                             for i in range(n_rangefinders)])
        return readings
    
    # ── Utilities ────────────────────────────────────────────────────
    
    def reset(self, q_init: Optional[np.ndarray] = None):
        """Reset to home position or specified joint config."""
        mujoco.mj_resetData(self.model, self.data)
        if q_init is not None:
            for i, addr in enumerate(self._q_addrs):
                self.data.qpos[addr] = q_init[i]
        # Reset object to table surface
        self.data.qpos[7:10] = [0.4, 0.0, 0.37]  # object start pos
        mujoco.mj_forward(self.model, self.data)
    
    def check_success(self, height_threshold: float = 0.1) -> bool:
        """
        Success = object lifted above height_threshold above table surface.
        Table surface is at z=0.32 (table geom top).
        """
        obj_z = self.get_object_pos()[2]
        return obj_z > (0.32 + height_threshold)
    
    def close(self):
        """Clean up renderer resources."""
        self.renderer_84.close()
        self.renderer_480.close()
```

---

## 6. EVENT CAMERA SIMULATION

### Academic Grounding

The MuJoCo-ESIM paper proposes building upon vid2e such that, when event data is required, the simulation automatically hands the render to the processing pipeline. The rendered images of the camera view are then directly processed.

Among major simulation engines, none provide a maintained, first-class DVS sensor. MuJoCo integrates event simulation through ESIM, making the ESIM/v2e workflow the standard approach.

We use **v2e** (the Python implementation of ESIM) to convert MuJoCo RGB frames to synthetic events.

### Installation

```bash
pip install v2e        # video-to-events converter
# OR from source for latest:
git clone https://github.com/SensorsINI/v2e.git
cd v2e && pip install -e .
```

### Event Camera Simulator

```python
# simulation/cameras/event_camera.py

import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass, field

@dataclass
class EventCameraConfig:
    """Configuration matching a DAVIS346 sensor (realistic defaults)."""
    height: int = 84          # match RGB resolution
    width: int = 84
    threshold_pos: float = 0.2    # positive event threshold (log intensity)
    threshold_neg: float = 0.2    # negative event threshold
    refractory_period: float = 0.001  # seconds between events per pixel
    noise_rate: float = 0.001    # Hz, background noise events

@dataclass
class Event:
    """Single DVS event."""
    t: float   # timestamp (s)
    x: int     # pixel x
    y: int     # pixel y
    p: int     # polarity: +1 (brightness increase) or -1 (decrease)

class EventCameraSimulator:
    """
    Simulates a Dynamic Vision Sensor (DVS) from MuJoCo RGB frames.
    
    Uses the per-pixel log-intensity change model:
        event fires when: |log(I_t / I_{t-1})| > threshold
    
    Reference: v2e (Hu et al., CVPR 2021), MuJoCo-ESIM (Palinauskas et al., ICONS 2023)
    """
    
    def __init__(self, config: Optional[EventCameraConfig] = None):
        self.cfg = config or EventCameraConfig()
        
        # Per-pixel state
        self._log_intensity: Optional[np.ndarray] = None  # [H, W]
        self._last_event_time: Optional[np.ndarray] = None  # [H, W]
        self._t: float = 0.0
        
        self._initialized = False
    
    def reset(self):
        """Reset simulator state (call at start of each episode)."""
        self._log_intensity = None
        self._last_event_time = None
        self._t = 0.0
        self._initialized = False
    
    def process_frame(
        self, 
        rgb_frame: np.ndarray, 
        dt: float,
        interpolate_steps: int = 5
    ) -> List[Event]:
        """
        Generate events from a new RGB frame.
        
        Args:
            rgb_frame:  [H, W, 3] uint8 RGB
            dt:         time elapsed since last frame (s)
            interpolate_steps: number of substeps for temporal interpolation
        
        Returns:
            List of Event objects (may be empty if no motion)
        """
        assert rgb_frame.shape == (self.cfg.height, self.cfg.width, 3)
        
        # Convert to grayscale, then log-luminance
        gray = 0.2126 * rgb_frame[:,:,0] + \
               0.7152 * rgb_frame[:,:,1] + \
               0.0722 * rgb_frame[:,:,2]
        gray = np.clip(gray, 1.0, 255.0)  # avoid log(0)
        log_I = np.log(gray)  # [H, W]
        
        if not self._initialized:
            self._log_intensity = log_I.copy()
            self._last_event_time = np.full(
                (self.cfg.height, self.cfg.width), self._t
            )
            self._initialized = True
            return []
        
        events = []
        substep_dt = dt / interpolate_steps
        
        # Simple linear interpolation between frames
        delta_log = log_I - self._log_intensity  # [H, W]
        
        for step in range(interpolate_steps):
            t_sub = self._t + step * substep_dt
            partial_delta = delta_log * (step + 1) / interpolate_steps
            
            # Positive events (brightness increase)
            pos_mask = (partial_delta > self.cfg.threshold_pos) & \
                       ((t_sub - self._last_event_time) > self.cfg.refractory_period)
            
            # Negative events (brightness decrease)
            neg_mask = (partial_delta < -self.cfg.threshold_neg) & \
                       ((t_sub - self._last_event_time) > self.cfg.refractory_period)
            
            for y, x in zip(*np.where(pos_mask)):
                events.append(Event(t=t_sub, x=int(x), y=int(y), p=+1))
                self._last_event_time[y, x] = t_sub
            
            for y, x in zip(*np.where(neg_mask)):
                events.append(Event(t=t_sub, x=int(x), y=int(y), p=-1))
                self._last_event_time[y, x] = t_sub
        
        # Add Poisson noise events
        noise_count = np.random.poisson(self.cfg.noise_rate * dt)
        for _ in range(noise_count):
            nx = np.random.randint(0, self.cfg.width)
            ny = np.random.randint(0, self.cfg.height)
            events.append(Event(
                t=self._t + np.random.uniform(0, dt),
                x=nx, y=ny, 
                p=np.random.choice([-1, +1])
            ))
        
        self._log_intensity = log_I.copy()
        self._t += dt
        
        return sorted(events, key=lambda e: e.t)
    
    def to_voxel_grid(
        self,
        events: List[Event],
        t_start: float,
        t_end: float,
        n_bins: int = 5
    ) -> np.ndarray:
        """
        Convert event list to voxel grid representation for neural network input.
        
        Standard representation from Zhu et al. (CVPR 2019).
        
        Returns:
            voxel_grid: [n_bins, H, W] float32
            - Positive bins: n_bins//2 + positive polarity events
            - Negative bins: n_bins//2 + negative polarity events
        """
        voxel = np.zeros((n_bins, self.cfg.height, self.cfg.width), dtype=np.float32)
        
        if not events or t_end <= t_start:
            return voxel
        
        dt = t_end - t_start
        
        for e in events:
            if not (t_start <= e.t < t_end):
                continue
            
            t_norm = (e.t - t_start) / dt  # [0, 1)
            
            if e.p == +1:
                bin_idx = int(t_norm * (n_bins // 2))
                bin_idx = min(bin_idx, n_bins // 2 - 1)
            else:
                bin_idx = (n_bins // 2) + int(t_norm * (n_bins // 2))
                bin_idx = min(bin_idx, n_bins - 1)
            
            voxel[bin_idx, e.y, e.x] += 1.0
        
        # Normalize
        if voxel.max() > 0:
            voxel = voxel / voxel.max()
        
        return voxel
```

### Validating Event Camera Output

The event camera is valid when:
1. Static scene → near-zero events (< noise_rate events/s)
2. Moving arm → events concentrated along motion edges
3. Fast motion → more events than slow motion (proportional to pixel velocity × contrast)

```python
# In tests/test_sensor_outputs.py:
def test_event_cam_static_vs_moving():
    env = XArmEnv(config)
    ev_sim = EventCameraSimulator()
    
    # Static frame
    frame1 = env.render_rgb()
    frame2 = env.render_rgb()  # identical
    static_events = ev_sim.process_frame(frame2, dt=0.067)  # 15fps
    
    # Moving frame (apply step)
    env.step(np.array([2.0, 0, 0, 0]))  # move joint 1
    frame3 = env.render_rgb()
    moving_events = ev_sim.process_frame(frame3, dt=0.067)
    
    assert len(moving_events) > len(static_events) * 3, \
        "Moving arm should generate significantly more events"
```

---

## 7. LIDAR SIMULATION

### Approach: MuJoCo Rangefinders

MuJoCo provides `rangefinder` sensors that measure distance to the nearest geometry along a specified direction. We arrange 16 or 32 rangefinders in a dome pattern around the EE site to simulate a simplified 3D LiDAR.

```python
# simulation/cameras/lidar_sensor.py

import numpy as np
from typing import List, Tuple

def generate_lidar_directions(n_horizontal: int = 8, n_vertical: int = 4) -> np.ndarray:
    """
    Generate unit direction vectors for a dome-shaped LiDAR.
    
    Returns:
        directions: [n_rays, 3] unit vectors
    """
    directions = []
    
    for v in range(n_vertical):
        elevation = np.pi/2 * v / (n_vertical - 1)  # 0 to 90 degrees
        for h in range(n_horizontal):
            azimuth = 2 * np.pi * h / n_horizontal
            dx = np.cos(elevation) * np.cos(azimuth)
            dy = np.cos(elevation) * np.sin(azimuth)
            dz = np.sin(elevation)
            directions.append([dx, dy, dz])
    
    return np.array(directions)  # [n_horizontal * n_vertical, 3]


def rangefinder_to_pointcloud(
    readings: np.ndarray,
    directions: np.ndarray,
    origin: np.ndarray,
    max_range: float = 1.0
) -> np.ndarray:
    """
    Convert rangefinder readings to 3D point cloud.
    
    Args:
        readings:   [n_rays] distances (m), -1 = no hit
        directions: [n_rays, 3] unit direction vectors
        origin:     [3] sensor origin in world frame
        max_range:  cutoff distance (m)
    
    Returns:
        points: [n_hits, 3] world-frame 3D points (only valid hits)
    """
    valid = (readings >= 0) & (readings < max_range)
    
    if not np.any(valid):
        return np.zeros((0, 3))
    
    points = origin[np.newaxis, :] + \
             readings[valid, np.newaxis] * directions[valid, :]
    
    return points


class LiDARProcessor:
    """
    Processes rangefinder readings into features for the fusion encoder.
    """
    
    def __init__(self, n_rays: int = 32, max_range: float = 1.0):
        self.n_rays = n_rays
        self.max_range = max_range
        self.directions = generate_lidar_directions(8, 4)  # 32 rays
    
    def readings_to_feature(self, readings: np.ndarray) -> np.ndarray:
        """
        Convert raw rangefinder distances to a fixed-size feature vector.
        
        Returns:
            feature: [n_rays + 3] = raw distances + [mean, min, max]
        """
        # Clip and normalize
        clipped = np.clip(readings, 0, self.max_range) / self.max_range
        # Replace no-hit (-1) with 1.0 (max normalized distance)
        clipped[readings < 0] = 1.0
        
        stats = np.array([clipped.mean(), clipped.min(), clipped.max()])
        return np.concatenate([clipped, stats])  # [n_rays + 3]
```

**Adding 32 rangefinders to MJCF:**

Generate the MJCF sensor block programmatically:
```python
# simulation/models/generate_lidar_sensors.py

def generate_lidar_sensor_xml(n_rays: int = 32, cutoff: float = 1.0) -> str:
    directions = generate_lidar_directions(8, 4)
    lines = []
    for i, d in enumerate(directions):
        # MuJoCo rangefinder direction is in local site frame
        lines.append(
            f'<rangefinder name="lidar_{i:02d}" '
            f'site="ee_site" cutoff="{cutoff}"/>'
        )
    return "\n    ".join(lines)
```

---

## 8. MULTIMODAL SENSOR FUSION

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FUSION ENCODER                                │
│                                                                 │
│  RGB [3,84,84] ──→ ResNet18 stub ──→ [256]                     │
│                                           │                     │
│  Events [5,84,84] ──→ Conv3D ────────→ [128]                   │
│                                           │                     │
│  LiDAR [35] ──→ MLP ─────────────────→ [64]                    │
│                                           │                     │
│  Proprio [4] ──→ Linear ─────────────→ [32]                    │
│                                           │                     │
│                               Concat → [480] → MLP → [256]     │
│                               (fused observation embedding)     │
└─────────────────────────────────────────────────────────────────┘
```

```python
# fusion/fusion_model.py

import torch
import torch.nn as nn
import yaml
from pathlib import Path

class RGBEncoder(nn.Module):
    """Lightweight ResNet-18 style encoder for 84x84 RGB."""
    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 42x42
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 21x21
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1), # 11x11
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),               # 4x4
            nn.Flatten(),                               # 1024
            nn.Linear(1024, out_dim),
            nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, 84, 84] float32 (normalized 0-1)"""
        return self.net(x)

class EventEncoder(nn.Module):
    """Encodes event voxel grids [n_bins, H, W]."""
    def __init__(self, n_bins: int = 5, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_bins, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(512, out_dim),
            nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_bins, 84, 84]"""
        return self.net(x)

class LiDAREncoder(nn.Module):
    """Encodes LiDAR rangefinder feature vector."""
    def __init__(self, in_dim: int = 35, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, out_dim), nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim]"""
        return self.net(x)

class ProprioEncoder(nn.Module):
    """Encodes joint state vector."""
    def __init__(self, in_dim: int = 4, out_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(),
            nn.Linear(32, out_dim), nn.ReLU()
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim]"""
        return self.net(x)

class MultimodalFusionEncoder(nn.Module):
    """
    Fuses RGB + Event + LiDAR + Proprioception into a single observation embedding.
    
    Output: [B, 256] observation embedding
    Used as input to SmolVLA (replacing or augmenting raw RGB).
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.rgb_enc    = RGBEncoder(out_dim=256)
        self.event_enc  = EventEncoder(n_bins=config.get("event_bins", 5), out_dim=128)
        self.lidar_enc  = LiDAREncoder(in_dim=config.get("lidar_dim", 35), out_dim=64)
        self.proprio_enc = ProprioEncoder(in_dim=config.get("proprio_dim", 4), out_dim=32)
        
        total_dim = 256 + 128 + 64 + 32  # 480
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
    
    def forward(
        self,
        rgb: torch.Tensor,       # [B, 3, 84, 84]
        events: torch.Tensor,    # [B, n_bins, 84, 84]
        lidar: torch.Tensor,     # [B, lidar_dim]
        proprio: torch.Tensor    # [B, 4]
    ) -> torch.Tensor:
        """Returns [B, 256] fused embedding."""
        
        f_rgb    = self.rgb_enc(rgb)        # [B, 256]
        f_event  = self.event_enc(events)   # [B, 128]
        f_lidar  = self.lidar_enc(lidar)    # [B, 64]
        f_proprio = self.proprio_enc(proprio) # [B, 32]
        
        fused = torch.cat([f_rgb, f_event, f_lidar, f_proprio], dim=-1)  # [B, 480]
        return self.fusion(fused)  # [B, 256]
    
    @classmethod
    def rgb_only(cls, config: dict) -> "MultimodalFusionEncoder":
        """Returns model configured for RGB-only mode (Phase 1)."""
        # Events, LiDAR set to zeros externally; architecture unchanged
        return cls(config)
```

### Modality Ablation Strategy

Phase 1: RGB only (match baseline)
Phase 2: RGB + Proprioception
Phase 3: RGB + Proprioception + Events
Phase 4: RGB + Proprioception + Events + LiDAR

Track success rate and tracking error at each phase → quantify the contribution of each sensor.

---

## 9. SMOLVLA INTEGRATION — CORRECTED ARCHITECTURE

### What SmolVLA Actually Outputs for xArm Dataset

From the dataset format: action_dim = 4 (absolute joint targets, not deltas). SmolVLA, when run on `lerobot/xarm_lift_medium` data or inference on dataset images, outputs:

```python
action_chunk: np.ndarray  # shape [10, 4]
# action_chunk[k] = [q1_target, q2_target, q3_target, gripper_target] at step k
# These are JOINT SPACE targets, not Cartesian EE deltas
# Units: radians for joints, meters for gripper
```

This is a 10-step plan at 15 Hz → covers 0.67 seconds of future motion.

### Corrected Action Processor

```python
# smolvla/action_processor.py

import numpy as np
from typing import Tuple

class ActionChunkProcessor:
    """
    Converts SmolVLA action_chunk to reference trajectory for SL-MPC.
    
    For lerobot/xarm_lift_medium: action_chunk is already in joint space,
    so NO IK is needed. Direct passthrough with validation.
    
    Key insight: action_chunk[k] = q_ref for timestep k.
    MPC tracks this reference at ~100 Hz (upsampled from 15 Hz chunk).
    """
    
    JOINT_LIMITS = np.array([
        [-2.618, 2.618],   # joint1
        [-2.059, 2.059],   # joint2
        [-3.927, 0.193],   # joint3
        [0.0, 0.085],      # gripper
    ])
    
    def __init__(self, control_hz: float = 100.0, vla_hz: float = 15.0):
        self.control_hz = control_hz
        self.vla_hz = vla_hz
        self.upsample_factor = int(control_hz / vla_hz)  # = 6.67 → 7
    
    def validate_chunk(self, chunk: np.ndarray) -> bool:
        """Returns True if chunk is within joint limits."""
        if chunk.shape != (10, 4):
            return False
        for step in range(10):
            q = chunk[step, :3]   # joint angles only (not gripper)
            for j in range(3):
                if not (self.JOINT_LIMITS[j, 0] <= q[j] <= self.JOINT_LIMITS[j, 1]):
                    return False
        return True
    
    def chunk_to_reference(
        self, 
        chunk: np.ndarray,
        current_q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 10-step action chunk to dense reference trajectory for MPC.
        
        Uses linear interpolation between chunk steps.
        
        Args:
            chunk:     [10, 4] joint targets from SmolVLA
            current_q: [4] current joint positions
        
        Returns:
            q_ref:    [N_mpc, 4] reference positions for MPC horizon
            qd_ref:   [N_mpc, 4] reference velocities (finite difference)
        """
        if not self.validate_chunk(chunk):
            # Fallback: hold current position
            q_ref  = np.tile(current_q, (10, 1))
            qd_ref = np.zeros_like(q_ref)
            return q_ref, qd_ref
        
        # Clip chunk to limits
        chunk_clipped = np.clip(chunk, self.JOINT_LIMITS[:, 0], 
                                       self.JOINT_LIMITS[:, 1])
        
        # Prepend current position as step 0
        full_traj = np.vstack([current_q[np.newaxis, :], chunk_clipped])  # [11, 4]
        
        # Linear interpolation to control frequency
        n_out = 10 * self.upsample_factor
        q_ref = np.zeros((n_out, 4))
        
        for k in range(n_out):
            # Which chunk interval?
            t_norm = k / self.upsample_factor  # in chunk steps
            i_low = int(t_norm)
            i_high = min(i_low + 1, 10)
            alpha = t_norm - i_low
            
            q_ref[k] = (1 - alpha) * full_traj[i_low] + alpha * full_traj[i_high]
        
        # Finite difference for velocity reference
        dt_control = 1.0 / self.control_hz
        qd_ref = np.diff(q_ref, axis=0, prepend=q_ref[:1]) / dt_control
        qd_ref = np.clip(qd_ref, -2.0, 2.0)  # velocity limit
        
        return q_ref, qd_ref
```

### SmolVLA Inference on Real Dataset Images

**Critical section: Validating SmolVLA actually works**

```python
# evaluation/benchmarks/run_smolvla_only.py
"""
Validates SmolVLA by running it on real dataset images and comparing
predicted action chunks to ground truth actions.

This is the ONLY way to know if SmolVLA actually works as System 2.
"""

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np, json, torch
from pathlib import Path
from tqdm import tqdm

def run_smolvla_dataset_eval(
    n_episodes: int = 10,
    smolvla_endpoint: str = None  # None = run locally (needs GPU)
):
    """
    For each episode in the dataset:
    1. Load real RGB image + state from dataset
    2. Query SmolVLA with the image + task description
    3. Compare first action of chunk to ground truth next action
    
    Metrics:
    - Mean Absolute Error (MAE) in joint space vs ground truth
    - Distribution of predicted actions
    """
    
    ds = LeRobotDataset("lerobot/xarm_lift_medium", root="data/cache")
    
    results = {
        "episodes_evaluated": 0,
        "mae_per_joint": [],      # [n_frames, 4]
        "chunk_mae":  [],         # first action prediction vs GT
        "episode_ids": []
    }
    
    task_instruction = "lift the object from the table"
    
    episode_starts = []
    for ep in range(n_episodes):
        # Find first frame of each episode
        for i in range(ds.num_frames):
            if ds[i]["episode_index"].item() == ep and \
               ds[i]["frame_index"].item() == 0:
                episode_starts.append(i)
                break
    
    for ep_idx, frame_start in enumerate(episode_starts):
        sample = ds[frame_start]
        
        # Real RGB image from dataset
        rgb = sample["observation.image"]  # [3, 84, 84] float32, 0-1
        state = sample["observation.state"].numpy()  # [4]
        gt_action = sample["action"].numpy()  # [4] next joint targets
        
        # Query SmolVLA
        if smolvla_endpoint:
            # Via HTTP (Colab server)
            import requests, base64, io
            from PIL import Image
            
            rgb_pil = Image.fromarray(
                (rgb.permute(1,2,0).numpy() * 255).astype(np.uint8)
            )
            buf = io.BytesIO()
            rgb_pil.save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            
            resp = requests.post(f"{smolvla_endpoint}/predict", json={
                "image_b64": b64,
                "instruction": task_instruction,
                "current_joints": state.tolist()
            }, timeout=5.0)
            
            if resp.status_code != 200:
                print(f"Episode {ep_idx}: VLA query failed")
                continue
            
            chunk = np.array(resp.json()["action_chunk"])  # [10, 4]
        else:
            # Local inference placeholder
            # In practice: load SmolVLA model locally
            print("Local SmolVLA inference not yet implemented")
            continue
        
        # Compare first predicted action to ground truth
        pred_action = chunk[0]  # SmolVLA's immediate next action
        mae = np.abs(pred_action - gt_action)
        
        results["mae_per_joint"].append(mae.tolist())
        results["chunk_mae"].append(float(mae.mean()))
        results["episode_ids"].append(ep_idx)
        results["episodes_evaluated"] += 1
    
    # Compute aggregate metrics
    if results["mae_per_joint"]:
        mae_arr = np.array(results["mae_per_joint"])
        results["summary"] = {
            "mean_mae":    float(mae_arr.mean()),
            "per_joint_mae": mae_arr.mean(axis=0).tolist(),
            "n_evaluated": results["episodes_evaluated"]
        }
        print(f"\nSmolVLA Evaluation Results:")
        print(f"  Episodes: {results['episodes_evaluated']}")
        print(f"  Mean MAE: {results['summary']['mean_mae']:.4f} rad")
        print(f"  Per-joint: {[f'{x:.4f}' for x in results['summary']['per_joint_mae']]}")
    
    # Save
    out_path = Path("evaluation/results/smolvla_eval.json")
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
    
    return results
```

---

## 10. STUART-LANDAU MPC — TRAJECTORY TRACKING

### 4-DOF Extension

The existing SL solver handles 2-3 DOF. Extend to 4-DOF (3 joints + gripper):

```
N_var = 4 × N_horizon (e.g., 4 × 10 = 40 oscillators)
```

The gripper joint has simpler dynamics (prismatic, low inertia) — include it in the state vector but with separate, smaller cost weight.

```python
# config/mpc.yaml
horizon: 10          # N
dt: 0.01             # control timestep (s) → 100 Hz
n_joints: 4          # 3 arm + 1 gripper

# Cost matrices (diagonal entries)
Q_pos: [100, 100, 100, 20]    # joint position error weights
Q_vel: [10, 10, 10, 5]        # joint velocity error weights
R:     [0.1, 0.1, 0.1, 0.05]  # torque cost weights
P:     [200, 200, 200, 40]    # terminal state cost (10x Q)

# SL solver settings
sl_n_steps: 500       # oscillator integration steps
sl_dt_osc:  0.001     # oscillator timestep
sl_mu:      1.0       # bifurcation parameter
rho_aug:    10.0      # augmented Lagrangian penalty
```

---

## 11. FULL SYSTEM INTEGRATION

### Control Loop

```python
# system/controller.py

import asyncio, threading, time, logging
import numpy as np
from enum import Enum, auto
from typing import Optional
from simulation.envs.xarm_env import XArmEnv
from smolvla.client import SmolVLAClient
from smolvla.action_processor import ActionChunkProcessor
from system.trajectory_buffer import TrajectoryBuffer
from mpc.sl_solver import StuartLandauSolver
from mpc.qp_builder import build_qp
from dynamics.dynamics import compute_M, compute_C, compute_G
from visualization.live_dashboard import LiveDashboard
from evaluation.metrics import MetricsTracker

logger = logging.getLogger(__name__)

class ControlState(Enum):
    IDLE        = auto()
    VLA_QUERY   = auto()
    EXECUTING   = auto()
    DONE        = auto()
    BLOCKED     = auto()  # VLA unavailable

class DualSystemController:
    """
    Full System 1 (SL-MPC) + System 2 (SmolVLA) controller.
    
    System 2 runs in background thread via asyncio.
    System 1 runs in main thread at control frequency.
    Communication via TrajectoryBuffer (lock-free on reads).
    """
    
    def __init__(self, env: XArmEnv, config: dict):
        self.env = env
        self.cfg = config
        
        self.sl_solver = StuartLandauSolver(
            n_joints=4,
            horizon=config["mpc"]["horizon"]
        )
        self.trajectory_buffer = TrajectoryBuffer(n_joints=4)
        self.action_processor = ActionChunkProcessor()
        self.vla_client = SmolVLAClient(config["smolvla"])
        self.metrics = MetricsTracker()
        
        self.state = ControlState.IDLE
        self._running = False
        self._event_loop = None
        self._vla_thread = None
        
        self.dashboard = LiveDashboard(n_joints=4) if config.get("show_dashboard") else None
    
    # ── System 1: SL-MPC (called at 100 Hz) ─────────────────────────
    
    def system1_step(self) -> np.ndarray:
        """One MPC step. Returns optimal torques [4]."""
        q    = self.env.get_joint_pos()
        qdot = self.env.get_joint_vel()
        
        # Get reference from trajectory buffer (from VLA's last chunk)
        N  = self.cfg["mpc"]["horizon"]
        dt = self.cfg["mpc"]["dt"]
        q_ref, qd_ref = self.trajectory_buffer.get_reference(q, N, dt)
        
        # Build and solve QP
        H, c, A_ineq, b_ineq = build_qp(q, qdot, q_ref, qd_ref, self.cfg["mpc"])
        tau = self.sl_solver.solve(H, c, A_ineq, b_ineq)
        
        self.metrics.log_mpc_step(q, qdot, q_ref[0], tau)
        return tau
    
    # ── System 2: SmolVLA (async, ~5 Hz) ────────────────────────────
    
    async def _system2_loop(self, instruction: str):
        """Background coroutine. Queries VLA and updates trajectory buffer."""
        await self.vla_client.start()
        
        while self._running:
            if self.state == ControlState.IDLE or \
               self.trajectory_buffer.chunk_expired():
                
                self.state = ControlState.VLA_QUERY
                rgb = self.env.render_rgb(size=84)  # [84, 84, 3]
                q   = self.env.get_joint_pos()
                
                response = await self.vla_client.query_async(
                    rgb_image=rgb,
                    instruction=instruction,
                    current_joints=q.tolist()
                )
                
                if response is not None:
                    chunk = np.array(response.action_chunk)  # [10, 4]
                    q_ref, qd_ref = self.action_processor.chunk_to_reference(
                        chunk, q
                    )
                    self.trajectory_buffer.update(q_ref, qd_ref)
                    self.metrics.log_vla_query(response.latency_ms)
                    self.state = ControlState.EXECUTING
                    logger.info(f"[System2] New chunk | lat={response.latency_ms:.0f}ms")
                else:
                    self.state = ControlState.BLOCKED
                    logger.warning("[System2] VLA query failed — holding last reference")
            
            await asyncio.sleep(0.2)  # 5 Hz
        
        await self.vla_client.stop()
    
    # ── Main Entry Point ─────────────────────────────────────────────
    
    def run_episode(
        self, 
        instruction: str, 
        max_steps: int = 1000,
        q_init: Optional[np.ndarray] = None
    ) -> dict:
        """
        Run one full episode with System 1 + System 2.
        
        Args:
            instruction:  Natural language task string
            max_steps:    Maximum control steps
            q_init:       Initial joint config (None = home)
        
        Returns:
            episode_result dict with metrics
        """
        self.env.reset(q_init)
        self.metrics.reset()
        self.trajectory_buffer.reset()
        self._running = True
        
        # Start event loop in background thread for System 2
        self._event_loop = asyncio.new_event_loop()
        self._vla_thread = threading.Thread(
            target=self._event_loop.run_forever, daemon=True
        )
        self._vla_thread.start()
        
        # Schedule System 2 as background task
        asyncio.run_coroutine_threadsafe(
            self._system2_loop(instruction), 
            self._event_loop
        )
        
        # System 1: main control loop
        if self.dashboard:
            self.dashboard.start()
        
        t_start = time.perf_counter()
        
        for step in range(max_steps):
            t_step = time.perf_counter()
            
            # MPC step
            tau = self.system1_step()
            self.env.step(tau)
            
            # Check success
            success = self.env.check_success()
            if success:
                logger.info(f"[Episode] SUCCESS at step {step}")
                break
            
            # Dashboard update (every 10 steps)
            if self.dashboard and step % 10 == 0:
                self.dashboard.update(self.metrics, self.env)
            
            # Timing
            elapsed = time.perf_counter() - t_step
            sleep_t = self.cfg["mpc"]["dt"] - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            elif sleep_t < -0.002:
                logger.warning(f"Control overrun: {-sleep_t*1000:.1f}ms")
        
        # Teardown
        self._running = False
        self._event_loop.call_soon_threadsafe(self._event_loop.stop)
        self._vla_thread.join(timeout=2.0)
        
        total_time = time.perf_counter() - t_start
        
        result = {
            "success": success,
            "steps": step + 1,
            "total_time_s": total_time,
            "final_tracking_error_rad": self.metrics.latest_tracking_error(),
            "mean_vla_latency_ms": self.metrics.mean_vla_latency(),
            "constraint_violations": self.metrics.constraint_violations,
            "controller_state": self.state.name
        }
        
        return result
```

---

## 12. BENCHMARKING & METRICS

### Metrics Definition

```python
# evaluation/metrics.py

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List
import json, time

@dataclass
class EpisodeResult:
    episode_idx: int
    success: bool
    steps_to_success: int
    total_steps: int
    tracking_error_rad: List[float]    # per-step ||q - q_ref||
    torques_nm: List[List[float]]      # per-step τ [4]
    vla_latencies_ms: List[float]      # per VLA query
    vla_query_count: int
    controller_state_log: List[str]
    object_final_height_m: float
    
    @property
    def mean_tracking_error(self) -> float:
        return float(np.mean(self.tracking_error_rad)) if self.tracking_error_rad else 0.0
    
    @property
    def mean_control_effort(self) -> float:
        if not self.torques_nm:
            return 0.0
        return float(np.mean([np.linalg.norm(t) for t in self.torques_nm]))

class BenchmarkResults:
    """Aggregates results across multiple episodes."""
    
    def __init__(self, benchmark_name: str):
        self.name = benchmark_name
        self.episodes: List[EpisodeResult] = []
        self.timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    def add(self, ep: EpisodeResult):
        self.episodes.append(ep)
    
    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.success for e in self.episodes) / len(self.episodes)
    
    @property
    def mean_tracking_error(self) -> float:
        return float(np.mean([e.mean_tracking_error for e in self.episodes]))
    
    @property
    def mean_vla_latency_ms(self) -> float:
        all_latencies = []
        for e in self.episodes:
            all_latencies.extend(e.vla_latencies_ms)
        return float(np.mean(all_latencies)) if all_latencies else 0.0
    
    def summary(self) -> dict:
        return {
            "benchmark": self.name,
            "timestamp": self.timestamp,
            "n_episodes": len(self.episodes),
            "success_rate": self.success_rate,
            "mean_tracking_error_rad": self.mean_tracking_error,
            "mean_vla_latency_ms": self.mean_vla_latency_ms,
            "mean_control_effort_nm": float(
                np.mean([e.mean_control_effort for e in self.episodes])
            ),
            "mean_steps_to_success": float(
                np.mean([e.steps_to_success for e in self.episodes 
                         if e.success])
            ) if any(e.success for e in self.episodes) else None
        }
    
    def save(self, path: str):
        out = {
            "summary": self.summary(),
            "episodes": [
                {
                    "idx": e.episode_idx,
                    "success": e.success,
                    "steps": e.total_steps,
                    "tracking_error_mean": e.mean_tracking_error,
                    "control_effort_mean": e.mean_control_effort,
                    "vla_latency_mean": float(np.mean(e.vla_latencies_ms))
                                        if e.vla_latencies_ms else 0.0
                }
                for e in self.episodes
            ]
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {path}")
```

### Benchmark Suite

| Benchmark | Description | N episodes | Metric | Expected baseline |
|-----------|-------------|-----------|--------|------------------|
| **B1: Dataset Replay** | Replay real dataset episodes in MuJoCo, track with SL-MPC | 50 | tracking error (rad) | < 0.1 rad |
| **B2: MPC Solo** | SL-MPC tracks sinusoidal reference, no VLA | 20 | tracking error (rad) | < 0.05 rad |
| **B3: VLA Prediction** | SmolVLA on real dataset images, compare to GT | 20 | joint MAE (rad) | < 0.2 rad |
| **B4: Full System** | VLA + MPC end-to-end in MuJoCo, "lift object" task | 30 | success rate (%) | report honestly |
| **B5: Sensor Ablation** | B4 with: RGB only / +proprio / +events / +lidar | 30 each | success rate Δ | quantify each modality |

**Report all numbers honestly. If success rate is 30%, write 30%.**

---

## 13. VISUALIZATION SUITE

### 1. Live Dashboard (During Experiments)

Six panels updating at ~5 Hz:
```python
# visualization/live_dashboard.py
# Panels:
# [0,0] Joint positions actual vs reference (4 joints, colored)
# [0,1] Control torques over time with limit lines (red dashed)
# [1,0] Joint tracking error ||q - q_ref|| (per joint + total norm)
# [1,1] QP cost over time (log scale) — SL solver convergence indicator
# [2,0] SmolVLA latency bar chart (per query, with 200ms target line)
# [2,1] EE trajectory in 3D (use mpl_toolkits.mplot3d)
```

### 2. Episode Replay Viewer

```python
# visualization/episode_viewer.py
"""
Replays any episode with synchronized:
- MuJoCo 3D render (480x480)
- RGB camera view (84x84, upscaled)
- Event camera voxel visualization
- LiDAR point cloud (3D scatter)
- Joint position time series
- Action chunk overlay

Usage:
    python -m visualization.episode_viewer --episode 42
"""
```

### 3. Results Figures (For Thesis)

```python
# visualization/report_generator.py
"""
Auto-generates publication-quality figures from benchmark results.

Figures:
  fig1_architecture.pdf    ← system block diagram (matplotlib)
  fig2_tracking.pdf        ← MPC tracking performance, 3 subplots
  fig3_success_rate.pdf    ← bar chart: success rate per benchmark
  fig4_latency.pdf         ← VLA latency distribution (violin + box)
  fig5_ablation.pdf        ← sensor ablation Δsuccess rate
  fig6_episode.pdf         ← single full episode visualization

Usage:
    python -m visualization.report_generator --results-dir evaluation/results/
"""
```

---

## 14. TESTING & VALIDATION PROTOCOL

### Phase-Gated Validation Gates

**Gate 0: Environment (Before Any Code)**
```
[ ] Dataset downloaded: lerobot/xarm_lift_medium, 800 episodes confirmed
[ ] Dataset log: logs/dataset_download.json exists and shows verified=True
[ ] MJCF loads: mujoco.MjModel.from_xml_path("simulation/models/xarm_4dof.xml") succeeds
[ ] Renderer works: env.render_rgb() returns [84, 84, 3] uint8
[ ] Agent memory files exist: docs/agent/AGENT_STATE.md, TODO.md, PROGRESS.md
```

**Gate 1: Dataset Audit (BEFORE any model or control code)**
```
[ ] notebooks/00_dataset_audit.ipynb runs end-to-end without error
[ ] Shows: 800 episodes, 20000 frames, image shape [3,84,84], action shape [4]
[ ] Shows: episode length distribution (mean, std, min, max) — printed and logged
[ ] Shows: sample images from 3 different episodes — visible, not black/corrupted
[ ] Shows: joint angle range for each DOF from full dataset
[ ] Log: logs/dataset_audit.json written with all above facts
```

**Gate 2: MuJoCo Validation**
```
[ ] Dataset replay: replay 5 episodes from dataset in MuJoCo using step_position()
[ ] FK match: env.get_ee_pos() vs analytical forward_kinematics(q) < 1mm error
[ ] Sensor check: event_camera returns events on motion, ~0 on static
[ ] LiDAR check: rangefinders return finite distances to table/objects
[ ] Render match: MuJoCo render visually resembles dataset images (write notebook)
[ ] Log: logs/gate2_validation.json with all pass/fail results
```

**Gate 3: Dynamics Validation**
```
[ ] M(q) PD: positive definite for 20 random configs
[ ] M(q) symmetric: np.allclose(M, M.T) for 10 random configs
[ ] G[0] = 0: for 10 random configs
[ ] Passivity: x^T(Mdot-2C)x < 1e-6 for random x,q,qdot
[ ] MuJoCo inverse dynamics comparison: < 5% error on 5 configs
[ ] Log: logs/gate3_dynamics.json
```

**Gate 4: SL-MPC Validation**
```
[ ] Point-to-point: from q=[0,0,0,0] to q_target, error < 0.05 rad in 500 steps
[ ] SL vs OSQP: < 5% cost deviation on 20 random QPs (log all 20 results)
[ ] Constraint satisfaction: torque limits never exceeded in 200-step run
[ ] Control rate: MPC runs at >= 100 Hz on test machine (measure actual Hz)
[ ] Log: logs/gate4_mpc.json with all 20 QP comparison results and timing
```

**Gate 5: SmolVLA Validation**
```
[ ] Server health: /health endpoint returns 200 from local machine
[ ] Inference: /predict returns action_chunk shape [10, 4]
[ ] On real images: evaluate on 10 dataset episodes, log MAE (not fabricated)
[ ] Latency: < 500ms per query on T4 GPU (log actual measurements)
[ ] Graceful fail: client returns None when server down, MPC continues
[ ] Log: logs/gate5_smolvla.json with actual MAE and latency measurements
```

**Gate 6: Full System**
```
[ ] End-to-end: 10 episodes of "lift the object" task, log actual success rate
[ ] No crashes: 10 episodes with no Python exceptions
[ ] Dashboard: all 6 panels update in real-time during episode
[ ] Sensor ablation: run 10 episodes each: RGB-only, +events, +lidar
[ ] Log: evaluation/results/benchmark_full_system.json
```

---

## 15. IMPLEMENTATION ROADMAP (PHASED)

### Phase 0 — Setup and Memory (Day 1)
```
Tasks:
  0.1  Create full project structure (mkdir -p for all dirs in §3)
  0.2  Create docs/agent/AGENT_STATE.md, TODO.md, PROGRESS.md
  0.3  Create requirements.txt with pinned versions
  0.4  Create config/paths.yaml, config/robot.yaml, config/mpc.yaml
  0.5  Run download_dataset.py → verify Gate 0
  0.6  Run data_inspector.py → verify Gate 1, write notebook 00

Exit: Gate 0 + Gate 1 checked. PROGRESS.md updated with download log.
```

### Phase 1 — Simulation Foundation (Days 2–3)
```
Tasks:
  1.1  Create simulation/models/xarm_4dof.xml
  1.2  Implement simulation/envs/xarm_env.py
  1.3  Implement simulation/cameras/rgb_camera.py (thin wrapper)
  1.4  Implement data/loaders/episode_player.py (replay from dataset)
  1.5  Write simulation/tests/test_env_loads.py, test_camera_renders.py
  1.6  Validate dataset replay (replay 5 episodes, log FK match error)
  1.7  Run notebook 01_env_validation.ipynb

Exit: Gate 2 fully checked. All tests green.
```

### Phase 2 — Dynamics & MPC (Days 4–6)
```
Tasks:
  2.1  dynamics/kinematics.py: FK, Jacobian, IK for 4-DOF xArm
  2.2  dynamics/dynamics.py: M(q), C(q,qdot), G(q)
  2.3  dynamics/tests/: all analytical property tests
  2.4  mpc/linearize.py: A, B matrices, ZOH
  2.5  mpc/qp_builder.py: H, c, A_ineq construction
  2.6  mpc/sl_solver.py: extend existing solver to 4-DOF (wrap, don't rewrite)
  2.7  mpc/tests/test_sl_vs_osqp.py: 20-trial comparison (log all)
  2.8  Run notebook 02_mpc_solo_test.ipynb

Exit: Gate 3 + Gate 4 fully checked.
```

### Phase 3 — Sensors (Days 7–8)
```
Tasks:
  3.1  simulation/cameras/event_camera.py: EventCameraSimulator
  3.2  simulation/cameras/lidar_sensor.py: LiDARProcessor
  3.3  Add 32 rangefinders to xarm_4dof.xml
  3.4  sensors/event_processing.py: voxel grid representation
  3.5  sensors/lidar_processing.py: point cloud to feature
  3.6  simulation/tests/test_sensor_outputs.py
  3.7  Run notebook 02_sensor_validation.ipynb with event + lidar plots

Exit: Gate 2 sensor items checked. Event camera shows motion-correlated events.
```

### Phase 4 — Fusion & SmolVLA (Days 9–11)
```
Tasks:
  4.1  fusion/encoders/: implement all 4 encoders
  4.2  fusion/fusion_model.py: MultimodalFusionEncoder
  4.3  fusion/tests/test_fusion_shapes.py: shape tests for all inputs
  4.4  smolvla/client.py: async HTTP client
  4.5  smolvla/action_processor.py: chunk_to_reference
  4.6  Set up Colab SmolVLA server (from original tech spec §10)
  4.7  Run evaluation/benchmarks/run_smolvla_only.py on 10 episodes
  4.8  Log actual MAE numbers to evaluation/results/

Exit: Gate 5 fully checked. Actual SmolVLA MAE logged (not fabricated).
```

### Phase 5 — Integration (Days 12–14)
```
Tasks:
  5.1  system/trajectory_buffer.py
  5.2  system/controller.py: DualSystemController
  5.3  system/state_machine.py
  5.4  visualization/live_dashboard.py
  5.5  visualization/episode_viewer.py
  5.6  Run 10 episodes of "lift the object" — log actual success rate
  5.7  Fix any integration bugs (expect this to take most of Phase 5)

Exit: Gate 6 partially checked (no crashes, actual success rate logged).
```

### Phase 6 — Benchmarking (Days 15–17)
```
Tasks:
  6.1  evaluation/benchmarks/run_dataset_replay.py (B1)
  6.2  evaluation/benchmarks/run_mpc_solo.py (B2)
  6.3  evaluation/benchmarks/run_smolvla_only.py (B3, more episodes)
  6.4  evaluation/benchmarks/run_full_system.py (B4, 30 episodes)
  6.5  Sensor ablation loop for B5
  6.6  Save all results to evaluation/results/*.json
  6.7  visualization/report_generator.py: generate thesis figures
  6.8  Run notebook 05_full_system.ipynb

Exit: Gate 6 fully checked. All 5 benchmarks have real results.
```

---

## 16. AGENT CODING STANDARDS

### Style
```python
# Type annotations: mandatory on all public functions
def compute_M(q: np.ndarray, config: dict) -> np.ndarray:
    """
    Compute 3x3 (or 4x4) mass matrix.
    
    Args:
        q: joint positions [n_joints], units: rad (joints 1-3), m (gripper)
    Returns:
        M: [n_joints, n_joints] mass matrix, units: kg⋅m²
    Raises:
        AssertionError if M is not PD (debug mode)
    """
```

### Logging (no print statements in production code)
```python
import logging
logger = logging.getLogger(__name__)

# Use these levels:
logger.debug("Every MPC step variable dump")       # disabled by default
logger.info("State transitions, VLA queries")      # enabled
logger.warning("IK failures, VLA timeouts")        # always visible
logger.error("Unrecoverable errors")               # always visible
```

### No bare print() calls (except in scripts/download/inspect utilities).

### Config Loading Pattern
```python
# ALWAYS load from config files. No hardcoded values.
import yaml
from pathlib import Path

def load_config(name: str) -> dict:
    path = Path("config") / f"{name}.yaml"
    assert path.exists(), f"Config not found: {path}"
    with open(path) as f:
        return yaml.safe_load(f)

mpc_cfg = load_config("mpc")
horizon = mpc_cfg["horizon"]  # correct
# horizon = 10                 # WRONG — hardcoded
```

### Test Structure
```python
# Every test file follows this pattern:
import pytest
import numpy as np

class TestMassMatrix:
    def setup_method(self):
        """Called before each test."""
        self.q_nominal = np.array([0.3, 0.5, -0.4, 0.02])
    
    def test_positive_definite(self):
        """Property: M must be PD for all valid q."""
        M = compute_M(self.q_nominal)
        eigvals = np.linalg.eigvals(M)
        assert np.all(eigvals > 0), \
            f"M not PD: eigvals={eigvals}"
    
    # Run with: pytest dynamics/tests/ -v
```

### Memory File Template

`docs/agent/AGENT_STATE.md`:
```markdown
# Agent State — Updated: [TIMESTAMP]

## Current Phase
Phase X: [name]

## Currently Working On
Task X.Y: [description]
File being modified: [path]

## Last Completed Task
Task X.Y-1: [description]
Result: [pass/fail + actual number or file path]

## Blockers
[None] OR [description of blocker + what info is needed]

## Next Task
Task X.Y+1: [description]
```

`docs/agent/PROGRESS.md`:
```markdown
# Progress Log

## [TIMESTAMP] Task X.Y — [description]
Status: COMPLETE
Result: [actual output — numbers, file path, test results]
Files created: [list]
Files modified: [list]
```

---

## APPENDIX A: Frequently Asked Questions

**Q: Can I download DROID / Open X-Embodiment / LSMO?**
A: No. DROID = 1.7TB. OXE requires RLDS tools. LSMO has no public access. Use `lerobot/xarm_lift_medium` only.

**Q: What if SmolVLA returns nonsense actions?**
A: Log it. Do not discard. Run `validate_chunk()` — if invalid, fall back to holding position. Report the failure rate as a metric.

**Q: What if my SL solver gives worse results than OSQP?**
A: That is a valid research finding. Log the cost ratio for each QP. Do not hide it. Document it in the thesis as "SL solver approximation quality vs OSQP on the 4-DOF xArm QP problem."

**Q: What if the success rate on the lift task is 0%?**
A: That is possible and should be reported. Likely causes: (1) SmolVLA not fine-tuned on xArm, (2) MuJoCo physics gap vs real robot, (3) IK/tracking failure. Document the cause analytically.

**Q: Do I need to fine-tune SmolVLA?**
A: No for now. Evaluate the pretrained model honestly. Fine-tuning is Phase 5 (post-thesis).

**Q: Where does the event camera data go?**
A: Into the MultimodalFusionEncoder (event_enc branch). It does NOT go into SmolVLA directly — SmolVLA only sees RGB. Events augment the fusion encoder that could later replace/augment SmolVLA's perception.

---

## APPENDIX B: Hardware Speed Note (LTSpice)

Your SL-MPC runs in Python simulation at ~100 Hz. On laptop, this is software-simulated.

The analog hardware argument for your thesis:
- Stuart-Landau oscillators in analog CMOS: convergence in ~microseconds
- Python simulation: ~10ms per solve (100 Hz ceiling)
- LTSpice simulation of the oscillator network: can model analog speeds (~ns to µs)
- **This is a thesis-level contribution**: showing that the solver architecture, if implemented in analog hardware, would achieve 10,000× speedup — enabling kHz-range MPC for robotic arms

For the thesis: (1) show the software system works, (2) run LTSpice circuit simulation of the SL oscillator network, (3) show convergence time in LTSpice vs Python, (4) extrapolate to full MPC loop performance. This is the "hardware neuroscience" angle that distinguishes your work.

---

*Document version: 2.0 — March 2026. Single source of truth for coding agent. Do not modify this document mid-phase. If changes needed, create a new versioned document.*
