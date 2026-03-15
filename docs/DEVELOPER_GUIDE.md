# Neuromorphic Robot Control - Comprehensive Developer Guide

**Document Version:** 2.0  
**Last Updated:** March 15, 2026  
**Target Audience:** Software developers planning next development phases  
**Status:** Phase 13 Complete (Real Sensor Fusion Ablation)

---

## TABLE OF CONTENTS

1. [System Overview](#1-system-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Complete Directory Structure](#3-complete-directory-structure)
4. [Core Components Deep-Dive](#4-core-components-deep-dive)
5. [Module Interaction Patterns](#5-module-interaction-patterns)
6. [Data Flow Diagrams](#6-data-flow-diagrams)
7. [Configuration System](#7-configuration-system)
8. [Running & Testing](#8-running--testing)
9. [Current Phase Status](#9-current-phase-status)
10. [Development Workflow](#10-development-workflow)

---

## 1. SYSTEM OVERVIEW

### Project Mission
Build a **dual-system neuromorphic robot controller** that combines:
- **System 1 (Execution)**: Stuart-Landau MPC controller running at 100-500 Hz for reactive torque control
- **System 2 (Planning)**: SmolVLA (Vision Language Action) model at 5 Hz for task-level decision making
- **Integration Layer**: Real-time sensor fusion from RGB, event cameras, LiDAR, and proprioception

### Key Features
✅ Real dataset validation (lerobot/utokyo_xarm_pick_and_place)  
✅ MuJoCo 3D arm simulation with physics  
✅ Multimodal sensor fusion with 5 operational modes  
✅ Async VLA integration with timeout handling  
✅ 100+ passing tests across all phases  
✅ Real-time benchmarking and metrics collection  

### Technical Stack
- **Language**: Python 3.10+
- **Simulation**: MuJoCo 3.x
- **ML Framework**: PyTorch, Transformers (HuggingFace)
- **VLA Model**: SmolVLA (lerobot/smolvla_base)
- **Dataset**: LeRobot (utokyo_xarm_pick_and_place)
- **Testing**: pytest, integration tests
- **Server**: FastAPI + uvicorn + asyncio
- **Optimization**: OSQP, scipy.integrate.solve_ivp

---

## 2. HIGH-LEVEL ARCHITECTURE

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUAL-SYSTEM CONTROLLER                       │
│  (DualSystemController: 100-500 Hz synchronous + 5 Hz async)   │
└──────────────────┬──────────────────────────────────────────────┘
                   │
        ┌──────────┴──────────┐
        │                     │
   ┌────▼─────────┐    ┌─────▼──────────┐
   │  SYSTEM 1    │    │   SYSTEM 2     │
   │  (Execution) │    │  (Planning)    │
   │  100-500 Hz  │    │     5 Hz       │
   └────┬─────────┘    └────┬───────────┘
        │                   │
   ┌────▼──────────────────▼┐
   │   STATE & OBSERVATIONS │
   │  Current: q, q̇, RGB   │
   │  Buffer: Reference τ   │
   └────┬────────────────────┘
        │
   ┌────▼─────────────────────────────┐
   │  SENSOR FUSION (Real Features)   │
   │  ├─ RGB Encoder (128-dim)        │
   │  ├─ Event Encoder (96-dim)       │
   │  ├─ LiDAR Encoder (64-dim)       │
   │  ├─ Proprioception (32-dim)      │
   │  └─ 5 Fusion Modes (M0-M4)       │
   └────┬──────────────────────────────┘
        │
        │     ┌──────────────────┐
        └────►│  MuJoCo Physics  │
              │  6-DOF xArm Sim  │
              │  + Cameras       │
              │  + Sensors       │
              └──────────────────┘

HORIZONTAL FLOW (Per Control Step):
Step 1: Read state (q, q̇) from simulator
Step 2: Render RGB image via MuJoCo camera
Step 3: Fuse multi-modal sensors (real feature extraction)
Step 4: ASYNC: Query SmolVLA for next action chunk (non-blocking)
Step 5: SYNC: Run SL-MPC solver on current reference trajectory
Step 6: Output optimal torque τ* to simulator
Step 7: Advance simulator by dt=10ms → repeat
```

### System 1: Stuart-Landau MPC Controller

**Purpose**: Fast, physics-aware trajectory tracking at control rate  
**Rate**: 100-500 Hz (typically 100 Hz = 10ms per step)  
**Input**: Current state (q, q̇), reference trajectory τ_ref  
**Output**: Optimal torque τ* ∈ [-20, 20] Nm  

**What it does**:
```python
# Simplified MPC loop (actual code in src/mpc/xarm_controller.py)
for step in range(horizon):
    q_ref = reference_trajectory[step]               # From VLA chunk
    q_error = q_ref - q_current
    stability_torque = compute_dynamics(q, qdot)     # M(q)qdotdot + C(q,qdot)
    tracking_torque = K_p * q_error + K_d * qdot_error
    tau_optimal = solve_qp_with_sl_solver(
        objective=||τ - τ_ref||²,
        constraints=[joint limits, torque limits]
    )
    return tau_optimal
```

**Solver Details**:
- Uses Stuart-Landau oscillators for continuous optimization
- Implements direct Lagrange multipliers (Arrow-Hurwicz algorithm)
- Convergence time: 2-5 seconds of solver time
- Handles nonlinear dynamics M(q), C(q,qdot), G(q)

### System 2: SmolVLA Task Planner

**Purpose**: High-level task understanding and action chunk planning  
**Rate**: 5 Hz (200ms between queries, async)  
**Input**: RGB image, robot state, language instruction  
**Output**: Action chunk [10 steps × 6-DOF + gripper]  

**What it does**:
```python
# Simplified VLA loop (actual code in src/smolvla/real_client.py)
async def vla_loop():
    while True:
        rgb_frame = current_observation.rgb
        state_vector = current_observation.state[0:6]  # 6 arm joints
        instruction = "pick up the object"
        
        # Non-blocking HTTP query (timeout 2s)
        action_chunk = await vla_client.predict(
            rgb_image=rgb_frame,
            state=state_vector,
            instruction=instruction
        )
        
        # Store in buffer for MPC to track
        trajectory_buffer.update(action_chunk)
        
        # Wait 200ms for next VLA query
        await asyncio.sleep(0.2)
```

**Model Details**:
- Base: lerobot/smolvla_base (huggingface)
- Output: 7-dimensional actions (6-DOF + gripper) per step × 10 steps
- Training data: utokyo_xarm pick-and-place dataset
- Does NOT involve training in current implementation (pre-trained only)

### Integration: Dual-System Sync

**Critical Design**: System 1 and System 2 run concurrently WITHOUT blocking each other.

```
Timeline (10 steps at 100 Hz):
[Step 0]  t=0ms    VLA QUERY STARTS (async)
          t=0ms    MPC Step 0 (q=0.5, ref from prev chunk)  → τ₀
          t=10ms   MPC Step 1 (q=0.6, same ref)            → τ₁
          ...
          t=50ms   VLA QUERY COMPLETES (200ms round-trip)
                   → New reference trajectory available
          t=90ms   MPC Step 9 (q=1.2, tracking new ref)    → τ₉
[Step 10] t=100ms  Next VLA QUERY STARTS
          t=100ms  MPC Step 10 (q=1.3)                     → τ₁₀
```

**Synchronization Primitive**: `TrajectoryBuffer`
- Thread-safe (GIL-based atomicity)
- MPC reads: ref_traj[t:t+N] (no wait)
- VLA writes: ref_traj[:] ← new_action_chunk (non-blocking)
- Guarantees fresh reference within 200ms

---

## 3. COMPLETE DIRECTORY STRUCTURE

### Root Level
```
neuromorphic-robot/
├── README.md                           # Quick start guide
├── REPO_STRUCTURE.md                   # File tree overview
├── requirements.txt                    # Python dependencies (pinned versions)
├── pytest.ini                          # Test configuration
├── pyproject.toml                      # Project metadata
├── .venv/                              # Python virtual environment
└── .gitignore                          # Git ignore patterns
```

### /config - Configuration Files
```
config/
├── config.yaml                         # Main config (solver params, MPC, robot)
├── logging.yaml                        # Logging configuration
├── robots/
│   ├── xarm_6dof.yaml                  # xArm 6-DOF parameters (joint limits, etc.)
│   └── xarm_4dof.yaml                  # Legacy 4-DOF config (deprecated)
└── solvers/
    ├── sl_neuromorphic.yaml             # Stuart-Landau solver params
    └── osqp.yaml                        # OSQP QP solver params
```

### /src - Main Source Code (400+ files)

#### src/main.py
Entry point for standalone simulation testing. Minimal usage in current setup (most work via scripts/).

#### src/core/ - Core Abstractions
```
core/
├── __init__.py
├── base_controller.py                  # Abstract base for all controllers
└── base_env.py                         # Abstract base for environments
```
**Serves**: Blueprint for extending controller types  
**Used By**: All concrete controller implementations

#### src/dynamics/ - Robot Dynamics Models
```
dynamics/
├── __init__.py
├── arm2dof.py                          # 2-DOF planar arm (legacy, Phases 1-3)
├── kinematics_3dof.py                  # 3-DOF spatial arm kinematics
│   ├── forward_kinematics(q) → (p, R)   # End-effector pose
│   ├── jacobian(q) → J                  # Velocity mapping
│   └── inverse_kinematics(p) → q        # Position to angles
├── lagrangian_3dof.py                  # 3-DOF dynamics (M, C, G matrices)
└── xarm_dynamics.py                    # 6-DOF xArm dynamics (if exists)
```

**Key Classes**:
- `Arm3DOF`: Planar 3-joint arm with RRR configuration
  - L0=0.1m (base), L1=0.25m, L2=0.2m
  - Max reach: 0.45m
  - Joint limits: q ∈ [-π, π] × [-π/2, π/2] × [-2π/3, 2π/3]

- `Arm2DOF`: Planar 2-joint arm (legacy, Phases 1-3 only)

**Usage**:
```python
from src.dynamics.kinematics_3dof import Arm3DOF
arm = Arm3DOF(L0=0.10, L1=0.25, L2=0.20)
p, R = arm.forward_kinematics(q=np.array([0.5, 0.2, 0.1]))
J = arm.jacobian(q)
```

#### src/environments/ - Environment Wrappers
```
environments/
├── __init__.py
└── mujoco_3dof_env.py                  # MuJoCo 3-DOF arm environment wrapper
```

**Interface**: Gym-like environment
- `reset()`: Reset to initial state
- `step(action)`: Apply action, get (obs, reward, done)
- `render()`: Render via MuJoCo viewer

#### src/mpc/ - Model Predictive Control
```
mpc/
├── __init__.py
├── xarm_controller.py                  # Main: 6-DOF xArm MPC controller (100+ lines)
│   ├── XArmMPCController.__init__()
│   ├── compute_inertia_matrix(q)       # M(q) ∈ ℝ⁸ˣ⁸
│   ├── compute_coriolis_torque(q, qdot) # C(q,qdot) ∈ ℝ⁸
│   ├── compute_gravity_torque(q)       # G(q) ∈ ℝ⁸
│   ├── solve_mpc(state, ref_traj)      # τ* = argmin ||τ - τ_ref||²
│   └── linearize_dynamics()            # A, B matrices for QP
├── linearize_3dof.py                   # Linearization for 3-DOF (legacyPhases 2-4)
├── qp_builder_3dof.py                  # QP construction for 3-DOF (legacy)
└── sl_solver.py                        # Stuart-Landau QP solver interface
```

**Key Algorithm**: MPC formulation
```
minimize:    Σ ||τ[k] - τ_ref[k]||²_Q  +  ||qdot_error[k]||²_R
subject to:  q_min ≤ q[k] ≤ q_max  ∀k ∈ [0, N)
             |qdot[k]| ≤ qdot_max  ∀k
             |τ[k]| ≤ τ_max  ∀k
             M(q)τ + C(q,qdot) = a_target (dynamics constraint)
```

#### src/solver/ - Optimization Solvers
```
solver/
├── __init__.py
├── stuart_landau_lagrange_direct.py    # MAIN: SL+Direct Lagrange solver (200+ lines)
│   ├── StuartLandauLagrangeDirect.__init__()
│   ├── solve(P, q, C, d, A_ineq, l, u) → (x*, λ_eq, λ_ineq)
│   ├── _ode_dynamics(t, state, constraints)  # Arrow-Hurwicz ODEs
│   └── convergence_check()
├── osqp_solver.py                      # OSQP wrapper (baseline QP solver)
├── stuart_landau_lagonn.py             # SL with Lagrange penalty encoding (legacy)
├── stuart_landau_3dof.py               # 3-DOF specific solver (legacy, Phases 2-3)
└── adaptive_mpc_controller.py           # Adaptive gain scheduling (deprecated)
```

**Stuart-Landau Solver Details**:
```python
# Continuous-time ODEs (Arrow-Hurwicz saddle-point algorithm):
dx/dt = (μ - x²)x - Px - q - C^T λ^eq - A_ineq^T (λ^up - λ^lo)
dλ_eq/dt = (Cx - d) / τ_lam_eq  [unbounded multipliers]
dλ_up/dt = max(0, (A_ineq x - u)) / τ_lam_ineq  [one-sided]
dλ_lo/dt = max(0, (l - A_ineq x)) / τ_lam_ineq  [one-sided]

Solve via scipy.integrate.solve_ivp (RK45) for T_solve seconds
Parameters:
  τ_x = 1.0             (decision variable time constant)
  τ_lam_eq = 0.1        (fast equality multiplier tracking)
  τ_lam_ineq = 0.5      (fast inequality multiplier tracking)
  T_solve = 2.0 s       (total solve time)
  convergence_tol = 1e-5
```

**Why Stuart-Landau?**
- Neuromorphic: Oscillator-based (biologically inspired)
- Robust: No matrix inversions (opposed to Newton methods)
- Flexible: Handles nonlinear constraints via continuous dynamics
- Convergent: Arrow-Hurwicz theory guarantees convergence

#### src/simulation/ - MuJoCo Physics Engine Integration
```
simulation/
├── __init__.py
├── envs/
│   ├── __init__.py
│   └── xarm_env.py                     # MAIN: XArmEnv wrapper (150+ lines)
│       ├── XArmEnv.__init__()          # MuJoCo model loading
│       ├── reset()
│       ├── step(tau)                   # Apply torque, integrate, return obs
│       ├── get_joint_pos()             # q ∈ ℝ⁶ (arm only)
│       ├── get_joint_vel()             # q̇ ∈ ℝ⁶
│       ├── render()
│       └── close()
├── models/
│   ├── xarm_6dof.xml                   # MJCF xArm model (6 arm + 2 gripper)
│   ├── xarm_4dof.xml                   # Legacy 4-DOF model (deprecated)
│   └── arm2dof.xml                     # Legacy 2-DOF planar model
├── cameras/
│   ├── __init__.py
│   └── event_camera_simple.py           # Event camera simulator (voxel grid)
│       ├── EventCameraSimulator
│       │   ├── process_frame(rgb) → events  [T, H, W]
│       │   └── frames_to_events(frames) → events
│       └── LiDARSimulator
│           ├── query_ranges(distances) → features
│           └── get_point_cloud() → xyz
└── tests/
    ├── test_env_loads.py               # Verify MuJoCo loads
    ├── test_camera_renders.py          # Verify cameras work
    └── test_sensor_outputs.py          # Verify sensor modalities
```

**XArmEnv Class**:
```python
# Usage example:
env = XArmEnv(model_path="assets/xarm_6dof.xml")
obs = env.reset()  # → {'q': array(6,), 'qdot': array(6,), 'rgb': array(H,W,3)}

for step in range(100):
    tau = np.array([1.0, 0.5, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0])  # 6-DOF + 2-gripper
    obs, reward, done, info = env.step(tau)
    if done:
        break

env.render()  # Display MuJoCo viewer
env.close()
```

**MJCF Model (xarm_6dof.xml)**:
- 6 revolute joints (arm main joints)
- 2 prismatic joints (parallel gripper fingers)
- Inertial properties matched to real xArm specs
- Contact geometry for gripper + objects
- 2-3 fixed cameras for RGB observation

#### src/smolvla/ - Vision Language Action Integration
```
smolvla/
├── __init__.py
├── real_client.py                      # MAIN: Async HTTP client (150+ lines)
│   ├── RealSmolVLAClient.__init__()    # Initialize with server URL
│   ├── predict(rgb, state, instruction) → action  [7,]
│   ├── _query_with_retries(payload)    # Exponential backoff
│   ├── health_check()                  # Verify server alive
│   ├── parse_response()                # Extract action from JSON
│   └── Stats: call_count, success_count, latency_ms
├── action_processor.py                 # Convert action chunk → joint reference
│   ├── ActionChunkProcessor
│   │   ├── process_chunk(action_chunk) → ref_traj [N, 6]
│   │   └── apply_ik(ee_delta) → joint_delta
│   └── Converts 7-D actions → 6-D joint space
└── vla_production_server.py             # Production VLA server (250+ lines)
    ├── load_model_on_startup()         # Cache model in memory
    ├── /health  GET endpoint
    │   └── Returns: {status, model_id, device, ready}
    ├── /predict POST endpoint
    │   ├── Input: {rgb_b64, state, instruction}
    │   ├── Output: {action, action_std, latency_ms, success}
    │   ├── Timeout protection: 10s max inference
    │   ├── Memory mgmt: cleanup_resources() every N requests
    │   └── Error handling: 504 on timeout, automatic recovery
    └── Global state: model, device, tokenizer (cached)
```

**API Contract** (VLA Server):
```
POST http://localhost:8000/predict
Content-Type: application/json

Request {
    "rgb_image_b64": "base64_jpeg_string",
    "state": [q1, q2, q3, q4, q5, q6],      # Optional, robot state
    "instruction": "pick up the object"     # Optional, task text
}

Response {
    "action": [a1, a2, a3, a4, a5, a6, a7], # 7-D action (6-DOF + gripper)
    "action_std": [s1, s2, ...],             # Action uncertainty
    "latency_ms": 45.2,                      # Server inference time
    "success": true
}
```

**Latency Profile**:
- Cold start (model load): 200-300s
- Warm inference: 20-80ms per query
- Network overhead (HTTP): 10-20ms
- Total end-to-end: 30-100ms including network

#### src/fusion/ - Multimodal Sensor Fusion
```
fusion/
├── __init__.py
├── encoders/
│   ├── __init__.py
│   └── real_fusion_simple.py            # MAIN: Real feature extraction (200+ lines)
│       ├── RealFusionEncoder class
│       ├── extract_rgb_features()       # 128-dim: color + gradients
│       ├── extract_event_features()     # 96-dim: temporal diff
│       ├── extract_lidar_features()     # 64-dim: spatial geometry
│       ├── extract_proprio_features()   # 32-dim: joint state
│       ├── encode(obs) → 256-dim fusion embedding
│       └── 5 Factory Methods
│           ├── rgb_only()              # M0: RGB baseline
│           ├── rgb_events()            # M1: RGB + events
│           ├── rgb_lidar()             # M2: RGB + geometry
│           ├── rgb_proprio()           # M3: RGB + state
│           └── full_fusion()           # M4: All modalities
└── sensors/
    ├── __init__.py
    └── (Reserved for future sensor abstractions)
```

**Fusion Modes**:
| Mode | Modalities | Dim | Purpose |
|------|-----------|-----|---------|
| M0 | RGB only | 128 | Baseline (visual only) |
| M1 | RGB + Events | 224 | Add temporal dynamics |
| M2 | RGB + LiDAR | 192 | Add spatial geometry |
| M3 | RGB + Proprio | 160 | Add robot state |
| M4 | RGB + Events + LiDAR + Proprio | 320 | Full fusion |

**Feature Extraction Details**:
```
RGB (128-dim):
  - Downsampled spatial: 64-dim (8×8 grid of color)
  - Color statistics: 32-dim (mean, std per channel)
  - Gradient patterns: 32-dim (edge maps, orientation)

Events (96-dim):
  - Frame difference optical flow: 96-dim (motion patterns)
  - Temporal voxel grid: [5 bins × 3×3 spatial] = 45-dim
  - Histogram: 51-dim

LiDAR (64-dim):
  - Point cloud features: 64-dim (corner detection, depth)
  - Extracted from image gradients (simulated LiDAR)

Proprioception (32-dim):
  - Joint angles: 6-dim (normalized to [-1, 1])
  - Joint velocities: 6-dim
  - Gripper state: 2-dim
  - Padding: 18-dim

Fusion Output:
  - Concatenate all enabled modalities
  - Normalize per-dimension
  - Final output: 256-dim embedding
```

#### src/integration/ - System Integration
```
integration/
├── __init__.py
├── dual_system_controller.py            # MAIN: Orchestrator (200+ lines)
│   ├── DualSystemController class
│   ├── step(q, qdot, rgb, instruction) → τ  [CRITICAL: < 20ms]
│   ├── update_trajectory_buffer()       # Non-blocking update
│   ├── vla_query_thread()               # Async VLA loop
│   ├── state_machine: INIT → TRACKING → GOAL_REACHED → ERROR
│   ├── Timing instrumentation
│   └── Safety checks
├── smolvla_server_client.py             # (Legacy wrapper, use real_client.py)
└── vla_query_thread.py                  # (Legacy thread impl, use asyncio)
```

**DualSystemController.step() Pseudocode**:
```python
def step(q, qdot, rgb, instruction) -> tau:
    # TIMING: Must return within 20ms (typical = 5-10ms)
    
    # 1. Update state (5ms)
    self.q_current = q.copy()
    self.qdot_current = qdot.copy()
    
    # 2. State machine transition (1ms)
    if self.state == INIT:
        self.state = TRACKING
    check_goal_arrival(q)
    
    # 3. Get reference trajectory from buffer (5ms)
    # Non-blocking read! If VLA hasn't queried yet, use default
    ref_trajectory = self.trajectory_buffer.get_reference()
    
    # 4. Solve MPC (8-10ms) VIA SYSTEM 1
    # Synchronous, guaranteed to complete
    tau_optimal = self.mpc_solver.solve(
        state=(q, qdot),
        reference_trajectory=ref_trajectory
    )
    
    # 5. Return torque (1ms)
    return tau_optimal
    
    # NOTE: VLA queries happen in SEPARATE async thread
    # They update trajectory_buffer without blocking main loop
```

**State Machine**:
```
INIT ──(first step)──> TRACKING ──(goal reached)──> GOAL_REACHED
      └─(error)──────────┬────────────────────────────┴──> ERROR
```

#### src/system/ - System Utilities
```
system/
├── __init__.py
├── controller.py                       # (Alias for dual_system_controller)
└── trajectory_buffer.py                # MAIN: Thread-safe trajectory buffer
    ├── TrajectoryBuffer class
    ├── update(action_chunk)            # Async writer (from VLA thread)
    ├── get_reference(t)                # Sync reader (from MPC)
    ├── Thread safety: GIL-based atomicity
    └── Provides: ref_trajectory [N, 6]
```

**TrajectoryBuffer Interface**:
```python
class TrajectoryBuffer:
    def __init__(self, horizon=10):
        self.horizon = horizon
        self.ref_trajectory = np.zeros((horizon, 6))  # Default: zero refs
    
    def update(self, action_chunk):
        """
        Called by VLA thread (async). Non-blocking.
        action_chunk: [10, 7] numpy array (10 steps × 6-DOF + gripper)
        """
        self.ref_trajectory = action_chunk[:10, :6]  # Extract arm joints
    
    def get_reference(self, t=0):
        """
        Called by MPC (sync). Non-blocking.
        Returns: [N, 6] reference trajectory for steps [t, t+N)
        """
        return self.ref_trajectory.copy()
```

#### src/robot/ - Robot Configuration
```
robot/
├── __init__.py
└── robot_config.py                     # xArm configuration loader
    ├── XArmConfig class
    ├── load_from_yaml()
    ├── Parameters: joint limits, mass, inertia
    └── Used by: dynamics models, MPC controllers
```

#### src/visualization/ - Plotting and Visualization
```
visualization/
├── __init__.py
├── visualizer.py                       # Matplotlib trajectory plots
│   ├── plot_trajectory_2d()
│   ├── plot_joint_angles()
│   └── plot_torques()
└── (Additional visualization utilities)
```

#### src/utils/ - Utility Modules
```
utils/
├── __init__.py
├── config.py                           # Load YAML configs
├── logger.py                           # Logging setup
├── data_utils.py                       # Data loading utilities
├── visualization.py                    # Plotting helpers
├── results_analyzer.py                 # Metrics analysis
├── report_generator.py                 # Auto-generate thesis figures
├── qp_inspector.py                     # Debug QP matrices
└── data_collector.py                   # Collect trajectories for analysis
```

### /tests - Integration Tests (100+ files)
```
tests/
├── __init__.py
├── test_phase0_health.py               # System initialization checks
├── test_xarm_env.py                    # 13 tests for simulation env
├── test_mpc_gate2.py                   # MPC solver validation (9 tests)
├── test_phase3_kkt.py                  # KKT conditions verification
├── test_phase5_integration.py           # Full system integration (8 tests)
├── test_encoder_simple.py              # Real fusion encoder (5 tests)
├── test_sl_gate3.py                    # SL solver gate (9 tests)
├── test_integration_e2e.py             # End-to-end benchmark
├── test_real_smolvla_server.py         # VLA server integration
├── test_state_machine_transitions.py   # Controller state machine
├── test_benchmark_suite.py             # Benchmark runner tests
├── test_lsmo_dataset.py                # (Legacy LSMO dataset tests)
├── test_comprehensive_eval.py          # Full evaluation suite
└── ... (50+ more test files for specific components)
```

**Test Categories**:
- **Unit Tests** (4 tests): Individual component validation
- **Integration Tests** (20+ tests): Component interaction
- **Gate Tests** (30+ tests): Phase validation gates
- **Benchmark Tests** (15+ tests): Performance benchmarks
- **E2E Tests** (20+ tests): End-to-end system validation

**Key Test Commands**:
```bash
pytest tests/test_xarm_env.py -v              # XArm environment (13 tests)
pytest tests/test_mpc_gate2.py -v             # MPC solver (9 tests)
pytest tests/test_phase5_integration.py -v    # Full integration (8 tests)
pytest tests/ -v --tb=short                   # All tests (100+ total)
```

### /evaluation - Benchmarking Results
```
evaluation/
├── __init__.py
├── benchmarks/
│   ├── __init__.py
│   ├── phase12_quick_test.py             # Quick 3-episode test
│   ├── phase13_final.py                  # 30-episode ablation study
│   ├── phase13_quick_ablation.py         # 3-episode per mode test
│   ├── run_dataset_replay.py             # Replay real dataset
│   ├── run_mpc_solo.py                   # MPC-only baseline
│   ├── run_smolvla_only.py               # VLA-only baseline
│   └── run_full_system.py                # Full System 1+2
└── results/
    ├── phase12_results_*.json             # Phase 12 benchmark outputs
    ├── phase13_ablation_*.json            # Phase 13 fusion ablation results
    ├── B1_mpc_*.json                      # Benchmark B1 results
    ├── B2_vla_*.json                      # Benchmark B2 results
    ├── B3_dual_*.json                     # Benchmark B3 results (dual system)
    ├── B4_real_*.json                     # Benchmark B4 results (real data)
    └── B5_sensor_*.json                   # Benchmark B5 results (sensor ablation)
```

### /scripts - Standalone Executable Scripts
```
scripts/
├── README.md                             # Script documentation
├── phase12_quick_test.py                 # Quick VLA benchmark (3 episodes)
├── phase13_final.py                      # Real sensor fusion ablation (30 episodes)
├── check_dataset_cache.py                # Verify dataset access
├── health_check_vla.py                   # VLA server health check
├── inspect_dataset.py                    # Analyze dataset statistics
├── test_vla_server.py                    # VLA server integration test
└── (Additional analysis and debug scripts)
```

### /notebooks - Jupyter Notebooks
```
notebooks/
├── 00_dataset_audit.ipynb                # Real dataset exploration
├── 01_env_validation.ipynb               # MuJoCo environment test
├── 02_sensor_validation.ipynb            # Sensor modality verification
├── 03_mpc_validation.ipynb               # MPC control law testing
├── 04_smolvla_validation.ipynb           # VLA model inference
└── 05_full_system.ipynb                  # End-to-end system test
```

### /data - Dataset Storage
```
data/
├── cache/
│   └── lerobot/                          # LeRobot dataset cache
│       └── utokyo_xarm_pick_and_place/   # Downloaded: 102 episodes
│           ├── episodes/                  # Episode data (parquet)
│           ├── videos/                    # MP4 videos
│           └── metadata.json
├── loaders/
│   ├── lerobot_loader.py                 # LeRobot dataset interface
│   ├── episode_player.py                 # Replay episodes in MuJoCo
│   └── data_inspector.py                 # Dataset statistics
├── download/
│   ├── download_dataset.py               # Download script
│   └── verify_dataset.py                 # Integrity check
└── dataset_001/                          # (Legacy synthetic dataset)
```

### /assets - CAD Models
```
assets/
├── xarm_6dof.xml                         # xArm 6-DOF MJCF  model (ACTIVE)
├── xarm_4dof.xml                         # xArm 4-DOF MJCF (deprecated)
├── arm2dof.xml                           # 2-DOF planar arm (legacy)
├── arm3dof.xml                           # 3-DOF planar arm (legacy)
└── (Gripper meshes, object models)
```

### /docs - Documentation
```
docs/
├── INDEX.md                              # Documentation index
├── 01-QUICKSTART.md                      # 5-minute setup
├── 02-GETTING_STARTED.md                 # Installation + usage
├── 03-SOLVERS.md                         # Solver deep-dive
├── 04-TESTING.md                         # How to run tests
├── 05-VISUALIZATION.md                   # MPC visualization
├── 06-BENCHMARKING.md                    # Benchmark methodology
├── 07-THEORY.md                          # Theory + math
├── ROADMAP.md                            # Implementation phases
├── PROJECT_STATUS.txt                    # Current status
├── DEVELOPER_GUIDE.md                    # ← THIS FILE
├── sensor_fusion_vla_mpc_techspec_v2.md  # Complete tech spec
├── agent/
│   ├── AGENT_STATE.md                    # AI agent working state
│   ├── TODO.md                           # Current task list
│   └── PROGRESS.md                       # Completed task log
└── archived/
    ├── (Prior phase documentation)
    └── (Legacy implementation docs)
```

### /logs - Log Files
```
logs/
├── phase12_benchmark.log                 # Phase 12 benchmark execution log
├── phase13_ablation_real.log             # Phase 13 fusion ablation log
├── vla_server.log                        # VLA server debug output
├── archived/
│   └── (Old logs from prior phases)
└── .gitkeep
```

---

## 4. CORE COMPONENTS DEEP-DIVE

### 4.1 MuJoCo Simulation Environment (XArmEnv)

**File**: [src/simulation/envs/xarm_env.py](src/simulation/envs/xarm_env.py)  
**Lines**: 150+

**Class Hierarchy**:
```python
XArmEnv
├── Attributes
│   ├── model: mujoco.MjModel          # Physics model
│   ├── data: mujoco.MjData            # Physics state
│   ├── renderer: GLFWRenderer          # Visualization
│   ├── q: ndarray [6]                 # Joint angles (arm only)
│   ├── qdot: ndarray [6]              # Joint velocities (arm only)
│   ├── tau: ndarray [8]               # Torque command (6 arm + 2 gripper)
│   ├── cameras: dict                  # Renderers for RGB, events, LiDAR
│   └── step_count: int
└── Methods
    ├── __init__(model_path, render_mode)
    ├── reset() → obs                  # Reset to initial state
    ├── step(tau: ndarray[8]) → (obs, reward, done, info)
    ├── get_joint_pos() → ndarray[6]   # Query q
    ├── get_joint_vel() → ndarray[6]   # Query q̇
    ├── render()                        # Show viewer
    └── close()                         # Cleanup
```

**Key Methods**:

```python
def reset(self) -> Dict:
    """Reset simulation to initial state."""
    # Implementation:
    # 1. Set all joint positions to 0
    # 2. Set all joint velocities to 0
    # 3. Reset simulation time to 0
    # 4. Return initial observation
    
    return {
        'q': self.data.qpos[0:6].copy(),        # Joint angles
        'qdot': self.data.qvel[0:6].copy(),     # Joint velocities
        'rgb': self._render_rgb(),               # Camera observation
    }

def step(self, tau: np.ndarray[8]) -> Tuple:
    """
    Apply torque and step simulation.
    
    Args:
        tau: [8] torque command (6 arm + 2 gripper)
    
    Returns:
        obs: Observation dict
        reward: Scalar reward (0 for now)
        done: Episode termination flag
        info: Debug info
    """
    # Implementation:
    # 1. Clamp tau to joint limits
    tau_clamped = np.clip(tau, self.tau_min, self.tau_max)
    
    # 2. Apply torque to actuators
    self.data.ctrl[:] = tau_clamped
    
    # 3. Step physics simulation
    mujoco.mj_step(self.model, self.data)  # Default dt=0.001
    
    # 4. Read state from simulation
    q_new = self.data.qpos[0:6].copy()
    qdot_new = self.data.qvel[0:6].copy()
    
    # 5. Render observation
    rgb = self._render_rgb()
    
    # 6. Check termination (time limit, collision, etc.)
    done = (self.step_count > 500) or self._check_collision()
    
    # 7. Compute reward (currently zero)
    reward = 0.0
    
    # 8. Increment step count
    self.step_count += 1
    
    return (
        {'q': q_new, 'qdot': qdot_new, 'rgb': rgb},
        reward,
        done,
        {'step': self.step_count}
    )
```

**Physical Model (xarm_6dof.xml)**:
```xml
<!-- Simplified structure -->
<mujoco>
  <worldbody>
    <body name="base" pos="0 0 0">
      <!-- 6 revolute arm joints -->
      <joint name="joint1" axis="0 0 1" type="hinge"/>
      <inertial mass="2.0" diaginv="0.01 0.01 0.01"/>
      
      <body name="link1">
        <joint name="joint2" axis="0 1 0" type="hinge"/>
        <!-- ... more links -->
      </body>
    </body>
    
    <!-- Gripper (2 prismatic joints for parallel fingers) -->
    <body name="gripper_left">
      <joint name="gripper_left" axis="1 0 0" type="slide"/>
    </body>
    <body name="gripper_right">
      <joint name="gripper_right" axis="1 0 0" type="slide"/>
    </body>
    
    <!-- Cameras (mounted on end-effector + fixed) -->
    <camera name="camera_ee" pos="0 0 0.1"/>
    <camera name="camera_base" pos="1 1 1"/>
  </worldbody>
  
  <actuator>
    <!-- Torque actuators for 8 DOF -->
    <motor joint="joint1" ctrlrange="-20 20"/>
    <!-- ... 7 more motors -->
  </actuator>
</mujoco>
```

**Integration Points**:
- Used by: `DualSystemController.step()` (via `env.step(tau)`)
- Provides: Observations (q, qdot, rgb) for next control loop iteration
- Interfaces with: Sensor fusion (cameras), MPC solver input

### 4.2 Stuart-Landau MPC Solver

**File**: [src/solver/stuart_landau_lagrange_direct.py](src/solver/stuart_landau_lagrange_direct.py)  
**Lines**: 200+

**Mathematical Formulation**:

**Continuous-time ODE system**:
```
dx/dt = (1/τ_x) · (μx - x³ - Px - q - C^T λ_eq - A_ineq^T (λ_up - λ_lo))
dλ_eq/dt = (1/τ_λ_eq) · (Cx - d)
dλ_up/dt = (1/τ_λ_ineq) · max(0, A_ineq x - u)
dλ_lo/dt = (1/τ_λ_ineq) · max(0, l - A_ineq x)
```

**Interpretation**:
- `(μx - x³)x`: Stuart-Landau oscillator (restoring force)
- `Px + q`: Quadratic cost gradient
- `C^T λ_eq`: Equality constraint forces
- `A_ineq^T (λ_up - λ_lo)`: Inequality constraint forces

**Arrow-Hurwicz Saddle-Point Algorithm**:
This is a **dual-ascent/primal-descent** algorithm:
- Decision variable `x` descends cost & constraint gradients
- Lagrange multipliers `λ` ascend to enforce constraints
- Convergence: O(1/T) for convex problems

**Solver Parameters**:
```python
class StuartLandauLagrangeDirect:
    def __init__(
        self,
        tau_x: float = 1.0,              # Time constant for x (slower = more stable)
        tau_lam_eq: float = 0.1,         # Time constant for λ_eq (fast = quick constraint satisfaction)
        tau_lam_ineq: float = 0.5,       # Time constant for λ_ineq (intermediate)
        mu_x: float = 0.0,               # Bifurcation parameter (0 = pure gradient flow)
        T_solve: float = 2.0,            # Total solve time (seconds)
        convergence_tol: float = 1e-5,   # Convergence threshold
        adaptive_annealing: bool = True   # Time-varying τ for better convergence
    ):
```

**Solve Method**:
```python
def solve(self, P, q, C, d, A_ineq, l, u) -> Tuple[np.ndarray, ...]:
    """
    QP Solution:
    minimize:    0.5 x^T P x + q^T x
    subject to:  Cx = d                (equality)
                 l ≤ A_ineq x ≤ u      (box inequality)
    
    Returns:
        x:        Optimal decision variable
        lambda_eq: Equality Lagrange multipliers
        lambda_up: Upper inequality multipliers
        lambda_lo: Lower inequality multipliers
    """
    # 1. Initialize state
    n = P.shape[0]
    x0 = np.zeros(n)
    lambda_eq0 = np.zeros(C.shape[0]) if C is not None else np.array([])
    lambda_up0 = np.zeros(A_ineq.shape[0]) if A_ineq is not None else np.array([])
    lambda_lo0 = np.zeros(A_ineq.shape[0]) if A_ineq is not None else np.array([])
    
    state0 = np.concatenate([x0, lambda_eq0, lambda_up0, lambda_lo0])
    
    # 2. Solve ODE for T_solve seconds
    result = solve_ivp(
        fun=self._ode_dynamics,
        t_span=(0, self.T_solve),
        y0=state0,
        args=(P, q, C, d, A_ineq, l, u),
        method='RK45',
        dense_output=True,
        max_step=0.05  # 50ms max timestep
    )
    
    # 3. Extract final state
    t_final = result.t[-1]
    state_final = result.y[:, -1]
    
    # 4. Unpack solution
    x = state_final[:n]
    lambda_eq = state_final[n:n+C.shape[0]]
    lambda_up = state_final[n+C.shape[0]:]
    lambda_lo = ...
    
    # 5. Log convergence info
    self.last_info = {
        'solve_time': t_final,
        'constraint_violation': np.linalg.norm(C @ x - d),
        'inequality_violation': np.max(np.concatenate([
            np.maximum(0, A_ineq @ x - u),
            np.maximum(0, l - A_ineq @ x)
        ])),
        'multiplier_norms': {
            'lambda_eq': np.linalg.norm(lambda_eq),
            'lambda_up': np.linalg.norm(lambda_up)
        }
    }
    
    return (x, lambda_eq, lambda_up, lambda_lo)
```

**Strengths**:
- ✅ No matrix inversions (robust to ill-conditioning)
- ✅ Continuous-time natural (neuromorphic interpretation)
- ✅ Handles nonlinear constraints naturally
- ✅ Annealing support for better convergence
- ✅ Convergence guarantees from Arrow-Hurwicz theory

**Limitations**:
- ⚠️ Slower than OSQP for small QPs (2-5s vs 10-50ms)
- ⚠️ Requires careful tuning of τ parameters
- ⚠️ No warm-starting from prior solutions

**Usage in MPC**:
```python
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect

solver = StuartLandauLagrangeDirect(
    tau_x=1.0,
    tau_lam_eq=0.1,
    tau_lam_ineq=0.5,
    T_solve=2.0
)

# Inside MPC loop:
tau_opt, lam_eq, lam_up, lam_lo = solver.solve(
    P=Q,                    # ∈ ℝ^(6×6), quadratic term
    q=q_cost,               # ∈ ℝ^6, linear term
    C=C_eq,                 # ∈ ℝ^(m_eq×6), equality matrix
    d=d_eq,                 # ∈ ℝ^m_eq, equality RHS
    A_ineq=A_box,          # ∈ ℝ^(m_ineq×6), box constraints
    l=lb,                   # ∈ ℝ^m_ineq, lower bound
    u=ub                    # ∈ ℝ^m_ineq, upper bound
)
```

### 4.3 Real Sensor Fusion Encoder

**File**: [src/fusion/encoders/real_fusion_simple.py](src/fusion/encoders/real_fusion_simple.py)  
**Lines**: 200+

**Architecture**:
```
Input Observation
    ├─[RGB] 84×84×3 uint8 image
    │   └─ RealFusionEncoder.extract_rgb_features()
    │       ├─ Downsampled RGB: 64-dim
    │       ├─ Color statistics: 32-dim
    │       └─ Gradient patterns: 32-dim
    │           → RGB embedding: 128-dim
    │
    ├─[Events] Temporal frame differences
    │   └─ RealFusionEncoder.extract_event_features()
    │       ├─ Optical flow: 45-dim
    │       ├─ Motion histogram: 51-dim
    │       → Event embedding: 96-dim
    │
    ├─[LiDAR] Spatial geometry from gradients
    │   └─ RealFusionEncoder.extract_lidar_features()
    │       ├─ Corner detection: 32-dim
    │       ├─ Depth features: 32-dim
    │       → LiDAR embedding: 64-dim
    │
    └─[Proprioception] Robot joint state
        └─ RealFusionEncoder.extract_proprio_features()
            ├─ Joint angles: 6-dim
            ├─ Joint velocities: 6-dim
            ├─ Gripper state: 2-dim
            └─ Padding: 18-dim
                → Proprio embedding: 32-dim

Fusion Concatenation (5 modes):
M0 (RGB):            128-dim
M1 (RGB+Events):     128+96 = 224-dim
M2 (RGB+LiDAR):      128+64 = 192-dim
M3 (RGB+Proprio):    128+32 = 160-dim
M4 (Full):           128+96+64+32 = 320-dim

Output: Normalized 256-dim (or mode-specific dim) embedding
```

**Class Methods**:
```python
class RealFusionEncoder:
    @classmethod
    def rgb_only(cls) -> RealFusionEncoder:
        """M0: Visual baseline (RGB only)."""
        return cls(use_rgb=True, use_events=False, use_lidar=False, use_proprio=False)
    
    @classmethod
    def rgb_events(cls) -> RealFusionEncoder:
        """M1: RGB + temporal events (add motion)."""
        return cls(use_rgb=True, use_events=True, use_lidar=False, use_proprio=False)
    
    @classmethod
    def rgb_lidar(cls) -> RealFusionEncoder:
        """M2: RGB + spatial geometry (add depth)."""
        return cls(use_rgb=True, use_events=False, use_lidar=True, use_proprio=False)
    
    @classmethod
    def rgb_proprio(cls) -> RealFusionEncoder:
        """M3: RGB + robot state (add self-awareness)."""
        return cls(use_rgb=True, use_events=False, use_lidar=False, use_proprio=True)
    
    @classmethod
    def full_fusion(cls) -> RealFusionEncoder:
        """M4: All modalities (maximum info)."""
        return cls(use_rgb=True, use_events=True, use_lidar=True, use_proprio=True)
    
    def encode(self, observation: Dict) -> np.ndarray:
        """
        Extract features from observation and fuse.
        
        Args:
            observation: {
                'rgb': ndarray [H, W, 3] uint8,
                'state': ndarray [6] (joint angles),  # Optional
                'prev_rgb': ndarray [H, W, 3] uint8   # Optional (for events)
            }
        
        Returns:
            embedding: ndarray [dim] float32
        """
        features = []
        if self.use_rgb:
            features.append(self.extract_rgb_features(observation['rgb']))
        if self.use_events:
            features.append(self.extract_event_features(
                observation['rgb'], 
                observation.get('prev_rgb')
            ))
        if self.use_lidar:
            features.append(self.extract_lidar_features(observation['rgb']))
        if self.use_proprio:
            features.append(self.extract_proprio_features(observation['state']))
        
        # Concatenate all features
        all_features = np.concatenate(features, axis=0)
        
        # Normalize per-dimension
        all_features = (all_features - all_features.mean()) / (all_features.std() + 1e-6)
        
        return all_features.astype(np.float32)
```

**Feature Extraction Details**:

**RGB Features (128-dim)**:
```
Input: 84×84×3 RGB uint8 image

Process:
  1. Normalize to float [0, 1]
  2. Compute color channels
  3. Downsample to 8×8 spatial grid
     └─ 64-dim feature (8×8 pixels × 1 value per pixel)
  4. Compute color statistics
     └─ 32-dim (mean, std for R, G, B, brightness, gradients)
  5. Compute gradients (edges)
     └─ 32-dim (edge map statistics)

Output: [64 + 32 + 32] = 128-dim feature vector
```

**Event Features (96-dim)**:
```
Input: Current frame + previous frame (or frame buffer)

Process:
  1. Convert both frames to grayscale
  2. Compute absolute difference per pixel
  3. Threshold at 0.15 (15% brightness change)
     → Binary event map
  4. Create temporal voxel grid [5 bins × H × W]
     └─ Use frame history to populate bins
  5. Compute optical flow from differences
     └─ 45-dim (sparse spatial grid)
  6. Compute histogram of event magnitudes
     └─ 51-dim (binned statistics)

Output: [45 + 51] = 96-dim feature vector
```

**LiDAR Features (64-dim)** (Simulated from RGB Gradients):
```
Input: RGB image

Process:
  1. Compute image gradients (Sobel-like)
     → Treat as "depth edges"
  2. Corner/edge detection via Harris corner response
     └─ 32-dim (corner locations + strength)
  3. Depth estimation from gradients
     └─ 32-dim (gradient magnitude histogram)

Output: [32 + 32] = 64-dim feature vector

Note: This is SIMULATED LiDAR (no real sensor).
      Future: Replace with actual v2e event camera integration.
```

**Proprioception Features (32-dim)**:
```
Input: Joint state vector [6] (angles only)

Process:
  1. Normalize joint angles to [-1, 1]
     → 6-dim
  2. Compute joint velocities (if available)
     → 6-dim (zeros if not available)
  3. Gripper state
     → 2-dim (0 = open, 1 = closed)
  4. Padding to 32-dim
     → 18-dim padding

Output: [6 + 6 + 2 + 18] = 32-dim feature vector
```

### 4.4 SmolVLA Integration & VLA Server

**Server File**: [vla/vla_production_server.py](vla/vla_production_server.py)  
**Client File**: [src/smolvla/real_client.py](src/smolvla/real_client.py)  
**Lines**: 250+ (server) + 150+ (client)

**VLA Server Architecture**:
```
┌────────────────────────────────────────────┐
│  FastAPI Server (vla_production_server.py)  │
├────────────────────────────────────────────┤
│  Endpoints:                                 │
│  ├─ GET /health                            │
│  │   └─ Check model loaded, device ready   │
│  │       Response: {status, model_id, ...} │
│  └─ POST /predict                          │
│      ├─ Input: {rgb_b64, state, instruction}
│      ├─ Process:
│      │   1. Decode base64 image
│      │   2. Preprocess image (256×256)
│      │   3. Build observation dict
│      │   4. Tokenize instruction
│      │   5. Run inference with 10s timeout
│      │   6. Resource cleanup on timeout
│      └─ Output: {action, latency_ms, success}
│
├────────────────────────────────────────────┤
│  Model Loading (One-time on startup):      │
│  ├─ Load SmolVLAPolicy from HuggingFace    │
│  ├─ Determine device (CUDA/MPS/CPU)        │
│  ├─ Cache model in memory                  │
│  └─ Ready for inference                    │
│
├────────────────────────────────────────────┤
│  Resource Management:                      │
│  ├─ CUDA memory cleanup every N requests   │
│  ├─ Model inference timeout: 10s           │
│  ├─ Request counter for cleanup triggers   │
│  ├─ Automatic recovery on timeout          │
│  └─ Aggressive cleanup on consecutive errors
│
└────────────────────────────────────────────┘
```

**Server Startup Process**:
```python
# Global state on server startup
def load_model_on_startup():
    global model, device, model_ready, tokenizer
    
    # 1. Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Metal (M1/M2/M3)
    else:
        device = torch.device("cpu")
    
    # 2. Load SmolVLA model
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    model = SmolVLAPolicy.from_pretrained(
        "lerobot/smolvla_base",
        cache_dir=CACHE_DIR
    )
    model.to(device)
    model.eval()
    
    # 3. Load tokenizer for instructions
    tokenizer = AutoTokenizer.from_pretrained("google/recurrent-gemma-2b")
    
    # 4. Mark as ready
    model_ready = True
    logger.info(f"[READY] Model loaded on {device}")

# Startup event (FastAPI)
@app.on_event("startup")
async def startup():
    load_model_on_startup()
```

**Inference Endpoint** (`/predict`):
```python
@app.post("/predict")
async def predict(request: PredictRequest) -> PredictResponse:
    """
    Main inference endpoint. 
    
    Critical constraint: Must timeout after 10s and return error.
    Prevents hanging connections.
    """
    
    t_start = time.perf_counter()
    
    try:
        # 1. Decode base64 image
        img_bytes = base64.b64decode(request.rgb_image_b64)
        pil_img = Image.open(io.BytesIO(img_bytes))
        rgb_array = np.array(pil_img)  # [H, W, 3] uint8
        
        # 2. Preprocess image
        rgb_resized = preprocess_image(rgb_array)  # [256, 256, 3]
        rgb_tensor = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0)
        rgb_tensor = rgb_tensor.to(device)  # [1, 3, 256, 256]
        
        # 3. Build observation dictionary
        observation = {
            "observation.images.camera1": rgb_tensor,
            "observation.images.camera2": rgb_tensor,  # Replicate for model
            "observation.images.camera3": rgb_tensor,
            "observation.state": torch.from_numpy(request.state).float().to(device),
        }
        
        # 4. Tokenize instruction
        tokens = tokenizer(
            request.instruction,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(device)
        # Add to observation
        observation["observation.language.tokens"] = tokens["input_ids"]
        observation["observation.language.attention_mask"] = tokens["attention_mask"]
        
        # 5. Run inference with timeout
        try:
            with torch.inference_mode():
                # Timeout wrapper: 10 seconds max
                action = await asyncio.wait_for(
                    asyncio.to_thread(
                        model.select_action,
                        observation
                    ),
                    timeout=10.0  # CRITICAL: 10s max inference
                )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Model inference timeout (>10.0s)"
            )
        
        # 6. Cleanup resources
        cleanup_resources()
        
        # 7. Build response
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        
        return PredictResponse(
            action=action.cpu().numpy().tolist(),
            action_std=[0.1] * len(action),  # Placeholder
            latency_ms=elapsed_ms,
            success=True
        )
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Client** (RealSmolVLAClient):
```python
# src/smolvla/real_client.py
class RealSmolVLAClient:
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout_s: float = 5.0,
        max_retries: int = 3,
    ):
        self.server_url = server_url
        self.endpoint = "/predict"
        self.full_url = f"{self.server_url}/{self.endpoint}"
        self.timeout_s = timeout_s
        self.max_retries = max_retries
    
    async def predict(
        self,
        rgb_image: np.ndarray,
        state: Optional[np.ndarray] = None,
        instruction: Optional[str] = None,
    ) -> np.ndarray:
        """
        Query VLA server for action prediction.
        
        Args:
            rgb_image: [H, W, 3] uint8 RGB
            state: [n_joints] float (optional)
            instruction: "pick up the object" (optional)
        
        Returns:
            action: [action_dim] float32
        
        Raises:
            Exception if server unavailable after retries
        """
        
        # Encode image to base64
        pil_img = Image.fromarray(rgb_image.astype(np.uint8))
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=85)
        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        # Build payload
        payload = {"rgb_image_b64": img_b64}
        if state is not None:
            payload["state"] = state.tolist()
        if instruction is not None:
            payload["instruction"] = instruction
        
        # Query with retries
        for attempt in range(self.max_retries):
            try:
                t_start = time.perf_counter()
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.full_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=self.timeout_s)
                    ) as response:
                        if response.status != 200:
                            raise RuntimeError(f"Server returned {response.status}")
                        
                        response_json = await response.json()
                        elapsed_ms = (time.perf_counter() - t_start) * 1000
                        
                        self.latency_ms.append(elapsed_ms)
                        self.success_count += 1
                        
                        return np.array(response_json["action"], dtype=np.float32)
            
            except Exception as e:
                if attempt == self.max_retries - 1:
                    self.fail_count += 1
                    raise
                # Exponential backoff
                await asyncio.sleep(0.1 * (2 ** attempt))
    
    async def health_check(self) -> bool:
        """Check if server is healthy."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.server_url}/health",
                    timeout=aiohttp.ClientTimeout(total=2.0)
                ) as response:
                    return response.status == 200
        except:
            return False
```

### 4.5 Dual-System Controller

**File**: [src/integration/dual_system_controller.py](src/integration/dual_system_controller.py)  
**Lines**: 200+

**State Machine**:
```
┌─────────┐
│  INIT   │  (First observation received)
└────┬────┘
     │ (transition on first step)
     ↓
┌──────────────────────────────────────────┐
│ TRACKING                                 │
│ └─ Following VLA action chunk             │
│ └─ MPC computing optimal torques          │
│ └─ Stepping simulation                    │
└──────┬───────────────────────────────────┘
       │ (goal reached check → True)
       ↓
┌──────────────────────────────────────────┐
│ GOAL_REACHED                             │
│ └─ Waiting for next VLA action chunk      │
│ └─ Hovering at goal with passive control │
└──────┬───────────────────────────────────┘
       │ (new chunk available)
       ↓
     Back to TRACKING

Error Cases:
TRACKING / GOAL_REACHED ──(exception)──> ERROR
```

**Step Function Timing**:
```
step() must complete in < 20ms (target: < 10ms)

Breakdown (typical):
  ├─ Update state observables (1-2ms)
  ├─ State machine transition (0.5-1ms)
  ├─ Get reference trajectory from buffer (0.5-1ms)
  ├─ MPC solver (8-12ms) ← DOMINANT
  ├─ Compute return torque (0.5-1ms)
  └─ Total: 10-18ms ✓

Key constraint: Step is PURELY SYNCHRONOUS
  ✓ No await/async
  ✓ No blocking I/O
  ✓ No network calls
  ✓ Reference trajectory always available (from previous VLA chunk)
```

**VLA Query Thread** (Async):
```python
# Runs in background, never blocks main control loop

async def vla_query_loop():
    """Async loop for SmolVLA queries (runs independently)."""
    
    # Initial warmup delay
    await asyncio.sleep(1.0)
    
    while self.running:
        try:
            # 1. Wait for query interval (5 Hz = 200ms)
            await asyncio.sleep(self.vla_interval)
            
            # 2. Get current observation (non-blocking read)
            rgb = self.rgb_current.copy()
            state = self.q_current[:6].copy()  # Arm joints only
            instruction = self.instruction
            
            if rgb is None or state is None:
                continue  # Skip if observation not ready
            
            # 3. Query VLA (blocking, but async)
            # If slow, the MAIN control loop continues unaffected
            try:
                action_chunk = await asyncio.wait_for(
                    self.vla_client.predict(
                        rgb_image=rgb,
                        state=state,
                        instruction=instruction
                    ),
                    timeout=5.0  # Safety timeout
                )
                
                # 4. Update trajectory buffer (non-blocking write)
                self.trajectory_buffer.update(action_chunk)
                
                # Log statistics
                self.vla_successes += 1
                
            except asyncio.TimeoutError:
                self.logger.warning("VLA query timeout")
                self.vla_timeouts += 1
            except Exception as e:
                self.logger.error(f"VLA query failed: {e}")
                self.vla_errors += 1
        
        except Exception as e:
            self.logger.error(f"VLA loop error: {e}")
            break
```

**Integration with MPC**:
```python
# Simplified control loop (actual code in dual_system_controller.py)

for step in range(1000):
    # READ OBSERVATIONS (from simulator)
    obs = env.step(tau_prev)     # Synchronous
    q = obs['q']
    qdot = obs['qdot']
    rgb = obs['rgb']
    
    # FETCH REFERENCE (non-blocking read from buffer)
    # If VLA hasn't queried yet, uses default ref (zeros or prev)
    ref_traj = trajectory_buffer.get_reference()
    
    # SOLVE MPC (synchronous, ~10ms)
    tau_opt = mpc_solver.solve(
        state=(q, qdot),
        reference=ref_traj,
        horizon=10,
        dt=0.01
    )
    
    # RETURN CONTROL
    tau_prev = tau_opt
    
    # Meanwhile, in BACKGROUND THREAD:
    # - VLA is querying server (async/await)
    # - When ready, updates trajectory_buffer
    # - Main loop unaffected (never waits)
```

---

## 5. MODULE INTERACTION PATTERNS

### 5.1 Full Pipeline: One Control Step

```
TIMELINE: step() execution at 100 Hz (every 10ms)

T=0.0ms
├─ [SYNC] DualSystemController.step() called
│  ├─ Input: q=[0.5, 0.2, ...], qdot=[0.1, 0.05, ...], rgb=
│  │
│  ├─ (1) Update internal state (1-2ms)
│  │   └─ self.q_current = q.copy()
│  │   └─ self.rgb_current = rgb.copy()
│  │
│  ├─ (2) State machine (0.5-1ms)
│  │   └─ if INIT: → TRACKING
│  │   └─ Check goal arrival
│  │
│  ├─ (3) Get reference trajectory (0.5-1ms)
│  │   └─ ref_traj = trajectory_buffer.get_reference()
│  │   └─ (Might be from previous VLA query, or default)
│  │
│  ├─ (4) Call MPC solver (8-12ms) ← MAIN COMPUTATION
│  │   ├─ Input: (q, qdot, ref_traj, Q, R, dt)
│  │   ├─ Process:
│  │   │  └─ Compute dynamics M(q), C(q,qdot), G(q)
│  │   │  └─ Linearize dynamics around current state
│  │   │  └─ Build QP matrices (P, q, C, A_ineq, bounds)
│  │   │  └─ Solve QP via Stuart-Landau solver (2-5s internal time)
│  │   │  └─ Extract optimal torque τ* from solution
│  │   └─ Output: τ_opt ∈ ℝ^8
│  │
│  ├─ (5) Return torque (0.5-1ms)
│  │   └─ return tau_opt
│  │
T=10ms (typical)
└─ [SYNC] Control function returns to simulator
   └─ Next observation ready at t=10ms
   └─ MPC is ready to solve again

PARALLEL (Background thread, doesn't block main loop):
T=0-10ms: VLA thread may be executing
├─ Waiting for query interval (e.g., 200ms = 20 steps)
├─ Or executing previous query (async/await, non-blocking)
├─ When result ready, calls trajectory_buffer.update(action_chunk)

Example Timeline over 200ms (20 control steps):
Step 0:   MPC compute, ref_traj = default
Step 1:   MPC compute, ref_traj = default
...
Step 5:   VLA query completes → trajectory_buffer.update(new_chunk)
Step 6:   MPC compute with NEW ref_traj from VLA
Step 7:   MPC compute, tracking new reference
...
Step 15:  MPC compute
Step 20:  Next VLA query fires (async)
```

### 5.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    RUN ONE EPISODE (1000 steps)                 │
└─────────────────────────────────────────────────────────────────┘
            │
            ├─── Initialize Environment
            │    └─ XArmEnv.reset()
            │       ├─ Load xarm_6dof.xml (MuJoCo model)
            │       ├─ Set q=0, qdot=0
            │       └─ Return: {q, qdot, rgb}
            │
            ├─── Launch DualSystemController
            │    └─ DualSystemController.__init__()
            │       ├─ Reset MPC solver state
            │       ├─ Initialize TrajectoryBuffer
            │       ├─ Connect to SmolVLA client
            │       └─ Start VLA query thread (asyncio)
            │
            ├─── FOR STEP = 0 to 999:
            │    │
            │    ├─ Current Observation
            │    │  ├─ XArmEnv.get_joint_pos() → q ∈ ℝ^6
            │    │  ├─ XArmEnv.get_joint_vel() → qdot ∈ ℝ^6
            │    │  └─ XArmEnv.render_rgb() → rgb ∈ ℝ^(H, W, 3)
            │    │
            │    ├──[MAIN CONTROL LOOP] ──┐
            │    │  DualSystemController.step(q, qdot, rgb, instruction)
            │    │  │
            │    │  ├─ (1) Synchronous: Update state
            │    │  │
            │    │  ├─ (2) Synchronous: State machine
            │    │  │
            │    │  ├─ (3) Synchronous: Fetch reference
            │    │  │     └─ TrajectoryBuffer.get_reference()
            │    │  │        (non-blocking read from VLA thread)
            │    │  │
            │    │  ├─ (4) Synchronous: MPC solve
            │    │  │     └─ XArmMPCController.solve_mpc(...)
            │    │  │        ├─ Compute M(q), C(q,qdot), G(q)
            │    │  │        ├─ Linearize dynamics
            │    │  │        ├─ Build QP: P, q, C, A_ineq, bounds
            │    │  │        ├─ Call SL solver
            │    │  │        │  └─ Solve ODE for 2-5 seconds (solver time)
            │    │  │        └─ Extract τ* from solution
            │    │  │
            │    │  └─ (5) Return: τ ∈ ℝ^8
            │    │  
            │    │  ┌─ Time taken: 8-12ms (< 20ms limit)
            │    │  │ Frequency: 100 Hz (10ms per step)
            │    └──┘
            │    │
            │    ├─ Apply Torque
            │    │  ├─ XArmEnv.step(τ)
            │    │  │  ├─ Clamp τ to limits
            │    │  │  ├─ mujoco.mj_step(model, data) [dt=1ms]
            │    │  │  ├─ Integrate 10 internal steps (10×1ms = 10ms)
            │    │  │  └─ Return new obs: {q', qdot', rgb', ...}
            │    │  │
            │    │  └─ Update episode metrics
            │    │     ├─ Track error: ||q' - q_ref||
            │    │     ├─ Accumulate reward
            │    │     └─ Check termination
            │    │
            │    ├─ PARALLEL (Background, non-blocking):
            │    │  │
            │    │  ├─ VLA Query Thread (async)
            │    │  │  └─ Every 200ms (every 20 steps):
            │    │  │     ├─ Get current rgb, state
            │    │  │     ├─ Query VLA server (async HTTP)
            │    │  │     │  └─ Wait_for: timeout 5s
            │    │  │     │     ├─ Encode image to base64
            │    │  │     │     ├─ POST /predict
            │    │  │     │     ├─ Server inference (20-80ms)
            │    │  │     │     └─ Receive action_chunk
            │    │  │     │
            │    │  │     └─ Update trajectory_buffer (non-blocking write)
            │    │  │        └─ ref_trajectory[:] = action_chunk[:, :6]
            │    │  │
            │    │  └─ (Main loop continues unaffected during VLA query)
            │    │
            │    └─ LOOP BACK to next step
            │
            └─── End Episode
                 ├─ Compute final metrics
                 ├─ Save results to evaluation/results/
                 └─ Print summary
```

### 5.3 Asynchronous Execution Pattern

```python
# Main control loop (synchronous, 100 Hz)
def control_loop():
    while True:
        obs = env.step(tau_prev)
        tau_new = controller.step(obs)  # Must complete < 20ms
        tau_prev = tau_new

# VLA async thread (5 Hz, 200ms between queries)
async def vla_loop():
    while True:
        await asyncio.sleep(0.2)  # Wait 200ms
        
        # Non-blocking read of current observation
        rgb = controller.rgb_current
        state = controller.q_current[:6]
        
        # Async query (might take 30-100ms, but doesn't block main loop)
        try:
            action_chunk = await vla_client.predict(rgb, state)
            # Non-blocking write to buffer
            trajectory_buffer.update(action_chunk)
        except:
            pass  # Silent fail, use default ref

# Main execution
# These run CONCURRENTLY:
# Coroutine 1: control_loop() [synchronous, runs every 10ms]
# Coroutine 2: vla_loop() [asynchronous, runs every 200ms]
# No blocking between them!

asyncio.run(asyncio.gather(
    control_loop(),      # Synchronous wrapped as async
    vla_loop()           # Fully async
))
```

---

## 6. DATA FLOW DIAGRAMS

### 6.1 Control Loop Data Flow

```
Input Data                Processing                Output Data
─────────────────────  ──────────────────────  ──────────────────────

Observation:           DualSystemController:    Torque Command:
  q (6-D)      ─┐      .step() {                 τ (8-D) ──┐
  qdot (6-D)   ─┼─────   ├─ State mach.         (6 arm  │
  rgb (H,W,3)  ─┤      └ │ + MPC solve        + 2 gripper)  │
  instruction  ─┘         └─ Return τ*           │
                        }                       │
                           │                     │
                           ├─ Queries      │
                           │  TrajectoryBuffer.get_reference()
                           │
                    Reference Trajectory (from VLA thread)
                           │
                           ├─ Builds
                           │  QP matrices ∈ ℝ^(n×n)
                           │
                    ┌──────────────────┐
                    │ SL QP Solver     │
                    │ ├─ P, q (cost)   │
                    │ ├─ C, d (eq)     │
                    │ ├─ A, bounds(ineq)
                    │ └─ Solves ODE    │
                    │    for 2-5s      │
                    └──────────────────┘
                           │
                           └──→ τ* ∈ ℝ^8
                                 │
                                 └→ RETURNS TO SIMULATOR
```

### 6.2 Sensor Fusion Data Flow

```
Raw Observations          Feature Extraction        Fusion Embedding
──────────────────────  ──────────────────────  ──────────────────────

RGB Image            RealFusionEncoder:
  (84×84×3) ──────────  ├─ extract_rgb_features()
              │   │      │   ├─ Downsample: 64-dim
              │   │      │   ├─ Color stats: 32-dim
              │   │      │   └─ Gradients: 32-dim
              │   │      │       → 128-dim RGB embedding
              │   │
              │   ├─ extract_event_features()
              │   │   ├─ Optical flow: 45-dim
              │   │   └─ Histogram: 51-dim
              │   │       → 96-dim Event embedding
              │
              └─ extract_lidar_features()
                  ├─ Corners: 32-dim
                  └─ Depth: 32-dim
                      → 64-dim LiDAR embedding

Robot State (6-D)  extract_proprio_features()
  q ─────────────────  → 32-dim Proprioception embedding
                  │
                  └──→ Concatenate [enabled modalities]
                        │
                        ├─ M0: [128] RGB only
                        ├─ M1: [128+96] RGB+Events
                        ├─ M2: [128+64] RGB+LiDAR
                        ├─ M3: [128+32] RGB+Proprio
                        └─ M4: [128+96+64+32] Full fusion
                            │
                            └──→ Final embedding (256-dim or less)
                                  │
                                  └──→ Can feed to downstream model
                                       (future: for classification/RL)
```

---

## 7. CONFIGURATION SYSTEM

### 7.1 Configuration Files

**config/config.yaml** (Main configuration):
```yaml
solver:
  neuromorphic:
    learning_rate: 0.01
    max_iterations: 1000
    tolerance: 1e-6
    decay_rate: 0.99
  osqp:
    eps_abs: 1e-4
    eps_rel: 1e-4
    max_iter: 5000

mpc:
  horizon: 10           # N steps ahead
  dt: 0.001            # 10ms per step (100 Hz)
  solver: "osqp"       # Choose solver
  weight_state: 1.0    # Q matrix weight
  weight_control: 0.01 # R matrix weight

robot:
  model: "arm_2dof"    # or "arm_3dof"
  dof: 2
  gravity: 9.81

simulation:
  render: true
  fps: 60
  episode_length: 1000

paths:
  assets: "assets/"
  data: "data/"
  results: "results/"
  models: "assets/"

logging:
  level: "INFO"
  format: "%(asct asctime)s - %(name)s - %(levelname)s - %(message)s"
```

**config/robots/xarm_6dof.yaml** (Robot parameters):
```yaml
name: xarm_6dof
num_joints: 6
actuators: 8  # 6 arm + 2 gripper

joint_limits:
  q_min: [-6.283, -3.665, -6.109, -4.555, -6.109, -6.283]
  q_max: [6.283, 3.665, 6.109, 4.555, 6.109, 6.283]
  
velocity_limits:
  qdot_max: [3.0, 2.5, 2.5, 2.0, 1.5, 2.0]

torque_limits:
  tau_max: [20.0, 15.0, 15.0, 10.0, 8.0, 6.0]

inertia:
  masses: [1.2, 0.8, 0.6, 0.5, 0.4, 0.3]  # kg per link

gripper:
  type: parallel
  num_fingers: 2
  max_force: 100  # N
```

### 7.2 Loading Configuration

```python
import yaml
from pathlib import Path

def load_config(config_path="config/config.yaml"):
    """Load YAML config into dict."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

# Usage
config = load_config()
mpc_horizon = config['mpc']['horizon']  # 10
dt = config['mpc']['dt']  # 0.001
solver_type = config['mpc']['solver']  # "osqp"
```

---

## 8. RUNNING & TESTING

### 8.1 Setup

```bash
# 1. Clone repo (assumed done)
cd /path/to/neuromorphic-robot

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  (Windows)

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python3 -c "import mujoco; print('MuJoCo ✓')"
python3 -c "import torch; print(f'PyTorch ✓ (Device: {torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")})')"
python3 -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print('LeRobot ✓')"
```

### 8.2 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_xarm_env.py -v

# Specific test function
pytest tests/test_xarm_env.py::test_env_renders -v

# With output
pytest tests/ -v -s

# Short summary
pytest tests/ --tb=short

# Stop on first failure
pytest tests/ -x

# Test count
pytest tests/ --collect-only | grep "test_" | wc -l
```

### 8.3 Running Benchmarks

```bash
# Quick test (3 episodes)
python3 scripts/phase12_quick_test.py

# Full ablation (30 episodes × 5 modes)
python3 scripts/phase13_final.py

# Check VLA server health
python3 scripts/health_check_vla.py

# Inspect dataset
python3 scripts/inspect_dataset.py
```

### 8.4 Starting VLA Server

```bash
# Terminal 1: Start server (runs in background)
cd /path/to/neuromorphic-robot
source .venv/bin/activate
python3 vla/vla_production_server.py

# Output:
# INFO:    Uvicorn running on http://0.0.0.0:8000
# INFO:    Press CTRL+C to quit
# [VLA Server] Loading model on startup...
# [VLA Server] Model loaded on device: cpu (or cuda/mps)

# Terminal 2: Send test request
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rgb_image_b64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    "instruction": "pick up the object"
  }'

# Response:
# {
#   "action": [0.1, 0.2, ..., 0.7],
#   "action_std": [...],
#   "latency_ms": 45.2,
#   "success": true
# }
```

---

## 9. CURRENT PHASE STATUS

### Phase 13: Real Sensor Fusion Ablation Study

**Status**: ✅ COMPLETE

**Completion Dates**:
- Stage 1 (Encoders): Mar 13, 2026 ✅
- Stage 2 (Simulators): Mar 14, 2026 ✅
- Stage 3-5 (Ablation): Mar 14-15, 2026 ✅

**Deliverables**:
1. **RealFusionEncoder** (src/fusion/encoders/real_fusion_simple.py)
   - 5 Factory methods for 5 fusion modes
   - Extract real features from RGB, events, LiDAR, proprioception
   - No synthetic data

2. **Event Camera Simulator** (src/simulation/cameras/event_camera_simple.py)
   - Frame-difference based event generation
   - Temporal voxel grid representation
   - 96-dim feature embedding

3.  **Ablation Study Results**
   - Tested 5 modes × 3+ episodes
   - M0 (RGB): Baseline
   - M1 (RGB+Events): Temporal dynamics
   - M2 (RGB+LiDAR): Spatial geometry
   - M3 (RGB+Proprio): Robot state
   - M4 (Full): Best performance (27.9ms latency)

**Key Findings**:
- Full fusion (M4) shows best compatibility with VLA
- No synthetic data used (all real feature extraction)
- Fusion overhead: 2-6ms (negligible)
- Integration with production VLA server working

**Known Issues** (RESOLVED):
- ✅ VLA Server timeout: Fixed with 10s timeout + cleanup
- ✅ Memory leaks: Fixed with gc.collect() + CUDA cleanup
- ✅ Resource exhaustion: Fixed with request-based cleanup

**Next Phase** (Phase 14):
- Integration of real event cameras (v2e)
- Fine-tuning VLA on robot observation
- Extended benchmarking (102 episodes)
- Thesis validation gates

---

## 10. DEVELOPMENT WORKFLOW

### 10.1 Typical Development Cycle

```
1. Define Feature / Fix
   ├─ Write down requirement
   ├─ Identify affected modules
   └─ Plan implementation

2. Implement
   ├─ Write code
   ├─ Place in correct directory
   ├─ Follow coding standards
   └─ Add logging

3. Test Locally
   ├─ Unit test for new function
   ├─ Integration test with dependencies
   ├─ Run full test suite
   └─ Verify no regressions

4. Run Benchmarks
   ├─ Quick test (3 episodes)
   ├─ Full test if time allows
   └─ Compare to baseline

5. Document
   ├─ Add docstrings
   ├─ Update this guide if needed
   └─ Credit (if collaborative)

6. Commit
   ├─ Clear commit message
   ├─ Reference phase/issue
   └─ Push to repo
```

### 10.2 Coding Standards

```python
# Style
PEP 8: Follow Python style guide
Lines: Max 100 characters
Naming: snake_case for functions/vars, CamelCase for classes

# Docstrings (Google style)
def forward_kinematics(self, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute end-effector pose from joint angles.
    
    Args:
        q: Joint angles [3] (radians)
    
    Returns:
        p: End-effector position [3] (meters)
        R: End-effector orientation (3×3 rotation matrix)
    
    Raises:
        ValueError: If q shape invalid
    """

# Type hints
from typing import Optional, Tuple, Dict, List
def step(self, q: np.ndarray, tau: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    ...

# Logging
import logging
logger = logging.getLogger(__name__)

logger.info("Model loaded successfully")
logger.warning("Singularity near workspace boundary")
logger.error(f"Joint {i} out of bounds: {q[i]}")
```

### 10.3 Adding New Features

**Example: Add new sensor modality to fusion**

1. **Create encoder function** in [src/fusion/encoders/real_fusion_simple.py](src/fusion/encoders/real_fusion_simple.py):
   ```python
   def extract_thermal_features(self, thermal_frame):
       """Extract features from thermal image."""
       # Process thermal frame
       # Return thermal_features (dim-D vector)
       return thermal_features
   ```

2. **Add to fusion class**:
   ```python
   class RealFusionEncoder:
       def __init__(self, ..., use_thermal=False):
           ...
           self.use_thermal = use_thermal
       
       def encode(self, observation):
           features = []
           if self.use_thermal:
               features.append(self.extract_thermal_features(observation['thermal']))
           ...
   ```

3. **Add factory method**:
   ```python
   @classmethod
   def rgb_thermal(cls):
       """M5: RGB + thermal."""
       return cls(use_rgb=True, use_thermal=True)
   ```

4. **Write test**:
   ```python
   # tests/test_thermal_encoder.py
   def test_thermal_features():
       encoder = RealFusionEncoder.rgb_thermal()
       thermal_frame = np.random.rand(84, 84).astype(np.float32)
       features = encoder.extract_thermal_features(thermal_frame)
       assert features.shape[0] > 0, "Features not extracted"
   ```

5. **Run tests**:
   ```bash
   pytest tests/test_thermal_encoder.py -v
   ```

6. **Update documentation**:
   - Add M5 to fusion modes table
   - Document thermal feature extraction
   - Add example usage

---

## APPENDIX: Key Files Reference

| File | Purpose | Key Classes |
|------|---------|-------------|
| src/main.py | Entry point | (None - standalone script) |
| src/simulation/envs/xarm_env.py | MuJoCo env | XArmEnv |
| src/mpc/xarm_controller.py | MPC controller | XArmMPCController |
| src/solver/stuart_landau_lagrange_direct.py | QP solver | StuartLandauLagrangeDirect |
| src/smolvla/real_client.py | VLA client | RealSmolVLAClient |
| src/fusion/encoders/real_fusion_simple.py | Sensor fusion | RealFusionEncoder |
| src/integration/dual_system_controller.py | Main orchestrator | DualSystemController |
| src/dynamics/kinematics_3dof.py | Kinematics | Arm3DOF |
| vla/vla_production_server.py | VLA server | (FastAPI app) |
| tests/test_xarm_env.py | Environment tests | (13 tests) |
| tests/test_mpc_gate2.py | MPC tests | (9 tests) |
| evaluation/benchmarks/phase13_final.py | Benchmarks | (Runner script) |

---

**END OF DEVELOPER GUIDE**

For questions about specific implementations, refer to the source files with their docstrings. For theoretical background, see docs/07-THEORY.md and docs/sensor_fusion_vla_mpc_techspec_v2.md.
