# Phase 8B: Dual-System Integration (In Progress)

**Status:** Phase 8A Complete ✓ → Phase 8B Starting

## What Was Completed (Phase 8A)

### SmolVLA Server ✓
- **Model:** LeRobot SmolVLA (450M param, distilled OpenVLA)
- **Architecture:** FastAPI + ngrok tunnel (non-blocking remote access)
- **Endpoint:** `/predict` accepts base64 RGB image, returns 3-DOF action
- **Latency:** ~700ms per query on Colab T4 GPU
- **Notebook:** `vla/smolvla_server.ipynb` (clean, 9 cells, production-ready)

### Key Implementation Details
```python
# Frame dict format (CRITICAL)
frame = {
    'observation.images.camera1': image_chw,  # [3, H, W] uint8
    'observation.images.camera2': image_chw,  # (replicate for 3-view input)
    'observation.images.camera3': image_chw,
    'observation.state': state_vector,         # [6] proprioceptive
    'task': 'reaching',                        # task description
}

# Dimension fix
for key, val in batch.items():
    if isinstance(val, np.ndarray) and val.ndim == 3:
        batch[key] = torch.from_numpy(val).unsqueeze(0).to(device)
    elif isinstance(val, torch.Tensor):
        batch[key] = val.to(device)

# Inference & postprocess
action = policy.select_action(batch)  # [1, n_actions]
action_dict = postprocess({"action": action})
action_4d = action_dict["action"][0].cpu().numpy()[:4]
```

## Next: Phase 8B - Integration Layer

### Objective
Implement non-blocking architecture:
- Local MPC loop runs at **100+ Hz** (guaranteed)
- VLA queries run at **1-5 Hz** in background (HTTP + async)
- Graceful fallback if VLA times out (hold last trajectory)

### Tasks (In Order)

#### Task 1: AsyncVLAClient (`src/smolvla_client/async_client.py`)
```python
class SmolVLAClient:
    def __init__(self, server_url: str, timeout_s: float = 1.0):
        self.url = server_url
        self.timeout = timeout_s
        self.session = None
    
    async def query_action(self, rgb: np.ndarray, state: np.ndarray) -> dict | None:
        """Query VLA prediction. Returns None on timeout."""
        # 1. Encode image to base64
        # 2. POST to /predict with 1-second timeout
        # 3. Return {"action": [...], "latency_ms": ...} or None
        # 4. Log latency at DEBUG level
```

**Tests:**
- [ ] `test_client_health_check` — GET /health succeeds
- [ ] `test_client_single_query` — Single /predict returns action
- [ ] `test_client_timeout` — Returns None after 1s without error

#### Task 2: TrajectoryBuffer (`src/smolvla_client/trajectory_buffer.py`)
```python
class TrajectoryBuffer:
    def update_subgoal(self, ee_target: np.ndarray, grasp: float):
        """Update goal from latest VLA prediction."""
        # Thread-safe (numpy ops are atomic under GIL)
    
    def get_reference_trajectory(self, q_curr: np.ndarray, N: int, dt: float):
        """Return reference trajectory [q_ref, qdot_ref] for MPC horizon."""
        # Use IK to interpolate from q_curr to ee_target
        # Return shapes: q_ref [N, 3], qdot_ref [N, 3]
    
    def is_goal_reached(self, q_curr: np.ndarray, tolerance: float) -> bool:
        """Check if EE is within tolerance of goal."""
```

**Tests:**
- [ ] `test_buffer_ik_interpolation` — Quintic spline correct
- [ ] `test_buffer_goal_detection` — Goal reached when close enough
- [ ] `test_buffer_thread_safety` — No race conditions

#### Task 3: DualSystemController (`src/integration/dual_system_controller.py`)
```python
class ControlState(Enum):
    INIT = 0
    WAITING_VLA = 1        # Fired async query
    TRACKING = 2           # Following trajectory
    GOAL_REACHED = 3       # At target, ready for next task
    ERROR = 4

class DualSystemController:
    def step(self, q: np.ndarray, qdot: np.ndarray, 
             rgb: np.ndarray, instruction: str) -> np.ndarray:
        """Main control loop (SYNCHRONOUS, ~500 Hz capable)."""
        # 1. Get latest subgoal from TrajectoryBuffer (instant)
        # 2. Generate reference trajectory
        # 3. Run MPC solver
        # 4. Check state transitions
        # 5. Return tau command
        # 6. (Background: fire VLA query if ready, handled separately)
```

**Key:** This function NEVER waits, NEVER calls async, NEVER reaches out to network.

**Tests:**
- [ ] `test_controller_speed` — Runs in < 10ms (target 500 Hz)
- [ ] `test_controller_state_machine` — Correct transitions at milestones
- [ ] `test_controller_reference_tracking` — Generates feasible trajectories

#### Task 4: Background VLA Poll Thread (`src/integration/vla_thread.py`)
```python
def poll_vla_background(vla_client: SmolVLAClient, 
                       trajectory_buffer: TrajectoryBuffer,
                       rgb_buffer: RGBBuffer,
                       instruction: str,
                       poll_interval_s: float = 0.2):
    """
    Run in separate thread with asyncio event loop.
    Poll VLA every 200ms, update trajectory buffer.
    Never blocks main control loop.
    """
    loop = asyncio.new_event_loop()
    
    async def poll_loop():
        while running:
            rgb = rgb_buffer.get_latest()  # Lock-free read
            try:
                result = await vla_client.query_action(rgb, state)
                if result:
                    action = result["action"]
                    trajectory_buffer.update_subgoal(action[:3], action[3])
                    logger.debug(f"VLA latency: {result['latency_ms']:.1f}ms")
            except asyncio.TimeoutError:
                logger.warning("VLA timeout, holding last trajectory")
            except Exception as e:
                logger.error(f"VLA error: {e}")
            
            await asyncio.sleep(poll_interval_s)
    
    loop.run_until_complete(poll_loop())
```

**Startup in main script:**
```python
vla_thread = threading.Thread(
    target=poll_vla_background,
    args=(vla_client, traj_buf, rgb_buf, instruction),
    daemon=True
).start()
```

**Tests:**
- [ ] `test_thread_runs_continuously` — No crashes over 1 minute
- [ ] `test_thread_nonblocking` — Main loop unaffected by thread
- [ ] `test_thread_graceful_failure` — System continues if thread dies

#### Task 5: Integration Tests (`tests/test_integration_e2e.py`)
```python
def test_dual_system_mpc_unaffected_by_vla_latency():
    """
    Slow VLA queries should NOT affect MPC loop rate.
    Mock VLA server that responds in 500ms.
    Measure MPC loop jitter: should be < 10%.
    """

def test_graceful_fallback_when_vla_down():
    """
    Kill VLA server mid-experiment.
    System should hold last known trajectory and continue.
    """

def test_pointing_task_with_vla():
    """
    Full simulation: get subgoal from VLA, track to it, 
    detect arrival, request new subgoal.
    """
```

### Success Criteria for Phase 8B
- [ ] All 3 client tests pass
- [ ] All 3 buffer tests pass  
- [ ] All 3 controller tests pass
- [ ] All 3 thread tests pass
- [ ] All 5 integration tests pass
- [ ] MPC loop timing variance < 10% during VLA queries
- [ ] System recovers gracefully from VLA timeouts
- [ ] State machine transitions logged clearly

### Estimated Effort
**5-7 hours** - Tasks can be parallelized (async client + buffer + thread can be skeletonized together)

### Architecture Diagram
```
┌─────────────────────────────────────────────────────────┐
│  Background Thread (asyncio)                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │ poll_vla_background()                              │  │
│  │  ├─ /predict query (200ms interval)               │  │
│  │  └─ update TrajectoryBuffer (lock-free write)    │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
          ↑ (reads trajectory)         ↓ (updates goal)
    TrajectoryBuffer (thread-safe)
          ↓ (reference traj)
┌─────────────────────────────────────────────────────────┐
│  Main Control Loop (100+ Hz, SYNCHRONOUS)               │
│  ┌───────────────────────────────────────────────────┐  │
│  │ DualSystemController.step()                       │  │
│  │  1. Get subgoal (instant)                        │  │
│  │  2. Generate reference trajectory                │  │
│  │  3. Run MPC solver (20-50ms)                    │  │
│  │  4. Return tau command                           │  │
│  │  (Never waits, never calls network)             │  │
│  └───────────────────────────────────────────────────┘  │
│  Target: ≥ 100 Hz, < 10 ms per step                    │
└─────────────────────────────────────────────────────────┘
```

## Files to Create
1. `src/smolvla_client/__init__.py`
2. `src/smolvla_client/async_client.py`
3. `src/smolvla_client/trajectory_buffer.py`
4. `src/integration/__init__.py`
5. `src/integration/dual_system_controller.py`
6. `src/integration/vla_thread.py`
7. `tests/test_integration_e2e.py`

## Version Control
- Existing MPC solver code: no changes needed
- Existing 3-DOF kinematics: no changes needed
- New code: clean interfaces, full type hints, docstrings

## Handoff to User
Once Phase 8B is complete (~ 1-2 days of focused work):
- SmolVLA is integrated with MPC **non-blocking**
- Can run pointing tasks end-to-end
- Ready for Phase 9: E2E testing suite + performance benchmarks

---
**Start Date:** 13 March 2026  
**Target Completion:** 15 March 2026  
**Next Gate:** Phase 9 (E2E testing + performance benchmarks)
