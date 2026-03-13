# Task 3: DualSystemController Implementation Plan

**Status:** Ready to Start  
**Duration:** 2-2.5 hours estimated  
**Dependencies:** Task 1 ✓, Task 2 ✓  
**Tests:** ~20 unit tests + state machine validation  
**Success Criteria:** All tests pass, step timing < 20ms, state transitions logged  

---

## Overview

The `DualSystemController` is the heart of Phase 8B integration. It orchestrates:
- **System 1 (synchronous):** MPC control loop running at 100-500 Hz
- **System 2 (asynchronous):** VLA queries via background thread at 1-5 Hz

**Critical invariant:** The main `step()` method is **purely synchronous** — it never calls `await`, never blocks on network I/O, and completes in < 20ms per iteration.

---

## Architecture

### State Machine

```
┌──────┐
│ INIT │  System booting, waiting for first observation
└──┬───┘
   │ (trajectory_buffer has subgoal)
   ▼
┌─────────────┐
│ TRACKING    │  Following trajectory toward current subgoal
└──┬────────┬─┘
   │        │ (goal_reached == True)
   │        ▼
   │    ┌──────────────┐
   │    │ GOAL_REACHED │  At target, ready for next VLA query
   │    └──────┬───────┘
   └───────────┘ (loop back to TRACKING on new subgoal)

Error paths:
  TRACKING → ERROR (if IK fails, MPC fails, etc.)
  GOAL_REACHED → ERROR (if can't compute trajectory)
  ERROR → TRACKING (manual recovery or next VLA success)
```

### Data Flow

```
RGB frame + instruction
       │
       ▼
┌─────────────────────────────────────┐
│ Async background thread (System 2)  │
│  SmolVLAClient.query_action()       │
│         ↓                           │
│  TrajectoryBuffer.update_subgoal()  │  [Thread-safe write via GIL]
└─────────────────────────────────────┘
       
       ▲ (reads latest subgoal, instant)
       │
┌─────────────────────────────────────────────────────┐
│ Main control loop (100+ Hz, System 1)              │
│  DualSystemController.step(q, qdot, rgb, instr)   │
│   1. Get reference trajectory (from buffer)         │
│   2. Run MPC solver                                 │
│   3. Check goal arrival (update state)              │
│   4. Return τ                                       │
│   Total: < 20ms per step (guaranteed)              │
└─────────────────────────────────────────────────────┘
       │
       ▼
    Robot control
```

---

## Implementation Details

### 1. State Enum

```python
# src/integration/dual_system_controller.py
from enum import Enum
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

class ControlState(Enum):
    """Control system state machine."""
    INIT = 0              # Initialization (first observation)
    TRACKING = 1         # Following reference trajectory to subgoal
    GOAL_REACHED = 2     # At subgoal, waiting for next VLA query
    ERROR = 3            # Unrecoverable error (logged at WARNING)
```

### 2. DualSystemController Class

```python
class DualSystemController:
    """
    Main control interface for dual-system architecture.
    
    System 1 (Synchronous, ~100-500 Hz):
      - MPC control loop
      - Guaranteed timing (< 20ms per step)
      - Never waits on network I/O
    
    System 2 (Asynchronous, 1-5 Hz):
      - SmolVLA queries in background thread
      - Non-blocking HTTP via aiohttp
      - Updates reference trajectory via TrajectoryBuffer
    
    Thread safety: TrajectoryBuffer uses GIL-based atomicity for numpy ops.
    No explicit locks needed.
    """
    
    def __init__(
        self,
        mpc_solver,                    # Existing Stuart-Landau solver
        smolvla_client,                # Task 1: SmolVLAClient instance
        trajectory_buffer,             # Task 2: TrajectoryBuffer instance
        logger_instance=None,
        mpc_horizon_steps: int = 10,
        control_dt_s: float = 0.01,
        vla_query_interval_s: float = 0.2
    ):
        """
        Initialize dual-system controller.
        
        Args:
            mpc_solver: Existing MPC solver with solve(state, ref_traj) → τ
            smolvla_client: SmolVLAClient for async VLA queries
            trajectory_buffer: TrajectoryBuffer for interpolated reference
            mpc_horizon_steps: MPC horizon N (default 10)
            control_dt_s: Control timestep (default 10ms → 100 Hz)
            vla_query_interval_s: VLA query interval (default 200ms → 5 Hz)
        """
        self.mpc_solver = mpc_solver
        self.vla_client = smolvla_client
        self.trajectory_buffer = trajectory_buffer
        self.logger = logger_instance or logging.getLogger(__name__)
        
        # Timing
        self.mpc_horizon = mpc_horizon_steps
        self.dt = control_dt_s
        self.vla_interval = vla_query_interval_s
        
        # State machine
        self.state = ControlState.INIT
        self._last_state = None
        
        # Observation tracking
        self.q_current = None
        self.qdot_current = None
        self.rgb_current = None
        self.instruction = None
        
        # Timing instrumentation
        self.step_count = 0
        self.step_times_ms = []
        self.last_vla_query_time = 0.0
        
        # Safety
        self.running = True
    
    def step(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        rgb: np.ndarray,
        instruction: str
    ) -> np.ndarray:
        """
        Main synchronous control loop step.
        
        **CRITICAL:** This function NEVER blocks, NEVER calls await,
        NEVER touches network I/O. Must complete in < 20ms.
        
        Args:
            q: Current joint angles [3] (rad)
            qdot: Current joint velocities [3] (rad/s)
            rgb: Current RGB frame [H, W, 3] uint8 (batched if needed)
            instruction: Task instruction string ("pick up", "place", etc.)
        
        Returns:
            tau: Optimal torque command [3] (N·m)
        
        Raises:
            Nothing — all errors logged at WARNING level, fallback returned
        """
        import time
        t0 = time.perf_counter()
        
        try:
            # 1. Update internal state
            self.q_current = q.copy()
            self.qdot_current = qdot.copy()
            self.rgb_current = rgb.copy() if isinstance(rgb, np.ndarray) else rgb
            self.instruction = instruction
            
            # 2. Check goal arrival (updates state machine if needed)
            self._check_goal_arrival(q)
            
            # 3. Get reference trajectory from buffer (instant, thread-safe)
            N = self.mpc_horizon
            q_ref, qdot_ref = self.trajectory_buffer.get_reference_trajectory(
                q, N=N, dt=self.dt
            )
            
            # 4. Prepare MPC state
            x_curr = np.concatenate([q, qdot])  # [6]
            x_ref = q_ref[-1]  # Terminal reference (last point of horizon)
            
            # 5. Run MPC solver
            # Input: current state, reference trajectory (or hold trajectory)
            # Output: optimal torques for next step
            tau = self.mpc_solver.solve(
                x_curr=x_curr,
                x_ref=x_ref,
                q_ref=q_ref,      # Full reference trajectory for cost
                qdot_ref=qdot_ref  # Velocity reference
            )
            
            # Ensure tau is [3] numpy array
            if not isinstance(tau, np.ndarray):
                tau = np.array(tau)
            if tau.shape != (3,):
                tau = tau.flatten()[:3]
            
            # 6. State machine transitions (for logging & diagnostics)
            self._update_state_machine()
            
            # 7. Measure timing
            elapsed = (time.perf_counter() - t0) * 1000  # ms
            self.step_times_ms.append(elapsed)
            if elapsed > 20:
                self.logger.warning(
                    f"[DualSystemController] Step {self.step_count} took {elapsed:.1f}ms "
                    f"(target < 20ms)"
                )
            
            self.step_count += 1
            return tau
        
        except Exception as e:
            # Graceful fallback: return zero torque (hold position)
            self.logger.warning(
                f"[DualSystemController] Error in step: {e}. "
                f"Returning zero torque (hold position)."
            )
            self.state = ControlState.ERROR
            return np.zeros(3)
    
    def _check_goal_arrival(self, q_current: np.ndarray):
        """
        Check if current position has reached the subgoal.
        
        Updates self.state if arrival detected.
        Uses TrajectoryBuffer.check_arrival() for hysteresis logic.
        """
        if self.trajectory_buffer.is_goal_reached(q_current):
            if self.state == ControlState.TRACKING:
                self.state = ControlState.GOAL_REACHED
                self.logger.info(
                    f"[DualSystemController] Goal reached at step {self.step_count}. "
                    f"Waiting for next subgoal."
                )
    
    def _update_state_machine(self):
        """
        Log state transitions for debugging.
        Called after every step.
        """
        if self.state != self._last_state:
            self.logger.info(
                f"[DualSystemController] State transition: "
                f"{self._last_state} → {self.state} at step {self.step_count}"
            )
            self._last_state = self.state
    
    def get_stats(self) -> dict:
        """Return controller statistics."""
        import numpy as np
        stats = {
            "step_count": self.step_count,
            "state": self.state.name,
            "step_time_mean_ms": np.mean(self.step_times_ms) if self.step_times_ms else 0,
            "step_time_max_ms": np.max(self.step_times_ms) if self.step_times_ms else 0,
            "step_time_p95_ms": np.percentile(self.step_times_ms, 95) if len(self.step_times_ms) > 20 else 0,
        }
        return stats
    
    def reset(self):
        """Reset controller to initial state."""
        self.state = ControlState.INIT
        self._last_state = None
        self.step_count = 0
        self.step_times_ms = []
        self.trajectory_buffer.reset()
        self.logger.info("[DualSystemController] Reset to INIT state")
```

---

## Unit Tests (20 total)

**File:** `tests/test_dual_system_controller.py`

### Test Categories

#### 1. Initialization (2 tests)
```python
def test_controller_init_default():
    """Controller initializes with defaults."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    assert controller.state == ControlState.INIT
    assert controller.step_count == 0

def test_controller_init_custom_horizon():
    """Custom horizon parameter accepted."""
    controller = DualSystemController(
        mpc, vla_client, traj_buf,
        mpc_horizon_steps=20
    )
    assert controller.mpc_horizon == 20
```

#### 2. Step Timing (5 tests)
```python
def test_controller_step_timing_under_20ms():
    """Single step completes in < 20ms."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.1]))
    
    q = np.array([0.0, 0.3, -0.3])
    qdot = np.array([0.0, 0.0, 0.0])
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    import time
    t0 = time.perf_counter()
    tau = controller.step(q, qdot, rgb, "reach target")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    assert elapsed_ms < 20, f"Step took {elapsed_ms:.1f}ms, expected < 20ms"
    assert isinstance(tau, np.ndarray)
    assert tau.shape == (3,)

def test_controller_step_consistent_timing():
    """Multiple steps have consistent timing (< 10% variance)."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.1]))
    
    timings = []
    for i in range(20):
        q = np.array([0.0, 0.3 + 0.01*i, -0.3])
        qdot = np.array([0.0, 0.01, 0.0])
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        t0 = time.perf_counter()
        tau = controller.step(q, qdot, rgb, "reach target")
        timings.append((time.perf_counter() - t0) * 1000)
    
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    variance = std_time / mean_time if mean_time > 0 else 0
    
    assert variance < 0.1, f"Timing variance {variance*100:.1f}% > 10%"

def test_controller_step_returns_valid_torque():
    """Step returns valid [3] torque vector."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.1]))
    
    q = np.array([0.0, 0.3, -0.3])
    qdot = np.zeros(3)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    tau = controller.step(q, qdot, rgb, "test")
    
    assert isinstance(tau, np.ndarray)
    assert tau.shape == (3,)
    assert np.all(np.isfinite(tau)), "Torque contains NaN or inf"

def test_controller_step_count_increments():
    """Step counter increments."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.1]))
    
    q = np.array([0.0, 0.3, -0.3])
    qdot = np.zeros(3)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    assert controller.step_count == 0
    controller.step(q, qdot, rgb, "test")
    assert controller.step_count == 1
    controller.step(q, qdot, rgb, "test")
    assert controller.step_count == 2

def test_controller_stats_available():
    """Statistics are recorded and accessible."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.1]))
    
    for i in range(5):
        q = np.array([0.0, 0.3, -0.3])
        qdot = np.zeros(3)
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        controller.step(q, qdot, rgb, "test")
    
    stats = controller.get_stats()
    assert stats["step_count"] == 5
    assert stats["step_time_mean_ms"] > 0
    assert stats["step_time_max_ms"] >= stats["step_time_mean_ms"]
```

#### 3. State Machine (8 tests)
```python
def test_controller_starts_in_init():
    """Controller starts in INIT state."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    assert controller.state == ControlState.INIT

def test_controller_transitions_to_tracking():
    """INIT → TRACKING when step called."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.1]))
    
    q = np.array([0.0, 0.3, -0.3])
    qdot = np.zeros(3)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    controller.step(q, qdot, rgb, "reach target")
    assert controller.state == ControlState.TRACKING

def test_controller_goal_reached_detection():
    """TRACKING → GOAL_REACHED when position close to target."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    
    q_far = np.array([0.0, 0.3, -0.3])
    q_near = np.array([0.01, 0.31, -0.29])  # Very close to subgoal
    
    controller.trajectory_buffer.update_subgoal(q_near)
    qdot = np.zeros(3)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # First step: at q_far, should be TRACKING
    controller.step(q_far, qdot, rgb, "test")
    assert controller.state == ControlState.TRACKING
    
    # Step at q_near: should detect goal reached
    for i in range(10):  # May need multiple steps for convergence
        controller.step(q_near, qdot, rgb, "test")
    
    assert controller.state == ControlState.GOAL_REACHED, \
        "Should reach GOAL_REACHED when close to subgoal"

def test_controller_error_on_invalid_state():
    """ERROR state set on exception."""
    controller = DualSystemController(
        mpc_solver=None,  # Will fail
        smolvla_client=vla_client,
        trajectory_buffer=traj_buf
    )
    
    q = np.array([0.0, 0.3, -0.3])
    qdot = np.zeros(3)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    controller.step(q, qdot, rgb, "test")
    assert controller.state == ControlState.ERROR

def test_controller_state_transitions_logged():
    """State transitions are logged at INFO level."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    
    # Capture log output
    import logging
    with caplog.at_level(logging.INFO):
        controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.1]))
        q = np.array([0.0, 0.3, -0.3])
        qdot = np.zeros(3)
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        controller.step(q, qdot, rgb, "test")
    
    # Should see state transition log
    assert "State transition" in caplog.text or "INIT" in caplog.text

def test_controller_reset_clears_state():
    """reset() returns to INIT state."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.1]))
    
    q = np.array([0.0, 0.3, -0.3])
    qdot = np.zeros(3)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    controller.step(q, qdot, rgb, "test")
    assert controller.state != ControlState.INIT
    
    controller.reset()
    assert controller.state == ControlState.INIT
    assert controller.step_count == 0
    assert len(controller.step_times_ms) == 0
```

#### 4. Reference Trajectory Integration (3 tests)
```python
def test_controller_uses_buffer_reference():
    """Controller uses TrajectoryBuffer reference trajectory."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    
    q_start = np.array([0.0, 0.3, -0.3])
    q_goal = np.array([0.2, 0.3, -0.1])
    
    controller.trajectory_buffer.update_subgoal(q_goal)
    qdot = np.zeros(3)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Get reference trajectory from buffer
    q_ref, qdot_ref = controller.trajectory_buffer.get_reference_trajectory(
        q_start, N=10, dt=0.01
    )
    
    # Now step controller
    tau = controller.step(q_start, qdot, rgb, "test")
    
    # Should return valid torque (reference was used)
    assert isinstance(tau, np.ndarray)
    assert tau.shape == (3,)

def test_controller_handles_hold_position():
    """When no subgoal set, controller holds position."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    # traj_buf has no subgoal yet (INIT state)
    
    q = np.array([0.0, 0.3, -0.3])
    qdot = np.zeros(3)
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    tau = controller.step(q, qdot, rgb, "test")
    
    # Should still return valid torque (hold position)
    assert isinstance(tau, np.ndarray)
    assert tau.shape == (3,)
```

#### 5. Non-blocking Verification (2 tests)
```python
def test_controller_doesnt_call_vla_client_in_step():
    """Main step() never calls SmolVLAClient (async)."""
    # This test verifies the critical invariant:
    # The main control loop is purely synchronous.
    
    controller = DualSystemController(mpc, vla_client, traj_buf)
    
    # Mock VLA client to track calls
    from unittest.mock import patch
    with patch.object(vla_client, 'query_action') as mock_query:
        controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.1]))
        
        q = np.array([0.0, 0.3, -0.3])
        qdot = np.zeros(3)
        rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        controller.step(q, qdot, rgb, "test")
        
        # VLA client should NOT be called
        mock_query.assert_not_called()

def test_controller_doesnt_await_in_step():
    """Main step() is synchronous (no await)."""
    controller = DualSystemController(mpc, vla_client, traj_buf)
    import inspect
    
    # Verify step() is a regular function, not async
    assert not inspect.iscoroutinefunction(controller.step), \
        "step() must be synchronous, not async"
```

---

## Integration with MPC Solver

The `step()` method calls:
```python
tau = self.mpc_solver.solve(
    x_curr=x_curr,     # [6] current state
    x_ref=x_ref,       # [3] terminal reference
    q_ref=q_ref,       # [N, 3] reference trajectory
    qdot_ref=qdot_ref  # [N, 3] reference velocities
)
```

**Assumption:** Existing MPC solver has a compatible interface:
```python
def solve(self, x_curr, x_ref, q_ref=None, qdot_ref=None) -> np.ndarray:
    """Return optimal [3] torque vector."""
```

If interface differs, adapt in `step()` method.

---

## Next Task: Task 4 (Background VLA Thread)

Task 4 will implement the background thread that:
1. Runs asyncio event loop in separate thread
2. Polls VLA every 200ms
3. Updates TrajectoryBuffer with new subgoal
4. Never blocks main control loop

**Dependency:** Task 3 ✓

---

## Files to Create/Modify

- **Create:** `src/integration/__init__.py`
- **Create:** `src/integration/dual_system_controller.py` (this implementation)
- **Create:** `tests/test_dual_system_controller.py` (all 20 tests above)

---

## Success Criteria (Definition of Done)

✅ All 20 unit tests pass  
✅ Step timing consistently < 20ms (measure over 100 steps)  
✅ State machine transitions logged correctly  
✅ No async calls in step() method  
✅ Graceful error handling (return zero torque on failure)  
✅ Statistics tracking works (get_stats() returns valid dict)  
✅ Reset functionality works (clear all state)  

---

## Implementation Checklist

- [ ] Create `src/integration/__init__.py`
- [ ] Implement `ControlState` enum
- [ ] Implement `DualSystemController.__init__()`
- [ ] Implement `DualSystemController.step()`
- [ ] Implement `_check_goal_arrival()`
- [ ] Implement `_update_state_machine()`
- [ ] Implement `get_stats()` and `reset()`
- [ ] Create `tests/test_dual_system_controller.py`
- [ ] Write all 20 tests (copy from above)
- [ ] Run pytest and verify all pass
- [ ] Measure timing: ensure < 20ms per step
- [ ] Review code for missing async/await calls
- [ ] Document any deviations from this plan

---

**Ready to implement!**
