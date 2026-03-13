# Phase 8B Implementation Plan
## Dual-System Integration Layer (SmolVLA ↔ MPC)

**Status:** Starting (Phase 8A ✓ complete)  
**Date:** 13 March 2026  
**Estimated Duration:** 5-7 hours  
**Approach:** Test-driven development with strict validation at each task

---

## System Overview

### Critical Invariant ⚠️
**The MPC loop MUST run at 100-500 Hz regardless of SmolVLA status. VLA queries never block System 1.**

Architecture:
```
Background Thread (asyncio):        Main Thread (Sync):
  SmolVLAClient                      DualSystemController.step()
    ├─ Query VLA (1-5 Hz)            ├─ Get reference trajectory (instant)
    ├─ HTTP POST to Colab            ├─ Compute dynamics (instant)
    ├─ Timeout 1-2s (non-blocking)   ├─ Build + solve QP (20-50ms)
    └─ Update TrajectoryBuffer       ├─ Check state machine
       (thread-safe, lock-free)       └─ Return τ command (real-time)
```

### Key Design Decisions (From Spec)
- **VLA Model:** SmolVLA 450M (LeRobot pretrained, running on Colab T4)
- **Communication:** FastAPI + ngrok HTTPS tunnel
- **Query Rate:** 1-5 Hz (200ms interval), latency ~700ms acceptable
- **Action Interface:** EE Cartesian subgoal [x, y, z] + grasp ∈ [0,1]
- **Thread Safety:** TrajectoryBuffer uses GIL for atomic numpy ops; no explicit locks
- **Graceful Fallback:** Hold last known subgoal if VLA timeout; continue MPC

---

## Tasks (Sequential, Each Fully Tested)

### TASK 1: SmolVLAClient (Async HTTP Interface)
**Duration:** 1.5-2 hours  
**File:** `src/smolvla_client/async_client.py`  
**Dependencies:** aiohttp, pydantic, pillow, numpy

#### 1.1 Interface Design
```python
@dataclass
class SmolVLAResponse:
    action_chunk: np.ndarray     # [chunk_size, 7] float
    subgoal_xyz: np.ndarray      # [3] float — EE target delta
    latency_ms: float            # query time
    timestamp: float             # wall time

class SmolVLAClient:
    async def query_action(
        self,
        rgb: np.ndarray,        # [H, W, 3] uint8
        instruction: str,       # natural language task
        current_joints: List[float]
    ) -> SmolVLAResponse | None:  # None on timeout/error
```

#### 1.2 Core Responsibilities
- [ ] Encode RGB image [H,W,3] uint8 → base64 JPEG
- [ ] POST to `{endpoint}/predict` with aiohttp (async)
- [ ] Handle timeouts gracefully (1-2s deadline)
- [ ] Parse JSON response → SmolVLAResponse dataclass
- [ ] Log latency at DEBUG level
- [ ] Return None on any failure (never raise exceptions)
- [ ] Connection pooling (reuse aiohttp.ClientSession)

#### 1.3 Critical Implementation Notes
- **Image encoding:** Resize to 224×224, JPEG quality 85 for bandwidth
- **Timeout handling:** `aiohttp.ClientTimeout(total=2.0)` with async context manager
- **Thread safety:** Use lock for latest_response if accessed across threads
- **No blocking:** All I/O is async; CPU work (base64, PIL) is negligible

#### 1.4 Unit Tests (`tests/test_smolvla_client.py`)

**Test 1: Health Check**
```python
@pytest.mark.asyncio
async def test_client_health_endpoint():
    """Verify /health endpoint reachable through ngrok."""
    client = SmolVLAClient(endpoint_url=NGROK_URL)
    resp = await client.health_check()
    assert resp == {"status": "ok", "model": "smolvla-base"}
```

**Test 2: Single Inference Query**
```python
@pytest.mark.asyncio
async def test_client_single_query():
    """Single RGB input → action_chunk response."""
    client = SmolVLAClient(endpoint_url=NGROK_URL)
    
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    response = await client.query_action(
        rgb, "pick up the object", [0.0, 0.3, -0.5]
    )
    
    assert response is not None
    assert response.action_chunk.shape[1] == 7  # [chunk, 7]
    assert response.subgoal_xyz.shape == (3,)
    assert response.latency_ms > 0
    assert response.timestamp > 0
```

**Test 3: Timeout Handling**
```python
@pytest.mark.asyncio
async def test_client_timeout():
    """Query to unreachable endpoint returns None, doesn't crash."""
    client = SmolVLAClient(
        endpoint_url="http://unreachable-xyz.invalid",
        timeout_s=0.1
    )
    
    rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    response = await client.query_action(rgb, "test", [0, 0, 0])
    
    # Should return None, not raise
    assert response is None
```

**Test 4: Image Encoding Correctness**
```python
def test_client_image_encoding():
    """Verify RGB array → base64 JPEG → PIL Image is lossless enough."""
    client = SmolVLAClient(endpoint_url="http://dummy")
    
    # Create known pattern
    rgb_orig = np.zeros((224, 224, 3), dtype=np.uint8)
    rgb_orig[:112, :, 0] = 255  # red top half
    
    # Encode → decode
    b64 = client._encode_image(rgb_orig)
    rgb_decoded = client._decode_image(b64)
    
    # Lossy JPEG, but overall structure preserved
    mse = np.mean((rgb_orig.astype(float) - rgb_decoded.astype(float))**2)
    assert mse < 100  # JPEG artifacts acceptable
```

#### 1.5 Validation Criteria
- ✅ All 4 unit tests pass
- ✅ Latency measurement accurate (wall clock vs actual HTTP RTT)
- ✅ No exceptions raised on network errors
- ✅ Memory stable (no leaks during 100 queries)

---

### TASK 2: TrajectoryBuffer (Reference Generator)
**Duration:** 1.5-2 hours  
**File:** `src/smolvla_client/trajectory_buffer.py`  
**Dependencies:** numpy, scipy.interpolate

#### 2.1 Responsibilities
- [ ] Store latest subgoal in joint space [3]
- [ ] Generate smooth reference trajectory for MPC horizon
- [ ] Interpolate from current position to subgoal using quintic spline
- [ ] Detect goal arrival (threshold-based)
- [ ] Thread-safe (GIL-based atomic reads)

#### 2.2 Core Methods
```python
class TrajectoryBuffer:
    def update_subgoal(self, q_goal: np.ndarray):
        """Called by System 2 when VLA returns new EE target."""
        # Convert EE target to joint space (via IK from dynamics module)
        # Store as current_subgoal_q
    
    def get_reference_trajectory(self, q_current, N: int, dt: float):
        """Return smooth reference for MPC horizon."""
        # Return: (q_ref [N, 3], qdot_ref [N, 3])
    
    def check_arrival(self, q_current) -> bool:
        """Detect when current position reaches subgoal.
        
        Returns True when ||q - q_goal|| < threshold_rad.
        Triggers next VLA query when True.
        """
```

#### 2.3 Interpolation Strategy
Use **quintic spline** (5th order polynomial) for smooth, continuous trajectories:

```
q(τ) = a₅τ⁵ + a₄τ⁴ + a₃τ³ + a₂τ² + a₁τ + a₀

Boundary conditions:
  q(0) = q₀, q̇(0) = 0
  q(1) = qf, q̇(1) = 0
  q̈(0) = q̈(1) = 0

Solve for [a₀, a₁, ..., a₅] subject to these constraints.
```

For each joint independently:
```python
from scipy.interpolate import CubicSpline

# Or use closed-form quintic:
def quintic_spline(q0, qf, T, t):
    """Quintic spline from q0 to qf over time T."""
    tau = t / T
    h00 = 1 - 10*tau**3 + 15*tau**4 - 6*tau**5
    h10 = tau - 6*tau**3 + 8*tau**4 - 3*tau**5
    h01 = 10*tau**3 - 15*tau**4 + 6*tau**5
    h11 = -4*tau**3 + 7*tau**4 - 3*tau**5
    return h00*q0 + h10*0 + h01*qf + h11*T*0
```

#### 2.4 Unit Tests (`tests/test_trajectory_buffer.py`)

**Test 1: Interpolation Correctness**
```python
def test_buffer_quintic_interpolation():
    """Verify smooth quintic spline between two joint positions."""
    buf = TrajectoryBuffer()
    q0 = np.array([0.0, 0.5, -0.3])
    qf = np.array([0.5, 0.2, 0.1])
    
    buf.update_subgoal(qf)
    q_ref, qdot_ref = buf.get_reference_trajectory(q0, N=10, dt=0.01)
    
    # Check boundary conditions
    assert np.allclose(q_ref[0], q0, atol=0.01), "start pos should be q0"
    assert np.allclose(q_ref[-1], qf, atol=0.01), "end pos should be qf"
    
    # Check smoothness (continuous ẋ)
    assert np.all(np.abs(qdot_ref) < 1.0), "velocities reasonable"
    
    # Check that path is monotonic (no oscillation)
    for j in range(3):
        diffs = np.diff(q_ref[:, j])
        assert np.all(diffs[:-1] * diffs[1:] >= 0), f"joint {j} should be monotonic"
```

**Test 2: Goal Detection**
```python
def test_buffer_goal_detection():
    """Detect when position reaches subgoal within threshold."""
    buf = TrajectoryBuffer(arrival_threshold_rad=0.05)
    
    q_curr = np.array([0.0, 0.0, 0.0])
    buf.update_subgoal(np.array([0.04, 0.03, 0.02]))  # Close to q_curr
    
    # First check should be True (goal is close)
    assert buf.check_arrival(q_curr) == True
```

**Test 3: Hold Position When No Goal**
```python
def test_buffer_hold_position_no_goal():
    """If no subgoal set, return hold-position trajectory."""
    buf = TrajectoryBuffer()
    q_curr = np.array([0.5, 0.2, -0.1])
    
    q_ref, qdot_ref = buf.get_reference_trajectory(q_curr, N=10, dt=0.01)
    
    # All positions should be q_curr
    assert np.allclose(q_ref, q_curr, atol=1e-6)
    # All velocities should be zero
    assert np.allclose(qdot_ref, 0, atol=1e-6)
```

#### 2.5 Validation Criteria
- ✅ All 3 unit tests pass
- ✅ Interpolation is smooth (continuous derivatives)
- ✅ Goal detection threshold works correctly
- ✅ Thread-safe (numpy GIL protects concurrent reads)

---

### TASK 3: DualSystemController (State Machine + Main Loop)
**Duration:** 2-2.5 hours  
**File:** `src/integration/dual_system_controller.py`  
**Dependencies:** numpy, enum, logging

#### 3.1 State Machine
```python
class ControlState(enum.Enum):
    INIT = 0          # System booting, waiting for first VLA query
    WAITING_VLA = 1   # Fired async query, waiting for response
    TRACKING = 2      # MPC executing toward current subgoal
    GOAL_REACHED = 3  # At target, ready for next task
    ERROR = 4         # Unrecoverable error (IK failed, etc.)
```

**State Transitions:**
```
  INIT ──request_subgoal──> WAITING_VLA
                                  │
                        (timeout 2s) ↓ (response)
                                  │
                            TRACKING ←─┐
                                  │     │
                          (arrival)→ GOAL_REACHED
                                  │
                        (new subgoal) →─┘
```

#### 3.2 Main Synchronous Interface
```python
class DualSystemController:
    def step(self, q: np.ndarray, qdot: np.ndarray, rgb: np.ndarray,
             instruction: str) -> np.ndarray:
        """
        Main control loop step (SYNCHRONOUS, ~100 Hz).
        
        NEVER blocks, NEVER calls async, NEVER waits on network.
        Called from simulation step in main thread.
        
        Returns: τ ∈ ℝ³ — torque command
        """
        # 1. Get reference trajectory (instant read from TrajectoryBuffer)
        # 2. Compute dynamics matrices (M, C, G)
        # 3. Build QP
        # 4. Solve with SL oscillator
        # 5. Log & transition state machine
        # 6. Return τ
```

#### 3.3 Async Background Loop
```python
async def poll_vla_background(
    controller: DualSystemController,
    interval_s: float = 0.2
):
    """
    Background asyncio coroutine (separate thread).
    Fires VLA queries every 200ms (non-blocking).
    Never blocks main MPC loop.
    """
    await controller.vla_client.start()
    
    while controller.running:
        q = controller.env.get_joint_positions()
        
        if controller.trajectory_buffer.check_arrival(q):
            controller.state = ControlState.WAITING_VLA
            
            # Query VLA (async, 1-2s timeout)
            rgb = controller.env.render_rgb(224, 224)
            response = await controller.vla_client.query_action(
                rgb, controller.instruction, q.tolist()
            )
            
            if response is not None:
                # Convert EE target to joint space (IK)
                ee_goal = controller.env.current_ee_pos() + response.subgoal_xyz
                try:
                    q_goal = controller.env.inverse_kinematics(ee_goal)
                    controller.trajectory_buffer.update_subgoal(q_goal)
                    controller.state = ControlState.TRACKING
                except ValueError:
                    logger.warning("IK failed, holding last trajectory")
                    controller.state = ControlState.ERROR
            else:
                # Timeout — hold last subgoal
                logger.warning("VLA timeout, holding trajectory")
        
        await asyncio.sleep(interval_s)
```

#### 3.4 Unit Tests (`tests/test_dual_system_controller.py`)

**Test 1: Synchronous Step Timing**
```python
def test_controller_step_timing():
    """MPC step completes in < 20ms (target ~10ms for 100 Hz)."""
    controller = DualSystemController(config)
    controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.2]))
    
    q = np.array([0.0, 0.3, -0.3])
    qdot = np.array([0.0, 0.0, 0.0])
    rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    import time
    t0 = time.perf_counter()
    tau = controller.step(q, qdot, rgb, "reach target")
    elapsed = (time.perf_counter() - t0) * 1000
    
    assert elapsed < 20, f"Step too slow: {elapsed:.1f}ms"
    assert tau.shape == (3,)
```

**Test 2: State Machine Transitions**
```python
def test_controller_state_transitions():
    """Verify correct state transitions."""
    controller = DualSystemController(config)
    
    assert controller.state == ControlState.INIT
    
    # Manually fire transition
    controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.2]))
    q = np.array([0.0, 0.3, -0.3])
    
    # After step, should still be tracking
    tau = controller.step(q, np.zeros(3), np.zeros((640, 480, 3), dtype=np.uint8), "test")
    assert controller.state in [ControlState.TRACKING, ControlState.WAITING_VLA]
```

**Test 3: Non-blocking Verification**
```python
@pytest.mark.asyncio
async def test_controller_async_non_blocking():
    """Verify System 1 (sync) is unaffected by System 2 (async) delay.
    
    Mock VLA server with intentional 500ms delay.
    Measure MPC loop timing variance: should be < 10%.
    """
    # [Implementation in integration tests, not unit test]
```

#### 3.5 Validation Criteria
- ✅ All 3 unit tests pass
- ✅ Step timing < 20ms consistently
- ✅ State transitions logged correctly
- ✅ No asyncio calls in step method
- ✅ No blocking on TrajectoryBuffer reads

---

### TASK 4: VLA Query Background Thread
**Duration:** 1-1.5 hours  
**File:** `src/integration/vla_query_thread.py`  
**Dependencies:** threading, asyncio

#### 4.1 Entry Point
```python
def poll_vla_background(
    controller: DualSystemController,
    poll_interval_s: float = 0.2
):
    """
    Entry point for background thread.
    Runs asyncio event loop in separate thread.
    Non-blocking from main thread perspective.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(
        controller.poll_vla_background_async(poll_interval_s)
    )
```

#### 4.2 Threading Model
```python
# In main script:
controller = DualSystemController(config)
controller.running = True

# Start background thread
vla_thread = threading.Thread(
    target=poll_vla_background,
    args=(controller, 0.2),  # 5 Hz poll rate
    daemon=True
)
vla_thread.start()

# Main loop (unaffected by vla_thread)
for step in range(max_steps):
    tau = controller.step(q, qdot, rgb, instruction)
    env.step(tau)
    
    # Measure loop frequency
    elapsed = measure_step_time()
    if elapsed > 15:  # target 10ms for 100 Hz
        logger.warning(f"MPC step slow: {elapsed:.1f}ms")
```

#### 4.3 Unit Tests (`tests/test_vla_thread.py`)

**Test 1: Thread Runs Continuously**
```python
def test_vla_thread_continuous():
    """Background thread runs without crashing for 1 minute."""
    controller = DualSystemController(config)
    controller.running = True
    
    vla_thread = threading.Thread(
        target=poll_vla_background,
        args=(controller, 0.5),  # 2 Hz for faster test
        daemon=True
    )
    vla_thread.start()
    
    import time
    time.sleep(5)  # Let it run
    
    controller.running = False
    vla_thread.join(timeout=2)
    
    assert not vla_thread.is_alive(), "Thread should exit cleanly"
```

**Test 2: Thread Doesn't Block Main**
```python
def test_vla_thread_non_blocking():
    """MPC loop unaffected by slow VLA responses."""
    # [Covered in integration tests]
```

#### 4.4 Validation Criteria
- ✅ Thread starts/stops cleanly
- ✅ No deadlocks or resource leaks
- ✅ Graceful handling of VLA timeouts (doesn't crash)

---

### TASK 5: Integration Tests (Async + Non-Blocking Verification)
**Duration:** 1.5-2 hours  
**File:** `tests/test_integration_phase8b.py`  
**Dependencies:** pytest, pytest-asyncio, unittest.mock

#### 5.1 Test 1: Mock VLA Server with Controlled Latency
```python
@pytest.mark.asyncio
async def test_vla_latency_doesnt_block_mpc():
    """
    Mock VLA with 500ms latency.
    Verify MPC loop timing variance < 10%.
    """
    # Create mock FastAPI app
    from fastapi.testclient import TestClient
    from unittest.mock import AsyncMock
    
    # Mock VLA response with 500ms delay
    async def slow_predict(request):
        await asyncio.sleep(0.5)
        return {
            "action_chunk": [[0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5]],
            "subgoal_xyz": [0.1, 0.2, 0.3],
            "latency_ms": 500,
            "status": "ok"
        }
    
    # Run MPC loop with mock
    controller = DualSystemController(config_with_mock_vla)
    
    # Measure 20 steps
    timings = []
    for _ in range(20):
        t0 = time.perf_counter()
        tau = controller.step(q, qdot, rgb, "test")
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
    
    # Compute statistics
    mean_time = np.mean(timings)
    std_time = np.std(timings)
    variance = std_time / mean_time  # coefficient of variation
    
    assert variance < 0.10, f"Timing variance {variance:.2%} > 10%"
    assert mean_time < 0.020, f"Mean step time {mean_time*1000:.1f}ms > 20ms"
```

#### 5.2 Test 2: Graceful Fallback on VLA Timeout
```python
@pytest.mark.asyncio
async def test_graceful_fallback_vla_timeout():
    """
    VLA server times out.
    System should return None, controller holds last trajectory.
    """
    controller = DualSystemController(config_with_unreachable_vla)
    
    q = np.array([0.0, 0.3, -0.3])
    qdot = np.zeros(3)
    rgb = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Last subgoal set
    controller.trajectory_buffer.update_subgoal(np.array([0.2, 0.3, -0.2]))
    
    # VLA times out, but system continues
    tau = controller.step(q, qdot, rgb, "reach target")
    
    # Should still have valid torque command
    assert tau is not None and not np.any(np.isnan(tau))
    # State should be ERROR or still TRACKING (holding trajectory)
    assert controller.state in [ControlState.TRACKING, ControlState.ERROR]
```

#### 5.3 Test 3: Full Pointing Task with Mock VLA
```python
@pytest.mark.asyncio
async def test_pointing_task_end_to_end():
    """
    Complete pointing task:
    1. Request subgoal from VLA
    2. MPC tracks to it
    3. Detect arrival
    4. Request next subgoal
    5. Complete in < 30 seconds
    """
    # Setup mock VLA that returns sequence of subgoals
    mock_vla_responses = [
        np.array([0.2, 0.3, -0.2]),
        np.array([0.0, 0.5, -0.1]),
        np.array([0.1, 0.2, 0.0]),
    ]
    response_idx = [0]
    
    async def mock_query(rgb, instruction, joints):
        resp = SmolVLAResponse(
            action_chunk=np.random.randn(10, 7),
            subgoal_xyz=mock_vla_responses[response_idx[0]] if response_idx[0] < len(mock_vla_responses) else None,
            latency_ms=100,
            timestamp=time.time()
        )
        response_idx[0] += 1
        return resp
    
    controller = DualSystemController(config)
    controller.vla_client.query_action = mock_query
    controller.running = True
    
    # Run for 30 seconds simulated time
    t_start = time.time()
    while time.time() - t_start < 30:
        tau = controller.step(q, qdot, rgb, "point to targets")
        # Simulate: q = q + qdot * dt, etc.
        q += np.random.randn(3) * 0.01  # Simulate reaching
    
    # Should have requested at least 2 subgoals
    assert response_idx[0] >= 2, "Multiple subgoals should be requested"
```

#### 5.4 Validation Criteria (Gate 4b)
- ✅ Timing variance test passes (< 10%)
- ✅ Graceful fallback test passes (no crash on VLA timeout)
- ✅ Full E2E test passes (task completes, multiple subgoals)

---

## Testing Strategy (Test-Driven Development)

### Order of Implementation
1. **Define interfaces** (dataclasses, method signatures)
2. **Write tests** (before implementation)
3. **Implement** (make tests pass)
4. **Validate** (run full test suite, check properties)
5. **Move to next task**

### Test Execution Workflow
```bash
# Task 1: SmolVLAClient
pytest tests/test_smolvla_client.py -v
pytest tests/test_smolvla_client.py::test_client_health_endpoint -v  # single test

# Task 2: TrajectoryBuffer
pytest tests/test_trajectory_buffer.py -v

# Task 3: DualSystemController
pytest tests/test_dual_system_controller.py -v

# Task 4 & 5: Integration tests
pytest tests/test_integration_phase8b.py -v
pytest tests/test_integration_phase8b.py::test_vla_latency_doesnt_block_mpc -v

# Full suite
pytest tests/ -v --tb=short
```

### Measurement & Validation

**Latency Measurements:** Use `time.perf_counter()` for sub-millisecond precision
```python
t0 = time.perf_counter()
result = function()
latency_ms = (time.perf_counter() - t0) * 1000
```

**Timing Histogram:** Track 100+ iterations
```python
timings = []
for _ in range(100):
    timings.append(measure_step_time())

print(f"Mean: {np.mean(timings):.2f}ms")
print(f"Median: {np.median(timings):.2f}ms")
print(f"95th percentile: {np.percentile(timings, 95):.2f}ms")
print(f"Std dev: {np.std(timings):.2f}ms")
```

---

## Files to Create

```
src/smolvla_client/
├── __init__.py
├── async_client.py          (Task 1)
├── trajectory_buffer.py     (Task 2)
└── tests/
    ├── __init__.py
    └── test_smolvla_client.py
    └── test_trajectory_buffer.py

src/integration/
├── __init__.py
├── dual_system_controller.py  (Task 3)
└── vla_query_thread.py        (Task 4)

tests/
├── test_integration_phase8b.py  (Task 5)
```

---

## Assumptions & Open Questions

### Assumptions
1. **Colab notebook is running:** FastAPI server is live, ngrok URL endpoint is known
2. **MPC solver works:** existing `src/solver/` code is functional
3. **Dynamics module exists:** `src/dynamics/` with FK, IK, M, C, G implemented
4. **MuJoCo environment configured:** `src/simulation/mujoco_env.py` provides state/control interface
5. **Thread safety via GIL:** NumPy array reads are atomic under Python GIL; no explicit locks needed for TrajectoryBuffer

### Open Questions (Clarify Before Proceeding)
1. **ngrok URL source:** Will Colab notebook output URL be copy-pasted into config, or read from file?
   - **Decision:** Start with manual config/smolvla_config.yaml with placeholder
2. **Image resolution:** Colab expects 224×224. What if simulation renders 480×640?
   - **Decision:** Resize in client.encode_image()
3. **IK failure mode:** What if inverse kinematics cannot reach EE target?
   - **Decision:** Log warning, hold last valid q, set state to ERROR
4. **Timeout duration:** 1s or 2s for VLA queries?
   - **Decision:** Start with 1s (acceptable latency), tune later

---

## Success Criteria (Definition of Done for Phase 8B)

✅ **SmolVLAClient**
- All unit tests pass
- Can successfully query Colab server
- Gracefully handles timeouts

✅ **TrajectoryBuffer**
- All unit tests pass
- Quintic spline interpolation is smooth
- Goal detection threshold works

✅ **DualSystemController**
- All unit tests pass
- Step timing < 20ms
- State machine transitions correct

✅ **VLA Query Thread**
- Runs in background without blocking
- Can survive > 1 minute continuous operation

✅ **Integration (Gate 4b)**
- MPC loop timing variance < 10% during VLA queries
- Graceful fallback when VLA timeout
- Full E2E task completes successfully

✅ **Code Quality**
- All methods have full type hints
- All docstrings include units and shapes
- No magic numbers (use config.yaml)
- Logging at appropriate levels (DEBUG, INFO, WARNING)

---

## Rollout After Phase 8B

Once Gate 4b ✓, proceed to:

**Phase 9:** Full system integration tests (E2E picking, multi-task)  
**Phase 10:** Observability dashboard, structured logging  

**Target:** Complete Phases 8B-10 by **17-18 March 2026**

---

**Status:** Plan complete. Ready for Task 1 implementation.
