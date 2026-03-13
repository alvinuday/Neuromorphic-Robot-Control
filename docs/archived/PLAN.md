# Implementation Plan: 3-DOF Spatial Arm + SmolVLA Integration
**Neuromorphic Robot Control — Phase 7-10**
**Created:** 13 March 2026

---

## Executive Summary

Extend the existing 2-DOF planar arm + Stuart-Landau oscillator MPC system (Phases 1-6 complete, 25 tests passing) to a **3-DOF spatial RRR arm** with **SmolVLA vision-language task intelligence**. Use a hybrid GPU strategy (local MPC first, fallback to Colab GPU if needed), strict gate-by-gate validation, and async architecture to prevent VLA queries from blocking the fast control loop.

**Total scope:** ~40 hours over 5-6 working days at 6-8 hrs/day.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         SYSTEM 2  (Slow, Deliberate)                         │
│                      SmolVLA — 450M param VLA (Colab)                        │
│                                                                              │
│  Input:  RGB frame (224×224) + language instruction + joint state            │
│  Output: EE target [x_goal, y_goal, z_goal] + grasp mode                    │
│  Rate:   1–5 Hz (async, does NOT block fast control)                         │
│  Where:  Google Colab T4 GPU → FastAPI → ngrok HTTPS tunnel → local         │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │  subgoal  (HTTP/2, ~200ms latency OK)
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 TRAJECTORY BUFFER (Local, non-blocking)                       │
│  Holds latest subgoal; interpolates reference trajectory for MPC             │
│  Graceful fallback if VLA times out                                          │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │  reference trajectory x_ref(t)
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                    SYSTEM 1  (Fast, Reactive)                                 │
│         Stuart-Landau Oscillator Network — 3-DOF QP Solver                   │
│                                                                              │
│  Input:  Current joint state q, q̇ + reference trajectory + constraints      │
│  Solving: MPC linearized around trajectory (N-step horizon)                  │
│  Output: Optimal torque command τ* for all 3 joints                          │
│  Rate:   100–500 Hz (consistent regardless of System 2 status)               │
│  Where:  Local machine (NumPy/JAX), fallback to Colab GPU if needed         │
└───────────────────────────┬──────────────────────────────────────────────────┘
                            │  τ*(t) command
                            ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                   MuJoCo 3-DOF Simulation / Real Arm                          │
│  RRR arm with M(q), C(q,q̇), G(q) dynamics fully modeled                     │
│  State feedback: q, q̇, EE position at control rate                           │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Design Decisions (Confirmed)

✅ **GPU Strategy:** Hybrid — local MPC first (100+ Hz target), fallback to Colab GPU if rate drops  
✅ **Action Space:** SmolVLA outputs EE Cartesian [x, y, z] + grasp signal; MPC does inverse kinematics  
✅ **Validation Approach:** Strict gate-by-gate testing (5 gates, each must pass before proceeding)

---

## Implementation Phases (Sequential Dependencies)

### PHASE 7A: Dynamics Extension (3-DOF Arm)
**Duration:** 6-8 hours  
**Generates:** New `src/dynamics/` module for 3-DOF kinematics & dynamics

**Deliverables:**
1. **Kinematics** (`src/dynamics/kinematics_3dof.py`)
   - Forward kinematics using DH parameters (Section 2-3 of techspec)
   - Jacobian J(q) ∈ ℝ⁶ˣ³ (geometric, Section 4)
   - Jacobian time-derivative J̇(q,q̇)
   - Inverse kinematics (damped pseudo-inverse for singularity robustness)
   
2. **Lagrangian Dynamics** (`src/dynamics/lagrangian_3dof.py`)
   - Mass matrix M(q): block-diagonal structure, PD + symmetric
   - Coriolis/centrifugal C(q,q̇) via Christoffel symbols
   - Gravity vector G(q): note G[0]=0 (azimuth joint decoupling)
   - Friction and damping terms for simulation realism
   
3. **MuJoCo Model** (`assets/arm3dof.xml`)
   - 3 revolute joints: azimuth (base), shoulder, elbow
   - Realistic masses + inertia from arm parameters
   - Sensors: joint positions/velocities, EE site at wrist
   - Cameras: overhead (for SmolVLA RGB 224×224) + side view
   - Validate: FK matches MuJoCo site positions (< 1mm error)

**Gate 1 Validation (MUST PASS before Phase 7B):**
- [ ] `test_3dof_home_position` — FK at q=[0,0,0] correct
- [ ] `test_3dof_positive_definite` — M(q) > 0 on 20 random configs
- [ ] `test_3dof_symmetric_M` — M(q) = M(q)ᵀ
- [ ] `test_3dof_block_structure` — M[0,1] = M[0,2] = 0
- [ ] `test_3dof_G_decoupling` — G[0] = 0 always
- [ ] `test_3dof_skew_symmetry` — (Ṁ - 2C) skew-symmetric (passivity)
- [ ] `test_3dof_energy_conservation` — G = ∇V
- [ ] `test_3dof_mujoco_inverse_dynamics` — < 5% error vs mujoco.mj_inverse()

---

### PHASE 7B: MPC Extension (3-DOF QP)
**Duration:** 4-6 hours  
**Depends on:** Phase 7A (Gate 1 ✓)  
**Generates:** `src/mpc/` module for linearization & QP construction

**Deliverables:**
1. **Linearization** (`src/mpc/linearize_3dof.py`)
   - Analytical state-space matrices A(t) ∈ ℝ⁶ˣ⁶, B(t) ∈ ℝ⁶ˣ³
   - Use JAX `jax.jacobian()` for automatic differentiation (complex A₂₁ term)
   - Zero-Order Hold (ZOH) discretization: Aₖ ≈ I + AΔt + A²(Δt)²/2
   - Test: one-step linearized model vs full nonlinear MuJoCo (O(Δt²) error expected)

2. **3-DOF QP Builder** (`src/mpc/qp_builder_3dof.py`)
   - Batch prediction matrices Sx ∈ ℝ⁶ᴺˣ⁶, Su ∈ ℝ⁶ᴺˣ³ᴺ
   - Hessian H = SuᵀQ̄Su + R̄ (now 9N×9N for N_var=3N)
   - Linear term c = SuᵀQ̄(Sx·x₀ - X_ref)
   - Constraint: torque limits, position limits, joint velocity limits
   - **Assert:** H is PSD before solving

3. **Warm-starting**
   - Shift solution from step k to initialize step k+1
   - Expected speedup: 2-3× reduction in solver iterations

**Gate 2 Validation (MUST PASS before Phase 7C):**
- [ ] `test_3dof_linearization_accuracy` — < 2% error one-step prediction
- [ ] `test_3dof_discrete_eigenvalues` — Aₖ eigenvalues |λ| ≤ 1 (stable)
- [ ] `test_3dof_H_is_psd` — Check on 10 random operating points
- [ ] `test_3dof_qp_solve_success` — OSQP solves each QP without infeasibility
- [ ] `test_3dof_warm_start_speedup` — Observed 2-3× reduction in iterations

---

### PHASE 7C: SL Solver Extension (2-DOF → 3-DOF)
**Duration:** 3-4 hours  
**Depends on:** Phase 7B (Gate 2 ✓)  
**Generates:** `src/solver/stuart_landau_3dof.py`

**Deliverables:**
1. **Scale existing solver** (adapt `stuart_landau_lagrange_direct.py`)
   - Network: N_var = 3 × N_horizon oscillators (30 for N=10)
   - Coupling W ← -H (from QP Hessian)
   - Bias b ← -c (from QP linear term)
   - **No new dynamics — same equations, just larger N_var**

2. **Validation vs OSQP**
   - Solve 20 random 3-DOF QPs with both SL and OSQP
   - Compare costs: target < 5% deviation
   - Check constraint satisfaction: both should be feasible

**Gate 3 Validation (MUST PASS before Phase 8B):**
- [ ] `test_3dof_sl_vs_osqp_cost` — < 5% cost deviation on 20 QPs
- [ ] `test_3dof_sl_constraint_satisfaction` — Constraints met
- [ ] `test_3dof_sl_convergence` — Reasonable iteration count (< 1000)
- [ ] `test_3dof_sl_eigenvalue_spectrum` — Couplings scale correctly

---

### PHASE 8A: SmolVLA Colab Deployment
**Duration:** 3-4 hours  
**Parallel work** (no dependencies, but must finish before Phase 8B)  
**Generates:** Colab notebook with FastAPI server + ngrok tunnel

**Deliverables:**
1. **Colab Notebook** (`vla/smolvla_server.ipynb`)
   ```python
   # Install & load
   !pip install "lerobot[smolvla]"
   from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
   policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base").to("cuda").eval()
   
   # FastAPI server
   from fastapi import FastAPI
   app = FastAPI()
   
   @app.post("/health")
   async def health():
       return {"status": "ok", "uptime_seconds": time.time() - start_time}
   
   @app.post("/predict")
   async def predict(rgb: np.ndarray, joints: np.ndarray, instruction: str):
       # Preprocess image + state + language
       # Run policy.select_action(batch)
       # Return denormalized EE target [x,y,z] + grasp ∈ [0,1]
       return {"action": [...], "latency_ms": ...}
   
   # Expose via ngrok
   from pyngrok import ngrok
   ngrok.connect(8000, bind_tls=True)  # HTTPS tunnel
   ```

2. **Input/Output Handling**
   - RGB: 224×224 preprocessing (PIL + torchvision normalize)
   - Joint state: proprioceptive embedding (LeRobot standard)
   - Language: tokenize via LeRobot tokenizer
   - Output: action in velocity/displacement space → denormalize using postprocessor

3. **Error Handling**
   - Graceful timeouts (1-second deadline)
   - Missing request fields → HTTP 400
   - GPU OOM/tensor errors → HTTP 500 with message
   - Always return latency in response

**Gate 4a Validation (MUST PASS before Phase 8B):**
- [ ] `test_colab_model_load` — 450M model loads without OOM on T4
- [ ] `test_colab_single_inference` — Single forward pass works
- [ ] `test_colab_ngrok_tunnel` — External access confirmed from local machine
- [ ] `test_colab_latency` — Per-query latency < 500ms (p50)

---

### PHASE 8B: Local Integration Layer (System 1 ↔ System 2)
**Duration:** 5-7 hours  
**Depends on:** Phase 7C (Gate 3 ✓) + Phase 8A (Gate 4a ✓)  
**Generates:** Complete async integration layer + non-blocking MPC

**Critical Invariant:** *MPC loop runs at 100-500 Hz regardless of VLA status. VLA never blocks main control loop.*

**Deliverables:**
1. **SmolVLA Async Client** (`src/smolvla_client/client.py`)
   ```python
   class SmolVLAClient:
       def __init__(self, ngrok_url, timeout_s=1.0):
           self.session = aiohttp.ClientSession()
           self.url = ngrok_url
       
       async def query_action(self, rgb: np.ndarray, joints: np.ndarray, instruction: str) 
               → SmolVLAResponse | None:
           # Async HTTP POST to Colab server
           # Timeout after 1 second
           # Log latency at DEBUG level
           # Return None on failure (graceful)
   ```
   - Uses `aiohttp` for HTTP/2 connection pooling
   - Timeout: 1 second (VLA may fail gracefully)
   - Logging: DEBUG for every query, WARNING on failures

2. **TrajectoryBuffer** (`src/smolvla_client/trajectory_buffer.py`)
   ```python
   class TrajectoryBuffer:
       def update_subgoal(self, p_goal: np.ndarray, grasp: float):
           """Update latest EE target from VLA"""
           # Store as current goal
       
       def get_reference_trajectory(self, q_curr, N, dt) → (q_ref, qdot_ref):
           """Return reference trajectory for MPC horizon"""
           # Interpolate from current q toward p_goal using IK
           # Return (q_ref [N×3], qdot_ref [N×3])
       
       def is_goal_reached(self, q_curr, tol=0.05) → bool:
           """Check if EE is within tolerance of goal"""
   ```
   - Thread-safe (numpy reads are atomic under GIL)
   - Interpolates via quintic spline toward subgoal
   - Detects goal arrival for state machine transitions

3. **DualSystemController State Machine** (`src/integration/dual_system_controller.py`)
   ```python
   class ControlState(enum.Enum):
       INIT = 0
       VLA_QUERY = 1      # Waiting for VLA response
       TRACKING = 2       # Following trajectory
       GOAL_REACHED = 3   # At target
       ERROR = 4
   
   class DualSystemController:
       def __init__(self, mpc, vla_client, trajectory_buffer, logger):
           self.state = ControlState.INIT
       
       def step(self, q: np.ndarray, qdot: np.ndarray, rgb: np.ndarray, 
                instruction: str) → np.ndarray:
           # Main loop method (NEVER async, called from fast loop)
           # 1. Get reference from trajectory_buffer (instant)
           # 2. Run MPC solver (20-50ms)
           # 3. Check state transitions
           # 4. Return tau command
           # Separately: fire VLA query in background (non-blocking)
   ```
   - Explicit state machine with transitions logged at INFO level
   - State transitions: INIT → VLA_QUERY (fire async) → TRACKING → GOAL_REACHED → VLA_QUERY

4. **Background VLA Query Thread** (`src/integration/vla_query_thread.py`)
   ```python
   def poll_vla_background(vla_client, trajectory_buffer, rgb_buffer, 
                          instruction, interval_s=0.2):
       """Run in background asyncio loop (separate thread)"""
       loop = asyncio.new_event_loop()
       asyncio.set_event_loop(loop)
       
       async def query_loop():
           while running:
               rgb = rgb_buffer.get_latest()  # Lock-free read
               try:
                   action = await vla_client.query_action(rgb, q, instruction)
                   trajectory_buffer.update_subgoal(action.position, action.grasp)
               except asyncio.TimeoutError:
                   logger.warning("VLA query timeout, holding last trajectory")
               await asyncio.sleep(interval_s)
       
       loop.run_until_complete(query_loop())
   
   # In main script:
   vla_thread = threading.Thread(target=poll_vla_background, daemon=True)
   vla_thread.start()
   ```
   - Runs in separate thread + asyncio event loop
   - Polls VLA every 200ms (non-blocking)
   - Main loop never calls `await` or `asyncio.run()`

5. **Integration Tests** (`tests/test_integration_3dof.py`)
   - Mock VLA server (FastAPI TestClient with intentional delays)
   - Verify MPC loop timing is unaffected by slow VLA
   - Verify graceful fallback when VLA times out
   - Verify state machine transitions

**Gate 4b Validation (MUST PASS before Phase 9):**
- [ ] `test_client_health_endpoint` — /health responds
- [ ] `test_client_single_action_query` — Single query works
- [ ] `test_client_timeout_handling` — Returns None on timeout, no crash
- [ ] `test_trajectory_buffer_interpolation` — Quintic spline correct
- [ ] `test_dual_system_non_blocking` — MPC loop timing variance < 10% during VLA query
- [ ] `test_dual_system_graceful_failure` — System continues if VLA down

---

### PHASE 9: System Integration & Testing
**Duration:** 8-10 hours  
**Depends on:** Phase 8B (Gate 4b ✓)  
**Generates:** 50+ tests, all passing; end-to-end validation

**Deliverables:**
1. **E2E Test Suite** (`tests/test_e2e_3dof.py`)
   - Point-to-point reaching: send target EE position, verify arm reaches within tolerance
   - Motion profile: verify smooth trajectories (no jerks)
   - Constraint enforcement: torque limits, joint limits never violated
   - Graceful handling: VLA timeouts, IK singularities, solver non-convergence

2. **Performance Tests** (`tests/test_performance_3dof.py`)
   - Control loop frequency: ≥ 100 Hz (measure 200-step run)
   - Solver timing: average < 20ms per step
   - Warm-starting: observed 2-3× speedup
   - Memory usage: stable (no leaks over 10-minute run)

3. **VLA Mock Tests** (`tests/test_vla_mock.py`)
   - Mock server with controlled latency
   - Verify non-blocking property
   - Test state machine transitions
   - Verify graceful fallback on timeout

**Gate 5 Validation (MUST PASS):**
- [ ] `test_3dof_point_to_point` — EE reaches target, error < 50mm after 500 steps
- [ ] `test_3dof_mpc_loop_frequency` — Consistent ≥ 100 Hz
- [ ] `test_3dof_torque_constraints` — Never violated in any run
- [ ] `test_3dof_warm_start_benefit` — 2-3× speedup confirmed
- [ ] `test_3dof_vla_non_blocking` — Timing variance < 10% during queries
- [ ] `test_3dof_graceful_degradation` — VLA down → system holds trajectory
- [ ] `test_3dof_picking_mock` — Simple mock pick-place completes in < 30s

---

### PHASE 10: Observability & Documentation
**Duration:** 4-6 hours  
**Parallel with Phase 9** (can start while testing is running)  
**Generates:** Live dashboard, structured logging, 3 validation notebooks

**Deliverables:**
1. **Structured Logging** (`src/utils/logger_3dof.py`)
   - JSON logs to `logs/run_{timestamp}.json`
   - Every MPC step: {timestamp, q, qdot, tau, mpc_cost, qp_solve_time_ms, vla_latency_ms, state_machine}
   - Every VLA query: {timestamp, latency_ms, success, action_output}
   - Every state transition: {old_state, new_state, reason}

2. **Live Dashboard** (`src/utils/observer_3dof.py`)
   - 6 matplotlib subplots updating every 5 control steps:
     * Joint angles (q vs q_ref)
     * Joint velocities (qdot)
     * Torques (tau, overlaid with ±tau_max)
     * MPC cost per step (real-time)
     * Solver convergence (iterations per step)
     * Timing histogram: MPC step duration, VLA latency percentiles
   - Auto-scale axes, grid, legends

3. **Validation Notebooks** (`notebooks/`)
   - `01_dynamics_validation.ipynb`
     * Load test results from Gate 1
     * Plots: M(q) eigenvalues, skew-symmetry error, G decoupling
     * Numerical comparison: analytical vs MuJoCo dynamics
   
   - `02_mpc_solo_test.ipynb`
     * Gate 2-3 results
     * Plots: linearization error, QP convergence, SL vs OSQP cost
     * Show constraint satisfaction
   
   - `03_full_system_test.ipynb`
     * Gate 5 results
     * Live dashboard plots
     * Reaching trajectory animation
     * Timing analysis: frequency histogram, latency CDF

4. **Documentation** (`docs/`)
   - `docs/08-3DOF_DYNAMICS.md` — Theory section 2-5 of techspec, derivations
   - `docs/09-SMOLVLA_INTEGRATION.md` — Setup guide, architecture, async design pattern
   - `docs/10-SCALING_3DOF.md` — Performance tuning, GPU fallback logic
   - `docs/PHASE_7-10_REPORT.md` — Final results, metrics, benchmarks

---

## Success Criteria (Definition of Done)

✅ **All 5 Gates pass** with documented results  
✅ **50+ tests**, all passing (unit + integration + E2E)  
✅ **Control loop:** consistent ≥ 100 Hz  
✅ **Reaching task:** position error < 50 mm after 500 steps  
✅ **Mock picking:** completes in < 30 seconds, success > 75%  
✅ **Memory:** zero leaks after 10-minute continuous run  
✅ **Code quality:** full type hints, docstrings, imports clean  
✅ **Reproducibility:** 3 validation notebooks with all plots & metrics saved  
✅ **Documentation:** complete, with examples and troubleshooting  
✅ **Ready for Phase 11:** Fine-tuning SmolVLA + real data collection

---

## Dependencies & Environment

```bash
# Core (existing)
numpy
scipy
matplotlib
osqp
pandas

# New for 3-DOF + SmolVLA
lerobot>=0.2.0          # SmolVLA + LeRobot utilities
torch>=2.2.0            # Colab
torchvision>=0.17
fastapi>=0.110          # Colab server
uvicorn>=0.27
pyngrok>=7.0            # ngrok tunnel
aiohttp>=3.9            # async HTTP client (local)
jax[cpu]>=0.4.25        # autodiff (linearization)
pytest-asyncio>=0.23    # async test support
```

---

## Risk Register & Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| MPC QP too large (N=10 → slow) | Low | High | Reduce N to 5; optimize sparse matrices; JAX JIT |
| Colab GPU OOM | Medium | High | Load model in float16; batch RGB; test memory upfront |
| ngrok tunnel instability | Real | Medium | Auto-reconnect logic; fallback to hold-last-trajectory |
| SL solver slow on 3-DOF | Low | Medium | Tune τ_x, μ_x; warm-start; compare vs OSQP early |
| IK failures at singularities | Medium | Low | Damped pseudo-inverse (λ=0.01); check det(JJᵀ); timeout gracefully |
| Async deadlock in integration | Low | High | Never `asyncio.run()` in main loop; extensive threading tests |
| Local CPU bottleneck (expected) | High | Low | Hybrid fallback to Colab GPU already in design |

---

## Timeline & Effort

| Phase | Hours | Days | Status |
|-------|-------|------|--------|
| 7A (Dynamics) | 6-8 | 1 | Planning |
| 7B (MPC) | 4-6 | 0.75 | Planning |
| 7C (SL Solver) | 3-4 | 0.5 | Planning |
| 8A (Colab Deploy) | 3-4 | 0.5 | Planning |
| 8B (Integration) | 5-7 | 1 | Planning |
| 9 (Testing) | 8-10 | 1.5 | Planning |
| 10 (Observability) | 4-6 | 1 | Planning |
| **TOTAL** | **33-45** | **5-6** | **Planning** |

**Recommended:** 6-8 hrs/day, 5-6 consecutive working days.

---

## Module Dependencies Graph

```
┌─────────────────────┐
│  Phase 7A: Dynamics │  (6-8 hrs)
│  kinematics, M,C,G  │
│    + MJCF arm       │
└──────────┬──────────┘
           │ (Gate 1 ✓)
           ▼
┌─────────────────────┐
│   Phase 7B: MPC     │  (4-6 hrs)
│   linearize, QP     │
└──────────┬──────────┘
           │ (Gate 2 ✓)
           ▼
┌─────────────────────┐      ┌──────────────────┐
│  Phase 7C: SL       │      │  Phase 8A: Colab │  (3-4 hrs, parallel)
│  Solver (3D scale)  │      │  FastAPI+ngrok   │
└──────────┬──────────┘      └────────┬─────────┘
           │  (Gate 3 ✓)             │ (Gate 4a ✓)
           └────────────┬────────────┘
                        ▼
           ┌──────────────────────────┐
           │   Phase 8B: Integration  │  (5-7 hrs)
           │   async client, buffer   │
           └────────────┬─────────────┘
                        │ (Gate 4b ✓)
                        ▼
           ┌──────────────────────────┐
           │    Phase 9: Testing      │  (8-10 hrs)
           │    E2E, performance      │
           └────────────┬─────────────┘
                        │ (Gate 5 ✓)
           ┌────────────┴──────────────┐
           ▼                           ▼
    ┌────────────────┐      ┌─────────────────┐
    │  Phase 10:     │      │  All gates pass │
    │  Observability │      │  System ready   │
    └────────────────┘      └─────────────────┘
```

---

## Verification Checklist

Before **starting Phase 7A**, confirm:

- [ ] **Decision 1 ✓:** GPU strategy is Hybrid (local first, fallback Colab)
- [ ] **Decision 2 ✓:** Action space is EE Cartesian [x,y,z] + grasp
- [ ] **Decision 3 ✓:** Validation approach is strict gate-by-gate
- [ ] **Colab setup:** Google Colab notebook linked & accessible
- [ ] **Local env:** `.venv` configured, `python -m pytest` runs
- [ ] **Dependencies:** All packages in requirements.txt installable
- [ ] **Tech spec:** Techspec document `docs/3darm_smolvla_sl_mpc_techspec.md` fully read
- [ ] **Existing code:** Existing SL solver interface understood from `src/solver/stuart_landau_lagrange_direct.py`

---

## Next Steps

1. **Alignment:** Confirm all decisions & checklist items above ✓
2. **Phase 7A start:** Read Section 2-5 of techspec (kinematics & dynamics)
3. **Create skeleton:** Stub files for all modules (imports, docstrings)
4. **Implement:** Start with FK in kinematics_3dof.py
5. **Test:** Write & run Gate 1 tests immediately after each function
6. **Iterate:** Short feedback loops (implement → test → refine)

---

**Status:** Planning phase complete. Ready for implementation on approval.  
**Last updated:** 13 March 2026  
**Estimated completion:** 18-20 March 2026 (assuming ~40 hrs available)

