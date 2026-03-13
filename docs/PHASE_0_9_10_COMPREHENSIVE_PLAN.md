# Phase 0, Phase 9, Phase 10: Comprehensive Validation, Benchmarking & Observability Plan

**Created:** 13 March 2026  
**Scope:** Full system validation, performance benchmarking on real datasets, observability infrastructure  
**Total Effort:** 40-60 hours  
**Status:** Planning → Implementation

---

## 1. Executive Summary

This document outlines an extensive validation and benchmarking campaign for the 3-DOF SmolVLA + MPC robot control system:

- **Phase 0:** Pre-flight validation (component health, dependencies, MuJoCo environment)
- **Phase 9:** Performance tuning and dataset benchmarking (OpenX Embodiment datasets)
- **Phase 10:** Observability, dashboards, final reporting

### Key Objectives

1. ✅ **Validate all components** work correctly in isolation and integration
2. ✅ **Benchmark performance** on standardized datasets (OpenX)
3. ✅ **Test on MuJoCo simulated robot** with realistic dynamics
4. ✅ **Create visualizations** (headless logs + interactive plots + video)
5. ✅ **Compare against baselines** (state-of-the-art VLA models)
6. ✅ **Generate comprehensive reports** with metrics and insights

### System Under Test

```
┌─────────────────────────────────────────────────────────────────┐
│  SmolVLA (450M params on Colab T4 GPU) + SL-Oscillator MPC      │
│                                                                 │
│  Input: RGB frame (224×224) + language instruction + joint q    │
│  Output: Joint torques τ* at 100+ Hz, EE position accuracy      │
│  Task: Reach targets, pick-and-place, follow trajectories       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase 0: Pre-Flight Validation (Sanity Checks)

**Duration:** 4-6 hours  
**Goal:** Ensure all components are healthy, dependencies installed, MuJoCo working

### 2.1 Component Health Checks

```python
# tests/phase0_component_health.py
class TestPhase0Health:
    def test_imports_all_modules(self):
        """Verify all src/ modules import without errors"""
        
    def test_mujoco_installation(self):
        """MuJoCo renders without GPU errors"""
        
    def test_dynamics_initialization(self):
        """3-DOF Arm3DOFDynamics initializes"""
        
    def test_mpc_solver_initialization(self):
        """StuartLandauLagrangeDirect initializes"""
        
    def test_smolvla_client_initialization(self):
        """RealSmolVLAClient can be configured"""
        
    def test_trajectory_buffer_initialization(self):
        """TrajectoryBuffer initializes correctly"""
        
    def test_dual_system_controller_initialization(self):
        """DualSystemController assembles all components"""
```

### 2.2 MuJoCo Environment Setup

**Create:** `src/environments/mujoco_3dof_env.py`

```python
class MuJoCo3DOFEnv:
    """OpenAI Gym-style environment for 3-DOF arm"""
    
    def __init__(self, render_mode=None, headless=False):
        # Load arm3dof.xml model
        # Create renderer if render_mode='human'
        # Initialize cameras (overhead, side)
        
    def reset(self) -> dict:
        """Reset to home position, return observation"""
        
    def step(self, action: np.ndarray) -> tuple:
        """Apply torque, step simulation, return state"""
        
    def render(self) -> np.ndarray | None:
        """Return RGB image or render if human mode"""
        
    def get_state(self) -> dict:
        """Return {q, qdot, ee_pos, ee_vel, ee_force}"""
```

### 2.3 Data Collection Utility

**Create:** `src/utils/data_collector.py`

```python
class DataCollector:
    """Collect trajectory + metrics during experiments"""
    
    def __init__(self, output_dir, task_name):
        self.metrics = {}  # timestamp, q, qdot, tau, mpc_cost, vla_latency
        
    def record_step(self, step: int, state: dict, action: np.ndarray, 
                    metrics: dict):
        """Record one control step"""
        
    def record_vla_query(self, latency_ms, success, instruction, action):
        """Record VLA query stats"""
        
    def record_task_completion(self, success, time_s, error_final):
        """Record task-level metrics"""
        
    def save_to_hdf5(self):
        """Save all collected data to compressed HDF5"""
```

### 2.4 Health Check Tests

```bash
# Run all Phase 0 checks
pytest tests/phase0_component_health.py -v --tb=short
pytest tests/phase0_mujoco_env.py -v --tb=short
pytest tests/phase0_data_collection.py -v --tb=short

# Verify no regressions in existing tests
pytest tests/ -k "not gate4b and not gate5" -v --tb=line
```

### 2.5 Success Criteria for Phase 0

- [ ] 100% of component health checks pass
- [ ] MuJoCo environment renders without errors (both headless + human)
- [ ] Data collector successfully logs trajectory data
- [ ] All 117 existing tests still pass (no regressions)
- [ ] Dependencies: `mujoco`, `h5py`, `lerobot`, `aiohttp` all installed

---

## 3. OpenX Dataset Integration

**Duration:** 4-6 hours  
**Goal:** Download relevant subsets of OpenX Embodiment dataset and create evaluation harness

### 3.1 Dataset Selection

**OpenX Embodiment:** https://github.com/google-deepmind/open_x_embodiment

**Relevant subsets for 3-DOF arm:**
1. **CALVIN** (CIRL) — Task sequence dataset
   - 15K trajectories, 30-sec clips
   - Language instructions: "place", "put", "pick", "move"
   - Simulated robot arm (K3 manipulator)
   - **Subset:** First 1000 trajectories (~200 unique tasks)
   
2. **BridgeData** — Real robot data (smaller)
   - Reach + push + grasp tasks
   - 50K trajectories across 40 hours
   - **Subset:** First 500 trajectories (human-verified good quality)
   
3. **Sim-to-Real Tasks** — Simulation data
   - Reaching, grasping in PyBullet
   - 100K+ trajectories
   - **Subset:** First 2000 trajectories for speed

### 3.2 Dataset Download & Preprocessing

**Create:** `src/datasets/openx_loader.py`

```python
class OpenXDataset:
    """Load OpenX Embodiment datasets"""
    
    @staticmethod
    def download_calvin(split='train', num_episodes=1000):
        """Download CALVIN subset from HuggingFace"""
        # from datasets import load_dataset
        # dataset = load_dataset('google_robotics/openx', 'calvin', split=split)
        
    @staticmethod
    def normalize_action(action, action_stats):
        """LinearNormalizer for action space"""
        
    @staticmethod
    def extract_language_instruction(trajectory):
        """Get instruction from trajectory metadata"""
        
    @staticmethod
    def extract_rgb_frames(trajectory, resolution=(224, 224)):
        """Extract RGB images, resize to VLA input size"""
        
    @staticmethod
    def extract_joint_angles(trajectory, dof=3):
        """Map proprioceptive state to 3-DOF joint angles"""
```

### 3.3 Dataset Tests

```python
# tests/test_openx_dataset.py
class TestOpenXDataset:
    def test_calvin_download(self):
        """Verify CALVIN dataset downloads successfully"""
        
    def test_calvin_episode_structure(self):
        """Check: RGB, joints, actions, language present"""
        
    def test_action_normalization(self):
        """Verify action normalization reversible"""
        
    def test_trajectory_sampling(self):
        """Sample random trajectories without errors"""
```

### 3.4 Success Criteria

- [ ] At least 1 OpenX dataset subset downloaded successfully
- [ ] 1000+ trajectories available locally
- [ ] Frames resizable to 224×224 without artifacts
- [ ] Actions mappable to 3-DOF joint space
- [ ] All dataset tests pass

---

## 4. MuJoCo 3-DOF Arm Tests (Validation)

**Duration:** 6-8 hours  
**Goal:** Test arm control in full simulation with realistic physics

### 4.1 Simulation Test Suite

**Create:** `tests/test_mujoco_3dof_tasks.py`

```python
class TestMuJoCo3DOFTasks:
    """Test control tasks in MuJoCo simulation"""
    
    def test_home_position_stability(self):
        """Hold home position for 10 seconds, no drift"""
        
    def test_single_joint_step_response(self):
        """Joint 1: step from 0→π/2, rise time < 2 sec"""
        
    def test_3dof_reaching_task(self):
        """Reach 10 random targets, error < 50mm"""
        
    def test_trajectory_tracking(self):
        """Follow sinusoidal reference, track error < 5%"""
        
    def test_constraint_enforcement(self):
        """Verify torque, velocity limits never violated"""
        
    def test_collision_detection(self):
        """Self-collision avoidance if enabled"""
        
    def test_real_world_noise(self):
        """Add ±5% torque noise, tracking still stable"""
        
    def test_long_run_stability(self):
        """100 random commands, 30 seconds total, no NaN"""
```

### 4.2 Visualization Tests

**Create:** `tests/test_visualization.py`

```python
class TestVisualization:
    def test_headless_render(self):
        """Render 100 frames to PNG without display"""
        
    def test_trajectory_plot_generation(self):
        """Create matplotlib plots of trajectories"""
        
    def test_video_generation(self):
        """Render simulation to MP4 video"""
        
    def test_summary_stats_plot(self):
        """Plot control metrics over time"""
```

### 4.3 Success Criteria

- [ ] All 8 task tests pass in MuJoCo
- [ ] Reaching accuracy < 50mm achieved
- [ ] At least 3 videos generated (reaching, tracking, picking)
- [ ] No NaN/Inf errors in 30-second runs

---

## 5. Phase 9: Performance Tuning & Benchmarking

**Duration:** 12-16 hours  
**Goal:** Measure, optimize, and benchmark on real datasets

### 5.1 Baseline Performance Measurement

**Create:** `src/benchmarks/baseline_benchmark.py`

```python
def benchmark_component_latencies():
    """Measure latency of each component"""
    # MPC solver: <20ms target
    # VLA inference: 700ms typical
    # Trajectory buffer: <1ms
    # State machine: <1ms
    
def benchmark_qp_solve_time():
    """Various problem sizes"""
    # N=5 horizon: ? ms
    # N=10 horizon: ? ms
    # N=20 horizon: ? ms
    # N=30 horizon: ? ms
    
def benchmark_sl_solver_convergence():
    """Number of iterations needed"""
    # Easy QP: ? iterations
    # Hard QP: ? iterations
    # Average: ? iterations
    
def benchmark_control_loop_frequency():
    """Run for 1 minute, measure actual Hz"""
    # Target: 100 Hz
    # Variance: < 10%
```

### 5.2 Profiling & Optimization

**Create:** `src/benchmarks/profiler.py`

```python
class SystemProfiler:
    """Profile all components with cProfile"""
    
    def profile_mpc_step(self):
        """cProfile MPC solver step"""
        
    def profile_vla_query(self):
        """Network I/O profiling"""
        
    def profile_control_loop(self):
        """Full integration profiling"""
```

### 5.3 Dataset Evaluation

**Create:** `tests/test_openx_performance.py`

```python
class TestOpenXPerformance:
    """Evaluate system on OpenX datasets"""
    
    def test_calvin_reaching(self):
        """100 reaching tasks from CALVIN"""
        # Metrics: success rate, accuracy, time-to-goal
        
    def test_calvin_manipulation(self):
        """50 pick-and-place tasks"""
        
    def test_bridgedata_eval(self):
        """BridgeData subset evaluation"""
        
    def test_sim_tasks_eval(self):
        """Simulation task performance"""
```

### 5.4 Comparison Against Baselines

**Create:** `src/benchmarks/baseline_comparison.py`

```
Compare against:
1. Simple reaching controller (IK only, no MPC)
2. Non-learning MPC (no VLA feedback)
3. Pure VLA (no MPC, just run inference)
4. Random actions
5. Hand-coded heuristics

Metrics:
- Success rate (%)
- Mean time-to-goal (sec)
- Tracking error (mm)
- Energy consumption (torque·dt)
- Robustness (std deviation across noise levels)
```

### 5.5 Success Criteria

- [ ] Control loop: 100+ Hz sustained
- [ ] MPC solver: <20ms mean
- [ ] Dataset eval: ≥75% success on ≥3 tasks
- [ ] Outperforms baselines on ≥2 metrics
- [ ] No performance degradation vs mock tests

---

## 6. Phase 10: Observability & Dashboards

**Duration:** 8-12 hours  
**Goal:** Create comprehensive dashboards, logging, and reporting

### 6.1 Structured Logging

**Create:** `src/utils/experiment_logger.py`

```python
class ExperimentLogger:
    """Structured JSON logging for all events"""
    
    def __init__(self, log_dir, experiment_name):
        # Output: logs/{experiment_name}/{timestamp}.jsonl
        
    def log_control_step(self, step: int, state: dict, action: np.ndarray, 
                         timing: dict, cost: float):
        """Log one control step: {q, qdot, tau, t_mpc, vla_latency, cost}"""
        
    def log_vla_query(self, instruction: str, rgb_shape: tuple, 
                     action: np.ndarray, latency_ms: float, success: bool):
        """Log VLA query event"""
        
    def log_task_event(self, event_type: str, task_name: str, 
                      details: dict):
        """Log high-level task events: start, end, success/fail"""
        
    def to_dataframe(self) -> pd.DataFrame:
        """Convert logs to pandas DataFrame for analysis"""
```

### 6.2 Real-Time Dashboard

**Create:** `src/utils/live_dashboard.py`

```python
class LiveDashboard:
    """Matplotlib dashboard updating every 5 steps"""
    
    def __init__(self, fig_size=(16, 10)):
        # 6 subplots
        
    def update(self, state: dict, action: np.ndarray, metrics: dict):
        """Update all plots"""
        
    # Subplots:
    # 1. Joint angles: q(t) vs q_ref(t)
    # 2. Joint torques: τ(t) with ±τ_max
    # 3. EE position: x(t) toward goal
    # 4. MPC cost per step
    # 5. Timing histogram: MPC, VLA latency percentiles
    # 6. State machine transitions (color-coded)
```

### 6.3 Offline Analysis Tools

**Create:** `notebooks/analysis_toolkit.ipynb`

Cells:
1. Load experiment data (pickle/HDF5/JSONL)
2. Plot trajectories with annotations
3. Compare multiple experiments side-by-side
4. Compute statistics (mean, std, quantiles)
5. Generate summary report (Markdown)

### 6.4 Video Generation

**Create:** `src/utils/video_recorder.py`

```python
class VideoRecorder:
    """Record simulation + overlay metrics as MP4"""
    
    def __init__(self, output_path, fps=30):
        
    def add_frame(self, rgb: np.ndarray, overlay_text: str):
        """Add MuJoCo frame with text overlay"""
        
    def finalize(self):
        """Write MP4 using ffmpeg"""
```

### 6.5 Comprehensive Reporting

**Create:** `docs/EVALUATION_REPORT.md` (auto-generated)

```markdown
# System Evaluation Report
**Generated:** <timestamp>
**Experiment:** <name>

## Executive Summary
- ✅ X tests passed out of Y
- ✅ Control loop: Z Hz
- ✅ Task success rate: W%

## Component Performance
- MPC solver: A ms (target: <20ms) ✅
- VLA inference: B ms (target: <700ms) ✅
- MPC + VLA integration: C ms (target: non-blocking) ✅

## Dataset Results
| Dataset | Task | Success | Time (s) | Accuracy | Notes |
|---------|------|---------|----------|----------|-------|
| CALVIN | Reaching | 85% | 4.2 | 25mm | Limited by arm workspace |

## Plots
[Embedded matplotlib PNGs]

## Recommendations
1. ...
2. ...
```

### 6.6 Success Criteria

- [ ] JSON logs for all experiments
- [ ] Live dashboard renders without lag
- [ ] 5+ videos generated (headless)
- [ ] Final report includes ≥10 plots
- [ ] Analysis notebook runs end-to-end

---

## 7. Detailed Implementation Timeline

### Week 1: Phase 0 + Phase 9 Part 1

| Day | Task | Deliverables | Est. Time |
|-----|------|--------------|-----------|
| 1 | Phase 0 health checks | 30+ passing tests | 4 hrs |
| 2 | MuJoCo env setup | 3DOFEnv class, 500 tests | 4 hrs |
| 3 | OpenX dataset integration | Dataset loader, 1000+ trajectories | 4 hrs |
| 4 | Dataset tests | All dataset tests pass | 3 hrs |

### Week 2: Phase 9 Continued, Phase 10

| Day | Task | Deliverables | Est. Time |
|-----|------|--------------|-----------|
| 5 | Benchmark baseline perf | Latency profiles, graphs | 4 hrs |
| 6 | Dataset eval (50+ tasks) | Success rate metrics | 6 hrs |
| 7 | Baseline comparisons | 5 baselines, comparison table | 4 hrs |
| 8 | Logging infrastructure | JSON logs, DataFrames | 3 hrs |
| 9 | Dashboards + video | Live plots, 5+ videos | 5 hrs |
| 10 | Final reporting | EVALUATION_REPORT.md | 3 hrs |

**Total Effort:** ~40 hours  
**Recommended Pace:** 6-8 hrs/day over 5-6 working days

---

## 8. Acceptance Criteria & Sign-Off

### Phase 0: Pre-Flight Validation
- [ ] All component health checks pass
- [ ] MuJoCo environment works headless and with display
- [ ] Data collection utility logs trajectories
- [ ] No regressions in existing 117 tests

### Phase 9: Performance & Benchmarking
- [ ] Control loop sustained ≥100 Hz
- [ ] MPC solver <20ms mean latency
- [ ] VLA queries ≤700ms p95
- [ ] ≥3 datasets evaluated
- [ ] ≥75% success on ≥2 tasks
- [ ] Outperforms ≥2 baselines

### Phase 10: Observability
- [ ] Experiment logs in structured JSON format
- [ ] Live dashboard renders without artifacts
- [ ] 5+ MP4 videos generated
- [ ] Final report with ≥20 plots/tables
- [ ] Analysis Jupyter notebook reproducible

### Overall System Validation
- [ ] **All 140+ tests pass** (117 existing + 23+ new)
- [ ] **System ready for real robot deployment**
- [ ] **Complete documentation** with troubleshooting
- [ ] **Reproducible results** on standard datasets

---

## 9. Failure Modes & Mitigation

| Risk | Likelihood | Mitigation |
|------|-----------|-----------|
| MuJoCo render fails on macOS | Medium | Use xvfb or Docker |
| Dataset download too slow | Medium | Cache locally, incremental download |
| VLA server timeouts during eval | Medium | Add retry logic, mock fallback |
| Memory usage spikes with video | Low | Stream video chunks, lower resolution |
| Profiling introduces overhead | Low | Use sampling profiler (py-spy) |

---

## 10. Resource Requirements

```
Hardware:
- Local: 4+ cores, 8GB RAM (simulation + analysis)
- Colab: T4 GPU (VLA inference)
- Disk: 50GB (datasets + logs + videos)

Software:
- mujoco 3.x
- opencv-python (video)
- pandas, matplotlib, seaborn
- h5py, torch, lerobot
- datasets (HuggingFace)

Time Budget:
- Phase 0: 4-6 hrs
- Phase 9: 12-16 hrs
- Phase 10: 8-12 hrs
- Total: 40-60 hrs
```

---

## 11. Success Metrics Summary

By the end of this plan:

```
✅ Component Validation: 30+ tests, 100% pass rate
✅ System Integration: 100+ Hz control loop, non-blocking
✅ Dataset Evaluation: 1000+ tasks tested, ≥75% success
✅ Baseline Comparison: Outperforms ≥2 handcrafted baselines
✅ Observability: Live dashboards, 10+ plots, videos
✅ Documentation: Reproducible, artifact-driven reporting
✅ Production Readiness: All systems validated and benchmarked
```

---

## 12. Next Steps

1. **Approve this plan** (review and sign-off)
2. **Start Phase 0** (health checks, MuJoCo setup)
3. **Parallel: OpenX dataset download** (cache locally)
4. **Phase 9 benchmarking** (profile all components)
5. **Phase 10 observability** (dashboards, reporting)
6. **Final validation** (full system test, sign-off)

---

**Document Owner:** AI Agent  
**Last Updated:** 13 March 2026  
**Status:** Ready for implementation.
