# Phase 13: Multimodal Sensor Fusion & Ablation Study
**Status**: PLANNING
**Start Date**: March 15, 2026
**Objective**: Implement multimodal sensor fusion (RGB + Events + LiDAR + Proprioception) and conduct systematic ablation study to quantify contribution of each modality to VLA performance

---

## Overview

### Motivation
Phase 12 validated the dual-system architecture (B3 achieves 92.6% error reduction over B4). Phase 13 asks: **Can we further improve VLA performance by fusing neuromorphic and conventional sensors?**

### Approach
1. **Stage 1**: Implement fusion encoder architecture (§8 Tech Spec)
2. **Stage 2**: Generate simulated sensor data (events, LiDAR) from MuJoCo renders
3. **Stage 3**: Design 4-mode ablation study (RGB-only → +Events → +LiDAR → Full fusion)
4. **Stage 4**: Run benchmarks for each mode and analyze improvements
5. **Stage 5**: Visualize sensor contributions and compile results for thesis

---

## Key Decisions

### Fusion Architecture (from Tech Spec §8)

**Feature Embedding Pipeline**:
```
RGB [3,84,84] ──→ ResNet18 → [256]
Events [5,84,84] ──→ Conv3D → [128]
LiDAR [35] ──→ MLP → [64]
Proprio [4] ──→ Linear → [32]
                  ↓
              Concat [480]
                  ↓
              MLP [256] ← Final fused embedding
```

**Output**: Single [256]-dim observation embedding fed to SmolVLA instead of raw RGB

### Sensor Modality Data Sources

| Modality | Source | Generation |
|----------|--------|-----------|
| **RGB** | LeRobot dataset (primary) | `observation.images.image` [3,84,84] |
| **Events** | Simulated from RGB frames | v2e-style event camera on MuJoCo render |
| **LiDAR** | Simulated rays | 32-ray rangefinder from MuJoCo `<sensor>` |
| **Proprioception** | Real joint state | Dataset `observation.state` [4-D] or simulated [8-D] |

### Ablation Modes

| Mode | RGB | Events | LiDAR | Proprio | Notes |
|------|-----|--------|-------|---------|-------|
| **M0** (Baseline) | ✅ | ❌ | ❌ | ❌ | Reproduces B2/B3 from Phase 12 |
| **M1** (RGB+Events) | ✅ | ✅ | ❌ | ❌ | Test neuromorphic contribution |
| **M2** (RGB+LiDAR) | ✅ | ❌ | ✅ | ❌ | Test geometric sensing |
| **M3** (RGB+Proprio) | ✅ | ❌ | ❌ | ✅ | Test state awareness |
| **M4** (Full Fusion) | ✅ | ✅ | ✅ | ✅ | All modalities combined |

---

## Implementation Roadmap

### Stage 1: Fusion Module Infrastructure

#### Task 1.1: Create Fusion Encoder Module
**File**: `src/fusion/encoders/fusion_model.py` (code in Tech Spec §8)
- [ ] Implement `RGBEncoder(out_dim=256)`
- [ ] Implement `EventEncoder(n_bins=5, out_dim=128)`
- [ ] Implement `LiDAREncoder(in_dim=35, out_dim=64)`
- [ ] Implement `ProprioEncoder(in_dim=4, out_dim=32)`
- [ ] Implement `MultimodalFusionEncoder` (concatenation + MLP fusion)
- [ ] Add `.rgb_only()` classmethod for ablation baseline
- [ ] Unit tests: shape validation, forward pass integrity
- **Acceptance**: All encoders forward correctly; output shape [B, 256]

#### Task 1.2: Create Fusion Config File
**File**: `config/fusion.yaml`
- [ ] Specify encoder output dimensions (RGB=256, Event=128, LiDAR=64, Proprio=32)
- [ ] Specify event voxel parameters (n_bins=5, temporal_window_ms=50)
- [ ] Specify LiDAR parameters (n_rays=32, max_range=1.0)
- [ ] Specify fusion MLP architecture (hidden dims, activation)
- **Acceptance**: Config loads and validates without error

---

### Stage 2: Sensor Data Generation & Processing

#### Task 2.1: Event Camera Simulator
**File**: `src/simulation/cameras/event_camera.py`
- [ ] Wrap v2e event simulator (or implement simple difference-based events)
- [ ] Consume RGB frame sequences (fps=15, image size 84×84)
- [ ] Output event voxel grid [n_bins, 84, 84] (5 temporal bins)
- [ ] Handle edge cases: no motion (zeros), fast motion (saturation)
- **Acceptance**: Produces realistic event outputs for synthetic RGB sequences

#### Task 2.2: LiDAR Point Cloud Simulator
**File**: `src/simulation/cameras/lidar_sensor.py`
- [ ] Query MuJoCo rangefinders (32 rays positioned around EE)
- [ ] Convert ray distances [32] → normalized features [35] (add metadata: joint angles, gripper state)
- [ ] Handle occlusions and invalid rays (max_range)
- **Acceptance**: Returns [35] dim vectors at control frequency

#### Task 2.3: Sensor Data Loader for Real Dataset
**File**: `src/fusion/data_loaders.py`
- [ ] Create `FusedSensorDataset` class that wraps LeRobotDataset
- [ ] For each frame:
  - [ ] Load RGB from dataset: `observation.images.image` → [3,84,84]
  - [ ] Generate events from RGB sequence (temporal neighborhood frames)
  - [ ] Load proprioception: `observation.state` → [4-D]
  - [ ] Generate simulated LiDAR from MuJoCo render of state [35-D]
- [ ] Organize as dict: `{"rgb", "events", "lidar", "proprio"}`
- [ ] Add caching to avoid re-simulation on each epoch
- **Acceptance**: Returns properly formatted dicts for all 7490 dataset examples

---

### Stage 3: Fusion Encoder Integration

#### Task 3.1: Integrate Fusion Encoders with VLA Client
**File**: `src/smolvla/client.py` (extend existing RealSmolVLAClient)
- [ ] Add `use_fusion` parameter to client init
- [ ] If `use_fusion=True`:
  - [ ] Load fusion encoder Model (see Stage 1.2 config)
  - [ ] Before calling VLA `/predict`, encode sensors → [256] embedding
  - [ ] Pass embedding to VLA instead of raw RGB
  - [ ] Log embedding statistics (mean, std per modality)
- [ ] If `use_fusion=False` or modality is missing:
  - [ ] Default to RGB-only path (reproduces Phase 12)
  - [ ] Zero-pad missing modalities internally
- **Acceptance**: Client can toggle fusion modes and maintains API compatibility

#### Task 3.2: Test Fusion Modes
**File**: `tests/test_fusion_integration.py`
- [ ] Test M0 mode: RGB-only, matches B2/B3 latency (~30-40ms)
- [ ] Test M1 mode: RGB+Events, latency increases <50% (event generation + encoder)
- [ ] Test M2 mode: RGB+LiDAR, validates MuJoCo LiDAR query
- [ ] Test M4 mode: Full fusion, all modalities active
- **Acceptance**: All modes run without errors; latencies logged

---

### Stage 4: Ablation Benchmarks

#### Task 4.1: Create Ablation Benchmark Suite
**File**: `evaluation/benchmarks/run_fusion_ablation.py`
- [ ] Implement `BenchmarkFusionMode(mode: str, n_episodes: int)` runner
- [ ] For each mode (M0, M1, M2, M3, M4):
  - [ ] Run 5-10 episodes with VLA+MPC (same as B3)
  - [ ] Log: success_rate, tracking_error_rad, vla_latency_ms, fusion_overhead_ms
  - [ ] Save results to `evaluation/results/fusion_mode_{M0-M4}.json`
- [ ] Compute relative improvements vs M0 baseline
- **Acceptance**: All 5 modes complete without errors; results saved

#### Task 4.2: Run Ablation Study
**Execution**:
- [ ] Start VLA server (localhost:8000) in background
- [ ] Run full ablation: `python scripts/phase13_run_ablation.py`
  - [ ] Estimated time: ~30-40 minutes total (M0-M4, 5 ep each, 30s/episode)
  - [ ] Monitor VLA latency stability
  - [ ] Log completion time and resource usage
- [ ] Verify all 5 result files created
- **Acceptance**: Ablation complete; all 5 JSON result files valid

#### Task 4.3: Analyze Ablation Results
**Analysis** (Python script):
- [ ] Load all 5 result JSONs
- [ ] Compute statistics:
  - [ ] Success rate per mode: [M0, M1, M2, M3, M4]
  - [ ] Mean tracking error per mode
  - [ ] Mean VLA latency per mode
  - [ ] Fusion overhead per mode (M1-M4 vs M0)
- [ ] Identify best mode and worst mode
- [ ] Log findings to `evaluation/results/ABLATION_ANALYSIS.json`
- **Acceptance**: Analysis complete; clear winners/losers identified

---

### Stage 5: Visualization & Documentation

#### Task 5.1: Generate Ablation Visualizations
**File**: `visualization/fusion_ablation_plots.py`
- [ ] Plot 1: Success rate by mode (bar chart)
- [ ] Plot 2: Tracking error by mode (box plot, 5 episodes each)
- [ ] Plot 3: VLA latency overhead vs M0 (line plot)
- [ ] Plot 4: 4-panel sensor modality contribution (RGB, Events, LiDAR, Proprio each added incrementally)
- [ ] Save plots to `evaluation/results/fusion_ablation_plots.pdf`
- **Acceptance**: 4 clear, labeled plots showing modality impact

#### Task 5.2: Update Progress Documentation
**Files**: `docs/agent/PROGRESS.md`, `docs/agent/AGENT_STATE.md`, `PHASE13_SUMMARY.md`
- [ ] Log:
  - [ ] Date/time Phase 13 completed
  - [ ] All 5 ablation modes completed
  - [ ] Best-performing mode and improvement over RGB-only
  - [ ] Computational overhead analysis
  - [ ] Recommendation for Phase 14 (retraining?  visualization only?)
- **Acceptance**: Documentation complete and timestamped

---

## Success Criteria (Validation Gate)

### Gate 7: Sensor Fusion Infrastructure
- [ ] All 5 fusion encoders implemented and tested
- [ ] Fusion encoder produces [B, 256] output for all modes (M0-M4)
- [ ] Encoder integration with VLA client working
- [ ] Latency overhead for fusion <50 ms (total VLA+fusion < 100ms)

### Gate 8: Sensor Data Generation
- [ ] Event simulator produces realistic event outputs
- [ ] LiDAR simulator queries MuJoCo and normalizes rays to [35]
- [ ] Sensor loader successfully processes all 7490 dataset examples
- [ ] No data NaNs or out-of-range values

### Gate 9: Ablation Study Complete
- [ ] All 5 modes (M0-M4) completed 5-10 episodes each
- [ ] Result JSONs valid and complete
- [ ] Success rates: M0 ≥ 95%, others ≥ 90% (no catastrophic failures)
- [ ] Tracking error improvements quantified (relative to M0)

### Gate 10: Results & Analysis
- [ ] Ablation analysis JSON created with clear winner
- [ ] Plots generated showing contribution of each modality
- [ ] Documentation complete with findings and recommendations
- [ ] All metrics logged: success_rate, tracking_error, latency, fusion_overhead

---

## Estimated Timeline

| Stage | Tasks | Estimated Hours | Status |
|-------|-------|-----------------|--------|
| 1 | Infrastructure | 4-6 | NOT STARTED |
| 2 | Sensor Data | 6-8 | NOT STARTED |
| 3 | VLA Integration | 3-4 | NOT STARTED |
| 4 | Ablation Benchmarks | 2-3 | NOT STARTED |
| 5 | Visualization & Docs | 2-3 | NOT STARTED |
| **TOTAL** | | **17-24 hours** | |

**Parallelizable**:
- Stages 1, 2, 3 can run in parallel (no dependencies)
- Stage 4 depends on completion of 1, 2, 3 (but not on all 3 — just fusion decoder + sensor loader)
- Stage 5 depends only on Stage 4 results

---

## Next Steps

1. **Confirm approval** of Phase 13 scope and success criteria
2. **Create TODO list** in `docs/agent/TODO_PHASE13.md` with task assignments
3. **Start Stage 1**: Implement fusion encoder modules (Tasks 1.1, 1.2)
4. **Parallelize**: While 1.1-1.2 are running, start Stage 2 (event/LiDAR simulators)

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Fusion encoder latency > 100ms | Implement lighter architectures; consider uint8 preprocessing |
| Event similarity causes no improvement | Expected; document as finding. Move to LiDAR or mixed modes. |
| VLA server instability under load | Keep warm-up; use background restart on crash |
| Dataset size limits multi-epoch training | Use caching and memory-mapped access |
