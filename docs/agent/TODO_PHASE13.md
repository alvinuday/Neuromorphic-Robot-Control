# Phase 13 TODO List

**Status**: READY FOR EXECUTION
**Start**: March 15, 2026
**Total Tasks**: 13 main tasks + 21 subtasks

---

## Stage 1: Fusion Module Infrastructure (Est. 4-6h)

### Task 1.1: Implement Fusion Encoders
- [ ] Create `src/fusion/encoders/` directory
- [ ] Create `src/fusion/encoders/__init__.py`
- [ ] Create `src/fusion/encoders/fusion_model.py`
  - [ ] Implement `RGBEncoder(out_dim=256)` class
  - [ ] Implement `EventEncoder(n_bins=5, out_dim=128)` class
  - [ ] Implement `LiDAREncoder(in_dim=35, out_dim=64)` class
  - [ ] Implement `ProprioEncoder(in_dim=4, out_dim=32)` class
  - [ ] Implement `MultimodalFusionEncoder` class (concat + fusion MLP)
  - [ ] Add classmethod `.rgb_only()` for M0 ablation baseline
- [ ] Create unit tests in `tests/fusion/test_fusion_model.py`
  - [ ] Test each encoder forward pass with dummy input
  - [ ] Verify output shapes match expectations
  - [ ] Test MultimodalFusionEncoder with all modalities
  - [ ] Test `.rgb_only()` variant

**Acceptance Criteria**:
- All encoders produce correct output shapes
- Tests pass; no import errors
- Code follows project style

**Time Est**: 3-4 hours

---

### Task 1.2: Create Fusion Config
- [ ] Create/update `config/fusion.yaml`
  - [ ] `rgb_encoder`: {out_dim: 256}
  - [ ] `event_encoder`: {n_bins: 5, out_dim: 128}
  - [ ] `lidar_encoder`: {in_dim: 35, out_dim: 64}
  - [ ] `proprio_encoder`: {in_dim: 4, out_dim: 32}
  - [ ] `fusion_mlp`: {layers: [480, 256], activations: ['relu', 'relu']}
- [ ] Test config loads in Python: `yaml.safe_load(open('config/fusion.yaml'))`

**Acceptance Criteria**:
- Config file valid YAML
- All keys present; values reasonable

**Time Est**: 1-2 hours

---

## Stage 2: Sensor Data Generation (Est. 6-8h)

### Task 2.1: Implement Event Camera Simulator
- [ ] Check if v2e library available; install if needed
- [ ] Create `src/simulation/cameras/event_camera.py`
  - [ ] Implement `EventCameraSimulator` class
  - [ ] Method: `frame_sequence_to_events(frames: List[np.ndarray])` → event voxel [n_bins, 84, 84]
  - [ ] Handle temporal binning (5 bins across 50ms window at 15fps)
  - [ ] Add edge cases: all-zero (no motion), saturation (fast motion)
- [ ] Test on synthetic RGB sequences

**Acceptance Criteria**:
- Event outputs are within [0, 1] range
- No NaN or Inf values
- Temporal structure visible in event grids

**Time Est**: 2-3 hours

---

### Task 2.2: Implement LiDAR Simulator
- [ ] Create `src/simulation/cameras/lidar_sensor.py`
  - [ ] Implement `LiDARSimulator` class
  - [ ] Method: `query_mujoco_rangefinders(env)` → [32] ray distances
  - [ ] Method: `normalize_to_feature_vector(rays, joint_angles, gripper)` → [35]
  - [ ] Handle ray occlusion and max-range clipping
  - [ ] Log ray-by-ray debug info
- [ ] Test on MuJoCo xArm environment

**Acceptance Criteria**:
- Returns [35]-dim vectors
- Ray distances in valid range [0, 1]
- No NaN values

**Time Est**: 2-3 hours

---

### Task 2.3: Create Fused Sensor Data Loader
- [ ] Create `src/fusion/data_loaders.py`
  - [ ] Implement `FusedSensorDataset` class (wraps LeRobotDataset)
  - [ ] For each frame, return dict: {rgb, events, lidar, proprio}
  - [ ] Cache event/LiDAR outputs in `data/cache/fusion_cache/` to avoid re-simulation
  - [ ] Handle dataset iteration; support batch loading
- [ ] Create validation script `scripts/validate_fusion_data.py`
  - [ ] Load 10 samples and verify all modalities present
  - [ ] Print shapes, ranges, no NaNs

**Acceptance Criteria**:
- All 7490 dataset examples loadable
- No NaN or out-of-range values
- Caching works and speeds up 2nd epoch

**Time Est**: 2-3 hours

---

## Stage 3: Fusion Integration with VLA (Est. 3-4h)

### Task 3.1: Integrate Fusion Encoder into VLA Client
- [ ] Update `src/smolvla/real_client.py` (RealSmolVLAClient)
  - [ ] Add `use_fusion` parameter to `__init__`
  - [ ] Add `fusion_config_path` parameter
  - [ ] Load fusion encoder model if `use_fusion=True`
  - [ ] Update `predict()` method:
    - [ ] Accept additional parameters: events, lidar, proprio
    - [ ] If all present: encode via fusion encoder → [256] embedding
    - [ ] Replace rgb image with embedding in VLA request
    - [ ] Log fusion overhead (time to encode)
  - [ ] Backward compatible: if `use_fusion=False`, works as Phase 12
- [ ] Update `tests/test_vla_client.py` to test both modes

**Acceptance Criteria**:
- VLA client initialized with fusion_mode parameter
- Fusion predictions work without errors
- Latency overhead logged (<50ms)

**Time Est**: 2-3 hours

---

### Task 3.2: Test Fusion Integration
- [ ] Create `tests/test_fusion_integration.py`
  - [ ] Test M0 mode (RGB-only): latency ~30-40ms
  - [ ] Test M1 mode (RGB+Events): latency increases <50%
  - [ ] Test M2 mode (RGB+LiDAR): latency increases <50%
  - [ ] Test M4 mode (Full): all modalities active
  - [ ] Latencies logged to JSON
- [ ] Run integration test: `pytest tests/test_fusion_integration.py -v`

**Acceptance Criteria**:
- All methods run without errors
- M0 latency ≤ 50ms (matches Phase 12 B2/B3)
- M1-M4 latencies <100ms total

**Time Est**: 1-2 hours

---

## Stage 4: Ablation Benchmarks (Est. 2-3h)

### Task 4.1: Create Ablation Benchmark Script
- [ ] Create `evaluation/benchmarks/run_fusion_ablation.py`
  - [ ] Implement `run_ablation_mode(mode: str, n_episodes: int)` function
  - [ ] For each mode (M0, M1, M2, M3, M4):
    - [ ] Initialize VLA client with fusion mode
    - [ ] Run 5-10 episodes with VLA+MPC (same as Phase 12 B3)
    - [ ] Log: success, tracking_error, vla_latency, fusion_overhead
  - [ ] Save results to `evaluation/results/fusion_mode_{M0-4}.json`
  - [ ] Print summary table

**Acceptance Criteria**:
- Script runs without crashes
- All 5 result JSONs created with valid data

**Time Est**: 1-2 hours

---

### Task 4.2: Run Ablation Study
- [ ] Start VLA server: `python src/smolvla/run_vla_server.py` (background)
- [ ] Wait for warmup (200s)
- [ ] Run ablation: `python scripts/phase13_run_ablation.py`
  - [ ] Config: 5-10 episodes per mode
  - [ ] Estimated total time: 30-40 minutes
  - [ ] Monitor: VLA server health, latency stability
- [ ] Verify all result files created: `evaluation/results/fusion_mode_M*.json`

**Acceptance Criteria**:
- All 5 modes complete
- All result files valid JSON
- Success rates: M0-M4 ≥ 90% (no catastrophic degradation)

**Time Est**: 1 hour (execution only; preparation in 4.1)

---

## Stage 5: Analysis & Visualization (Est. 2-3h)

### Task 5.1: Analyze Ablation Results
- [ ] Create `evaluation/analysis/fusion_ablation_analysis.py`
  - [ ] Load all 5 result JSONs
  - [ ] Compute per-mode statistics:
    - [ ] Mean success_rate, tracking_error_rad, vla_latency_ms
    - [ ] Std dev, min, max
  - [ ] Compute improvements vs M0 baseline (%)
  - [ ] Identify best and worst modes
  - [ ] Save analysis to `evaluation/results/FUSION_ABLATION_ANALYSIS.json`
  - [ ] Print formatted table

**Acceptance Criteria**:
- Analysis JSON complete with all statistics
- Clear winner and loser identified
- Table printable for thesis

**Time Est**: 1.5 hours

---

### Task 5.2: Generate Visualizations
- [ ] Create `visualization/fusion_ablation_plots.py`
  - [ ] Plot 1: Success rate by mode (bar chart, M0-M4)
  - [ ] Plot 2: Mean tracking error by mode (bar + error bars)
  - [ ] Plot 3: Latency overhead vs M0 (line plot, cumulative)
  - [ ] Plot 4: Contribution per modality (stacked bar: RGB, +Events, +LiDAR, +Proprio)
  - [ ] Save to `evaluation/results/fusion_ablation_plots.pdf` (multi-page)
  - [ ] Add legends, labels, grid

**Acceptance Criteria**:
- 4 plots generated and saved
- All plots readable (clear labels, legends)
- PDF file created

**Time Est**: 1-2 hours

---

### Task 5.3: Update Documentation
- [ ] Update `docs/agent/AGENT_STATE.md`
  - [ ] Mark Phase 13 complete
  - [ ] Note best-performing mode
  - [ ] Key findings (which modalities help most)
- [ ] Update `docs/agent/PROGRESS.md`
  - [ ] Add Phase 13 completion entry with timestamp
  - [ ] Log all 5 benchmark results with paths
  - [ ] Log analysis and visualization file paths
- [ ] Create `docs/PHASE13_SUMMARY.md`
  - [ ] Executive summary of ablation findings
  - [ ] Recommendation: Keep best mode for Phase 14?
  - [ ] Comparison with Phase 12 (B2/B3)

**Acceptance Criteria**:
- Documentation complete and timestamped
- All file paths correct
- Summary accessible for thesis writing

**Time Est**: 1 hour

---

## Success Criteria (Validation Gate)

- [ ] **Gate 7**: All fusion encoders implemented, tested, produce [B, 256] output
- [ ] **Gate 8**: Sensor simulators (events, LiDAR) produce realistic outputs; no NaNs
- [ ] **Gate 9**: All 5 ablation modes complete; success rates ≥ 90%
- [ ] **Gate 10**: Analysis complete; visualizations generated; documentation updated

---

## Execution Notes

**Start Order**:
1. Execute Task 1.1 & 1.2 (infrastructure)
2. Parallelize Tasks 2.1, 2.2, 2.3 (sensor data)
3. Execute Task 3.1 & 3.2 (VLA integration)
4. Execute Task 4.1, then 4.2 (benchmarks)
5. Execute Task 5.1, 5.2, 5.3 (analysis)

**Critical Path**: 1.1 → 1.2 → 2.1/2.2/2.3 → 3.1/3.2 → 4.1 → 4.2 → 5.1/5.2/5.3

**Estimated Total Time**: 17-24 hours (can parallelize to ~12-16 hours with concurrent work)

**Decision Points**:
- After 4.2: If best mode < 90% success, investigate issues before proceeding to 5.x
- After 5.1: If improvements negligible (<5%), document as negative result and proceed to Phase 14

---

## Related Documentation

- Phase 13 Plan: [PHASE13_PLAN.md](PHASE13_PLAN.md)
- Tech Spec (Fusion): [docs/sensor_fusion_vla_mpc_techspec_v2.md](../../sensor_fusion_vla_mpc_techspec_v2.md) §8
- Phase 12 Results: [Memory: phase_12_complete.md]
