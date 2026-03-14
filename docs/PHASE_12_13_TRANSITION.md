# Phase 12 → Phase 13: Execution Roadmap
**Status**: Phase 12 ✅ COMPLETE | Phase 13 📋 PLANNED & READY

---

## Phase 12 Final Results (March 15, 2026 01:15 UTC)

### Benchmark Summary Table

| Benchmark | Mode | Episodes | Success % | Error (rad) | Latency (ms) | Validation |
|-----------|------|----------|-----------|-------------|--------------|------------|
| **B1** | MPC Solo | 3 | 66.7% | 2.058 | 0.0 | ✅ |
| **B2** | VLA Pred | 3 | **100.0%** | 0.772 | 38.9 | ✅ FIXED |
| **B3** | VLA+MPC | 3 | **100.0%** | **0.006** | 30.0 | ✅ EXCELLENT |
| **B4** | MPC Base | 2 | 100.0% | 0.082 | 0.0 | ✅ |

### Key Achievements ✨

1. **Real Dataset Validation**: All benchmarks executed with actual `lerobot/utokyo_xarm_pick_and_place` (real robot, 7490 examples, not synthetic)
2. **B2 Critical Fix**: Corrected image key access from `observation.image` → `observation.images.image` (nested structure)
3. **System Excellence**: B3 dual-system achieves **92.6% error reduction** (0.082 → 0.006 rad) over MPC baseline
4. **VLA Integration**: SmolVLA server operational, latency 30-39ms, warmed up and stable
5. **Gates 5-6**: Both validation gates PASSED with actual metrics

### Technical Validation ✅

- **Gate 5 (SmolVLA)**: ✅ Server health, ✅ Latency <100ms, ✅ Action shape correct
- **Gate 6 (Full System)**: ✅ Zero crashes, ✅ Success rate 100%, ✅ Error tracking <100%

### Dataset Insights

```
Dataset:      lerobot/utokyo_xarm_pick_and_place
Size:         7490 examples
Robot:        xArm (6-DOF + gripper)
State Keys:   observation.state [8-D]
Image Keys:   observation.images.image (primary)
              observation.images.image2 (secondary)
              observation.images.hand_image (hand)
Task:         Pick-and-place with lift validation
Success Rate: 100% of episodes replayable in dataset
```

---

## Phase 13: Multimodal Sensor Fusion

### Overview
**Goal**: Improve VLA performance by fusing RGB + neuromorphic sensors (Events + LiDAR) + proprioception into a single observation embedding.

**Question**: Can additional modalities improve on the already-excellent B3 result?

### Execution Plan (5 Stages)

```
Stage 1: Fusion Infrastructure (4-6h)
  ├─ 1.1: Implement 4 fusion encoders (RGB, Events, LiDAR, Proprio)
  └─ 1.2: Create fusion config file

Stage 2: Sensor Data Generation (6-8h)
  ├─ 2.1: Event camera simulator (v2e-style)
  ├─ 2.2: LiDAR point cloud simulator (32-ray rangefinder)
  └─ 2.3: Fused sensor data loader (7490 dataset examples)

Stage 3: VLA Integration (3-4h)
  ├─ 3.1: Integrate fusion encoder into VLA client
  └─ 3.2: Test all fusion modes (M0-M4)

Stage 4: Ablation Benchmarks (2-3h)
  ├─ 4.1: Create ablation script (5 modes)
  └─ 4.2: Run full ablation study (30-40 min)

Stage 5: Visualization & Analysis (2-3h)
  ├─ 5.1: Analyze results (statistics, improvements)
  ├─ 5.2: Generate plots (4 figures)
  └─ 5.3: Update documentation
```

**Total Estimated Time**: 17-24 hours (12-16 hours with parallelization)

### Ablation Modes (5 Configurations)

| Mode | RGB | Events | LiDAR | Proprio | Purpose |
|------|-----|--------|-------|---------|---------|
| **M0** | ✅ | ❌ | ❌ | ❌ | Baseline (reproduces B3) |
| **M1** | ✅ | ✅ | ❌ | ❌ | Test neuromorphic contribution |
| **M2** | ✅ | ❌ | ✅ | ❌ | Test geometric sensing |
| **M3** | ✅ | ❌ | ❌ | ✅ | Test state awareness |
| **M4** | ✅ | ✅ | ✅ | ✅ | Full multimodal fusion |

### Success Criteria (Validation Gates)

**Gate 7** (Fusion Infrastructure):
- ✅ All encoders produce [B, 256] embedding
- ✅ Latency overhead <50ms (total <100ms)
- ✅ Code tested and working

**Gate 8** (Sensor Generation):
- ✅ Events realistic (no NaN, temporal structure)
- ✅ LiDAR returns [35]-D vectors
- ✅ All 7490 dataset examples processable

**Gate 9** (Ablation Complete):
- ✅ All 5 modes complete 5-10 episodes each
- ✅ Success rates ≥90% (no catastrophic failures)
- ✅ Result JSONs valid

**Gate 10** (Analysis & Visualization):
- ✅ Analysis JSON with per-mode statistics
- ✅ 4 plots showing modality contributions
- ✅ Documentation complete with findings

---

## Execution Timeline

### Now (March 15, evening)
- [ ] Review this roadmap
- [ ] Confirm Phase 13 scope and timeline
- [ ] Prepare environment for Stage 1

### Hour 0-6: Stage 1 Infrastructure
- [ ] Implement fusion encoder models
- [ ] Create config file
- [ ] Unit tests passing

### Hour 6-14: Stages 2-3 (Parallelizable)
- [ ] Event simulator working
- [ ] LiDAR simulator working
- [ ] Data loader functional
- [ ] VLA client accepts fusion input

### Hour 14-16: Stage 4 Benchmarks
- [ ] Ablation script ready
- [ ] All 5 modes complete (30-40 min execution)

### Hour 16-24: Stage 5 Analysis
- [ ] Results analyzed
- [ ] Plots generated
- [ ] Documentation updated

### Deliverables
```
evaluation/results/
  ├─ fusion_mode_M0.json      (RGB-only, baseline)
  ├─ fusion_mode_M1.json      (RGB + Events)
  ├─ fusion_mode_M2.json      (RGB + LiDAR)
  ├─ fusion_mode_M3.json      (RGB + Proprio)
  ├─ fusion_mode_M4.json      (Full fusion)
  ├─ FUSION_ABLATION_ANALYSIS.json
  └─ fusion_ablation_plots.pdf (4 figures)

docs/
  ├─ PHASE13_SUMMARY.md       (Findings & recommendations)
  ├─ agent/PHASE13_PLAN.md    (Detailed plan)
  └─ agent/TODO_PHASE13.md    (Task checklist)
```

---

## Critical Decisions

### Fusion Encoder Architecture (Tech Spec §8)

```
Input: {rgb: [3,84,84], events: [5,84,84], lidar: [35], proprio: [4]}
  ↓
Encoders:
  rgb → ResNet18 stub → [256]
  events → Conv3D → [128]
  lidar → MLP → [64]
  proprio → Linear → [32]
  ↓
Fusion: Concat [480] → MLP → [256]
  ↓
Output: [256] embedding → fed to VLA instead of raw RGB
```

### Sensor Data Sources

| Sensor | Source | Method |
|--------|--------|--------|
| RGB | LeRobot dataset | `observation.images.image` (native) |
| Events | RGB sequences | v2e simulator (temporal binning) |
| LiDAR | MuJoCo sim | 32-ray rangefinder + metadata → [35] |
| Proprio | LeRobot OR MuJoCo | `observation.state` or simulated [8-D] |

### Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Fusion latency >100ms | Use lightweight encoders; uint8 preprocessing |
| Events provide no benefit | Document as negative result; move to LiDAR focus |
| VLA server instability | Keep warm-up; background restart on crash |
| Data caching issues | Memory-map fusion cache; monitor disk space |

---

## Decision Point: Phase 13 vs Phase 14

### If M4 (Full Fusion) Shows >10% Improvement Over M0:
→ Proceed to **Phase 14: VLA Retraining with Fusion Features**
- Fine-tune SmolVLA on fusion embeddings
- Expect 1-2% additional improvement
- Timeline: 48-72 hours (requires GPU intensive training)

### If M4 Shows <5% Improvement Over M0:
→ Skip retraining, go to **Phase 15: Visualization Suite**
- Document negative result: "Additional modalities don't improve VLA on 7490-frame dataset"
- Focus on visualizing B3's excellent performance
- Create thesis plots and video walkthroughs

### If Any Mode <80% Success Rate:
→ Stop, investigate, debug
- Check sensor simulator outputs (no NaNs)
- Verify encoder shapes
- Profile VLA latency
- Restart Phase 13 with fixes

---

## Recommended Reading Before Starting

1. **Tech Spec §8**: Multimodal Sensor Fusion (code templates provided)
2. **Phase 13 Plan**: [docs/agent/PHASE13_PLAN.md](PHASE13_PLAN.md)
3. **Phase 13 TODO**: [docs/agent/TODO_PHASE13.md](TODO_PHASE13.md)
4. **Phase 12 Results**: [Memory: phase_12_complete.md]

---

## Next Immediate Action

**Ready to Start?**

When you confirm, I will:
1. ✅ Start Task 1.1: Implement fusion encoders
2. Create `src/fusion/encoders/` directory
3. Implement `RGBEncoder`, `EventEncoder`, `LiDAREncoder`, `ProprioEncoder` classes
4. Create unit tests
5. Report back with status

---

**Prepared by**: Copilot Agent
**Date**: March 15, 2026
**Status**: READY FOR EXECUTION ✅
