# TODO — Phase 5-6 Benchmark Validation + Phase 9-10 Future Roadmap

## Phase 5-6: SENSOR FUSION & BENCHMARKING (THIS SESSION)

### ✅ Completed
- [x] Event camera simulator (EventCameraSimulator + EventFrameProcessor)
- [x] LiDAR processor (32 rangefinder rays, normalized features)
- [x] XArmEnv multimodal _get_obs (RGB, events, LiDAR, proprio)
- [x] Sensor preprocessing (SensorFusionProcessor, numpy-based)
- [x] Deferred neural fusion to Phase 9-10 with documented rationale

### 📋 BENCHMARKS B1-B5 (IN-PROGRESS)
- [ ] **B1: MPC Solo** — Track dataset with MPC only (50 episodes)
  - Success rate, tracking error (rad), latency (ms)
  - Output: evaluation/results/B1_mpc_solo.json

- [ ] **B2: VLA + MPC** — Full dual-system on dataset (50 episodes)
  - Success rate, VLA latency, tracking error
  - Output: evaluation/results/B2_vla_mpc.json

- [ ] **B3: VLA Only** — Single-system baseline, no MPC (50 episodes)
  - Success rate, VLA latency
  - Output: evaluation/results/B3_vla_only.json

- [ ] **B4: Full End-to-End** — Closed-loop with object (30 episodes)
  - Success rate, latency breakdown
  - Output: evaluation/results/B4_full_system.json

- [ ] **B5: Sensor Ablation** — Test modality importance (30 episodes each)
  - RGB-only, RGB+events, RGB+events+LiDAR
  - Output: evaluation/results/B5_ablation.json

### 🔐 GATES VALIDATION (Pending Benchmarks)
- [ ] **Gate 5 (SmolVLA)**: Server health, latency, action shape
  - Output: logs/gate5_validation.json

- [ ] **Gate 6 (Full System)**: No crashes, honest success rate
  - Output: logs/gate6_validation.json

### 📝 DOCUMENTATION
- [ ] Update PROGRESS.md with actual B1-B5 results
- [ ] Update tech spec §12 with achieved metrics
- [ ] AGENT_STATE.md → Gates 5-6 VALIDATED ✅

---

## PHASE 9-10: FUSION ENCODER ROADMAP (DEFERRED, CONDITIONAL)

### Only proceed if B1-B5 benchmarks show success rate <80%

#### Phase 9: Lightweight Fusion
```
If benchmarks show need:
- [ ] 9.1 — Implement sklearn PCA fusion (64-dim reduction)
- [ ] 9.2 — Evaluate PCA vs raw features
- [ ] 9.3 — Benchmark performance improvement
```

#### Phase 10: Neural Fusion (Only if Phase 9 shows >5% improvement)
```
- [ ] 10.1 — Implement torch-based encoders
- [ ] 10.2 — Create MultimodalFusionEncoder
- [ ] 10.3 — Fine-tune on lerobot dataset
- [ ] 10.4 — Compare with/without fusion
- [ ] 10.5 — Consider VLA modification (if major gain)
```

---

## KEY PRINCIPLES

✅ **Run benchmarks FIRST** — Validate system with real VLA before any modifications
✅ **Log real numbers** — Never fabricate metrics
✅ **Defer is pragmatic** — Neural fusion deferral is intentional, not incomplete
✅ **Honest evaluation** — If success rate is 30%, report 30%
✅ **Document decisions** — All deferred work is in AGENT_STATE + TODO + Phase 9-10 roadmap

