# TODO — Phase 12: Comprehensive Benchmarking (2026-03-15)

## STAGE 1: Benchmark Execution & Validation (THIS STAGE)

### Subtask 1.1: Verify Benchmark API Compatibility
- [ ] 1.1.1 — Read benchmark file headers and structure
- [ ] 1.1.2 — Cross-check RealSmolVLAClient calls vs test_vla_api.py format
- [ ] 1.1.3 — Verify payload structure: rgb_b64, state, instruction, language_tokens
- [ ] 1.1.4 — Verify response parsing: action, action_std, latency_ms, success
- [ ] 1.1.5 — Document any API mismatches found
- [ ] 1.1.6 — Fix any payload format issues
- [ ] 1.1.7 — Test single VLA query manually before full benchmark

### Subtask 1.2: Pre-Benchmark Health Checks
- [ ] 1.2.1 — Verify VLA server still healthy (quick health check)
- [ ] 1.2.2 — Verify environment loads without error
- [ ] 1.2.3 — Verify dataset is accessible
- [ ] 1.2.4 — Verify result directory exists (evaluation/results/)
- [ ] 1.2.5 — Verify logging is configured

### Subtask 1.3: Execute B1 Benchmark (MPC Solo)
- [ ] 1.3.1 — Start timer for B1
- [ ] 1.3.2 — Run: python evaluation/benchmarks/run_b1_b5_comprehensive.py (B1 only)
- [ ] 1.3.3 — Monitor output for errors
- [ ] 1.3.4 — Wait for completion (expected ~5 min)
- [ ] 1.3.5 — Verify B1_mpc_solo.json was created
- [ ] 1.3.6 — Parse and log success_rate, mean_tracking_error

### Subtask 1.4: Execute B2 Benchmark (VLA Prediction)
- [ ] 1.4.1 — Health check VLA server (should be warm now)
- [ ] 1.4.2 — Run: python evaluation/benchmarks/run_b1_b5_comprehensive.py (B2 only)
- [ ] 1.4.3 — Monitor VLA query responses
- [ ] 1.4.4 — Watch for timeout errors (should be ~20-55ms per query)
- [ ] 1.4.5 — Wait for completion (expected ~30 min)
- [ ] 1.4.6 — Verify B2_vla_prediction.json was created
- [ ] 1.4.7 — Parse and log mean_vla_latency, mean_action_error

### Subtask 1.5: Execute B3 Benchmark (Full Dual-System)
- [ ] 1.5.1 — Health check VLA server
- [ ] 1.5.2 — Run: python evaluation/benchmarks/run_b1_b5_comprehensive.py (B3 only)
- [ ] 1.5.3 — Monitor both MPC and VLA integration
- [ ] 1.5.4 — Watch for any control instabilities
- [ ] 1.5.5 — Wait for completion (expected ~45 min)
- [ ] 1.5.6 — Verify B3_dual_system.json was created
- [ ] 1.5.7 — Parse and log success_rate, mean_tracking_error, system_latency

### Subtask 1.6: Execute B4 Benchmark (MPC Baseline)
- [ ] 1.6.1 — Run: python evaluation/benchmarks/run_b1_b5_comprehensive.py (B4 only)
- [ ] 1.6.2 — Wait for completion (expected ~3 min)
- [ ] 1.6.3 — Verify B4_mpc_baseline.json was created
- [ ] 1.6.4 — Parse and log MPC-only metrics (should be very accurate)

### Subtask 1.7: Execute B5 Benchmark (Sensor Ablation)
- [ ] 1.7.1 — Health check VLA server
- [ ] 1.7.2 — Run: python evaluation/benchmarks/run_b1_b5_comprehensive.py (B5 only)
- [ ] 1.7.3 — Monitor sensor ablation experiments
- [ ] 1.7.4 — Watch for modality-specific error patterns
- [ ] 1.7.5 — Wait for completion (expected ~60 min)
- [ ] 1.7.6 — Verify B5_sensor_ablation.json was created
- [ ] 1.7.7 — Parse RGB-only, RGB+Events, RGB+Events+LiDAR metrics

## STAGE 2: Results Analysis & Documentation

### Subtask 2.1: Parse All Benchmark Results
- [ ] 2.1.1 — Load and parse B1_mpc_solo.json
- [ ] 2.1.2 — Load and parse B2_vla_prediction.json
- [ ] 2.1.3 — Load and parse B3_dual_system.json
- [ ] 2.1.4 — Load and parse B4_mpc_baseline.json
- [ ] 2.1.5 — Load and parse B5_sensor_ablation.json
- [ ] 2.1.6 — Create summary table (benchmark name, success_rate, latency, error)

### Subtask 2.2: Validate Against Tech Spec §12
- [ ] 2.2.1 — Compare B1 metrics to expected values
- [ ] 2.2.2 — Compare B2 metrics to expected values
- [ ] 2.2.3 — Compare B3 metrics to expected values
- [ ] 2.2.4 — Compare B4 metrics to expected values
- [ ] 2.2.5 — Compare B5 modality impacts to expected trends

### Subtask 2.3: Document in PROGRESS.md
- [ ] 2.3.1 — Add timestamp and benchmark execution log entry
- [ ] 2.3.2 — Include actual metrics (success_rate %, latency ms, error rad)
- [ ] 2.3.3 — Log any anomalies or unexpected results
- [ ] 2.3.4 — Note any failures with root cause analysis
- [ ] 2.3.5 — Add decision logicfor Phase 12 → Phase 13 flow

## STAGE 3: Gate Validation (Gates 5-6)

### Subtask 3.1: Validate Gate 5 (SmolVLA)
- [ ] 3.1.1 — Extract server health status from benchmarks (should all pass)
- [ ] 3.1.2 — Extract action shape from B2/B3 outputs (should be [7])
- [ ] 3.1.3 — Extract latency stats from B2 (mean, max, min)
- [ ] 3.1.4 — Verify action_std is present in all responses
- [ ] 3.1.5 — Document Gate 5: PASS or FAIL with reasoning
- [ ] 3.1.6 — Update PROGRESS.md with Gate 5 status

### Subtask 3.2: Validate Gate 6 (Full System)
- [ ] 3.2.1 — Check B3 for any crashes or exceptions (should be 0)
- [ ] 3.2.2 — Extract actual success_rate from B3 (do NOT estimate)
- [ ] 3.2.3 — Calculate mean control latency from B3
- [ ] 3.2.4 — Verify MPC is running at 100-300 Hz (can check from logs)
- [ ] 3.2.5 — Document Gate 6: PASS or FAIL with reasoning
- [ ] 3.2.6 — Update PROGRESS.md with Gate 6 status

## STAGE 4: Phase Transition Decision

### Subtask 4.1: Success Rate Analysis
- [ ] 4.1.1 — If B3 success_rate >= 80%:  ✅ PROCEED to Stage 2 (Sensor Fusion)
- [ ] 4.1.2 — If B3 success_rate < 80%:  ⚠️ ANALYZE failure modes before proceeding
- [ ] 4.1.3 — Document decision in AGENT_STATE.md

### Subtask 4.2: Plan Stage 2 (Sensor Fusion)
- [ ] 4.2.1 — Review tech spec §8 (Multimodal Sensor Fusion)
- [ ] 4.2.2 — Design feature encoder architecture
- [ ] 4.2.3 — List required changes to VLA client for fusion
- [ ] 4.2.4 — Plan VLA retraining approach
- [ ] 4.2.5 — Update TODO for Phase 12.2

## QUALITY GATES

### Gate G1: No Fabricated Numbers
- ✅ All metrics come from actual benchmark JSON files
- ✅ No hand-calculated estimates
- ✅ No synthetic success rates

### Gate G2: No VLA Server Restarts
- ✅ Keep server running throughout all benchmarks (do NOT restart)
- ✅ Hot reload should handle minor changes
- ✅ If server dies, pause and restart once, log time lost

### Gate G3: All Results Logged
- ✅ Each benchmark result saved as JSON
- ✅ Each result documented in PROGRESS.md
- ✅ Summary table created comparing all benchmarks

### Gate G4: Every Failure Explained
- ✅ If benchmark fails: describe what failed and why
- ✅ Log exception traces from Python
- ✅ Recommend remediation

---

## Time Estimates
- B1 (MPC Solo):              ~5 minutes
- B2 (VLA Prediction):        ~30 minutes (server queries slow)
- B3 (Full Dual-System):      ~45 minutes
- B4 (MPC Baseline):          ~3 minutes
- B5 (Sensor Ablation):       ~60 minutes
- **Total Stage 1:**          **~2.5-3 hours**

- Results analysis:           ~30 minutes
- Gate validation:            ~20 minutes
- **Total Stage 2-3:**        **~50 minutes**

- **GRAND TOTAL:**            **~3.5-4 hours**

---

## Success Criteria

### For Phase 12 Stage 1 COMPLETE:
1. ✅ B1 executed and logged (MPC solo baseline metrics documented)
2. ✅ B2 executed and logged (VLA prediction accuracy documented)
3. ✅ B3 executed and logged (full system end-to-end metrics documented)
4. ✅ B4 executed and logged (MPC-only comparison available)
5. ✅ B5 executed and logged (sensor ablation impact documented)
6. ✅ Gates 5-6 validated (actual metrics, not estimated)
7. ✅ All results in PROGRESS.md (no missing numbers)
8. ✅ AGENT_STATE.md updated with Phase 13 decision

### For Phase 12 Stage 2 PREPARATION:
1. ✅ Sensor fusion feature requirements identified
2. ✅ Encoder architecture sketched
3. ✅ Retraining approach defined
4. ✅ Updated TODO for Phase 12.2 created
