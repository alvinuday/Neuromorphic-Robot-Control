# ⚠️ CRITICAL FINDINGS: Dataset & Benchmark Audit

**Date**: March 15, 2026
**Status**: PHASE 12 INCOMPLETE - MAJOR DISCREPANCIES FOUND

---

## 📊 ACTUAL DATASET vs WHAT WE THOUGHT

### **What We Thought We Were Using**
```
Dataset:  lerobot/xarm_lift_medium
Episodes: 800
Frames:   20000
Images:   [3, 84, 84]
State:    [4]
```

### **What We Actually Loaded** ✅ (from inspect_dataset.py, NOW OFFICIAL)

```
Dataset:  lerobot/utokyo_xarm_pick_and_place
Episodes: 102 (CONFIRMED - this is our actual dataset)
Frames:   7490 (avg 73.4 per episode)
Images:   [3, 224, 224] ← Main camera images
State:    [8] dimensions (6-DOF xArm + gripper)
Action:   [7] dimensions (6-DOF xArm + gripper command)
```

### **Key Facts - NOW CORRECTED**
| Aspect | ACTUAL (Now Using) | Old Tech Spec (REMOVED) | Status |
|--------|----------|--------|-------|
| Dataset | utokyo_xarm_pick_and_place | xarm_lift_medium | ✅ CORRECTED |
| Episodes | 102 | 800 (hypothetical) | ✅ CORRECTED |
| Frames | 7,490 | 20,000 (hypothetical) | ✅ CORRECTED |
| Image size | [3, 224, 224] | [3, 84, 84] (old spec) | ✅ CORRECTED |
| State dim | 8 | 4 (old spec) | ✅ CORRECTED |
| Action dim | 7 | N/A (old spec) | ✅ CORRECTED |

---

## 🔴 ISSUES WITH PHASE 12 BENCHMARKS (3-5 episodes)

### **Issue 1: Insufficient Coverage**
```
B1: 3 episodes / 102 available  = 2.9%   COVERAGE
B2: 3 episodes / 102 available  = 2.9%   COVERAGE ❌
B3: 3 episodes (sim)             N/A
B4: 2 episodes (sim)             N/A

Statistical Significance: INVALID
Could be luck/bad luck, not representative
```

### **Issue 2: Image Size Mismatch**
- Dataset provides: **[3, 224, 224]**
- VLA probably expects: 84×84 or 224×224 (depends on SmolVLA model)
- Current code: May silently resize/crop, leading to unpredictable behavior
- B2 accuracy metrics: Possibly invalid due to image preprocessing

### **Issue 3: Dataset Reference Mismatch**
- Code says: `lerobot/xarm_lift_medium`
- Actually loaded: `lerobot/utokyo_xarm_pick_and_place`
- These are DIFFERENT datasets with different characteristics

### **Issue 4: B1 and B4 Distinction Still Unclear**
```
What we claimed:
  B1 = "Dataset Replay with MPC"
  B4 = "MPC-only Baseline"

Reality check:
  B1 = Tracks REAL dataset trajectories (makes sense)
  B4 = Tracks SINE WAVE in simulation (different task)
```
✅ These ARE different benchmarks, but B4 is NOT a fair comparison for B1

---

## ✅ WHAT BENCHMARKS SHOULD ACTUALLY TEST

### **B1: Dataset Replay with MPC Solo**
```
Purpose:    Validate that MPC can track real robot trajectories
Input:      102 dataset episodes with state/action trajectories
Process:    Extract trajectory from dataset, use MPC to track it
Metrics:    Tracking error (lower = better MPC)
Episodes:   Use 80 out of 102 (leave 22 for testing)
Time:       ~6-7 minutes (80 episodes × 5 sec)
```

### **B2: VLA Prediction Accuracy**
```
Purpose:    Validate that SmolVLA makes reasonable predictions on real images
Input:      Real dataset images [3, 224, 224] + states
Process:    Query VLA → Get predictions → Compare to ground truth
Metrics:    Prediction error, VLA latency, inference accuracy
Episodes:   80 out of 102 (sample frames from each)
Time:       ~40 minutes (80 episodes × VLA latency ~30sec/ep)
Notes:      MUST handle image size: [3, 224, 224] properly
```

### **B3: Full Dual-System (VLA + MPC)**
```
Purpose:    Test VLA + MPC integration on simulated lift task
Input:      MuJoCo simulation (NOT dataset-based)
Process:    VLA queries + MPC execution on lift task
Metrics:    Success rate, tracking error, control latency
Episodes:   50 in simulation
Time:       ~17 minutes
Status:     ✅ Makes sense, NOT a dataset comparison
```

### **B4: MPC-Only Baseline**
```
Purpose:    Establish baseline for MPC performance alone (NO VLA)
Input:      Sine-wave reference in MuJoCo simulation
Process:    MPC tracks artificial reference curve
Metrics:    Tracking error on sine wave
Episodes:   30 in simulation
Time:       ~2-3 minutes
Status:     ✅ Makes sense as a baseline
Caveat:     NOT directly comparable to B1 (different task, sim vs real)
```

---

## 🔴 WHAT WENT WRONG IN PHASE 12

### **Root Causes**
1. **Incomplete verification**: Ran with 3-5 episodes without checking full dataset
2. **Assumed dataset size**: Thought 800 episodes available, actually only 102
3. **Didn't validate image format**: Assumed 84×84, actually 224×224
4. **Didn't check dataset name**: Code referenced different dataset than actually loaded
5. **No transparency logging**: Made claims without actually verifying each component

### **Why This Matters for Phase 13**
- Cannot proceed to sensor fusion ablation with statistically invalid benchmarks
- Need proper B1-B4 baselines before testing multimodal fusion improvements
- Image size mismatch will break fusion encoder expectations
- 102-episode limit is actually fine, but need to use ALL of them

---

## ✅ CORRECTED BENCHMARK PLAN

### **New Benchmark Script**
- **File**: `scripts/phase12_run_benchmarks_corrected.py`
- **Status**: CREATED ✅
- **Key changes**:
  - ✅ Uses actual dataset: `lerobot/utokyo_xarm_pick_and_place`
  - ✅ Sets proper episode counts: B1=80, B2=80, B3=50, B4=30
  - ✅ Handles [3, 224, 224] images correctly
  - ✅ Validates state [8] and action [7] dimensions
  - ✅ Proper logging and transparency

### **Execution Plan**

```
Step 1: Start VLA server (if not already running)
  python3 vla/vla_production_server.py &
  (Allow 200s warmup)

Step 2: Run corrected benchmarks
  python3 scripts/phase12_run_benchmarks_corrected.py
  (Estimated: 70-80 minutes total)
  
Step 3: Validate results
  - Check all 4 JSON files created
  - Verify metrics make sense (no NaN, not all zero)
  - B1: Success > 70%, error reasonable
  - B2: VLA latency 20-50ms
  - B3: Success rate depends onMPC (target >60%)
  - B4: Baseline established

Step 4: Document findings
  - Update PROGRESS.md with actual metrics
  - Record dataset verification
  - Confirm all episode counts
  
Step 5: THEN proceed to Phase 13
  - Only after B1-B4 validated
  - With full dataset coverage (80+ episodes each)
  - Proper image formats verified
```

---

## 📋 VALIDATION CHECKLIST (Before Phase 13)

- [ ] **Dataset confirmed**: lerobot/utokyo_xarm_pick_and_place, 102 episodes
- [ ] **Image format verified**: [3, 224, 224] uint8 or float32
- [ ] **B1 completed**: 80 dataset episodes, tracking error logged
- [ ] **B2 completed**: 80 VLA predictions, latency verified (20-50ms)
- [ ] **B3 completed**: 50 sim episodes, success rate recorded
- [ ] **B4 completed**: 30 baseline episodes, error baseline established
- [ ] **All JSON files created**: B1, B2, B3, B4 result files
- [ ] **Metrics reasonable**: No NaN, not all zeros, within expected ranges
- [ ] **Coverage adequate**: 80/102 episodes = 78% coverage for real dataset
- [ ] **Comparison valid**: B3 vs B4 improvement quantified (should show VLA helps)
- [ ] **Documentation complete**: PROGRESS.md updated with actual data
- [ ] **No anomalies**: All benchmarks completed without blocking errors

---

## ⏭️ NEXT ACTIONS (PRIORITY ORDER)

### 🔴 **ACTION 1: Run Corrected Benchmarks** (BLOCKING)
```bash
# Option A: If VLA still running from Phase 12
python3 scripts/phase12_run_benchmarks_corrected.py

# Option B: Start fresh
python3 vla/vla_production_server.py &
sleep 200s  # Wait for warmup
python3 scripts/phase12_run_benchmarks_corrected.py
```
**Estimated time**: 70-80 minutes  
**Output**: 4 JSON files in `evaluation/results/B{1-4}_*.json`

---

### 🟡 **ACTION 2: Validate Results** (POST-EXECUTION)
```bash
python3 << 'EOF'
import json
import numpy as np

for b in ['B1', 'B2', 'B3', 'B4']:
    try:
        name_map = {
            'B1': 'dataset_replay_mpc_solo',
            'B2': 'vla_prediction_accuracy',
            'B3': 'full_dual_system',
            'B4': 'mpc_only_baseline'
        }
        fn = f'evaluation/results/{b}_{name_map[b]}.json'
        data = json.load(open(fn))
        s = data['summary']
        
        print(f"\n{b}: {s['benchmark']}")
        print(f"  Episodes: {s['n_episodes']}")
        print(f"  Success: {s['success_rate']*100:.1f}%")
        print(f"  Error: {s['mean_tracking_error_rad']:.6f} rad")
        if s['mean_vla_latency_ms'] > 0:
            print(f"  VLA Latency: {s['mean_vla_latency_ms']:.1f} ms")
    except Exception as e:
        print(f"{b}: ERROR - {e}")
EOF
```

---

### 🟢 **ACTION 3: Document Findings**
Update `docs/agent/PROGRESS.md`:
```markdown
## Phase 12 (CORRECTED RUN)

**Date**: March 15, 2026 [TIME]
**Dataset**: lerobot/utokyo_xarm_pick_and_place (102 episodes, 7490 frames)
**Episode Coverage**: B1=80/102 (78%), B2=80/102 (78%), B3=50 sim, B4=30 sim

**Results**:
- B1 (Dataset Replay MPC): [SUCCESS RATE] success, [ERROR] rad error
- B2 (VLA Prediction): [SUCCESS RATE] success, [LATENCY] ms VLA latency  
- B3 (Dual-System): [SUCCESS RATE] success, [ERROR] rad error, [LATENCY] ms
- B4 (MPC Baseline): [SUCCESS RATE] success, [ERROR] rad error

**Gates Validated**:
- Gate 5 (SmolVLA): [PASS/FAIL] - Server health, latency, action shape
- Gate 6 (Full System): [PASS/FAIL] - No crashes, success rate, latency

**Key Findings**:
- [List any anomalies or interesting results]
- [Compare B3 vs B4 improvement]
- [Note any image format issues]

**Approval**:  [YES/NO] - Ready for Phase 13?
```

---

### 🟢 **ACTION 4: THEN Proceed to Phase 13**
Once above is complete:
- ✅ Phase 12 properly validated
- ✅  Baseline benchmarks established with full dataset coverage
- ✅ Ready to design sensor fusion ablation (Phase 13)

---

## 💡 KEY TAKEAWAYS

1. **Dataset We're Using**: `lerobot/utokyo_xarm_pick_and_place` (102 episodes, 224×224 images)
2. **Episode Coverage**: Need 80+ per benchmark (not 3-5)
3. **Benchmarks Make Sense**:
   - B1 = Dataset tracking (real)
   - B2 = VLA inference (real images)
   - B3 = Full system sim (controlled test)
   - B4 = MPC baseline (reference)
4. **Phase 13 Blocked**: Until Phase 12 is properly validated
5. **Timeline**: ~2 hours more work (80 min benchmarks + 30 min validation)

---

## 🎯 PHASE 13 PREREQUISITE

**We will NOT start Phase 13 until:**
- ✅ B1-B4 benchmarks run on full available dataset (80+ episodes each)
- ✅ All 4 JSON results valid and pass sanity checks
- ✅ Image format verified (224×224 → SmolVLA compatibility)
- ✅ Metrics documented and approved
- ✅ User confirms: "Go ahead with Phase 13"

---

**Prepared by**: Copilot  
**Status**: WAITING FOR BENCHMARK EXECUTION  
**Next Check**: After corrected benchmarks complete
