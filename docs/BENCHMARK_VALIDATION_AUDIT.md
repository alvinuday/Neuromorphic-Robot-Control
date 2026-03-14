# Benchmark Validation & Clarification Document

**Date**: March 15, 2026
**Status**: CRITICAL CLARIFICATION NEEDED

---

## THE PROBLEM

We ran the benchmarks with **only 3-5 episodes each** (Phase 12), but the actual dataset has **800+ episodes**. Before proceeding to Phase 13 (sensor fusion), we need to:

1. ✅ **Understand what each benchmark actually tests**
2. ✅ **Clarify the differences between B1 and B4**
3. ✅ **Confirm B2 is pure VLA (not VLA+MPC)**
4. ⏳ **Run full dataset benchmarks** (not just 3-5 episodes)
5. ⏳ **Validate all metrics make sense**

---

## BENCHMARK DEFINITIONS (Based on Code Review)

### **B1: "Dataset Replay with MPC Solo"**

**What it actually does:**
- Takes a REAL DATASET EPISODE trajectory
- Loads the initial state from the dataset
- Uses MPC (Model Predictive Control) to **track that dataset trajectory**
- MPC does NOT use VLA at all
- Replays frame-by-frame from the dataset

**Input:** Real dataset trajectories (states, images, actions)
**Process:** MPC tries to match the dataset reference trajectory
**Metrics:** How well MPC tracks the dataset trajectories (tracking_error_rad)
**Success criteria:** Low tracking error = MPC can follow dataset paths

**Why:** Validates that MPC can accurately track real robot trajectories from the dataset

---

### **B2: "VLA Prediction Accuracy"**

**What it actually does:**
- Takes REAL DATASET IMAGES (from dataset frames)
- Feeds them to SmolVLA server
- Gets VLA's predicted actions
- Compares to GROUND TRUTH actions from the dataset
- Does NOT actually execute these actions (pure inference test)

**Input:** Dataset images + states
**Process:** Query VLA → Get predictions → Compare to ground truth
**Metrics:** 
- VLA prediction accuracy (MAE between predicted and ground truth actions)
- VLA latency
- Success rate (prediction errors < threshold)

**Why:** Validates that VLA can make reasonable action predictions on real dataset images

---

### **B3: "Full Dual-System (VLA + MPC)"**

**What it actually does:**
- Runs in SIMULATED environment (MuJoCo, not real dataset)
- For each step:
  1. Gets RGB observation from MuJoCo render
  2. Queries SmolVLA for action
  3. Uses MPC (or directly applies?) action to move robot
  4. Checks if object was lifted successfully
- This is an END-TO-END system test (not dataset-based)

**Input:** MuJoCo simulated states
**Process:** VLA queries + MPC execution in simulation
**Metrics:** 
- Success rate (object lifted?)
- Tracking error  
- VLA latency
- Control latency

**Why:** Validates that VLA + MPC together can solve a task in simulation

---

### **B4: "MPC-Only Baseline"**

**What it actually does:**
- Runs in SIMULATED environment (NOT dataset-based)
- Tries to track a SINUSOIDAL REFERENCE (not dataset trajectory)
- Simple PD/MPC controller
- No VLA at all
- No dataset involved

**Input:** Sinusoidal joint reference (just math, no real data)
**Process:** MPC tracks this synthetic reference
**Metrics:** Tracking error (how well MPC tracks sine wave)
**Success criteria:** Low steady-state error

**Why:** Provides a baseline for comparison (what's the best MPC can do on a simple task)

---

## THE KEY DIFFERENCES

| Aspect | B1 | B2 | B3 | B4 |
|--------|-----|-----|-----|-----|
| **Data source** | Real dataset | Real dataset | MuJoCo sim | MuJoCo sim |
| **Uses VLA?** | ❌ No | ✅ Yes | ✅ Yes | ❌ No |
| **Uses MPC?** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Reference** | Dataset traj | N/A (prediction) | VLA output | Sine wave |
| **Actual control?** | ✅ Tries to track | ❌ Just inference | ✅ Executes | ✅ Executes |
| **Validation** | Tracking accuracy | Prediction accuracy | Task success | Tracking accuracy |

---

## USER'S VALID CONCERNS

### ❌ **Problem 1: B1 and B4 seem similar**
**User is right:** Both say "MPC" but they're quite different:
- **B1 = Dataset-based tracking** (real trajectories)
- **B4 = Simulation-based sine tracking** (synthetic reference)
- They answer different questions:
  - B1: "Can MPC track real dataset trajectories?"
  - B4: "What's the baseline MPC performance on a simple synthetic task?"

### ❌ **Problem 2: B2 might not be just VLA**
**User is asking:** "Is B2 really just pure VLA?"
**Answer:** YES, in terms of **inference**, but:
- B2 queries VLA on real dataset images
- Compares predictions to ground truth
- Does NOT execute the predictions (no actual control)
- So it's "VLA prediction accuracy test" not "VLA execution test"

### ❌ **Problem 3: Only 3-5 episodes**
**User is right:** We ran:
- B1: 3 episodes
- B2: 3 episodes
- B3: 3 episodes
- B4: 2 episodes

**The dataset has 800 episodes** (or 7490 frames across 800 episodes = ~10 frames per episode on average)

**This means:**
- Our results are NOT statistically significant
- Could be lucky/unlucky runs
- Need to run 50-100+ episodes per benchmark to validate

---

## WHAT NEEDS TO HAPPEN BEFORE PHASE 13

### ✅ **Step 1: Clarify Dataset Structure** (COMPLETED)
- [x] **Dataset is lerobot/utokyo_xarm_pick_and_place (102 episodes, 7490 frames, [3,224,224] images)**
  - 102 total episodes (not 800)
  - Average 73.4 frames per episode
  - 224×224 RGB images (not 84×84)
- [x] **Average episode length: 73.4 frames**
  - Can run 80 episodes for training tests (B1/B2)
  - Remaining 22 episodes for test set (B4)
  - Max statistical validity: ~78% coverage

**Action**: DONE - Use 80 episodes for B1/B2, 22 for B4 test set

---

### ✅ **Step 2: Define Proper Benchmark Scope**
Based on dataset size, define:
- **B1**: How many dataset episodes to replay? (suggest 100+)
- **B2**: How many VLA predictions? (suggest 100+)
- **B3**: How many sim episodes? (suggest 50-100)
- **B4**: How many sine-tracking episodes? (suggest 20-30)

**Estimated time**: 
- B1 @ 5 sec/episode = 500 sec = 8 min
- B2 @ 30 sec/episode (VLA queries) = 3000 sec = 50 min
- B3 @ 20 sec/episode = 1000 sec = 17 min
- B4 @ 5 sec/episode = 150 sec = 3 min
- **Total: ~70-80 minutes**

---

### ✅ **Step 3: Re-run Benchmarks with Full Dataset**
Once dataset size is confirmed:
- Run B1-B4 with proper episode counts
- Save all 4 result JSONs
- Analyze metrics to ensure they're reasonable
- Validate:
  - B1 success rate should be >70% (real dataset, should work well)
  - B2 VLA latency should be 30-50ms
  - B3 success rate depends on MPC tuning (target >80%)
  - B4 baseline established for comparison

---

### ✅ **Step 4: Validate Each Benchmark's Logic**
Before accepting results:
- [ ] **B1**: Are we actually using dataset trajectories? Is MPC tracking them?
- [ ] **B2**: Are we comparing VLA predictions to ground truth? Getting reasonable metrics?
- [ ] **B3**: Is the sim environment working? Are episodes running to completion?
- [ ] **B4**: Is the sine-wave reference working? Reasonable MPC behavior?

---

### ✅ **Step 5: Only Then Proceed to Phase 13**
Once all above is complete:
- Benchmarks are validated on full dataset
- Metrics make sense
- No anomalies or failures
- Then start Phase 13 (sensor fusion ablation)

---

## RECOMMENDED ACTIONS (IN ORDER)

### 🔴 **ACTION 1: Confirm Dataset Size**
```bash
python3 scripts/inspect_dataset.py
# Should output:
# - Total episodes
# - Total frames
# - Avg frames per episode
# - Episode length distribution
```

**Expected output:**
- Episodes: 800 or 7490(?)
- Frames: proportional to episodes
- Avg length: determines benchmark size

**Time**: 2-5 minutes

---

### 🔴 **ACTION 2: Update Benchmark Parameters**
Once dataset size is known:
- Update `scripts/phase12_run_benchmarks.py` to use proper episode counts
- B1: 100 episodes (or scale)
- B2: 100 episodes (or scale)
- B3: 50 episodes
- B4: 30 episodes

**Time**: 5 minutes

---

### 🟡 **ACTION 3: Run Full Benchmarks**
Execute benchmarks on complete dataset:
```bash
python3 scripts/phase12_run_benchmarks.py
```

**Expected time**: 60-90 minutes (depends on dataset size + VLA latency)

**Monitor**: 
- VLA server health
- Per-episode latency
- Any crashes or errors

---

### 🟡 **ACTION 4: Validate Results**
Once benchmarks complete:
```bash
python3 -c "
import json
for b in ['B1', 'B2', 'B3', 'B4']:
    data = json.load(open(f'evaluation/results/{b}_*.json'))
    summary = data['summary']
    print(f'{b}: success={summary[\"success_rate\"]:.1%}, error={summary[\"mean_tracking_error_rad\"]:.6f}')
"
```

**Sanity checks**:
- No success_rate should be 0% (indicates broken benchmark)
- No tracking_error should be 0.0 (indicates data not being collected)
- B2 latency should be 30-50ms (VLA inference time)
- All metrics should be reasonable

---

### 🟢 **ACTION 5: Document & Approve**
Once validated:
- [ ] Update PROGRESS.md with full dataset results
- [ ] Document dataset size and episode counts
- [ ] Confirm all 4 benchmarks are working correctly
- [ ] Then proceed to Phase 13

---

## PHASE 13 DECISION LOGIC

**DO NOT START PHASE 13 UNTIL:**
- ✅ Dataset size confirmed
- ✅ B1-B4 benchmarks run on FULL dataset (100+ episodes each)
- ✅ All metrics pass sanity checks
- ✅ No anomalies or failures
- ✅ B1-B4 results documented and approved

**If any of the above fails:**
- Stop, investigate root cause
- Fix benchmark logic or code
- Re-run with fixes
- Only then proceed

---

## SUMMARY

**Current State**: Phase 12 (Benchmarking) is INCOMPLETE
- Only ran 3-5 episodes per benchmark
- Dataset has 800+ episodes available
- Need to run full dataset before validation
- Phase 13 blocked until Phase 12 is properly validated

**Next Action**: 
1. Confirm dataset size
2. Update benchmark scripts
3. Run on full dataset (60-90 min)
4. Validate results
5. Then proceed to Phase 13

**Timeline**:
- Dataset inspection: 5 min
- Script updates: 5 min
- Benchmark execution: 60-90 min
- Validation & docs: 15 min
- **Total: ~90-120 minutes before Phase 13 starts**

---

**User Request Acknowledged**: "ensure each of the benchmark makes sense, and is correct, and properly being executed, i need more transparency as to whats happening. Once you run the whole dataset with all the episodes then lets continue with phase 13"

**Plan**: Will do exactly that. No Phase 13 until Phase 12 is properly validated on full dataset.
