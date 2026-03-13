# Honest Benchmarking Plan
## MPC Solvers & VLA Integration

**Status**: Starting Fresh  
**Goal**: Rigorous, unbiased comparison of ALL available solvers  
**Methodology**: Identical problem formulations, proper measurement, full transparency

---

## 1. Audit of Available Solvers

### 1.1 StuartLandauLagrangeDirect (SL MPC)
- **File**: `src/solver/stuart_landau_lagrange_direct.py`
- **Type**: Continuous-time constrained optimization via Arrow-Hurwicz saddle-point
- **Problem**: min ||Px + q||² s.t. Cx = d, l ≤ Ax ≤ u
- **Status**: ✓ IMPLEMENTED - MAIN SOLVER
- **Test needed**: Performance on MPC horizon problems

### 1.2 OSQP Solver Wrapper
- **File**: `src/solver/osqp_solver.py`
- **Type**: Open-source quadratic programming
- **Problem**: min 0.5 x^T Q x + p^T x  s.t. l ≤ Ax ≤ u
- **Status**: ✓ IMPLEMENTED - May be available
- **Test needed**: Compare against SL on identical QP problems

### 1.3 Phase4MPCController
- **File**: `src/solver/phase4_mpc_controller.py`
- **Type**: Receding horizon MPC wrapper around SL solver
- **Problem**: Full MPC trajectory optimization
- **Status**: ✓ IMPLEMENTED - For 2-DOF systems
- **Test needed**: Performance on 2-DOF trajectory problems

### 1.4 AdaptiveMPCController
- **File**: `src/solver/adaptive_mpc_controller.py`
- **Type**: Generic MPC wrapper (DOF-agnostic)
- **Problem**: Full MPC for any robot DOF
- **Status**: ✓ IMPLEMENTED - But needs testing
- **Test needed**: Works with 6-DOF systems? What solver does it use?

### 1.5 Stuart-Landau Variants
- `stuart_landau_lagonn.py` - Lagrange + NN variant
- `stuart_landau_lagonn_full.py` - Full version
- `stuart_landau_3dof.py` - 3-DOF specific
- **Status**: NEED TO AUDIT - Unclear which is current

---

## 2. Test Problem Formulations

### 2.1 Small QP (Quick Unit Test)
```
min ||x||² subject to -1 ≤ x ≤ 1
Problem: (10-variable toy)
```
**Solvers to test**: SL, OSQP  
**Expected**: Both should solve in microseconds

### 2.2 Medium QP (MPC Step)
```
min τ_x ||x_k - x_ref||² + τ_u ||u_k||²  
subject to: x_{k+1} = A x_k + B u_k (implicit)
            -u_max ≤ u_k ≤ u_max
Problem: 40-50 variable, 20+ constraints
```
**Solvers to test**: SL, OSQP  
**Expected**: < 10ms per solve for real-time

### 2.3 Full MPC Horizon Problem (2-DOF)
```
Receding horizon N=10, x ∈ R⁴, u ∈ R²
min: sum of tracking costs + control costs
s.t: dynamics + bounds
```
**Solvers to test**: Phase4MPC (uses SL internally)  
**Expected**: Should solve in milliseconds

### 2.4 Full MPC Horizon Problem (6-DOF)  
```
Receding horizon N=20, x ∈ R¹², u ∈ R⁶
min: sum of tracking costs + control costs
s.t: dynamics + bounds
```
**Solvers to test**: AdaptiveMPC (uses ?)  
**Expected**: Should solve in milliseconds

### 2.5 VLA + MPC Integration
```
Vision-Language Model predicts action → MPC trajectory
```
**Solvers to test**: SL MPC with SmolVLA server  
**Expected**: Image encoding + inference + MPC < 100ms

---

## 3. Benchmarking Protocol

### 3.1 For Each Solver
1. **Initialize** once before benchmark
2. **Run** on identical N=100 random problem instances
3. **Measure** wall-clock time (no overhead)
4. **Record** all samples (not just mean)
5. **Report** mean, median, std, min, max, P95, P99

### 3.2 Fairness Rules
- ✓ All solvers solve THE SAME problem
- ✓ Solver setup time measured separately
- ✓ No warm-starting one solver and not others
- ✓ No simplified problems for some solvers
- ✓ Same tolerance/accuracy targets for all
- ✓ Run on same machine, same CPU conditions

### 3.3 What I Will NOT Do This Time
- ❌ Create fake toy problems
- ❌ Include setup/initialization in solve time
- ❌ Use different problem formulations per solver
- ❌ Make claims without experimental validation
- ❌ Apply misleading statistical tricks

---

## 4. VLA Integration Testing

### 4.1 Components to Test
1. **Image Encoding**: RGB array → base64 PNG (using PIL)
2. **SmolVLA Server**: Send request, measure latency
3. **MPC Solver**: Convert VLA output → trajectory
4. **Full Pipeline**: Image → VLA → MPC → control signal

### 4.2 Latency Budget (100 Hz = 10ms per cycle)
- Image acquisition: ~5ms
- Image encoding: < 1ms (target)
- VLA inference: < 100ms (external server)
- MPC solve: < 10ms (per MPC step)
- **TOTAL**: Should fit in 100-200ms for non-critical control

### 4.3 Tests to Run
1. Unit test: Image encoding (PIL) - 1000x
2. Integration test: VLA server connectivity
3. End-to-end: Dummy image → VLA → MPC trajectory
4. Performance: Measure all components on real LSMO task

---

## 5. Execution Plan

### Phase A: Solver Audit & Validation
- [ ] Read and understand each solver
- [ ] Write simple unit tests for each
- [ ] Document actual capabilities
- [ ] Identify which solver each high-level controller uses

### Phase B: Fair Benchmarking
- [ ] Implement identical QP test suites
- [ ] Run SL vs OSQP on small/medium/large problems
- [ ] Document assumptions and fairness
- [ ] Generate honest comparison report

### Phase C: Full MPC Testing
- [ ] Test Phase4MPC on 2-DOF problems
- [ ] Test AdaptiveMPC on 6-DOF problems
- [ ] **Test SL + OSQP properly** - no more fake comparisons
- [ ] Document which is faster for what workload

### Phase D: VLA Integration Testing
- [ ] Implement PIL image encoding (proper)
- [ ] Test SmolVLA server connectivity
- [ ] Measure end-to-end latency
- [ ] Validate on real LSMO trajectory data

### Phase E: Final Report
- [ ] Generate honest benchmark report
- [ ] Clear performance metrics
- [ ] Recommendations for deployment
- [ ] Documentation for your thesis

---

## 6. Success Criteria

✓ **Honest**: All claims backed by actual measurements  
✓ **Complete**: All solvers tested fairly  
✓ **Reproducible**: Clear methodology, code published  
✓ **Critical**: Identify any real weaknesses  
✓ **Practical**: Answer "what should I use for deployment?"

---

## 7. Known Issues to Investigate

1. **AdaptiveMPC solver backend**: Uses what? SL or something else?
2. **OSQP availability**: Is osqp package installed?
3. **VLA server state**: Still online? correct endpoint?
4. **Image encoding**: Need PIL for PNG compression
5. **6-DOF vs 2-DOF**: Do all solvers support both?

---

**Next Step**: Start Phase A - Audit the solvers properly

Generated: 2026-03-14  
Status: CLEAN START - No fake data, proper methodology
