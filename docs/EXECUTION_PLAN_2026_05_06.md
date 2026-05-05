# Neuromorphic MPC Verification & Implementation Plan
**Date**: May 6, 2026 | **Status**: EXECUTING  
**Goal**: Verify and cross-validate all calculations (MD file ↔ Webapp), implement complete SNN infrastructure, benchmark against OSQP, deliver validated documentation

---

## Executive Summary

This plan systematically validates the neuromorphic robot control pipeline:
1. **Phase 1**: Extract and verify webapp implementation details
2. **Phase 2**: Parse and hand-calculate all derivations in SNN_MPC_Complete_Derivation.md
3. **Phase 3**: Validate calculation consistency between MD file and webapp code
4. **Phase 4**: Complete/optimize SNN implementation (PIPG + LIF neurons + oscillators)
5. **Phase 5**: Run comprehensive benchmarks (OSQP vs SNN) with timing, accuracy, energy proxies
6. **Phase 6**: Adjust MD file values to match webapp (ground truth) & improve explanations
7. **Phase 7**: Generate comprehensive report with results, plots, validation tables

---

## Phase 1: Extract Webapp Implementation Details
**Status**: PENDING

### 1.1 Webapp QP Building
- [ ] Read `/webapp/server.py` fully (POST /api/build, /api/solve endpoints)
- [ ] Extract MPC matrices generation (Q, p, A_eq, b_eq, A_ineq) for 2-DOF arm
- [ ] Verify linearization point (θ* = [45°, 45°])
- [ ] Extract horizon N, time step dt, arm parameters (m1, m2, l1, l2, g)
- [ ] Document exact matrix dimensions and structure

### 1.2 Webapp Solver Integration
- [ ] Map solver options (OSQP, Stuart-Landau, others)
- [ ] Extract solution decomposition logic (_decompose_z)
- [ ] Find KKT verification implementation
- [ ] Document solve timing instrumentation

### 1.3 Benchmark Results
- [ ] Parse all JSON files in evaluation/results/
- [ ] Extract B5_solver_comparison (OSQP vs SL timing data)
- [ ] Document condition numbers, solve times, constraint violations

---

## Phase 2: Hand Calculate All Derivations
**Status**: PENDING

### 2.1 Robot Physics (Section 1-2 of MD)
- [ ] Verify Lagrangian derivation matches webapp
- [ ] Compute M(θ*), C(θ*, 0), G(θ*) numerically
- [ ] Check linearization Jacobians (A_c, B_c)
- [ ] Discretize to A_d, B_d with dt=0.02s

### 2.2 MPC Formulation (Section 3-4 of MD)
- [ ] Build Toeplitz lifting matrices for N=10 horizon
- [ ] Derive condensed H matrix and f vector from cost function
- [ ] Construct inequality constraints (input saturation, joint limits)
- [ ] Compare hand-calculated H, f, A_ineq with webapp output

### 2.3 KKT Conditions (Section 5 of MD)
- [ ] Derive KKT stationarity, complementarity, dual feasibility
- [ ] Implement KKT verification: residuals, constraint violations, duality gap
- [ ] Compare hand-verified KKT against webapp solver output

### 2.4 PIPG Algorithm (Section 6 of MD)
- [ ] Trace PIPG iterations: gradient step → projection → Lagrange update
- [ ] Document parameter choices (step sizes μ, α, β)
- [ ] Verify convergence criteria (tolerance ε, max iterations)

### 2.5 SNN Dynamics (Section 7 of MD)
- [ ] Map PIPG equations to LIF neuron spikes and membrane potentials
- [ ] Verify phase/amplitude encoding of dual variables
- [ ] Check constraint handling (equality → phase, inequality → amplitude)

### 2.6 Full Hand Calculation Example (Section 8 of MD)
- [ ] Walk through 5 PIPG iterations manually
- [ ] Track convergence: ||∇L||, constraint violations, ||ΔU||
- [ ] Compare to numerical reference solution

---

## Phase 3: Cross-Validate MD File ↔ Webapp
**Status**: PENDING

### 3.1 Numerical Values Verification
- [ ] Extract parameters from MD: m1, m2, l1, l2, g, θ*, N, dt
- [ ] Compare vs webapp default config
- [ ] Verify all computed matrices (M*, A_c, B_c, etc.) match
- [ ] Document any discrepancies and resolution

### 3.2 Example Trajectories
- [ ] Run MD's numerical example (initial x0=[0,0,0,0], goal x*=[π/4, π/4, 0, 0])
- [ ] Compare predicted trajectory vs webapp simulation
- [ ] Verify end-effector positions (FK)
- [ ] Check input torques and constraint satisfaction

### 3.3 Solver Output Validation
- [ ] Run OSQP on MD's QP matrices
- [ ] Extract solution ΔU*=[u0, u1, ..., u9]
- [ ] Verify KKT conditions
- [ ] Compare solver time, iterations, accuracy vs webapp logging

### 3.4 Create Verification Table
- [ ] Document all values and cross-references
- [ ] Flag any MD ↔ webapp differences
- [ ] Prepare corrections

---

## Phase 4: Complete/Optimize SNN Implementation
**Status**: PENDING

### 4.1 Review Existing Solvers
- [ ] Examine `src/solver/stuart_landau_lagonn_full.py` (current implementation)
- [ ] Verify PIPG algorithm correctness
- [ ] Check ODE integration method (RK45, step size)
- [ ] Document current solve times, convergence behavior

### 4.2 SNN Dynamics (LIF + Oscillators)
- [ ] Implement/verify LIF neuron model: dv/dt = (I - v)/τ + spike reset
- [ ] Implement oscillator dynamics for decision variables (complex amplitude)
- [ ] Verify constraint enforcement: equality→phase, inequality→amplitude
- [ ] Add noise/robustness modeling (optional: Poisson spike trains)

### 4.3 Gradient & Projection Neurons
- [ ] Implement gradient computation neurons (∇L computation)
- [ ] Implement projection neurons for constraints
- [ ] Verify iterative updates match PIPG equations
- [ ] Add convergence monitoring

### 4.4 Optimization & Tuning
- [ ] Tune ODE solver parameters (RK45 tolerance, step size)
- [ ] Implement adaptive annealing schedule (τ(t))
- [ ] Add early stopping (convergence detection)
- [ ] Target solve time: 100–500ms (balance accuracy vs neuromorphic fidelity)

### 4.5 Modular SNN Architecture
```
snn_solver.py
├── lif_neuron.py          # LIF model, spike generation
├── oscillator_dynamics.py # Complex amplitude, phase encoding
├── gradient_layer.py      # Computes ∇L from current state
├── projection_layer.py    # Projects onto constraint sets
├── pipg_flow.py           # PIPG recurrent dynamics
└── snn_solver_main.py     # ODE integrator, convergence monitoring
```

---

## Phase 5: Comprehensive Benchmarking
**Status**: PENDING

### 5.1 Benchmark Suite Design
- [ ] Create 100 random QP instances (mix well-conditioned & ill-conditioned)
- [ ] Vary problem sizes: n ∈ {20, 40, 80, 160} (decision variables)
- [ ] Vary condition numbers: κ ∈ {10, 100, 1000} (HLS spectrum)
- [ ] Ground truth: OSQP (double precision)

### 5.2 Run Solvers
- [ ] **OSQP**: Measure wall-clock time, iterations, KKT residuals
- [ ] **SNN (Stuart-Landau)**: Measure solve time, trajectory convergence
- [ ] **Proposed SNN (optimized)**: Same metrics
- [ ] Ensure all produce valid solutions (KKT feasible)

### 5.3 Metrics Collection
- [ ] **Accuracy**: ||z_snn - z_osqp||_2 / ||z_osqp||_2 (relative error)
- [ ] **Timing**: Wall-clock solve time (ms)
- [ ] **Iterations**: Number of PIPG steps / OSQP iterations to convergence
- [ ] **Feasibility**: Max constraint violation, KKT residuals
- [ ] **Scalability**: Time vs problem size (n, κ)

### 5.4 Comparison vs Research Paper
- [ ] Extract Mangalore et al. benchmark data (Table 1, Fig 3)
- [ ] Compare SNN solve time, accuracy, energy-delay product
- [ ] Document advantages/limitations of our implementation

### 5.5 Generate Plots
- [ ] Solve time vs problem size (n)
- [ ] Solve time vs condition number (κ)
- [ ] Accuracy vs solve time (Pareto frontier)
- [ ] Convergence curves (residuals vs iteration)

---

## Phase 6: Validate & Adjust MD File
**Status**: PENDING

### 6.1 Numerical Value Audit
- [ ] Go through MD section by section
- [ ] Verify each computed value (M*, A_c, B_c, H, f, etc.)
- [ ] Check against webapp ground truth
- [ ] Identify deviations (rounding, precision, formula errors)

### 6.2 Formula Verification
- [ ] Verify all KKT equations
- [ ] Check PIPG recurrence formulas
- [ ] Validate hand-calculation example (5 iterations)

### 6.3 Update Values (Preserve Explanations)
- [ ] If discrepancies found, update numerical values only
- [ ] Keep all explanations intact
- [ ] Enhance explanations where helpful (cross-references, intuition)
- [ ] Add "Verification Status" checkmarks

### 6.4 Add Cross-References
- [ ] Link to webapp implementation
- [ ] Reference benchmark results
- [ ] Cite benchmark plots/tables from Phase 5

---

## Phase 7: Generate Comprehensive Report
**Status**: PENDING

### 7.1 Verification Report
- [ ] Executive summary: all checks passed/failed
- [ ] Detailed findings: MD vs webapp
- [ ] Cross-validation tables with all numerical values
- [ ] Identified issues and resolutions

### 7.2 Benchmark Report
- [ ] Benchmark methodology
- [ ] Problem instances (size distribution, condition numbers)
- [ ] Results tables (OSQP vs SNN: time, accuracy, iterations)
- [ ] Plots (convergence, scalability, Pareto frontier)
- [ ] Comparison vs Mangalore et al. (qualitative discussion)

### 7.3 Implementation Documentation
- [ ] SNN architecture diagram
- [ ] PIPG algorithm pseudo-code
- [ ] LIF neuron equations
- [ ] Constraint encoding scheme
- [ ] Tuning parameter guide

### 7.4 Updated MD File
- [ ] Corrected numerical values
- [ ] Enhanced explanations (as needed)
- [ ] New section: "Webapp Implementation" with endpoint descriptions
- [ ] New appendix: "Benchmark Results" with plots & tables

---

## Deliverables

| Item | Location | Purpose |
|------|----------|---------|
| **Verification Report** | docs/VERIFICATION_REPORT_2026_05_06.md | Cross-validation results |
| **Benchmark Report** | docs/BENCHMARK_RESULTS_2026_05_06.md | SNN vs OSQP metrics |
| **Updated MD File** | SNN_MPC_Complete_Derivation.md | Corrected + enhanced |
| **SNN Source Code** | src/solver/snn_solver_*.py | Modular implementation |
| **Benchmark Plots** | docs/plots/ | convergence, scalability, Pareto |
| **Test Suite** | tests/test_snn_calculations.py | Validation tests |

---

## Timeline

| Phase | Tasks | Est. Time |
|-------|-------|-----------|
| 1 | Extract webapp details | 1 hour |
| 2 | Hand calculations | 3–4 hours |
| 3 | Cross-validation | 2 hours |
| 4 | SNN implementation | 4–5 hours |
| 5 | Benchmarking | 2–3 hours |
| 6 | MD file audit & fixes | 1–2 hours |
| 7 | Report generation | 2 hours |
| **Total** | **~18–20 hours** | |

---

## Success Criteria

- ✅ All MD numerical values verified against webapp
- ✅ Hand calculations trace through 5 PIPG iterations correctly
- ✅ SNN solver produces valid (KKT feasible) solutions
- ✅ SNN accuracy within 1e-3 relative error of OSQP on test problems
- ✅ Benchmark suite runs 100+ instances, documents timing/accuracy
- ✅ Updated MD file integrates webapp details without redundancy
- ✅ All deliverables completed and validated

---

## Notes

- **Ground Truth**: Webapp implementation (evaluate/ + benchmark results) is treated as reference
- **No Hallucinations**: All calculations verified numerically; no estimated values
- **Modular SNN**: Can be extended later to real Loihi 2 or other neuromorphic hardware
- **Documentation**: Enhanced MD file serves as both theory reference and implementation guide

---

*Status Updates will be appended below as phases complete.*
