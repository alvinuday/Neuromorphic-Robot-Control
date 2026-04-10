# Neuromorphic MPC for 2-DOF Arm (Code-Grounded Version)
### Stuart-Landau Intuition, Implemented Equations, Variants, and Validation Protocol (No-Fallback Audit)

> This document is intentionally aligned to the current repository implementation.
> Ground truth is code behavior, not older theoretical drafts.

---

## Table of Contents

1. [What Changed and Why](#1-what-changed-and-why)
2. [What Oscillators Are (Beginner-Friendly)](#2-what-oscillators-are-beginner-friendly)
3. [Implemented 2-DOF Dynamics (Exact Code Model)](#3-implemented-2-dof-dynamics-exact-code-model)
4. [Implemented MPC QP Formulation (Exact Code Form)](#4-implemented-mpc-qp-formulation-exact-code-form)
5. [Implemented Solver Variants and Their Status](#5-implemented-solver-variants-and-their-status)
6. [Canonical Baseline and Validation Rule](#6-canonical-baseline-and-validation-rule)
7. [End-to-End 2-DOF Numerical Check You Can Do by Calculator](#7-end-to-end-2-dof-numerical-check-you-can-do-by-calculator)
8. [How to Run Full Validation in This Repo](#8-how-to-run-full-validation-in-this-repo)
9. [Current Validation Results (As of This Edit)](#9-current-validation-results-as-of-this-edit)
10. [What to Tell Supervisor Tomorrow](#10-what-to-tell-supervisor-tomorrow)
11. [CMOS/LTSpice Next Steps](#11-cmosltspice-next-steps)

---

## 1. What Changed and Why

Earlier versions mixed three different things:

1. A pedagogical point-mass derivation.
2. A full-space QP implementation with slacks.
3. Multiple experimental oscillator solvers.

This file now uses **repository code as source of truth**:

- Dynamics: `src/dynamics/arm2dof.py`
- QP builder: `src/mpc/qp_builder.py`
- OSQP baseline: `src/solver/osqp_solver.py`
- Canonical neuromorphic attempt: `src/solver/stuart_landau_lagrange_direct.py`
- Validation harness: `scripts/validate_2dof_sl_vs_osqp.py`

---

## 2. What Oscillators Are (Beginner-Friendly)

An oscillator is any system that repeats motion in time (electrical voltage, current, mechanical position, etc.).

A Stuart-Landau oscillator (SL) is the canonical nonlinear oscillator near a Hopf bifurcation:

$$
\dot{z} = (\mu + i\omega)z - \gamma |z|^2 z
$$

with $z \in \mathbb{C}$. Intuition:

- $(\mu + i\omega)z$ wants to grow and rotate.
- $-\gamma |z|^2 z$ limits amplitude, so it does not blow up.

In optimization-style dynamics, we keep this amplitude-limiting idea and add forces from cost gradients and constraints.

In this repo, the practical solver implementation uses **real-valued state dynamics** with SL-like cubic damping and constraint feedback, then checks feasibility against OSQP.

---

## 3. Implemented 2-DOF Dynamics (Exact Code Model)

The implemented arm model uses distributed link inertia terms (not point masses at ends).

State:

$$
x = \begin{bmatrix} q_1 & q_2 & \dot q_1 & \dot q_2 \end{bmatrix}^\top
$$

Dynamics:

$$
\dot x = \begin{bmatrix}
\dot q \\
M^{-1}(q)\left(\tau - C(q,\dot q)\dot q - G(q)\right)
\end{bmatrix}
$$

with

$$
M(q)=\begin{bmatrix}
I_{11} & I_{12} \\
I_{12} & I_{22}
\end{bmatrix}
$$

$$
I_{11}=\frac{m_1 l_1^2}{3} + m_2\left(l_1^2 + \frac{l_2^2}{3} + l_1 l_2\cos q_2\right)
$$

$$
I_{12}=m_2\left(\frac{l_2^2}{3} + \frac{1}{2}l_1 l_2\cos q_2\right),\quad
I_{22}=\frac{m_2 l_2^2}{3}
$$

Coriolis form used in code:

$$
h = -m_2 l_1 l_2 \sin q_2
$$

$$
C = \begin{bmatrix}
h\dot q_2 & h(\dot q_1+\dot q_2) \\
-h\dot q_1 & 0
\end{bmatrix}
$$

Gravity form used in code:

$$
G_1 = \left(\frac{m_1 l_1}{2}+m_2 l_1\right)g\sin q_1 + \frac{m_2 l_2}{2}g\sin(q_1+q_2)
$$

$$
G_2 = \frac{m_2 l_2}{2}g\sin(q_1+q_2)
$$

Important: this is why old point-mass matrices from previous draft do not numerically match code.

---

## 4. Implemented MPC QP Formulation (Exact Code Form)

The implemented QP is **full-space trajectory optimization with slacks**, not lifted-input-only $U$.

Decision vector structure:

$$
z = [x_0, u_0, x_1, u_1, \dots, x_N, s_0, s_1, \dots, s_N]
$$

Dimensions (2-DOF):

- $n_x=4$, $n_u=2$
- base decision count: $N(n_x+n_u)+n_x$
- slack count: $(N+1)\cdot 2$

Objective:

$$
\min_z \frac{1}{2}z^\top Q z + p^\top z
$$

with stage/terminal tracking and control penalties plus slack penalty $Q_s$.

Equality constraints:

1. Initial state lock: $x_0 = x_{\text{measured}}$
2. Linearized Euler dynamics per stage:

$$
A_k = I + \Delta t A_c,\quad B_k = \Delta t B_c,
$$

$$
x_{k+1} - A_k x_k - B_k u_k = c_k
$$

where

$$
c_k = \Delta t\left(f(x_{\bar k},u_{\bar k}) - A_c x_{\bar k} - B_c u_{\bar k}\right)
$$

Inequality constraints:

- hard torque bounds
- soft joint-angle bounds using slacks
- nonnegative slacks

---

## 5. Implemented Solver Variants and Their Status

### 5.1 Canonical Baseline

`OSQPSolver` in `src/solver/osqp_solver.py`

Standard form:

$$
\min_x \frac{1}{2}x^\top P x + q^\top x\quad \text{s.t.}\quad l \le Ax \le u
$$

This is the baseline ground truth for thesis validation.

### 5.2 Canonical Neuromorphic Attempt (Current)

`StuartLandauLagrangeDirect` in `src/solver/stuart_landau_lagrange_direct.py`

Current practical behavior:

1. Attempts SL-style continuous dynamics with fixed-step integration.
2. Splits equality rows from inequality rows internally.
3. Applies numerical guards and clipping.
4. **Default policy is now no fallback** (`fallback_to_osqp=False`) so SL behavior is measured directly.

Best tested no-fallback configuration for this solver on current 2-DOF MPC QPs:

- `tau_x=0.1`
- `tau_lam=0.01`
- `constraint_penalty=50.0`
- `damping=0.1`
- `use_dual=True`

This setting reduces equality residual compared to other SL variants, but still does not reach OSQP-level optimality.

### 5.3 Other Variants (Experimental/Legacy)

- `src/solver/stuart_landau_lagonn.py`
- `src/solver/stuart_landau_lagonn_full.py`
- `src/mpc/sl_solver.py`

`StuartLandauLagONNFull` was extended with a deterministic post-oscillator active-set KKT refinement (no OSQP call). With this extension enabled, it is currently the closest no-fallback variant to OSQP.

### 5.4 Why It Looked Better Before

Two concrete reasons:

1. In older LagONN code, convergence reporting used `sol.status == 0` as converged.
2. In `solve_ivp`, `status == 0` means integration reached end of time span, not event-based convergence.

So runs could be labeled `converged=True` while still far from KKT/OSQP. This has now been corrected in both:

- `src/solver/stuart_landau_lagonn.py`
- `src/solver/stuart_landau_lagonn_full.py`

### 5.5 Root Causes of LagONN Non-Convergence (Confirmed)

Code-level investigation identified four concrete issues:

1. Equality dual dead-zone in phase dynamics:
   - Old form used $\dot\phi \propto -\sin(\phi)(Cx-d)$.
   - At initialization $\phi=0$, $\sin(\phi)=0$, so equality dual update could stall.
2. Bounded equality multiplier encoding:
   - Old mapping $\lambda^{eq}=\cos(\phi)$ limits multipliers to $[-1,1]$.
   - This is often insufficient for KKT multipliers in full-space MPC QPs.
3. Irreversible inequality dual windup:
   - Pure ReLU accumulation increased dual amplitudes but did not leak/relax them.
4. Expensive/fragile integration behavior:
   - Adaptive RK flow frequently spent long time without reaching practical KKT quality.

Implemented fixes:

1. Equality dual now uses unconstrained dual flow (stored in `phi_eq`).
2. Inequality duals use projected-leaky dynamics.
3. Solvers use fixed-step projected integration with clipping guards.
4. `StuartLandauLagONNFull` adds an explicit active-set KKT refinement step (still no OSQP fallback).

### 5.6 Complete Implemented SL LagONN Full System (Step-by-Step)

This subsection writes the solver in the same form as implemented in `src/solver/stuart_landau_lagonn_full.py`.

#### 5.6.1 Optimization target

We solve the QP

$$
\min_x\; \frac{1}{2}x^\top P x + q^\top x
$$

subject to

$$
C x = d,
$$

$$
l \le A_c x \le u.
$$

#### 5.6.2 State variables represented by oscillator network

The continuous-time state is

$$
y = [x,\;\phi^{eq},\;\lambda^{up},\;\lambda^{lo}],
$$

where

1. $x \in \mathbb{R}^n$: primal decision variables.
2. $\phi^{eq} \in \mathbb{R}^{m_{eq}}$: equality dual variables (unbounded in current implementation).
3. $\lambda^{up},\lambda^{lo} \in \mathbb{R}_{\ge 0}^{m}$: inequality dual amplitudes.

#### 5.6.3 Forces used in primal dynamics

Cost gradient:

$$
g(x) = P x + q.
$$

Stuart-Landau nonlinear restoring term:

$$
r_{SL}(x) = (\mu_x - x\odot x)\odot x.
$$

Equality residual:

$$
r_{eq}(x) = Cx - d.
$$

Inequality violation signals:

$$
r_{up}(x) = \max(0, A_c x - u),
$$

$$
r_{lo}(x) = \max(0, l - A_c x).
$$

Net inequality dual:

$$
\lambda^{net} = \lambda^{up} - \lambda^{lo}.
$$

#### 5.6.4 Implemented ODEs

Primal update:

$$
	au_x \dot{x} = r_{SL}(x) - g(x)
- s_\lambda C^\top \phi^{eq}
- s_\lambda A_c^\top \lambda^{net}
- \rho_{eq} C^\top r_{eq}(x)
- \rho_{in} A_c^\top \big(r_{up}(x)-r_{lo}(x)\big),
$$

where $s_\lambda$ is `lagrange_scale`, $\rho_{eq}$ is `eq_penalty`, and $\rho_{in}$ is `ineq_penalty`.

Equality dual update (dead-zone removed):

$$
	au_{eq} \dot{\phi}^{eq} = r_{eq}(x).
$$

Inequality dual updates with leakage and projection:

$$
	au_{in} \dot{\lambda}^{up} = s_\lambda\big(r_{up}(x) - \eta\lambda^{up}\big),
$$

$$
	au_{in} \dot{\lambda}^{lo} = s_\lambda\big(r_{lo}(x) - \eta\lambda^{lo}\big),
$$

followed by projection

$$
\lambda^{up} \leftarrow \max(0,\lambda^{up}),\quad
\lambda^{lo} \leftarrow \max(0,\lambda^{lo}).
$$

Here $\eta$ is `dual_leak`.

#### 5.6.5 Numerical integration actually used in code

The current implementation uses explicit fixed-step integration with safeguards:

$$
y_{k+1} = y_k + \alpha_k \;\mathrm{clip}(\dot{y}_k),
$$

$$
\alpha_k = \frac{\Delta t}{1 + 0.01\|\dot{y}_k\|_2},
$$

with state clipping to avoid overflow and stopping when

$$
\|\dot{y}_k\|_2 < \varepsilon_{conv}.
$$

#### 5.6.6 Warm start

When no warm start is provided, primal starts from regularized unconstrained minimizer:

$$
x_0 \approx -(P + 10^{-6}I)^{-1}q.
$$

This improves basin-of-attraction behavior relative to zero initialization.

### 5.7 What Deterministic Active-Set KKT Refinement Means

Short answer: yes, this step is currently classical deterministic linear algebra on CPU, not oscillator-only dynamics.

After oscillator evolution, solver builds an active set:

1. Include equality constraints $Cx=d$.
2. Include upper inequalities where $A_cx-u > -\epsilon_a$.
3. Include lower inequalities where $l-A_cx > -\epsilon_a$.

Then solve equality-constrained QP KKT system:

$$
\begin{bmatrix}
P + \delta I & A_{act}^\top \\
A_{act} & 0
\end{bmatrix}
\begin{bmatrix}
x \\
\nu
\end{bmatrix}
=
\begin{bmatrix}
-q \\
b_{act}
\end{bmatrix}.
$$

This is deterministic (no randomness), finite-step, and typically very accurate.

Important interpretation for thesis:

1. It is not OSQP fallback.
2. It is still digital/classical matrix solve in software.
3. Therefore the current best result is a hybrid algorithmic pipeline:
   - analog-inspired oscillator phase
   - deterministic algebraic refinement phase

### 5.8 Can KKT Refinement Be Done With Oscillators Only?

In principle: possible as analog computing architecture.

In current repository: not implemented.

To do it oscillator-only, you need an analog circuit that solves linear saddle-point equations directly, for example via continuous-time primal-dual network:

$$
\dot{x} = -\nabla_x\mathcal{L}(x,\nu),\quad
\dot{\nu} = +\nabla_{\nu}\mathcal{L}(x,\nu),
$$

for

$$
\mathcal{L}(x,\nu)=\frac{1}{2}x^\top P x + q^\top x + \nu^\top(A_{act}x-b_{act}).
$$

Practical hardware notes:

1. This becomes an analog linear solver block with transconductance-weighted couplings.
2. You still need active-set detection logic; fully analog active-set selection is hard and may need comparator/latch networks.
3. Exact matrix inversion is replaced by network settling; accuracy depends on mismatch/noise/bandwidth.

So for pure analog hardware, recommended path is:

1. Keep oscillator phase for active-set emergence.
2. Replace digital KKT solve with an analog primal-dual linear solver macrocell.
3. Add analog comparator network for active constraint gating.

This preserves a fully analog solver concept, but requires significant circuit research and verification.

### 5.9 Intel Loihi Paper vs Our Current QP: Why They Can Stop Without KKT Refinement

The archived paper in this repo (`docs/archived/Neuromorphic QP Intel Loihi.md`) uses projected primal-dual mechanics very close in spirit to ours, but under different practical assumptions.

Core solver pattern in the paper:

$$
x_{t+1} = r_X\big(x_t - a_t(Qx_t + p + A^Tv_t)\big)
$$

$$
v_t = i_G\big(w_t + b_t(Ax_t-k)\big),\quad
w_{t+1} = w_t + b_t(Ax_{t+1}-k)
$$

with $i_G(\cdot)=\max(0,\cdot)$ and annealed gains ($a_t$ reduced, $b_t$ increased).

Why this can work well for their reported setting:

1. They benchmark to approximate accuracy (paper uses an 8% optimality target relative to OSQP).
2. They run warm-started iterative MPC and accept approximate per-iteration solves.
3. They use preconditioning (Ruiz) and hardware-specific fixed-point scheduling.
4. Their stopping criterion is performance/energy at acceptable control quality, not strict per-instance KKT tightness.

Why we still need deterministic refinement in this repo (for current claim level):

1. Our thesis comparison currently expects near-OSQP objective and very small equality residual on each tested QP instance.
2. Our full-space MPC transcription has many hard equality rows from dynamics and initial-state locking.
3. Oscillator-only flows here often remain feasible-ish but stall before tight optimality.
4. The post-oscillator active-set KKT solve removes this final optimality/residual gap deterministically.

So: the difference is not that their mechanics are unrelated; it is mostly target accuracy, stopping policy, and formulation-specific conditioning.

---

## 6. Canonical Baseline and Validation Rule

For every QP instance:

1. Build QP once from same $x_0$ and reference.
2. Solve with OSQP.
3. Solve with SL variant.
4. Report:

$$
\text{rel\_obj\_gap}=\frac{|J_{SL}-J_{OSQP}|}{|J_{OSQP}|+10^{-12}}
$$

$$
\|u_{0,SL}-u_{0,OSQP}\|_2
$$

and primal feasibility residuals.

If SL falls back, explicitly report it as fallback, not pure-SL convergence.

---

## 7. End-to-End 2-DOF Numerical Check You Can Do by Calculator

Use implemented model parameters:

$$
m_1=m_2=1,\; l_1=0.5,\; l_2=0.4,\; g=9.81,\; q_1=\pi/4,\; q_2=\pi/3
$$

### 7.1 Compute implemented inertia matrix

$$
\cos q_2=0.5
$$

$$
I_{11}=\frac{1\cdot 0.5^2}{3}+1\left(0.5^2+\frac{0.4^2}{3}+0.5\cdot0.4\cdot0.5\right)
=0.083333+0.403333=0.486666
$$

$$
I_{12}=1\left(\frac{0.4^2}{3}+\frac{1}{2}0.5\cdot0.4\cdot0.5\right)=0.053333+0.050000=0.103333
$$

$$
I_{22}=\frac{1\cdot0.4^2}{3}=0.053333
$$

So

$$
M=\begin{bmatrix}0.486666 & 0.103333 \\
0.103333 & 0.053333\end{bmatrix}
$$

and

$$
M^{-1}\approx\begin{bmatrix}3.4909 & -6.7636 \\
-6.7636 & 31.8545\end{bmatrix}
$$

### 7.2 Compute implemented gravity

$$
\sin q_1=0.7071,\;\sin(q_1+q_2)=\sin(105^\circ)=0.9659
$$

$$
G_1=\left(\frac{1\cdot0.5}{2}+1\cdot0.5\right)9.81\sin q_1 + \frac{1\cdot0.4}{2}9.81\sin(q_1+q_2)
\approx 7.0977
$$

$$
G_2=\frac{1\cdot0.4}{2}9.81\sin(q_1+q_2)\approx1.8951
$$

These values should match code-level checks.

---

## 8. How to Run Full Validation in This Repo

### 8.1 Canonical comparison script

```bash
"/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/python" scripts/validate_2dof_sl_vs_osqp.py --cases 3 --horizon 5 --t_solve 1.5 --dt 0.0005
```

Optional (explicitly enable fallback):

```bash
"/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/python" scripts/validate_2dof_sl_vs_osqp.py --cases 3 --horizon 5 --t_solve 1.5 --dt 0.0005 --allow_fallback
```

### 8.2 Interpreting output

Per case, check:

- `rel_obj_gap`
- `u0_err`
- `viol_osqp`
- `viol_sl`
- `status_sl`

If `status_sl=fallback_osqp:solved`, the final answer is OSQP-backed.

### 8.3 Optional stress sweeps

Run multiple horizons and initial states by editing case list in `scripts/validate_2dof_sl_vs_osqp.py`.

### 8.4 Exact test cases used in comparison scripts

The canonical three cases in current scripts are:

1. $x_0=[0.10,-0.05,0.02,-0.01],\; x_g=[0.60,-0.20,0,0]$
2. $x_0=[-0.15,0.08,-0.03,0.02],\; x_g=[0.30,0.35,0,0]$
3. $x_0=[0.05,0.12,0.01,-0.02],\; x_g=[-0.40,0.20,0,0]$

These are used by:

1. `scripts/validate_2dof_sl_vs_osqp.py`
2. `scripts/compare_sl_variants_no_fallback.py`

### 8.5 How to read the no-fallback comparison metrics

For each case:

1. `rel`: relative objective gap to OSQP.
2. `u0`: norm difference in first control action.
3. `eq`: max equality residual.
4. `ineq`: max upper inequality violation.

Interpretation hierarchy:

1. Feasibility first: `eq` and `ineq` should be near machine tolerance.
2. Then optimality: `rel` should be near zero.
3. Then control similarity: `u0` should be small.

In current results, oscillator-only variants are feasible-ish but not optimal, while refined LagONN Full is both feasible and near-optimal on tested QPs.

---

## 9. Current Validation Results (As of This Edit)

### 9.1 Variant comparison (no fallback)

Measured on identical QPs (no fallback), command:

```bash
PYTHONPATH=. "/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/python" -u scripts/compare_sl_variants_no_fallback.py --cases 3 --t_solve 1.0
```

Case metrics against OSQP (`rel_obj_gap` shown as percentage):

1. Case 0:
   - LagONN: 126.76%
   - LagONN Full (refined): 0.00%
   - Direct SL (no fallback): 121.91%
   - Direct SL + PIPG-style inequality channel: 100.00%
2. Case 1:
   - LagONN: 939.17%
   - LagONN Full (refined): 0.01%
   - Direct SL (no fallback): 935.01%
   - Direct SL + PIPG-style inequality channel: 100.00%
3. Case 2:
   - LagONN: 809.51%
   - LagONN Full (refined): 0.01%
   - Direct SL (no fallback): 805.15%
   - Direct SL + PIPG-style inequality channel: 100.00%

Mean relative objective gap:

- LagONN: 625.15%
- LagONN Full (refined): 0.0067%
- Direct SL (no fallback): 620.69%
- Direct SL + PIPG-style inequality channel: 100.00%

Additional control-error check from same run (`||u0_{SL}-u0_{OSQP}||_2`):

1. Case 0: 0.0005
2. Case 1: 0.0130
3. Case 2: 0.0313

Mean: 0.0149

Additional residual behavior for the implemented PIPG-style variant (same run):

1. Case 0: `eq=1.000e-01`
2. Case 1: `eq=1.500e-01`
3. Case 2: `eq=1.200e-01`

Mean equality residual: `1.233e-01`

Interpretation:

1. The PIPG-style inequality channel reduced equality residual relative to the basic direct no-fallback variant.
2. But objective/control mismatch stayed large (`rel≈1.0`, large `u0` mismatch), so this alone did not remove the need for final deterministic refinement on these QPs.
3. Increasing `t_solve` from 1.0 to 5.0 seconds did not materially change this behavior in our tests.

### 9.2 Direct-SL validator (no fallback)

Command:

```bash
PYTHONPATH=. "/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/python" -u scripts/validate_2dof_sl_vs_osqp.py --cases 3 --horizon 5 --t_solve 1.5 --dt 0.0005
```

Output summary (from this run):

- `mean_rel_obj_gap=6.1501`
- `mean_u0_err=24.7961`
- `mean_viol_osqp=4.983e-06`
- `mean_viol_sl=3.586e-01`

Per-case relative objective gaps from same run:

1. Case 0: 120.10%
2. Case 1: 929.30%
3. Case 2: 795.62%

Mean relative objective gap (best tuned Direct SL): 615.01%

Observed status behavior:

- LagONN and LagONN Full no longer report false convergence when only time limit is reached.
- LagONN Full now reports `status=refined_active_set` when oscillator phase did not meet convergence tolerance but active-set refinement produced the final solution.
- Direct SL no-fallback typically exits with `status=max_iter` on these QPs.

### 9.3 Harness tests executed

Command:

```bash
PYTHONPATH=. "/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/python" tests/test_stuart_landau_lagonn.py -v
```

Result:

- Passed: 4/6
- Failed: 2/6
- Failing tests:
   - `4. Medium MPC QP (N=20)`: failure reason `Solver did not converge`.
   - `6. Convergence Metrics`: failure reason `N=5: solver did not converge`.

Commands and results for full LagONN harness:

```bash
PYTHONPATH=. "/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/python" tests/test_sl_full_lagonn.py -t 3
PYTHONPATH=. "/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/python" tests/test_sl_full_lagonn.py -t 5
PYTHONPATH=. "/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/python" tests/test_sl_full_lagonn.py -t 6
```

- Test 3: PASS
- Test 5: PASS
- Test 6: PASS

Test 4 behavior:

```bash
PYTHONPATH=. "/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/python" tests/test_sl_full_lagonn.py -t 4
```

- Did not complete within 180 seconds in this environment (no final summary line captured).

### 9.4 Full pytest suite status

Command:

```bash
PYTHONPATH=. "/Users/alvin/Documents/Alvin/College/Academics/Master's Thesis/Code/Neuromorphic-Robot-Control/.venv/bin/pytest" -q --maxfail=1
```

Current first blocker:

- `tests/test_benchmark_suite.py` fails at import with `ModuleNotFoundError: No module named 'src.benchmark'`.
- Because collection fails early, full-suite pass/fail cannot be claimed until this import path issue is fixed.

Conclusion:

- Pure oscillator-only LagONN and Direct-SL still underperform on this full-space/slack MPC QP.
- Extended LagONN Full (oscillator phase + deterministic active-set KKT refinement, no OSQP fallback) is now near-OSQP on tested cases.

---

## 10. What to Tell Supervisor Tomorrow

Suggested clear narrative:

1. We have a robust 2-DOF MPC pipeline with OSQP ground truth.
2. Pure oscillator-only SL variants are still not OSQP-accurate on this QP transcription.
3. An extended no-fallback LagONN Full path now reaches near-OSQP quality by combining oscillator dynamics with deterministic active-set KKT refinement.
4. Convergence/status flags were corrected to avoid false-positive "converged" labeling.
5. Thesis claims should clearly separate:
   - baseline guarantee (OSQP)
   - oscillator-only capability (still limited)
   - hybrid neuromorphic+deterministic refinement capability (currently strongest no-fallback result)

---

## 11. CMOS/LTSpice Next Steps

This section is for next-phase hardware direction, not current validated baseline.

### 11.1 Ring oscillator basics

A CMOS ring oscillator is an odd number of inverters in a loop.

- Each inverter contributes delay $t_p$.
- Oscillation frequency (rough):

$$
 f \approx \frac{1}{2 N t_p}
$$

where $N$ is the number of inverter stages.

### 11.2 Mapping concept for optimization hardware

For SL/LagONN-style hardware prototype:

1. One oscillator (or one state variable) per primal/dual variable.
2. Coupling implemented by transconductance paths (weighted current injection).
3. Nonlinearity/saturation via device transfer or explicit limiter blocks.
4. Constraint projection approximated by rectifier-like blocks (ReLU behavior).

### 11.2.1 Suggested block-level mapping for SL LagONN Full

Map each equation to analog blocks:

1. Primal block $x_i$:
   - integrator capacitor for $x_i$ state.
   - cubic damping via translinear/nonlinear gm cell implementing $(\mu_x-x_i^2)x_i$.
   - current injections for $-(Px+q)_i$, $-(C^T\phi)_i$, and $-(A^T\lambda^{net})_i$.
2. Equality dual block $\phi_j$:
   - integrator with input current proportional to $(Cx-d)_j$.
3. Inequality dual blocks $\lambda_k^{up},\lambda_k^{lo}$:
   - rectified residual current source for violation.
   - leakage branch $-\eta\lambda$.
   - diode-connected projection to enforce nonnegativity.
4. Coupling matrix realization:
   - programmable transconductance matrix (OTA array or switched-gm bank).

### 11.2.2 Where digital control is still hiding today

If you keep deterministic active-set KKT refinement as currently implemented, digital is used in:

1. Active constraint index selection.
2. KKT matrix assembly.
3. Linear system solve.

For a strict analog-only target, these three must be replaced by analog network mechanisms.

### 11.3 LTSpice practical setup (first prototype)

1. Build 3-stage ring oscillator (CMOS inverters) and verify oscillation.
2. Measure amplitude and frequency sensitivity vs supply and device sizing.
3. Add one coupling branch between two oscillators via controlled current source.
4. Add a limiter block to emulate amplitude saturation.
5. Sweep coupling gain and confirm convergence/stability regions.

### 11.3.1 LTSpice roadmap for full solver prototype

A practical staged roadmap:

1. Stage A: single-variable SL cell
   - verify limit-cycle amplitude control and damping.
2. Stage B: coupled primal pair
   - implement 2x2 symmetric $P$ couplings and linear term $q$ current injections.
3. Stage C: add one equality dual channel
   - realize $Cx-d$ sensing and feedback into primal channels.
4. Stage D: add one inequality dual pair
   - realize upper/lower rectified violation with leakage.
5. Stage E: small QP tile (for example $n=4,m_{eq}=2,m=4$)
   - compare settled point with software reference.
6. Stage F: active-set analog candidate
   - explore comparator/latch gating for active constraints.
7. Stage G: analog linear refinement network
   - prototype a continuous-time primal-dual linear solver replacing digital KKT stage.

### 11.3.2 Verification signals to record in LTSpice

In addition to frequency/amplitude, log optimization-relevant signals:

1. Residual probes: $Cx-d$, $A_cx-u$, $l-A_cx$.
2. Dual state trajectories: $\phi^{eq}$ and $\lambda^{up/lo}$.
3. Primal objective proxy: implement a measurement macro for $\frac{1}{2}x^TPx+q^Tx$.
4. Settling time to residual thresholds.
5. Monte Carlo mismatch impact on residual floor.

### 11.4 What to log from LTSpice

- Frequency, amplitude, settling time.
- Phase relationship between coupled nodes.
- Power estimate from supply current.
- Sensitivity to process/voltage/temperature corners.

### 11.5 Thesis-safe statement today

Hardware mapping is a concrete next step with clear LTSpice milestones, but current thesis-grade closed-loop validation remains software-level with OSQP baseline and documented SL research variants.

For strict analog-only claims:

1. Oscillator-only SL LagONN Full is conceptually compatible with ring-oscillator/analog networks.
2. Deterministic active-set KKT refinement in the current repository is classical digital computation.
3. A pure analog replacement is possible in principle but is future work requiring dedicated circuit blocks for active-set gating and linear saddle-point solve.

---

## References (Code-Grounded)

- `src/dynamics/arm2dof.py`
- `src/mpc/qp_builder.py`
- `src/solver/osqp_solver.py`
- `src/solver/stuart_landau_lagonn.py`
- `src/solver/stuart_landau_lagonn_full.py`
- `src/solver/stuart_landau_lagrange_direct.py`
- `scripts/validate_2dof_sl_vs_osqp.py`
- `scripts/compare_sl_variants_no_fallback.py`
- `scripts/tune_direct_no_fallback.py`
- `tests/test_stuart_landau_lagonn.py`
- `tests/test_sl_full_lagonn.py`

