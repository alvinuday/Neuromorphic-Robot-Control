# Neuromorphic QP-MPC via Stuart-Landau Oscillators and LagONN
## Complete Derivation, Analysis, and Implementation Guide

**Grounded in:** Delacour 2025 (LagONN) · Mangalore et al. 2024 (Loihi/PIPG) · Wang & Roychowdhury (OIM) · Bhowmik Group (SHNO/Ring-OIM, IIT Bombay)

**Context:** M.Sc.–B.E. Dual Degree Thesis — Neuromorphic Edge Intelligence for Industrial Robotics — IIT Bombay, 2025–26

---

## Abstract

We present a complete, end-to-end derivation of a neuromorphic solver for convex Quadratic Programming (QP) problems arising in Model Predictive Control (MPC) of robotic systems. Starting from the standard OSQP form of the QP, we map decision variables onto the amplitudes of Stuart-Landau (SL) oscillators and constraints onto a Lagrangian oscillator network (LagONN), synthesising the independently developed Intel Loihi PIPG approach (Mangalore et al., IEEE RAM 2024) and the LagONN saddle-point framework (Delacour, *Neuromorph. Comput. Eng.* 5 044004, 2025). We show that the combined SL+LagONN dynamics implement a continuous-time analog of the Proportional-Integral Projected Gradient (PIPG) algorithm; that the saddle-point equilibrium coincides with the KKT solution of the QP up to a controllable bias O(μ_x); and that the architecture maps directly onto the CMOS ring-oscillator and FeFET-coupled oscillator platforms of the Bhowmik group at IIT Bombay. A concrete implementation roadmap is provided for a 2-DOF planar robotic arm MPC, benchmarked against OSQP.

---

## Table of Contents

| # | Section |
|---|---------|
| I | Problem Formulation: OSQP QP Form |
| II | Background: Oscillator Ising Machines & Stuart-Landau Theory |
| III | Background: LagONN (Delacour 2025) |
| IV | Background: Intel Loihi PIPG (Mangalore et al. 2024) |
| V | Core Derivation: SL Encoding of Decision Variables |
| VI | Core Derivation: Quadratic Cost → SL Couplings |
| VII | Core Derivation: Equality Constraints via LagONN |
| VIII | Core Derivation: Inequality Constraints (OSQP Form) |
| IX | Complete SL+LagONN Dynamical System |
| X | Saddle-Point Analysis and Convergence |
| XI | MPC Warm-Start and Receding Horizon |
| XII | Software Implementation (Python / SciPy) |
| XIII | Hardware Mapping — Bhowmik Group Platform |
| XIV | Benchmarking Against OSQP |
| XV | References |

---

## Section I — Problem Formulation: OSQP QP Form

### 1.1 Standard OSQP QP

OSQP [Stellato et al., *Math. Program. Comput.* 12(4), 2020] solves the canonical convex QP:

```
minimize    (1/2) x^T P x  +  q^T x                                  (OSQP)
subject to  l  <=  A_c x  <=  u

x ∈ R^n,   A_c ∈ R^{m×n},   P ∈ S_+^n (sym. pos.-semidefinite)
q, l, u ∈ R^m   (l_i may be -∞, u_i may be +∞)
```

OSQP internally converts this to an ADMM problem by introducing a slack variable `z = A_c x`, splitting the constraint into an equality `A_c x - z = 0` and a box `l ≤ z ≤ u`. The augmented Lagrangian is:

```
L_ρ(x, z, y) = (1/2) x^T P x  +  q^T x
             + y^T (A_c x - z)
             + (ρ/2) || A_c x - z ||_2^2                             (L_ρ)

y ∈ R^m  (dual variable / Lagrange multiplier)
ρ > 0    (penalty / step size parameter)
```

OSQP ADMM update equations per iteration k:

```
x^{k+1} = (P + ρ A_c^T A_c)^{-1} (-q - A_c^T (y^k - ρ z^k))
z^{k+1} = Π_{[l,u]} ( A_c x^{k+1} + y^k / ρ )                      (ADMM-OSQP)
y^{k+1} = y^k  +  ρ ( A_c x^{k+1} - z^{k+1} )
```

`Π_{[l,u]}` denotes element-wise clipping (projection onto [l_i, u_i]). KKT conditions at the solution (x\*, z\*, y\*):

```
Px* + q + A_c^T y*  = 0          [stationarity]
A_c x* - z*          = 0          [primal feasibility]
l_i <= z_i* <= u_i                [box feasibility]
y_i*(z_i* - l_i) = 0, y_i*(u_i - z_i*) = 0   [complementarity]     (KKT-OSQP)
```

### 1.2 MPC QP for a 2-DOF Planar Arm

For the 2-DOF arm, the state vector is `s = [θ₁, θ₂, θ̇₁, θ̇₂]^T ∈ R^4` and the control input is `u = [τ₁, τ₂]^T ∈ R^2`. A horizon-N MPC formulates the QP over the stacked decision vector:

```
ξ = [u_0^T, ..., u_{N-1}^T, s_1^T, ..., s_N^T]^T  ∈ R^{n_ξ}
n_ξ = N*(n_u + n_s)  =  N*(2 + 4)  =  6N                            (MPC-2DOF)

Cost matrices (horizon tracking + effort):
  P = blkdiag(R, Q_f, ..., Q, R, Q_f)  ∈ S_+^{n_ξ}
  q = stacked linear tracking terms

Constraint matrix A_c encodes:
  - Dynamics:  s_{k+1} = A_d s_k + B_d u_k  (linearised arm model)
  - Torque limits:  τ_min <= u_k <= τ_max
  - Joint angle / velocity limits

For horizon N = 20:  n_ξ = 120,  m constraints ~ 200-300
```

The MPC QP is then in exact OSQP form with P, q, A_c, l, u computed from the current state measurement at each control step (~20–50 ms for the arm).

---

## Section II — Background: Oscillator Ising Machines and Stuart-Landau Theory

### 2.1 Coupled Nonlinear Oscillators and the Ising Hamiltonian

Wang and Roychowdhury [OIM, Springer UCNC 2019; *Natural Computing* 2021] showed that networks of coupled self-sustaining nonlinear oscillators naturally minimise an Ising Hamiltonian when driven by sub-harmonic injection locking (SHIL). The foundational result is:

> **Theorem (Wang-Roychowdhury 2019)**
> Under SHIL at frequency 2ω₀, the phase dynamics of n coupled oscillators are governed by the Lyapunov function:
>
> `E(φ) = -(1/2) Σ_{i,j} J_{ij} cos(φ_i - φ_j) - Σ_i h_i cos(φ_i)`
>
> which is equivalent to the Ising Hamiltonian `H = -Σ_{i<j} J_{ij} σ_i σ_j - Σ_i h_i σ_i` when φ_i binarises to {0, π} → σ_i = {+1, -1}. Coupled oscillator dynamics therefore minimise H, implementing an Ising machine.

Bhowmik and colleagues at IIT Bombay have implemented this principle using two physical platforms:

- **Spin Hall Nano-Oscillators (SHNOs):** Hemadribhotla et al. / Bhowmik group [*IEEE Trans. Magn.*, 2021] demonstrated dipole-coupled SHNO arrays where micromagnetic dynamics (LLGS equations) realise the Kuramoto / Stuart-Landau model. The synchronisation range between two SHNOs as a function of physical distance maps directly to J_{ij} coupling coefficients.

- **CMOS Ring Oscillator Arrays:** Bhowmik group [2024] demonstrated improved computation time using electronic ring oscillator networks vs classical optimisation algorithms (SA) for Max-Cut. FeFET-coupled CMOS ring oscillator arrays [Bhowmik group, ISCAS 2024] use ferroelectric FET coupling to provide programmable, non-volatile J_{ij} weights — directly enabling on-chip encoding of QP matrix P.

### 2.2 The Stuart-Landau Oscillator: Normal Form of a Hopf Bifurcation

Any smooth self-sustaining oscillator near a Hopf bifurcation is locally equivalent to the Stuart-Landau (SL) equation [Cross and Hohenberg, *Rev. Mod. Phys.* 65, 1993; Kuramoto, 1984]. The SL oscillator is therefore the canonical, physics-agnostic model for all physical oscillator platforms — SHNOs, CMOS ring oscillators, LC oscillators, and VO₂-based relaxation oscillators alike. The complex ODE is:

```
τ  dA_i/dt  =  ( μ_i  +  iω_i  -  |A_i|² ) A_i  +  Σ_j K_{ij} A_j  +  I_i(t)    (SL)

where  A_i(t) = r_i(t) exp(iφ_i(t))  ∈ C   (complex amplitude)
       r_i = |A_i|      : instantaneous amplitude (radius in phase plane)
       φ_i = arg(A_i)   : instantaneous phase
       μ_i > 0          : bifurcation / gain parameter
       ω_i              : natural angular frequency
       K_{ij} ∈ C       : complex coupling weight between oscillators i, j
       I_i(t)           : external injection / forcing term
       τ                : oscillator time constant
```

In polar coordinates (r_i, φ_i), separating real and imaginary parts:

```
τ  dr_i/dt     =  ( μ_i - r_i² ) r_i  +  Re[ Σ_j K_{ij} A_j exp(-iφ_i) ]  +  Re[I_i exp(-iφ_i)]
τ  r_i dφ_i/dt =                         Im[ Σ_j K_{ij} A_j exp(-iφ_i) ]  +  Im[I_i exp(-iφ_i)]  (SL-polar)

Isolated oscillator (K=0, I=0): limit cycle  r_i* = √μ_i,  φ_i(t) = ω_i t + φ₀
```

### 2.3 SHIL: Phase Binarisation for Ising Computing

When an external signal at frequency 2ω₀ is injected into oscillator i (SHIL), the phase equation acquires an additional term:

```
τ  dφ_i/dt  =  -K_SHIL sin(2φ_i)  +  Σ_j K_{ij}/r_i * Im[A_j exp(-iφ_i)]   (SHIL)

Stable fixed points:  φ_i ∈ {0, π}  (binary phase states)
Under strong SHIL:  φ_i → {0, π}  regardless of initial conditions
This binarises A_i → r_i * {+1, -1}  ⟹  binary Ising spin σ_i
```

> **Key design insight for QP:** For continuous QP optimisation, we *suppress* SHIL and operate the oscillators in the amplitude regime, where r_i encodes a continuous decision variable. SHIL is only applied in binary Ising machines. The Bhowmik group's CMOS ring oscillator platform can operate in either mode by switching the SHIL injection signal on/off, making it directly applicable to our continuous QP solver.

---

## Section III — Background: LagONN (Delacour 2025)

### 3.1 The Constrained ONN Problem

Delacour et al. [*Neuromorphic Computing and Engineering* 5, 044004, 2025, DOI: 10.1088/2634-4386/ae0eab] address a fundamental limitation of standard Oscillatory Neural Networks (ONNs): penalty-method constraint enforcement is unreliable — large penalties cause slow convergence; small penalties permit constraint violation at local minima. The paper introduces the Lagrange Oscillatory Neural Network (LagONN) to resolve this.

### 3.2 Standard XY-ONN Energy

```
E(φ) = -(1/2) Σ_{i,j} J_{ij} cos(φ_i - φ_j)  -  Σ_i h_i cos(φ_i)

Standard ONN dynamics (gradient descent on E):
  τ  dφ_i/dt  =  -dE/dφ_i  =  -Σ_j J_{ij} sin(φ_i - φ_j)  +  h_i sin(φ_i)     (XY-ONN)

For constrained problems, constraint g(φ)=0 is enforced by adding a penalty:
  E_pen = E + λ*g(φ)²   [unreliable, tuning-sensitive]
```

### 3.3 LagONN: Lagrange Oscillators for Hard Constraints

For m constraints Z_m(φ) = 0 (where Z_m ∈ C is the complex constraint energy), LagONN adds one Lagrange oscillator φ_m^λ per constraint. The Lagrange function is [Eq. 6 of Delacour 2025]:

```
L_T(φ, φ^λ) = Σ_m u_m^λ · Z_m
             = Σ_m [ cos(φ_m^λ) Re[Z_m] + sin(φ_m^λ) Im[Z_m] ]               (LagONN-L)

where  u_m^λ = ( cos(φ_m^λ), sin(φ_m^λ) )  is the unit vector
encoding the real and imaginary parts of the Lagrange multiplier λ_m.
```

LagONN saddle-point dynamics [Eq. 17 of Delacour 2025]:

```
Primal oscillators (gradient DESCENT on L_T + E):
  τ  dφ_x/dt  =  -dE/dφ_x  -  Σ_m u_m^λ · (dZ_m/dφ_x)                        (D1)

Lagrange oscillators (gradient ASCENT on L_T):
  τ_λ  dφ_m^λ/dt  =  +dL_T/dφ_m^λ
                   =  -sin(φ_m^λ) Re[Z_m] + cos(φ_m^λ) Im[Z_m]                (D2)

Interpretation:
  - Primal oscillators descend: minimise cost subject to constraint
  - Lagrange oscillators ascend: maximise violation signal, enforcing Z_m → 0
```

> **Theorem 2 (Delacour 2025) — Saddle-Point Existence**
>
> For any satisfiable constrained problem, L_T has at least one saddle point (φ\*, φ^{λ\*}) satisfying `L_T(φ*, φ^{λ*}) = 0`, i.e., all constraints Z_m(φ\*) = 0 are exactly satisfied.
>
> *Proof sketch:* By strong duality of the primal-dual formulation, `max_{φ^λ} min_φ L_T = min_φ max_{φ^λ} L_T = 0`. The saddle exists at the feasible optimal φ\*.

### 3.4 Connection to QP Lagrangian

This result is directly applicable to QP. The standard Lagrangian for a constrained QP with equality constraints g(x) = Cx - d = 0 is:

```
L(x, λ) = (1/2) x^T P x + q^T x  +  λ^T (Cx - d)                   (L_QP)

This is identical to LagONN's L_T with:
  φ_x    <->  x   (decision variables, continuous amplitudes in SL mapping)
  Z_m    <->  (Cx - d)_m   (real-valued constraint residuals)
  φ_m^λ  <->  angle of Lagrange multiplier
  λ_m    <->  cos(φ_m^λ)   (for real constraints, Im[Z_m]=0)

The LagONN paper proves the saddle-point dynamics for any differentiable
constraint, of which QP equality constraints are a special (linear) case.
```

---

## Section IV — Background: Intel Loihi PIPG (Mangalore et al. 2024)

### 4.1 QP on Loihi 2 via PIPG

Mangalore, Fonseca Guerra, Risbud, Stratmann, Wild [*IEEE Robotics and Automation Magazine*, DOI: 10.1109/MRA.2024.3415005, 2024] implement a convex QP solver on Intel Loihi 2 using the Proportional-Integral Projected Gradient (PIPG) algorithm [Yu, Elango, Acikmese; *IEEE Control Syst. Lett.* 5(6), 2021]:

```
minimize   f(x) = (1/2) x^T Q x + p^T x                              (Loihi-QP)
subject to  g(x) = Ax - k  ≤  0

x ∈ R^L,  A ∈ R^{M×L},  Q ∈ S_+^L,  p, k ∈ R^M
```

PIPG discrete-time update equations [Eqs 6–8 in Mangalore et al.]:

```
x_{t+1} = Π_X( x_t - α_t (Q x_t + p + A^T v_t) )                   (6)

v_t     = θ_G( v_{t-1} + β_t (A x_t - k) )                          (7)

w_{t+1} = w_t + β_t (A x_{t+1} - k)                                  (8)

Π_X  : projection onto feasible set X (box constraints on x)
θ_G  : element-wise ReLU  (enforces v_t ≥ 0 for inequalities)
α_t  : step-size (decays toward 0)
β_t  : dual step-size (grows over time)
```

### 4.2 Neuromorphic Mapping on Loihi 2

The PIPG equations are interpreted as a two-layer event-based recurrent neural network [Fig. 2(b) in Mangalore et al.]:

| PIPG Variable | Loihi 2 Neuron Type | State Update |
|---|---|---|
| x_i (decision vars) | Gradient Descent Neuron A_i | Eq. (6): GD + constraint correction |
| v_m (constraint vars) | Constraint Check Neuron B_m | Eq. (7): ReLU-projected PI update |
| Q matrix | Synaptic weights between A neurons | Sparse matrix-vector multiply |
| A matrix | Synaptic weights A→B | Sparse matrix-vector multiply |
| A^T matrix | Synaptic weights B→A | Constraint correction injection |

### 4.3 Key Results (Mangalore et al.)

Applied to the ANYmal quadruped robot MPC (N=100 horizon, ~7,248 neurons, ~405,504 synaptic weights on a single Loihi 2 chip):

- Solution converges within ~55 PIPG iterations to within **8%** of OSQP optimal
- TTS scales roughly linearly with problem size; Loihi outperforms OSQP at large N
- **203× energy reduction** vs laptop-class CPU OSQP
- **520× energy-delay product (EDP) reduction** vs laptop CPU at largest problem
- Warm-starting between MPC iterations stabilises the approximate solutions

> **Our position:** The Loihi solver uses digital integrate-and-fire neurons with graded spikes to approximate the PIPG gradient flow. Our SL+LagONN approach implements the **same mathematical algorithm** — continuous-time PIPG / Arrow-Hurwicz saddle-point flow — but in analog oscillator hardware, providing a physical, energy-efficient substrate that does not require neuromorphic chip access. The Bhowmik group's CMOS ring oscillator + FeFET platform at IIT Bombay can implement this directly.

---

## Section V — Core Derivation: SL Encoding of Decision Variables

### 5.1 Amplitude Encoding Principle

We encode each decision variable x_i ∈ R as the real-valued amplitude of Stuart-Landau oscillator i. Operating without SHIL (or with very weak SHIL fixing phase to 0), the oscillator state is:

```
A_i(t) = r_i(t) exp(i·0)  =  r_i(t) ∈ R   (real amplitude, phase locked to 0)

Decision variable encoding:
  x_i  :=  a_i * r_i  +  b_i                                         (Encode)

where scaling coefficients a_i, b_i map r_i ∈ [0, r_max] to x_i ∈ [x_i^min, x_i^max]:
  a_i = (x_i^max - x_i^min) / r_max
  b_i = x_i^min

For symmetric problems with x_i ∈ [-x_max, +x_max]:
  Use signed amplitude: A_i ∈ R, A_i ∈ [-√μ_i, +√μ_i]
  x_i = A_i  (direct identification, a_i=1, b_i=0)
```

The signed amplitude interpretation is achieved by operating the SL oscillator below the Hopf bifurcation in an overdamped regime (μ_i < 0) for unconstrained variables, or by allowing A_i to oscillate about a non-zero mean set by the forcing terms. For QP decision variables that are naturally real-valued (torques, positions), we adopt the **direct signed amplitude approach**: x_i = A_i ∈ R, exploiting the full range [-√μ_i, +√μ_i].

### 5.2 The SL Restoring Term as Soft Box Constraint

The SL restoring nonlinearity `(μ_i - A_i²) A_i` serves a critical role: it keeps the amplitude bounded near A_i ∈ [-√μ_i, +√μ_i], acting as a differentiable soft proxy for a box constraint. This is exploited in two ways:

1. **Box constraint satisfaction:** For variables with bounds |x_i| ≤ x_max^i, set μ_i = (x_max^i)². The SL term exponentially penalises amplitude exceeding √μ_i, providing a smooth analogue of Π_{[l_i,u_i]} in the discrete PIPG.

2. **Tikhonov regularisation:** For very small μ_i, the SL term becomes approximately -x_i² · x_i = -x_i³, which provides a weak norm-like penalty. In the limit μ_i → 0, the SL term vanishes and we recover pure gradient flow.

```
Effect on equilibrium with cost gradient alone:
  (μ_i - x_i*²) x_i* = (Px* + q)_i   [SL equilibrium condition for x_i]

For |x_i*| << √μ_i  (operating point well below saturation):
  μ_i x_i* ≈ (Px* + q)_i
  => x_i* ≈ (Px* + q)_i / μ_i   [biased by μ_i]                     (SL-equil)

Therefore: choose μ_i >> max|(Px*+q)_i| / |x_i*| to minimise bias,
OR add constraint terms to cancel the SL term exactly at the optimum.
Our LagONN augmentation (Section VII) handles this correctly.
```

### 5.3 The SL Oscillator with External Forcing

With coupling and external forcing, the SL amplitude equation becomes:

```
τ  dA_i/dt  =  (μ_i - A_i²) A_i  +  F_i^ext(t)                      (SL-FP)

F_i^ext(t) includes cost gradient coupling, constraint correction,
            and Lagrange multiplier injection (derived in Sections VI–VIII)

Fixed-point condition:  F_i^ext(A*) = -(μ_i - A_i*²) A_i*

In the limit μ_i small (weak SL restoring):
  0 ≈ F_i^ext(A*)  =>  A_i* is a zero of F^ext
  This is exactly the KKT stationarity condition.
```

---

## Section VI — Core Derivation: Quadratic Cost to SL Couplings

### 6.1 Identifying QP Cost with ONN Hamiltonian

The QP objective f(x) = (1/2) x^T P x + q^T x must be expressed as the energy function of the SL network. Compare with the ONN Hamiltonian:

```
QP cost:         f(x) = (1/2) Σ_{i,j} P_{ij} x_i x_j  +  Σ_i q_i x_i

ONN Hamiltonian: H(A) = -(1/2) Σ_{i,j} J_{ij} A_i A_j  -  Σ_i h_i A_i
                       (real amplitude, XY coupling for real-valued variables)

Mapping: minimising f(x) <-> minimising H(A) with x_i = A_i, requires
   J_{ij}  = -P_{ij}       (off-diagonal coupling weights)            (Cost-Map)
   h_i     = -q_i          (external field / linear bias)
   J_{ii}  = 0             (self-coupling absorbed into SL restoring term)

Note: P must be PSD for convex QP; -P is NSD, which is appropriate since
the ONN Hamiltonian H should have a minimum, not a maximum.
(A PSD P means H(A)=f(A) is convex, and the ONN gradient descent on H
 converges to its minimum, which is the QP optimum.)
```

### 6.2 Gradient Descent Dynamics on the Cost

Adding the cost gradient to the SL oscillator, the amplitude dynamics for decision variable x_i become:

```
τ_x  dx_i/dt  =  (μ_x - x_i²) x_i  -  df/dx_i
              =  (μ_x - x_i²) x_i  -  Σ_j P_{ij} x_j  -  q_i

Matrix form (all n decision variables simultaneously):
  τ_x  dx/dt  =  diag(μ_x - x²) x  -  Px  -  q                      (VI.1)

where  diag(μ_x - x²) x  denotes element-wise  (μ_x - x_i²) x_i.

Equivalently, define D(x) = diag(μ_x - x_i²):
  τ_x  dx/dt  =  D(x) x  -  Px  -  q                                 (VI.2)
```

### 6.3 Relation to the Loihi PIPG x-update

```
Loihi [Eq. 3, no constraints]:  x_{t+1} = x_t - α (Qx_t + p)

Continuous-time limit (α small, τ_x = 1/α_eff):
  τ_x  dx/dt  =  -(Px + q)                                [pure GD]  (GD-compare)

Our SL equation (VI.2) = pure GD  +  SL restoring term D(x)x:
  τ_x  dx/dt  =  D(x)x  -  Px  -  q

The SL restoring term D(x)x provides:
  (a) Amplitude bounding (soft box constraint on x)
  (b) Non-zero equilibrium (phase-coherent output for hardware readout)
  (c) Physical meaning: nonlinear gain in the oscillator circuit
```

### 6.4 Physical Realisation in the Bhowmik Group Platform

In the CMOS ring oscillator implementation of Bhowmik's group:

- Each ring oscillator i corresponds to one decision variable x_i
- The natural oscillation amplitude corresponds to √μ_i, set by the supply voltage and transistor sizing
- Resistive coupling between oscillators implements J_{ij} = -P_{ij}: positive coupling (excitatory, in-phase pull) for negative P_{ij}, negative coupling (inhibitory, anti-phase pull) for positive P_{ij}
- In the FeFET-coupled array, coupling conductances G_{ij} are set via ferroelectric polarisation (non-volatile, programmable), directly encoding the QP matrix P_{ij}. This is equivalent to programming synaptic weights on Loihi 2.
- The external field h_i = -q_i is set by injecting a DC current bias into oscillator i, which shifts the equilibrium amplitude.

---

## Section VII — Core Derivation: Equality Constraints via LagONN

### 7.1 Equality Constraint Structure

From OSQP, equality constraints arise from the slack variable formulation `A_c x = z`, as well as from explicit equality constraints in the MPC (dynamics equations `s_{k+1} = A_d s_k + B_d u_k`). Collecting all equalities:

```
C x  =  d    (m_eq equality constraints)                              (Eq-def)

For OSQP: C = A_c, d = z (but z is also a variable, see Section VIII)
For MPC dynamics: rows of A_c corresponding to l_i = u_i = b_i

The constraint residual vector:  G(x) = Cx - d  ∈ R^{m_eq}
At optimum: G(x*) = 0   (primal feasibility)
```

### 7.2 Lagrangian and Saddle-Point Formulation

Introduce Lagrange multipliers λ^eq ∈ R^{m_eq} (unconstrained sign). The Lagrangian is:

```
L^eq(x, λ^eq) = f(x)  +  (λ^eq)^T (Cx - d)
              = (1/2) x^T P x + q^T x  +  (λ^eq)^T (Cx - d)         (Lagrangian-eq)

KKT stationarity w.r.t. x:   Px + q + C^T λ^eq = 0
KKT primal feasibility:       Cx - d = 0

These define the saddle point (x*, λ^{eq*}) of L^eq.
```

### 7.3 LagONN Phase Encoding of λ^eq

Following Delacour 2025 Eq. 17, each equality Lagrange multiplier λ_m^eq is encoded as the cosine of a Lagrange oscillator phase:

```
λ_m^eq(t)  =  u_m cos( φ_m^{λ,eq}(t) )

where u_m > 0 is an amplitude scale (set to 1 for simplicity) and
φ_m^{λ,eq} ∈ [0, 2π) is the Lagrange oscillator phase.

Since G_m(x) = (Cx - d)_m is real-valued, Im[Z_m] = 0,
and L_T = Σ_m λ_m^eq G_m = Σ_m cos(φ_m^{λ,eq}) G_m(x).

Lagrange oscillator ascent dynamics (from LagONN Eq. D2):
  τ_eq  dφ_m^{λ,eq}/dt  =  +dL_T / dφ_m^{λ,eq}
                         =  -sin(φ_m^{λ,eq}) G_m(x)
                         =  -sin(φ_m^{λ,eq}) (Cx - d)_m             (VII.1)

At saddle: dφ_m^{λ,eq}/dt = 0  =>  sin(φ_m^{λ,eq}) G_m = 0
  Case 1: G_m(x*) = 0   [constraint satisfied — desired!]
  Case 2: φ_m^{λ,eq} ∈ {0, π}   [λ_m frozen, degenerate]
  Case 1 is the global saddle by Theorem 2 (Delacour 2025).
```

### 7.4 Modified Decision Variable Dynamics with Equality Constraints

Adding the equality constraint correction to the decision variable dynamics (gradient descent on the full Lagrangian L^eq with respect to x):

```
τ_x  dx_i/dt  =  (μ_x - x_i²) x_i  -  df/dx_i  -  dL_T/dx_i
              =  (μ_x - x_i²) x_i  -  (Px+q)_i  -  (C^T λ^eq)_i

Matrix form:
  τ_x  dx/dt  =  D(x)x  -  Px  -  q  -  C^T λ^eq                    (VII.2)

where  λ_m^eq = cos(φ_m^{λ,eq})  from (VII.1).

Coupled system (VII.1)+(VII.2) implements the continuous-time Arrow-Hurwicz
saddle-point algorithm for the equality-constrained QP.
```

---

## Section VIII — Core Derivation: Inequality Constraints (OSQP Form)

### 8.1 OSQP Box Constraints: Upper and Lower Bounds

The OSQP constraint `l ≤ A_c x ≤ u` decomposes into m independent constraints per row k:

```
(A_c x)_k  ≤  u_k    [upper bound]
(A_c x)_k  ≥  l_k    [lower bound]                                   (OSQP-ineq)

One-sided inequality formulation:
  g_k^{up}(x) = (A_c x)_k - u_k  ≤  0   (satisfied when ≤ 0)
  g_k^{lo}(x) = l_k - (A_c x)_k  ≤  0   (satisfied when ≤ 0)

KKT conditions for inequality constraints:
  λ_k^{up} ≥ 0,  g_k^{up}(x*) ≤ 0,  λ_k^{up} g_k^{up}(x*) = 0
  λ_k^{lo} ≥ 0,  g_k^{lo}(x*) ≤ 0,  λ_k^{lo} g_k^{lo}(x*) = 0
```

### 8.2 Non-Negative Amplitude Encoding for Inequality Multipliers

For inequality constraints, λ_k^{up}, λ_k^{lo} ≥ 0. Unlike equality multipliers (unconstrained sign), inequality multipliers require a non-negativity constraint. We use **amplitude encoding**:

```
λ_k^{up}(t)  =  r_k^{up}(t)  ≥  0   (SL amplitude, always non-negative)
λ_k^{lo}(t)  =  r_k^{lo}(t)  ≥  0   (SL amplitude, always non-negative)     (Lam-amp)

Physical interpretation: these are the amplitudes of additional 'Lagrange'
SL oscillators — one per active constraint. Their natural equilibrium
amplitude (without coupling) is √ν_λ, but constraint forces
push them away from equilibrium in proportion to constraint violation.

This is the continuous-time analog of the θ_G (ReLU) operation
in the Loihi PIPG (Eq. 7 of Mangalore et al.).
```

### 8.3 Inequality Lagrange Oscillator Dynamics

Following the PIPG dual ascent (gradient ascent on the Lagrangian w.r.t. inequality multipliers), with projection onto the non-negative orthant (implemented by amplitude encoding):

```
Upper bound Lagrange oscillators (k = 1...m):
  τ_ineq  dr_k^{up}/dt  =  max( 0,  (A_c x - u)_k )
                         =  ReLU[ g_k^{up}(x) ]                      (VIII.1a)

Lower bound Lagrange oscillators (k = 1...m):
  τ_ineq  dr_k^{lo}/dt  =  max( 0,  (l - A_c x)_k )
                         =  ReLU[ g_k^{lo}(x) ]                      (VIII.1b)

Interpretation:
  - Constraint satisfied (g_k ≤ 0): dr_k/dt = 0 -> λ_k stays constant
  - Constraint violated  (g_k > 0): dr_k/dt > 0 -> λ_k increases
  - Growing λ_k drives the decision variables to correct the violation
  - Non-negativity is automatic (amplitude increases monotonically when violated)

Complementary slackness at equilibrium:
  r_k^{up*} * g_k^{up}(x*) = 0   [λ_k=0 if constraint inactive,
                                    g_k=0 if constraint active with λ_k>0]
```

### 8.4 Contribution to Decision Variable Dynamics

The inequality constraint correction enters the decision variable dynamics through the gradient of the Lagrangian w.r.t. x:

```
Full Lagrangian (equality + inequality):
  L(x, λ^eq, λ^up, λ^lo)
  = f(x)  +  (λ^eq)^T (Cx - d)
           +  (λ^up)^T (A_c x - u)
           +  (λ^lo)^T (l - A_c x)

Gradient w.r.t. x_i:
  dL/dx_i = (Px+q)_i + (C^T λ^eq)_i + (A_c^T λ^up)_i - (A_c^T λ^lo)_i

Net constraint injection into decision oscillator i:
  F_i^{constr} = -(C^T λ^eq)_i - (A_c^T (λ^up - λ^lo))_i            (VIII.2)
```

### 8.5 Combined Slack Variable Formulation (ADMM-aligned)

The OSQP slack variable `z = A_c x` with `l ≤ z ≤ u` can also be treated using an augmented Lagrangian / ADMM structure, which aligns more directly with the OSQP implementation. In this variant:

```
Introduce slack SL oscillators:  z_k ∈ R  (m additional oscillators)
encoded as bounded amplitudes:  z_k ∈ [l_k, u_k]  (soft via SL restoring)

Augmented Lagrangian:
  L_ρ(x, z, y) = (1/2)x^T P x + q^T x
               + y^T (A_c x - z)
               + (ρ/2) ||A_c x - z||²

Coupled SL dynamics for ADMM-like flow:
  τ_x  dx/dt = D(x)x - Px - q - A_c^T y - ρ A_c^T (A_c x - z)      (VIII.3)
  τ_z  dz/dt = D_z(z)z + y + ρ (A_c x - z)   [soft-clipped to [l,u]] (VIII.4)
  τ_y  dy/dt = A_c x - z                                              (VIII.5)

D_z(z)z = diag(ν_z - z_k²) z  (SL restoring for slack oscillators)
          provides soft projection onto [l_k, u_k] when ν_z = max(l_k², u_k²)
```

> **Note:** Both formulations (VIII.1+VII.2 and VIII.3–5) are equivalent at equilibrium. The ADMM formulation (VIII.3–5) is simpler to implement in software (maps directly to OSQP structure) and is preferred for the Python simulation. The LagONN formulation (VIII.1+VII.2) is preferred for analog hardware because phase-encoded multipliers (λ^eq) and amplitude-encoded multipliers (λ^up, λ^lo) map to distinct physical oscillator types.

---

## Section IX — Complete SL+LagONN Dynamical System

### 9.1 Full System Equations

Collecting all oscillator dynamics, the complete neuromorphic QP solver is a system of ODEs in the variables: `x ∈ R^n` (decision), `φ^eq ∈ R^{m_eq}` (equality Lagrange phases), `λ^up ∈ R_+^m` (upper bound amplitudes), `λ^lo ∈ R_+^m` (lower bound amplitudes).

> **Complete SL+LagONN System for OSQP QP**
>
> **Decision Variable Oscillators (n SL oscillators)**
> ```
> τ_x  dx_i/dt = (μ_x - x_i²) x_i - (Px+q)_i
>                - (C^T λ^eq)_i - (A_c^T λ^net)_i               (IX.1)
>
> where  λ^net = λ^up - λ^lo  (net constraint force)
>        C^T λ^eq  uses  λ_m^eq = cos(φ_m^eq)
> ```
>
> **Equality Lagrange Oscillators (m_eq phase oscillators)**
> ```
> τ_eq  dφ_m^eq/dt = -sin(φ_m^eq) (Cx - d)_m                   (IX.2)
>        (λ_m^eq = cos(φ_m^eq))
> ```
>
> **Upper Bound Lagrange Oscillators (m amplitude oscillators)**
> ```
> τ_ineq  dλ_k^up/dt = max(0, (A_c x - u)_k)                    (IX.3)
> ```
>
> **Lower Bound Lagrange Oscillators (m amplitude oscillators)**
> ```
> τ_ineq  dλ_k^lo/dt = max(0, (l - A_c x)_k)                    (IX.4)
> ```
>
> **Total oscillators:** n + m_eq + 2m
> **For 2-DOF arm MPC (N=20):** n=120, m≈240, m_eq≈80 ⟹ ~560 oscillators

### 9.2 Hyperparameter Summary

| Parameter | Role | Recommended Value | Effect of Mistuning |
|---|---|---|---|
| τ_x | Decision oscillator time constant | 1.0 (normalised) | Larger: slower convergence |
| τ_eq | Equality Lagrange time constant | τ_x × 0.5 | Too large: weak constraint enforcement |
| τ_ineq | Ineq. Lagrange time constant | τ_x × 1.0 | Too small: oscillatory λ |
| μ_x | SL bifurcation parameter | max(|x_i^max|)² | Too large: biased solution; too small: no bounding |
| ρ | ADMM penalty (ADMM variant) | 1 / σ_max(P) | Too large: slow x-update; too small: poor constraint |
| dt | ODE integrator step size | 0.01–0.05 | Too large: instability; too small: slow |

### 9.3 Annealing Schedule (Following Loihi PIPG)

Mangalore et al. use a decaying step size α_t and growing penalty β_t for accelerated convergence. In our continuous-time system, this translates to time-varying τ_x and τ_ineq:

```
τ_x(t)     =  τ_x^0 / (1 + γ_x · t)       [decreasing: faster updates late]
τ_ineq(t)  =  τ_ineq^0 · (1 + γ_ineq · t) [increasing: stronger enforcement]   (Anneal)

Or equivalently (following Loihi's halving/doubling schedule):
  At time t_k = k · T_anneal:  τ_x <- τ_x / 2,  τ_ineq <- τ_ineq * 2

This is implemented in hardware by scaling coupling conductances
or capacitor values at discrete checkpoints.

Typical schedule: 5–10 annealing steps over total solve time T_solve.
```

---

## Section X — Saddle-Point Analysis and Convergence

### 10.1 Lyapunov Function for the Complete System

Define the primal Lagrangian as the candidate Lyapunov function. For the equality-constrained case (inequalities treated as equalities with slacks), the system (IX.1)–(IX.2) is an Arrow-Hurwicz gradient flow:

```
Primal Lyapunov candidate:  V(x, φ^eq) = L^eq(x, cos(φ^eq))
                          = (1/2) x^T P x + q^T x + Σ_m cos(φ_m^eq)(Cx-d)_m

Time derivative along system trajectories:
  dV/dt = (dV/dx)^T dx/dt + Σ_m (dV/dφ_m^eq) dφ_m^eq/dt

From (IX.1) (ignoring SL restoring term for clarity):
  dx_i/dt = -(1/τ_x) dL^eq/dx_i
  => (dV/dx)^T dx/dt = -(1/τ_x) ||dL^eq/dx||²  ≤  0   [descent in x]    (Lyapunov)

From (IX.2):
  dL^eq/dφ_m^eq = -sin(φ_m^eq)(Cx-d)_m
  dphi_m/dt = -sin(φ_m)(Cx-d)_m   =>   this IS gradient ASCENT in λ_m
  Therefore: (dV/dφ^eq) dφ^eq/dt = +(1/τ_eq) ||dL^eq/dφ^eq||²  ≥  0

Combined:  dV/dt = -(1/τ_x)||∇_x L||² + (1/τ_eq)||∇_φ L||²

V is NOT a strict Lyapunov function (it can increase due to ascent in φ).
The saddle-point structure is essential: x descent and φ^λ ascent together
drive the system toward the saddle (x*, φ^{λ*}).
```

### 10.2 Convergence by Minimax Duality

> **Convergence Result** (from Delacour 2025 Theorem 2 + Convex QP Theory)
>
> For the equality-constrained convex QP with P ∈ S_+^n (PSD) and Slater condition satisfied (feasible interior point exists):
>
> 1. **Strong duality holds:** `min_x max_λ L(x,λ) = max_λ min_x L(x,λ) = f*`
>
> 2. The saddle point (x\*, λ\*) satisfies KKT conditions (IX.1=0, IX.2=0, IX.3=0, IX.4=0).
>
> 3. For the continuous-time Arrow-Hurwicz flow (IX.1)–(IX.2), convergence is guaranteed when `τ_eq / τ_x < 2 / σ_max(CC^T)` (time-scale condition).
>
> 4. The SL restoring term introduces a bias of order μ_x at the equilibrium: `|x_i* - x_i^QP| = O(μ_x / σ_min(P))` as μ_x → 0. For `μ_x << σ_min(P)` the bias is negligible.
>
> 5. For the inequality-augmented system with ReLU Lagrange dynamics, convergence to KKT is guaranteed by PIPG theory (Yu et al. 2021).

### 10.3 Effect of SL Restoring Term on Solution Accuracy

The bias introduced by the SL term can be quantified exactly. At equilibrium of the full system (IX.1) with constraints satisfied:

```
(μ_x - x_i*²) x_i*  =  (Px* + q)_i  +  (C^T λ^{eq*} + A_c^T λ^{net*})_i

The RHS is the KKT residual: at the exact QP optimum, RHS = 0.
So the SL equilibrium satisfies:  (μ_x - x_i*²) x_i* = 0             (SL-bias)
Solutions: x_i* = 0  OR  x_i*² = μ_x => x_i* = ±√μ_x

This is incorrect unless μ_x is tuned! Remedy options:
  Option A: Set μ_x very small (epsilon-SL): x_i* ≈ 0 biased.
            Then add a separate projection at readout.
  Option B: Time-scale separation: τ_x << τ_SL_restoring.
            Run gradient flow fast, SL restoring slow -> SL barely acts.
  Option C: Modify SL as:  (μ_x - (x_i - x_i^c)²)(x_i - x_i^c)
            where x_i^c is a moving center updated by gradient flow.
  Option D (RECOMMENDED): Use ADMM formulation (VIII.3-5) where SL
            acts purely as a soft box constraint for z oscillators,
            and x oscillators use pure gradient flow (μ_x -> 0).
```

### 10.4 Recommended Implementation Choice

> **Recommended Architecture: Hybrid SL/Gradient Flow**
>
> **x oscillators:** Pure gradient flow — `τ_x dx/dt = -dL/dx`, μ_x = 0. Box constraints l_i ≤ x_i ≤ u_i enforced by clipping at readout (or by separate bounded SL oscillators for bound enforcement).
>
> **z oscillators (slack):** SL with `μ_z = max(l_k², u_k²)`. Provides soft projection onto [l_k, u_k]. Coupled to x via ADMM penalty `ρ*(A_c x - z)`.
>
> **y oscillators (ADMM dual):** Lagrange amplitude y_k (sign-free). `τ_y dy/dt = A_c x - z` — pure dual ascent, no SL restoring.
>
> This exactly implements the continuous-time ADMM / PIPG algorithm, coincides with the Loihi PIPG approach, and maps cleanly to hardware.

---

## Section XI — MPC Warm-Start and Receding Horizon

### 11.1 Receding Horizon MPC Loop

In MPC, at each time step k the QP changes slightly: the initial state changes, the horizon shifts by one step, and the cost/constraint matrices may change slightly. The warm-start strategy re-uses the previous solution as the initial condition for the oscillator network:

```
MPC Loop (Algorithm):
  1. Measure current state s_k
  2. Form QP: compute P(s_k), q(s_k), A_c(s_k), l(s_k), u(s_k)
  3. Update oscillator couplings: J_{ij} <- -P_{ij}, h_i <- -q_i,
                                  A matrix in constraint oscillators
  4. Warm-start: x_0 = x_{k-1}^{shifted}  (shift previous solution by 1 step)
                 λ^{eq/up/lo}_0 = λ_{k-1}  (previous multipliers)
  5. Evolve ODE for T_solve oscillation cycles (or until convergence)
  6. Read out: u_k* = x_0*(first n_u components of decision vector)
  7. Apply u_k* to the arm, advance to step k+1                       (MPC-loop)

Convergence monitoring: ||dx/dt||² + ||dφ^eq/dt||² < ε_conv
Typical T_solve: 50–200 oscillation cycles (depends on μ_x, τ, problem size)
```

### 11.2 Warm-Start Stability

Mangalore et al. verified (for the Loihi implementation) that warm-starting with approximate solutions is stable in closed-loop MPC: since only the first 20–30 ms of the MPC solution is used before re-solving, small optimality gaps (8%) compound but do not destabilise the arm if the warm-start is close to the true optimum. For the 2-DOF arm:

```
Warm-start quality: ||x_0 - x*_k||_P ≤ δ  (distance to true optimum)

Theoretical convergence bound (PIPG theory, Yu et al. 2021):
  After T iterations:  ||x_T - x*||_P ≤ ρ^T · ||x_0 - x*||_P        (Warm-start)
  where  ρ = 1 - α σ_min(P) / (1 + α σ_max(P))  < 1

With warm-start: δ_{k+1} ≤ ρ^T · δ_k + O(||ΔQP||)
  ΔQP: change in QP matrices between steps (bounded by arm dynamics)
  For slowly varying MPC: δ stays small -> fast convergence.

Empirical guideline (from Mangalore et al.): ~55 PIPG iterations sufficient
for 8% optimality across all problem sizes tested (264–8424 variables).
Our 2-DOF arm (120 variables) is much smaller -> fewer iterations needed.
```

---

## Section XII — Software Implementation (Python / SciPy)

### 12.1 Overview and Dependencies

The software implementation simulates the SL+LagONN dynamics in Python, using `scipy.integrate.solve_ivp` with the Fehlberg / RK45 integrator (consistent with Delacour 2025 who uses the Fehlberg ODE solver). The implementation is structured in four components:

- **Component 1 — QP Builder:** Constructs P, q, A_c, l, u from the 2-DOF arm model using CasADi or sympy linearisation.
- **Component 2 — SL+LagONN ODE:** Defines the right-hand side of (IX.1)–(IX.4) as a Python function.
- **Component 3 — Solver Wrapper:** Calls `solve_ivp`, implements warm-start, monitors convergence.
- **Component 4 — Benchmark:** Compares against OSQP using identical QP instances, measures TTS and solution quality.

### 12.2 Core ODE Function

The central ODE function implementing equations (IX.1)–(IX.4):

```python
def sl_lagonn_ode(t, state, P, q, C, d, Ac, l_vec, u_vec, params):
    n, m_eq, m = params['n'], params['m_eq'], params['m']
    mu_x   = params['mu_x']
    tau_x  = params['tau_x']
    tau_eq = params['tau_eq']
    tau_ineq = params['tau_ineq']

    x      = state[:n]                    # decision variables
    phi_eq = state[n:n+m_eq]              # equality Lagrange phases
    lam_up = state[n+m_eq:n+m_eq+m]       # upper bound multipliers (>=0)
    lam_lo = state[n+m_eq+m:]             # lower bound multipliers (>=0)

    lam_eq  = np.cos(phi_eq)              # phase encoding of equality multipliers
    lam_net = lam_up - lam_lo             # net inequality force

    # ── Decision oscillator dynamics (IX.1) ───────────────────────────────
    SL_restore = (mu_x - x**2) * x
    cost_grad  = P @ x + q
    eq_force   = C.T @ lam_eq             if m_eq > 0 else 0.0
    ineq_force = Ac.T @ lam_net
    dx = (1/tau_x) * (SL_restore - cost_grad - eq_force - ineq_force)

    # ── Equality Lagrange oscillators (IX.2) ──────────────────────────────
    G_eq  = C @ x - d
    dphi  = -(1/tau_eq) * np.sin(phi_eq) * G_eq   if m_eq > 0 else np.array([])

    # ── Inequality Lagrange oscillators (IX.3, IX.4) ──────────────────────
    viol_up = np.maximum(0.0, Ac @ x - u_vec)
    viol_lo = np.maximum(0.0, l_vec - Ac @ x)
    dlam_up = (1/tau_ineq) * viol_up
    dlam_lo = (1/tau_ineq) * viol_lo

    return np.concatenate([dx, dphi, dlam_up, dlam_lo])
```

### 12.3 Solver Wrapper with Warm-Start

```python
def solve_qp_sl_lagonn(P, q, Ac, l_vec, u_vec,
                        C=None, d=None,
                        x0=None, lam0=None,
                        params=None, T_solve=20.0, tol=1e-3):
    n = len(q)
    m = len(l_vec)
    m_eq = len(d) if d is not None else 0

    # Warm start: use previous solution or zero
    if x0 is None:   x0   = np.zeros(n)
    if lam0 is None: lam0 = np.zeros(m_eq + 2*m)
    state0 = np.concatenate([x0, lam0])

    # Convergence monitoring via event function
    def converged(t, y, *args):
        dydt = sl_lagonn_ode(t, y, *args)
        return np.linalg.norm(dydt) - tol   # event fires when norm < tol
    converged.terminal  = True
    converged.direction = -1

    sol = solve_ivp(
        sl_lagonn_ode, [0, T_solve], state0,
        args=(P, q, C if C is not None else np.zeros((0,n)),
              d if d is not None else np.zeros(0),
              Ac, l_vec, u_vec, params),
        method='RK45', events=converged, dense_output=True,
        rtol=1e-4, atol=1e-6
    )

    x_star   = sol.y[:n, -1]
    lam_star = sol.y[n:, -1]
    TTS      = sol.t[-1]          # time to solution
    return x_star, lam_star, TTS
```

### 12.4 ADMM Variant (Simpler, Recommended for Initial Testing)

For initial validation, the ADMM-aligned formulation (VIII.3–5) is simpler to implement and directly comparable to OSQP. No equality Lagrange phases are needed — all constraints use the y oscillator:

```python
def admm_sl_ode(t, state, P, q, Ac, l_vec, u_vec, params):
    n = params['n'];  m = params['m']
    tau_x = params['tau_x'];  tau_z = params['tau_z']
    tau_y = params['tau_y'];  rho   = params['rho']
    nu_z  = params['nu_z']    # SL restoring for slack oscillators

    x = state[:n]
    z = state[n:n+m]
    y = state[n+m:]

    # (VIII.3): x gradient flow + ADMM coupling
    dx = -(1/tau_x) * (P @ x + q + Ac.T @ y + rho * Ac.T @ (Ac @ x - z))

    # (VIII.4): z SL oscillator, soft-clipped to [l, u]
    dz_raw = (1/tau_z) * (y + rho * (Ac @ x - z))
    SL_z   = (nu_z - z**2) * z         # soft box restoring
    dz     = dz_raw + SL_z

    # (VIII.5): y dual ascent (pure integral)
    dy = (1/tau_y) * (Ac @ x - z)

    return np.concatenate([dx, dz, dy])

# Convergence criterion
# ||Ac@x - z||² + ||Px + q + Ac^T @ y||² < epsilon
```

> **Recommendation:** Implement and test the ADMM-SL variant first (Phase 1), then layer in the LagONN Lagrange phase oscillators (Phase 2) for improved constraint handling.

---

## Section XIII — Hardware Mapping: Bhowmik Group Platform (IIT Bombay)

### 13.1 Available Physical Platforms

Prof. Bhowmik's group at IIT Bombay has demonstrated and simulated two primary oscillator platforms, both modelled as Stuart-Landau systems in the weakly nonlinear regime:

| Platform | Physical Oscillator | SL Parameters | Coupling Mechanism | Key Publication |
|---|---|---|---|---|
| CMOS Ring Oscillator | 5-/7-stage CMOS inverter ring | ω₀~GHz, μ~supply bias | Resistive/capacitive coupling networks | Bhowmik et al., ISCAS 2024 |
| FeFET-Coupled Ring Osc. | Ring osc. + FeFET coupling gates | Programmable μ_i via FeFET state | FeFET conductance G_{ij} = J_{ij} weight (non-volatile) | Bhowmik group, 2024 preprint |
| SHNO Array | Spin Hall nano-oscillator | ω₀~GHz, μ~bias current | Dipole coupling, distance-dependent | Hemadribhotla et al. 2021 |

### 13.2 FeFET-Coupled CMOS Ring Oscillator: Recommended Platform

The FeFET-coupled ring oscillator array is the most directly applicable platform for QP because it provides:

- **Programmable non-volatile weights:** FeFET polarisation state encodes coupling conductance G_{ij} proportional to P_{ij}. Changing the QP matrix between MPC iterations requires writing new FeFET states — achievable at microsecond timescales via voltage pulses.
- **Real-valued amplitude:** Without SHIL, ring oscillators operate in the amplitude regime — the output voltage amplitude is the decision variable x_i.
- **External current injection for q:** A DC current bias shifts the oscillator equilibrium amplitude, implementing the linear cost term q_i.
- **On-chip ReLU for Lagrange oscillators:** The ReLU in (IX.3)–(IX.4) is implemented as a half-wave rectifier circuit at the Lagrange oscillator input — a standard CMOS circuit. This provides hardware constraint enforcement without digital logic.

### 13.3 Mapping QP Matrices to Hardware Parameters

```
QP Parameter      ->  Hardware Parameter
─────────────────────────────────────────────────────────────────────
P_{ij}  (i ≠ j)   ->  Coupling conductance G_{ij} = |P_{ij}| / V_ref
                       Sign: excitatory (G>0) for P_{ij}<0,
                             inhibitory  (G<0) for P_{ij}>0

P_{ii}            ->  Self-feedback resistor at oscillator i: R_i = 1/P_{ii}

q_i               ->  DC bias current I_i = -q_i / V_ref  at oscillator i

A_c (constraint)  ->  Coupling from decision osc to Lagrange osc:
                       G_{ki}^{constr} = (A_c)_{ki}

u_k, l_k (bounds) ->  Threshold voltages for Lagrange osc k:
                       V_thresh^{up,k} = u_k · V_ref / a_k
                       V_thresh^{lo,k} = l_k · V_ref / a_k

Lagrange feedback  ->  Current injection back to decision osc i
                        from Lagrange osc k: weight = (A_c)_{ki}
```

### 13.4 Simulation Methodology for Hardware Validation

Before tape-out, the hardware is validated through the following simulation hierarchy (consistent with Bhowmik group methodology):

**Level 1 — Algorithm Simulation (Python):** Validate SL+LagONN ODE (IX.1)–(IX.4) against OSQP on MPC-arm QP instances. Target: solution within 8% of OSQP optimal within 100 ms simulation time.

**Level 2 — Kuramoto/SL Network Simulation (Python):** Replace ideal gradient flow with full SL oscillator model including phase noise (Wiener process term). Validate that noise does not degrade QP solution beyond acceptable threshold.

**Level 3 — Circuit Simulation (SPICE):** Build transistor-level CMOS ring oscillator model in SPICE, add FeFET coupling circuits. Validate coupling conductances produce correct J_{ij} mapping. Check input/output voltage scales.

**Level 4 — MuJoCo Closed-Loop Test:** Run 2-DOF arm simulation in MuJoCo, replace OSQP solver with SL+LagONN (Python implementation), measure arm trajectory tracking quality and constraint satisfaction.

---

## Section XIV — Benchmarking Against OSQP

### 14.1 Benchmarking Metrics

Following Mangalore et al. (Loihi paper) and Delacour 2025 (LagONN paper):

| Metric | Definition | Target (vs OSQP) |
|---|---|---|
| TTS (ms) | Time for solution quality to reach within ε of OSQP optimal | < 10 ms per MPC iteration |
| ETS (mJ) | Energy consumed from problem setup to solution readout | < 1% of CPU OSQP |
| EDP (mJ·ms) | Energy × delay product | < 1/100 of CPU OSQP |
| Optimality gap (%) | `\|\|f(x*_ours) - f(x*_OSQP)\|\| / f(x*_OSQP) × 100` | < 8% (Loihi standard) |
| Constraint violation | `max_k max(0, (A_c x* - u)_k, (l - A_c x*)_k) / \|\|x*\|\|` | < 1e-3 |
| TTS (# ODE steps) | Oscillator cycles to convergence (hardware proxy) | < 200 cycles |

### 14.2 Test Dataset

Following Delacour 2025 (Figure 5 methodology) and Mangalore et al.:

```
Dataset: 2-DOF planar arm MPC QP instances
  - Generated from simulated arm trajectories in MuJoCo
  - 6 problem sizes: N = 5, 10, 15, 20, 30, 40  (horizon steps)
  - Corresponding n_ξ = 30, 60, 90, 120, 180, 240  variables
  - 10 QP instances per size (varying difficulty / condition number)
  - Total: 60 QP instances

For each instance, report: mean TTS, std TTS, optimality gap, max violation.
Present as boxplots (Delacour style) or mean ± std curves (Mangalore style).

OSQP reference: OSQP v0.6+ with default settings (accurate mode),
  run on same CPU to establish baseline TTS and f*.
```

### 14.3 Implementation Roadmap (Phased)

| Phase | Task | Tool / Method | Success Criterion |
|---|---|---|---|
| **1** *(current)* | ADMM-SL baseline (no LagONN) | Python scipy.integrate vs OSQP | Converges to OSQP ±8% in all 60 instances |
| **2** | Add LagONN equality phase oscillators | Python + full (IX.1)–(IX.2) | Equality constraints satisfied to 1e-3 |
| **3** | Add LagONN inequality amplitude oscillators | Python + full (IX.1)–(IX.4) | All KKT conditions satisfied to 1e-3 |
| **4** | MuJoCo closed-loop MPC test | MuJoCo + Python SL-solver | Arm tracks trajectory within 5% of OSQP baseline |
| **5** | Hardware simulation (SPICE / Kuramoto) | SPICE or Bhowmik group SL simulator | Match Phase 3 accuracy with noise ≤ 10% |
| **6** | Hardware validation | Bhowmik group CMOS/SHNO platform | TTS and EDP gains vs CPU OSQP |

---

## Section XV — References

1. **Delacour, C. et al.** (2025). Lagrange oscillatory neural network for solving constrained combinatorial optimization problems. *Neuromorphic Computing and Engineering* 5, 044004. DOI: 10.1088/2634-4386/ae0eab. GitHub: [github.com/corentindelacour/Lagrange-oscillatory-neural-network](https://github.com/corentindelacour/Lagrange-oscillatory-neural-network)

2. **Mangalore, A.R., Fonseca Guerra, G.A., Risbud, S.R., Stratmann, P., and Wild, A.** (2024). Neuromorphic Quadratic Programming for Efficient and Scalable Model Predictive Control. *IEEE Robotics and Automation Magazine*. DOI: 10.1109/MRA.2024.3415005

3. **Yu, Y., Elango, P., and Acikmese, B.** (2021). Proportional-integral projected gradient method for model predictive control. *IEEE Control Systems Letters* 5(6), 2174–2179. DOI: 10.1109/LCSYS.2020.3044977

4. **Wang, T. and Roychowdhury, J.** (2019/2021). OIM: Oscillator-Based Ising Machines for Solving Combinatorial Optimisation Problems. Springer UCNC 2019; *Natural Computing* 2021. arXiv: 1903.07163

5. **Stellato, B., Banjac, G., Goulart, P., Bemporad, A., and Boyd, S.** (2020). OSQP: An operator splitting solver for quadratic programs. *Mathematical Programming Computation* 12(4), 637–672. DOI: 10.1007/s12532-020-00179-2

6. **Hemadribhotla, S.V., Muduli, P.K., Garg, N., and Bhowmik, D.** (2021). Kuramoto-model-based data classification using synchronization dynamics of uniform-mode spin Hall nano-oscillators. *IEEE Transactions on Magnetics*. DOI: 10.1109/TMAG.2021.3101637

7. **Bhowmik, D. et al.** (2024). Improved Computation Time of Electronic Ring Oscillator Networks Compared to a Popular Classical Optimization Algorithm for the Max-Cut Problem. [Bhowmik group, IIT Bombay]

8. **Bhowmik, D. et al.** (2024). Symbol Detection in a MIMO Wireless Communication System Using a FeFET-coupled CMOS Ring Oscillator Array. [Bhowmik group, IIT Bombay — demonstrates FeFET programmable coupling]

9. **Bhowmik, D.** (2024). Neuromorphic and Ising Computing using Emerging Non-Volatile Memory Devices for Edge Applications: Wireless Communication and Robotics. [Bhowmik group — directly covers robotics applications]

10. **Bhowmik, D.** (2024). Spintronics-Based Neuromorphic Computing. Springer Nature. ISBN: 978-981-97-4445-9. [Comprehensive book including SHNO oscillator networks and ONN chapters]

11. **Davies, M. et al.** (2021). Advancing neuromorphic computing with Loihi: A survey of results and outlook. *Proceedings of the IEEE* 109(5), 911–934. DOI: 10.1109/JPROC.2021.3067593

12. **Orchard, G. et al.** (2021). Efficient neuromorphic signal processing with Loihi 2. *Proc. IEEE Workshop on Signal Processing Systems (SiPS)*, 254–259. DOI: 10.1109/SiPS52927.2021.00053

13. **Sajeeb, M. et al.** (2025). Phase copying in scalable oscillator networks. arXiv: 2503.01177

14. **Pedretti, G. et al.** (2025). GNSAT-N. *npj Unconventional Computing* 2, 7. [Comparison target for LagONN benchmarking in Delacour 2025]

15. **Cross, M.C. and Hohenberg, P.C.** (1993). Pattern formation outside of equilibrium. *Reviews of Modern Physics* 65(3), 851. [Standard reference for Stuart-Landau normal form]

16. **Sleiman, J.-P., Farshidian, F., Minniti, M.V., Hutter, M.** (2021). A unified MPC framework for whole-body dynamic locomotion and manipulation. *IEEE Robotics and Automation Letters* 6(3), 4688–4695. [ANYmal MPC reference, also cited in Mangalore et al.]

17. **Graber, M. and Hofmann, K.** (2024). An integrated coupled oscillator network to solve optimization problems. *Communications Engineering* 3, 126. [1440-oscillator OIM chip: validates scalability of oscillator approach]

18. **Mancoo, A., Keemink, S., and Machens, C.K.** (2020). Understanding spiking networks through convex optimization. *Advances in NeurIPS*, vol. 33, 8824–8835. [Cited in Loihi paper: convex QP via spiking networks]

---

*Document prepared for M.Sc. Thesis — Neuromorphic Edge Intelligence for Industrial Robotics — IIT Bombay, 2025–26. All derivations are grounded in the cited academic literature; equations (IX.1)–(IX.4) constitute the original contribution synthesising Delacour 2025, Mangalore et al. 2024, and SL oscillator theory.*
