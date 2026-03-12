# Problem Statement & Application

Definition of the control problem and robot specification.

## Robot Arm Specification

### 2-DOF Planar Manipulator

```
        q1 (shoulder)
         ↓
    ●========●========●
    |  l1=0.5m  l2=0.5m
    |  m1=1kg   m2=1kg
    |
    ●━━━ (fixed base)
   
    q0 = base angle (shoulder)
    q1 = link angle (elbow)
```

### Kinematics

**End-effector position**:
```
x = l1*cos(q0) + l2*cos(q0+q1)
y = l1*sin(q0) + l2*sin(q0+q1)
```

**Joint limits**: 
- q0, q1 ∈ [-π, π] unrestricted (free rotation)

**Control constraints**:
- τ0, τ1 ∈ [-50, 50] Nm (motor torque limits)

---

## Dynamics Model

### Equations of Motion

Lagrangian mechanics with gravity:

```
M(q) q̈ = τ - C(q, q̇) q̇ - g(q)
```

Where:
- **M(q)**: Mass/inertia matrix (2×2, position-dependent)
- **C(q, q̇)**: Coriolis/centrifugal terms
- **g(q)**: Gravity vector
- **τ**: Applied torques (controls)

### Simplifications Used

In this project, we use **unit mass double integrator** approximation:

```
q̇ = dq
d(dq) = u  (control directly affects acceleration, simplified)
```

This is **not** physically accurate but allows focusing on MPC without heavy nonlinear dynamics.

For **accurate physics** simulation, MuJoCo computes full Lagrangian dynamics.

---

## Control Problem

### Objective

**Reach a target configuration and minimize energy**:

```
J = Σ(t=0 to N-1) ||q(t) - q_ref||²_Q + ||u(t)||²_R + ||q(N) - q_ref||²_Qf
```

Where:
- **Q** = cost weight on tracking error (2×2, typically 10×I)
- **R** = cost weight on control effort (2×2, typically 0.1×I)
- **Q_f** = terminal cost (2×2, typically 2×I)
- **N** = prediction horizon (typically 10 steps)

### Constraints

1. **Dynamics**: x(t+1) = A x(t) + B u(t) for prediction
2. **Control bounds**: u ∈ [-50, 50]
3. **Angle bounds**: q ∈ [-π, π]

---

## Tasks

### 1. Reach Task

**Objective**: Move arm from rest to target angle

**Parameters**:
- Initial: q = [0, 0]
- Target: q = [π/6, π/6] ≈ [30°, 30°]
- Duration: 5 seconds
- Success: Error < 0.1 rad at end

**Difficulty**: Easy - single point target

### 2. Trajectory Tracking (Circle)

**Objective**: Follow a reference trajectory

**Parameters**:
- Center: [π/4, π/4]
- Radius: 0.2 rad
- Period: 10 seconds
- Success: Maintain error < 0.3 rad throughout

**Reference trajectory** (parametric):
```
q_0(t) = π/4 + 0.2*cos(2πt/10)
q_1(t) = π/4 + 0.2*sin(2πt/10)
```

**Difficulty**: Medium - continuous tracking

### 3. Multi-Point Tracking (Square)

**Objective**: Visit sequence of target points

**Parameters**:
- Corners: [π/6, π/6], [π/3, π/6], [π/3, π/3], [π/6, π/3]
- Dwell time: 2.5 sec per corner
- Total duration: 10 seconds
- Success: Settle at each corner

**Trajectory**:
```
t ∈ [0, 2.5s]:     Go to corner 1
t ∈ [2.5, 5s]:     Go to corner 2
t ∈ [5, 7.5s]:     Go to corner 3
t ∈ [7.5, 10s]:    Go to corner 4
```

**Difficulty**: Hard - discrete target switching

---

## Performance Metrics

### Control Performance

| Metric | Formula | Target |
|--------|---------|--------|
| **Tracking Error** | `\|\|q - q_ref\|\|` | < 0.3 rad |
| **Settling Time** | Time to reach error | < 3 sec |
| **Overshoot** | Max error above target | < 20% |
| **Steady-State Error** | Final error | < 0.01 rad |

### Optimality

| Metric | Formula | Excellent |
|--------|---------|-----------|
| **Optimality Gap** | `(f_solver - f_opt)/f_opt` | < 1% |
| **Constraint Viol.** | `max(\|Cx-d\|, max(0,Acx-u))` | < 1e-6 |
| **Convergence** | Iterations to solution | < 1000 |

### Real-Time Capability

| Metric | Target |
|--------|--------|
| **Solve Time** | < 100 ms (10Hz control) |
| **CPU Usage** | < 50% single core |
| **Memory** | < 100 MB |

---

## Solver Comparison Problem

All solvers are evaluated on same QP instance:

**Standard 2×2 Problem**:
```
min 0.5 x^T P x + q^T x
s.t. Cx = d
     l ≤ Ax ≤ u
```

**Expected Optimal Solution**:
- x_opt = [1.0, 2.0] (example)
- f_opt ≈ 1.234

**Solver Requirements**:
1. ✅ Find solution within 1% of optimal
2. ✅ Satisfy constraints to machine precision
3. ✅ Complete in reasonable time

---

## Research Questions

This project addresses:

1. **Can neuromorphic (oscillator-based) solvers achieve optimal control?**
   → YES (SL+DirectLag converges globally)

2. **How do they compare to traditional solvers?**
   → Slower (100ms vs 10ms) but more hardware-friendly

3. **Are they suitable for real-time control?**
   → For 10Hz systems: YES; for 100Hz+ systems: needs optimization

4. **Can they be efficiently implemented on neuromorphic hardware?**
   → Theoretically yes (parallel oscillators); practical demo pending

---

## Related Work

**Convex Optimization**: Boyd & Vandenberghe (2004)
**OSQP Solver**: Stellato et al. (2020)
**iLQR**: Tassa et al. (2012)
**Neuromorphic Computing**: Indiveri & Horiuchi (2011)

---

**See also**: [07-THEORY.md](../07-THEORY.md), [03-SOLVERS.md](../03-SOLVERS.md)
