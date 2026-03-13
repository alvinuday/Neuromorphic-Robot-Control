# Neuromorphic Robot Control: Technical Overview

This document provides a detailed explanation of the mathematics and implementation of the robot control system.

## 1. Robot Dynamics (Arm 2-DOF)

The robot is modeled as a 2-Degree-of-Freedom (2-DOF) manipulator in the vertical plane.

### Mathematical Model
The dynamics are governed by the standard manipulator equation:
$$M(\theta)\ddot{\theta} + C(\theta, \dot{\theta})\dot{\theta} + G(\theta) = \tau$$

Where:
- $\theta = [\theta_1, \theta_2]^T$ are the joint angles.
- $M(\theta)$ is the **Inertia Matrix**, representing the mass and moment of inertia of the links.
- $C(\theta, \dot{\theta})$ is the **Coriolis and Centrifugal Matrix**, representing velocity-dependent forces.
- $G(\theta)$ is the **Gravity Vector**, representing the gravitational torque on each joint.
- $\tau$ is the **Control Torque** applied by the motors.

### Implementation: `src/dynamics/arm2dof.py`
- Uses **CasADi** for symbolic computation.
- Computes $M, C, G$ matrices based on link lengths ($l_1, l_2$) and masses ($m_1, m_2$).
- Provides Jacobian functions $A = \frac{\partial \dot{x}}{\partial x}$ and $B = \frac{\partial \dot{x}}{\partial u}$ for linearization, where $x = [\theta, \dot{\theta}]^T$.

---

## 2. Model Predictive Control (MPC)

MPC solves an optimization problem at each time step to find the optimal control sequence.

### QP Formulation
The continuous dynamics $\dot{x} = f(x, u)$ are discretized and linearized:
$$x_{k+1} = A_k x_k + B_k u_k + c_k$$

The objective is to minimize a quadratic cost function:
$$J = \sum_{k=0}^{N-1} (x_k - x_{ref,k})^T Q (x_k - x_{ref,k}) + u_k^T R u_k + (x_N - x_{ref,N})^T Q_f (x_N - x_{ref,N})$$

This is converted into a standard **Quadratic Programming (QP)** form:
$$\min_z \frac{1}{2} z^T \mathcal{Q} z + p^T z$$
subject to:
- Equality constraints (Dynamics): $A_{eq} z = b_{eq}$
- Inequality constraints (Bounds): $A_{ineq} z \le k_{ineq}$

Where $z = [x_0, u_0, x_1, u_1, \dots, x_N]^T$ is the decision vector.

### Implementation: `src/mpc/qp_builder.py`
- `build_qp`: Constructs the large matrices $\mathcal{Q}, p, A_{eq}, b_{eq}, A_{ineq}, k_{ineq}$ by stacking local costs and constraints over the horizon $N$.
- `linearize`: Uses `A_fun` and `B_fun` from the arm model to compute $A_k$ and $B_k$ at each step of the reference trajectory.

---

## 3. Solvers

The system supports two types of solvers:

### OSQP (Classical)
- **Operator Splitting Quadratic Program**: A standard numerical solver for convex QPs.
- It uses the Alternating Direction Method of Multipliers (ADMM) to solve the optimization problem efficiently.
- **Implementation**: `src/solver/osqp_solver.py` acts as a wrapper for the `osqp` library.

### SHO & ALM (Neuromorphic)
This is a novel approach using **Spiking/Simulated Harmonic Oscillators**.

#### Augmented Lagrangian Method (ALM)
To handle constraints in a neuromorphic framework, we use ALM. The constrained problem is converted into a sequence of unconstrained problems:
$$\mathcal{L}(z, \lambda) = \frac{1}{2} z^T Q z + p^T z + \lambda^T(Az - b) + \frac{\rho}{2} \|Az - b\|^2$$
The dual variable $\lambda$ is updated iteratively: $\lambda \leftarrow \lambda + \rho(Az - b)$.

#### Oscillator Ising Machine (OIM)
1.  **Continuous to Binary**: The continuous variables $z$ are encoded using $n$ bits per variable. If $z \in \mathbb{R}^{d}$, we use $N = d \times n$ binary variables $s \in \{0, 1\}^N$. The mapping is:
    $$z = x_{min} + C s$$
    Where $C$ is a scaling matrix that maps bits to their respective continuous magnitudes (e.g., $2^{-1}, 2^{-2}, \dots$).

    Substituting this into the quadratic cost:
    $$f(s) = \frac{1}{2} s^T (C^T Q C) s + (x_{min}^T Q C + p^T C) s + \text{const}$$
    This is a **QUBO** problem: $\min \frac{1}{2} s^T Q_{qubo} s + p_{qubo}^T s$.

2.  **QUBO to Ising**: The binary variables $s \in \{0, 1\}$ are mapped to spins $\sigma \in \{-1, 1\}$ using $s = \frac{\sigma + 1}{2}$. This transforms the QUBO into an Ising model:
    $$H(\sigma) = -\sum_{i < j} J_{ij} \sigma_i \sigma_j - \sum_i h_i \sigma_i$$
    Where $J = -\frac{1}{4} Q_{qubo}$ and $h = -\frac{1}{2} (p_{qubo} + \sum Q_{qubo})$.

3.  **Dynamics**: The Ising model's ground state is found by simulating a network of phase-coupled oscillators (e.g., Kuramoto-like oscillators):
    $$\dot{\phi}_i = \omega_i + \sum_j J_{ij} \sin(\phi_j - \phi_i) + h_i$$
    As the system evolves, the oscillators synchronize or anti-synchronize. The final state is decoded by:
    $$\sigma_i = \text{sign}(\cos(\phi_i))$$

#### Augmented Lagrangian Method (ALM) Loop
The SHO solver wraps the OIM in an ALM loop to handle the constraints $Az = b$:
1.  Solve the Ising model for the current penalty problem.
2.  Update $z$ from decoded spins.
3.  Update multipliers: $\lambda \leftarrow \lambda + \rho(Az - b)$.
4.  Increase $\rho$ (optional) and repeat until convergence.

---

## 4. Codebase Structure

| Directory | Purpose | Key Files |
| :--- | :--- | :--- |
| `src/dynamics` | Robot physical models | `arm2dof.py` (Dynamics & Kinematics) |
| `src/mpc` | Optimization formulation | `qp_builder.py` (Linearization & QP construction) |
| `src/solver` | Optimization engines | `osqp_solver.py` (Classical), `sho_solver.py` (Neuromorphic) |
| `src/utils` | Helper functions | Hardware/Simulation interfaces |
| `main.py` | Entry point | Connects dynamics, MPC, and solver in a loop |
