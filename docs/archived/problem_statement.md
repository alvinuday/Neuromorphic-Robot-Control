# Problem Statement: Neuromorphic MPC for 2-DOF Robotic Arm

## 1. System Dynamics
We model a 2-DOF planar robotic arm with joint angles $q = [\theta_1, \theta_2]^T$. The dynamics are governed by the Euler-Lagrange equations:
$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + G(q) = \tau$$
where:
- $M(q)$ is the inertia matrix.
- $C(q, \dot{q})$ is the Coriolis and centrifugal matrix.
- $G(q)$ is the gravity vector.
- $\tau$ is the vector of joint torques (control inputs).

The state is defined as $x = [q^T, \dot{q}^T]^T \in \mathbb{R}^4$.

## 2. MPC Formulation
We solve a discrete-time optimal control problem over a horizon $N$:
$$\min \sum_{k=0}^{N-1} (\|x_k - x_{ref,k}\|^2_Q + \|u_k\|^2_R) + \|x_N - x_{ref,N}\|^2_{Q_f}$$
Subject to:
- $x_{k+1} = f(x_k, u_k)$ (Nonlinear dynamics)
- $u_{min} \leq u_k \leq u_{max}$
- $q_{min} \leq q_k \leq q_{max}$

At each step, we linearize the dynamics around a reference trajectory to obtain a **Quadratic Programming (QP)** problem of the form:
$$\min_z \frac{1}{2} z^T Q z + p^T z$$
$$\text{subject to } A_{eq} z = b_{eq}, \quad A_{ineq} z \leq k_{ineq}$$

## 3. Neuromorphic Solving via Oscillator Ising Machines
To solve the QP on neuromorphic hardware, we map it to the Ising model.

### 3.1 Augmented Lagrangian Method (ALM)
Constraints are handled using the Augmented Lagrangian:
$$\mathcal{L}(z, \lambda) = \frac{1}{2} z^T Q z + p^T z + \lambda^T(Az - b) + \frac{\rho}{2} \|Az - b\|^2$$
We iteratively solve for $z$ and update the dual variables $\lambda \leftarrow \lambda + \rho(Az - b)$.

### 3.2 Binary Encoding
Continuous variables $z$ are encoded using $n_{bits}$ binary variables $s \in \{0, 1\}$:
$$z \approx z_{min} + \text{Range} \cdot \sum_{j=0}^{n-1} 2^{-(j+1)} s_j$$

### 3.3 Ising Mapping
The resulting QUBO problem (Quadratic Unconstrained Binary Optimization) is mapped to an Ising Hamiltonian:
$$H = -\sum_{i,j} J_{ij} \sigma_i \sigma_j - \sum_i h_i \sigma_i$$
where $\sigma_i \in \{-1, +1\}$ represent the spins.

### 3.4 Oscillator Dynamics
We simulate the Ising machine using coupled phase oscillators (Kuramoto-like model):
$$\frac{d\phi_i}{dt} = \sum_j J_{ij} \sin(\phi_j - \phi_i) + h_i$$
The steady-state phases $\phi_i$ are then decoded back to spins ($\sigma_i = \text{sgn}(\cos \phi_i)$) and finally to the control inputs $u$.
