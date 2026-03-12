# Theory & Mathematical Background

Deep dive into the mathematics and theory behind the solvers.

## Overview

This project solves **Quadratic Programming (QP)** problems using different solvers:

$$\min_x \frac{1}{2}x^T P x + q^T x$$
$$\text{subject to } Cx = d, \quad l \leq Ax \leq u$$

Where:
- **x**: Decision variables (n dimensions)
- **P**: Hessian matrix (n×n positive semidefinite)
- **q**: Linear term (n)
- **C, d**: Equality constraints (m_eq × n, m_eq)
- **A, l, u**: Inequality box constraints (m_ineq × n)

### Application: Model Predictive Control (MPC)

For a 2-DOF arm, we predict N steps ahead and optimize control inputs:

$$J = \sum_{k=0}^{N-1} ||x_k - x_{ref}||_Q^2 + ||u_k||_R^2 + ||x_N - x_{ref}||_{Q_f}^2$$

This becomes a QP with:
- Decision variables: u = [u₀, u₁, ..., u_{N-1}]
- Subject to: x_{k+1} = A x_k + B u_k (dynamics)

---

## Stuart-Landau + Direct Lagrange (SL+DirectLag)

### Concept

Model optimization as a continuous-time dynamical system:
- Decision variables x → oscillator state
- Constraints → Lagrange multiplier dynamics
- Convergence → solution found

### Equations

**Decision variable dynamics**:
$$\tau_x \frac{dx}{dt} = (\mu - x^2)x - \nabla_x L(x, \lambda)$$

Where L(x,λ) is the Lagrangian:
$$L(x, \lambda) = \frac{1}{2}x^T P x + q^T x - \lambda^T(Cx - d) - \mu^T(Ax - b)$$

**Lagrange multiplier dynamics**:
$$\tau_\lambda \frac{d\lambda}{dt} = Cx - d \quad \text{(equality)}$$
$$\tau_\mu \frac{d\mu}{dt} = \max(0, Ax - b) \quad \text{(inequality, one-sided)}$$

### Advantages

✅ **Hardware-friendly**: Natural parallelism across variables  
✅ **Constraint satisfaction**: λ dynamics directly enforce constraints  
✅ **Global convergence**: Arrow-Hurwicz saddle-point algorithm  
✅ **Machine precision**: Convergence to < 1e-7 accuracy  

### Disadvantages

❌ **Slower than OSQP**: ~50-100ms vs ~8ms  
❌ **Requires ODE integration**: More computation per iteration  
❌ **Parameter tuning**: τ_x, τ_λ, τ_μ affect convergence speed  

---

## OSQP (Operator Splitting QP)

### Concept

Use **Alternating Direction Method of Multipliers (ADMM)**:
- Split problem into simpler subproblems
- Solve each subproblem separately
- Coordinate via dual variables
- Repeat until convergence

### Algorithm

1. **x-update**: Solve (P + ρI) x = q + ρ(z - u)
2. **z-update**: Proximal operator on constraints
3. **u-update**: Dual variable scaling: u ← u + (Ax - z)
4. **Check convergence**: Are primal and dual residuals small?

### Advantages

✅ **Fast**: ~5-50ms depending on problem size  
✅ **Reliable**: Industry standard (used in robotics, finance)  
✅ **Optimal**: Guaranteed globally optimal solution  
✅ **Robust**: Handles many constraint types  

### Disadvantages

❌ **ADMM tuning**: ρ parameter affects convergence  
❌ **Requires library**: OSQP package dependency  
❌ **Sequential**: Hard to parallelize  

---

## iLQR (Iterative Linear Quadratic Regulator)

### Concept

Trajectory optimization via linearization:
1. Expand dynamics around nominal trajectory
2. Solve infinite-horizon LQR (Riccati equations)
3. Take gradient step on real nonlinear problem
4. Repeat until local convergence

### Algorithm

1. **Forward pass**: Rollout with current policy
2. **Backward pass**: Solve Riccati equations (optimal gain)
3. **Line search**: Find step size α that reduces cost
4. **Update**: x ← x + α · Δx

### Advantages

✅ **Fast**: ~10-20ms per solve  
✅ **Trajectory-focused**: Good for tracking control  
✅ **Local convergence**: Fast once near optimum  

### Disadvantages

❌ **Local optima**: Only finds nearby solution  
❌ **Sensitive initialization**: Needs good starting point  
❌ **Constraint handling**: Hard constraints difficult  

---

## KKT Optimality Conditions

A solution x* is optimal iff KKT conditions hold:

### 1. Stationarity
$$\nabla_x L = P x^* + q - C^T \lambda^* - A^T \mu^* = 0$$

### 2. Primal feasibility
$$Cx^* = d, \quad l \leq Ax^* \leq u$$

### 3. Dual feasibility
$$\mu^* \geq 0$$

### 4. Complementary slackness
$$\mu_i^* (Ax^* - u)_i = 0 \text{ and } \mu_i^* (l - Ax^*)_i = 0$$

**All 4 conditions → solution is optimal!**

---

## Comparison

### Speed Hierarchy
OSQP (fastest) > iLQR > Neuromorphic > PID (slowest but simplest)

### Optimality Hierarchy
Neuromorphic = OSQP (globally optimal) > iLQR (local) > PID (heuristic)

### Hardware Suitability
Neuromorphic (parallel) > iLQR (sequential) > OSQP (very sequential)

---

## Further Reading

- Boyd & Vandenberghe, "Convex Optimization" (theory)
- Stellato et al., "OSQP: An Operator Splitting Solver" (OSQP)
- Tassa et al., "Learning and Generalization in Biped Locomotion" (iLQR)
- Rosenbluth "The Use of Electronic Computers in Making Sound Movies" (oscillator theory)

---

## Next Steps

- **Practical examples**: [03-SOLVERS.md](03-SOLVERS.md)
- **Benchmarking**: [06-BENCHMARKING.md](06-BENCHMARKING.md)
- **API reference**: [REFERENCE/API_REFERENCE.md](REFERENCE/API_REFERENCE.md)

---

**Questions?** See [INDEX.md](INDEX.md) for other docs.
