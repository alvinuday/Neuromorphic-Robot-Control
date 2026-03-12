# Data Format Reference

Specification of data formats used throughout the project.

## QP Problem Format

QP problems are represented as tuples:

```python
qp = (P, q, C, d, Ac, l_vec, u_vec)
```

### Components

| Variable | Shape | Type | Description |
|----------|-------|------|-------------|
| P | (n, n) | float64 | Hessian (positive semidefinite) |
| q | (n,) | float64 | Linear cost term |
| C | (m_eq, n) | float64 | Equality constraint matrix |
| d | (m_eq,) | float64 | Equality RHS |
| Ac | (m_ineq, n) | float64 | Inequality constraint matrix |
| l_vec | (m_ineq,) | float64 | Inequality lower bounds |
| u_vec | (m_ineq,) | float64 | Inequality upper bounds |

### Example

```python
import numpy as np

# Problem: min 0.5 x[0]² + x[1]² - 2*x[0] - 4*x[1]
#          s.t. x[0] + x[1] = 1
#               0 <= x[0], x[1] <= 10

n = 2
P = np.diag([1.0, 2.0])  # Hessian: diag(1, 2)
q = np.array([-2.0, -4.0])

m_eq = 1
C = np.array([[1.0, 1.0]])  # x₀ + x₁ = 1
d = np.array([1.0])

m_ineq = 2
Ac = np.eye(2)
l_vec = np.array([0.0, 0.0])  # x >= 0
u_vec = np.array([10.0, 10.0])  # x <= 10

qp = (P, q, C, d, Ac, l_vec, u_vec)
```

---

## Solver Output Format

```python
x_opt = solver.solve(qp)
info = solver.get_info()
```

### Solution

- **Type**: numpy.ndarray
- **Shape**: (n,) - the optimized decision variables
- **Example**: `array([0.5, 0.5])` - two decision variables

### Info Dictionary

```python
{
    'solve_time': 0.0082,           # Wall-clock seconds
    'objective': 1.2345,            # Objective value f(x*)
    'eq_violation': 1.3e-7,         # max(|Cx - d|)
    'ineq_violation': 2.1e-7,       # max(Acx - u, l - Acx)
    'iterations': 47,               # Number of iterations
    'status': 'solved',             # 'solved', 'timeout', 'diverged'
    'solver': 'OSQP'                # Solver name
}
```

---

## MPC State Format

### State Vector

```python
x = [q0, q1, dq0, dq1]  # Shape: (4,)
```

Where:
- `q0, q1`: Joint angles (radians)
- `dq0, dq1`: Joint velocities (rad/sec)

### Control Vector

```python
u = [tau0, tau1]  # Shape: (2,)  [Nm]
```

Torque commands, clipped to [-50, 50] Nm

### Target Position

```python
q_target = [q0_ref, q1_ref]  # Shape: (2,)
dq_target = [dq0_ref, dq1_ref]  # Shape: (2,), usually zeros
```

---

## Benchmark Result Format

### CSV Output

Benchmark results are saved as CSV:

```
solver,problem_size,solve_time,constraint_violation,optimality_gap,iterations
OSQP,2x2,0.0082,1.3e-7,0.0,47
OSQP,2x2,0.0081,1.2e-7,0.0,48
iLQR,2x2,0.0123,1.5e-5,0.008,8
Neuromorphic,2x2,0.0984,8.2e-7,0.0,746
```

### Python Dictionary

```python
{
    'solver': 'OSQP',
    'problem_size': '2x2',
    'solve_time': 0.0082,
    'constraint_violation': 1.3e-7,
    'optimality_gap': 0.0,
    'iterations': 47
}
```

---

## Trajectory Format

### Reference Trajectory

```python
q_ref = [q0(t), q1(t)]  # Position over time
dq_ref = [dq0(t), dq1(t)]  # Velocity over time
```

### Example: Circle in Joint Space

```python
# Parameters
center = np.array([np.pi/4, np.pi/4])
radius = 0.2
period = 10.0  # seconds

# Parametric form
theta = 2 * np.pi * t / period
q_ref = center + radius * np.array([np.cos(theta), np.sin(theta)])
dq_ref = radius * (2*np.pi/period) * np.array([-np.sin(theta), np.cos(theta)])
```

---

## File Storage Formats

### Numpy Archives (.npz)

Used to store QP instances:

```python
np.savez('qp_step_0000.npz',
    P=P, q=q, C=C, d=d, Ac=Ac, l_vec=l_vec, u_vec=u_vec
)

# Load
data = np.load('qp_step_0000.npz')
P = data['P']
q = data['q']
```

### Metadata (CSV)

```csv
step,problem_size,solver,solve_time,constraint_violation
0,2x2,OSQP,0.0082,1.3e-7
1,2x2,OSQP,0.0081,1.2e-7
2,2x2,iLQR,0.0123,1.5e-5
```

---

## XML Model Format (MuJoCo)

MuJoCo models are defined in XML:

```xml
<?xml version="1.0"?>
<mujoco model="arm2dof">
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  
  <worldbody>
    <body name="base">
      <joint name="q0" type="hinge" range="-3.14 3.14"/>
      <body name="link1">
        ...
      </body>
    </body>
  </worldbody>
  
  <actuator>
    <motor name="tau0" joint="q0" ctrlrange="-50 50"/>
    <motor name="tau1" joint="q1" ctrlrange="-50 50"/>
  </actuator>
</mujoco>
```

---

## Numerical Precision

### Typical Values

| Metric | Value |
|--------|-------|
| Constraint Violation | 1e-7 to 1e-6 |
| Optimality Gap | < 1% |
| Solve Time | 8ms to 100ms |
| State Magnitude | 0 to π radians |
| Control Magnitude | ±50 Nm |

### Machine Precision

- **Float64 epsilon**: ~2.22e-16
- **Practical tolerance**: 1e-6 (1000x epsilon, accounts for accumulation)
- **Constraint satisfaction**: < 1e-6 considered excellent

---

**See also**: [PROBLEM_STATEMENT.md](PROBLEM_STATEMENT.md), [API_REFERENCE.md](API_REFERENCE.md)
