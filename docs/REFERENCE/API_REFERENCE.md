# API Reference

Complete API documentation for the solver and controller classes.

## Solver Classes

### StuartLandauLagrangeDirect

```python
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect

solver = StuartLandauLagrangeDirect(
    tau_x=1.0,              # Decision variable time constant
    tau_lam_eq=0.1,         # Equality multiplier time constant
    tau_lam_ineq=0.5,       # Inequality multiplier time constant
    mu_x=0.0,               # Stuart-Landau bifurcation parameter
    T_solve=30.0,           # Total solve time (seconds)
    convergence_tol=1e-6,   # Convergence threshold
    adaptive_annealing=True # Use time-varying tau
)
```

**Methods**:

```python
x_opt = solver.solve(qp_matrices, verbose=False)
```

**Input**: 
- `qp_matrices`: tuple `(P, q, C, d, Ac, l_vec, u_vec)`
  - P: Hessian (n×n)
  - q: Linear term (n)
  - C: Equality constraint matrix (m_eq×n)
  - d: Equality RHS (m_eq)
  - Ac: Inequality constraint matrix (m_ineq×n)
  - l_vec: Inequality lower bounds (m_ineq)
  - u_vec: Inequality upper bounds (m_ineq)

**Output**:
- `x_opt`: Optimal solution (n)

**Example**:
```python
import numpy as np

P = np.eye(4)
q = np.array([1, 2, -1, -2])
C = np.array([[1, 0, 1, 0]])
d = np.array([1.0])
Ac = np.eye(4)
l = np.zeros(4)
u = np.array([10, 10, 10, 10])

x = solver.solve((P, q, C, d, Ac, l, u))
print(f"Solution: {x}")
```

---

### Phase4MPCController

```python
from src.solver.phase4_mpc_controller import Phase4MPCController

mpc = Phase4MPCController(
    N=10,           # Prediction horizon (steps)
    dt=0.002,       # Time step (seconds)
    Qx=None,        # State cost (default: I_4)
    Qf=None,        # Terminal cost (default: 2*I_4)
    R=None,         # Control cost (default: 0.1*I_2)
    tau_min=-50.0,  # Min control
    tau_max=50.0,   # Max control
    theta_min=-np.pi,  # Min angle
    theta_max=np.pi    # Max angle
)
```

**Methods**:

```python
u_opt, info = mpc.solve_step(x_current, x_target)
```

**Input**:
- `x_current`: Current state [q1, q2, dq1, dq2] (4)
- `x_target`: Target state [q1_ref, q2_ref, dq1_ref, dq2_ref] (4)

**Output**:
- `u_opt`: Optimal control [tau1, tau2] (2)  
- `info`: Dictionary with keys:
  - `'solve_time'`: Wall-clock time (seconds)
  - `'constraint_eq_violation'`: Equality constraint error
  - `'constraint_ineq_violation'`: Inequality constraint error
  - `'objective'`: Cost value

**Example**:
```python
x = np.array([0.0, 0.0, 0.0, 0.0])  # At rest, origin
x_ref = np.array([np.pi/6, np.pi/6, 0, 0])  # Target 30° each joint

u, info = mpc.solve_step(x, x_ref)
print(f"Control: {u}")
print(f"Solve time: {info['solve_time']:.4f}s")
```

---

## Benchmark Framework

### create_solver Factory

```python
from src.benchmark.benchmark_solvers import create_solver

# Create any solver
osqp_solver = create_solver('osqp')
ilqr_solver = create_solver('ilqr')
neuro_solver = create_solver('neuromorphic')
```

### Generic Solver Interface

All solvers inherit from `QPSolver` and provide:

```python
x_opt = solver.solve(P, q, C, d, Ac, l, u)
info = solver.get_info()
```

**Returns**:
- `x_opt`: Solution vector
- `info`: Dictionary with:
  - `'solve_time'`: Wall-clock time
  - `'objective'`: Objective value
  - `'eq_violation'`: Equality constraint error
  - `'ineq_violation'`: Inequality constraint error
  - `'solver'`: Solver name string

---

## MuJoCo Integration

### InteractiveArmController

```python
from src.mujoco.mujoco_interactive_controller import InteractiveArmController

controller = InteractiveArmController(
    model_path='/path/to/arm2dof.xml',
    task='reach',           # 'reach', 'circle', 'square'
    controller_type='osqp'  # 'pid', 'osqp', 'ilqr', 'neuromorphic'
)

controller.run()  # Launch interactive viewer
```

**Tasks**:
- `'reach'`: Move to π/6, π/6 (5 seconds)
- `'circle'`: Trace circle in joint space (continuous)
- `'square'`: Visit 4 corners (10 seconds)

---

## Trajectory Generators

```python
from src.mujoco.mujoco_interactive_controller import TrajectoryGenerator

# Get trajectory function and duration
traj_fn, duration = TrajectoryGenerator.reach_trajectory(
    duration=5.0,
    target=np.array([np.pi/6, np.pi/6])
)

# Use trajectory
q_ref, dq_ref = traj_fn(t=0.5)  # At t=0.5 seconds
```

**Available**:
- `reach_trajectory(duration, target)`
- `circle_trajectory(duration, center, radius)`
- `square_trajectory(duration, corner_duration)`

---

## Testing Utilities

### Running Specific Tests

```bash
# One test
pytest tests/test_lagrange_direct.py::test_solve_simple_qp -v

# All in file
pytest tests/test_lagrange_direct.py -v

# Pattern matching
pytest tests/ -k "benchmark" -v

# Show output
pytest tests/test_benchmark_suite.py -v -s
```

---

## Common Patterns

### Simple QP Solve

```python
import numpy as np
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect

# Problem: min 0.5 x^T x + x subject to x >= 0
P = np.eye(5)
q = np.ones(5)
C = np.zeros((0, 5))  # No equality constraints
d = np.zeros(0)
Ac = np.eye(5)  # x >= 0
l = np.zeros(5)
u = np.full(5, 100)

solver = StuartLandauLagrangeDirect()
x = solver.solve((P, q, C, d, Ac, l, u))
print(f"Solution: {x}")  # Should be zeros (x >= 0, minimizes 0.5x²+x)
```

### MPC Control Loop

```python
import numpy as np
import mujoco
from src.solver.phase4_mpc_controller import Phase4MPCController

model = mujoco.MjModel.from_xml_path('assets/arm2dof.xml')
data = mujoco.MjData(model)
mpc = Phase4MPCController(N=10, dt=0.002)

target_pos = np.array([np.pi/6, np.pi/6])

for step in range(1000):
    x = np.concatenate([data.qpos, data.qvel])
    
    u, info = mpc.solve_step(x, target_pos)
    data.ctrl[:] = np.clip(u, -50, 50)
    
    mujoco.mj_step(model, data)
    
    if step % 100 == 0:
        error = np.linalg.norm(target_pos - data.qpos)
        print(f"Step {step}: error={error:.4f}")
```

---

## Error Handling

**Common exceptions**:

```python
# Infeasible QP
try:
    x = solver.solve(...)
except ValueError as e:
    print(f"Solver error: {e}")

# File not found
from pathlib import Path
if not Path('assets/arm2dof.xml').exists():
    raise FileNotFoundError("Model not found")

# Module import
try:
    import osqp
except ImportError:
    print("Install with: pip install osqp")
```

---

**See also**: [INDEX.md](../INDEX.md), [07-THEORY.md](../07-THEORY.md)
