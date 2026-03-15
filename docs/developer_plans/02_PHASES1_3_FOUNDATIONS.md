# PHASES 1–3 — FOUNDATIONS: ABSTRACTIONS, SOLVERS, MUJOCO
**Requires**: Phase 0 complete, AUDIT_REPORT.md written.

---

## PHASE 1 — CORE ABSTRACTIONS

**Goal**: Establish the modular base. Every component in this project inherits from these.

### 1.1 — `src/core/base_solver.py`

```python
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict

class BaseQPSolver(ABC):
    """
    Abstract base for QP solvers. Standard OSQP-style interface.

    Solves:
        minimize    0.5 * x^T P x + q^T x
        subject to  l <= A x <= u

    Equality constraints are encoded as l[i] == u[i].
    """

    @abstractmethod
    def solve(
        self,
        P: np.ndarray,   # [n, n] positive semi-definite cost matrix
        q: np.ndarray,   # [n]    linear cost vector
        A: np.ndarray,   # [m, n] constraint matrix
        l: np.ndarray,   # [m]    lower bounds  (-inf for one-sided)
        u: np.ndarray,   # [m]    upper bounds  (+inf for one-sided)
    ) -> Tuple[np.ndarray, Dict]:
        """
        Returns:
            x:    [n] optimal primal solution
            info: dict with keys:
                    solve_time_ms     (float)
                    obj_val           (float)  — 0.5 x^T P x + q^T x
                    constraint_viol   (float)  — max(0, Ax-u, l-Ax).max()
                    status            (str)    — "optimal"|"max_iter"|"error"
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable solver name, e.g. 'StuartLandauLagrange' or 'OSQP'."""
        pass
```

### 1.2 — `src/core/base_controller.py`

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseController(ABC):
    @abstractmethod
    def step(self, state: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Compute control output. MUST be synchronous. MUST return ndarray."""
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
```

### 1.3 — `src/core/base_env.py`

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseEnv(ABC):
    @abstractmethod
    def reset(self) -> dict:
        """Reset to initial state. Returns observation dict."""
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple:
        """Apply action. Returns (obs, reward, done, info)."""
        pass

    @abstractmethod
    def close(self) -> None:
        pass
```

### 1.4 — `config/robots/xarm_6dof.yaml` (canonical)

```yaml
robot:
  name: "xarm_6dof"
  n_joints: 6
  n_gripper: 2
  n_total_dof: 8          # 6 arm + 2 gripper

  joint_limits:
    q_min: [-6.283, -3.665, -6.109, -4.555, -6.109, -6.283]   # rad
    q_max: [ 6.283,  3.665,  6.109,  4.555,  6.109,  6.283]

  velocity_limits:
    qdot_max: [3.0, 2.5, 2.5, 2.0, 1.5, 2.0]   # rad/s

  torque_limits:
    tau_max: [20.0, 15.0, 15.0, 10.0, 8.0, 6.0, 5.0, 5.0]    # Nm (6 arm + 2 gripper)

  dynamics:
    link_masses:   [1.2, 0.8, 0.6, 0.5, 0.4, 0.3]   # kg
    link_lengths:  [0.267, 0.289, 0.078, 0.346, 0.076, 0.097]  # m
    gravity: 9.81

  mjcf_path: "assets/xarm_6dof.xml"
```

### 1.5 — Tests for Phase 1

`tests/unit/test_abstractions.py`:
```python
def test_base_solver_import():
    from src.core.base_solver import BaseQPSolver

def test_base_controller_import():
    from src.core.base_controller import BaseController

def test_base_env_import():
    from src.core.base_env import BaseEnv

def test_xarm_config_loads():
    import yaml
    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)
    assert cfg['robot']['n_joints'] == 6
    assert len(cfg['robot']['joint_limits']['q_min']) == 6
    assert len(cfg['robot']['torque_limits']['tau_max']) == 8
```

**Phase 1 Gate**: `pytest tests/unit/test_abstractions.py -v` — ALL PASS

---

## PHASE 2 — SOLVERS (SL + OSQP)

**Goal**: Both solvers must solve a real QP and return numerically correct answers.
This phase validates the **core thesis contribution** (SL solver).

### 2.1 — Validation QP (used in all solver tests)

```
minimize    x₁² + x₂²           (P = 2I, q = 0)
subject to  x₁ + x₂ = 1         (equality, encoded as l=u=1)
            -5 ≤ x₁ ≤ 5         (box)
            -5 ≤ x₂ ≤ 5

Analytic solution: x₁ = x₂ = 0.5,  obj = 0.5
```

OSQP-style matrices:
```python
import numpy as np
P = np.array([[2., 0.], [0., 2.]])
q = np.array([0., 0.])
A = np.array([
    [1., 1.],   # row 0: equality x1+x2=1 (l=u=1)
    [1., 0.],   # row 1: x1 box
    [0., 1.],   # row 2: x2 box
])
l = np.array([ 1., -5., -5.])
u = np.array([ 1.,  5.,  5.])
```

### 2.2 — `src/solver/stuart_landau_lagrange_direct.py`

**Mathematical form** (Arrow-Hurwicz saddle-point on the SL-augmented Lagrangian):

```
dx/dt     = (μ - |x|²)x/τ_x - (Px + q)/τ_x - Aᵀ(λ_up - λ_lo)/τ_x
dλ_up/dt  = max(0, Ax - u) / τ_λ
dλ_lo/dt  = max(0, l - Ax) / τ_λ
```

Equality constraints encoded by having `l[i] == u[i]`, so both λ_up and λ_lo are active.

**Implementation skeleton** (fill in details):

```python
from scipy.integrate import solve_ivp
import numpy as np
import time
from src.core.base_solver import BaseQPSolver

class StuartLandauLagrangeDirect(BaseQPSolver):
    """
    Continuous-time QP solver via Stuart-Landau oscillator dynamics
    and Arrow-Hurwicz saddle-point algorithm.
    
    Neuromorphic motivation: the ODE mimics analog neuronal dynamics —
    no matrix inversions, no line searches, purely differential equations.
    
    EXPECTED TIMING: 2000–8000ms wall clock for n=6 QPs.
    This is CORRECT behavior. Do not optimize for speed.
    """
    
    def __init__(
        self,
        tau_x: float = 1.0,      # primal time constant
        tau_lam: float = 0.2,    # dual (multiplier) time constant
        mu: float = 0.1,         # SL bifurcation parameter
        T_solve: float = 3.0,    # solver time horizon (seconds of ODE time)
        rtol: float = 1e-4,
        atol: float = 1e-6,
    ):
        self.tau_x = tau_x
        self.tau_lam = tau_lam
        self.mu = mu
        self.T_solve = T_solve
        self.rtol = rtol
        self.atol = atol

    @property
    def name(self) -> str:
        return "StuartLandauLagrange"

    def _ode_rhs(self, t, y, P, q, A, l, u):
        n = P.shape[0]
        m = A.shape[0]
        x      = y[:n]
        lam_up = y[n:n+m]
        lam_lo = y[n+m:]

        Ax = A @ x

        # Stuart-Landau restoring term + gradient of quadratic cost
        sl_term = (self.mu - float(x @ x)) * x
        grad_f  = P @ x + q
        dual_force = A.T @ (lam_up - lam_lo)

        dx      = (sl_term - grad_f - dual_force) / self.tau_x
        dlam_up = np.maximum(0.0, Ax - u) / self.tau_lam
        dlam_lo = np.maximum(0.0, l - Ax) / self.tau_lam

        return np.concatenate([dx, dlam_up, dlam_lo])

    def solve(self, P, q, A, l, u) -> tuple:
        n, m = P.shape[0], A.shape[0]
        y0 = np.zeros(n + 2*m)
        
        t_wall_start = time.perf_counter()
        result = solve_ivp(
            fun=self._ode_rhs,
            t_span=(0.0, self.T_solve),
            y0=y0,
            args=(P, q, A, l, u),
            method='RK45',
            rtol=self.rtol,
            atol=self.atol,
            max_step=0.1,
            dense_output=False,
        )
        wall_ms = (time.perf_counter() - t_wall_start) * 1000.0

        x = result.y[:n, -1]
        Ax = A @ x
        viol = float(np.maximum(0, Ax - u).max() + np.maximum(0, l - Ax).max())
        obj  = float(0.5 * x @ P @ x + q @ x)

        status = "optimal" if result.success else "max_iter"
        info = {
            'solve_time_ms':   wall_ms,
            'obj_val':         obj,
            'constraint_viol': viol,
            'status':          status,
            'ode_nfev':        result.nfev,
        }
        return x, info
```

### 2.3 — `src/solver/osqp_solver.py`

```python
import osqp
import scipy.sparse as sp
import numpy as np
import time
from src.core.base_solver import BaseQPSolver

class OSQPSolver(BaseQPSolver):
    """OSQP wrapper. Baseline QP solver. Fast (~5–50ms for n=6)."""
    
    def __init__(self, eps_abs=1e-4, eps_rel=1e-4, max_iter=10000, verbose=False):
        self.settings = dict(eps_abs=eps_abs, eps_rel=eps_rel,
                             max_iter=max_iter, verbose=verbose)

    @property
    def name(self) -> str:
        return "OSQP"

    def solve(self, P, q, A, l, u) -> tuple:
        P_sp = sp.csc_matrix(P)
        A_sp = sp.csc_matrix(A)
        
        prob = osqp.OSQP()
        prob.setup(P_sp, q, A_sp, l, u, **self.settings)
        
        t0 = time.perf_counter()
        res = prob.solve()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        
        x = res.x if res.x is not None else np.zeros(P.shape[0])
        Ax = A @ x
        viol = float(np.maximum(0, Ax - u).max() + np.maximum(0, l - Ax).max())
        obj  = float(0.5 * x @ P @ x + q @ x)
        
        status_map = {osqp.constant('OSQP_SOLVED'): 'optimal'}
        status = status_map.get(res.info.status_val, 'max_iter')
        
        return x, {
            'solve_time_ms':   wall_ms,
            'obj_val':         obj,
            'constraint_viol': viol,
            'status':          status,
        }
```

### 2.4 — Tests for Phase 2

`tests/unit/test_sl_solver.py` and `tests/unit/test_osqp_solver.py` must share the same QP suite:

```python
# tests/conftest.py  (shared fixtures)
import numpy as np, pytest

@pytest.fixture
def validation_qp():
    """x1+x2=1, minimize x1^2+x2^2. Solution: [0.5, 0.5]."""
    P = np.array([[2.,0.],[0.,2.]])
    q = np.zeros(2)
    A = np.array([[1.,1.],[1.,0.],[0.,1.]])
    l = np.array([1.,-5.,-5.])
    u = np.array([1., 5., 5.])
    return P, q, A, l, u

@pytest.fixture
def box_qp():
    """minimize (x-10)^2 s.t. -1<=x<=1. Solution: x=1."""
    P = np.array([[2.]]); q = np.array([-20.])
    A = np.array([[1.]]); l = np.array([-1.]); u = np.array([1.])
    return P, q, A, l, u
```

```python
# tests/unit/test_sl_solver.py
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
import numpy as np, pytest

@pytest.mark.slow
def test_validation_qp(validation_qp):
    P, q, A, l, u = validation_qp
    solver = StuartLandauLagrangeDirect(T_solve=3.0)
    x, info = solver.solve(P, q, A, l, u)
    assert abs(x[0] - 0.5) < 0.02, f"x[0]={x[0]:.4f} expected 0.5"
    assert abs(x[1] - 0.5) < 0.02, f"x[1]={x[1]:.4f} expected 0.5"
    assert info['constraint_viol'] < 0.05
    assert info['status'] in ('optimal', 'max_iter')
    print(f"\nSL: x={x}, time={info['solve_time_ms']:.0f}ms, status={info['status']}")

def test_box_qp(box_qp):
    P, q, A, l, u = box_qp
    solver = StuartLandauLagrangeDirect(T_solve=2.0)
    x, info = solver.solve(P, q, A, l, u)
    assert abs(x[0] - 1.0) < 0.05, f"x={x[0]:.4f} expected 1.0"

def test_info_dict_keys(validation_qp):
    P, q, A, l, u = validation_qp
    solver = StuartLandauLagrangeDirect(T_solve=1.0)
    _, info = solver.solve(P, q, A, l, u)
    for key in ('solve_time_ms', 'obj_val', 'constraint_viol', 'status'):
        assert key in info, f"Missing key: {key}"

def test_different_inputs_different_outputs():
    """Solver must not return same x for different q vectors."""
    solver = StuartLandauLagrangeDirect(T_solve=1.5)
    P = np.eye(2) * 2
    A = np.eye(2); l = np.array([-5.,-5.]); u = np.array([5.,5.])
    x1, _ = solver.solve(P, np.array([0., 0.]), A, l, u)
    x2, _ = solver.solve(P, np.array([2., 0.]), A, l, u)
    assert not np.allclose(x1, x2), "Different inputs must yield different outputs"
```

```python
# tests/unit/test_osqp_solver.py  (same tests, tighter tolerances)
from src.solver.osqp_solver import OSQPSolver
import numpy as np, pytest

def test_validation_qp(validation_qp):
    P, q, A, l, u = validation_qp
    x, info = OSQPSolver().solve(P, q, A, l, u)
    assert abs(x[0] - 0.5) < 0.001
    assert abs(x[1] - 0.5) < 0.001
    assert info['status'] == 'optimal'
    assert info['solve_time_ms'] < 200

def test_solve_time_fast(validation_qp):
    P, q, A, l, u = validation_qp
    _, info = OSQPSolver().solve(P, q, A, l, u)
    assert info['solve_time_ms'] < 200, f"OSQP too slow: {info['solve_time_ms']:.1f}ms"
```

**Phase 2 Gate**: `pytest tests/unit/test_sl_solver.py tests/unit/test_osqp_solver.py -v`

Note: SL solver tests are marked `@pytest.mark.slow`. Run them with `pytest -m slow` separately if CI is time-constrained.

---

## PHASE 3 — MUJOCO ENVIRONMENT

**Goal**: `XArmEnv` must load real physics and show joint motion when torque is applied.

### 3.1 — `assets/xarm_6dof.xml` (canonical MJCF)

The XML must load in MuJoCo 3.x. Key structural requirements:

```xml
<mujoco model="xarm_6dof">
  <compiler angle="radian" meshdir="meshes/"/>
  <option gravity="0 0 -9.81" timestep="0.001" integrator="RK4"/>

  <worldbody>
    <!-- Static base -->
    <body name="link0" pos="0 0 0">
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.01 0.01 0.01"/>
      <geom type="cylinder" size="0.05 0.05" rgba="0.5 0.5 0.5 1"/>

      <!-- Joint 1: shoulder rotation (z-axis) -->
      <body name="link1" pos="0 0 0.267">
        <joint name="joint1" type="hinge" axis="0 0 1"
               range="-6.283 6.283" damping="0.5"/>
        <inertial mass="1.2" pos="0 0 0.1" diaginertia="0.005 0.005 0.002"/>
        <geom type="capsule" size="0.04 0.13" rgba="0.3 0.5 0.8 1"/>

        <!-- Joint 2: shoulder pitch (y-axis) -->
        <body name="link2" pos="0 0 0.289">
          <joint name="joint2" type="hinge" axis="0 1 0"
                 range="-3.665 3.665" damping="0.5"/>
          <inertial mass="0.8" pos="0 0 0.12" diaginertia="0.003 0.003 0.001"/>
          <geom type="capsule" size="0.035 0.13" rgba="0.3 0.5 0.8 1"/>

          <body name="link3" pos="0 0 0.078">
            <joint name="joint3" type="hinge" axis="0 1 0"
                   range="-6.109 6.109" damping="0.3"/>
            <inertial mass="0.6" pos="0 0 0.15" diaginertia="0.002 0.002 0.001"/>
            <geom type="capsule" size="0.03 0.16" rgba="0.4 0.6 0.9 1"/>

            <body name="link4" pos="0 0 0.346">
              <joint name="joint4" type="hinge" axis="1 0 0"
                     range="-4.555 4.555" damping="0.2"/>
              <inertial mass="0.5" pos="0 0 0.03" diaginertia="0.001 0.001 0.0005"/>
              <geom type="capsule" size="0.025 0.04" rgba="0.5 0.5 0.7 1"/>

              <body name="link5" pos="0 0 0.076">
                <joint name="joint5" type="hinge" axis="0 1 0"
                       range="-6.109 6.109" damping="0.2"/>
                <inertial mass="0.4" pos="0 0 0.04" diaginertia="0.0008 0.0008 0.0003"/>
                <geom type="capsule" size="0.02 0.04" rgba="0.5 0.5 0.7 1"/>

                <body name="link6" pos="0 0 0.097">
                  <joint name="joint6" type="hinge" axis="0 0 1"
                         range="-6.283 6.283" damping="0.1"/>
                  <inertial mass="0.3" pos="0 0 0.02" diaginertia="0.0005 0.0005 0.0002"/>
                  <geom type="cylinder" size="0.02 0.02" rgba="0.6 0.4 0.3 1"/>

                  <!-- Gripper base -->
                  <body name="gripper_base" pos="0 0 0.05">
                    <geom type="box" size="0.03 0.02 0.01" rgba="0.3 0.3 0.3 1"/>
                    <!-- Left finger -->
                    <body name="finger_left" pos="-0.015 0 0.02">
                      <joint name="finger_joint1" type="slide" axis="1 0 0"
                             range="0 0.04" damping="10.0"/>
                      <inertial mass="0.05" pos="0 0 0.02" diaginertia="0.00001 0.00001 0.00001"/>
                      <geom type="box" size="0.005 0.008 0.025" rgba="0.2 0.2 0.2 1"/>
                    </body>
                    <!-- Right finger -->
                    <body name="finger_right" pos="0.015 0 0.02">
                      <joint name="finger_joint2" type="slide" axis="-1 0 0"
                             range="0 0.04" damping="10.0"/>
                      <inertial mass="0.05" pos="0 0 0.02" diaginertia="0.00001 0.00001 0.00001"/>
                      <geom type="box" size="0.005 0.008 0.025" rgba="0.2 0.2 0.2 1"/>
                    </body>
                    <!-- End-effector site -->
                    <site name="ee_site" pos="0 0 0.06" size="0.01"/>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Table -->
    <body name="table" pos="0.5 0 0.4">
      <geom type="box" size="0.25 0.25 0.02" rgba="0.7 0.5 0.3 1"
            contype="1" conaffinity="1"/>
    </body>

    <!-- Graspable object (red block) -->
    <body name="red_block" pos="0.5 0 0.44">
      <freejoint name="block_freejoint"/>
      <inertial mass="0.05" pos="0 0 0" diaginertia="0.0001 0.0001 0.0001"/>
      <geom type="box" size="0.02 0.02 0.02" rgba="0.9 0.1 0.1 1"
            contype="1" conaffinity="1"/>
    </body>

    <!-- Fixed camera: front view -->
    <camera name="camera_front" pos="1.2 0.0 0.8" xyaxes="-1 0 0 0 -0.5 1"/>
    <!-- Fixed camera: side view -->
    <camera name="camera_side"  pos="0.0 1.2 0.8" xyaxes="0 -1 0 0.5 0 1"/>
    <!-- Wrist camera (on EE) -->
    <camera name="camera_wrist" pos="0 0 0.06" mode="fixed"/>
  </worldbody>

  <actuator>
    <motor name="act1" joint="joint1"       ctrlrange="-20 20" gear="1"/>
    <motor name="act2" joint="joint2"       ctrlrange="-15 15" gear="1"/>
    <motor name="act3" joint="joint3"       ctrlrange="-15 15" gear="1"/>
    <motor name="act4" joint="joint4"       ctrlrange="-10 10" gear="1"/>
    <motor name="act5" joint="joint5"       ctrlrange="-8 8"   gear="1"/>
    <motor name="act6" joint="joint6"       ctrlrange="-6 6"   gear="1"/>
    <motor name="act7" joint="finger_joint1" ctrlrange="0 5"   gear="1"/>
    <motor name="act8" joint="finger_joint2" ctrlrange="0 5"   gear="1"/>
  </actuator>
</mujoco>
```

### 3.2 — `src/simulation/envs/xarm_env.py`

```python
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')  # headless default

import mujoco
import numpy as np
from src.core.base_env import BaseEnv

class XArmEnv(BaseEnv):
    """
    MuJoCo-backed xArm 6-DOF + gripper environment.
    
    Observation:
        q:    [6] arm joint angles (rad)
        qdot: [6] arm joint velocities (rad/s)
        rgb:  [H, W, 3] uint8 from camera_front
    
    Action:
        tau: [8] torques (6 arm + 2 gripper fingers), Nm
    """
    
    H, W = 240, 320   # render resolution
    
    def __init__(self, model_path: str = "assets/xarm_6dof.xml",
                 render_mode: str = "offscreen",
                 max_steps: int = 500,
                 camera: str = "camera_front"):
        self.model_path = model_path
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.camera_name = camera

        # Load model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data  = mujoco.MjData(self.model)

        # Renderer for offscreen RGB
        self.renderer = mujoco.Renderer(self.model, height=self.H, width=self.W)

        # Torque limits
        self.tau_min = -self.model.actuator_ctrlrange[:, 0]  # negative lower
        self.tau_max =  self.model.actuator_ctrlrange[:, 1]

        self._step_count = 0

    def reset(self) -> dict:
        mujoco.mj_resetData(self.model, self.data)
        self._step_count = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, tau: np.ndarray) -> tuple:
        assert tau.shape == (8,), f"Expected tau shape (8,), got {tau.shape}"
        tau_clamped = np.clip(tau, -self.model.actuator_ctrlrange[:, 1],
                                    self.model.actuator_ctrlrange[:, 1])
        self.data.ctrl[:] = tau_clamped
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        obs = self._get_obs()
        done = self._step_count >= self.max_steps
        return obs, 0.0, done, {'step': self._step_count}

    def _get_obs(self) -> dict:
        q    = self.data.qpos[:6].copy()
        qdot = self.data.qvel[:6].copy()
        rgb  = self._render_rgb()
        return {'q': q, 'qdot': qdot, 'rgb': rgb}

    def _render_rgb(self) -> np.ndarray:
        self.renderer.update_scene(self.data, camera=self.camera_name)
        return self.renderer.render().copy()   # [H, W, 3] uint8

    def get_ee_pose(self) -> tuple:
        """Return (position [3], quat [4]) of ee_site."""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
        pos  = self.data.site_xpos[site_id].copy()
        mat  = self.data.site_xmat[site_id].reshape(3, 3)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, mat.flatten())
        return pos, quat

    def close(self):
        self.renderer.close()
```

### 3.3 — `src/simulation/cameras/event_camera.py`

```python
import numpy as np

class EventCameraSimulator:
    """Simulate event camera via frame-difference thresholding."""
    def __init__(self, threshold=0.15, n_bins=5):
        self.threshold = threshold
        self.n_bins = n_bins

    def process_frame(self, frame: np.ndarray, prev_frame: np.ndarray) -> np.ndarray:
        """Returns [H, W] float32 event map (+-polarity)."""
        curr = frame.mean(axis=2).astype(np.float32) / 255.
        prev = prev_frame.mean(axis=2).astype(np.float32) / 255.
        diff = curr - prev
        events = np.where(np.abs(diff) > self.threshold, np.sign(diff), 0.).astype(np.float32)
        return events

    def extract_features(self, events: np.ndarray) -> np.ndarray:
        """Returns 96-dim feature vector from event map."""
        H, W = events.shape
        # 5x5 spatial grid statistics (25 cells × 2 stats = 50-dim)
        cells = []
        for i in range(5):
            for j in range(5):
                r0, r1 = i*H//5, (i+1)*H//5
                c0, c1 = j*W//5, (j+1)*W//5
                patch = events[r0:r1, c0:c1]
                cells.extend([patch.mean(), patch.std()])
        spatial = np.array(cells, dtype=np.float32)  # [50]
        # Magnitude histogram (46 bins)
        hist, _ = np.histogram(events.ravel(), bins=46, range=(-1.0, 1.0))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
        return np.concatenate([spatial, hist])   # [96]

class LiDARSimulator:
    """Simulate LiDAR from RGB gradient analysis."""
    def extract_features(self, rgb: np.ndarray) -> np.ndarray:
        """Returns 64-dim feature vector."""
        gray = rgb.mean(axis=2).astype(np.float32) / 255.
        gx = np.abs(np.diff(gray, axis=1))   # [H, W-1]
        gy = np.abs(np.diff(gray, axis=0))   # [H-1, W]
        # Depth proxy: gradient magnitude histogram (32 bins each axis)
        hx, _ = np.histogram(gx.ravel(), bins=32, range=(0, 0.5))
        hy, _ = np.histogram(gy.ravel(), bins=32, range=(0, 0.5))
        hx = hx.astype(np.float32) / (hx.sum() + 1e-6)
        hy = hy.astype(np.float32) / (hy.sum() + 1e-6)
        return np.concatenate([hx, hy])   # [64]
```

### 3.4 — Tests for Phase 3

`tests/integration/test_xarm_env.py`:

```python
import os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, pytest
from src.simulation.envs.xarm_env import XArmEnv

@pytest.fixture(scope='module')
def env():
    e = XArmEnv(render_mode='offscreen')
    yield e
    e.close()

def test_model_loads(env):
    assert env.model is not None

def test_reset_shapes(env):
    obs = env.reset()
    assert obs['q'].shape == (6,)
    assert obs['qdot'].shape == (6,)
    assert obs['rgb'].ndim == 3 and obs['rgb'].shape[2] == 3

def test_rgb_not_all_black(env):
    obs = env.reset()
    assert obs['rgb'].sum() > 0, "RGB is all black — rendering broken"

def test_step_changes_joint_state(env):
    obs0 = env.reset()
    q0 = obs0['q'].copy()
    tau = np.array([5., 0., 0., 0., 0., 0., 0., 0.])
    for _ in range(30):
        obs, _, _, _ = env.step(tau)
    assert abs(obs['q'][0] - q0[0]) > 0.001, "Arm didn't move with applied torque"

def test_torque_clamping_no_crash(env):
    env.reset()
    tau = np.ones(8) * 9999   # Way beyond limits
    obs, r, done, info = env.step(tau)
    assert obs is not None    # No crash

def test_100_steps_complete(env):
    env.reset()
    tau = np.zeros(8)
    for _ in range(100):
        obs, _, done, _ = env.step(tau)
    assert obs is not None

def test_done_after_max_steps():
    e = XArmEnv(render_mode='offscreen', max_steps=5)
    e.reset()
    tau = np.zeros(8)
    done = False
    for _ in range(6):
        _, _, done, _ = e.step(tau)
    assert done
    e.close()

def test_ee_pose_shape(env):
    env.reset()
    pos, quat = env.get_ee_pose()
    assert pos.shape == (3,)
    assert quat.shape == (4,)
```

**Phase 3 Gate**: `pytest tests/integration/test_xarm_env.py -v` — ALL PASS

*Next: Read `03_PHASES4_6_MPC_FUSION_VLA.md`*
