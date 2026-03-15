# PHASES 4–6 — MPC CONTROLLER, SENSOR FUSION, SMOLVLA
**Requires**: Phases 1–3 complete and all their gates PASSED.

---

## PHASE 4 — MPC CONTROLLER

**Goal**: `XArmMPCController` must compute physically meaningful torques that drive the arm toward its reference.

### 4.1 — `src/dynamics/xarm_dynamics.py`

Simplified analytical dynamics using a diagonal inertia approximation. Sufficient for MPC linearization.

```python
import numpy as np
import yaml
from typing import Optional

class XArmDynamics:
    """
    Simplified 6-DOF xArm dynamics using diagonal-dominant approximation.
    
    Full M(q) computation from URDF is not needed here — the diagonal
    approximation keeps M(q) positive-definite everywhere, avoiding
    singularities in the MPC linearization.
    
    Gravity vector G(q) uses the full geometric chain.
    """
    
    def __init__(self, config: dict):
        cfg = config['robot']
        self.masses   = np.array(cfg['dynamics']['link_masses'])     # [6]
        self.lengths  = np.array(cfg['dynamics']['link_lengths'])    # [6]
        self.g        = cfg['dynamics']['gravity']
        self.n        = 6

    def inertia_matrix(self, q: np.ndarray) -> np.ndarray:
        """M(q) — [6,6] positive definite.  Diagonal rod-approximation."""
        assert q.shape == (6,)
        diag = self.masses * self.lengths**2 / 3.0 + 0.05   # damping floor
        # Add small off-diagonal coupling via sin(q) scaling
        M = np.diag(diag)
        for i in range(1, self.n):
            coupling = 0.02 * self.masses[i] * abs(np.sin(q[i]))
            M[i-1, i] = M[i, i-1] = coupling
        return M

    def coriolis_vector(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """C(q,qdot)·qdot — [6] simplified centrifugal/Coriolis."""
        assert q.shape == (6,) and qdot.shape == (6,)
        # Simplified: C·qdot ≈ (m * l^2 / 2) * sin(q) * qdot^2
        c = self.masses * self.lengths**2 / 2.0 * np.sin(q) * qdot**2
        return c

    def gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """G(q) — [6] gravity torques using geometric chain."""
        assert q.shape == (6,)
        G = np.zeros(6)
        # Cumulative effect: each joint must support all links above it
        z_height = 0.0
        for i in range(self.n - 1, -1, -1):
            # Gravity contribution from link i+1...n to joint i
            mass_above = self.masses[i:].sum()
            length_i   = self.lengths[i]
            # Only joints with vertical component contribute
            # (joints 1, 2, 4: shoulder pitch, elbow, wrist pitch)
            vertical_gain = abs(np.cos(q[i])) if i in (1, 2, 4) else 0.0
            G[i] = mass_above * self.g * length_i * vertical_gain / 2.0
        return G

    def forward_dynamics(self, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray) -> np.ndarray:
        """qddot = M(q)^-1 * (tau - C*qdot - G(q))"""
        M = self.inertia_matrix(q)
        C = self.coriolis_vector(q, qdot)
        G = self.gravity_vector(q)
        return np.linalg.solve(M, tau - C - G)
```

### 4.2 — `src/mpc/xarm_mpc_controller.py`

```python
import numpy as np
import yaml
from typing import Optional, Dict, Tuple
from src.core.base_controller import BaseController
from src.core.base_solver import BaseQPSolver
from src.dynamics.xarm_dynamics import XArmDynamics

class XArmMPCController(BaseController):
    """
    Single-step torque MPC for xArm 6-DOF.

    Formulation (single-step lookahead, linearized dynamics):
    
        minimize   (q_next - q_ref)^T Q (q_next - q_ref) + tau^T R tau
        subject to  q_next  = q + dt * (qdot + dt * M^-1(tau - C - G))
                   |tau_i| <= tau_max_i
    
    After substitution, this becomes a standard QP in tau.
    
    Args:
        solver:       BaseQPSolver instance (SL or OSQP)
        robot_config: dict loaded from config/robots/xarm_6dof.yaml
        dt:           control timestep (seconds)
        Q:            state cost matrix [6,6]  (default: identity)
        R:            input cost matrix [6,6]  (default: 0.01 * identity)
    """

    def __init__(
        self,
        solver: BaseQPSolver,
        robot_config: dict,
        dt: float = 0.01,
        Q: Optional[np.ndarray] = None,
        R: Optional[np.ndarray] = None,
    ):
        self.solver = solver
        self.dt     = dt
        self.dynamics = XArmDynamics(robot_config)
        rc = robot_config['robot']

        self.tau_max = np.array(rc['torque_limits']['tau_max'][:6])
        self.tau_min = -self.tau_max

        n = 6
        self.Q = Q if Q is not None else np.eye(n)
        self.R = R if R is not None else 0.01 * np.eye(n)

        # For QP inspector
        self._last_qp: Dict = {}

    def reset(self) -> None:
        self._last_qp = {}

    def step(self, state: Tuple, reference: np.ndarray) -> np.ndarray:
        """
        Args:
            state:     (q [6], qdot [6])
            reference: [N, 6] or [6] reference joint angles (only first row used)
        
        Returns:
            tau: [8] torques (6 arm + 2 gripper = 0)
        """
        q, qdot = state
        q_ref = reference[0] if reference.ndim == 2 else reference

        # Build QP
        P, qv, A, l, u = self._build_qp(q, qdot, q_ref)

        # Solve
        tau_arm, info = self.solver.solve(P, qv, A, l, u)

        # Cache for QP inspector
        self._last_qp = {'P': P, 'q': qv, 'A': A, 'l': l, 'u': u,
                         'solution': tau_arm, 'info': info}

        # Append zero gripper torques
        tau_full = np.append(tau_arm, [0., 0.])
        return tau_full

    def _build_qp(self, q, qdot, q_ref):
        """Build QP matrices for torque optimization."""
        M  = self.dynamics.inertia_matrix(q)     # [6,6]
        C  = self.dynamics.coriolis_vector(q, qdot)  # [6]
        G  = self.dynamics.gravity_vector(q)         # [6]
        M_inv = np.linalg.solve(M, np.eye(6))

        dt = self.dt
        # q_next = q + dt*qdot + dt^2 * M_inv @ (tau - C - G)
        # error(tau) = q_next - q_ref = const + dt^2 * M_inv @ tau - dt^2 * M_inv @ (C+G)
        #
        # cost = error^T Q error + tau^T R tau
        #      = tau^T [A_d^T Q A_d + R] tau + 2 b^T Q A_d tau  + const
        # where A_d = dt^2 * M_inv,  b = q + dt*qdot - q_ref - dt^2 * M_inv@(C+G)

        A_d   = (dt**2) * M_inv                       # [6,6]
        b     = q + dt * qdot - q_ref - A_d @ (C + G) # [6]

        P_qp  = A_d.T @ self.Q @ A_d + self.R          # [6,6] symmetric PSD
        q_qp  = A_d.T @ self.Q @ b                      # [6]

        # Symmetrize P (numerical safety)
        P_qp  = 0.5 * (P_qp + P_qp.T)

        # Box constraints on tau
        A_box = np.eye(6)
        l_box = self.tau_min
        u_box = self.tau_max

        return P_qp, q_qp, A_box, l_box, u_box

    def get_last_qp_matrices(self) -> Dict:
        """Return QP matrices for QP inspector webapp."""
        return self._last_qp
```

### 4.3 — Tests for Phase 4

`tests/integration/test_mpc_controller.py`:

```python
import numpy as np, pytest, yaml
from src.mpc.xarm_mpc_controller import XArmMPCController
from src.solver.osqp_solver import OSQPSolver

@pytest.fixture
def config():
    with open("config/robots/xarm_6dof.yaml") as f:
        return yaml.safe_load(f)

@pytest.fixture
def mpc(config):
    return XArmMPCController(solver=OSQPSolver(), robot_config=config, dt=0.01)

def test_output_shape(mpc):
    q = np.zeros(6); qdot = np.zeros(6)
    ref = np.zeros((10, 6))
    tau = mpc.step((q, qdot), ref)
    assert tau.shape == (8,), f"Expected (8,), got {tau.shape}"

def test_output_within_limits(mpc, config):
    q = np.zeros(6); qdot = np.zeros(6)
    ref = np.ones((10, 6))
    tau = mpc.step((q, qdot), ref)
    tau_max = config['robot']['torque_limits']['tau_max']
    for i in range(6):
        assert abs(tau[i]) <= tau_max[i] + 1e-3, f"Joint {i} torque out of limits"

def test_positive_tracking_direction(mpc):
    """Positive reference error on joint 0 must yield positive torque on joint 0."""
    q    = np.zeros(6); qdot = np.zeros(6)
    ref  = np.zeros((10, 6))
    ref[:, 0] = 0.5    # joint 0 reference = +0.5 rad, current = 0
    tau  = mpc.step((q, qdot), ref)
    assert tau[0] > 0, f"Expected positive torque toward +ref, got {tau[0]:.4f}"

def test_zero_error_near_zero_torque(mpc):
    """When reference == current, torque should be small (gravity comp only)."""
    q = np.array([0., 0., 0., 0., 0., 0.])
    ref = np.tile(q, (10, 1))
    tau = mpc.step((q, np.zeros(6)), ref)
    # Allow some gravity compensation torque
    assert np.abs(tau[:6]).max() < 15.0

def test_qp_matrices_available(mpc):
    mpc.step((np.zeros(6), np.zeros(6)), np.zeros((10, 6)))
    qp = mpc.get_last_qp_matrices()
    for key in ('P', 'q', 'A', 'l', 'u', 'solution', 'info'):
        assert key in qp, f"Missing key: {key}"

def test_sl_solver_works(config):
    from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
    mpc = XArmMPCController(
        solver=StuartLandauLagrangeDirect(T_solve=2.0),
        robot_config=config
    )
    tau = mpc.step((np.zeros(6), np.zeros(6)), np.zeros((10, 6)))
    assert tau.shape == (8,)
```

**Phase 4 Gate**: `pytest tests/integration/test_mpc_controller.py -v` — ALL PASS

---

## PHASE 5 — SENSOR FUSION ENCODER

**Goal**: All 5 fusion modes produce non-trivial, dimensionally correct, input-dependent features.

### 5.1 — `src/fusion/real_fusion_encoder.py`

```python
import numpy as np
import time
from src.simulation.cameras.event_camera import EventCameraSimulator, LiDARSimulator

class RealFusionEncoder:
    """
    Multimodal sensor fusion.  All feature extraction uses numpy only —
    no pretrained neural networks, no random outputs.
    
    Feature dimensions (fixed):
        RGB:            128-dim   (spatial grid + channel stats + gradients)
        Events:          96-dim   (spatial histogram + polarity stats)
        LiDAR (sim):     64-dim   (gradient-derived depth features)
        Proprioception:  32-dim   (joint angles + velocities + gripper)
    
    Fusion modes:
        M0: RGB only              → 128-dim
        M1: RGB + Events          → 224-dim
        M2: RGB + LiDAR           → 192-dim
        M3: RGB + Proprio         → 160-dim
        M4: RGB + Events + LiDAR + Proprio → 320-dim
    """
    
    DIM_RGB    = 128
    DIM_EVENT  =  96
    DIM_LIDAR  =  64
    DIM_PROPRIO =  32

    # Factory methods ─────────────────────────────────────────────────────────

    @classmethod
    def mode_rgb_only(cls):        # M0
        return cls(rgb=True)

    @classmethod
    def mode_rgb_events(cls):      # M1
        return cls(rgb=True, events=True)

    @classmethod
    def mode_rgb_lidar(cls):       # M2
        return cls(rgb=True, lidar=True)

    @classmethod
    def mode_rgb_proprio(cls):     # M3
        return cls(rgb=True, proprio=True)

    @classmethod
    def mode_full(cls):            # M4
        return cls(rgb=True, events=True, lidar=True, proprio=True)

    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, rgb=False, events=False, lidar=False, proprio=False):
        self.use_rgb    = rgb
        self.use_events = events
        self.use_lidar  = lidar
        self.use_proprio = proprio
        self._event_cam = EventCameraSimulator() if events else None
        self._lidar_sim = LiDARSimulator()        if lidar  else None

    @property
    def feature_dim(self) -> int:
        d = 0
        if self.use_rgb:     d += self.DIM_RGB
        if self.use_events:  d += self.DIM_EVENT
        if self.use_lidar:   d += self.DIM_LIDAR
        if self.use_proprio: d += self.DIM_PROPRIO
        return d

    def encode(self, observation: dict) -> np.ndarray:
        """
        Args:
            observation: dict with keys:
                'rgb':      [H, W, 3] uint8  (required)
                'prev_rgb': [H, W, 3] uint8  (for events, optional — zeros if missing)
                'state':    [6+]      float  (for proprio, optional — zeros if missing)
        Returns:
            features: [feature_dim] float32
        """
        rgb  = observation['rgb'].astype(np.uint8)
        parts = []
        if self.use_rgb:
            parts.append(self._extract_rgb(rgb))
        if self.use_events:
            prev = observation.get('prev_rgb', np.zeros_like(rgb))
            events = self._event_cam.process_frame(rgb, prev)
            parts.append(self._event_cam.extract_features(events))
        if self.use_lidar:
            parts.append(self._lidar_sim.extract_features(rgb))
        if self.use_proprio:
            state = observation.get('state', np.zeros(7))
            parts.append(self._extract_proprio(state))
        feat = np.concatenate(parts).astype(np.float32)
        assert feat.shape == (self.feature_dim,), \
            f"Dim mismatch: {feat.shape} vs expected {(self.feature_dim,)}"
        return feat

    def encode_with_timing(self, observation: dict):
        t0 = time.perf_counter()
        feat = self.encode(observation)
        ms   = (time.perf_counter() - t0) * 1000.
        return feat, ms

    # ── Feature extractors ───────────────────────────────────────────────────

    def _extract_rgb(self, rgb: np.ndarray) -> np.ndarray:
        """128-dim: spatial(64) + channel_stats(32) + gradient(32)."""
        img = rgb.astype(np.float32) / 255.
        H, W = img.shape[:2]

        # 1. 8×8 spatial grid → 64-dim
        bh, bw = max(1, H // 8), max(1, W // 8)
        spatial = img[:bh*8, :bw*8].reshape(8, bh, 8, bw, 3).mean(axis=(1, 3, 4))
        spatial = spatial.flatten()                   # [64]

        # 2. Per-channel statistics → 32-dim (≈10 stats × 3 channels, capped)
        stats = []
        for c in range(3):
            ch = img[:, :, c]
            stats += [ch.mean(), ch.std(), float(np.percentile(ch, 25)),
                      float(np.percentile(ch, 75)), float(np.median(ch)),
                      ch.min(), ch.max(), np.var(ch),
                      float((ch > 0.5).mean()), float((ch < 0.1).mean())]
        stats = np.array(stats[:30], dtype=np.float32)
        stats = np.pad(stats, (0, 32 - len(stats)))  # [32]

        # 3. Gradient features → 32-dim
        gray = img.mean(axis=2)
        gx   = np.abs(np.diff(gray, axis=1)).mean(axis=1)  # [H]
        gy   = np.abs(np.diff(gray, axis=0)).mean(axis=1)  # [H-1]
        gx16 = np.interp(np.linspace(0, len(gx)-1, 16), np.arange(len(gx)), gx)
        gy16 = np.interp(np.linspace(0, len(gy)-1, 16), np.arange(len(gy)), gy)
        grad = np.concatenate([gx16, gy16]).astype(np.float32)  # [32]

        return np.concatenate([spatial, stats, grad])  # [128]

    def _extract_proprio(self, state: np.ndarray) -> np.ndarray:
        """32-dim: norm_angles(6) + velocities(6) + gripper(2) + pad(18)."""
        q   = np.asarray(state, dtype=np.float32)
        q6  = q[:6] / np.pi                          # normalize to [-1,1]
        vel = np.zeros(6, dtype=np.float32)          # no velocity in state
        grip = np.array([float(q[6]) if len(q) > 6 else 0.], dtype=np.float32)
        raw  = np.concatenate([q6, vel, grip])       # [13]
        return np.pad(raw, (0, 32 - len(raw))).astype(np.float32)  # [32]
```

### 5.2 — Tests for Phase 5

`tests/unit/test_fusion_encoder.py`:

```python
import numpy as np, pytest
from src.fusion.real_fusion_encoder import RealFusionEncoder

def _dummy_rgb(seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (84, 84, 3), dtype=np.uint8)

def _obs(seed=0):
    rgb = _dummy_rgb(seed)
    return {'rgb': rgb, 'prev_rgb': _dummy_rgb(seed+1), 'state': np.zeros(7)}

@pytest.mark.parametrize("mode,factory,expected_dim", [
    ("M0", RealFusionEncoder.mode_rgb_only,    128),
    ("M1", RealFusionEncoder.mode_rgb_events,  224),
    ("M2", RealFusionEncoder.mode_rgb_lidar,   192),
    ("M3", RealFusionEncoder.mode_rgb_proprio, 160),
    ("M4", RealFusionEncoder.mode_full,        320),
])
def test_output_shape(mode, factory, expected_dim):
    enc  = factory()
    feat = enc.encode(_obs())
    assert feat.shape == (expected_dim,), f"{mode}: expected {expected_dim}, got {feat.shape}"
    assert feat.dtype == np.float32
    assert enc.feature_dim == expected_dim

@pytest.mark.parametrize("factory", [
    RealFusionEncoder.mode_rgb_only,
    RealFusionEncoder.mode_full,
])
def test_features_not_all_zero(factory):
    feat = factory().encode(_obs())
    assert np.abs(feat).sum() > 0.1, "Features are all zero"

@pytest.mark.parametrize("factory", [
    RealFusionEncoder.mode_rgb_only,
    RealFusionEncoder.mode_rgb_events,
])
def test_different_images_different_features(factory):
    enc = factory()
    f1 = enc.encode(_obs(seed=0))
    f2 = enc.encode(_obs(seed=99))
    assert not np.allclose(f1, f2), "Different inputs → same output (fake!)"

def test_missing_prev_rgb_handled():
    enc = RealFusionEncoder.mode_rgb_events()
    obs = {'rgb': _dummy_rgb(), 'state': np.zeros(7)}  # No prev_rgb
    feat = enc.encode(obs)   # Must not crash
    assert feat.shape == (224,)

def test_latency_reasonable():
    enc = RealFusionEncoder.mode_full()
    _, ms = enc.encode_with_timing(_obs())
    assert ms < 200, f"Fusion took {ms:.1f}ms — too slow for real-time"
```

**Phase 5 Gate**: `pytest tests/unit/test_fusion_encoder.py -v` — ALL PASS

---

## PHASE 6 — SMOLVLA: MOCK + SERVER + CLIENT

**Goal**: Server infrastructure works in mock mode. Real SmolVLA is optional.

### 6.1 — `src/smolvla/mock_vla.py`

```python
import numpy as np
import time

class MockVLAServer:
    """
    Deterministic mock VLA for testing without model weights.
    
    CRITICAL CONTRACT:
        Every output dict MUST contain "source": "MOCK".
        This prevents mock metrics from being confused with real SmolVLA metrics.
    
    Action is computed deterministically from state (not randomly) so that
    tests are reproducible and results are comparable across runs.
    """
    
    RESPONSE_TIME_MS = 5.0   # Realistic mock latency

    def predict(
        self,
        rgb: np.ndarray,       # [H, W, 3] uint8 (not used by mock, but accepted)
        state: np.ndarray,     # [6] arm joint angles
        instruction: str = "pick up the red block",
    ) -> dict:
        """Returns deterministic 7-D action from state."""
        t0 = time.perf_counter()
        s  = np.asarray(state[:6], dtype=np.float64)
        
        # Deterministic target: slowly move joints toward a "reach" pose
        reach_pose = np.array([0.3, -0.5, 0.8, -0.2, 0.1, 0.0])
        action_6   = s + 0.05 * (reach_pose - s)   # 5% of error per step
        action_7   = np.append(action_6, 0.0)       # gripper open

        elapsed_ms = (time.perf_counter() - t0) * 1000. + self.RESPONSE_TIME_MS
        return {
            'action':     action_7.tolist(),
            'action_std': [0.01] * 7,
            'latency_ms': elapsed_ms,
            'success':    True,
            'source':     'MOCK',   # NEVER REMOVE THIS FIELD
        }
```

### 6.2 — `src/smolvla/vla_server.py`

FastAPI server supporting both mock and real modes:

```python
#!/usr/bin/env python3
"""
VLA inference server. Two startup modes:
    python vla_server.py --mode mock   (fast, no GPU required)
    python vla_server.py --mode real   (slow load, GPU recommended)

Endpoints:
    GET  /health   → {status, mode, device, ready}
    POST /predict  → {action, action_std, latency_ms, success, source}
    GET  /stats    → {call_count, success_count, fail_count, mean_latency_ms}
"""
import argparse, asyncio, base64, io, time, logging
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logger = logging.getLogger("vla_server")
app  = FastAPI(title="VLA Inference Server")

# ── Global state ──────────────────────────────────────────────────────────────
_mode        = "mock"
_ready       = False
_model       = None    # Real model (loaded on startup in real mode)
_mock_server = None    # MockVLAServer instance
_call_count  = 0
_success_count = 0
_fail_count  = 0
_latencies   = []

# ── Request/Response schemas ──────────────────────────────────────────────────

class PredictRequest(BaseModel):
    rgb_image_b64: str
    state:         list  = [0.]*6
    instruction:   str   = "pick up the red block"

class PredictResponse(BaseModel):
    action:     list
    action_std: list
    latency_ms: float
    success:    bool
    source:     str   # "MOCK" or "SmolVLA"

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok" if _ready else "loading",
            "mode": _mode, "ready": _ready}

@app.get("/stats")
async def stats():
    mean_lat = float(np.mean(_latencies)) if _latencies else 0.0
    return {"call_count": _call_count, "success_count": _success_count,
            "fail_count": _fail_count, "mean_latency_ms": mean_lat}

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    global _call_count, _success_count, _fail_count, _latencies
    _call_count += 1
    try:
        img_bytes = base64.b64decode(req.rgb_image_b64)
        from PIL import Image
        pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        rgb = np.array(pil, dtype=np.uint8)
        state = np.array(req.state, dtype=np.float32)

        if _mode == "mock":
            result = _mock_server.predict(rgb, state, req.instruction)
        else:
            result = await _infer_real(rgb, state, req.instruction)

        _success_count += 1
        _latencies.append(result['latency_ms'])
        if len(_latencies) > 1000: _latencies = _latencies[-500:]
        return PredictResponse(**result)
    except Exception as e:
        _fail_count += 1
        logger.error(f"Predict failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _infer_real(rgb, state, instruction):
    """Real SmolVLA inference — only called in --mode real."""
    import torch
    t0 = time.perf_counter()
    obs = {
        "observation.images.top": torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).float()/255.,
        "observation.state": torch.tensor(state).unsqueeze(0),
    }
    with torch.no_grad():
        action = await asyncio.to_thread(_model.select_action, obs)
    ms = (time.perf_counter() - t0) * 1000.
    return {
        'action':     action.squeeze().cpu().numpy().tolist(),
        'action_std': [0.05]*7,
        'latency_ms': ms,
        'success':    True,
        'source':     'SmolVLA',
    }

# ── Startup ───────────────────────────────────────────────────────────────────

def _startup_mock():
    global _mock_server, _ready
    from src.smolvla.mock_vla import MockVLAServer
    _mock_server = MockVLAServer()
    _ready = True
    logger.info("[VLA Server] Mock mode ready.")

def _startup_real():
    global _model, _ready
    logger.info("[VLA Server] Loading SmolVLA model...")
    try:
        from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    except ImportError:
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    _model.to(device).eval()
    _ready = True
    logger.info(f"[VLA Server] SmolVLA loaded on {device}.")

@app.on_event("startup")
async def on_startup():
    global _mode
    if _mode == "mock":
        _startup_mock()
    else:
        await asyncio.to_thread(_startup_real)

# ── CLI entry ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "real"], default="mock")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    _mode = args.mode
    uvicorn.run(app, host="0.0.0.0", port=args.port)
```

### 6.3 — `src/smolvla/vla_client.py`

```python
import base64, io, time, logging
import numpy as np
import requests
from PIL import Image
from src.smolvla.mock_vla import MockVLAServer

logger = logging.getLogger(__name__)

class VLAClient:
    """
    HTTP client for the VLA server.
    
    mock_mode=True:  calls MockVLAServer directly (no HTTP overhead, for unit tests)
    mock_mode=False: calls server at server_url (for integration)
    """
    def __init__(self, server_url="http://localhost:8000",
                 mock_mode=False, timeout_s=5.0):
        self.server_url = server_url
        self.mock_mode  = mock_mode
        self.timeout_s  = timeout_s
        self._mock      = MockVLAServer() if mock_mode else None
        self.call_count = 0
        self.fail_count = 0
        self._latencies = []

    def predict(self, rgb: np.ndarray, state: np.ndarray,
                instruction: str = "pick up the red block") -> dict:
        """Synchronous predict. Returns full response dict."""
        self.call_count += 1
        try:
            if self.mock_mode:
                result = self._mock.predict(rgb, state, instruction)
            else:
                result = self._http_predict(rgb, state, instruction)
            self._latencies.append(result['latency_ms'])
            return result
        except Exception as e:
            self.fail_count += 1
            logger.warning(f"VLA predict failed: {e}")
            raise

    def _http_predict(self, rgb, state, instruction):
        pil = Image.fromarray(rgb)
        buf = io.BytesIO()
        pil.save(buf, format='JPEG', quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {"rgb_image_b64": b64, "state": state.tolist(),
                   "instruction": instruction}
        resp = requests.post(f"{self.server_url}/predict",
                             json=payload, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> bool:
        if self.mock_mode:
            return True
        try:
            r = requests.get(f"{self.server_url}/health", timeout=2.)
            return r.status_code == 200
        except:
            return False

    @property
    def mean_latency_ms(self):
        return float(np.mean(self._latencies)) if self._latencies else 0.
```

### 6.4 — Tests for Phase 6

`tests/integration/test_vla_mock.py`:

```python
import numpy as np, pytest
from src.smolvla.mock_vla import MockVLAServer
from src.smolvla.vla_client import VLAClient

def _dummy_rgb():
    return np.zeros((84, 84, 3), dtype=np.uint8)

def test_mock_server_returns_action():
    srv = MockVLAServer()
    res = srv.predict(_dummy_rgb(), np.zeros(6))
    assert 'action' in res
    assert len(res['action']) == 7

def test_mock_source_field():
    res = MockVLAServer().predict(_dummy_rgb(), np.zeros(6))
    assert res['source'] == 'MOCK', "MOCK source field missing!"

def test_mock_deterministic():
    srv = MockVLAServer()
    state = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    r1 = srv.predict(_dummy_rgb(), state)
    r2 = srv.predict(_dummy_rgb(), state)
    assert r1['action'] == r2['action'], "Mock must be deterministic"

def test_client_mock_mode():
    client = VLAClient(mock_mode=True)
    res = client.predict(_dummy_rgb(), np.zeros(6))
    assert res['source'] == 'MOCK'
    assert client.health_check()

def test_client_mock_different_states():
    client = VLAClient(mock_mode=True)
    r1 = client.predict(_dummy_rgb(), np.zeros(6))
    r2 = client.predict(_dummy_rgb(), np.ones(6) * 0.5)
    assert r1['action'] != r2['action'], "Different states must yield different actions"
```

**Phase 6 Gate**: `pytest tests/integration/test_vla_mock.py -v` — ALL PASS

*Next: Read `04_PHASES7_9_INTEGRATION.md`*
