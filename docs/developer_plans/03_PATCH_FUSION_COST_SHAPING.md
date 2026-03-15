# PATCH: PHASES 4–6 — ADAPTIVE COST SHAPING VIA SENSOR FUSION
**This file supersedes Phase 4 in `03_PHASES4_6_MPC_FUSION_VLA.md`.**
**Phase 5 (fusion encoder) and Phase 6 (SmolVLA) are unchanged.**

---

## WHAT CHANGED AND WHY

The original `XArmMPCController.step()` used static `Q` and `R` matrices set at construction time and never modified. The sensor fusion encoder ran but its output was discarded (`_ = self.fusion.encode(...)`).

The fix introduces:
1. `FusionCostAdapter` — converts fusion feature vectors into physically meaningful Q/R scaling signals
2. `XArmMPCController.step()` — accepts optional `Q_override` and `R_override` per-call
3. `DualSystemController.step()` — feeds fusion output through the adapter, passes adapted costs to MPC (covered in the companion patch file `04_PATCH_DUAL_CONTROLLER_B4.md`)

---

## PHYSICAL INTERPRETATION OF EACH MODALITY → COST MAPPING

```
Modality      Feature Signal          Physical Meaning          MPC Effect
──────────────────────────────────────────────────────────────────────────────
Events        Mean abs spatial        Fast motion happening     ↑ R (smooth torques)
              activity in 5×5 grid    in scene                  during dynamics

LiDAR (sim)   Power in high-gradient  Object with sharp edges   ↑ Q (tight tracking)
              histogram bins          close to gripper          near target

Proprio       |q_normalized|          Joint is near its         ↑ R[i,i] (conservative
              per joint               range limit               per-joint torques)

RGB           Edge density            Complex, structured       ↑ Q uniform
              (gradient features)     scene needing precision   (scale up baseline)
```

The scaling formulas are **additive and bounded**:
```
Q_adapted[i,i] = Q_base[i,i] * (1 + α_prox * prox_score * w_ee[i] + α_rgb * rgb_score)
R_adapted[i,i] = R_base[i,i] * (1 + α_motion * motion_score + α_limit * limit_score[i])

All α parameters in [0, 5].  All scores in [0, 1].  Q/R scale in [1, 6].
```
This is guaranteed PSD (Q, R are always positive definite).

---

## NEW FILE: `src/fusion/cost_adapter.py`

```python
"""
FusionCostAdapter
─────────────────
Maps RealFusionEncoder output features to MPC cost matrix scale factors.

Each modality contributes to a physically motivated cost signal:
    Events      → motion_score    → R scaling (torque smoothness)
    LiDAR       → proximity_score → Q scaling (wrist joint precision)
    Proprio     → limit_scores    → per-joint R scaling
    RGB         → scene_score     → Q uniform scaling

Design constraints:
    1. All scores ∈ [0, 1]   — bounded, no runaway amplification
    2. All scales ≥ 1.0      — fusion only ADDS cost shaping, never removes it
    3. All scales ≤ max_scale — configurable hard cap (default 5.0)
    4. Works with any fusion mode — missing modalities use neutral scores (0.0)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# Feature vector layout constants — must match real_fusion_encoder.py exactly
_DIM_RGB    = 128
_DIM_EVENT  =  96
_DIM_LIDAR  =  64
_DIM_PROPRIO =  32

# RGB feature layout (within its 128-dim block)
_RGB_SPATIAL_SLICE   = slice(0,   64)   # 8×8 spatial grid
_RGB_STATS_SLICE     = slice(64,  96)   # channel statistics
_RGB_GRADIENT_SLICE  = slice(96, 128)   # edge/gradient features

# Event feature layout (within its 96-dim block)
# [0:50]  = 5×5 spatial grid (mean at even indices, std at odd)
# [50:96] = 46-bin polarity histogram
_EVENT_CELL_MEANS = slice(0, 50, 2)   # 25 mean values from spatial grid

# LiDAR feature layout (within its 64-dim block)
# [0:32]  = horizontal gradient histogram (bins 0..0.5)
# [32:64] = vertical gradient histogram
# High bins (indices 20-31, 52-63) correspond to strong depth edges
_LIDAR_HIGH_H = slice(20, 32)
_LIDAR_HIGH_V = slice(52, 64)

# Proprio feature layout (within its 32-dim block)
# [0:6] = q / pi  (normalized angles, ∈ [-1, 1])
_PROPRIO_Q_NORM = slice(0, 6)


@dataclass
class CostScales:
    """Output of FusionCostAdapter. All values ≥ 1.0."""
    Q_scale:       float            = 1.0   # uniform Q multiplier
    Q_joint_diag:  np.ndarray = field(default_factory=lambda: np.ones(6))
    R_scale:       float            = 1.0   # uniform R multiplier
    R_joint_diag:  np.ndarray = field(default_factory=lambda: np.ones(6))

    # Diagnostic — which signals drove the scaling
    motion_score:    float = 0.0
    proximity_score: float = 0.0
    scene_score:     float = 0.0
    limit_scores:    np.ndarray = field(default_factory=lambda: np.zeros(6))


class FusionCostAdapter:
    """
    Converts a fusion feature vector to MPC cost matrix scale factors.

    Args:
        fusion_mode:  which modalities are enabled (M0–M4)
                      Inferred from feature vector length if not given.
        alpha_motion:    R scale gain for motion (default 2.0)
        alpha_proximity: Q scale gain for proximity (default 3.0)
        alpha_scene:     Q scale gain for scene complexity (default 1.0)
        alpha_limit:     per-joint R gain for joint limits (default 2.0)
        max_scale:       hard cap on any scale factor (default 5.0)

    EE joint weighting:
        Joints 3,4,5 (wrist joints) receive 2× the proximity boost.
        Rational: end-effector precision matters more for grasping.
    """

    # Joints closest to end-effector get higher Q weight under proximity
    _EE_JOINT_WEIGHTS = np.array([0.5, 0.5, 0.8, 1.0, 1.3, 1.8])

    def __init__(
        self,
        alpha_motion:    float = 2.0,
        alpha_proximity: float = 3.0,
        alpha_scene:     float = 1.0,
        alpha_limit:     float = 2.0,
        max_scale:       float = 5.0,
    ):
        self.alpha_motion    = alpha_motion
        self.alpha_proximity = alpha_proximity
        self.alpha_scene     = alpha_scene
        self.alpha_limit     = alpha_limit
        self.max_scale       = max_scale

    # ── Public API ─────────────────────────────────────────────────────────

    def adapt(self, features: np.ndarray, active_modes: dict) -> CostScales:
        """
        Compute cost scales from a fused feature vector.

        Args:
            features:     [D] float32 feature vector from RealFusionEncoder
            active_modes: dict with boolean keys 'rgb','events','lidar','proprio'
                          matching the encoder that produced `features`

        Returns:
            CostScales with Q_scale, Q_joint_diag, R_scale, R_joint_diag
        """
        offset = 0
        rgb_feat  = event_feat = lidar_feat = proprio_feat = None

        if active_modes.get('rgb', False):
            rgb_feat = features[offset: offset + _DIM_RGB]
            offset  += _DIM_RGB
        if active_modes.get('events', False):
            event_feat = features[offset: offset + _DIM_EVENT]
            offset    += _DIM_EVENT
        if active_modes.get('lidar', False):
            lidar_feat = features[offset: offset + _DIM_LIDAR]
            offset    += _DIM_LIDAR
        if active_modes.get('proprio', False):
            proprio_feat = features[offset: offset + _DIM_PROPRIO]
            offset      += _DIM_PROPRIO

        motion_score    = self._motion_score(event_feat)
        proximity_score = self._proximity_score(lidar_feat)
        scene_score     = self._scene_score(rgb_feat)
        limit_scores    = self._limit_scores(proprio_feat)

        scales = self._build_scales(motion_score, proximity_score,
                                    scene_score, limit_scores)
        scales.motion_score    = motion_score
        scales.proximity_score = proximity_score
        scales.scene_score     = scene_score
        scales.limit_scores    = limit_scores
        return scales

    # ── Score extractors ───────────────────────────────────────────────────

    def _motion_score(self, event_feat: Optional[np.ndarray]) -> float:
        """
        Events → motion score ∈ [0, 1].

        Extracts mean spatial cell activity (cell means at even indices of
        the 5×5 spatial grid). High mean absolute activity = fast motion.

        Physical: arm moving quickly → events fire everywhere →
                  mean |cell_activity| → high → R↑ (smooth torques).
        """
        if event_feat is None:
            return 0.0
        cell_means = event_feat[_EVENT_CELL_MEANS]          # [25]
        raw = float(np.mean(np.abs(cell_means)))
        return float(np.clip(raw * 4.0, 0.0, 1.0))

    def _proximity_score(self, lidar_feat: Optional[np.ndarray]) -> float:
        """
        LiDAR (sim) → proximity score ∈ [0, 1].

        Power concentrated in high-gradient histogram bins = sharp depth
        edges = object with a clear boundary is close to the gripper.

        Physical: object nearby → Q↑ on wrist joints → tighter EE tracking.
        """
        if lidar_feat is None:
            return 0.0
        # High bins in horizontal + vertical histogram → strong depth discontinuity
        high_power = float(lidar_feat[_LIDAR_HIGH_H].sum() +
                           lidar_feat[_LIDAR_HIGH_V].sum())
        return float(np.clip(high_power * 1.5, 0.0, 1.0))

    def _scene_score(self, rgb_feat: Optional[np.ndarray]) -> float:
        """
        RGB → scene complexity score ∈ [0, 1].

        Uses gradient features (last 32 dims of RGB block).
        High mean gradient = rich, structured scene requiring precision.

        Physical: complex scene → small Q boost across all joints.
        """
        if rgb_feat is None:
            return 0.0
        grad = rgb_feat[_RGB_GRADIENT_SLICE]                 # [32]
        return float(np.clip(float(np.mean(np.abs(grad))) * 3.0, 0.0, 1.0))

    def _limit_scores(self, proprio_feat: Optional[np.ndarray]) -> np.ndarray:
        """
        Proprio → per-joint limit score ∈ [0, 1]^6.

        Normalized joint angles are stored at proprio_feat[0:6] = q / pi.
        |q / pi| = 0 at center, 1 at ±pi limit.

        Physical: joint near limit → conservative torques → R[i,i]↑.
        """
        if proprio_feat is None:
            return np.zeros(6)
        q_norm = np.abs(proprio_feat[_PROPRIO_Q_NORM])       # [6] ∈ [0, 1]
        return np.clip(q_norm, 0.0, 1.0)

    # ── Scale assembly ──────────────────────────────────────────────────────

    def _build_scales(self, motion, proximity, scene, limits) -> CostScales:
        cap = self.max_scale

        # Q: uniform = scene complexity boost + proximity boost
        Q_unif = np.clip(1.0 + self.alpha_scene * scene
                             + self.alpha_proximity * proximity * 0.3,
                         1.0, cap)

        # Q per-joint: EE joints amplified by proximity
        w = self._EE_JOINT_WEIGHTS / self._EE_JOINT_WEIGHTS.mean()
        Q_joint = np.clip(1.0 + self.alpha_proximity * proximity * w,
                          1.0, cap)

        # R: uniform = motion smoothness
        R_unif = np.clip(1.0 + self.alpha_motion * motion, 1.0, cap)

        # R per-joint: joint-limit penalty
        R_joint = np.clip(1.0 + self.alpha_limit * limits, 1.0, cap)

        return CostScales(
            Q_scale      = float(Q_unif),
            Q_joint_diag = Q_joint,
            R_scale      = float(R_unif),
            R_joint_diag = R_joint,
        )
```

---

## UPDATED: `src/mpc/xarm_mpc_controller.py`

**Changes from original plan:**
- `step()` accepts optional `cost_scales: CostScales` argument
- `_build_qp()` uses adapted Q/R when scales are provided
- `get_active_costs()` exposes current Q/R for diagnostics

```python
import numpy as np
from typing import Optional, Dict, Tuple
from src.core.base_controller import BaseController
from src.core.base_solver import BaseQPSolver
from src.dynamics.xarm_dynamics import XArmDynamics
from src.fusion.cost_adapter import CostScales   # NEW

class XArmMPCController(BaseController):
    """
    Single-step torque MPC for xArm 6-DOF.

    Formulation:
        minimize   (q_next - q_ref)^T Q (q_next - q_ref) + tau^T R tau
        subject to  q_next = q + dt*(qdot + dt*M^-1*(tau - C - G))
                    |tau_i| ≤ tau_max_i

    cost_scales (optional, from FusionCostAdapter):
        Scales Q and R per-step based on sensor fusion signals.
        When None, uses static Q_base and R_base.
    """

    def __init__(
        self,
        solver:       BaseQPSolver,
        robot_config: dict,
        dt:           float = 0.01,
        Q:            Optional[np.ndarray] = None,
        R:            Optional[np.ndarray] = None,
    ):
        self.solver   = solver
        self.dt       = dt
        self.dynamics = XArmDynamics(robot_config)
        rc            = robot_config['robot']

        self.tau_max = np.array(rc['torque_limits']['tau_max'][:6])
        self.tau_min = -self.tau_max

        n = 6
        self.Q_base = Q if Q is not None else np.eye(n)
        self.R_base = R if R is not None else 0.01 * np.eye(n)

        # For QP inspector and diagnostics
        self._last_qp:     Dict = {}
        self._last_Q_used: np.ndarray = self.Q_base.copy()
        self._last_R_used: np.ndarray = self.R_base.copy()

    def reset(self) -> None:
        self._last_qp     = {}
        self._last_Q_used = self.Q_base.copy()
        self._last_R_used = self.R_base.copy()

    def step(
        self,
        state:       Tuple,
        reference:   np.ndarray,
        cost_scales: Optional[CostScales] = None,   # ← NEW
    ) -> np.ndarray:
        """
        Args:
            state:       (q [6], qdot [6])
            reference:   [N, 6] or [6] reference joint angles
            cost_scales: optional CostScales from FusionCostAdapter

        Returns:
            tau: [8] torques (6 arm + 2 gripper = 0)
        """
        q, qdot = state
        q_ref   = reference[0] if reference.ndim == 2 else reference

        Q_eff, R_eff = self._apply_scales(cost_scales)

        P, qv, A, l, u = self._build_qp(q, qdot, q_ref, Q_eff, R_eff)
        tau_arm, info  = self.solver.solve(P, qv, A, l, u)

        self._last_qp     = {'P': P, 'q': qv, 'A': A, 'l': l, 'u': u,
                             'solution': tau_arm, 'info': info,
                             'Q_used': Q_eff.tolist(), 'R_used': R_eff.tolist()}
        self._last_Q_used = Q_eff
        self._last_R_used = R_eff

        return np.append(tau_arm, [0., 0.])   # append zero gripper torques

    # ── Private helpers ─────────────────────────────────────────────────────

    def _apply_scales(self, scales: Optional[CostScales]) -> Tuple[np.ndarray, np.ndarray]:
        """Apply CostScales to Q_base and R_base → (Q_eff, R_eff)."""
        if scales is None:
            return self.Q_base.copy(), self.R_base.copy()

        # Q: uniform scale × per-joint diagonal
        Q_eff = self.Q_base * scales.Q_scale * np.diag(scales.Q_joint_diag)
        Q_eff = 0.5 * (Q_eff + Q_eff.T)   # symmetrize

        # R: uniform scale × per-joint diagonal
        R_eff = self.R_base * scales.R_scale * np.diag(scales.R_joint_diag)
        R_eff = 0.5 * (R_eff + R_eff.T)

        return Q_eff, R_eff

    def _build_qp(self, q, qdot, q_ref, Q, R):
        """Build QP matrices using provided Q and R."""
        M     = self.dynamics.inertia_matrix(q)
        C     = self.dynamics.coriolis_vector(q, qdot)
        G     = self.dynamics.gravity_vector(q)
        M_inv = np.linalg.solve(M, np.eye(6))

        dt  = self.dt
        A_d = (dt**2) * M_inv
        b   = q + dt * qdot - q_ref - A_d @ (C + G)

        P_qp = 0.5 * (A_d.T @ Q @ A_d + R)
        P_qp = P_qp + P_qp.T   # fully symmetrize
        q_qp = A_d.T @ Q @ b

        return P_qp, q_qp, np.eye(6), self.tau_min, self.tau_max

    def get_last_qp_matrices(self) -> Dict:
        """For QP inspector webapp."""
        return self._last_qp

    def get_active_costs(self) -> Dict:
        """Return currently active Q/R matrices for diagnostics."""
        return {'Q': self._last_Q_used, 'R': self._last_R_used}
```

---

## UPDATED TESTS FOR PHASE 4

`tests/integration/test_mpc_controller.py` — add to existing tests:

```python
# ── NEW: adaptive cost shaping tests ─────────────────────────────────────────

from src.fusion.cost_adapter import FusionCostAdapter, CostScales

def test_step_accepts_cost_scales(mpc):
    """step() must accept cost_scales without crashing."""
    scales = CostScales(Q_scale=2.0, Q_joint_diag=np.ones(6)*2,
                        R_scale=1.5, R_joint_diag=np.ones(6)*1.5)
    tau = mpc.step((np.zeros(6), np.zeros(6)), np.zeros((10,6)),
                   cost_scales=scales)
    assert tau.shape == (8,)

def test_high_Q_scale_increases_tracking_torque(mpc):
    """Doubling Q should increase torques toward a non-zero reference."""
    q = np.zeros(6); ref = np.zeros((10,6)); ref[:,0] = 0.5
    tau_base = mpc.step((q, np.zeros(6)), ref, cost_scales=None)

    high_Q = CostScales(Q_scale=5.0, Q_joint_diag=np.ones(6),
                        R_scale=1.0,  R_joint_diag=np.ones(6))
    tau_highQ = mpc.step((q, np.zeros(6)), ref, cost_scales=high_Q)

    # Higher Q → bigger torque toward reference
    assert abs(tau_highQ[0]) > abs(tau_base[0]), \
        f"High Q should increase tracking torque: {tau_base[0]:.3f} vs {tau_highQ[0]:.3f}"

def test_high_R_scale_reduces_torque_magnitude(mpc):
    """Higher R penalizes torque → smaller magnitude output."""
    q = np.zeros(6); ref = np.zeros((10,6)); ref[:,0] = 0.5
    tau_base = mpc.step((q, np.zeros(6)), ref, cost_scales=None)

    high_R = CostScales(Q_scale=1.0, Q_joint_diag=np.ones(6),
                        R_scale=10.0, R_joint_diag=np.ones(6)*2)
    tau_highR = mpc.step((q, np.zeros(6)), ref, cost_scales=high_R)

    assert abs(tau_highR[0]) < abs(tau_base[0]), \
        f"High R should reduce torque: {tau_base[0]:.3f} vs {tau_highR[0]:.3f}"

def test_fusion_cost_adapter_integration(config):
    """Full pipeline: fusion features → adapter → MPC scales."""
    import numpy as np
    from src.fusion.real_fusion_encoder import RealFusionEncoder

    encoder = RealFusionEncoder.mode_full()
    adapter = FusionCostAdapter()
    mpc     = XArmMPCController(OSQPSolver(), config)
    modes   = {'rgb': True, 'events': True, 'lidar': True, 'proprio': True}

    rng = np.random.default_rng(0)
    obs = {
        'rgb':      rng.integers(0, 200, (84,84,3), dtype=np.uint8),
        'prev_rgb': rng.integers(0, 200, (84,84,3), dtype=np.uint8),
        'state':    np.zeros(7),
    }
    features = encoder.encode(obs)
    scales   = adapter.adapt(features, modes)

    assert scales.Q_scale  >= 1.0
    assert scales.R_scale  >= 1.0
    assert all(scales.Q_joint_diag >= 1.0)
    assert all(scales.R_joint_diag >= 1.0)

    tau = mpc.step((np.zeros(6), np.zeros(6)), np.zeros((10,6)),
                   cost_scales=scales)
    assert tau.shape == (8,)

def test_cost_adapter_neutral_with_no_modalities():
    """M0 (RGB-only) should produce near-neutral scales (≈1.0)."""
    adapter  = FusionCostAdapter()
    # M0 has no events, lidar, or proprio — all scores → 0
    scales   = adapter.adapt(np.zeros(128),
                             {'rgb': True, 'events': False,
                              'lidar': False, 'proprio': False})
    # scene_score may be tiny (near-zero features), scales ≈ 1.0
    assert scales.Q_scale  < 2.0, "RGB-only should give near-neutral Q scale"
    assert scales.R_scale  < 2.0, "RGB-only should give near-neutral R scale"

def test_cost_adapter_scores_in_bounds():
    """All scores must be in [0, 1], all scales in [1, max_scale]."""
    adapter = FusionCostAdapter(max_scale=5.0)
    from src.fusion.real_fusion_encoder import RealFusionEncoder
    enc     = RealFusionEncoder.mode_full()
    rng     = np.random.default_rng(42)
    obs     = {'rgb': rng.integers(0,255,(84,84,3),dtype=np.uint8),
               'prev_rgb': rng.integers(0,255,(84,84,3),dtype=np.uint8),
               'state': rng.standard_normal(7)}
    feat    = enc.encode(obs)
    modes   = {'rgb':True,'events':True,'lidar':True,'proprio':True}
    scales  = adapter.adapt(feat, modes)

    assert 0.0 <= scales.motion_score    <= 1.0
    assert 0.0 <= scales.proximity_score <= 1.0
    assert 0.0 <= scales.scene_score     <= 1.0
    assert all(0.0 <= s <= 1.0 for s in scales.limit_scores)

    assert 1.0 <= scales.Q_scale  <= 5.0
    assert 1.0 <= scales.R_scale  <= 5.0
    assert all(1.0 <= s <= 5.0 for s in scales.Q_joint_diag)
    assert all(1.0 <= s <= 5.0 for s in scales.R_joint_diag)
```

**Phase 4 Gate (updated)**: `pytest tests/integration/test_mpc_controller.py -v` — ALL PASS including the new adaptive cost shaping tests.

---

## HOW TO ADD THIS TO `config/solvers/sl_neuromorphic.yaml`

Add a `fusion_cost_adapter` section so alpha values are configurable:

```yaml
fusion_cost_adapter:
  alpha_motion:    2.0   # R gain: fast motion → smoother torques
  alpha_proximity: 3.0   # Q gain: near object → tighter EE tracking
  alpha_scene:     1.0   # Q gain: complex scene → higher precision
  alpha_limit:     2.0   # R gain: near joint limit → conservative torques
  max_scale:       5.0   # hard cap: no scale factor exceeds this
```

*Next: Read `04_PATCH_DUAL_CONTROLLER_B4.md` for the DualSystemController update and new B4 benchmark.*
