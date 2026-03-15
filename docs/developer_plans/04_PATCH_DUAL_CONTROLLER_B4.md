# PATCH: DUAL CONTROLLER + B4 SENSOR ABLATION BENCHMARK
**This file supersedes Phase 7.2 and Phase 9 B4 in `04_PHASES7_9_INTEGRATION.md`.**
**Phase 7.1 (TrajectoryBuffer), 7.3 (action_processor), Phase 8 (dataset/GIF),
B1/B2/B3/B5 benchmarks are all unchanged.**

---

## WHAT CHANGED AND WHY

`DualSystemController.step()` previously discarded the fusion output with `_ = self.fusion.encode(...)`. It now:

1. Calls `FusionCostAdapter.adapt(features, active_modes)` to convert features → `CostScales`
2. Passes `cost_scales` to `mpc.step()` so Q/R are shaped per-step by sensor state
3. Records all scaling signals in `info` dict for benchmarking and webapp visualization

The B4 benchmark is redesigned around **three scenarios** that specifically stress each modality, so we can actually observe whether the scaling helps, rather than just measuring average RMSE across arbitrary runs.

---

## UPDATED: `src/integration/dual_controller.py`

```python
import threading, time, logging
import numpy as np
from typing import Optional

from src.mpc.xarm_mpc_controller import XArmMPCController
from src.smolvla.vla_client import VLAClient
from src.fusion.real_fusion_encoder import RealFusionEncoder
from src.fusion.cost_adapter import FusionCostAdapter, CostScales
from src.integration.trajectory_buffer import TrajectoryBuffer
from src.smolvla.action_processor import process_action_chunk

logger = logging.getLogger(__name__)


class DualSystemController:
    """
    System 1 (MPC at 100 Hz) + System 2 (VLA at 5 Hz background thread).

    Fusion integration via adaptive cost shaping:
        Every step():
          1. Encode sensors → feature vector (all enabled modalities)
          2. FusionCostAdapter maps features → CostScales (Q/R scaling)
          3. MPC uses adapted Q/R → physically informed torques
          4. VLA still receives only (RGB, state, instruction) — unchanged API

    active_modes:  dict telling the adapter which modalities are active,
                   must match the encoder factory used.
                   e.g. RealFusionEncoder.mode_full() → all True
    """

    # Canonical map from factory method name → active_modes dict
    _FACTORY_TO_MODES = {
        'mode_rgb_only':    {'rgb': True,  'events': False, 'lidar': False, 'proprio': False},
        'mode_rgb_events':  {'rgb': True,  'events': True,  'lidar': False, 'proprio': False},
        'mode_rgb_lidar':   {'rgb': True,  'events': False, 'lidar': True,  'proprio': False},
        'mode_rgb_proprio': {'rgb': True,  'events': False, 'lidar': False, 'proprio': True},
        'mode_full':        {'rgb': True,  'events': True,  'lidar': True,  'proprio': True},
    }

    def __init__(
        self,
        mpc:            XArmMPCController,
        vla_client:     VLAClient,
        fusion_encoder: RealFusionEncoder,
        instruction:    str   = "pick up the red block",
        vla_interval:   float = 0.2,
        vla_timeout:    float = 5.0,
        cost_adapter:   Optional[FusionCostAdapter] = None,
        fusion_mode_name: str = 'mode_full',   # used to look up active_modes
    ):
        self.mpc      = mpc
        self.vla      = vla_client
        self.fusion   = fusion_encoder
        self.instruction  = instruction
        self.vla_interval = vla_interval
        self.vla_timeout  = vla_timeout

        # Adapter + mode map
        self.adapter      = cost_adapter or FusionCostAdapter()
        self.active_modes = self._FACTORY_TO_MODES.get(
            fusion_mode_name,
            {'rgb': True, 'events': False, 'lidar': False, 'proprio': False}
        )

        # Trajectory buffer (written by VLA thread, read by MPC)
        self.buffer = TrajectoryBuffer(horizon=10, action_dim=6)

        # Shared observation buffers (step() writes, VLA thread reads)
        self._rgb_buf   = None
        self._state_buf = None
        self._prev_rgb  = None
        self._buf_lock  = threading.Lock()

        # Diagnostics — all lists are appended per-step
        self.mpc_times_ms    = []
        self.fusion_times_ms = []
        self.motion_scores   = []
        self.proximity_scores= []
        self.Q_scales        = []
        self.R_scales        = []
        self.vla_calls   = 0
        self.vla_success = 0
        self.vla_fail    = 0

        self._running = False
        self._thread  = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._vla_loop, daemon=True,
                                         name="VLA-Thread")
        self._thread.start()
        logger.info("DualController started (VLA thread running).")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        logger.info("DualController stopped.")

    # ── Main control step (synchronous, call at 100 Hz) ────────────────────

    def step(self, obs: dict) -> tuple:
        """
        Args:
            obs: {'q': [6], 'qdot': [6], 'rgb': [H,W,3]}

        Returns:
            tau:  [8]  torque command
            info: dict with timing, fusion scores, cost scales, VLA stats
        """
        t_step = time.perf_counter()
        q, qdot, rgb = obs['q'], obs['qdot'], obs['rgb']

        # ── 1. Encode sensors ─────────────────────────────────────────────
        obs_aug = {
            'rgb':   rgb,
            'state': np.append(q, 0.),   # [7]: 6 joints + gripper placeholder
        }
        if self._prev_rgb is not None:
            obs_aug['prev_rgb'] = self._prev_rgb

        t_fuse = time.perf_counter()
        features, fuse_ms = self.fusion.encode_with_timing(obs_aug)
        self._prev_rgb = rgb.copy()
        self.fusion_times_ms.append(fuse_ms)

        # ── 2. Compute adaptive cost scales ──────────────────────────────
        scales = self.adapter.adapt(features, self.active_modes)

        self.motion_scores.append(scales.motion_score)
        self.proximity_scores.append(scales.proximity_score)
        self.Q_scales.append(scales.Q_scale)
        self.R_scales.append(scales.R_scale)

        # ── 3. Update shared buffers for VLA thread ────────────────────
        with self._buf_lock:
            self._rgb_buf   = rgb.copy()
            self._state_buf = q.copy()

        # ── 4. Get trajectory reference (non-blocking) ──────────────────
        ref = self.buffer.get_reference()    # [10, 6]

        # ── 5. MPC solve with adapted costs ─────────────────────────────
        t_mpc = time.perf_counter()
        tau   = self.mpc.step((q, qdot), ref, cost_scales=scales)
        mpc_ms = (time.perf_counter() - t_mpc) * 1000.
        self.mpc_times_ms.append(mpc_ms)

        total_ms = (time.perf_counter() - t_step) * 1000.

        info = {
            'total_time_ms':    total_ms,
            'mpc_time_ms':      mpc_ms,
            'fusion_time_ms':   fuse_ms,
            # fusion signals
            'motion_score':     scales.motion_score,
            'proximity_score':  scales.proximity_score,
            'scene_score':      scales.scene_score,
            'Q_scale':          scales.Q_scale,
            'R_scale':          scales.R_scale,
            'Q_joint_diag':     scales.Q_joint_diag.tolist(),
            'R_joint_diag':     scales.R_joint_diag.tolist(),
            # VLA / buffer stats
            'buffer_updates':   self.buffer.update_count,
            'buffer_staleness_s': self.buffer.staleness_s,
            'vla_success':      self.vla_success,
            'vla_fail':         self.vla_fail,
        }
        return tau, info

    def get_stats(self) -> dict:
        def safe_mean(lst): return float(np.mean(lst)) if lst else 0.
        def safe_max(lst):  return float(np.max(lst))  if lst else 0.
        return {
            'mean_mpc_ms':       safe_mean(self.mpc_times_ms),
            'max_mpc_ms':        safe_max(self.mpc_times_ms),
            'mean_fusion_ms':    safe_mean(self.fusion_times_ms),
            'mean_motion_score': safe_mean(self.motion_scores),
            'mean_prox_score':   safe_mean(self.proximity_scores),
            'mean_Q_scale':      safe_mean(self.Q_scales),
            'mean_R_scale':      safe_mean(self.R_scales),
            'vla_calls':         self.vla_calls,
            'vla_success':       self.vla_success,
            'vla_fail':          self.vla_fail,
            'buffer_updates':    self.buffer.update_count,
        }

    # ── VLA background thread ──────────────────────────────────────────────

    def _vla_loop(self):
        logger.info("VLA background thread started.")
        while self._running:
            time.sleep(self.vla_interval)
            with self._buf_lock:
                rgb   = self._rgb_buf
                state = self._state_buf
            if rgb is None or state is None:
                continue
            self._do_vla_query(rgb, state)

    def _do_vla_query(self, rgb, state):
        self.vla_calls += 1
        try:
            # VLA API is UNCHANGED: only RGB + state + instruction
            # Fusion features are NOT passed to VLA (would require retraining)
            result     = self.vla.predict(rgb, state, self.instruction)
            action_7   = np.array(result['action'], dtype=np.float64)
            chunk      = process_action_chunk(action_7, horizon=10)  # [10, 6]
            self.buffer.update(chunk)
            self.vla_success += 1
        except Exception as e:
            self.vla_fail += 1
            logger.warning(f"VLA query failed: {e}")
```

---

## UPDATED TESTS FOR PHASE 7

Add to `tests/integration/test_dual_controller.py`:

```python
# ── NEW: fusion → MPC cost shaping integration tests ─────────────────────────

def test_fusion_scores_recorded_in_info(controller):
    """step() info dict must contain fusion scores and cost scales."""
    tau, info = controller.step(_make_obs())
    required = ['motion_score', 'proximity_score', 'Q_scale', 'R_scale',
                'fusion_time_ms', 'Q_joint_diag', 'R_joint_diag']
    for key in required:
        assert key in info, f"Missing info key: {key}"

def test_fusion_scores_in_valid_range(controller):
    """All fusion scores must be in [0, 1], all scales in [1, 5]."""
    _, info = controller.step(_make_obs())
    assert 0.0 <= info['motion_score']    <= 1.0
    assert 0.0 <= info['proximity_score'] <= 1.0
    assert 1.0 <= info['Q_scale']         <= 5.0
    assert 1.0 <= info['R_scale']         <= 5.0

def test_cost_scales_different_for_different_scenes():
    """Visually different scenes should produce different cost scales."""
    import yaml
    from src.fusion.real_fusion_encoder import RealFusionEncoder
    from src.fusion.cost_adapter import FusionCostAdapter
    from src.mpc.xarm_mpc_controller import XArmMPCController
    from src.solver.osqp_solver import OSQPSolver

    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)

    enc   = RealFusionEncoder.mode_full()
    adap  = FusionCostAdapter()
    modes = {'rgb':True,'events':True,'lidar':True,'proprio':True}
    rng   = np.random.default_rng(0)

    # Scene A: dark, static
    obs_a = {'rgb': np.zeros((84,84,3), dtype=np.uint8),
             'prev_rgb': np.zeros((84,84,3), dtype=np.uint8),
             'state': np.zeros(7)}
    # Scene B: bright, high-gradient (lots of edges = object nearby)
    rgb_b = rng.integers(100, 255, (84,84,3), dtype=np.uint8)
    obs_b = {'rgb': rgb_b,
             'prev_rgb': np.zeros((84,84,3), dtype=np.uint8),
             'state': np.ones(7) * 0.8}  # near joint limits

    feat_a = enc.encode(obs_a); scales_a = adap.adapt(feat_a, modes)
    feat_b = enc.encode(obs_b); scales_b = adap.adapt(feat_b, modes)

    # The two scenes must produce different Q and/or R scales
    Q_diff = abs(scales_a.Q_scale - scales_b.Q_scale)
    R_diff = abs(scales_a.R_scale - scales_b.R_scale)
    assert Q_diff > 0.01 or R_diff > 0.01, \
        "Different scenes should produce different cost scales"

def test_m0_vs_m4_cost_scales_differ():
    """M0 (RGB only) and M4 (full) should produce different scales on same scene."""
    from src.fusion.real_fusion_encoder import RealFusionEncoder
    from src.fusion.cost_adapter import FusionCostAdapter

    rng = np.random.default_rng(1)
    rgb = rng.integers(50, 200, (84,84,3), dtype=np.uint8)
    obs = {'rgb': rgb, 'prev_rgb': rng.integers(50,200,(84,84,3),dtype=np.uint8),
           'state': np.array([1.5, 0.8, -1.2, 0.3, 0.5, -0.9, 0.])}

    adapter = FusionCostAdapter()

    f_m0 = RealFusionEncoder.mode_rgb_only().encode(obs)
    f_m4 = RealFusionEncoder.mode_full().encode(obs)

    s_m0 = adapter.adapt(f_m0, {'rgb':True,'events':False,'lidar':False,'proprio':False})
    s_m4 = adapter.adapt(f_m4, {'rgb':True,'events':True,'lidar':True,'proprio':True})

    # M4 must have a higher R scale (events + limits active vs M0)
    assert s_m4.R_scale >= s_m0.R_scale, \
        f"M4 R_scale ({s_m4.R_scale:.3f}) should be >= M0 ({s_m0.R_scale:.3f})"
```

---

## UPDATED: B4 — `evaluation/benchmarks/sensor_ablation.py`

**Redesign rationale**: Instead of running all modes on identical generic tracking, we run three
scenarios that specifically stress the sensor fusion signals. This gives the benchmark
discriminative power — if fusion is working, M4 should beat M0 in each scenario that
the missing modalities are sensitive to.

```python
#!/usr/bin/env python3
"""
B4 — Sensor Fusion Ablation Study
══════════════════════════════════
Tests all 5 fusion modes (M0–M4) across 3 scenarios.

Scenarios
─────────
A. Precision Reach   — slow approach to a precise target joint config
   Tests: proximity_score → Q shaping → tighter EE tracking (LiDAR)
   Expected winner: M2 (RGB+LiDAR) or M4 (Full)

B. Dynamic Tracking  — fast-changing sinusoidal reference  
   Tests: motion_score → R shaping → smoother torques (Events)
   Expected winner: M1 (RGB+Events) or M4 (Full)

C. Near-Limit Motion — reference that takes joints near their limits
   Tests: limit_score → per-joint R shaping (Proprio)
   Expected winner: M3 (RGB+Proprio) or M4 (Full)

Metrics per scenario × mode (3 episodes each = 45 runs total):
    tracking_rmse_rad:    mean ||q - q_ref||_2 per step
    torque_smoothness:    1 / (1 + mean ||Δτ||)  — higher is smoother
    task_success_rate:    fraction of episodes where min error < threshold
    time_to_reach_steps:  steps until first success (-1 if never)
    mean_Q_scale:         mean adaptive Q multiplier (shows fusion working)
    mean_R_scale:         mean adaptive R multiplier
    mean_fusion_ms:       feature extraction latency
"""

import json, time, os, yaml, datetime, platform, sys
import numpy as np

os.environ.setdefault('MUJOCO_GL', 'osmesa')

from src.simulation.envs.xarm_env import XArmEnv
from src.mpc.xarm_mpc_controller import XArmMPCController
from src.solver.osqp_solver import OSQPSolver
from src.smolvla.vla_client import VLAClient
from src.fusion.real_fusion_encoder import RealFusionEncoder
from src.fusion.cost_adapter import FusionCostAdapter
from src.integration.dual_controller import DualSystemController

# ── Scenario definitions ──────────────────────────────────────────────────────

def make_reference_precision(n_steps=200):
    """
    Scenario A: slow ramp to a precise target.
    Joint 0 → 0.3 rad over 100 steps, hold for 100 steps.
    Low speed → motion score ≈ 0, proximity matters.
    """
    ref = np.zeros((n_steps, 6))
    target = np.array([0.3, -0.2, 0.1, 0.05, -0.1, 0.0])
    for i in range(n_steps):
        alpha = min(1.0, i / 100.)
        ref[i] = target * alpha
    return ref, target

def make_reference_dynamic(n_steps=200):
    """
    Scenario B: fast sinusoidal reference (ω = 2π rad/s at 100Hz).
    High qdot → large frame differences → events fire → motion score↑.
    """
    t = np.arange(n_steps) * 0.01    # 0..1.99 s
    ref = np.zeros((n_steps, 6))
    ref[:, 0] = 0.4 * np.sin(2 * np.pi * 0.8 * t)
    ref[:, 1] = 0.2 * np.sin(2 * np.pi * 1.2 * t + 0.5)
    ref[:, 2] = 0.15 * np.sin(2 * np.pi * 1.5 * t + 1.0)
    target = ref[-1]
    return ref, target

def make_reference_near_limits(n_steps=200):
    """
    Scenario C: target near joint limits (80% of range).
    Joint 0 max = 6.283 → target = 5.0 rad.
    Near-limit → proprio limit_scores high → R[i,i]↑.
    """
    ref = np.zeros((n_steps, 6))
    target = np.array([5.0, 3.0, -5.5, 4.0, -5.5, 5.5])  # ~80% of limits
    for i in range(n_steps):
        alpha = min(1.0, i / 150.)
        ref[i] = target * alpha
    return ref, target

SCENARIOS = {
    'A_precision':    (make_reference_precision,  {'threshold_rad': 0.05}),
    'B_dynamic':      (make_reference_dynamic,    {'threshold_rad': 0.15}),
    'C_near_limits':  (make_reference_near_limits, {'threshold_rad': 0.20}),
}

FUSION_MODES = {
    'M0_rgb':         (RealFusionEncoder.mode_rgb_only,    'mode_rgb_only'),
    'M1_events':      (RealFusionEncoder.mode_rgb_events,  'mode_rgb_events'),
    'M2_lidar':       (RealFusionEncoder.mode_rgb_lidar,   'mode_rgb_lidar'),
    'M3_proprio':     (RealFusionEncoder.mode_rgb_proprio, 'mode_rgb_proprio'),
    'M4_full':        (RealFusionEncoder.mode_full,        'mode_full'),
}

N_EPISODES = 3
SUCCESS_THRESHOLD = 0.10   # rad (default; scenario overrides via 'threshold_rad')

# ── Helper functions ──────────────────────────────────────────────────────────

def evaluate_success(q_traj, q_target, threshold):
    errors = [float(np.linalg.norm(np.array(q) - q_target)) for q in q_traj]
    min_err = min(errors)
    t_reach = next((i for i, e in enumerate(errors) if e < threshold), -1)
    return {
        'success':        min_err < threshold,
        'min_error_rad':  min_err,
        'time_to_reach':  t_reach,
        'final_error_rad': errors[-1],
    }

def torque_smoothness(torques):
    """1 / (1 + mean ||τ[t+1] - τ[t]||). Range (0,1]. Higher = smoother."""
    if len(torques) < 2:
        return 1.0
    deltas = [np.linalg.norm(np.array(torques[i+1]) - np.array(torques[i]))
              for i in range(len(torques)-1)]
    return float(1.0 / (1.0 + np.mean(deltas)))

def run_episode(env, ctrl, ref_fn, episode_seed, threshold):
    """Run one episode. Returns per-step metrics."""
    rng   = np.random.default_rng(episode_seed)
    ref, target = ref_fn()
    n_steps = len(ref)

    obs = env.reset()
    # Apply small random perturbation for reproducible variety across episodes
    perturb = rng.standard_normal(8) * 0.01
    for _ in range(3):
        obs, _, _, _ = env.step(perturb)

    q_traj = []; torques = []
    Q_scales = []; R_scales = []; fuse_ms_list = []
    rmse_per_step = []

    for step in range(n_steps):
        tau, info = ctrl.step(obs)
        obs, _, done, _ = env.step(tau)

        q_traj.append(obs['q'].tolist())
        torques.append(tau[:6].tolist())
        Q_scales.append(info['Q_scale'])
        R_scales.append(info['R_scale'])
        fuse_ms_list.append(info['fusion_time_ms'])
        rmse_per_step.append(float(np.sqrt(np.mean((obs['q'] - ref[step])**2))))
        if done:
            break

    success_info = evaluate_success(q_traj, target, threshold)
    return {
        'tracking_rmse_rad':  float(np.mean(rmse_per_step)),
        'torque_smoothness':  torque_smoothness(torques),
        'mean_Q_scale':       float(np.mean(Q_scales)),
        'mean_R_scale':       float(np.mean(R_scales)),
        'mean_fusion_ms':     float(np.mean(fuse_ms_list)),
        'n_steps_completed':  len(q_traj),
        **success_info,
    }

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)

    env     = XArmEnv(render_mode='offscreen')
    vla     = VLAClient(mock_mode=True)
    adapter = FusionCostAdapter()
    results = {}

    for scenario_name, (ref_fn, scenario_cfg) in SCENARIOS.items():
        threshold = scenario_cfg.get('threshold_rad', SUCCESS_THRESHOLD)
        results[scenario_name] = {}
        print(f"\n══ Scenario {scenario_name} ══")

        for mode_name, (enc_factory, factory_name) in FUSION_MODES.items():
            encoder = enc_factory()
            mpc     = XArmMPCController(OSQPSolver(), cfg)
            ctrl    = DualSystemController(
                mpc, vla, encoder,
                cost_adapter=adapter,
                fusion_mode_name=factory_name,
            )
            ctrl.start()

            episodes_data = []
            for ep in range(N_EPISODES):
                ep_result = run_episode(env, ctrl, ref_fn, episode_seed=ep*7+42,
                                        threshold=threshold)
                episodes_data.append(ep_result)
                print(f"  {mode_name} ep{ep}: "
                      f"RMSE={ep_result['tracking_rmse_rad']:.4f}  "
                      f"smooth={ep_result['torque_smoothness']:.3f}  "
                      f"success={ep_result['success']}  "
                      f"Q_scale={ep_result['mean_Q_scale']:.2f}  "
                      f"R_scale={ep_result['mean_R_scale']:.2f}")

            ctrl.stop()

            # Aggregate across episodes
            results[scenario_name][mode_name] = {
                'tracking_rmse_rad':  float(np.mean([e['tracking_rmse_rad']  for e in episodes_data])),
                'torque_smoothness':  float(np.mean([e['torque_smoothness']  for e in episodes_data])),
                'task_success_rate':  float(np.mean([float(e['success'])     for e in episodes_data])),
                'mean_time_to_reach': float(np.mean([e['time_to_reach']      for e in episodes_data
                                                     if e['time_to_reach'] >= 0] or [-1])),
                'mean_Q_scale':       float(np.mean([e['mean_Q_scale']       for e in episodes_data])),
                'mean_R_scale':       float(np.mean([e['mean_R_scale']       for e in episodes_data])),
                'mean_fusion_ms':     float(np.mean([e['mean_fusion_ms']     for e in episodes_data])),
                'episodes':           episodes_data,
            }

    env.close()

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n\n══ ABLATION SUMMARY ══")
    for scenario_name in SCENARIOS:
        print(f"\nScenario {scenario_name}:")
        print(f"  {'Mode':<16} {'RMSE':>8} {'Smooth':>8} {'Success':>8} {'Q_scale':>8} {'R_scale':>8}")
        for mode_name in FUSION_MODES:
            r = results[scenario_name][mode_name]
            print(f"  {mode_name:<16} "
                  f"{r['tracking_rmse_rad']:8.4f} "
                  f"{r['torque_smoothness']:8.3f} "
                  f"{r['task_success_rate']:8.2f} "
                  f"{r['mean_Q_scale']:8.2f} "
                  f"{r['mean_R_scale']:8.2f}")

    # Per-scenario winner
    winners = {}
    for scenario_name in SCENARIOS:
        # Winner = lowest RMSE
        best_mode = min(FUSION_MODES.keys(),
                        key=lambda m: results[scenario_name][m]['tracking_rmse_rad'])
        winners[scenario_name] = {
            'best_mode': best_mode,
            'best_rmse': results[scenario_name][best_mode]['tracking_rmse_rad'],
            'worst_mode': max(FUSION_MODES.keys(),
                             key=lambda m: results[scenario_name][m]['tracking_rmse_rad']),
        }
        print(f"\nScenario {scenario_name} winner: {best_mode} "
              f"(RMSE={winners[scenario_name]['best_rmse']:.4f})")

    # Overall winner (lowest mean RMSE across all scenarios)
    mode_mean_rmse = {}
    for mode_name in FUSION_MODES:
        mode_mean_rmse[mode_name] = float(np.mean(
            [results[s][mode_name]['tracking_rmse_rad'] for s in SCENARIOS]
        ))
    overall_winner = min(mode_mean_rmse, key=mode_mean_rmse.get)
    print(f"\nOverall winner: {overall_winner} "
          f"(mean RMSE = {mode_mean_rmse[overall_winner]:.4f})")

    # ── Save JSON ──────────────────────────────────────────────────────────
    output = {
        "benchmark_id": "B4_sensor_ablation",
        "timestamp":    datetime.datetime.now().isoformat(),
        "config": {
            "n_episodes_per_cell": N_EPISODES,
            "scenarios":           list(SCENARIOS.keys()),
            "fusion_modes":        list(FUSION_MODES.keys()),
            "vla_mode":            "MOCK",
            "solver":              "osqp",
        },
        "environment": {
            "python":   sys.version.split()[0],
            "platform": platform.platform(),
        },
        "scenarios": results,
        "winners":   winners,
        "overall_winner": overall_winner,
        "mode_mean_rmse": mode_mean_rmse,
    }

    os.makedirs("evaluation/results", exist_ok=True)
    path = f"evaluation/results/B4_sensor_ablation_{int(time.time())}.json"
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nB4 saved: {path}")

if __name__ == "__main__":
    main()
```

---

## EXPECTED B4 RESULTS (document honestly)

The benchmark produces a **5×3 matrix** (mode × scenario). The expected pattern, if fusion cost shaping is working:

| Mode | Scenario A (precision) | Scenario B (dynamic) | Scenario C (limits) | Hypothesis |
|------|------------------------|----------------------|---------------------|------------|
| M0 RGB | baseline | baseline | baseline | No adaptation → static Q/R |
| M1 Events | ≈M0 | **lower RMSE** | ≈M0 | Motion score → smoother R |
| M2 LiDAR | **lower RMSE** | ≈M0 | ≈M0 | Proximity → tighter Q |
| M3 Proprio | ≈M0 | ≈M0 | **lower RMSE** | Limit score → safer R |
| M4 Full | **best** | **best** | **best** | All signals combined |

**If M4 doesn't win every scenario**: that's still a valid scientific result. It means either:
- The alpha gains need tuning (document recommended values)
- The simulated LiDAR/events don't produce strong enough signals for those scenarios
- The MuJoCo dynamics are simple enough that cost shaping has small marginal effect

**Do not fake or curate results.** Report what actually happened and explain it.

---

## NEW TEST: `tests/integration/test_fusion_mpc_pipeline.py`

```python
"""Integration test: fusion encoder → cost adapter → MPC → meaningful torque change."""
import os; os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, pytest, yaml

@pytest.fixture(scope='module')
def config():
    with open("config/robots/xarm_6dof.yaml") as f:
        return yaml.safe_load(f)

def test_m4_different_torques_than_m0_on_active_scene(config):
    """
    Full pipeline test: M4 fusion must produce different torques than M0
    when given a scene with high-gradient content (LiDAR/events active).
    """
    from src.fusion.real_fusion_encoder import RealFusionEncoder
    from src.fusion.cost_adapter import FusionCostAdapter
    from src.mpc.xarm_mpc_controller import XArmMPCController
    from src.solver.osqp_solver import OSQPSolver

    # High-gradient image = LiDAR proximity score will be non-zero
    rng = np.random.default_rng(7)
    # Checkerboard-like pattern → lots of edges
    img  = np.zeros((84, 84, 3), dtype=np.uint8)
    img[::8, :] = 200; img[:, ::8] = 200   # grid pattern

    obs = {'rgb': img,
           'prev_rgb': rng.integers(50, 150, (84,84,3), dtype=np.uint8),
           'state': np.array([2.5, 1.5, -2.0, 1.8, -2.5, 2.0, 0.])}

    adapter = FusionCostAdapter(alpha_proximity=3.0, alpha_motion=2.0, alpha_limit=2.0)
    q   = np.zeros(6); qdot = np.zeros(6)
    ref = np.zeros((10, 6)); ref[:, 0] = 0.5

    # M0: RGB only → near-neutral scales
    f_m0 = RealFusionEncoder.mode_rgb_only().encode(obs)
    s_m0 = adapter.adapt(f_m0, {'rgb':True,'events':False,'lidar':False,'proprio':False})
    mpc  = XArmMPCController(OSQPSolver(), config)
    tau_m0 = mpc.step((q, qdot), ref, cost_scales=s_m0)

    # M4: full fusion → events + lidar + proprio all active
    f_m4 = RealFusionEncoder.mode_full().encode(obs)
    s_m4 = adapter.adapt(f_m4, {'rgb':True,'events':True,'lidar':True,'proprio':True})
    tau_m4 = mpc.step((q, qdot), ref, cost_scales=s_m4)

    print(f"\nM0 scales: Q={s_m0.Q_scale:.2f} R={s_m0.R_scale:.2f}")
    print(f"M4 scales: Q={s_m4.Q_scale:.2f} R={s_m4.R_scale:.2f}")
    print(f"M0 tau[:3]: {tau_m0[:3].round(3)}")
    print(f"M4 tau[:3]: {tau_m4[:3].round(3)}")

    # Scales must differ (because M4 has extra active modalities)
    assert s_m4.Q_scale != s_m0.Q_scale or s_m4.R_scale != s_m0.R_scale, \
        "M4 and M0 must produce different cost scales"

    # Torques must differ (different Q/R → different QP solution)
    assert not np.allclose(tau_m0, tau_m4, atol=1e-4), \
        "Different cost scales must yield different torques"
```

**Phase 7 Gate (updated)**: `pytest tests/integration/test_dual_controller.py tests/integration/test_fusion_mpc_pipeline.py -v` — ALL PASS

---

## SUMMARY: THE COMPLETE FUSION PIPELINE (for thesis documentation)

```
Observation at step t
        │
        ├─ rgb [H,W,3]  ──────────────────────────────────────────────────┐
        │                                                                  │
        ├─ prev_rgb [H,W,3]  ──────────────────────────────────────────┐  │
        │                                                               │  │
        └─ state [7]  ────────────────────────────────────────────┐    │  │
                                                                   │    │  │
                                         ┌─────────────────────────────────────────┐
                                         │        RealFusionEncoder (M0–M4)        │
                                         │                                          │
                                         │  extract_rgb(rgb)      → 128-dim        │
                                         │  extract_events(rgb, prev_rgb) → 96-dim │
                                         │  extract_lidar(rgb)     → 64-dim        │
                                         │  extract_proprio(state) → 32-dim        │
                                         │                                          │
                                         │  concatenate → features [D]             │
                                         └──────────────┬──────────────────────────┘
                                                        │
                                         ┌──────────────▼──────────────────────────┐
                                         │         FusionCostAdapter               │
                                         │                                          │
                                         │  motion_score    ← events cells         │
                                         │  proximity_score ← lidar high bins      │
                                         │  scene_score     ← rgb gradients        │
                                         │  limit_scores    ← proprio q_norm       │
                                         │                                          │
                                         │  Q_scale, Q_joint_diag  (tighten pos)  │
                                         │  R_scale, R_joint_diag  (smooth τ)     │
                                         └──────────────┬──────────────────────────┘
                                                        │  CostScales
                              ┌─────────────────────────▼──────────────────────────┐
                              │           XArmMPCController.step()                  │
                              │                                                      │
                              │  Q_eff = Q_base × Q_scale × diag(Q_joint_diag)     │
                              │  R_eff = R_base × R_scale × diag(R_joint_diag)     │
                              │                                                      │
                              │  Build QP(Q_eff, R_eff, dynamics, ref)             │
                              │  Solve QP → τ* (SL or OSQP)                        │
                              └──────────────────────┬───────────────────────────────┘
                                                     │  τ* [8]
                                                     ▼
                                            XArmEnv.step(τ*)
                                                     │
                                           obs at step t+1


Separate VLA path (unchanged API, runs async):
  rgb [H,W,3] + state [6] + instruction
        → VLA server → action_chunk [10,7]
        → TrajectoryBuffer (reference for MPC)
```
