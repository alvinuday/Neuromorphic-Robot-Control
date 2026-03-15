# PHASES 7–9 — DUAL CONTROLLER, DATASET, BENCHMARKS
**Requires**: Phases 1–6 complete and all their gates PASSED.

---

## PHASE 7 — DUAL SYSTEM CONTROLLER + TRAJECTORY BUFFER

**Goal**: `DualSystemController` runs >50 Hz (with MockVLA) without deadlocks.

### 7.1 — `src/integration/trajectory_buffer.py`

```python
import threading, time
import numpy as np

class TrajectoryBuffer:
    """
    Thread-safe buffer for VLA action chunks.
    
    VLA thread writes.  MPC thread reads.  Never blocks MPC.
    """
    def __init__(self, horizon: int = 10, action_dim: int = 6):
        self._horizon    = horizon
        self._action_dim = action_dim
        self._traj       = np.zeros((horizon, action_dim), dtype=np.float64)
        self._lock       = threading.Lock()
        self._update_count     = 0
        self._last_update_time = None

    def update(self, action_chunk: np.ndarray) -> None:
        """Called by VLA background thread. Replaces full trajectory."""
        chunk = np.asarray(action_chunk, dtype=np.float64)
        n = min(chunk.shape[0], self._horizon)
        with self._lock:
            self._traj[:n] = chunk[:n, :self._action_dim]
            if n < self._horizon:
                self._traj[n:] = chunk[-1:, :self._action_dim]
            self._update_count    += 1
            self._last_update_time = time.time()

    def get_reference(self) -> np.ndarray:
        """Called by MPC — non-blocking read."""
        with self._lock:
            return self._traj.copy()

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def staleness_s(self) -> float:
        if self._last_update_time is None:
            return float('inf')
        return time.time() - self._last_update_time
```

### 7.2 — `src/integration/dual_controller.py`

```python
import threading, time, logging
import numpy as np
from src.mpc.xarm_mpc_controller import XArmMPCController
from src.smolvla.vla_client import VLAClient
from src.fusion.real_fusion_encoder import RealFusionEncoder
from src.integration.trajectory_buffer import TrajectoryBuffer
from src.smolvla.action_processor import process_action_chunk

logger = logging.getLogger(__name__)

class DualSystemController:
    """
    System 1 (MPC, synchronous) + System 2 (VLA, background thread).
    
    Control loop calls .step() at 100 Hz.
    VLA background thread queries every vla_interval seconds.
    """

    def __init__(
        self,
        mpc:            XArmMPCController,
        vla_client:     VLAClient,
        fusion_encoder: RealFusionEncoder,
        instruction:    str  = "pick up the red block",
        vla_interval:   float = 0.2,     # 5 Hz
        vla_timeout:    float = 5.0,
    ):
        self.mpc      = mpc
        self.vla      = vla_client
        self.fusion   = fusion_encoder
        self.instruction  = instruction
        self.vla_interval = vla_interval
        self.vla_timeout  = vla_timeout

        self.buffer = TrajectoryBuffer(horizon=10, action_dim=6)

        # Shared state (written by step(), read by VLA thread)
        self._rgb_buf   = None
        self._state_buf = None
        self._buf_lock  = threading.Lock()

        # Stats
        self.vla_calls    = 0
        self.vla_success  = 0
        self.vla_fail     = 0
        self.mpc_times_ms = []

        self._running = False
        self._thread  = None

    # ── Public API ─────────────────────────────────────────────────────────

    def start(self):
        """Launch VLA background thread."""
        self._running = True
        self._thread  = threading.Thread(target=self._vla_loop, daemon=True)
        self._thread.start()
        logger.info("DualController started. VLA thread running.")

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("DualController stopped.")

    def step(self, obs: dict) -> tuple:
        """
        Main control step. SYNCHRONOUS. Call at 100 Hz.
        
        Args:
            obs: {'q': [6], 'qdot': [6], 'rgb': [H,W,3]}
        Returns:
            tau:  [8] torque command
            info: dict with timing and buffer stats
        """
        t0 = time.perf_counter()
        q, qdot, rgb = obs['q'], obs['qdot'], obs['rgb']

        # Encode sensors
        obs_aug = {
            'rgb':   rgb,
            'state': np.append(q, 0.),   # 7-dim: 6 joints + 1 gripper
        }
        if hasattr(self, '_prev_rgb') and self._prev_rgb is not None:
            obs_aug['prev_rgb'] = self._prev_rgb
        _ = self.fusion.encode(obs_aug)   # features available for future VLA feed-in
        self._prev_rgb = rgb.copy()

        # Update shared buffers for VLA thread
        with self._buf_lock:
            self._rgb_buf   = rgb.copy()
            self._state_buf = q.copy()

        # Get reference from trajectory buffer (non-blocking)
        ref = self.buffer.get_reference()   # [10, 6]

        # MPC solve
        tau = self.mpc.step((q, qdot), ref)

        ms = (time.perf_counter() - t0) * 1000.
        self.mpc_times_ms.append(ms)

        info = {
            'mpc_time_ms':     ms,
            'buffer_updates':  self.buffer.update_count,
            'buffer_staleness_s': self.buffer.staleness_s,
            'vla_success':     self.vla_success,
            'vla_fail':        self.vla_fail,
        }
        return tau, info

    def get_stats(self) -> dict:
        mpc_ms = self.mpc_times_ms or [0.]
        return {
            'mean_mpc_ms':    float(np.mean(mpc_ms)),
            'max_mpc_ms':     float(np.max(mpc_ms)),
            'vla_calls':      self.vla_calls,
            'vla_success':    self.vla_success,
            'vla_fail':       self.vla_fail,
            'buffer_updates': self.buffer.update_count,
        }

    # ── VLA background thread ───────────────────────────────────────────────

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
            result = self.vla.predict(rgb, state, self.instruction)
            action_7 = np.array(result['action'], dtype=np.float64)
            chunk = process_action_chunk(action_7, horizon=10)  # [10, 6]
            self.buffer.update(chunk)
            self.vla_success += 1
        except Exception as e:
            self.vla_fail += 1
            logger.warning(f"VLA query failed: {e}")
```

### 7.3 — `src/smolvla/action_processor.py`

```python
import numpy as np

def process_action_chunk(action_7d: np.ndarray, horizon: int = 10) -> np.ndarray:
    """
    Convert 7-D VLA action (6 joints + 1 gripper) to [horizon, 6] trajectory.
    
    The action is treated as a single waypoint; we tile it to fill the horizon.
    Future: support multi-step action chunks from VLA.
    """
    a = np.asarray(action_7d, dtype=np.float64)
    arm_action = a[:6]   # drop gripper
    return np.tile(arm_action, (horizon, 1))   # [horizon, 6]
```

### 7.4 — Tests for Phase 7

`tests/integration/test_dual_controller.py`:

```python
import os; os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, pytest, time, yaml
from src.mpc.xarm_mpc_controller import XArmMPCController
from src.solver.osqp_solver import OSQPSolver
from src.smolvla.vla_client import VLAClient
from src.fusion.real_fusion_encoder import RealFusionEncoder
from src.integration.dual_controller import DualSystemController
from src.integration.trajectory_buffer import TrajectoryBuffer

def _make_obs(seed=0):
    rng = np.random.default_rng(seed)
    return {
        'q':    np.zeros(6),
        'qdot': np.zeros(6),
        'rgb':  rng.integers(0, 200, (84, 84, 3), dtype=np.uint8),
    }

@pytest.fixture(scope='module')
def controller():
    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)
    mpc = XArmMPCController(OSQPSolver(), cfg)
    vla = VLAClient(mock_mode=True)
    enc = RealFusionEncoder.mode_rgb_only()
    ctrl = DualSystemController(mpc, vla, enc)
    ctrl.start()
    yield ctrl
    ctrl.stop()

def test_single_step(controller):
    tau, info = controller.step(_make_obs())
    assert tau.shape == (8,)
    assert 'mpc_time_ms' in info

def test_10_steps_no_crash(controller):
    for i in range(10):
        tau, info = controller.step(_make_obs(i))
        assert tau.shape == (8,)

def test_control_rate_achievable(controller):
    """50 steps must complete in < 10s (i.e. >5 Hz minimum)."""
    t0 = time.perf_counter()
    for i in range(50):
        controller.step(_make_obs(i))
    elapsed = time.perf_counter() - t0
    assert elapsed < 10.0, f"50 steps took {elapsed:.1f}s — too slow"

def test_vla_thread_updates_buffer(controller):
    """After waiting 0.5s, VLA thread should have updated the buffer."""
    time.sleep(0.5)
    assert controller.buffer.update_count > 0, "VLA thread never updated buffer"

def test_trajectory_buffer_thread_safety():
    buf = TrajectoryBuffer(horizon=5, action_dim=6)
    import threading
    errors = []
    def writer():
        for _ in range(100):
            try:
                buf.update(np.random.randn(5, 6))
            except Exception as e:
                errors.append(str(e))
    def reader():
        for _ in range(100):
            try:
                _ = buf.get_reference()
            except Exception as e:
                errors.append(str(e))
    threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert not errors, f"Thread-safety violations: {errors}"
```

**Phase 7 Gate**: `pytest tests/integration/test_dual_controller.py -v` — ALL PASS

---

## PHASE 8 — LEROBOT DATASET + GIF RECORDER

**Goal**: Load real episodes, replay in MuJoCo, record GIFs with visible motion.

### 8.1 — `data/loaders/lerobot_loader.py`

```python
import numpy as np
from pathlib import Path

class DatasetNotAvailableError(Exception):
    pass

class LeRobotDatasetLoader:
    """
    Load lerobot/utokyo_xarm_pick_and_place dataset.
    
    Dataset schema (verified against HF dataset card):
        observation.state:  [7]  — 6 arm joints (rad) + 1 gripper (0=open)
        action:             [7]  — target joint positions (position control)
        episode_index:      int
        frame_index:        int
    
    NOTE: Dataset uses position control (actions = target joint angles).
    Our MPC uses torque control.
    Evaluation methodology: use dataset states as MPC reference trajectories.
    """
    DATASET_REPO = "lerobot/utokyo_xarm_pick_and_place"

    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = cache_dir
        self._dataset  = None
        self._load()

    def _load(self):
        try:
            from datasets import load_dataset
            self._dataset = load_dataset(
                self.DATASET_REPO,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
                split="train",
            )
        except Exception as e:
            raise DatasetNotAvailableError(
                f"Cannot load {self.DATASET_REPO}: {e}\n"
                "Run with internet access or skip dataset tests."
            ) from e

    def get_info(self) -> dict:
        episodes = sorted(set(self._dataset['episode_index']))
        return {
            'n_episodes':   len(episodes),
            'n_steps_total': len(self._dataset),
            'action_dim':    7,
            'state_dim':     7,
        }

    def load_episode(self, episode_idx: int) -> dict:
        ep = self._dataset.filter(lambda x: x['episode_index'] == episode_idx)
        states  = np.array(ep['observation.state'], dtype=np.float32)   # [T, 7]
        actions = np.array(ep['action'],             dtype=np.float32)   # [T, 7]
        return {'states': states, 'actions': actions, 'n_steps': len(states)}
```

### 8.2 — `src/visualization/episode_recorder.py`

```python
import numpy as np
from pathlib import Path
from contextlib import contextmanager

class EpisodeRecorder:
    """Record MuJoCo RGB frames to GIF using imageio."""

    def __init__(self, fps: int = 10, resize: tuple = (320, 240)):
        self.fps    = fps
        self.resize = resize
        self._frames = []

    def start(self):
        self._frames = []

    def add_frame(self, frame: np.ndarray):
        """Add [H, W, 3] uint8 frame."""
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)
        # Resize if needed
        if frame.shape[:2] != (self.resize[1], self.resize[0]):
            from PIL import Image
            pil = Image.fromarray(frame).resize(self.resize)
            frame = np.array(pil)
        self._frames.append(frame)

    def save(self, path: str) -> str:
        import imageio.v3 as iio
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(path, self._frames, loop=0, fps=self.fps)
        size_kb = Path(path).stat().st_size / 1024
        print(f"GIF saved: {path}  ({len(self._frames)} frames, {size_kb:.1f} KB)")
        return path

    @contextmanager
    def recording(self, path: str):
        self.start()
        try:
            yield self
        finally:
            self.save(path)
```

### 8.3 — Tests for Phase 8

`tests/integration/test_dataset_and_gif.py`:

```python
import os; os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, pytest
from pathlib import Path

# ── Dataset tests (skip if unavailable) ──────────────────────────────────────

try:
    from data.loaders.lerobot_loader import LeRobotDatasetLoader, DatasetNotAvailableError
    _loader = LeRobotDatasetLoader()
    DATASET_AVAILABLE = True
except Exception:
    DATASET_AVAILABLE = False

@pytest.mark.skipif(not DATASET_AVAILABLE, reason="LeRobot dataset not available")
def test_dataset_episode_shapes():
    ep = _loader.load_episode(0)
    assert ep['states'].shape[1] == 7
    assert ep['actions'].shape[1] == 7
    assert ep['n_steps'] > 10

@pytest.mark.skipif(not DATASET_AVAILABLE, reason="LeRobot dataset not available")
def test_dataset_info():
    info = _loader.get_info()
    assert info['n_episodes'] > 10
    assert info['action_dim'] == 7

# ── GIF recorder tests ───────────────────────────────────────────────────────

def test_gif_recorder_saves_file(tmp_path):
    from src.visualization.episode_recorder import EpisodeRecorder
    rec = EpisodeRecorder(fps=10)
    rec.start()
    for _ in range(10):
        frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
        rec.add_frame(frame)
    path = rec.save(str(tmp_path / "test.gif"))
    assert Path(path).exists()
    assert Path(path).stat().st_size > 5_000, "GIF file too small — probably empty"

def test_gif_context_manager(tmp_path):
    from src.visualization.episode_recorder import EpisodeRecorder
    from src.simulation.envs.xarm_env import XArmEnv
    rec = EpisodeRecorder()
    env = XArmEnv(render_mode='offscreen')
    obs = env.reset()
    gif_path = str(tmp_path / "env_test.gif")
    with rec.recording(gif_path):
        tau = np.zeros(8); tau[0] = 3.0
        for _ in range(20):
            obs, _, _, _ = env.step(tau)
            rec.add_frame(obs['rgb'])
    env.close()
    assert Path(gif_path).stat().st_size > 10_000
```

**Phase 8 Gate**: GIF tests PASS (no skip). Dataset tests PASS or SKIP (never FAIL if dataset unavailable).

---

## PHASE 9 — BENCHMARKS B1–B5

**Goal**: Five benchmark scripts run end-to-end and produce real JSON results. No placeholder numbers.

### JSON output schema (ALL benchmarks must follow this)

```python
import json, datetime, platform, sys

def make_result_header(benchmark_id: str, config: dict) -> dict:
    return {
        "benchmark_id": benchmark_id,
        "timestamp":    datetime.datetime.now().isoformat(),
        "config":       config,
        "environment": {
            "python":   sys.version.split()[0],
            "platform": platform.platform(),
            "mujoco":   _mujoco_version(),
            "vla_mode": config.get("vla_mode", "MOCK"),
        }
    }

def _mujoco_version():
    try:
        import mujoco; return mujoco.__version__
    except: return "unknown"
```

### B1 — `evaluation/benchmarks/run_mpc_solo.py`

```
Setup:    XArmEnv + MPC (no VLA). Reference = sinusoidal trajectory.
Episodes: 5   Steps: 200 each
Metrics:  per-joint tracking RMSE, mean/std solve time
Solvers:  run once with OSQP, once with SL (separately)
```

```python
# Minimal skeleton — agent must fill in full implementation
import json, time, numpy as np, yaml, os
os.environ.setdefault('MUJOCO_GL', 'osmesa')

from src.simulation.envs.xarm_env import XArmEnv
from src.mpc.xarm_mpc_controller import XArmMPCController
from src.solver.osqp_solver import OSQPSolver

def sinusoidal_ref(t, n_joints=6, freq=0.5, amp=0.3):
    """Generate sinusoidal reference trajectory [N, 6]."""
    steps = np.arange(200)
    refs  = amp * np.sin(2 * np.pi * freq * (steps + t) * 0.01)
    return np.tile(refs[:, None], (1, n_joints))   # [200, 6]

def run_episode(env, mpc, episode_idx):
    obs   = env.reset()
    rmses = []
    solve_times = []
    ref   = sinusoidal_ref(episode_idx * 200)
    for step in range(200):
        q_ref = ref[step]
        tau   = mpc.step((obs['q'], obs['qdot']), ref[step:])
        obs, _, done, _ = env.step(tau)
        rmse  = float(np.sqrt(np.mean((obs['q'] - q_ref)**2)))
        rmses.append(rmse)
        info  = mpc.get_last_qp_matrices()
        if 'info' in info:
            solve_times.append(info['info'].get('solve_time_ms', 0))
        if done: break
    return {'rmse': float(np.mean(rmses)), 'solve_ms': float(np.mean(solve_times or [0]))}

def main():
    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)
    
    env  = XArmEnv(render_mode='offscreen')
    mpc  = XArmMPCController(OSQPSolver(), cfg)
    
    episodes = [run_episode(env, mpc, i) for i in range(5)]
    env.close()
    
    result = {
        **make_result_header("B1_mpc_solo_osqp", {"solver": "osqp", "n_episodes": 5}),
        "results": {
            "episodes": episodes,
            "mean_rmse":     float(np.mean([e['rmse']     for e in episodes])),
            "mean_solve_ms": float(np.mean([e['solve_ms'] for e in episodes])),
        }
    }
    
    import os
    os.makedirs("evaluation/results", exist_ok=True)
    path = f"evaluation/results/B1_mpc_solo_osqp_{int(time.time())}.json"
    with open(path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"B1 complete. RMSE={result['results']['mean_rmse']:.4f}")
    print(f"Saved: {path}")

if __name__ == "__main__":
    main()
```

### B2 — `evaluation/benchmarks/run_vla_solo.py`

```
Setup:    XArmEnv + MockVLA only (execute VLA actions without MPC).
Episodes: 5   Steps: 100 each
Metrics:  joint range covered, action smoothness (diff between consecutive actions)
Note:     All results labeled "MOCK" in JSON
```

### B3 — `evaluation/benchmarks/run_dual_system.py`

```
Setup:    XArmEnv + DualSystemController (OSQP-MPC + MockVLA)
Episodes: 5   Steps: 200 each
Metrics:  MPC tracking RMSE, VLA query rate achieved, buffer staleness stats
```

### B4 — `evaluation/benchmarks/sensor_ablation.py`

```
For each mode in {M0, M1, M2, M3, M4}:
    Run 3 episodes with DualSystemController
    Measure: fusion latency (ms), feature norm, MPC tracking RMSE
    
Output includes comparison table.
Expected winner: M4 should have lowest RMSE (more state info → better tracking).
```

Critical: each mode runs REAL encoding and REAL MPC. If feature_dim is different for each mode, the results will differ. That's expected and correct.

### B5 — `evaluation/benchmarks/solver_comparison.py` ← THESIS CORE

This is the most important benchmark. Must be rigorous.

```python
"""
Compares SL solver vs OSQP solver on a suite of 50 QPs.

QP Suite design:
    - Small (n=6, m=6):    MPC-sized problems (most common case)
    - Medium (n=12, m=12): Extended horizon
    - Ill-conditioned:     κ(P) = 100–1000 (challenging for OSQP)
    - Well-conditioned:    κ(P) = 1–10

For each QP:
    - Run OSQP 10 times, record mean/std time + objective value
    - Run SL 10 times (or 3 if time-constrained), record same
    - OSQP solution = reference; SL objective error = |obj_SL - obj_OSQP| / |obj_OSQP|

Expected result (be honest about this in the report):
    OSQP: ~5–50ms, near-perfect accuracy
    SL:   ~2000–8000ms, 0.1–5% objective error (depends on conditioning)
    
Thesis claim: SL is more robust on ill-conditioned problems.
Test this claim explicitly.
"""

import numpy as np, json, time, os, yaml
from src.solver.osqp_solver import OSQPSolver
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect

def generate_qp_suite():
    """50 QPs of varying size and conditioning."""
    rng   = np.random.default_rng(42)  # fixed seed for reproducibility
    suite = []
    
    for _ in range(25):  # well-conditioned, n=6
        n = 6
        U = np.linalg.qr(rng.standard_normal((n,n)))[0]
        D = np.diag(rng.uniform(1, 10, n))
        P = U @ D @ U.T; P = 0.5*(P+P.T) + n*np.eye(n)
        q = rng.standard_normal(n)
        A = np.eye(n)
        l = -5*np.ones(n); u = 5*np.ones(n)
        suite.append({'P': P, 'q': q, 'A': A, 'l': l, 'u': u,
                      'type': 'well_cond_n6', 'kappa': float(D.diagonal().max()/D.diagonal().min())})
    
    for _ in range(25):  # ill-conditioned, n=6
        n = 6
        U = np.linalg.qr(rng.standard_normal((n,n)))[0]
        D = np.diag(np.logspace(0, 3, n))   # κ = 1000
        P = U @ D @ U.T; P = 0.5*(P+P.T)
        q = rng.standard_normal(n)
        A = np.eye(n); l = -5*np.ones(n); u = 5*np.ones(n)
        suite.append({'P': P, 'q': q, 'A': A, 'l': l, 'u': u,
                      'type': 'ill_cond_n6', 'kappa': float(D.diagonal().max()/D.diagonal().min())})
    return suite

def run_solver_on_suite(solver, suite, n_runs=3):
    results = []
    for qp in suite:
        times = []; objs = []; viols = []
        for _ in range(n_runs):
            x, info = solver.solve(qp['P'], qp['q'], qp['A'], qp['l'], qp['u'])
            times.append(info['solve_time_ms'])
            objs.append(info['obj_val'])
            viols.append(info['constraint_viol'])
        results.append({
            'type':     qp['type'],
            'kappa':    qp['kappa'],
            'mean_ms':  float(np.mean(times)),
            'std_ms':   float(np.std(times)),
            'mean_obj': float(np.mean(objs)),
            'mean_viol':float(np.mean(viols)),
        })
    return results

def main():
    suite   = generate_qp_suite()
    osqp    = OSQPSolver()
    sl      = StuartLandauLagrangeDirect(T_solve=3.0)
    
    print(f"Running OSQP on {len(suite)} QPs...")
    osqp_res = run_solver_on_suite(osqp, suite, n_runs=5)
    
    print(f"Running SL on {len(suite)} QPs (this is SLOW — expected ~{len(suite)*3*3:.0f}s)...")
    sl_res = run_solver_on_suite(sl, suite, n_runs=3)
    
    # Compute objective error: (sl_obj - osqp_obj) / |osqp_obj|
    for i, (s, o) in enumerate(zip(sl_res, osqp_res)):
        denom = abs(o['mean_obj']) + 1e-10
        s['obj_error_pct'] = float(abs(s['mean_obj'] - o['mean_obj']) / denom * 100)
    
    # Summary tables by type
    for qtype in ('well_cond_n6', 'ill_cond_n6'):
        sl_sub   = [r for r in sl_res   if r['type'] == qtype]
        osqp_sub = [r for r in osqp_res if r['type'] == qtype]
        print(f"\n── {qtype} ──")
        print(f"  OSQP: {np.mean([r['mean_ms'] for r in osqp_sub]):.1f}ms (mean)")
        print(f"  SL:   {np.mean([r['mean_ms'] for r in sl_sub]):.1f}ms (mean)")
        print(f"  SL obj error: {np.mean([r['obj_error_pct'] for r in sl_sub]):.2f}%")
    
    result = {
        "benchmark_id": "B5_solver_comparison",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_qps": len(suite),
        "osqp": osqp_res,
        "sl":   sl_res,
        "summary": {
            "osqp_mean_ms_well": float(np.mean([r['mean_ms'] for r in osqp_res if 'well' in r['type']])),
            "sl_mean_ms_well":   float(np.mean([r['mean_ms'] for r in sl_res   if 'well' in r['type']])),
            "osqp_mean_ms_ill":  float(np.mean([r['mean_ms'] for r in osqp_res if 'ill'  in r['type']])),
            "sl_mean_ms_ill":    float(np.mean([r['mean_ms'] for r in sl_res   if 'ill'  in r['type']])),
            "sl_mean_obj_error_pct": float(np.mean([r['obj_error_pct'] for r in sl_res])),
        }
    }
    os.makedirs("evaluation/results", exist_ok=True)
    path = f"evaluation/results/B5_solver_comparison_{int(time.time())}.json"
    with open(path, 'w') as f:
        json.dump(result, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nB5 saved: {path}")
    print("Summary:", json.dumps(result['summary'], indent=2))

if __name__ == "__main__":
    main()
```

**Phase 9 Gate**: Each benchmark script exits with code 0 AND produces a JSON file in `evaluation/results/` with non-zero, non-identical metric values.

*Next: Read `05_PHASES10_12_WEBAPP_E2E.md`*
