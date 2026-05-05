#!/usr/bin/env python3
"""
Webapp backend for the Neuromorphic Robot Control system.

Serves:
  GET  /              → existing QP inspector (static/index.html) — UNCHANGED
  GET  /mujoco        → MuJoCo visualization page (viz/mujoco_viz.html)
  GET  /dashboard     → Benchmark dashboard (viz/dashboard.html)
  POST /api/solve_qp  → Run QP with chosen solver, return matrices + solution
  GET  /api/results   → All benchmark JSONs from evaluation/results/
  POST /api/run_episode → Run one sim episode, return trajectory + GIF path

Usage:
  python webapp/server.py
  # Serves at http://localhost:3000
"""
import json
import glob
import os
import time
import uuid
import asyncio
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
from pydantic import BaseModel

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Clean environment for MuJoCo (unset on macOS)
if 'MUJOCO_GL' in os.environ:
    del os.environ['MUJOCO_GL']

app = FastAPI(title="Neuromorphic Robot Control")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

WEBAPP_DIR = Path(__file__).parent
STATIC_DIR = WEBAPP_DIR / "static"
VIZ_DIR = WEBAPP_DIR / "viz"
RESULTS_DIR = Path("evaluation/results")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

ROOT_DIR = WEBAPP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def _np(obj):
    """Recursively convert numpy types to JSON-safe Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float64, np.float32)):
        v = float(obj)
        if np.isnan(v):
            return None
        if np.isinf(v):
            return 1e30 if v > 0 else -1e30
        return v
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np(v) for v in obj]
    return obj


def _parse_config(data: dict):
    """Extract arm/mpc/qp config from a JSON request body."""
    from src.dynamics.arm2dof import Arm2DOF
    from src.mpc.qp_builder import MPCBuilder

    params = data.get('params', {})
    m1 = float(params.get('m1', 1.0))
    m2 = float(params.get('m2', 1.0))
    l1 = float(params.get('l1', 0.5))
    l2 = float(params.get('l2', 0.5))
    g = float(params.get('g', 9.81))

    N = int(data.get('N', 1))
    dt = float(data.get('dt', 0.1))

    x0 = np.array(data.get('x0', [0, 0, 0, 0]), dtype=float)
    x_goal = np.array(data.get('x_goal', [np.pi / 3, np.pi / 6, 0, 0]), dtype=float)

    weights = data.get('weights', {})
    Qx = np.diag(weights['Qx']) if 'Qx' in weights else None
    Qf = np.diag(weights['Qf']) if 'Qf' in weights else None
    R = np.diag(weights['R']) if 'R' in weights else None
    Qs = float(weights['Qs']) if 'Qs' in weights else None

    bnd = data.get('bounds', {})
    tau_max = float(bnd.get('tau_max', 50.0))
    tau_min = float(bnd.get('tau_min', -50.0))
    theta_max_deg = float(bnd.get('theta_max', 180.0))
    theta_min_deg = float(bnd.get('theta_min', -180.0))
    theta_max = np.array([np.deg2rad(theta_max_deg)] * 2)
    theta_min = np.array([np.deg2rad(theta_min_deg)] * 2)

    bounds = {
        'theta_min': theta_min,
        'theta_max': theta_max,
        'tau_min': np.array([tau_min, tau_min]),
        'tau_max': np.array([tau_max, tau_max]),
    }

    arm = Arm2DOF(m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    mpc = MPCBuilder(arm, N=N, dt=dt, Qx=Qx, Qf=Qf, R=R, Qs=Qs, bounds=bounds)
    return arm, mpc, x0, x_goal, N, dt


def _z_labels(N: int, nx: int = 4, nu: int = 2, nq: int = 2):
    """Generate human-readable labels for every element of z."""
    state_names = ['q1', 'q2', 'dq1', 'dq2']
    ctrl_names = ['tau1', 'tau2']
    labels = []
    for k in range(N):
        for s in state_names:
            labels.append(f'x{k}[{s}]')
        for c in ctrl_names:
            labels.append(f'u{k}[{c}]')
    for s in state_names:
        labels.append(f'x{N}[{s}]')
    for k in range(N + 1):
        for j in range(nq):
            labels.append(f's{k}[{j}]')
    return labels


def _decompose_z(z_star: np.ndarray, N: int, nx: int = 4, nu: int = 2, nq: int = 2):
    """Split z* into per-step x_k, u_k and slacks."""
    x_steps = []
    u_steps = []
    for k in range(N):
        start = k * (nx + nu)
        x_steps.append(z_star[start:start + nx].tolist())
        u_steps.append(z_star[start + nx:start + nx + nu].tolist())
    start_xN = N * (nx + nu)
    x_steps.append(z_star[start_xN:start_xN + nx].tolist())
    n_z = N * (nx + nu) + nx
    slacks = z_star[n_z:].tolist()
    return {'x_k': x_steps, 'u_k': u_steps, 'slacks': slacks}

# Serve existing QP inspector (NEVER modify files in STATIC_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
# Serve new viz pages
app.mount("/viz", StaticFiles(directory=VIZ_DIR), name="viz")

# ── Page routes ────────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    """Serve the QP inspector (existing, unchanged)."""
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/mujoco")
async def mujoco_page():
    """Serve the MuJoCo visualization page."""
    return FileResponse(VIZ_DIR / "mujoco_viz.html")

@app.get("/dashboard")
async def dashboard_page():
    """Serve the benchmark dashboard page."""
    return FileResponse(VIZ_DIR / "dashboard.html")


@app.get("/api/cases")
async def api_cases():
    """Return canonical MPC inspector presets for dropdown selection."""
    from src.utils.qp_inspector import CANONICAL_CASES

    cases = []
    for name, cfg in CANONICAL_CASES.items():
        cases.append({
            'name': name,
            'description': cfg.get('description', ''),
            'g': cfg.get('g', 9.81),
            'q0': cfg.get('q0', [0, 0]),
            'dq0': cfg.get('dq0', [0, 0]),
            'x_goal': _np(cfg.get('x_goal', [np.pi / 3, np.pi / 6, 0, 0])),
            'N': cfg.get('N', 1),
            'dt': cfg.get('dt', 0.1),
            'tau_max': cfg.get('tau_max', 50.0),
        })
    return {'cases': _np(cases)}


@app.post("/api/build")
async def api_build(data: dict):
    """Build 2DOF MPC QP matrices and linearisation details for frontend inspector."""
    try:
        arm, mpc, x0, x_goal, N, dt = _parse_config(data)
        ref_traj = mpc.build_reference_trajectory(x0, x_goal)
        qp_matrices = mpc.build_qp(x0, ref_traj)
        Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices

        nx, nu, nq = arm.nx, arm.nu, arm.nq
        n_z = N * (nx + nu) + nx
        n_slack = (N + 1) * nq
        n_z_total = n_z + n_slack

        # Linearisation details at k=0 for matrix inspector tabs.
        x_bar = ref_traj[0]
        q_bar = x_bar[:nq]
        th1, th2 = q_bar[0], q_bar[1]
        m1, m2, l1, l2, g = arm.m1, arm.m2, arm.l1, arm.l2, arm.g
        G1 = (m1 * l1 / 2 + m2 * l1) * g * np.cos(th1) + m2 * g * l2 / 2 * np.cos(th1 + th2)
        G2 = m2 * g * l2 / 2 * np.cos(th1 + th2)
        u_bar = np.array([G1, G2])

        f_val = np.array(mpc.f_fun(x_bar, u_bar)).flatten()
        Ac = np.array(mpc.A_fun(x_bar, u_bar))
        Bc = np.array(mpc.B_fun(x_bar, u_bar))
        Ad = np.eye(nx) + dt * Ac
        Bd = dt * Bc
        c_k = dt * (f_val - Ac @ x_bar - Bc @ u_bar)

        M_num = np.array(arm.M_fun(q_bar))
        try:
            M_inv = np.linalg.inv(M_num)
        except np.linalg.LinAlgError:
            M_inv = np.full_like(M_num, np.nan)

        G_vec = np.array([G1, G2])
        dG11 = (m1 * l1 / 2 + m2 * l1) * g * np.cos(th1) + m2 * l2 / 2 * g * np.cos(th1 + th2)
        dG12 = m2 * l2 / 2 * g * np.cos(th1 + th2)
        dG21 = dG12
        dG22 = dG12
        dG_dq = np.array([[dG11, dG12], [dG21, dG22]])

        z_labels = _z_labels(N, nx, nu, nq)
        fk_points = []
        for k in range(N + 1):
            pts = arm.forward_kinematics(ref_traj[k][:nq])
            fk_points.append(pts.tolist())

        result = {
            'dimensions': {
                'nx': nx,
                'nu': nu,
                'nq': nq,
                'N': N,
                'dt': dt,
                'n_z': n_z,
                'n_slack': n_slack,
                'n_z_total': n_z_total,
                'n_eq': A_eq.shape[0],
                'n_ineq': A_ineq.shape[0],
            },
            'linearisation': {
                'x_bar': x_bar,
                'u_bar': u_bar,
                'f_val': f_val,
                'Ac': Ac,
                'Bc': Bc,
                'Ad': Ad,
                'Bd': Bd,
                'c_k': c_k,
                'M': M_num,
                'M_inv': M_inv,
                'G': G_vec,
                'dG_dq': dG_dq,
            },
            'qp': {
                'Q': Q,
                'p': p,
                'A_eq': A_eq,
                'b_eq': b_eq,
                'A_ineq': A_ineq,
                'k_ineq': k_ineq,
            },
            'ref_traj': ref_traj,
            'ref_fk': fk_points,
            'x0': x0,
            'x_goal': x_goal,
            'z_labels': z_labels,
            'params': {'m1': arm.m1, 'm2': arm.m2, 'l1': arm.l1, 'l2': arm.l2, 'g': arm.g},
        }
        return _np(result)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)


@app.post("/api/solve")
async def api_solve(data: dict):
    """Solve 2DOF MPC QP with OSQP and return diagnostics for frontend inspector."""
    try:
        from src.solver.osqp_solver import OSQPSolver
        from src.utils.qp_inspector import verify_solution

        arm, mpc, x0, x_goal, N, dt = _parse_config(data)
        ref_traj = mpc.build_reference_trajectory(x0, x_goal)
        qp_matrices = mpc.build_qp(x0, ref_traj)
        Q, p, A_eq, b_eq, A_ineq, k_ineq = qp_matrices

        nx, nu, nq = arm.nx, arm.nu, arm.nq
        solver = OSQPSolver()
        A_all = np.vstack([A_eq, A_ineq])
        l_all = np.concatenate([b_eq, np.full(A_ineq.shape[0], -1e30)])
        u_all = np.concatenate([b_eq, k_ineq])
        z_star, _solve_info = solver.solve(Q, p, A_all, l_all, u_all)

        if z_star is None:
            return {'error': 'OSQP returned infeasible/unsolved', 'status': 'infeasible'}

        vr = verify_solution(z_star, qp_matrices, tol=1e-4, verbose=False)
        decomp = _decompose_z(z_star, N, nx, nu, nq)

        pred_fk = []
        for xk in decomp['x_k']:
            pts = arm.forward_kinematics(np.array(xk[:nq]))
            pred_fk.append(pts.tolist())

        result = {
            'status': 'solved',
            'z_star': z_star,
            'z_decomposed': decomp,
            'predicted_fk': pred_fk,
            'objective': vr['objective'],
            'eq_norm': vr['eq_norm'],
            'ineq_viol': vr['ineq_viol'],
            'kkt_resid': vr['kkt_resid'],
            'comp_slack': vr['comp_slack'],
            'passed': vr['passed'],
        }
        return _np(result)
    except Exception as e:
        return JSONResponse({'error': str(e), 'status': 'error'}, status_code=400)


@app.post("/api/snn_solve")
async def api_snn_solve(data: dict):
    """Solve the MPC QP using the modular PIPG-SNN solver."""
    try:
        from src.solver.pipg_snn_solver import PIPGSNNSolver
        from src.solver.hardware_estimator import HardwareEstimator

        arm, mpc, x0, x_goal, N, dt = _parse_config(data)
        ref_traj = mpc.build_reference_trajectory(x0, x_goal)
        qp_matrices = mpc.build_qp(x0, ref_traj)
        
        # Condense the QP for SNN efficiency
        Q_c, p_c, A_c, k_c = mpc.condense_qp(*qp_matrices)
        
        # SNN Solver Hyperparameters
        snn_cfg = data.get('snn_config', {})
        solver = PIPGSNNSolver(
            alpha0=float(snn_cfg.get('alpha0', 0.01)),
            beta0=float(snn_cfg.get('beta0', 0.1)),
            T_anneal=int(snn_cfg.get('T_anneal', 100)),
            max_iter=int(snn_cfg.get('max_iter', 200)),
            tol=float(snn_cfg.get('tol', 1e-4))
        )
        
        # PIPG assumes Ax <= b. For condensed QP, l is -inf.
        l_c = np.full(k_c.shape, -1e30)
        res = solver.solve(Q_c, p_c, A_c, l_c, k_c)
        
        # Estimate hardware performance
        est = HardwareEstimator()
        hw_stats = est.estimate_snn(Q_c.shape[0], A_c.shape[0], len(res.history))
        
        # Format history for frontend
        history_data = []
        for h in res.history:
            history_data.append({
                't': h.t,
                'x': h.x.tolist(),
                'cost': float(h.cost),
                'max_viol': float(h.max_viol),
                'alpha': h.alpha,
                'beta': h.beta
            })

        return _np({
            'status': res.status,
            'objective': res.objective,
            'ineq_viol': res.ineq_viol,
            'solve_time_ms': res.solve_time_ms,
            'history': history_data,
            'hardware': hw_stats,
            'z_star': res.z_star,
            'dimensions': {
                'n_vars': Q_c.shape[0],
                'n_constr': A_c.shape[0]
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({'error': str(e)}, status_code=400)


@app.get("/api/snn_hardware")
async def api_snn_hardware(n_vars: int = 10, n_constr: int = 20, n_iters: int = 100):
    """Return hardware timing estimates for arbitrary SNN sizes."""
    try:
        from src.solver.hardware_estimator import HardwareEstimator
        est = HardwareEstimator()
        stats = est.estimate_snn(n_vars, n_constr, n_iters)
        return _np(stats)
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)


@app.get("/api/snn_lif")
async def api_snn_lif(current: float = 0.5, duration: float = 100.0):
    """Run a single LIF neuron simulation for visualization."""
    try:
        from src.solver.lif_neuron_sim import LIFNeuronSim
        sim = LIFNeuronSim(n_neurons=1)
        v_hist, s_hist = sim.simulate(np.array([current]), duration)
        return _np({
            'v': v_hist.flatten(),
            'spikes': s_hist.flatten(),
            't': np.arange(0, duration, sim.p.dt).tolist()
        })
    except Exception as e:
        return JSONResponse({'error': str(e)}, status_code=400)

# ── API routes ────────────────────────────────────────────────────────────────

class QPRequest(BaseModel):
    n: int = 6
    solver: str = "osqp"
    condition_number: float = 1.0
    custom_P: Optional[list] = None
    custom_q: Optional[list] = None
    custom_A: Optional[list] = None
    custom_l: Optional[list] = None
    custom_u: Optional[list] = None

@app.post("/api/solve_qp")
async def api_solve_qp(req: QPRequest):
    """Run a QP solve and return matrices + solution."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    n = req.n
    if req.custom_P is not None:
        P = np.array(req.custom_P)
        q = np.array(req.custom_q)
        A = np.array(req.custom_A)
        l = np.array(req.custom_l)
        u = np.array(req.custom_u)
    else:
        rng = np.random.default_rng(42)
        kappa = req.condition_number
        D = np.diag(np.linspace(1, kappa, n))
        U, _ = np.linalg.qr(rng.standard_normal((n, n)))
        P = U @ D @ U.T
        P = 0.5 * (P + P.T) + 0.1 * np.eye(n)
        q = rng.standard_normal(n)
        A = np.eye(n)
        l = -5 * np.ones(n)
        u = 5 * np.ones(n)

    if req.solver == "osqp":
        from src.solver.osqp_solver import OSQPSolver
        solver = OSQPSolver()
    else:
        from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
        solver = StuartLandauLagrangeDirect(T_solve=3.0)

    x, info = solver.solve(P, q, A, l, u)

    return {
        "solver": req.solver,
        "n": n,
        "solution": x.tolist(),
        "info": info,
        "matrices": {
            "P": P.tolist(),
            "q": q.tolist(),
            "A": A.tolist(),
            "l": l.tolist(),
            "u": u.tolist(),
        }
    }

@app.get("/api/results")
async def api_results():
    """Return all benchmark JSON files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(RESULTS_DIR / "*.json")), reverse=True)
    results = []
    for f in files[:50]:
        try:
            with open(f) as fh:
                data = json.load(fh)
            results.append({"file": Path(f).name, "data": data})
        except Exception:
            pass
    return {"count": len(results), "results": results}

class EpisodeRequest(BaseModel):
    n_steps: int = 100
    solver: str = "osqp"
    record_gif: bool = True
    tau_scale: float = 3.0
    task: str = "reach"
    camera: str = "cam_side"
    gif_width: int = 640
    gif_height: int = 480
    controller_mode: str = "mpc_only"
    vla_server_url: str = "http://localhost:8000"
    dataset_episode_idx: int = 0


def _build_reference(task: str, step_idx: int, horizon: int = 10, dt: float = 0.05) -> np.ndarray:
    """Build a horizon of joint references for a simple task library."""
    t = (step_idx + np.arange(horizon)) * dt
    ref = np.zeros((horizon, 6), dtype=np.float32)

    if task == "wave":
        ref[:, 0] = 0.4 * np.sin(1.2 * t)
        ref[:, 1] = -0.8 + 0.25 * np.sin(0.9 * t)
        ref[:, 3] = 0.7 * np.sin(1.8 * t)
        ref[:, 5] = 0.4 * np.sin(2.0 * t)
    elif task == "pick_place":
        # Periodic sequence resembling approach/lift/place in joint space.
        phase = (t % 4.0)
        ref[:, 0] = 0.3 * np.sin(0.8 * t)
        ref[:, 1] = np.where(phase < 2.0, -1.0 + 0.25 * phase, -0.5 - 0.25 * (phase - 2.0))
        ref[:, 2] = np.where(phase < 2.0, 0.2 + 0.3 * phase, 0.8 - 0.3 * (phase - 2.0))
        ref[:, 3] = 0.35 * np.sin(1.0 * t + 0.5)
        ref[:, 4] = -0.2 + 0.15 * np.sin(1.2 * t)
    else:
        # reach
        ref[:, 0] = 0.55
        ref[:, 1] = -0.95
        ref[:, 2] = 0.65
        ref[:, 3] = 0.35
        ref[:, 4] = -0.25
        ref[:, 5] = 0.15

    return ref


def _vla_action_to_ref(vla_action: np.ndarray, horizon: int = 10) -> np.ndarray:
    """Convert VLA action vector into a fixed MPC reference horizon for 6 joints."""
    act = np.asarray(vla_action, dtype=np.float32).flatten()
    if act.size < 6:
        padded = np.zeros(6, dtype=np.float32)
        padded[:act.size] = act
        act = padded
    target_q = act[:6]
    return np.repeat(target_q[None, :], horizon, axis=0)


async def _fetch_real_vla_action(server_url: str, rgb: np.ndarray, state_q: np.ndarray, task: str) -> np.ndarray:
    """Fetch one action from real VLA server and return 6-DOF target candidate."""
    from src.smolvla.real_client import RealSmolVLAClient

    client = RealSmolVLAClient(server_url=server_url, timeout_s=3.0, max_retries=1)
    action = await client.predict(rgb_image=rgb, state=state_q, instruction=task)
    return np.asarray(action, dtype=np.float32)


def _dataset_reference_horizon(states: np.ndarray, step_idx: int, horizon: int = 10) -> np.ndarray:
    """Build a horizon from dataset states [T, 8] using the first 6 arm joints."""
    if states.shape[0] == 0:
        return np.zeros((horizon, 6), dtype=np.float32)
    out = np.zeros((horizon, 6), dtype=np.float32)
    last = states.shape[0] - 1
    for i in range(horizon):
        idx = min(step_idx + i, last)
        out[i] = states[idx, :6]
    return out


@app.get("/api/vla_status")
async def api_vla_status(url: str = "http://localhost:8000"):
    """Check real SmolVLA server availability."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.smolvla.real_client import RealSmolVLAClient
        client = RealSmolVLAClient(server_url=url, timeout_s=2.0, max_retries=1)
        healthy = await client.health_check()
        return {"healthy": bool(healthy), "url": url}
    except Exception as e:
        return {"healthy": False, "url": url, "error": str(e)}


@app.get("/api/dataset_info")
async def api_dataset_info():
    """Return dataset metadata for dashboard controls."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from data.loaders.lerobot_loader import LeRobotDatasetLoader
        loader = LeRobotDatasetLoader()
        info = loader.get_info()
        return {"available": True, "info": info}
    except Exception as e:
        return {"available": False, "error": str(e)}

@app.post("/api/run_episode")
async def api_run_episode(req: EpisodeRequest):
    """Run a simulation episode and optionally record GIF."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))

        from src.simulation.envs.xarm_env import XArmEnv
        from src.mpc.xarm_mpc_controller import XArmMPCController
        from src.solver.osqp_solver import OSQPSolver
        from src.visualization.episode_recorder import EpisodeRecorder
        from src.smolvla.mock_vla import MockVLAServer
        from data.loaders.lerobot_loader import LeRobotDatasetLoader
        import yaml

        with open("config/robots/xarm_6dof.yaml") as f:
            cfg = yaml.safe_load(f)

        env = XArmEnv(render_mode='offscreen')
        env.set_camera(req.camera)
        if req.solver == "osqp":
            mpc = XArmMPCController(OSQPSolver(), cfg)
        else:
            from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
            mpc = XArmMPCController(StuartLandauLagrangeDirect(T_solve=3.0), cfg)

        gif_path = None
        rec = EpisodeRecorder(fps=12, resize=(req.gif_width, req.gif_height)) if req.record_gif else None

        requested_mode = req.controller_mode
        effective_mode = requested_mode
        vla_calls = 0
        vla_failures = 0
        vla_query_interval = 8
        mock_vla = MockVLAServer() if requested_mode == "mpc_mock_vla" else None
        real_vla_enabled = requested_mode == "mpc_real_vla"

        dataset_states = None
        dataset_steps = None
        if req.task == "dataset_replay":
            loader = LeRobotDatasetLoader()
            episode = loader.load_episode(req.dataset_episode_idx)
            dataset_states = episode["states"]
            dataset_steps = int(episode["n_steps"])

        obs = env.reset()
        traj = []
        torques = []
        ref_source = []

        total_steps = req.n_steps
        if dataset_steps is not None:
            total_steps = min(total_steps, dataset_steps)

        for step in range(total_steps):
            if req.task == "dataset_replay" and dataset_states is not None:
                ref = _dataset_reference_horizon(dataset_states, step)
                ref_source.append("dataset")
            else:
                ref = _build_reference(req.task, step)

            # True dataset replay: directly apply recorded joint poses.
            # This shows what the dataset actually did and avoids MPC-tracking artifacts.
            if req.task == "dataset_replay" and dataset_states is not None:
                env.set_arm_state(dataset_states[step, :6])
                obs = env._get_obs()
                traj.append(obs['q'].tolist())
                torques.append([0.0] * 8)
                ref_source[-1:] = ["dataset_direct"]
                if rec:
                    rec.add_frame(env.render_frame(width=req.gif_width, height=req.gif_height))
                continue

            if requested_mode == "mpc_mock_vla" and mock_vla is not None and step % vla_query_interval == 0:
                out = mock_vla.predict(obs['rgb'], obs['q'], instruction=req.task)
                vla_action = np.asarray(out.get('action', []), dtype=np.float32)
                if vla_action.size >= 6:
                    ref = _vla_action_to_ref(vla_action)
                    vla_calls += 1
                    ref_source[-1:] = ["mock_vla"]
                else:
                    vla_failures += 1
                    if req.task != "dataset_replay":
                        ref_source[-1:] = ["task_library"]
            elif requested_mode == "mpc_real_vla" and real_vla_enabled and step % vla_query_interval == 0:
                try:
                    vla_action = await _fetch_real_vla_action(req.vla_server_url, obs['rgb'], obs['q'], req.task)
                    if vla_action.size >= 6:
                        ref = _vla_action_to_ref(vla_action)
                        vla_calls += 1
                        ref_source[-1:] = ["real_vla"]
                    else:
                        vla_failures += 1
                        if req.task != "dataset_replay":
                            ref_source[-1:] = ["task_library"]
                except Exception:
                    vla_failures += 1
                    if req.task != "dataset_replay":
                        ref_source[-1:] = ["task_library"]
                    # Degrade gracefully to MPC-only behavior after first hard failure.
                    effective_mode = "mpc_only_fallback"
                    real_vla_enabled = False
            elif req.task != "dataset_replay":
                ref_source.append("task_library")

            tau = mpc.step((obs['q'], obs['qdot']), ref)
            obs, _, done, _ = env.step(tau)
            traj.append(obs['q'].tolist())
            torques.append(np.asarray(tau).tolist())
            if rec:
                rec.add_frame(env.render_frame(width=req.gif_width, height=req.gif_height))
            if done:
                break

        env.close()

        if rec:
            gif_id = uuid.uuid4().hex[:8]
            gif_path = str(OUTPUTS_DIR / f"episode_{gif_id}.gif")
            rec.save(gif_path)

        return {
            "n_steps_completed": len(traj),
            "trajectory": traj,
            "torques": torques,
            "task": req.task,
            "dataset_episode_idx": req.dataset_episode_idx,
            "dataset_steps": dataset_steps,
            "camera": req.camera,
            "solver": req.solver,
            "controller_mode_requested": requested_mode,
            "controller_mode_effective": effective_mode,
            "vla_server_url": req.vla_server_url,
            "vla_calls": vla_calls,
            "vla_failures": vla_failures,
            "reference_source": ref_source,
            "gif_path": gif_path,
            "gif_url": f"/outputs/{Path(gif_path).name}" if gif_path else None,
        }
    except Exception as e:
        return JSONResponse({"error": str(e), "status": "failed", "type": type(e).__name__}, status_code=500)


@app.get("/api/gifs")
async def api_gifs():
    """List recent generated GIF files from outputs directory."""
    gif_files = sorted(OUTPUTS_DIR.glob("episode_*.gif"), key=lambda p: p.stat().st_mtime, reverse=True)
    data = []
    for gif in gif_files[:30]:
        stat = gif.stat()
        data.append({
            "name": gif.name,
            "url": f"/outputs/{gif.name}",
            "size_kb": round(stat.st_size / 1024, 1),
            "modified": stat.st_mtime,
        })
    return {"count": len(data), "gifs": data}

# Serve GIF outputs
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=False)
