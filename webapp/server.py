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
from pathlib import Path
from typing import Optional

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
