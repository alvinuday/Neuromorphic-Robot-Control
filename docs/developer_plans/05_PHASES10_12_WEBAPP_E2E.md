# PHASES 10–12 — WEBAPP, E2E DEMO, FINAL DOCUMENTATION
**Requires**: Phases 1–9 complete. At least B1 and B5 benchmark JSONs must exist.

---

## PHASE 10 — WEBAPP EXTENSION

**Goal**: Extend the existing QP Inspector with MuJoCo visualization and benchmark dashboard — without touching any existing HTML/CSS/JS.

### Sacred Rule Verification

Before starting Phase 10:
```bash
md5sum webapp/static/index.html webapp/static/style.css webapp/static/app.js \
    > docs/agent/webapp_checksums_before.txt
cat docs/agent/webapp_checksums_before.txt
```

At the end of Phase 10, run again and verify checksums are identical:
```bash
md5sum webapp/static/index.html webapp/static/style.css webapp/static/app.js \
    > docs/agent/webapp_checksums_after.txt
diff docs/agent/webapp_checksums_before.txt docs/agent/webapp_checksums_after.txt
# Must show NO DIFFERENCE
```

### 10.1 — `webapp/server.py`

FastAPI backend. Uses port **3000** (port 8000 is the VLA server).

```python
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
  WS   /ws/live       → Stream live MPC data (optional, Phase 11)
"""
import json, glob, os, time, uuid
from pathlib import Path
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="Neuromorphic Robot Control")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

WEBAPP_DIR  = Path(__file__).parent
STATIC_DIR  = WEBAPP_DIR / "static"
VIZ_DIR     = WEBAPP_DIR / "viz"
RESULTS_DIR = Path("evaluation/results")
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Serve existing QP inspector (NEVER modify files in STATIC_DIR)
app.mount("/static",  StaticFiles(directory=STATIC_DIR), name="static")
# Serve new viz pages
app.mount("/viz",     StaticFiles(directory=VIZ_DIR),    name="viz")

# ── Page routes ───────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/mujoco")
async def mujoco_page():
    return FileResponse(VIZ_DIR / "mujoco_viz.html")

@app.get("/dashboard")
async def dashboard_page():
    return FileResponse(VIZ_DIR / "dashboard.html")

# ── API routes ────────────────────────────────────────────────────────────────

class QPRequest(BaseModel):
    n: int = 6                  # problem dimension
    solver: str = "osqp"        # "osqp" or "sl"
    condition_number: float = 1.0  # for generated test QP
    custom_P: Optional[list] = None
    custom_q: Optional[list] = None
    custom_A: Optional[list] = None
    custom_l: Optional[list] = None
    custom_u: Optional[list] = None

@app.post("/api/solve_qp")
async def api_solve_qp(req: QPRequest):
    """Run a QP solve and return matrices + solution."""
    import os; os.environ.setdefault('MUJOCO_GL', 'osmesa')

    n = req.n
    if req.custom_P is not None:
        P = np.array(req.custom_P); q = np.array(req.custom_q)
        A = np.array(req.custom_A); l = np.array(req.custom_l)
        u = np.array(req.custom_u)
    else:
        rng = np.random.default_rng(42)
        kappa = req.condition_number
        D = np.diag(np.linspace(1, kappa, n))
        U = np.linalg.qr(rng.standard_normal((n, n)))[0]
        P = U @ D @ U.T; P = 0.5*(P + P.T) + 0.1*np.eye(n)
        q = rng.standard_normal(n)
        A = np.eye(n); l = -5*np.ones(n); u = 5*np.ones(n)

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
            "P": P.tolist(), "q": q.tolist(),
            "A": A.tolist(), "l": l.tolist(), "u": u.tolist(),
        }
    }

@app.get("/api/results")
async def api_results():
    """Return all benchmark JSON files."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(RESULTS_DIR / "*.json")), reverse=True)
    results = []
    for f in files[:50]:   # cap at 50 most recent
        try:
            with open(f) as fh:
                data = json.load(fh)
            results.append({"file": Path(f).name, "data": data})
        except Exception:
            pass
    return {"count": len(results), "results": results}

class EpisodeRequest(BaseModel):
    n_steps:     int   = 100
    solver:      str   = "osqp"
    record_gif:  bool  = True
    tau_scale:   float = 3.0   # scale applied to joint 0 for visible motion

@app.post("/api/run_episode")
async def api_run_episode(req: EpisodeRequest):
    """Run a simulation episode and optionally record GIF."""
    import os; os.environ.setdefault('MUJOCO_GL', 'osmesa')

    from src.simulation.envs.xarm_env import XArmEnv
    from src.mpc.xarm_mpc_controller import XArmMPCController
    from src.solver.osqp_solver import OSQPSolver
    from src.visualization.episode_recorder import EpisodeRecorder
    import yaml

    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)

    env = XArmEnv(render_mode='offscreen')
    mpc = XArmMPCController(OSQPSolver(), cfg)

    gif_path = None
    rec = EpisodeRecorder(fps=10) if req.record_gif else None
    if rec: rec.start()

    obs = env.reset()
    traj = []
    ref  = np.zeros((10, 6))
    ref[:, 0] = 0.5  # simple target: joint 0 → 0.5 rad

    for step in range(req.n_steps):
        tau = mpc.step((obs['q'], obs['qdot']), ref)
        obs, _, done, _ = env.step(tau)
        traj.append(obs['q'].tolist())
        if rec: rec.add_frame(obs['rgb'])
        if done: break

    env.close()

    if rec:
        gif_id = uuid.uuid4().hex[:8]
        gif_path = str(OUTPUTS_DIR / f"episode_{gif_id}.gif")
        rec.save(gif_path)

    return {
        "n_steps_completed": len(traj),
        "trajectory": traj,
        "gif_path": gif_path,
        "gif_url": f"/outputs/{Path(gif_path).name}" if gif_path else None,
    }

# Serve GIF outputs
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=False)
```

### 10.2 — `webapp/viz/mujoco_viz.html`

New page (independent of existing QP inspector):

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MuJoCo Visualization — Neuromorphic Robot</title>
<style>
  body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }
  h1   { color: #00d4ff; }
  .tab-bar { display: flex; gap: 8px; margin-bottom: 20px; }
  .tab-btn { padding: 8px 16px; background: #16213e; border: 1px solid #00d4ff44;
             color: #a0c4ff; cursor: pointer; border-radius: 4px; }
  .tab-btn.active { background: #0f3460; color: #00d4ff; border-color: #00d4ff; }
  .tab-panel { display: none; }
  .tab-panel.active { display: block; }
  .card  { background: #16213e; border-radius: 8px; padding: 16px; margin: 12px 0; }
  button.action { background: #0f3460; color: #00d4ff; border: 1px solid #00d4ff66;
                  padding: 8px 20px; border-radius: 4px; cursor: pointer; font-size: 14px; }
  button.action:hover { background: #00d4ff22; }
  canvas { display: block; max-width: 100%; }
  img.gif-preview { max-width: 400px; border: 1px solid #00d4ff44; border-radius: 4px; }
  pre  { background: #0a0a1a; padding: 10px; border-radius: 4px; font-size: 12px;
         overflow-x: auto; color: #88c0d0; max-height: 300px; overflow-y: auto; }
  .status { color: #88ff88; }
  .error  { color: #ff6b6b; }
</style>
</head>
<body>
<h1>🤖 MuJoCo Visualization</h1>
<p style="color:#88c0d0">Simulation, episode replay, and QP matrix inspection.</p>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('run')">▶ Run Episode</button>
  <button class="tab-btn" onclick="switchTab('gallery')">🎞 GIF Gallery</button>
  <button class="tab-btn" onclick="switchTab('qp')">📊 QP Matrices</button>
  <button class="tab-btn" onclick="switchTab('back')">← Back to QP Inspector</button>
</div>

<!-- Tab: Run Episode -->
<div class="tab-panel active" id="tab-run">
  <div class="card">
    <h3>Run Simulation Episode</h3>
    <label>Steps: <input type="number" id="n-steps" value="100" min="10" max="500"></label> &nbsp;
    <label>Solver: <select id="solver-sel">
      <option value="osqp">OSQP (fast)</option>
      <option value="sl">Stuart-Landau (slow ~3s/QP)</option>
    </select></label> &nbsp;
    <label><input type="checkbox" id="record-gif" checked> Record GIF</label>
    <br><br>
    <button class="action" onclick="runEpisode()">▶ Run Episode</button>
    <span id="run-status" style="margin-left:12px"></span>
  </div>
  <div class="card" id="episode-result" style="display:none">
    <h3>Result</h3>
    <div id="gif-container"></div>
    <canvas id="traj-canvas" width="700" height="250"></canvas>
    <pre id="episode-json"></pre>
  </div>
</div>

<!-- Tab: GIF Gallery -->
<div class="tab-panel" id="tab-gallery">
  <div class="card">
    <h3>Saved Episode GIFs</h3>
    <button class="action" onclick="loadGallery()">🔄 Refresh</button>
    <div id="gallery-grid" style="display:flex;flex-wrap:wrap;gap:12px;margin-top:12px"></div>
  </div>
</div>

<!-- Tab: QP Matrices -->
<div class="tab-panel" id="tab-qp">
  <div class="card">
    <h3>QP Matrix Heatmap</h3>
    <p>Run a quick QP to inspect the matrices:</p>
    <label>n (dim): <input type="number" id="qp-n" value="6" min="2" max="12"></label> &nbsp;
    <label>κ(P): <input type="number" id="qp-kappa" value="1" step="10"></label> &nbsp;
    <label>Solver: <select id="qp-solver">
      <option value="osqp">OSQP</option>
      <option value="sl">SL</option>
    </select></label>
    <br><br>
    <button class="action" onclick="solveQP()">Solve & Inspect</button>
    <pre id="qp-result"></pre>
    <canvas id="qp-heatmap" width="700" height="300"></canvas>
  </div>
</div>

<script>
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  if (name === 'back') { window.location.href = '/'; return; }
  event.target.classList.add('active');
  document.getElementById('tab-' + name).classList.add('active');
}

async function runEpisode() {
  const status = document.getElementById('run-status');
  status.innerHTML = '<span class="status">Running...</span>';
  const body = {
    n_steps:    parseInt(document.getElementById('n-steps').value),
    solver:     document.getElementById('solver-sel').value,
    record_gif: document.getElementById('record-gif').checked,
  };
  try {
    const r = await fetch('/api/run_episode', { method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
    const data = await r.json();
    status.innerHTML = `<span class="status">✓ ${data.n_steps_completed} steps</span>`;
    
    const container = document.getElementById('episode-result');
    container.style.display = 'block';
    
    if (data.gif_url) {
      document.getElementById('gif-container').innerHTML =
        `<img class="gif-preview" src="${data.gif_url}?t=${Date.now()}" alt="Episode GIF">`;
    }
    document.getElementById('episode-json').textContent =
      JSON.stringify({steps: data.n_steps_completed, gif: data.gif_url}, null, 2);
    
    drawTrajectory(data.trajectory);
  } catch(e) {
    status.innerHTML = `<span class="error">Error: ${e}</span>`;
  }
}

function drawTrajectory(traj) {
  const canvas = document.getElementById('traj-canvas');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0a0a1a'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  const colors = ['#ff6b6b','#ffd93d','#6bcb77','#4d96ff','#c77dff','#ff9a3c'];
  const T = traj.length, W = canvas.width - 60, H = canvas.height - 40;
  
  ctx.fillStyle = '#88c0d0'; ctx.font = '12px monospace';
  ctx.fillText('Joint trajectories (rad)', 5, 15);
  
  for (let j = 0; j < 6; j++) {
    const vals = traj.map(t => t[j]);
    const vmin = Math.min(...vals), vmax = Math.max(...vals);
    const range = vmax - vmin || 0.01;
    ctx.strokeStyle = colors[j]; ctx.lineWidth = 1.5;
    ctx.beginPath();
    vals.forEach((v, i) => {
      const x = 30 + (i / (T-1)) * W;
      const y = 30 + (1 - (v - vmin) / range) * H;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = colors[j];
    ctx.fillText(`J${j+1}`, canvas.width - 25, 35 + j * 14);
  }
}

async function solveQP() {
  const body = {
    n: parseInt(document.getElementById('qp-n').value),
    solver: document.getElementById('qp-solver').value,
    condition_number: parseFloat(document.getElementById('qp-kappa').value),
  };
  const r = await fetch('/api/solve_qp', { method:'POST',
    headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
  const data = await r.json();
  document.getElementById('qp-result').textContent = JSON.stringify({
    solver: data.solver, solution: data.solution.map(x => +x.toFixed(4)),
    info: data.info
  }, null, 2);
  drawHeatmap(data.matrices.P);
}

function drawHeatmap(P) {
  const canvas = document.getElementById('qp-heatmap');
  const ctx = canvas.getContext('2d');
  const n = P.length;
  const cell = Math.floor(Math.min(canvas.width, canvas.height) / (n + 1));
  const vals = P.flat();
  const vmax = Math.max(...vals.map(Math.abs));
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#0a0a1a'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#88c0d0'; ctx.font = '12px monospace';
  ctx.fillText('P matrix heatmap', 5, 15);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const v = P[i][j] / (vmax || 1);
    const r = v > 0 ? Math.round(v * 200) : 0;
    const b = v < 0 ? Math.round(-v * 200) : 0;
    ctx.fillStyle = `rgb(${r},30,${b})`;
    ctx.fillRect(30 + j*cell, 25 + i*cell, cell-1, cell-1);
  }
}

async function loadGallery() {
  const grid = document.getElementById('gallery-grid');
  grid.innerHTML = 'Loading...';
  try {
    const r = await fetch('/api/results');
    const data = await r.json();
    grid.innerHTML = data.count === 0 ? '<p>No results yet. Run an episode first.</p>' : '';
  } catch(e) {
    grid.innerHTML = `<span class="error">${e}</span>`;
  }
}
</script>
</body>
</html>
```

### 10.3 — `webapp/viz/dashboard.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Benchmark Dashboard — Neuromorphic Robot</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  body { font-family: monospace; background: #1a1a2e; color: #e0e0e0; margin: 0; padding: 20px; }
  h1   { color: #00d4ff; }
  .nav { margin-bottom: 20px; }
  .nav a { color: #4d96ff; text-decoration: none; margin-right: 16px; }
  .card { background: #16213e; border-radius: 8px; padding: 16px; margin: 12px 0; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  table  { width: 100%; border-collapse: collapse; }
  th     { color: #00d4ff; padding: 6px; text-align: left; border-bottom: 1px solid #00d4ff33; }
  td     { padding: 5px 6px; border-bottom: 1px solid #ffffff11; font-size: 13px; }
  .tag-mock { background: #ffd93d22; color: #ffd93d; padding: 1px 6px; border-radius: 3px; font-size: 11px; }
  button.refresh { background: #0f3460; color: #00d4ff; border: 1px solid #00d4ff44;
                   padding: 6px 14px; border-radius: 4px; cursor: pointer; }
</style>
</head>
<body>
<h1>📊 Benchmark Dashboard</h1>
<div class="nav">
  <a href="/">← QP Inspector</a>
  <a href="/mujoco">🤖 MuJoCo Viz</a>
  <button class="refresh" onclick="loadAll()">🔄 Refresh</button>
</div>

<div class="grid2">
  <div class="card">
    <h3>Solver Comparison (B5)</h3>
    <canvas id="solver-chart" height="200"></canvas>
    <p id="solver-note" style="font-size:12px;color:#88c0d0"></p>
  </div>
  <div class="card">
    <h3>Sensor Ablation (B4)</h3>
    <canvas id="ablation-chart" height="200"></canvas>
  </div>
</div>

<div class="card">
  <h3>All Benchmark Results</h3>
  <table id="results-table">
    <thead><tr><th>File</th><th>ID</th><th>Timestamp</th><th>VLA Mode</th><th>Key Metric</th></tr></thead>
    <tbody id="results-body"><tr><td colspan="5">Loading...</td></tr></tbody>
  </table>
</div>

<script>
let solverChart = null, ablationChart = null;

async function loadAll() {
  const r = await fetch('/api/results');
  const data = await r.json();
  renderTable(data.results);
  
  const b5 = data.results.find(r => r.data.benchmark_id?.startsWith('B5'));
  if (b5) renderSolverChart(b5.data);
  
  const b4 = data.results.find(r => r.data.benchmark_id?.startsWith('B4'));
  if (b4) renderAblationChart(b4.data);
}

function renderTable(results) {
  const tbody = document.getElementById('results-body');
  if (!results.length) { tbody.innerHTML = '<tr><td colspan="5">No results yet. Run benchmarks first.</td></tr>'; return; }
  tbody.innerHTML = results.map(r => {
    const d = r.data;
    const vla = d.environment?.vla_mode || d.config?.vla_mode || '?';
    const tag = vla === 'MOCK' ? '<span class="tag-mock">MOCK</span>' : vla;
    const metric = extractKeyMetric(d);
    return `<tr>
      <td>${r.file}</td>
      <td>${d.benchmark_id || '—'}</td>
      <td>${(d.timestamp||'').slice(0,19)}</td>
      <td>${tag}</td>
      <td>${metric}</td>
    </tr>`;
  }).join('');
}

function extractKeyMetric(d) {
  if (!d.results) return '—';
  const r = d.results;
  if (r.mean_rmse      !== undefined) return `RMSE: ${r.mean_rmse.toFixed(4)}`;
  if (r.mean_solve_ms  !== undefined) return `Solve: ${r.mean_solve_ms.toFixed(1)}ms`;
  if (d.summary?.sl_mean_ms_well !== undefined)
    return `SL: ${d.summary.sl_mean_ms_well.toFixed(0)}ms | OSQP: ${d.summary.osqp_mean_ms_well.toFixed(0)}ms`;
  return '—';
}

function renderSolverChart(data) {
  if (!data.summary) return;
  const s = data.summary;
  const ctx = document.getElementById('solver-chart').getContext('2d');
  if (solverChart) solverChart.destroy();
  solverChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Well-conditioned', 'Ill-conditioned'],
      datasets: [
        { label: 'OSQP (ms)', data: [s.osqp_mean_ms_well, s.osqp_mean_ms_ill],
          backgroundColor: '#4d96ff88', borderColor: '#4d96ff', borderWidth: 1 },
        { label: 'SL (ms)',   data: [s.sl_mean_ms_well, s.sl_mean_ms_ill],
          backgroundColor: '#ff6b6b88', borderColor: '#ff6b6b', borderWidth: 1 },
      ]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#e0e0e0' } } },
      scales: {
        x: { ticks: { color: '#88c0d0' }, grid: { color: '#ffffff11' } },
        y: { ticks: { color: '#88c0d0' }, grid: { color: '#ffffff11' },
             title: { display: true, text: 'Time (ms)', color: '#88c0d0' } }
      }
    }
  });
  document.getElementById('solver-note').textContent =
    `SL obj error: ${(s.sl_mean_obj_error_pct||0).toFixed(2)}% vs OSQP reference`;
}

function renderAblationChart(data) {
  if (!data.modes) return;
  const ctx = document.getElementById('ablation-chart').getContext('2d');
  if (ablationChart) ablationChart.destroy();
  const modes = Object.keys(data.modes);
  const rmses = modes.map(m => data.modes[m].mean_tracking_rmse || 0);
  ablationChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: modes,
      datasets: [{ label: 'Tracking RMSE', data: rmses,
        backgroundColor: modes.map((_, i) => `hsl(${i*60+200},60%,55%)`),
        borderWidth: 1 }]
    },
    options: {
      responsive: true,
      plugins: { legend: { labels: { color: '#e0e0e0' } } },
      scales: {
        x: { ticks: { color: '#88c0d0' }, grid: { color: '#ffffff11' } },
        y: { ticks: { color: '#88c0d0' }, grid: { color: '#ffffff11' } }
      }
    }
  });
}

loadAll();
</script>
</body>
</html>
```

### 10.4 — Phase 10 Gate

```bash
# Start webapp
python webapp/server.py &
sleep 3

# Verify existing QP inspector untouched
curl -s http://localhost:3000/ | grep -c "QP Inspector"   # should be > 0

# Verify new pages serve correctly
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/mujoco    # 200
curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/dashboard # 200

# Verify API works
curl -s -X POST http://localhost:3000/api/solve_qp \
    -H "Content-Type: application/json" \
    -d '{"n":6,"solver":"osqp"}' | python -m json.tool | grep "solution"

# Kill webapp
kill %1

# Verify existing files unchanged
diff docs/agent/webapp_checksums_before.txt docs/agent/webapp_checksums_after.txt
echo "EXIT CODE: $?  (0 = files unchanged)"
```

---

## PHASE 11 — END-TO-END DEMO

### 11.1 — `scripts/run_demo.py`

```python
#!/usr/bin/env python3
"""
End-to-end pick-and-place demonstration.

Usage:
    python scripts/run_demo.py --solver osqp --steps 200 --gif outputs/demo_osqp.gif
    python scripts/run_demo.py --solver sl   --steps 100 --gif outputs/demo_sl.gif
"""
import argparse, json, time, numpy as np, yaml, os
os.environ.setdefault('MUJOCO_GL', 'osmesa')
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--solver', choices=['osqp','sl'], default='osqp')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--gif', default='outputs/demo.gif')
    parser.add_argument('--vla', choices=['mock'], default='mock')  # real = future
    args = parser.parse_args()

    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)

    from src.simulation.envs.xarm_env import XArmEnv
    from src.mpc.xarm_mpc_controller import XArmMPCController
    from src.smolvla.vla_client import VLAClient
    from src.fusion.real_fusion_encoder import RealFusionEncoder
    from src.integration.dual_controller import DualSystemController
    from src.visualization.episode_recorder import EpisodeRecorder

    if args.solver == 'osqp':
        from src.solver.osqp_solver import OSQPSolver as Solver
    else:
        from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect as Solver

    env     = XArmEnv(render_mode='offscreen')
    mpc     = XArmMPCController(Solver(), cfg)
    vla     = VLAClient(mock_mode=True)
    fusion  = RealFusionEncoder.mode_full()
    ctrl    = DualSystemController(mpc, vla, fusion, instruction="pick up the red block")
    ctrl.start()

    rec = EpisodeRecorder(fps=10)
    rec.start()

    obs    = env.reset()
    rmses  = []
    # Simple reference: joint 0 moves to 0.4 rad
    ref    = np.zeros((10, 6)); ref[:, 0] = 0.4

    print(f"Running {args.steps} steps with {args.solver.upper()} solver + MockVLA...")
    t0 = time.perf_counter()
    for step in range(args.steps):
        tau, info = ctrl.step(obs)
        obs, _, done, _ = env.step(tau)
        rec.add_frame(obs['rgb'])
        rmse = float(np.sqrt(np.mean((obs['q'] - ref[0])**2)))
        rmses.append(rmse)
        if step % 50 == 0:
            print(f"  step {step:3d} | q[0]={obs['q'][0]:.3f} | RMSE={rmse:.4f} | "
                  f"VLA calls={info['vla_success']} | MPC {info['mpc_time_ms']:.1f}ms")
        if done: break

    elapsed = time.perf_counter() - t0
    ctrl.stop()
    env.close()

    gif_path = rec.save(args.gif)
    stats    = ctrl.get_stats()

    result = {
        'solver':          args.solver,
        'vla_mode':        'MOCK',
        'n_steps':         len(rmses),
        'elapsed_s':       elapsed,
        'hz':              len(rmses) / elapsed,
        'mean_rmse':       float(np.mean(rmses)),
        'final_q0':        float(obs['q'][0]),
        'target_q0':       0.4,
        'gif':             gif_path,
        **stats,
    }
    print("\n── DEMO COMPLETE ──────────────────────────────")
    print(json.dumps(result, indent=2))
    print(f"GIF saved: {gif_path}")

if __name__ == "__main__":
    main()
```

### 11.2 — E2E Tests

`tests/e2e/test_episode_run.py`:
```python
import os; os.environ.setdefault('MUJOCO_GL', 'osmesa')
import numpy as np, pytest, subprocess, sys
from pathlib import Path

def test_mpc_osqp_episode():
    """50 steps of OSQP-MPC without VLA."""
    import yaml
    from src.simulation.envs.xarm_env import XArmEnv
    from src.mpc.xarm_mpc_controller import XArmMPCController
    from src.solver.osqp_solver import OSQPSolver
    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)
    env = XArmEnv(render_mode='offscreen')
    mpc = XArmMPCController(OSQPSolver(), cfg)
    obs = env.reset()
    q0  = obs['q'].copy()
    ref = np.zeros((10, 6)); ref[:, 0] = 0.3
    for _ in range(50):
        tau = mpc.step((obs['q'], obs['qdot']), ref)
        obs, _, _, _ = env.step(tau)
    env.close()
    # Arm must have moved toward reference
    assert abs(obs['q'][0]) > 0.001, "Arm didn't move at all"

def test_dual_controller_episode():
    import yaml, time
    from src.simulation.envs.xarm_env import XArmEnv
    from src.mpc.xarm_mpc_controller import XArmMPCController
    from src.solver.osqp_solver import OSQPSolver
    from src.smolvla.vla_client import VLAClient
    from src.fusion.real_fusion_encoder import RealFusionEncoder
    from src.integration.dual_controller import DualSystemController
    with open("config/robots/xarm_6dof.yaml") as f:
        cfg = yaml.safe_load(f)
    env  = XArmEnv(render_mode='offscreen')
    ctrl = DualSystemController(
        XArmMPCController(OSQPSolver(), cfg),
        VLAClient(mock_mode=True),
        RealFusionEncoder.mode_rgb_only()
    )
    ctrl.start()
    obs = env.reset()
    for _ in range(30):
        tau, _ = ctrl.step(obs)
        obs, _, done, _ = env.step(tau)
    ctrl.stop(); env.close()
    assert obs is not None

def test_gif_produced(tmp_path):
    from src.simulation.envs.xarm_env import XArmEnv
    from src.visualization.episode_recorder import EpisodeRecorder
    env = XArmEnv(render_mode='offscreen')
    obs = env.reset()
    gif_path = str(tmp_path / "test_episode.gif")
    with EpisodeRecorder(fps=10).recording(gif_path) as rec:
        tau = np.zeros(8); tau[0] = 2.0
        for _ in range(20):
            obs, _, _, _ = env.step(tau)
            rec.add_frame(obs['rgb'])
    env.close()
    size = Path(gif_path).stat().st_size
    assert size > 10_000, f"GIF too small ({size} bytes) — frames all identical?"

def test_benchmarks_produced_output():
    """At least one benchmark JSON must exist in evaluation/results/."""
    results = list(Path("evaluation/results").glob("*.json"))
    assert len(results) > 0, \
        "No benchmark results found. Run at least: python evaluation/benchmarks/run_mpc_solo.py"
```

**Phase 11 Gate**: `pytest tests/e2e/ -v` — ALL PASS

---

## PHASE 12 — FINAL DOCUMENTATION & STATE UPDATE

### 12.1 — Required README.md sections

```markdown
# Neuromorphic Robot Control

[one-line description]

## Quick Start
[exact install commands]
[how to run the webapp]
[how to run all tests]

## Architecture
[ASCII diagram from 00_INDEX.md]

## Running Benchmarks
[in order: B1 → B2 → B3 → B4 → B5]

## Benchmark Results (Actual Numbers)
[paste actual results from evaluation/results/ — NOT made-up numbers]

## Known Limitations
- SL solver is 2000–8000ms per QP (expected; see theory)
- SmolVLA requires GPU for real inference; MockVLA used in all benchmarks
- Dataset tests require internet access to download lerobot/utokyo_xarm_pick_and_place
- MuJoCo headless rendering requires osmesa (set MUJOCO_GL=osmesa)
```

### 12.2 — Final AGENT_STATE.md

```markdown
## Phase 12 — COMPLETE (Final State)

### Full Test Suite Results
[paste output of: pytest tests/ -v --tb=short]

### Benchmark Summaries
B1 (MPC solo, OSQP):  RMSE = X.XXXX, solve_ms = XX.X
B5 (Solver compare):  SL/OSQP speedup = Xx, SL obj error = X.XX%
B4 (Sensor ablation): M4 RMSE = X.XXXX (best), M0 RMSE = X.XXXX (worst)

### What Was Hallucinated in Original Repo
[specific list with file names and evidence]

### What Was Built From Scratch
[specific list]

### Webapp Integrity
[confirm checksums match — paste diff output]

### How to Run Everything
[exact sequence of commands]
```

---

*End of Phase 10–12 plan. All phases complete.*
