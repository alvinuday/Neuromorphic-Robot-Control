# CODING PROMPT FOR AGENT
---

You are implementing a complete robotic control system from scratch (or near-scratch). The repository at `neuromorphic-robot/` exists but is partially broken — previous development sessions introduced stubs, fake benchmark results, and hallucinated progress. Your job is to audit, clean, rebuild, test, and validate everything.

You have 7 plan files in `docs/agent/`:
- `00_INDEX.md` — philosophy, rules, target structure
- `01_PHASE0_AUDIT_CLEANUP.md` — audit script, cleanup rules
- `02_PHASES1_3_FOUNDATIONS.md` — ABCs, SL solver, OSQP solver, MuJoCo env
- `03_PHASES4_6_MPC_FUSION_VLA.md` — MPC controller, sensor fusion, SmolVLA
- `04_PHASES7_9_INTEGRATION.md` — dual controller, dataset, benchmarks
- `05_PHASES10_12_WEBAPP_E2E.md` — webapp extension, E2E demo, docs

**Read all 7 files now before writing any code.**

---

## YOUR ABSOLUTE RULES

1. **Read the plan files completely before starting.** Use `cat docs/agent/00_INDEX.md` etc.

2. **Phase complete = `pytest -v` shows green.** Not "I think it works." Not "I reviewed the code." The tests must actually pass.

3. **Never fake a result.** If a benchmark can't run, say so and log the error. If an MPC outputs the same torque regardless of input, that is a bug — fix it.

4. **Never modify these three files under any circumstances:**
   - `webapp/static/index.html`
   - `webapp/static/style.css`
   - `webapp/static/app.js`
   Verify this with `git diff webapp/static/` at the end.

5. **MockVLA outputs must always carry `"source": "MOCK"`.** Any result using MockVLA is labeled as such in JSON outputs.

6. **Write `docs/agent/AGENT_STATE.md` after every phase.** Include: what passed, what failed, actual pytest output snippets (not paraphrased).

7. **Log every deleted/moved file** in `docs/agent/CLEANUP_LOG.md`.

8. **The SL solver being slow (2000–8000ms per QP) is correct behavior.** Do not "fix" this by reducing T_solve to fake speed or returning early. Document the timing honestly.

---

## START HERE: EXACT SEQUENCE OF COMMANDS

```bash
# 0. Read all plan files
cat docs/agent/00_INDEX.md
cat docs/agent/01_PHASE0_AUDIT_CLEANUP.md
cat docs/agent/02_PHASES1_3_FOUNDATIONS.md
cat docs/agent/03_PHASES4_6_MPC_FUSION_VLA.md
cat docs/agent/04_PHASES7_9_INTEGRATION.md
cat docs/agent/05_PHASES10_12_WEBAPP_E2E.md
```

Then execute phases in order:

```bash
# Phase 0: Audit
python scripts/audit/audit_repo.py
bash scripts/audit/archive_legacy.sh
# → Produces docs/agent/AUDIT_REPORT.md and docs/agent/CLEANUP_LOG.md

# Phase 1: Abstractions
# Create: src/core/base_solver.py, base_controller.py, base_env.py
# Create/verify: config/robots/xarm_6dof.yaml
pytest tests/unit/test_abstractions.py -v                # MUST PASS before Phase 2

# Phase 2: Solvers
# Create/fix: src/solver/stuart_landau_lagrange_direct.py
# Create/fix: src/solver/osqp_solver.py
pytest tests/unit/test_sl_solver.py tests/unit/test_osqp_solver.py -v

# Phase 3: MuJoCo Environment
# Create/fix: assets/xarm_6dof.xml (with table + red block)
# Create/fix: src/simulation/envs/xarm_env.py
# Create: src/simulation/cameras/event_camera.py
export MUJOCO_GL=osmesa   # must be set for headless
pytest tests/integration/test_xarm_env.py -v             # MUST PASS before Phase 4

# Phase 4: MPC Controller
# Create: src/dynamics/xarm_dynamics.py
# Create/fix: src/mpc/xarm_mpc_controller.py
pytest tests/integration/test_mpc_controller.py -v

# Phase 5: Sensor Fusion
# Create/fix: src/fusion/real_fusion_encoder.py
pytest tests/unit/test_fusion_encoder.py -v

# Phase 6: SmolVLA
# Create: src/smolvla/mock_vla.py
# Create/fix: src/smolvla/vla_server.py (supports --mode mock)
# Create/fix: src/smolvla/vla_client.py
# Create: src/smolvla/action_processor.py
pytest tests/integration/test_vla_mock.py -v

# Phase 7: Dual Controller
# Create/fix: src/integration/trajectory_buffer.py
# Create/fix: src/integration/dual_controller.py
pytest tests/integration/test_dual_controller.py -v

# Phase 8: Dataset + GIF
# Create: data/loaders/lerobot_loader.py
# Create: src/visualization/episode_recorder.py
pytest tests/integration/test_dataset_and_gif.py -v

# Phase 9: Benchmarks (run in order)
python evaluation/benchmarks/run_mpc_solo.py       # produces B1 JSON
python evaluation/benchmarks/run_vla_solo.py       # produces B2 JSON
python evaluation/benchmarks/run_dual_system.py    # produces B3 JSON
python evaluation/benchmarks/sensor_ablation.py    # produces B4 JSON (slow)
python evaluation/benchmarks/solver_comparison.py  # produces B5 JSON (very slow)
ls -la evaluation/results/*.json                   # must exist with non-zero sizes

# Phase 10: Webapp
# Create: webapp/server.py
# Create: webapp/viz/mujoco_viz.html
# Create: webapp/viz/dashboard.html
# VERIFY checksums of existing files (must be unchanged):
md5sum webapp/static/index.html webapp/static/style.css webapp/static/app.js

# Phase 11: E2E
python scripts/run_demo.py --solver osqp --steps 100 --gif outputs/demo_osqp.gif
python scripts/run_demo.py --solver sl   --steps 30  --gif outputs/demo_sl.gif
pytest tests/e2e/ -v

# Phase 12: Final docs
# Update: README.md, docs/agent/AGENT_STATE.md (final)
pytest tests/ -v --tb=short 2>&1 | tee docs/agent/final_test_results.txt
```

---

## VALIDATION COMMANDS (run after each phase to self-verify)

```bash
# Phase 2: SL solver correctness
python -c "
from src.solver.stuart_landau_lagrange_direct import StuartLandauLagrangeDirect
import numpy as np
P=np.array([[2.,0.],[0.,2.]]); q=np.zeros(2)
A=np.array([[1.,1.],[1.,0.],[0.,1.]]); l=np.array([1.,-5.,-5.]); u=np.array([1.,5.,5.])
x,info=StuartLandauLagrangeDirect(T_solve=3.0).solve(P,q,A,l,u)
print(f'x={x.round(3)}, status={info[\"status\"]}, time={info[\"solve_time_ms\"]:.0f}ms')
assert abs(x[0]-0.5)<0.03 and abs(x[1]-0.5)<0.03, f'FAILED: x={x}'
print('SL SOLVER: PASS')
"

# Phase 3: MuJoCo env
python -c "
import os; os.environ['MUJOCO_GL']='osmesa'
from src.simulation.envs.xarm_env import XArmEnv
import numpy as np
env=XArmEnv(render_mode='offscreen')
obs=env.reset()
print(f'q={obs[\"q\"].shape} rgb={obs[\"rgb\"].shape} rgb_sum={obs[\"rgb\"].sum()}')
tau=np.zeros(8); tau[0]=5.
for _ in range(30): obs,_,_,_=env.step(tau)
print(f'q[0] after 30 steps with tau[0]=5: {obs[\"q\"][0]:.4f}')
assert abs(obs['q'][0])>0.01,'ARM DIDNT MOVE'
env.close(); print('MUJOCO ENV: PASS')
"

# Phase 5: Fusion encoder
python -c "
from src.fusion.real_fusion_encoder import RealFusionEncoder
import numpy as np
rng=np.random.default_rng(0)
obs1={'rgb':rng.integers(0,200,(84,84,3),dtype=np.uint8),'state':np.zeros(7)}
obs2={'rgb':rng.integers(0,200,(84,84,3),dtype=np.uint8),'state':np.zeros(7)}
enc=RealFusionEncoder.mode_full()
f1=enc.encode(obs1); f2=enc.encode(obs2)
print(f'M4 dim={f1.shape[0]} (expected 320)')
print(f'Different: {not np.allclose(f1,f2)}')
assert f1.shape==(320,) and not np.allclose(f1,f2)
print('FUSION: PASS')
"
```

---

## WHAT TO DO IF SOMETHING IS BROKEN

### MuJoCo won't render headlessly
```bash
export MUJOCO_GL=osmesa
# If osmesa missing: sudo apt-get install libosmesa6-dev libgl1-mesa-glx
# If still failing, use MUJOCO_GL=egl (requires GPU driver)
```

### SL solver ODE diverges
The Arrow-Hurwicz ODE must be exactly:
```
dx/dt      = [ (μ - |x|²)x - Px - q - Aᵀ(λ_up - λ_lo) ] / τ_x
dλ_up/dt   = max(0, Ax - u) / τ_λ
dλ_lo/dt   = max(0, l - Ax) / τ_λ
```
Check signs carefully. If oscillating, reduce tau_x (try 0.5).

### OSQP import error
```bash
pip install osqp
python -c "import osqp; print(osqp.__version__)"
```
OSQP requires scipy sparse matrices — always convert P and A with `scipy.sparse.csc_matrix`.

### LeRobot dataset unavailable
Dataset tests use `@pytest.mark.skipif(not DATASET_AVAILABLE, ...)`. They should SKIP, not FAIL. If they fail with code errors (not "unavailable"), fix the code.

### SmolVLA can't load
Use MockVLA for all testing. Document in AGENT_STATE.md: "SmolVLA not loaded — [reason]. All VLA operations use MockVLA with source='MOCK'."

---

## WHAT THE AGENT PRODUCES (DELIVERABLES)

At completion, the following must exist and work:

```
# Tests pass
pytest tests/ -v --tb=short   ← all green (dataset tests may be skipped)

# Benchmarks have real results
ls evaluation/results/*.json   ← B1, B2, B3, B4, B5 JSONs with actual numbers

# GIFs show arm motion  
ls outputs/*.gif               ← demo_osqp.gif, demo_sl.gif (>50KB each)

# Webapp serves correctly
python webapp/server.py &
curl http://localhost:3000/           ← existing QP inspector (unchanged)
curl http://localhost:3000/mujoco     ← new MuJoCo viz page
curl http://localhost:3000/dashboard  ← new benchmark dashboard

# Webapp files are unchanged
git diff webapp/static/   ← ZERO changes
```

---

## AGENT MEMORY PROTOCOL

Every time you complete a phase, write to `docs/agent/AGENT_STATE.md`:

```markdown
## Phase N — [NAME] — [COMPLETE/IN PROGRESS/BLOCKED]
Date: [timestamp]

### Tests Run
[paste the pytest output line: "X passed, Y failed in Z.Xs"]

### What Was Done
[bullet list of files created/modified]

### Issues Encountered
[honest description of any problems and how they were resolved]

### Blockers
[anything that couldn't be resolved; why]

### Ready for Phase N+1?
[YES/NO]
```

**Do not summarize in vague language.** "Tests pass" is not enough. Paste the actual output.

---

*End of coding prompt. Begin with: `cat docs/agent/00_INDEX.md`*
