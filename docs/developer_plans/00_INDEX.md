# NEUROMORPHIC ROBOT CONTROL — AGENT PLAN INDEX
**Version**: 1.0 | **Date**: 2026-03-15 | **Author**: Plan written for coding agent

---

## HOW TO USE THESE FILES

Give the agent **all 7 files** at session start. They are meant to be read in order but referenced non-linearly during execution.

```
00_INDEX.md                     ← THIS FILE — philosophy, rules, overview
01_PHASE0_AUDIT_CLEANUP.md      ← Run first: audit existing code, delete junk
02_PHASES1_3_FOUNDATIONS.md     ← ABCs, SL solver, OSQP solver, MuJoCo env
03_PHASES4_6_MPC_FUSION_VLA.md  ← MPC controller, sensor fusion, SmolVLA
04_PHASES7_9_INTEGRATION.md     ← Dual controller, dataset, 5 benchmarks
05_PHASES10_12_WEBAPP_E2E.md    ← Webapp extension, E2E demo, final docs
CODING_PROMPT.md                ← Paste this into your agent (Claude Code etc.)
```

Place all 7 files inside `docs/agent/` in the repository before the agent starts.

---

## WHAT WE ARE BUILDING

A fully executable, validated dual-system robotic controller:

```
[LeRobot Dataset: utokyo_xarm_pick_and_place]
         │
         ▼
[Episode Loader] ──► reference trajectories
         │
         ▼
[MuJoCo xArm 6-DOF + gripper]  ◄──── [Sensor Fusion M0–M4]
         │                                    │
         │                          [RGB, Events(sim), LiDAR(sim), Proprio]
         ▼
[Dual System Controller]
    │
    ├── System 1 (sync, every 10ms)
    │       └── [XArmMPCController]
    │               ├── solver=SL  (thesis contribution)
    │               └── solver=OSQP (baseline)
    │
    └── System 2 (async, every 200ms)
            └── [SmolVLA Server]  ← MockVLA for offline, RealVLA optional
                      │
                      └── [TrajectoryBuffer] ──► System 1 reference
         │
         ▼
[Benchmarks B1–B5]  ──► JSON results ──► [Webapp Dashboard]
         │
         └── [QP Inspector + MuJoCo Viz + Ablation Charts]
                  (existing HTML untouched, new pages added)
```

---

## THE 5 DELIVERABLES (mapping to your requirements)

| Req | Deliverable | Plan File |
|-----|-------------|-----------|
| 1 | QP Inspector webapp extended with MuJoCo viz tabs, benchmark dashboard | `05_PHASES10_12_WEBAPP_E2E.md` |
| 2 | SL-MPC, OSQP-MPC, SmolVLA, Sensor Fusion — all implemented and tested | `02`, `03`, `04` |
| 3 | Modular robot config — works for any robot via YAML | `02_PHASES1_3_FOUNDATIONS.md` |
| 4 | MuJoCo sim, GIF recorder, interactive dashboards for xArm 6-DOF | `04_PHASES7_9_INTEGRATION.md` + `05` |
| 5 | LeRobot dataset eval, sensor ablation, SL vs OSQP comparison | `04_PHASES7_9_INTEGRATION.md` |

---

## ABSOLUTE RULES (agent must never break these)

### The Anti-Hallucination Rules

1. **A phase is only COMPLETE when `pytest -v` shows all its tests GREEN.** Never mark a phase done by inspecting code. Run the tests.
2. **Never write a benchmark result that isn't computed.** If a number appears in a JSON, it must come from actual execution, not from a hardcoded dict or a `time.sleep` fake.
3. **Never use `mock.patch` to make a test pass that would otherwise fail due to missing functionality.** Fix the functionality.
4. **MockVLA outputs must carry `"source": "MOCK"` in every response dict.** Never let mock data flow into a result labeled as SmolVLA performance.
5. **If a file is deleted, it goes to `_archive/` first and gets logged in `docs/agent/CLEANUP_LOG.md`.** Never silent-delete.
6. **Write `docs/agent/AGENT_STATE.md` after every phase.** This is the agent's working memory. It records: what passed, what failed, actual pytest output snippets, and why anything was skipped.

### The Webapp Sacred Rule

```
webapp/static/index.html   ─── DO NOT TOUCH
webapp/static/style.css    ─── DO NOT TOUCH  
webapp/static/app.js       ─── DO NOT TOUCH
```

New functionality is added via new files only. Verify with:
```bash
git diff webapp/static/  # Must show zero changes
```

### The Solver Honesty Rule

The SL solver is expected to be **slow** (2,000–8,000ms per QP on n=6 problems). This is NOT a bug. It is documented as the expected behavior of a continuous-time ODE-based solver. Do not "fix" the solver by reducing `T_solve` to fake speed or by returning early without convergence. If it's slow, document it. That is the thesis point.

---

## PHASE EXECUTION ORDER

```
Phase 0  → always first (audit before writing any code)
Phase 1  → always second (abstractions before implementations)
Phase 2  → solvers (SL + OSQP) — required by Phase 4
Phase 3  → MuJoCo env — required by Phases 7, 8, 9
Phase 4  → MPC controller — requires Phase 2 + 3
Phase 5  → Sensor fusion — requires Phase 3
Phase 6  → SmolVLA mock — independent
Phase 7  → Dual controller — requires 4 + 5 + 6
Phase 8  → Dataset + GIF recorder — requires Phase 3
Phase 9  → Benchmarks — requires all of 1–8
Phase 10 → Webapp extension — requires Phase 9 results
Phase 11 → E2E demo — requires all of 1–10
Phase 12 → Docs — last
```

**Skip-allowed (if env constraint)**:
- Real SmolVLA inference (needs GPU) → use MockVLA, document skip
- LeRobot dataset download (needs internet) → mark tests `skipif`, document skip
- MuJoCo headless GL → use osmesa or EGL, document setup

**Never skip**: Solver tests, MPC tests, fusion encoder tests, env loading tests.

---

## TARGET DIRECTORY STRUCTURE (canonical end state)

```
neuromorphic-robot/
├── README.md
├── requirements.txt
├── pyproject.toml
├── pytest.ini
│
├── config/
│   ├── config.yaml
│   ├── robots/
│   │   └── xarm_6dof.yaml          # ONLY robot config (4dof deleted)
│   └── solvers/
│       ├── sl_neuromorphic.yaml
│       └── osqp.yaml
│
├── src/
│   ├── core/
│   │   ├── base_controller.py
│   │   ├── base_env.py
│   │   └── base_solver.py          # NEW
│   ├── dynamics/
│   │   └── xarm_dynamics.py        # 6-DOF only (2dof, 3dof deleted)
│   ├── robot/
│   │   └── robot_config.py
│   ├── simulation/
│   │   ├── envs/xarm_env.py
│   │   ├── models/xarm_6dof.xml    # (or assets/ — one canonical location)
│   │   └── cameras/event_camera.py
│   ├── solver/
│   │   ├── stuart_landau_lagrange_direct.py
│   │   └── osqp_solver.py
│   ├── mpc/
│   │   └── xarm_mpc_controller.py
│   ├── fusion/
│   │   └── real_fusion_encoder.py
│   ├── smolvla/
│   │   ├── vla_server.py
│   │   ├── vla_client.py
│   │   ├── mock_vla.py             # NEW
│   │   └── action_processor.py
│   ├── integration/
│   │   ├── dual_controller.py
│   │   └── trajectory_buffer.py
│   └── visualization/
│       ├── episode_recorder.py     # NEW — GIF output
│       └── mujoco_dashboard.py     # NEW — offline charts
│
├── data/
│   ├── loaders/
│   │   ├── lerobot_loader.py
│   │   └── episode_player.py
│   └── cache/                      # gitignored
│
├── assets/
│   └── xarm_6dof.xml               # Only active MJCF
│
├── evaluation/
│   ├── benchmarks/
│   │   ├── run_mpc_solo.py         # B1
│   │   ├── run_vla_solo.py         # B2
│   │   ├── run_dual_system.py      # B3
│   │   ├── sensor_ablation.py      # B4
│   │   └── solver_comparison.py    # B5
│   └── results/                    # gitignored (only real runs go here)
│
├── webapp/
│   ├── server.py                   # NEW FastAPI backend
│   ├── static/
│   │   ├── index.html              # SACRED — do not touch
│   │   ├── style.css               # SACRED — do not touch
│   │   └── app.js                  # SACRED — do not touch
│   └── viz/
│       ├── mujoco_viz.html         # NEW
│       └── dashboard.html          # NEW
│
├── tests/
│   ├── unit/
│   │   ├── test_sl_solver.py
│   │   ├── test_osqp_solver.py
│   │   ├── test_xarm_dynamics.py
│   │   ├── test_fusion_encoder.py
│   │   └── test_trajectory_buffer.py
│   ├── integration/
│   │   ├── test_xarm_env.py
│   │   ├── test_mpc_controller.py
│   │   ├── test_dual_controller.py
│   │   └── test_vla_mock.py
│   └── e2e/
│       ├── test_episode_run.py
│       └── test_benchmarks_produce_output.py
│
├── docs/
│   └── agent/
│       ├── 00_INDEX.md             ← (this file)
│       ├── 01_PHASE0_AUDIT_CLEANUP.md
│       ├── 02_PHASES1_3_FOUNDATIONS.md
│       ├── 03_PHASES4_6_MPC_FUSION_VLA.md
│       ├── 04_PHASES7_9_INTEGRATION.md
│       ├── 05_PHASES10_12_WEBAPP_E2E.md
│       ├── CODING_PROMPT.md
│       ├── AGENT_STATE.md          ← agent writes here after each phase
│       ├── AUDIT_REPORT.md         ← output of Phase 0 audit script
│       └── CLEANUP_LOG.md          ← every deleted file logged here
│
└── _archive/                       # Deleted files (not in active tree)
```

---

## WHAT IS CERTAINLY HALLUCINATED IN THE EXISTING REPO

Based on analysis of the developer guide and typical AI-assisted development patterns:

| Component | Red Flag |
|-----------|----------|
| `evaluation/results/*.json` | Pre-existing JSON files with suspiciously neat metrics (e.g., `M4_latency: 27.9ms`) are almost certainly hardcoded, not measured. **Delete all and re-run.** |
| Tests claiming "100+ passing" | With no GPU and unverified MuJoCo, most integration tests were likely mocked. **Verify every test imports and runs real code.** |
| Sensor ablation results | `phase13_final.py` says "30 episodes × 5 modes" but results files have identical structures — high probability of fake loops. **Re-run from scratch.** |
| SL solver convergence | The ODE may exist but `T_solve=2.0s` may have been tuned to return early. Verify the validation QP: x = [0.5, 0.5]. |
| Fusion encoder features | If `extract_rgb_features()` returns `np.random.randn(128)`, it's fake. Verify features differ across images. |

---

## DEPENDENCY INSTALLATION (exact, in order)

```bash
# 1. Core scientific
pip install numpy scipy matplotlib pyyaml

# 2. MuJoCo (must be 3.x)
pip install mujoco
python -c "import mujoco; print(mujoco.__version__)"

# 3. QP Solver
pip install osqp

# 4. Web server
pip install fastapi uvicorn requests aiohttp Pillow

# 5. GIF
pip install imageio

# 6. Testing
pip install pytest pytest-asyncio

# 7. Dataset (optional — skip if no internet)
pip install datasets huggingface_hub

# 8. LeRobot + SmolVLA (optional — skip if no GPU)
pip install lerobot torch torchvision

# 9. Headless rendering (Linux)
# sudo apt-get install libosmesa6-dev libgl1-mesa-glx
export MUJOCO_GL=osmesa  # Add to .env or shell profile
```

---

*Next: Read `01_PHASE0_AUDIT_CLEANUP.md`*
