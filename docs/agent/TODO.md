# TODO — Phase 0-8 Task Checklist

## Phase 0: Governance & Setup

- [x] 0.1 — Create docs/agent/ memory files (AGENT_STATE.md, TODO.md, PROGRESS.md)
- [ ] 0.2 — Delete conflicting LSMO/OpenX dataset scripts (7 files)
- [ ] 0.3 — Build source-to-target migration matrix for existing code
- [ ] 0.4 — Create minimal config base (paths.yaml, robot.yaml)

## Phase 1: Canonical Structure Migration

- [ ] 1.1 — Create canonical top-level directories (data/, simulation/, sensors/, fusion/, smolvla, system/, evaluation/)
- [ ] 1.2 — Move existing SL solver to src/mpc/sl_solver.py
- [ ] 1.3 — Consolidate SmolVLA into single src/smolvla/ package
- [ ] 1.4 — Move environment logic to simulation/envs/xarm_env.py
- [ ] 1.5 — Move MJCF models to simulation/models/
- [ ] 1.6 — Consolidate controller/buffer to src/system/
- [ ] 1.7 — Consolidate benchmarks to evaluation/benchmarks/
- [ ] 1.8 — Create temporary compatibility shims for old import paths

## Phase 2: LeRobot Dataset Pipeline

- [ ] 2.1 — Implement data/download/download_dataset.py (with hash logging)
- [ ] 2.2 — Implement data/loaders/lerobot_loader.py
- [ ] 2.3 — Implement data/loaders/data_inspector.py
- [ ] 2.4 — Implement data/loaders/episode_player.py
- [ ] 2.5 — Download lerobot/xarm_lift_medium and verify
- [ ] 2.6 — Create logs/dataset_download.json with verification

## Phase 3: xArm 4-DOF Simulation & Sensors

- [ ] 3.1 — Create/validate simulation/models/xarm_4dof.xml
- [ ] 3.2 — Implement simulation/envs/xarm_env.py (XArmEnv class)
- [ ] 3.3 — Implement simulation/cameras/rgb_camera.py (84x84 render)
- [ ] 3.4 — Implement simulation/cameras/event_camera.py (v2e wrapper)
- [ ] 3.5 — Implement simulation/cameras/lidar_sensor.py (rangefinders)
- [ ] 3.6 — Implement sensors/event_processing.py
- [ ] 3.7 — Implement sensors/lidar_processing.py
- [ ] 3.8 — Create simulation/tests/ with env/camera/sensor tests

## Phase 4: Dynamics, MPC & SmolVLA Alignment

- [ ] 4.1 — Consolidate dynamics to canonical 4-DOF contract
- [ ] 4.2 — Move SL solver to src/mpc/ and extend to 4-DOF
- [ ] 4.3 — Validate dynamics/MPC property tests pass
- [ ] 4.4 — Implement smolvla/action_processor.py
- [ ] 4.5 — Validate SmolVLA client behavior and fallback semantics

## Phase 5: Full Integration

- [ ] 5.1 — Consolidate system/controller.py (dual system)
- [ ] 5.2 — Consolidate system/trajectory_buffer.py
- [ ] 5.3 — Wire full control loop with metrics logging
- [ ] 5.4 — Implement visualization/live_dashboard.py
- [ ] 5.5 — Implement visualization/episode_viewer.py
- [ ] 5.6 — Create sample integration test run

## Phase 6: Benchmarks B1-B5

- [ ] 6.1 — Implement evaluation/benchmarks/run_dataset_replay.py (B1)
- [ ] 6.2 — Implement evaluation/benchmarks/run_mpc_solo.py (B2)
- [ ] 6.3 — Implement evaluation/benchmarks/run_smolvla_only.py (B3)
- [ ] 6.4 — Implement evaluation/benchmarks/run_full_system.py (B4)
- [ ] 6.5 — Implement evaluation/benchmarks/run_sensor_ablation.py (B5)
- [ ] 6.6 — Run all benchmarks and log results

## Phase 7: Validation Gates Execution

- [ ] 7.1 — Execute Gate 0: Environment setup
- [ ] 7.2 — Execute Gate 1: Dataset audit
- [ ] 7.3 — Execute Gate 2: MuJoCo validation
- [ ] 7.4 — Execute Gate 3: Dynamics validation
- [ ] 7.5 — Execute Gate 4: SL-MPC validation
- [ ] 7.6 — Execute Gate 5: SmolVLA validation
- [ ] 7.7 — Execute Gate 6: Full system validation

## Phase 8: Hardening & Handoff

- [ ] 8.1 — Full regression test suite pass
- [ ] 8.2 — Remove temporary compatibility shims
- [ ] 8.3 — Clean up dead code and temp files
- [ ] 8.4 — Final documentation and progress matrix
