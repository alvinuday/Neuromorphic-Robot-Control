# Neuromorphic QP Solver for Robot Control

**Stuart-Landau + Direct Lagrange Multiplier solver for Model Predictive Control** on a 2-DOF planar robot arm.

**Status**: ✅ **Phases 1-6 COMPLETE** (25/25 tests passing, ~15s runtime)

## 📚 Documentation

All documentation is organized in the [`docs/`](docs) folder:

| Document | Purpose |
|----------|---------|
| [**INDEX.md**](docs/INDEX.md) | Main documentation index & navigation |
| [**01-QUICKSTART.md**](docs/01-QUICKSTART.md) | 5-minute quick start |
| [**02-GETTING_STARTED.md**](docs/02-GETTING_STARTED.md) | Installation & setup |
| [**03-SOLVERS.md**](docs/03-SOLVERS.md) | Solver descriptions (OSQP, iLQR, Neuromorphic) |
| [**04-TESTING.md**](docs/04-TESTING.md) | How to run tests & benchmarks |
| [**05-VISUALIZATION.md**](docs/05-VISUALIZATION.md) | Interactive MPC controller |
| [**06-BENCHMARKING.md**](docs/06-BENCHMARKING.md) | Benchmark methodology & results |
| [**07-THEORY.md**](docs/07-THEORY.md) | Theory, math, and background |
| [**ROADMAP.md**](docs/ROADMAP.md) | Implementation phases & future work |
| [**PROJECT_STATUS.txt**](docs/PROJECT_STATUS.txt) | Current project status |
| [**REFERENCE/**](docs/REFERENCE) | Technical API reference |

## 🚀 Quick Start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Test
python -m pytest tests/ -v

# 3. View interactive MPC controller
mjpython src/mujoco/mujoco_interactive_controller.py --task reach --controller osqp
```

See [**01-QUICKSTART.md**](docs/01-QUICKSTART.md) for more.

## 📊 Project Structure

```
src/
├── solver/               # QP solvers (SL+DirectLag, etc.)
├── benchmark/           # Solver benchmarking framework
├── mujoco/              # MuJoCo arm + interactive viewer
├── mpc/                 # MPC controller
├── dynamics/            # Robot arm dynamics
└── utils/              # Utilities

tests/                   # 25 tests (all passing)
docs/                    # Full documentation (see table above)
assets/                  # MuJoCo XML models
```

## ✅ Features

- **SL+DirectLag Solver**: Neuromorphic QP solver with direct Lagrange multipliers
- **MPC Controller**: 10-step horizon @ 500Hz
- **MuJoCo Integration**: 2-DOF arm with gravity dynamics
- **Interactive Viewer**: Multiple controllers (PID, OSQP-MPC, iLQR, Neuromorphic)
- **Benchmarking**: Compare solvers on QP performance & accuracy
- **Full Test Suite**: 25 tests covering all components

## 📖 Next Steps

Start with [**docs/INDEX.md**](docs/INDEX.md) to navigate all documentation.
