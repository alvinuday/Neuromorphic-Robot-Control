# Neuromorphic Robot Control - Master's Thesis

## Project Structure

```
.
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Test configuration
├── config/                          # Configuration files
│   ├── config.yaml
│   ├── logging.yaml
│   ├── robots/
│   └── solvers/
├── assets/                          # CAD/URDF files
│   ├── arm2dof.xml
│   └── arm3dof.xml
├── data/                            # Dataset storage
│   ├── cache/                       # LeRobot dataset cache
│   ├── dataset_001/                 # Local simulation data
│   ├── __init__.py
│   ├── loaders/
│   └── openx_cache/
├── docs/                            # Documentation
│   ├── 01-QUICKSTART.md
│   ├── 02-GETTING_STARTED.md
│   ├── [API reference & theory docs]
│   └── [archived phase documentation]
├── evaluation/                      # Results & benchmarks
│   ├── benchmarks/                  # Benchmark runners
│   ├── results/                     # JSON results from experiments
│   └── test_*.py                    # Integration tests
├── logs/                            # Log files
│   ├── archived/                    # Old logs
│   ├── bench.log                    # Latest benchmark log
│   └── final_bench.log
├── notebooks/                       # Jupyter notebooks
├── scripts/                         # Executable scripts
│   ├── README.md                    # Script documentation (START HERE)
│   ├── phase12_quick_test.py        # VLA benchmarking
│   ├── phase13_final.py             # Real sensor fusion ablation
│   ├── check_dataset_cache.py       # Utilities
│   ├── health_check_vla.py
│   ├── inspect_dataset.py
│   └── test_vla_server.py
├── src/                             # Main source code
│   ├── main.py
│   ├── core/                        # Core modules
│   ├── dynamics/                    # Dynamics models
│   ├── environments/                # Robot environments
│   ├── fusion/                      # Multi-modal sensor fusion
│   │   ├── encoders/
│   │   │   ├── real_fusion_simple.py    # ACTIVE: real sensor encoder
│   │   │   └── __init__.py
│   │   ├── sensors/
│   │   └── __init__.py
│   ├── integration/                 # VLA integration
│   ├── mpc/                         # Model predictive control
│   ├── mujoco/                      # MuJoCo simulation
│   ├── robot/                       # Robot control
│   ├── ros2_arm_viz/                # ROS2 visualization
│   ├── simulation/                  # Simulation environment
│   │   ├── cameras/                 # Camera simulators
│   │   │   └── event_camera_simple.py
│   │   ├── envs/
│   │   ├── models/
│   │   └── tests/
│   ├── smolvla/                     # SmolVLA client/server
│   │   ├── real_client.py
│   │   └── vla_production_server.py
│   ├── solver/                      # Optimization solvers
│   ├── system/                      # System utilities
│   ├── utils/                       # Helper utilities
│   ├── visualization/               # Plotting & visualization
│   └── __init__.py
├── tests/                           # Integration tests
│   ├── test_*.py                    # Various integration tests
│   └── __init__.py
├── test_encoder_simple.py           # Unit test for sensor encoder
├── vla/                             # VLA server (legacy location)
│   ├── vla_production_server.py     # Main VLA server
│   └── ...
└── web_app/                         # Web interface (if present)
```

## Quick Start

### 1. Environment Setup
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verify Installation
```bash
python3 test_encoder_simple.py          # Verify sensor encoder works
python3 scripts/check_dataset_cache.py  # Check dataset access
python3 scripts/health_check_vla.py     # Verify VLA server
```

### 3. Run Experiments
See [scripts/README.md](scripts/README.md) for detailed usage.

**Phase 12: Quick VLA test**
```bash
python3 scripts/phase12_quick_test.py
```

**Phase 13: Real sensor fusion ablation (30 episodes)**
```bash
python3 scripts/phase13_final.py
```

## Key Components

### Real Sensor Fusion (`src/fusion/encoders/real_fusion_simple.py`)
Extracts REAL sensor features from observations:
- **RGB**: Color/gradient/edge statistics (128-dim)
- **Events**: Frame difference optical flow (96-dim)
- **LiDAR**: Corner/edge/depth geometry (64-dim)
- **Proprioception**: Robot joint state (32-dim)
- **Output**: 256-dim embeddings across 5 fusion modes

### VLA Server (`src/smolvla/vla_production_server.py`)
Real SmolVLA server with:
- Multi-modal sensor fusion support
- Async query handling
- Memory optimization

### Datasets
- **LeRobot** (`utokyo_xarm_pick_and_place`): 102 episodes, 7490 frames, 3 cameras
- Cached in `data/cache/`

## Experiment Results

### Phase 13: Real Sensor Fusion Ablation (30 episodes × 5 modes)
- **File**: `evaluation/results/phase13_ablation_real_*.json`
- **Modes Tested**:
  - M0: RGB only
  - M1: RGB + Events
  - M2: RGB + LiDAR
  - M3: RGB + Proprioception
  - M4: Full Fusion
- **Runtime**: ~3-4 minutes
- **Key Finding**: All features extracted from REAL data (no mocking)

## Notes

- Removed 20+ redundant script variants during cleanup
- Archived old logs in `logs/archived/`
- Clean repository structure with organized modules
- Only active, tested scripts remain in `scripts/`

## Next Steps

1. Analyze Phase 13 ablation results
2. Create thesis visualizations
3. Phase 14: Full integration testing (if needed)

See [docs/](docs/) for detailed technical documentation.
