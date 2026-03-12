# Neuromorphic Robot Control - 2-DOF Arm MPC Dataset

This project generates a dataset of Quadratic Programming (QP) problems approximating the Model Predictive Control (MPC) of a 2-DOF planar robot arm. It also includes a simulation of a Neuromorphic "Spin-Hall Oscillator" (SHO) solver to compare against a classical OSQP solver.

## Setup

1.  **Activate Environment**:
    ```bash
    source .venv/bin/activate
    ```
2.  **Install Dependencies** (if not done):
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script from the project root:

```bash
# Important: Set PYTHONPATH to include current directory
export PYTHONPATH=$PYTHONPATH:.

# Generate Dataset (OSQP mode)
python src/main.py --mode osqp --steps 50 --dataset_dir data/my_dataset

# Run Comparison (OSQP vs SHO)
python src/main.py --mode compare --steps 50 --dataset_dir data/comparison_run
```

## Dataset Structure

The `data/` directory will contain folders for each run. Inside a run folder:
- `qp_step_XXXX.npz`: compressed numpy file containing `Q`, `p`, `A_eq`, `b_eq`, `A_ineq`, `k_ineq` for that time step.
- `metadata.csv`: Summary of steps.
- `sim.mp4`: Video visualization of the arm.
- `sim.png`: Static plots of angles/velocities.

## ROS2 and RViz visualization

You can visualize the 2-DOF arm running MPC in **RViz2**. The ROS2 workspace is **this repo** (package under `src/ros2_arm_viz/`).

- **One-command run** (from repo root, after one-time setup):  
  `./scripts/run_ros2_arm_viz.sh`
- **Headless verification** (no RViz; for CI or to confirm the pipeline):  
  `./scripts/verify_ros2_arm_viz.sh`  
  Starts the launch with `use_rviz:=false`, then checks that `/joint_states` and `/robot_description` are published. Exit 0 = pass.
- **One-time setup**: In your ROS2 conda env (e.g. `ros_env`):  
  `pip install -r requirements.txt` and `micromamba install -c conda-forge freetype`.  
  Then build: `colcon build --packages-select ros2_arm_viz` (with `PATH` including your ROS2 env).
- **Full guide**: [docs/ROS2_RVIZ_SETUP.md](docs/ROS2_RVIZ_SETUP.md) — fixing RViz freetype, how TF/joint_states/robot_state_publisher work, and troubleshooting.

## Solvers
- **OSQP**: Classical ADMM-based QP solver. Used as the baseline.
- **SHO**: Simulated Ising Machine using coupled phase oscillators (Kuramoto model) to verify the feasibility of neuromorphic control.
