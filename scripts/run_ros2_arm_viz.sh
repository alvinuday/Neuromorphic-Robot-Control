#!/usr/bin/env bash
# One-command run: build (if needed), source, and launch the 2-DOF arm MPC in RViz.
# From repo root:
#   ./scripts/run_ros2_arm_viz.sh           # with RViz
#   ./scripts/run_ros2_arm_viz.sh --no-rviz # nodes only (e.g. you open RViz yourself)

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure ROS2 is on PATH (conda ros_env)
export PATH="/Users/alvin/y/envs/ros_env/bin:${PATH:-}"

# Build if install missing
if [[ ! -f "$REPO_ROOT/install/setup.bash" ]]; then
  echo "Building ros2_arm_viz..."
  colcon build --packages-select ros2_arm_viz
fi

# Source base ROS2 then workspace (must use bash: zsh breaks colcon setup.bash)
source /Users/alvin/y/envs/ros_env/setup.bash
source "$REPO_ROOT/install/setup.bash"
export NEUROMORPHIC_ARM_PROJECT_ROOT="$REPO_ROOT"

# Optional: fix RViz libfreetype on macOS
export DYLD_LIBRARY_PATH="${CONDA_PREFIX:-/Users/alvin/y/envs/ros_env}/lib:${DYLD_LIBRARY_PATH:-}"

# MPC node needs casadi, osqp, etc. in the same env as ros2
if ! python3 -c "import casadi" 2>/dev/null; then
  echo "Install project deps in this env first: pip install -r requirements.txt"
  exit 1
fi

USE_RVIZ="true"
for arg in "$@"; do
  if [[ "$arg" == "--no-rviz" ]]; then USE_RVIZ="false"; break; fi
done

echo "Launching (use_rviz=$USE_RVIZ). Stop with Ctrl+C."
exec ros2 launch ros2_arm_viz display_arm_mpc.launch.py "use_rviz:=$USE_RVIZ"
