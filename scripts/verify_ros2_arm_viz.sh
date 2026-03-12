#!/usr/bin/env bash
# Headless verification: start the ROS2 arm viz launch (no RViz), then run a
# Python checker that subscribes to /joint_states and /robot_description.
# Exit 0 only if both topics get data. Use in CI or to confirm the pipeline.
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PATH="${PATH:-}:/Users/alvin/y/envs/ros_env/bin"
export CONDA_PREFIX="${CONDA_PREFIX:-/Users/alvin/y/envs/ros_env}"
export NEUROMORPHIC_ARM_PROJECT_ROOT="$REPO_ROOT"

# Ensure workspace is built and sourced
if [[ ! -f "$REPO_ROOT/install/setup.bash" ]]; then
  echo "Build first: colcon build --packages-select ros2_arm_viz"
  exit 1
fi

source /Users/alvin/y/envs/ros_env/setup.bash
source "$REPO_ROOT/install/setup.bash"

echo "Starting launch (use_rviz:=false)..."
./scripts/run_ros2_arm_viz.sh --no-rviz &
LAUNCH_PID=$!
trap 'kill $LAUNCH_PID 2>/dev/null; wait $LAUNCH_PID 2>/dev/null; true' EXIT

echo "Waiting 10s for nodes and topics..."
sleep 10

echo "Running verification (subscribe to /joint_states and /robot_description)..."
if python3 "$REPO_ROOT/tests/verify_ros2_arm_viz.py"; then
  echo "Verification passed."
  exit 0
else
  echo "Verification failed."
  exit 1
fi
