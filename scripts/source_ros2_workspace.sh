#!/usr/bin/env bash
# Source base ROS2 (conda ros_env) and this repo's workspace so you can run
#   ros2 launch ros2_arm_viz display_arm_mpc.launch.py
# Use from bash:  bash scripts/source_ros2_workspace.sh
# (Under zsh, colcon's setup.bash resolves paths wrong; run this script with bash.)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1) Base ROS2 from conda (robot_state_publisher, rviz2, etc.)
CONDA_ROS="/Users/alvin/y/envs/ros_env/setup.bash"
if [[ -f "$CONDA_ROS" ]]; then
  # shellcheck source=/dev/null
  source "$CONDA_ROS"
else
  echo "Not found: $CONDA_ROS (activate ros_env or set CONDA_ROS)"
  exit 1
fi

# 2) This repo's workspace (ros2_arm_viz)
if [[ ! -f "$REPO_ROOT/install/setup.bash" ]]; then
  echo "Build first from repo root: colcon build --packages-select ros2_arm_viz"
  exit 1
fi
# shellcheck source=/dev/null
source "$REPO_ROOT/install/setup.bash"

export NEUROMORPHIC_ARM_PROJECT_ROOT="$REPO_ROOT"
echo "ROS2 workspace ready (repo=$REPO_ROOT)"
