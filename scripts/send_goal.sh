#!/usr/bin/env bash
# Publish a new goal to the MPC node. Run this in a second terminal while the launch is running.
# Usage: ./scripts/send_goal.sh [q1] [q2]   (radians; default 0.5 0.8)
# Example: ./scripts/send_goal.sh 0 0
#          ./scripts/send_goal.sh 1.57 0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
q1="${1:-0.5}"
q2="${2:-0.8}"

export PATH="${PATH:-}:/Users/alvin/y/envs/ros_env/bin"
source /Users/alvin/y/envs/ros_env/setup.bash 2>/dev/null || true
source "$REPO_ROOT/install/setup.bash" 2>/dev/null || true

echo "Publishing goal q1=$q1, q2=$q2 (rad) to /goal_joint_position"
echo "Make sure the launch is running in another terminal (./scripts/run_ros2_arm_viz.sh)"
ros2 topic pub --once /goal_joint_position std_msgs/Float64MultiArray "{data: [$q1, $q2]}"
