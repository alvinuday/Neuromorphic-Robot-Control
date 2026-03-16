"""
MJPC: Multi-step Joint Motion Planning & Control
=================================================

Integrated module for robot motion planning with MPC control.

Components:
  - motion_planning: Quintic trajectory generation
  - arm_mpc: Model Predictive Controller with MuJoCo
  - sensors: Robot state estimation
"""

from .motion_planning import SmoothTrajectoryPlanner, MotionPlanningSequence

__all__ = [
    'SmoothTrajectoryPlanner',
    'MotionPlanningSequence',
]
