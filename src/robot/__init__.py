"""Robot abstraction and configuration module."""

from src.robot.robot_config import (
    RobotConfig,
    JointSpec,
    RobotManager,
    create_3dof_arm,
    create_cobotta_6dof
)

__all__ = [
    'RobotConfig',
    'JointSpec',
    'RobotManager',
    'create_3dof_arm',
    'create_cobotta_6dof'
]
