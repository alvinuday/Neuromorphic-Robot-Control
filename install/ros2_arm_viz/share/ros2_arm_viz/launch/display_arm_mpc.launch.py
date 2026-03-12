#!/usr/bin/env python3
"""Launch robot_state_publisher, MPC joint state node, and optionally RViz."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_dir = get_package_share_directory("ros2_arm_viz")
    urdf_path = os.path.join(pkg_dir, "urdf", "planar_arm_2dof.urdf")
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    use_rviz_arg = DeclareLaunchArgument(
        "use_rviz",
        default_value="true",
        description="Launch RViz2 for visualization",
    )
    use_rviz = LaunchConfiguration("use_rviz", default="true")

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    # Publish URDF to /robot_description so RViz RobotModel can subscribe (many setups don't publish it by default)
    robot_description_publisher_node = Node(
        package="ros2_arm_viz",
        executable="robot_description_publisher_node",
        name="robot_description_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    joint_state_mpc_node = Node(
        package="ros2_arm_viz",
        executable="joint_state_mpc_node",
        name="joint_state_mpc_node",
        output="screen",
    )

    # So RViz2 finds libfreetype etc. on macOS (conda); launch may not pass env to children
    rviz_env = {}
    if os.environ.get("DYLD_LIBRARY_PATH"):
        rviz_env["DYLD_LIBRARY_PATH"] = os.environ["DYLD_LIBRARY_PATH"]
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = os.path.join(conda_prefix, "lib")
        rviz_env["DYLD_LIBRARY_PATH"] = conda_lib + os.pathsep + rviz_env.get("DYLD_LIBRARY_PATH", "")

    # Start RViz after a short delay so /robot_description is already published (transient_local).
    rviz_config_path = os.path.join(pkg_dir, "rviz", "planar_arm.rviz")
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_path] if os.path.isfile(rviz_config_path) else [],
        condition=IfCondition(use_rviz),
        additional_env=rviz_env,
    )
    rviz_delayed = TimerAction(period=1.0, actions=[rviz_node])

    return LaunchDescription([
        use_rviz_arg,
        robot_state_publisher_node,
        robot_description_publisher_node,
        joint_state_mpc_node,
        rviz_delayed,
    ])
