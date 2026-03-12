#!/usr/bin/env python3
"""
ROS2 node: runs the 2-DOF arm MPC simulation and publishes joint states to /joint_states
so RViz (with robot_state_publisher) can visualize the arm.
"""

import os
import sys

# Ensure project root is on path so we can import src.dynamics, src.mpc, src.solver
def _find_project_root():
    # When run from colcon install, __file__ is in install space; use env if set
    root = os.environ.get("NEUROMORPHIC_ARM_PROJECT_ROOT", "").strip()
    if root and os.path.isfile(os.path.join(root, "src", "main.py")):
        return root
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "requirements.txt")) or os.path.isfile(
            os.path.join(d, "src", "main.py")
        ):
            return d
        d = os.path.dirname(d)
    return None


_root = _find_project_root()
if _root and _root not in sys.path:
    sys.path.insert(0, _root)
if _root is None:
    raise RuntimeError(
        "Project root not found. Set NEUROMORPHIC_ARM_PROJECT_ROOT to the path of "
        "Neuromorphic-Robot-Control (the repo containing src/main.py), or run from that repo."
    )

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# Imports from the main project (require project root on path)
from src.dynamics.arm2dof import Arm2DOF
from src.mpc.qp_builder import MPCBuilder
from src.solver.osqp_solver import OSQPSolver


class JointStateMPCNode(Node):
    """Runs MPC simulation and publishes joint states at each step."""

    def __init__(self):
        super().__init__("joint_state_mpc_node")

        self.declare_parameter("steps", 200)
        self.declare_parameter("horizon", 20)
        self.declare_parameter("dt", 0.02)
        self.declare_parameter("q0", [0.0, 0.0])
        self.declare_parameter("dq0", [0.0, 0.0])
        self.declare_parameter("goal", [np.pi / 3, np.pi / 6])
        self.declare_parameter("publish_rate", 50.0)  # Hz

        steps = self.get_parameter("steps").value
        horizon = self.get_parameter("horizon").value
        dt = self.get_parameter("dt").value
        q0 = self.get_parameter("q0").value
        dq0 = self.get_parameter("dq0").value
        goal = self.get_parameter("goal").value
        publish_rate = self.get_parameter("publish_rate").value

        self.get_logger().info(
            f"MPC viz: steps={steps}, goal={goal}, dt={dt}, rate={publish_rate} Hz"
        )

        self.arm = Arm2DOF(m1=1.0, m2=1.0, l1=0.5, l2=0.5, g=9.81)
        self.mpc = MPCBuilder(self.arm, N=horizon, dt=dt)
        self.solver = OSQPSolver()

        self.x = np.concatenate([np.array(q0, dtype=float), np.array(dq0, dtype=float)])
        self.x_goal = np.array([goal[0], goal[1], 0.0, 0.0], dtype=float)
        self.dt = dt
        self.steps = steps
        self.step_index = 0

        self.pub = self.create_publisher(JointState, "joint_states", 10)
        self.timer = self.create_timer(1.0 / publish_rate, self.timer_callback)
        self._trajectory = None  # precomputed trajectory (optional replay mode)
        self.create_subscription(
            Float64MultiArray,
            "/goal_joint_position",
            self._goal_callback,
            10,
        )

    def _goal_callback(self, msg):
        if len(msg.data) >= 2:
            self.x_goal = np.array(
                [float(msg.data[0]), float(msg.data[1]), 0.0, 0.0],
                dtype=float,
            )
            self.step_index = 0  # restart MPC toward new goal
            self.get_logger().info(f"New goal: q1={msg.data[0]:.3f}, q2={msg.data[1]:.3f}")

    def timer_callback(self):
        if self.step_index >= self.steps:
            # Hold last pose
            self._publish_state(self.x)
            return

        # Build reference and QP, solve, step
        x_ref_traj = self.mpc.build_reference_trajectory(self.x, self.x_goal)
        qp_matrices = self.mpc.build_qp(self.x, x_ref_traj)
        z = self.solver.solve(qp_matrices)
        u = z[self.arm.nx : self.arm.nx + self.arm.nu]
        self.x = self.arm.step_dynamics(self.x, u, self.dt)
        self.step_index += 1

        self._publish_state(self.x)

    def _publish_state(self, x):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.name = ["joint1", "joint2"]
        msg.position = [float(x[0]), float(x[1])]
        msg.velocity = [float(x[2]), float(x[3])]
        msg.effort = []
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointStateMPCNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
