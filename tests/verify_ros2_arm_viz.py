#!/usr/bin/env python3
"""
Headless verification: confirm /joint_states and /robot_description are published.
Run while the launch is running (use_rviz:=false). Exit 0 if both get data, 1 otherwise.
"""
import sys
import time

def main():
    try:
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
        from sensor_msgs.msg import JointState
        from std_msgs.msg import String
    except ImportError as e:
        print("SKIP: rclpy not available:", e, file=sys.stderr)
        return 0

    rclpy.init()
    node = Node("verify_ros2_arm_viz")

    joint_ok = [False]
    desc_ok = [False]

    def on_joint(msg):
        joint_ok[0] = True

    def on_desc(msg):
        desc_ok[0] = True

    node.create_subscription(JointState, "joint_states", on_joint, 10)
    # Publisher uses TRANSIENT_LOCAL (latched); subscriber must match to receive it
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    node.create_subscription(String, "robot_description", on_desc, qos)

    timeout = 15.0
    start = time.monotonic()
    while (time.monotonic() - start) < timeout:
        rclpy.spin_once(node, timeout_sec=0.5)
        if joint_ok[0] and desc_ok[0]:
            break

    try:
        node.destroy_node()
        rclpy.shutdown()
    except Exception:
        pass

    if joint_ok[0] and desc_ok[0]:
        print("OK: /joint_states and /robot_description both received.")
        return 0
    if not joint_ok[0]:
        print("FAIL: no message on /joint_states", file=sys.stderr)
    if not desc_ok[0]:
        print("FAIL: no message on /robot_description", file=sys.stderr)
    return 1

if __name__ == "__main__":
    sys.exit(main())
