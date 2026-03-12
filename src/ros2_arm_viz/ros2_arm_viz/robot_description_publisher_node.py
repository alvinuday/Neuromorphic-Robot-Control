#!/usr/bin/env python3
"""Publishes robot_description (URDF string) to /robot_description so RViz RobotModel can display it."""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import String


def main(args=None):
    rclpy.init(args=args)
    node = Node("robot_description_publisher")

    node.declare_parameter("robot_description", "")
    desc = node.get_parameter("robot_description").get_parameter_value().string_value
    if not desc:
        node.get_logger().error("robot_description parameter is empty")
        rclpy.shutdown()
        return

    # Transient local = latched; late subscribers (e.g. RViz) still get the last message
    qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
    pub = node.create_publisher(String, "robot_description", qos)
    msg = String()
    msg.data = desc
    pub.publish(msg)
    node.get_logger().info("Published robot_description to /robot_description")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
