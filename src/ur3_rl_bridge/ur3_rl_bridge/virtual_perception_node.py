#!/usr/bin/env python3

from typing import List

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster


class VirtualPerceptionNode(Node):
    """Publishes camera->target TF as a sim-first stand-in for real perception."""

    def __init__(self) -> None:
        super().__init__("virtual_perception_node")

        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("object_frame", "target_object")
        self.declare_parameter("publish_rate_hz", 30.0)
        self.declare_parameter("object_pose_camera_xyz_quat", [0.30, 0.00, 0.25, 0.0, 0.0, 0.0, 1.0])

        self._br = TransformBroadcaster(self)
        rate = float(self.get_parameter("publish_rate_hz").value)
        self._timer = self.create_timer(max(1.0 / rate, 1e-3), self._on_timer)
        self.get_logger().info("Virtual perception TF publisher started.")

    def _on_timer(self) -> None:
        pose: List[float] = list(self.get_parameter("object_pose_camera_xyz_quat").value)
        if len(pose) != 7:
            self.get_logger().warning("object_pose_camera_xyz_quat must be length 7; skipping publish")
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = str(self.get_parameter("camera_frame").value)
        t.child_frame_id = str(self.get_parameter("object_frame").value)
        t.transform.translation.x = float(pose[0])
        t.transform.translation.y = float(pose[1])
        t.transform.translation.z = float(pose[2])
        t.transform.rotation.x = float(pose[3])
        t.transform.rotation.y = float(pose[4])
        t.transform.rotation.z = float(pose[5])
        t.transform.rotation.w = float(pose[6])
        self._br.sendTransform(t)


def main() -> None:
    rclpy.init()
    node = VirtualPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

