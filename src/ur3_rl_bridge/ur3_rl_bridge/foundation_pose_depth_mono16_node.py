#!/usr/bin/env python3
"""Convert MuJoCo 32FC1 depth (meters) to MONO16 millimeters for Isaac ConvertMetricNode."""

from __future__ import annotations

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class FoundationPoseDepthMono16Node(Node):
    def __init__(self) -> None:
        super().__init__("foundation_pose_depth_mono16")
        self.declare_parameter("input_topic", "/rl_camera/noisy/depth")
        self.declare_parameter("output_topic", "/fp_bridge/depth_mono16")
        self.declare_parameter("invalid_depth_m", 0.0)

        in_topic = str(self.get_parameter("input_topic").value)
        out_topic = str(self.get_parameter("output_topic").value)
        self._invalid = float(self.get_parameter("invalid_depth_m").value)

        self._pub = self.create_publisher(Image, out_topic, 10)
        self.create_subscription(Image, in_topic, self._on_depth, 10)
        self.get_logger().info(f"Depth MONO16 bridge: {in_topic} -> {out_topic}")

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding != "32FC1":
            self.get_logger().warn(f"Expected 32FC1 depth, got '{msg.encoding}'", throttle_duration_sec=5.0)
            return
        depth_m = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
        depth_mm = np.clip(depth_m * 1000.0, 0.0, 65535.0).astype(np.uint16)
        if self._invalid == 0.0:
            depth_mm[~np.isfinite(depth_m) | (depth_m <= 0.0)] = 0

        out = Image()
        out.header = msg.header
        out.height = msg.height
        out.width = msg.width
        out.encoding = "mono16"
        out.is_bigendian = 0
        out.step = msg.width * 2
        out.data = depth_mm.tobytes()
        self._pub.publish(out)


def main() -> None:
    rclpy.init()
    node = FoundationPoseDepthMono16Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
