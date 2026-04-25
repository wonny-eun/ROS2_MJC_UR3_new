#!/usr/bin/env python3
"""Visualize MuJoCo rl_camera topics in OpenCV windows."""

import os
import threading

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def _jet_on_gray(gray_u8: np.ndarray) -> np.ndarray:
    """BGR uint8 colormap for OpenCV display."""
    return cv2.applyColorMap(gray_u8, cv2.COLORMAP_JET)


class MujocoRlCameraPreview(Node):
    def __init__(self):
        super().__init__("mujoco_rl_camera_preview")
        self.declare_parameter("rgb_topic", "/rl_camera/color")
        self.declare_parameter("depth_topic", "/rl_camera/depth")
        self.declare_parameter("show_rgb", True)
        self.declare_parameter("show_depth", True)
        self.declare_parameter("display_hz", 15.0)

        self._rgb = None
        self._depth_bgr = None
        self._lock = threading.Lock()
        self._show_rgb = self.get_parameter("show_rgb").get_parameter_value().bool_value
        self._show_depth = self.get_parameter("show_depth").get_parameter_value().bool_value
        hz = float(self.get_parameter("display_hz").get_parameter_value().double_value)
        period = 1.0 / max(hz, 1.0)
        self.create_timer(period, self._display_tick)

        self._gui_ok = bool(os.environ.get("DISPLAY"))
        if not self._gui_ok:
            self.get_logger().error(
                "DISPLAY is not set. Camera preview window cannot open. "
                "Set DISPLAY (for example ':0') and relaunch."
            )
        else:
            if self._show_rgb:
                cv2.namedWindow("rl_camera RGB", cv2.WINDOW_NORMAL)
            if self._show_depth:
                cv2.namedWindow("rl_camera depth", cv2.WINDOW_NORMAL)

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        if self._show_rgb:
            self.create_subscription(Image, rgb_topic, self._on_rgb, 1)
            self.get_logger().info(f"Subscribing RGB: {rgb_topic}")
        if self._show_depth:
            self.create_subscription(Image, depth_topic, self._on_depth, 1)
            self.get_logger().info(f"Subscribing depth: {depth_topic}")

    def _on_rgb(self, msg: Image):
        if msg.encoding != "rgb8" or msg.height * msg.width * 3 != len(msg.data):
            self.get_logger().warn("Expected rgb8 Image")
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        with self._lock:
            self._rgb = arr.copy()

    def _on_depth(self, msg: Image):
        try:
            if msg.encoding != "32FC1":
                self.get_logger().warn("Expected depth 32FC1")
                return
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            d = np.clip(arr, 0.0, 5.0)
            gray = (d / 5.0 * 255.0).astype(np.uint8)
            colored = _jet_on_gray(gray)
            with self._lock:
                self._depth_bgr = colored
        except Exception as e:
            self.get_logger().warn(f"Depth decode failed: {e}")

    def _display_tick(self):
        if not self._gui_ok:
            return
        with self._lock:
            rgb = None if self._rgb is None else self._rgb.copy()
            depth = None if self._depth_bgr is None else self._depth_bgr.copy()

        if self._show_rgb and rgb is not None:
            cv2.imshow("rl_camera RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if self._show_depth and depth is not None:
            cv2.imshow("rl_camera depth", depth)
        # keep HighGUI responsive
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = MujocoRlCameraPreview()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
