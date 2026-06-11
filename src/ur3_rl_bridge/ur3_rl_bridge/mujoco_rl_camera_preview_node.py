#!/usr/bin/env python3
"""Visualize ROS camera RGB/depth topics in OpenCV windows (MuJoCo sim or RealSense)."""

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


def _depth_meters_to_colormap_bgr(depth_m: np.ndarray, max_depth_m: float) -> np.ndarray:
    d = np.clip(depth_m, 0.0, max_depth_m)
    gray = (d / max_depth_m * 255.0).astype(np.uint8)
    return _jet_on_gray(gray)


class MujocoRlCameraPreview(Node):
    def __init__(self):
        super().__init__("mujoco_rl_camera_preview")
        self.declare_parameter("rgb_topic", "/rl_camera/color")
        self.declare_parameter("depth_topic", "/rl_camera/depth")
        self.declare_parameter("show_rgb", True)
        self.declare_parameter("show_depth", True)
        self.declare_parameter("display_hz", 15.0)
        self.declare_parameter("depth_max_m", 5.0)
        self.declare_parameter("rgb_window_title", "rl_camera RGB")
        self.declare_parameter("depth_window_title", "rl_camera depth")

        self._rgb_bgr = None
        self._depth_bgr = None
        self._lock = threading.Lock()
        self._show_rgb = self.get_parameter("show_rgb").get_parameter_value().bool_value
        self._show_depth = self.get_parameter("show_depth").get_parameter_value().bool_value
        self._depth_max_m = float(self.get_parameter("depth_max_m").get_parameter_value().double_value)
        self._rgb_window = self.get_parameter("rgb_window_title").get_parameter_value().string_value
        self._depth_window = self.get_parameter("depth_window_title").get_parameter_value().string_value
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
                cv2.namedWindow(self._rgb_window, cv2.WINDOW_NORMAL)
            if self._show_depth:
                cv2.namedWindow(self._depth_window, cv2.WINDOW_NORMAL)

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        if self._show_rgb:
            self.create_subscription(Image, rgb_topic, self._on_rgb, 1)
            self.get_logger().info(f"Subscribing RGB: {rgb_topic}")
        if self._show_depth:
            self.create_subscription(Image, depth_topic, self._on_depth, 1)
            self.get_logger().info(f"Subscribing depth: {depth_topic}")

    def _on_rgb(self, msg: Image) -> None:
        try:
            if msg.encoding == "rgb8":
                if msg.height * msg.width * 3 != len(msg.data):
                    raise ValueError("rgb8 size mismatch")
                rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "bgr8":
                if msg.height * msg.width * 3 != len(msg.data):
                    raise ValueError("bgr8 size mismatch")
                bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3)).copy()
            else:
                self.get_logger().warn(f"Unsupported RGB encoding '{msg.encoding}' (expected rgb8 or bgr8)")
                return
            with self._lock:
                self._rgb_bgr = bgr
        except Exception as exc:
            self.get_logger().warn(f"RGB decode failed: {exc}")

    def _on_depth(self, msg: Image) -> None:
        try:
            if msg.encoding == "32FC1":
                arr = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
                colored = _depth_meters_to_colormap_bgr(arr, self._depth_max_m)
            elif msg.encoding == "16UC1":
                arr = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
                depth_m = arr.astype(np.float32) * 0.001
                colored = _depth_meters_to_colormap_bgr(depth_m, self._depth_max_m)
            else:
                self.get_logger().warn(
                    f"Unsupported depth encoding '{msg.encoding}' (expected 32FC1 or 16UC1)"
                )
                return
            with self._lock:
                self._depth_bgr = colored
        except Exception as exc:
            self.get_logger().warn(f"Depth decode failed: {exc}")

    def _display_tick(self) -> None:
        if not self._gui_ok:
            return
        with self._lock:
            rgb = None if self._rgb_bgr is None else self._rgb_bgr.copy()
            depth = None if self._depth_bgr is None else self._depth_bgr.copy()

        if self._show_rgb and rgb is not None:
            cv2.imshow(self._rgb_window, rgb)
        if self._show_depth and depth is not None:
            cv2.imshow(self._depth_window, depth)
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


if __name__ == "__main__":
    main()
