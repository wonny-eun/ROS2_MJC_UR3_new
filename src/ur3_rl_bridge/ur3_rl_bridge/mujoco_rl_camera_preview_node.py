#!/usr/bin/env python3
"""Visualize MuJoCo rl_camera topics without OpenCV (avoids NumPy 2 / cv2 binary mismatches)."""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def _jet_on_gray(gray_u8: np.ndarray) -> np.ndarray:
    """RGB uint8 colormap from scalar 0–255 (no OpenCV)."""
    try:
        import matplotlib.cm as cm
    except ImportError:
        return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    rgba = cm.jet(gray_u8.astype(np.float32) / 255.0)  # h,w,4
    return (rgba[:, :, :3] * 255.0).astype(np.uint8)


class MujocoRlCameraPreview(Node):
    def __init__(self):
        super().__init__('mujoco_rl_camera_preview')
        self.declare_parameter('rgb_topic', '/rl_camera/color')
        self.declare_parameter('depth_topic', '/rl_camera/depth')
        self.declare_parameter('show_depth', True)
        self.declare_parameter('display_hz', 15.0)

        self._rgb = None
        self._depth_rgb = None
        self._im_rgb = None
        self._im_depth = None
        self._lock = threading.Lock()
        self._show_depth = self.get_parameter('show_depth').get_parameter_value().bool_value
        hz = float(self.get_parameter('display_hz').get_parameter_value().double_value)
        period = 1.0 / max(hz, 1.0)
        self.create_timer(period, self._display_tick)

        import matplotlib

        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt

        self._plt = plt
        ncols = 2 if self._show_depth else 1
        self._fig, self._axes = plt.subplots(1, ncols, figsize=(4 * ncols, 3))
        if ncols == 1:
            self._ax_rgb = self._axes
            self._ax_depth = None
        else:
            self._ax_rgb, self._ax_depth = self._axes
        self._ax_rgb.set_title('rl_camera RGB')
        self._ax_rgb.axis('off')
        if self._ax_depth is not None:
            self._ax_depth.set_title('rl_camera depth')
            self._ax_depth.axis('off')
        self._fig.tight_layout()
        self._fig.show()
        plt.ion()

        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.create_subscription(Image, rgb_topic, self._on_rgb, 1)
        self.get_logger().info('Subscribing RGB: %s' % rgb_topic)
        if self._show_depth:
            self.create_subscription(Image, depth_topic, self._on_depth, 1)
            self.get_logger().info('Subscribing depth: %s' % depth_topic)

    def _on_rgb(self, msg: Image):
        if msg.encoding != 'rgb8' or msg.height * msg.width * 3 != len(msg.data):
            self.get_logger().warn('Expected rgb8 Image')
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        with self._lock:
            self._rgb = arr.copy()

    def _on_depth(self, msg: Image):
        try:
            if msg.encoding != '32FC1':
                self.get_logger().warn('Expected depth 32FC1')
                return
            arr = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            d = np.clip(arr, 0.0, 5.0)
            gray = (d / 5.0 * 255.0).astype(np.uint8)
            colored = _jet_on_gray(gray)
            with self._lock:
                self._depth_rgb = colored
        except Exception as e:
            self.get_logger().warn('Depth decode failed: %s' % e)

    def _display_tick(self):
        with self._lock:
            rgb = None if self._rgb is None else self._rgb.copy()
            depth = None if self._depth_rgb is None else self._depth_rgb.copy()
        # Do not call imshow() every frame — it stacks AxesImage artists and TkAgg often keeps
        # showing the first frame. Update a single image with set_data instead.
        if rgb is not None:
            if self._im_rgb is None:
                self._im_rgb = self._ax_rgb.imshow(rgb)
            else:
                self._im_rgb.set_data(rgb)
        if self._show_depth and self._ax_depth is not None and depth is not None:
            if self._im_depth is None:
                self._im_depth = self._ax_depth.imshow(depth)
            else:
                self._im_depth.set_data(depth)
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        self._plt.pause(0.001)


def main():
    rclpy.init()
    node = MujocoRlCameraPreview()
    try:
        # spin_once so SIGINT/SIGTERM can stop the process (spin() blocks badly with matplotlib).
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            import matplotlib.pyplot as plt

            plt.close('all')
        except Exception:
            pass
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
