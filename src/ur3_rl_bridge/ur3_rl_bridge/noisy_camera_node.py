#!/usr/bin/env python3
"""Add RealSense-like RGB/depth noise to live MuJoCo camera topics."""

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


def _apply_depth_shadow(
    depth: np.ndarray,
    *,
    direction: str,
    k_px_m: float,
    min_radius_px: int,
    max_radius_px: int,
    edge_threshold_m: float,
    max_depth_m: float,
) -> np.ndarray:
    if k_px_m <= 0.0 or max_radius_px <= 0 or edge_threshold_m <= 0.0:
        return depth

    out = depth.copy()
    valid = np.isfinite(depth) & (depth > 0.0)
    if depth.shape[1] < 2:
        return out

    left = depth[:, :-1]
    right = depth[:, 1:]
    valid_pair = valid[:, :-1] & valid[:, 1:]
    direction = direction.lower()

    if direction == "left":
        # Background is on the left, foreground object edge is on the right.
        edge = valid_pair & ((left - right) > edge_threshold_m)
        edge_rows, edge_cols = np.nonzero(edge)
        fg_depth = right[edge_rows, edge_cols]
        for row, col, z in zip(edge_rows.tolist(), edge_cols.tolist(), fg_depth.tolist()):
            if max_depth_m > 0.0 and float(z) > max_depth_m:
                continue
            radius = int(np.clip(round(k_px_m / max(float(z), 1e-6)), min_radius_px, max_radius_px))
            x0 = max(0, col - radius + 1)
            out[row, x0 : col + 1] = 0.0
    elif direction == "right":
        # Foreground object edge is on the left, background is on the right.
        edge = valid_pair & ((right - left) > edge_threshold_m)
        edge_rows, edge_cols = np.nonzero(edge)
        fg_depth = left[edge_rows, edge_cols]
        width = depth.shape[1]
        for row, col, z in zip(edge_rows.tolist(), edge_cols.tolist(), fg_depth.tolist()):
            if max_depth_m > 0.0 and float(z) > max_depth_m:
                continue
            radius = int(np.clip(round(k_px_m / max(float(z), 1e-6)), min_radius_px, max_radius_px))
            x0 = col + 1
            x1 = min(width, x0 + radius)
            out[row, x0:x1] = 0.0

    return out


def _copy_header(src: Image, dst: Image) -> None:
    dst.header.stamp = src.header.stamp
    dst.header.frame_id = src.header.frame_id


class NoisyCameraNode(Node):
    def __init__(self):
        super().__init__("noisy_camera_node")
        self.declare_parameter("rgb_in_topic", "/rl_camera/color")
        self.declare_parameter("depth_in_topic", "/rl_camera/depth")
        self.declare_parameter("rgb_out_topic", "/rl_camera/noisy/color")
        self.declare_parameter("depth_out_topic", "/rl_camera/noisy/depth")
        self.declare_parameter("rgb_noise_sigma", 3.0)
        self.declare_parameter("rgb_salt_pepper_prob", 0.003)
        self.declare_parameter("rgb_brightness_jitter", 8.0)
        self.declare_parameter("rgb_contrast_jitter", 0.08)
        self.declare_parameter("depth_noise_sigma_m", 0.002)
        self.declare_parameter("depth_quant_step_m", 0.001)
        self.declare_parameter("depth_dropout_prob", 0.02)
        self.declare_parameter("depth_edge_dropout_prob", 0.10)
        self.declare_parameter("depth_edge_threshold_m", 0.01)
        self.declare_parameter("depth_shadow_enable", True)
        self.declare_parameter("depth_shadow_direction", "left")
        self.declare_parameter("depth_shadow_k_px_m", 1.5)
        self.declare_parameter("depth_shadow_min_radius_px", 1)
        self.declare_parameter("depth_shadow_max_radius_px", 8)
        self.declare_parameter("depth_shadow_edge_threshold_m", 0.015)
        self.declare_parameter("depth_shadow_max_depth_m", 4.0)
        self.declare_parameter("seed", 0)

        self._rgb_sigma = float(self.get_parameter("rgb_noise_sigma").value)
        self._rgb_salt_pepper = float(self.get_parameter("rgb_salt_pepper_prob").value)
        self._rgb_brightness_jitter = float(self.get_parameter("rgb_brightness_jitter").value)
        self._rgb_contrast_jitter = float(self.get_parameter("rgb_contrast_jitter").value)
        self._depth_sigma = float(self.get_parameter("depth_noise_sigma_m").value)
        self._depth_quant = float(self.get_parameter("depth_quant_step_m").value)
        self._depth_dropout = float(self.get_parameter("depth_dropout_prob").value)
        self._depth_edge_dropout = float(self.get_parameter("depth_edge_dropout_prob").value)
        self._depth_edge_threshold = float(self.get_parameter("depth_edge_threshold_m").value)
        self._depth_shadow_enable = bool(self.get_parameter("depth_shadow_enable").value)
        self._depth_shadow_direction = str(self.get_parameter("depth_shadow_direction").value)
        self._depth_shadow_k = float(self.get_parameter("depth_shadow_k_px_m").value)
        self._depth_shadow_min_radius = int(self.get_parameter("depth_shadow_min_radius_px").value)
        self._depth_shadow_max_radius = int(self.get_parameter("depth_shadow_max_radius_px").value)
        self._depth_shadow_edge_threshold = float(self.get_parameter("depth_shadow_edge_threshold_m").value)
        self._depth_shadow_max_depth = float(self.get_parameter("depth_shadow_max_depth_m").value)
        self._rng = np.random.default_rng(int(self.get_parameter("seed").value))

        rgb_in = str(self.get_parameter("rgb_in_topic").value)
        depth_in = str(self.get_parameter("depth_in_topic").value)
        rgb_out = str(self.get_parameter("rgb_out_topic").value)
        depth_out = str(self.get_parameter("depth_out_topic").value)

        self._rgb_pub = self.create_publisher(Image, rgb_out, 1)
        self._depth_pub = self.create_publisher(Image, depth_out, 1)
        self.create_subscription(Image, rgb_in, self._on_rgb, 1)
        self.create_subscription(Image, depth_in, self._on_depth, 1)

        self.get_logger().info(
            f"Noisy RGB: {rgb_in} -> {rgb_out}, sigma={self._rgb_sigma}, "
            f"salt_pepper={self._rgb_salt_pepper}, brightness_jitter={self._rgb_brightness_jitter}, "
            f"contrast_jitter={self._rgb_contrast_jitter}"
        )
        self.get_logger().info(
            f"Noisy depth: {depth_in} -> {depth_out}, sigma={self._depth_sigma}m, "
            f"quant={self._depth_quant}m, dropout={self._depth_dropout}, "
            f"edge_dropout={self._depth_edge_dropout}, shadow={self._depth_shadow_enable}, "
            f"shadow_direction={self._depth_shadow_direction}, shadow_k={self._depth_shadow_k}, "
            f"shadow_max_depth={self._depth_shadow_max_depth}m"
        )

    def _on_rgb(self, msg: Image) -> None:
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().warn(f"Unsupported RGB encoding '{msg.encoding}', expected rgb8 or bgr8.")
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3)).astype(np.float32)
        if self._rgb_contrast_jitter > 0.0:
            contrast = 1.0 + self._rng.uniform(-self._rgb_contrast_jitter, self._rgb_contrast_jitter)
            arr = (arr - 127.5) * contrast + 127.5
        if self._rgb_brightness_jitter > 0.0:
            arr += self._rng.uniform(-self._rgb_brightness_jitter, self._rgb_brightness_jitter)
        if self._rgb_sigma > 0.0:
            arr += self._rng.normal(0.0, self._rgb_sigma, size=arr.shape).astype(np.float32)
        out_arr = np.clip(arr, 0, 255).astype(np.uint8)
        if self._rgb_salt_pepper > 0.0:
            salt = self._rng.random(out_arr.shape[:2]) < (self._rgb_salt_pepper * 0.5)
            pepper = self._rng.random(out_arr.shape[:2]) < (self._rgb_salt_pepper * 0.5)
            out_arr[salt, :] = 255
            out_arr[pepper, :] = 0

        out = Image()
        _copy_header(msg, out)
        out.height = msg.height
        out.width = msg.width
        out.encoding = msg.encoding
        out.is_bigendian = msg.is_bigendian
        out.step = int(msg.width * 3)
        out.data = out_arr.tobytes()
        self._rgb_pub.publish(out)

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding != "32FC1":
            self.get_logger().warn(f"Unsupported depth encoding '{msg.encoding}', expected 32FC1.")
            return
        depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width)).copy()
        valid = np.isfinite(depth) & (depth > 0.0)
        if self._depth_sigma > 0.0:
            depth[valid] += self._rng.normal(0.0, self._depth_sigma, size=int(valid.sum())).astype(np.float32)
        if self._depth_quant > 0.0:
            depth[valid] = np.round(depth[valid] / self._depth_quant) * self._depth_quant
        if self._depth_dropout > 0.0:
            drop = self._rng.random(depth.shape) < self._depth_dropout
            depth[valid & drop] = 0.0
        if self._depth_edge_dropout > 0.0:
            edge_valid = np.isfinite(depth) & (depth > 0.0)
            edge = np.zeros(depth.shape, dtype=bool)
            dz_x = np.abs(depth[:, 1:] - depth[:, :-1])
            dz_y = np.abs(depth[1:, :] - depth[:-1, :])
            valid_x = edge_valid[:, 1:] & edge_valid[:, :-1]
            valid_y = edge_valid[1:, :] & edge_valid[:-1, :]
            edge[:, 1:] |= valid_x & (dz_x > self._depth_edge_threshold)
            edge[:, :-1] |= valid_x & (dz_x > self._depth_edge_threshold)
            edge[1:, :] |= valid_y & (dz_y > self._depth_edge_threshold)
            edge[:-1, :] |= valid_y & (dz_y > self._depth_edge_threshold)
            drop_edge = self._rng.random(depth.shape) < self._depth_edge_dropout
            depth[edge_valid & edge & drop_edge] = 0.0
        if self._depth_shadow_enable:
            depth = _apply_depth_shadow(
                depth,
                direction=self._depth_shadow_direction,
                k_px_m=self._depth_shadow_k,
                min_radius_px=self._depth_shadow_min_radius,
                max_radius_px=self._depth_shadow_max_radius,
                edge_threshold_m=self._depth_shadow_edge_threshold,
                max_depth_m=self._depth_shadow_max_depth,
            )
        depth = np.where(depth < 0.0, 0.0, depth).astype(np.float32)

        out = Image()
        _copy_header(msg, out)
        out.height = msg.height
        out.width = msg.width
        out.encoding = "32FC1"
        out.is_bigendian = msg.is_bigendian
        out.step = int(msg.width * 4)
        out.data = depth.tobytes()
        self._depth_pub.publish(out)


def main():
    rclpy.init()
    node = NoisyCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
