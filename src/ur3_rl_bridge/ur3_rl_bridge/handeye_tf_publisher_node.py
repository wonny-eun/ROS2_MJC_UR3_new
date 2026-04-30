#!/usr/bin/env python3
"""Publish calibrated hand-eye transform from a YAML file."""

import math
from pathlib import Path
from typing import Any, Dict

import rclpy
import yaml
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


def _quat_from_rpy(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    return {
        "x": sr * cp * cy - cr * sp * sy,
        "y": cr * sp * cy + sr * cp * sy,
        "z": cr * cp * sy - sr * sp * cy,
        "w": cr * cp * cy + sr * sp * sy,
    }


def _require_map(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"'{key}' must be a map in the hand-eye YAML file")
    return value


def _xyz(data: Dict[str, Any]) -> Dict[str, float]:
    return {
        "x": float(data.get("x", 0.0)),
        "y": float(data.get("y", 0.0)),
        "z": float(data.get("z", 0.0)),
    }


def _rotation(data: Dict[str, Any]) -> Dict[str, float]:
    if "rotation_xyzw" in data:
        q = _require_map(data, "rotation_xyzw")
        return {
            "x": float(q.get("x", 0.0)),
            "y": float(q.get("y", 0.0)),
            "z": float(q.get("z", 0.0)),
            "w": float(q.get("w", 1.0)),
        }
    if "rotation_rpy" in data:
        rpy = _require_map(data, "rotation_rpy")
        return _quat_from_rpy(
            float(rpy.get("roll", 0.0)),
            float(rpy.get("pitch", 0.0)),
            float(rpy.get("yaw", 0.0)),
        )
    raise ValueError("Hand-eye YAML must contain either 'rotation_xyzw' or 'rotation_rpy'")


class HandeyeTfPublisher(Node):
    def __init__(self):
        super().__init__("handeye_tf_publisher")
        self.declare_parameter("calibration_file", "")

        calibration_file = str(self.get_parameter("calibration_file").value)
        if not calibration_file:
            raise ValueError("Parameter 'calibration_file' is required")

        path = Path(calibration_file).expanduser()
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid hand-eye YAML: {path}")

        parent_frame = str(data.get("parent_frame", "tool0"))
        child_frame = str(data.get("child_frame", "camera_color_optical_frame"))
        translation = _xyz(_require_map(data, "translation"))
        rotation = _rotation(data)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = parent_frame
        tf_msg.child_frame_id = child_frame
        tf_msg.transform.translation.x = translation["x"]
        tf_msg.transform.translation.y = translation["y"]
        tf_msg.transform.translation.z = translation["z"]
        tf_msg.transform.rotation.x = rotation["x"]
        tf_msg.transform.rotation.y = rotation["y"]
        tf_msg.transform.rotation.z = rotation["z"]
        tf_msg.transform.rotation.w = rotation["w"]

        self._broadcaster = StaticTransformBroadcaster(self)
        self._broadcaster.sendTransform(tf_msg)
        self.get_logger().info(
            f"Published hand-eye TF {parent_frame} -> {child_frame} from {path}"
        )


def main():
    rclpy.init()
    node = HandeyeTfPublisher()
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
