#!/usr/bin/env python3
"""Compute a tool0 target from a desired TCP target and TCP YAML."""

import math
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster


def _quat_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return np.array(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ],
        dtype=np.float64,
    )


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rot_to_quat(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array(
            [
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    idx = int(np.argmax([rot[0, 0], rot[1, 1], rot[2, 2]]))
    if idx == 0:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        q = np.array([0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[2, 1] - rot[1, 2]) / s])
    elif idx == 1:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        q = np.array([(rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s, (rot[0, 2] - rot[2, 0]) / s])
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        q = np.array([(rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s, (rot[1, 0] - rot[0, 1]) / s])
    return q / np.linalg.norm(q)


def _transform_from_xyz_quat(xyz: Dict[str, Any], quat: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = _quat_to_rot(quat)
    tf[:3, 3] = [float(xyz.get("x", 0.0)), float(xyz.get("y", 0.0)), float(xyz.get("z", 0.0))]
    return tf


def _load_yaml(path: str) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a map: {path}")
    return data


def _rotation_from_yaml(data: Dict[str, Any]) -> np.ndarray:
    if "rotation_xyzw" in data:
        q = data["rotation_xyzw"]
        return np.array([float(q.get("x", 0.0)), float(q.get("y", 0.0)), float(q.get("z", 0.0)), float(q.get("w", 1.0))])
    rpy = data.get("rotation_rpy", {})
    return _quat_from_rpy(float(rpy.get("roll", 0.0)), float(rpy.get("pitch", 0.0)), float(rpy.get("yaw", 0.0)))


def _pose_to_msg(frame_id: str, tf: np.ndarray) -> PoseStamped:
    q = _rot_to_quat(tf[:3, :3])
    msg = PoseStamped()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(tf[0, 3])
    msg.pose.position.y = float(tf[1, 3])
    msg.pose.position.z = float(tf[2, 3])
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


def _tf_to_msg(parent: str, child: str, tf: np.ndarray) -> TransformStamped:
    q = _rot_to_quat(tf[:3, :3])
    msg = TransformStamped()
    msg.header.frame_id = parent
    msg.child_frame_id = child
    msg.transform.translation.x = float(tf[0, 3])
    msg.transform.translation.y = float(tf[1, 3])
    msg.transform.translation.z = float(tf[2, 3])
    msg.transform.rotation.x = float(q[0])
    msg.transform.rotation.y = float(q[1])
    msg.transform.rotation.z = float(q[2])
    msg.transform.rotation.w = float(q[3])
    return msg


def _load_target(task_file: str) -> Tuple[str, float, str, np.ndarray]:
    task = _load_yaml(task_file)
    frame = str(task.get("task_frame", "base_link"))
    table_z = float(task.get("table_surface_z_m", 0.79))
    target = task.get("target_tcp", {})
    if not isinstance(target, dict):
        raise ValueError("'target_tcp' must be a map in task YAML")
    xyz = dict(target.get("translation", {}))
    xyz["z"] = float(xyz.get("z", 0.0)) + table_z
    tf = _transform_from_xyz_quat(xyz, _rotation_from_yaml(target))
    child = str(target.get("child_frame", "desired_gripper_tip"))
    return frame, table_z, child, tf


class TcpTaskPoseTest(Node):
    def __init__(self):
        super().__init__("tcp_task_pose_test")
        self.declare_parameter("tcp_file", "")
        self.declare_parameter("task_file", "")
        self.declare_parameter("publish_debug_tf", True)

        tcp_file = str(self.get_parameter("tcp_file").value)
        task_file = str(self.get_parameter("task_file").value)
        if not tcp_file or not task_file:
            raise ValueError("'tcp_file' and 'task_file' parameters are required")

        tcp = _load_yaml(tcp_file)
        tool_frame = str(tcp.get("parent_frame", "tool0"))
        tip_frame = str(tcp.get("child_frame", "gripper_tip"))
        tool_to_tip = _transform_from_xyz_quat(tcp.get("translation", {}), _rotation_from_yaml(tcp))

        task_frame, table_z, target_child, task_to_tip = _load_target(task_file)
        task_to_tool = task_to_tip @ np.linalg.inv(tool_to_tip)

        self._tcp_pub = self.create_publisher(PoseStamped, "target_tcp_pose", 1)
        self._tool_pub = self.create_publisher(PoseStamped, "target_tool0_pose", 1)
        self._tcp_msg = _pose_to_msg(task_frame, task_to_tip)
        self._tool_msg = _pose_to_msg(task_frame, task_to_tool)

        self._tf_msgs = []
        if bool(self.get_parameter("publish_debug_tf").value):
            self._broadcaster = StaticTransformBroadcaster(self)
            self._tf_msgs = [
                _tf_to_msg(task_frame, target_child, task_to_tip),
                _tf_to_msg(task_frame, "desired_tool0", task_to_tool),
                _tf_to_msg(tool_frame, tip_frame, tool_to_tip),
            ]
        self.create_timer(0.5, self._publish)

        self.get_logger().info(f"table-relative z uses table_surface_z_m={table_z:.3f}; task z=0 means world z={table_z:.3f}")
        self.get_logger().info(
            "target tool0 pose: "
            f"x={task_to_tool[0, 3]:.4f}, y={task_to_tool[1, 3]:.4f}, z={task_to_tool[2, 3]:.4f}"
        )

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        self._tcp_msg.header.stamp = stamp
        self._tool_msg.header.stamp = stamp
        self._tcp_pub.publish(self._tcp_msg)
        self._tool_pub.publish(self._tool_msg)
        for msg in self._tf_msgs:
            msg.header.stamp = stamp
        if self._tf_msgs:
            self._broadcaster.sendTransform(self._tf_msgs)


def main():
    rclpy.init()
    node = TcpTaskPoseTest()
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
