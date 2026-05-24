#!/usr/bin/env python3
"""Publish fp_object TF from Isaac FoundationPose /output (Detection3DArray).

Isaac normally broadcasts this frame; this node is a fallback when only /output is active.
Optional moving-average filter smooths jitter on translation and orientation.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from vision_msgs.msg import Detection3DArray

_PoseSample = Tuple[np.ndarray, np.ndarray]  # translation (3,), quaternion xyzw (4,)


def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def _average_poses(samples: list[_PoseSample]) -> _PoseSample:
    """Mean translation + hemisphere-corrected quaternion average."""
    if not samples:
        raise ValueError("samples must be non-empty")
    if len(samples) == 1:
        t, q = samples[0]
        return t.copy(), _normalize_quaternion(q)

    translations = np.stack([s[0] for s in samples], axis=0)
    t_mean = np.mean(translations, axis=0)

    ref = _normalize_quaternion(samples[0][1])
    acc = np.zeros(4, dtype=np.float64)
    for _t, q in samples:
        qn = _normalize_quaternion(q)
        if float(np.dot(ref, qn)) < 0.0:
            qn = -qn
        acc += qn
    q_mean = _normalize_quaternion(acc)
    return t_mean, q_mean


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    a = _normalize_quaternion(q1)
    b = _normalize_quaternion(q2)
    d = abs(float(np.dot(a, b)))
    d = min(1.0, max(0.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


class FoundationPoseOutputTfNode(Node):
    def __init__(self) -> None:
        super().__init__("foundation_pose_output_tf")
        self.declare_parameter("output_topic", "/output")
        self.declare_parameter("child_frame", "fp_object")
        self.declare_parameter("min_score", 0.0)
        self.declare_parameter("tf_filter_enable", True)
        self.declare_parameter("tf_filter_window", 10)
        # If true, publish only after the buffer has this many samples (then keep sliding).
        self.declare_parameter("tf_filter_require_full_window", False)
        # Freeze output once averaged TF stays stable for N consecutive frames.
        self.declare_parameter("tf_filter_lock_on_stable", True)
        self.declare_parameter("tf_filter_lock_stable_frames", 10)
        self.declare_parameter("tf_filter_lock_pos_tol_m", 0.003)
        self.declare_parameter("tf_filter_lock_rot_tol_deg", 1.0)

        output_topic = str(self.get_parameter("output_topic").value)
        self._child = str(self.get_parameter("child_frame").value).strip()
        self._min_score = float(self.get_parameter("min_score").value)
        self._filter_enable = bool(self.get_parameter("tf_filter_enable").value)
        window = max(1, int(self.get_parameter("tf_filter_window").value))
        self._require_full = bool(self.get_parameter("tf_filter_require_full_window").value)
        self._window = window
        self._pose_buffer: Deque[_PoseSample] = deque(maxlen=window)
        self._lock_on_stable = bool(self.get_parameter("tf_filter_lock_on_stable").value)
        self._lock_stable_frames = max(1, int(self.get_parameter("tf_filter_lock_stable_frames").value))
        self._lock_pos_tol_m = max(0.0, float(self.get_parameter("tf_filter_lock_pos_tol_m").value))
        self._lock_rot_tol_deg = max(0.0, float(self.get_parameter("tf_filter_lock_rot_tol_deg").value))
        self._stable_counter = 0
        self._last_filtered: Optional[_PoseSample] = None
        self._locked_pose: Optional[_PoseSample] = None

        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Detection3DArray, output_topic, self._on_output, 10)
        filt = f"MA(window={window})" if self._filter_enable else "off"
        self.get_logger().info(
            f"Republishing TF {self._child} from {output_topic} "
            f"(score >= {self._min_score}, filter={filt})"
        )
        if self._lock_on_stable:
            self.get_logger().info(
                "TF lock enabled: "
                f"stable_frames={self._lock_stable_frames}, "
                f"pos_tol={self._lock_pos_tol_m:.4f} m, rot_tol={self._lock_rot_tol_deg:.3f} deg."
            )

    def _push_sample(self, pose) -> Optional[_PoseSample]:
        t = np.array(
            [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
            dtype=np.float64,
        )
        q = _normalize_quaternion(
            np.array(
                [
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ],
                dtype=np.float64,
            )
        )
        self._pose_buffer.append((t, q))
        if self._require_full and len(self._pose_buffer) < self._window:
            return None
        return _average_poses(list(self._pose_buffer))

    def _publish_tf(self, parent: str, child: str, stamp, t_vec: np.ndarray, q_vec: np.ndarray) -> None:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = parent
        t.child_frame_id = child
        t.transform.translation.x = float(t_vec[0])
        t.transform.translation.y = float(t_vec[1])
        t.transform.translation.z = float(t_vec[2])
        t.transform.rotation.x = float(q_vec[0])
        t.transform.rotation.y = float(q_vec[1])
        t.transform.rotation.z = float(q_vec[2])
        t.transform.rotation.w = float(q_vec[3])
        self._tf_broadcaster.sendTransform(t)

    def _on_output(self, msg: Detection3DArray) -> None:
        if not msg.detections:
            return
        det = msg.detections[0]
        if not det.results:
            return
        hyp = det.results[0]
        if hyp.hypothesis.score < self._min_score:
            return

        parent = (det.header.frame_id or msg.header.frame_id or "").strip()
        if not parent:
            return

        stamp = det.header.stamp if det.header.stamp.sec or det.header.stamp.nanosec else msg.header.stamp
        if self._locked_pose is not None:
            t_lock, q_lock = self._locked_pose
            self._publish_tf(parent, self._child, stamp, t_lock, q_lock)
            return

        pose = hyp.pose.pose
        if self._filter_enable:
            filtered = self._push_sample(pose)
            if filtered is None:
                return
            t_vec, q_vec = filtered
        else:
            t_vec = np.array(
                [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
                dtype=np.float64,
            )
            q_vec = _normalize_quaternion(
                np.array(
                    [
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                    ],
                    dtype=np.float64,
                )
            )
        if self._lock_on_stable:
            if self._last_filtered is not None:
                t_prev, q_prev = self._last_filtered
                dp = float(np.linalg.norm(t_vec - t_prev))
                dr = _quat_angle_deg(q_vec, q_prev)
                if dp <= self._lock_pos_tol_m and dr <= self._lock_rot_tol_deg:
                    self._stable_counter += 1
                else:
                    self._stable_counter = 0
            self._last_filtered = (t_vec.copy(), q_vec.copy())
            if self._stable_counter >= self._lock_stable_frames:
                self._locked_pose = (t_vec.copy(), q_vec.copy())
                self.get_logger().info(
                    f"TF locked at stable pose after {self._stable_counter + 1} frames "
                    f"(window={self._window}, pos_tol={self._lock_pos_tol_m:.4f} m, "
                    f"rot_tol={self._lock_rot_tol_deg:.3f} deg)."
                )

        self._publish_tf(parent, self._child, stamp, t_vec, q_vec)


def main() -> None:
    rclpy.init()
    node = FoundationPoseOutputTfNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
