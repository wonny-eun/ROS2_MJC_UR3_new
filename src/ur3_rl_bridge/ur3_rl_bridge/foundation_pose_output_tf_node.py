#!/usr/bin/env python3
"""Publish fp_object TF from Isaac FoundationPose /output (Detection3DArray).

Optional filter smooths jitter and can republish in ``reference_frame`` with
world-upright orientation (object +Z parallel to base +Z), fixing camera-frame
Z-up that appears tilted in RViz.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Sequence, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from vision_msgs.msg import Detection3DArray

from ur3_rl_bridge.fp_pose_utils import (
    PoseSample,
    average_poses_in_base,
    is_upside_down_in_base,
    matrix_to_pose,
    normalize_quaternion,
    object_z_tilt_from_vertical_deg,
    pose_to_matrix,
    quat_xyzw_to_rot,
    upright_pose_in_base,
)

_PoseSample = PoseSample


def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    a = normalize_quaternion(q1)
    b = normalize_quaternion(q2)
    d = abs(float(np.dot(a, b)))
    d = min(1.0, max(0.0, d))
    return float(np.degrees(2.0 * np.arccos(d)))


def _parse_axis_param(value) -> Optional[Tuple[float, float, float]]:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) != 3:
            return None
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    if isinstance(value, (list, tuple)) and len(value) == 3:
        return (float(value[0]), float(value[1]), float(value[2]))
    return None


class FoundationPoseOutputTfNode(Node):
    def __init__(self) -> None:
        super().__init__("foundation_pose_output_tf")
        self.declare_parameter("output_topic", "/output")
        self.declare_parameter("child_frame", "fp_object")
        self.declare_parameter("min_score", 0.0)
        self.declare_parameter("reference_frame", "base_link")
        self.declare_parameter("upright_in_base", True)
        self.declare_parameter("long_axis_in_object", [])
        self.declare_parameter("short_axis_in_object", [0.0, 1.0, 0.0])
        self.declare_parameter("tf_filter_enable", True)
        self.declare_parameter("tf_filter_window", 10)
        self.declare_parameter("tf_filter_require_full_window", False)
        self.declare_parameter("tf_filter_use_yaw_median", True)
        self.declare_parameter("tf_filter_lock_on_stable", True)
        self.declare_parameter("tf_filter_lock_stable_frames", 10)
        self.declare_parameter("tf_filter_lock_min_samples", 30)
        self.declare_parameter("tf_filter_lock_pos_tol_m", 0.003)
        self.declare_parameter("tf_filter_lock_rot_tol_deg", 1.0)
        self.declare_parameter("tf_lookup_timeout_sec", 0.15)

        output_topic = str(self.get_parameter("output_topic").value)
        self._child = str(self.get_parameter("child_frame").value).strip()
        self._min_score = float(self.get_parameter("min_score").value)
        self._reference_frame = str(self.get_parameter("reference_frame").value).strip()
        self._upright_in_base = bool(self.get_parameter("upright_in_base").value)
        self._long_axis = _parse_axis_param(self.get_parameter("long_axis_in_object").value)
        self._short_axis = _parse_axis_param(self.get_parameter("short_axis_in_object").value) or (
            0.0,
            1.0,
            0.0,
        )
        self._filter_enable = bool(self.get_parameter("tf_filter_enable").value)
        window = max(1, int(self.get_parameter("tf_filter_window").value))
        self._require_full = bool(self.get_parameter("tf_filter_require_full_window").value)
        self._use_yaw_median = bool(self.get_parameter("tf_filter_use_yaw_median").value)
        self._window = window
        self._pose_buffer: Deque[_PoseSample] = deque(maxlen=window)
        self._lock_on_stable = bool(self.get_parameter("tf_filter_lock_on_stable").value)
        self._lock_stable_frames = max(1, int(self.get_parameter("tf_filter_lock_stable_frames").value))
        self._lock_min_samples = max(1, int(self.get_parameter("tf_filter_lock_min_samples").value))
        self._lock_pos_tol_m = max(0.0, float(self.get_parameter("tf_filter_lock_pos_tol_m").value))
        self._lock_rot_tol_deg = max(0.0, float(self.get_parameter("tf_filter_lock_rot_tol_deg").value))
        self._tf_lookup_timeout = max(0.01, float(self.get_parameter("tf_lookup_timeout_sec").value))
        self._stable_counter = 0
        self._last_filtered: Optional[_PoseSample] = None
        self._locked_pose: Optional[_PoseSample] = None
        self._total_samples = 0
        self._tf_fail_counter = 0
        self._flip_corrected_samples = 0

        self._tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)
        self._tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Detection3DArray, output_topic, self._on_output, 10)

        filt = f"MA(window={window}, yaw_median={self._use_yaw_median})" if self._filter_enable else "off"
        upright_txt = (
            f"upright_in_{self._reference_frame!r}, long_axis={self._long_axis}"
            if self._upright_in_base
            else "raw parent frame"
        )
        self.get_logger().info(
            f"Republishing TF {self._child} from {output_topic} "
            f"(score >= {self._min_score}, filter={filt}, {upright_txt})"
        )
        if self._lock_on_stable:
            self.get_logger().info(
                "TF lock enabled: "
                f"stable_frames={self._lock_stable_frames}, min_samples={self._lock_min_samples}, "
                f"pos_tol={self._lock_pos_tol_m:.4f} m, rot_tol={self._lock_rot_tol_deg:.3f} deg."
            )

    def _lookup_T_reference_parent(self, parent: str, stamp) -> Optional[np.ndarray]:
        if not self._reference_frame or parent == self._reference_frame:
            return np.eye(4, dtype=np.float64)
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._reference_frame,
                parent,
                stamp if (stamp.sec or stamp.nanosec) else Time(),
                timeout=rclpy.duration.Duration(seconds=self._tf_lookup_timeout),
            )
        except Exception:  # noqa: BLE001
            self._tf_fail_counter += 1
            if self._tf_fail_counter in (1, 20, 100):
                self.get_logger().warn(
                    f"TF lookup {self._reference_frame} <- {parent!r} failed "
                    f"(count={self._tf_fail_counter}); skipping sample.",
                    throttle_duration_sec=15.0,
                )
            return None
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        return pose_to_matrix(
            np.array([t.x, t.y, t.z], dtype=np.float64),
            np.array([q.x, q.y, q.z, q.w], dtype=np.float64),
        )

    def _sample_to_reference(self, parent: str, stamp, t_vec: np.ndarray, q_vec: np.ndarray) -> Optional[_PoseSample]:
        T_ref_parent = self._lookup_T_reference_parent(parent, stamp)
        if T_ref_parent is None:
            return None
        T_parent_obj = pose_to_matrix(t_vec, q_vec)
        T_ref_obj = T_ref_parent @ T_parent_obj
        t_ref, q_ref = matrix_to_pose(T_ref_obj)
        if not self._upright_in_base:
            return t_ref, q_ref
        R_ref = quat_xyzw_to_rot(q_ref)
        if is_upside_down_in_base(R_ref):
            self._flip_corrected_samples += 1
        return upright_pose_in_base(
            t_ref,
            R_ref,
            long_axis_object=self._long_axis,
            short_axis_object=self._short_axis,
        )

    def _push_sample(self, sample: _PoseSample) -> Optional[_PoseSample]:
        self._pose_buffer.append(sample)
        self._total_samples += 1
        if self._require_full and len(self._pose_buffer) < self._window:
            return None
        return average_poses_in_base(
            list(self._pose_buffer),
            use_yaw_median=self._use_yaw_median,
            long_axis_object=self._long_axis,
            short_axis_object=self._short_axis,
        )

    def _publish_parent_frame(self) -> str:
        if self._upright_in_base and self._reference_frame:
            return self._reference_frame
        return ""

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
        publish_parent = self._publish_parent_frame() or parent

        if self._locked_pose is not None:
            t_lock, q_lock = self._locked_pose
            self._publish_tf(publish_parent, self._child, stamp, t_lock, q_lock)
            return

        pose = hyp.pose.pose
        t_raw = np.array(
            [float(pose.position.x), float(pose.position.y), float(pose.position.z)],
            dtype=np.float64,
        )
        q_raw = normalize_quaternion(
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

        ref_sample = self._sample_to_reference(parent, stamp, t_raw, q_raw)
        if ref_sample is None:
            return

        if self._filter_enable:
            filtered = self._push_sample(ref_sample)
            if filtered is None:
                return
            t_vec, q_vec = filtered
        else:
            t_vec, q_vec = ref_sample

        if self._lock_on_stable and self._total_samples >= self._lock_min_samples:
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
                tilt = object_z_tilt_from_vertical_deg(quat_xyzw_to_rot(q_vec))
                self.get_logger().info(
                    f"TF locked in {publish_parent!r} after {self._stable_counter + 1} stable frames "
                    f"(total_samples={self._total_samples}, window={self._window}, "
                    f"Z tilt={tilt:.2f}°, fp_inverted_samples={self._flip_corrected_samples}, "
                    f"pos_tol={self._lock_pos_tol_m:.4f} m, rot_tol={self._lock_rot_tol_deg:.3f} deg)."
                )

        self._publish_tf(publish_parent, self._child, stamp, t_vec, q_vec)


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
