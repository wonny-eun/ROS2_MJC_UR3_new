#!/usr/bin/env python3
"""Config-driven action sequencer for UR3 MoveIt motions."""

from __future__ import annotations

import math
import os
import select
import signal
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import cv2
import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup, MoveGroupSequence
from moveit_msgs.msg import (
    AttachedCollisionObject,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MotionSequenceItem,
    MoveItErrorCodes,
    PlanningOptions,
    PlanningScene,
    RobotState,
    RobotTrajectory,
    WorkspaceParameters,
)
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]


DEFAULT_CONFIG_FILE = (
    "/home/wonny/ur3_control/src/ROS2_MuJoCo_UR3/src/ur3_pick_task/"
    "config/actions/ur3_action_sequence.yaml"
)
DEFAULT_TCP_FILE = (
    "/home/wonny/ur3_control/src/ROS2_MuJoCo_UR3/src/ur3_rl_bridge/"
    "config/tcp/gripper_tip.yaml"
)

DEFAULT_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


class SequenceInterrupted(Exception):
    """Raised when the operator stops the sequence (Ctrl+C or ~/stop_sequence service)."""


def _merge_vision_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Defaults + optional `vision:` block in ur3_action_sequence.yaml."""
    defaults: Dict[str, Any] = {
        "rgb_topic": "/rl_camera/color",
        "depth_topic": "/rl_camera/depth",
        "camera_info_topic": "/rl_camera/camera_info",
        # Depth + intrinsics match MuJoCo rl_camera; adjust if your hand-eye uses another optical frame.
        "camera_optical_frame": "rl_camera_frame",
        "min_confidence": 0.8,
        "target_distance_m": 0.4,
        "max_iterations": 5,
        # Softer default for convergence; tighten in YAML once look-at IK is stable.
        "center_tolerance_px": 55.0,
        "distance_tolerance_m": 0.025,
        "step_gain": 0.45,
        "depth_roi_half_px": 5,
        "yolo_iou": 0.5,
        "yolo_model_path": "",
        "wait_image_timeout_sec": 8.0,
        "default_target_class": "",
        # yolo_visual_center: cartesian optical centering (default), then optical approach.
        "approach_max_iterations": 8,
        "approach_step_max_m": 0.12,
        # Centering: "servo"/"smooth" = velocity IBVS (smooth); "cartesian" = streamed Cartesian paths.
        "yolo_visual_center_centering_mode": "servo",
        "center_max_iterations": 40,
        "yolo_center_step_gain": 0.35,
        "yolo_center_max_step_m": 0.008,
        "yolo_center_max_duration_sec": 60.0,
        "yolo_center_path_segments": 20,
        "yolo_center_cartesian_max_step_m": 0.002,
        "yolo_center_velocity_scaling": 0.05,
        "yolo_center_acceleration_scaling": 0.05,
        "yolo_center_redetect_after_misses": 30,
        "yolo_center_settle_frames": 25,
        "yolo_center_settle_period_sec": 0.06,
        "yolo_center_use_raw_detection": True,
        "yolo_center_nudge_sign": 1.0,
        "yolo_center_min_move_m": 0.0003,
        "yolo_visual_center_acquire_min_confidence": 0.25,
        "yolo_visual_center_require_centered_on_complete": True,
        "yolo_visual_center_avoid_collisions": True,
        # MoveIt Servo (optional): publish Twist to delta_twist_cmds while bbox centers in image.
        "yolo_servo_twist_topic": "/servo_node/delta_twist_cmds",
        "yolo_servo_start_service": "/servo_node/start_servo",
        "yolo_servo_unpause_service": "/servo_node/unpause_servo",
        "yolo_servo_pause_service": "/servo_node/pause_servo",
        "yolo_servo_command_frame": "base_link",
        "yolo_servo_use_optical_ibvs": True,
        "yolo_servo_ibvs_sign_u": 1.0,
        "yolo_servo_ibvs_sign_v": -1.0,
        "yolo_servo_max_axis_offset_px": 95.0,
        "yolo_servo_velocity_in_base": True,
        "yolo_servo_cartesian_fallback": True,
        "yolo_servo_rate_hz": 25.0,
        "yolo_servo_max_duration_sec": 90.0,
        "yolo_servo_use_raw_only": False,
        "yolo_servo_use_smoothed_for_twist": True,
        "yolo_servo_offset_ema_alpha": 0.28,
        "yolo_servo_velocity_ema_alpha": 0.28,
        "yolo_servo_max_twist_delta_m_s": 0.004,
        "yolo_servo_cartesian_speed_scale": 0.45,
        "yolo_servo_diverge_correct_after": 0,
        "yolo_servo_cartesian_nudge_every_sec": 0.0,
        "yolo_servo_diverge_min_metric_px": 120.0,
        "yolo_servo_match_cartesian_gain": False,
        "yolo_servo_gain_scale": 0.22,
        "yolo_servo_linear_gain": 0.35,
        "yolo_servo_ibvs_sign": -1.0,
        "yolo_servo_cartesian_nudge_min_metric_px": 80.0,
        "yolo_servo_diverge_abort_streak": 0,
        "yolo_servo_cmd_on_keepalive": True,
        "yolo_servo_slow_error_px": 100.0,
        "yolo_servo_near_center_ramp_px": 100.0,
        "yolo_servo_near_center_min_scale": 0.35,
        "yolo_servo_deadband_px": 8.0,
        "yolo_servo_fine_pass": True,
        "yolo_servo_fine_pass_max_duration_sec": 20.0,
        "yolo_servo_cross_brake_frames": 0,
        "yolo_servo_warmup_raw_frames": 2,
        "yolo_servo_redetect_after_misses": 15,
        "yolo_servo_angular_gain": 0.0,
        "yolo_servo_use_linear": True,
        "yolo_servo_use_angular": False,
        "yolo_servo_max_linear_m_s": 0.008,
        "yolo_servo_max_angular_rad_s": 0.15,
        "yolo_servo_stable_frames": 3,
        # After centering, re-lock tool0 to pick lookat_vector + tcp_roll_rad before approach.
        "yolo_visual_center_restore_pick_orientation_before_approach": False,
        # Plan approach as one Cartesian path (N waypoints, one trajectory) instead of stop-and-go per step.
        "yolo_visual_center_smooth_approach": True,
        # True: at most one look-at joint move, then one approach Cartesian move (no iterative servo stripes).
        "yolo_visual_center_single_motion": False,
        # Single-motion caps how far optical Z translates in one approach hop (metres).
        "yolo_visual_center_single_motion_max_translation_m": 2.0,
        # Ultralytics predict() conf= (lower keeps more boxes; we still require score >= min_confidence).
        "yolo_predict_conf_floor": 0.01,
        # Match YOLO class name to target_class case-insensitively (training export often differs).
        "yolo_class_match_case_insensitive": True,
        # Bootstrap: retry until a valid YOLO+depth sample primes the tracker (keep-alive needs had_good).
        "yolo_visual_center_acquire_max_attempts": 60,
        "yolo_visual_center_acquire_spin_sec": 0.05,
        # After N consecutive raw misses: return to last joint pose where YOLO worked, nudge camera to re-find.
        "yolo_visual_center_redetect_after_misses": 10,
        "yolo_visual_center_redetect_max_sessions": 3,
        "yolo_visual_center_redetect_velocity_scaling": 0.15,
        "yolo_visual_center_redetect_nudge_step_m": 0.012,
        "yolo_visual_center_redetect_nudge_max_trials": 16,
        "yolo_redetect_skip_snap_back_after_servo": True,
        # Do not servo closer than this camera-to-object depth (metres); keeps FOV / YOLO reliable.
        "yolo_visual_center_min_detection_depth_m": 0.35,
        # yolo_visual_center: stabilize Ray / centroid when YOLO flickers off.
        "yolo_tracking_median_window": 5,
        "yolo_tracking_ema_alpha": 0.35,
        "yolo_tracking_keep_alive_max_misses": 15,
        "yolo_tracking_log_keep_alive_every": 4,
    }
    vis = config.get("vision")
    if isinstance(vis, dict):
        defaults.update(vis)
    return defaults


class _YoloDetectionTrack:
    """Median-window + EMA on successful YOLO+depth rays; brief dropout reuses last filtered sample (keep-alive)."""

    __slots__ = (
        "median_window",
        "ema_alpha",
        "max_keep_alive_misses",
        "log_every",
        "z_hist",
        "du_hist",
        "dv_hist",
        "xc_hist",
        "yc_hist",
        "u_hist",
        "v_hist",
        "ema_z",
        "ema_du",
        "ema_dv",
        "ema_xc",
        "ema_yc",
        "ema_u",
        "ema_v",
        "last_conf",
        "last_out",
        "miss_count",
        "had_good",
        "_alive_log_ctr",
    )

    def __init__(
        self,
        median_window: int,
        ema_alpha: float,
        max_keep_alive_misses: int,
        log_every: int,
    ) -> None:
        self.median_window = max(1, int(median_window))
        self.ema_alpha = float(max(0.0, min(1.0, ema_alpha)))
        self.max_keep_alive_misses = int(max(0, max_keep_alive_misses))
        self.log_every = max(1, int(log_every))
        mw = self.median_window
        self.z_hist: deque[float] = deque(maxlen=mw)
        self.du_hist: deque[float] = deque(maxlen=mw)
        self.dv_hist: deque[float] = deque(maxlen=mw)
        self.xc_hist: deque[float] = deque(maxlen=mw)
        self.yc_hist: deque[float] = deque(maxlen=mw)
        self.u_hist: deque[float] = deque(maxlen=mw)
        self.v_hist: deque[float] = deque(maxlen=mw)
        self.ema_z: Optional[float] = None
        self.ema_du: Optional[float] = None
        self.ema_dv: Optional[float] = None
        self.ema_xc: Optional[float] = None
        self.ema_yc: Optional[float] = None
        self.ema_u: Optional[float] = None
        self.ema_v: Optional[float] = None
        self.last_conf = -1.0
        self.last_out: Optional[tuple[int, int, float, float, float, float, float, float]] = None
        self.miss_count = 0
        self.had_good = False
        self._alive_log_ctr = 0

    def reset(self) -> None:
        """Clear filter state after relocating to a new viewing pose."""
        self.z_hist.clear()
        self.du_hist.clear()
        self.dv_hist.clear()
        self.xc_hist.clear()
        self.yc_hist.clear()
        self.u_hist.clear()
        self.v_hist.clear()
        self.ema_z = None
        self.ema_du = None
        self.ema_dv = None
        self.ema_xc = None
        self.ema_yc = None
        self.ema_u = None
        self.ema_v = None
        self.last_conf = -1.0
        self.last_out = None
        self.miss_count = 0
        self.had_good = False
        self._alive_log_ctr = 0

    @staticmethod
    def _median_deque(samples: deque[float]) -> Optional[float]:
        if not samples:
            return None
        return float(statistics.median(list(samples)))

    def _apply_ema(self, prev: Optional[float], new_val: float) -> float:
        if prev is None:
            return new_val
        a = self.ema_alpha
        return a * new_val + (1.0 - a) * prev

    def push(
        self,
        det: Optional[tuple[int, int, float, float, float, float, float, float]],
        *,
        log_fn: Optional[Callable[[str], None]] = None,
        phase_tag: str = "",
    ) -> tuple[bool, Optional[tuple[int, int, float, float, float, float, float, float]], bool]:
        """
        Return (success, filtered_detection_tuple, keep_alive_used).
        Filtered tuple matches _yolo_detection_center_ray outputs.
        """
        if det is not None:
            u, v, x_cam, y_cam, z_ray, du, dv, conf = det
            self.miss_count = 0
            self._alive_log_ctr = 0
            self.had_good = True
            self.last_conf = float(conf)
            self.z_hist.append(float(z_ray))
            self.du_hist.append(float(du))
            self.dv_hist.append(float(dv))
            self.xc_hist.append(float(x_cam))
            self.yc_hist.append(float(y_cam))
            self.u_hist.append(float(u))
            self.v_hist.append(float(v))

            z_med = self._median_deque(self.z_hist)
            du_med = self._median_deque(self.du_hist)
            dv_med = self._median_deque(self.dv_hist)
            xc_med = self._median_deque(self.xc_hist)
            yc_med = self._median_deque(self.yc_hist)
            u_med = self._median_deque(self.u_hist)
            v_med = self._median_deque(self.v_hist)

            z_f = self._apply_ema(self.ema_z, z_med if z_med is not None else float(z_ray))
            du_f = self._apply_ema(self.ema_du, du_med if du_med is not None else float(du))
            dv_f = self._apply_ema(self.ema_dv, dv_med if dv_med is not None else float(dv))
            xc_f = self._apply_ema(self.ema_xc, xc_med if xc_med is not None else float(x_cam))
            yc_f = self._apply_ema(self.ema_yc, yc_med if yc_med is not None else float(y_cam))
            u_f = self._apply_ema(self.ema_u, u_med if u_med is not None else float(u))
            v_f = self._apply_ema(self.ema_v, v_med if v_med is not None else float(v))

            self.ema_z = z_f
            self.ema_du = du_f
            self.ema_dv = dv_f
            self.ema_xc = xc_f
            self.ema_yc = yc_f
            self.ema_u = u_f
            self.ema_v = v_f

            out = (
                int(round(u_f)),
                int(round(v_f)),
                xc_f,
                yc_f,
                z_f,
                du_f,
                dv_f,
                self.last_conf,
            )
            self.last_out = out
            return True, out, False

        self.miss_count += 1
        if not self.had_good:
            return False, None, False
        if self.max_keep_alive_misses > 0 and self.miss_count > self.max_keep_alive_misses:
            return False, None, False
        if self.last_out is None:
            return False, None, False

        self._alive_log_ctr += 1
        if log_fn and self._alive_log_ctr % self.log_every == 0:
            log_fn(f"{phase_tag} keep-alive ({self._alive_log_ctr} misses): replaying filtered ray (conf_was={self.last_conf:.3f})")
        return True, self.last_out, True


def _load_yaml(path: str) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a map: {path}")
    return data


def _duration_from_seconds(seconds: float) -> DurationMsg:
    seconds = max(float(seconds), 0.0)
    sec = int(seconds)
    return DurationMsg(sec=sec, nanosec=int((seconds - sec) * 1e9))


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
    q = q / np.linalg.norm(q)
    x, y, z, w = q.tolist()
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _moveit_error_label(code: int) -> str:
    """Return MoveIt constant name when `code` matches a MoveItErrorCodes integer."""
    for name in dir(MoveItErrorCodes):
        if name.startswith("_"):
            continue
        val = getattr(MoveItErrorCodes, name)
        if isinstance(val, int) and val == code:
            return name
    return ""


def _rot_to_quat(rot: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        q = np.array(
            [
                (rot[2, 1] - rot[1, 2]) / s,
                (rot[0, 2] - rot[2, 0]) / s,
                (rot[1, 0] - rot[0, 1]) / s,
                0.25 * s,
            ],
            dtype=np.float64,
        )
    else:
        idx = int(np.argmax([rot[0, 0], rot[1, 1], rot[2, 2]]))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            q = np.array(
                [
                    0.25 * s,
                    (rot[0, 1] + rot[1, 0]) / s,
                    (rot[0, 2] + rot[2, 0]) / s,
                    (rot[2, 1] - rot[1, 2]) / s,
                ],
                dtype=np.float64,
            )
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            q = np.array(
                [
                    (rot[0, 1] + rot[1, 0]) / s,
                    0.25 * s,
                    (rot[1, 2] + rot[2, 1]) / s,
                    (rot[0, 2] - rot[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            q = np.array(
                [
                    (rot[0, 2] + rot[2, 0]) / s,
                    (rot[1, 2] + rot[2, 1]) / s,
                    0.25 * s,
                    (rot[1, 0] - rot[0, 1]) / s,
                ],
                dtype=np.float64,
            )
    return q / np.linalg.norm(q)


def _transform_from_yaml(data: Dict[str, Any]) -> np.ndarray:
    xyz = data.get("translation", {})
    if "rotation_xyzw" in data:
        q = data["rotation_xyzw"]
        quat = np.array(
            [
                float(q.get("x", 0.0)),
                float(q.get("y", 0.0)),
                float(q.get("z", 0.0)),
                float(q.get("w", 1.0)),
            ],
            dtype=np.float64,
        )
    else:
        rpy = data.get("rotation_rpy", {})
        quat = _quat_from_rpy(
            float(rpy.get("roll", 0.0)),
            float(rpy.get("pitch", 0.0)),
            float(rpy.get("yaw", 0.0)),
        )

    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = _quat_to_rot(quat)
    tf[:3, 3] = [
        float(xyz.get("x", 0.0)),
        float(xyz.get("y", 0.0)),
        float(xyz.get("z", 0.0)),
    ]
    return tf


def _vector3(value: Any, label: str) -> np.ndarray:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 numbers")
    return np.array([float(v) for v in value], dtype=np.float64)


def _normalize(v: np.ndarray, label: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f"{label} vector is zero")
    return v / n


def _orthonormalize_rotation(rot: np.ndarray) -> np.ndarray:
    """Project a near-rotation 3x3 matrix onto SO(3) (removes cross-product / float drift)."""
    u, _, vh = np.linalg.svd(rot, full_matrices=True)
    r = u @ vh
    if np.linalg.det(r) < 0.0:
        u[:, -1] *= -1.0
        r = u @ vh
    return r


def _normalize_lookat_vector(raw: np.ndarray) -> np.ndarray:
    """
    Unit vector for TCP +Z. Snaps nearly-axis-aligned inputs (e.g. [0,0,-1] with YAML float noise)
    to exact ±X/±Y/±Z so look-at frames are not tilted by tiny parsing errors.
    """
    v = np.asarray(raw, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-15:
        raise ValueError("lookat_vector is zero")
    u = v / n
    a = np.abs(u)
    order = np.argsort(a)
    sm, mid, lg = int(order[0]), int(order[1]), int(order[2])
    # Dominant component ~±1 and other two negligible → exact cardinal direction
    if a[lg] > 1.0 - 1e-9 and a[sm] + a[mid] < 1e-5:
        out = np.zeros(3, dtype=np.float64)
        out[lg] = 1.0 if u[lg] >= 0.0 else -1.0
        return out
    return u


def _scaling(value: Any, label: str, default: float) -> float:
    out = default if value is None else float(value)
    if not 0.0 < out <= 1.0:
        raise ValueError(f"{label} must be in (0.0, 1.0], got {out}")
    return out


def _tcp_rotation_from_lookat(lookat: np.ndarray, roll_rad: float = 0.0) -> np.ndarray:
    z_axis = _normalize(lookat, "lookat")
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(z_axis, up_hint))) > 0.98:
        up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    x_axis = _normalize(np.cross(up_hint, z_axis), "computed x")
    y_axis = _normalize(np.cross(z_axis, x_axis), "computed y")
    base_rot = np.column_stack((x_axis, y_axis, z_axis))

    cr = math.cos(roll_rad)
    sr = math.sin(roll_rad)
    local_roll = np.array(
        [[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return _orthonormalize_rotation(base_rot @ local_roll)


def _pose_from_transform(tf: np.ndarray) -> Pose:
    rot = _orthonormalize_rotation(tf[:3, :3])
    q = _rot_to_quat(rot)
    msg = Pose()
    msg.position.x = float(tf[0, 3])
    msg.position.y = float(tf[1, 3])
    msg.position.z = float(tf[2, 3])
    msg.orientation.x = float(q[0])
    msg.orientation.y = float(q[1])
    msg.orientation.z = float(q[2])
    msg.orientation.w = float(q[3])
    return msg


class MotionManager(Node):
    """Load a YAML sequence and execute robot actions one by one."""

    def __init__(self) -> None:
        super().__init__("action_sequencer")
        self.declare_parameter("config_file", DEFAULT_CONFIG_FILE)
        self.declare_parameter("sequence_name", "scan_and_approach")
        self.declare_parameter("object_name", "")
        self.declare_parameter("prompt_for_object", True)
        self.declare_parameter("continue_on_failure", False)
        self.declare_parameter("stop_sequence_service", "~/stop_sequence")
        self.declare_parameter("publish_action_completed_topic", "")

        self.config_file = str(self.get_parameter("config_file").value)
        self.sequence_name = str(self.get_parameter("sequence_name").value)
        self.object_name = str(self.get_parameter("object_name").value).strip()
        self.prompt_for_object = bool(self.get_parameter("prompt_for_object").value)
        self.continue_on_failure = bool(self.get_parameter("continue_on_failure").value)

        self.config = _load_yaml(self.config_file)
        self.sequence_names = self._select_sequence_names(self.config, self.sequence_name)
        self.sequence_name = " -> ".join(self.sequence_names)
        self.motion_cfg = self._load_motion_config(self.config)
        self._forbidden_zone_presets = MotionManager._load_named_preset_map(self.config, "forbidden_zone_presets")
        self._planning_attach_presets = MotionManager._load_named_preset_map(self.config, "planning_attach_presets")
        self._planning_detach_presets = MotionManager._load_named_preset_map(self.config, "planning_detach_presets")
        self._mujoco_weld_presets = MotionManager._load_named_preset_map(self.config, "mujoco_weld_presets")
        self.sequence = self._load_sequences(self.config, self.sequence_names)
        self.joint_names = [str(name) for name in self.motion_cfg.get("joint_names", DEFAULT_JOINT_NAMES)]
        self.tool_to_tcp = _transform_from_yaml(_load_yaml(str(self.motion_cfg["tcp_file"])))
        self.vision_cfg = _merge_vision_config(self.config)
        self._latest_joint_state: Optional[JointState] = None
        self._vis_lock = threading.Lock()
        self._vis_bgr: Optional[np.ndarray] = None
        self._vis_depth: Optional[np.ndarray] = None
        self._vis_info: Optional[CameraInfo] = None
        self._tf_buffer = Buffer(cache_time=RclDuration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)
        self._yolo_model: Any = None
        self._yolo_loaded_path = ""
        self._vision_subscriptions: list[Any] = []
        self._servo_twist_pub: Any = None
        self._servo_twist_topic = ""
        self._servo_trigger_clients: dict[str, Any] = {}

        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)
        self._ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self._cartesian_client = self.create_client(
            GetCartesianPath,
            str(self.motion_cfg["cartesian_path_service"]),
        )
        self._move_group_client = ActionClient(
            self,
            MoveGroup,
            str(self.motion_cfg["move_group_action"]),
        )
        self._sequence_client = ActionClient(
            self,
            MoveGroupSequence,
            str(self.motion_cfg["sequence_action"]),
        )
        self._execute_trajectory_client = ActionClient(
            self,
            ExecuteTrajectory,
            str(self.motion_cfg["execute_trajectory_action"]),
        )
        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.motion_cfg["trajectory_action"]),
        )
        self._weld_pose_publishers: dict[str, Any] = {}
        self._trigger_clients: dict[str, Any] = {}
        self._planning_scene_pub = self.create_publisher(
            PlanningScene,
            str(self.motion_cfg.get("planning_scene_topic", "/planning_scene")),
            10,
        )
        self._abort_requested: bool = False
        act_topic = str(self.get_parameter("publish_action_completed_topic").value).strip()
        self._action_completed_pub = (
            self.create_publisher(String, act_topic, 10) if act_topic else None
        )

        stop_svc = str(self.get_parameter("stop_sequence_service").value)
        self.create_service(Trigger, stop_svc, self._stop_sequence_cb)
        self.get_logger().info(
            "Stop: Space (TTY), Ctrl+C in this terminal, or "
            f"`ros2 service call {stop_svc.strip()} std_srvs/srv/Trigger {{}}` "
            "(resolve ~/ via `ros2 service list | grep stop`)."
        )

    def _stop_sequence_cb(self, _request: Any, response: Trigger.Response) -> Trigger.Response:
        self._abort_requested = True
        self.get_logger().warn("Stop sequence requested — aborting after current interrupt point.")
        response.success = True
        response.message = "Sequence stop requested; active motions will be cancelled where supported."
        return response

    def _publish_action_completed(self, action_name: str) -> None:
        if self._action_completed_pub is None:
            return
        msg = String()
        msg.data = str(action_name)
        self._action_completed_pub.publish(msg)

    def _raise_if_abort(self, goal_handle: Any = None) -> None:
        if not self._abort_requested:
            return
        if goal_handle is not None:
            try:
                cancel_future = goal_handle.cancel_goal_async()
                deadline = time.monotonic() + 2.0
                while not cancel_future.done() and time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=0.02)
            except Exception as exc:  # noqa: BLE001
                self.get_logger().warn(f"Could not cancel motion goal after stop: {exc}")
        raise SequenceInterrupted()

    def _spin_until_future_complete_abortable(
        self,
        future: Any,
        timeout_sec: float,
        goal_handle: Any = None,
    ) -> None:
        deadline = time.monotonic() + float(timeout_sec)
        while rclpy.ok() and not future.done():
            self._raise_if_abort(goal_handle)
            if time.monotonic() >= deadline:
                return
            rclpy.spin_once(self, timeout_sec=0.01)

    def _wait_for_action_server_abortable(self, client: ActionClient, timeout_sec: float) -> bool:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            self._raise_if_abort()
            if client.server_is_ready():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return client.server_is_ready()

    def _wait_for_service_abortable(self, client: Any, timeout_sec: float) -> bool:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline:
            self._raise_if_abort()
            if client.service_is_ready():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return client.service_is_ready()

    def _sleep_interruptible(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        end = time.monotonic() + float(seconds)
        while time.monotonic() < end:
            self._raise_if_abort()
            rclpy.spin_once(self, timeout_sec=min(0.05, end - time.monotonic()))

    def _on_joint_states(self, msg: JointState) -> None:
        if msg.name and msg.position:
            self._latest_joint_state = msg

    def _on_vision_rgb(self, msg: Image) -> None:
        try:
            if msg.encoding == "rgb8":
                rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            elif msg.encoding == "bgr8":
                bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3)).copy()
            else:
                return
            with self._vis_lock:
                self._vis_bgr = bgr
        except Exception:
            return

    def _on_vision_depth(self, msg: Image) -> None:
        try:
            if msg.encoding != "32FC1":
                return
            d = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            with self._vis_lock:
                self._vis_depth = d.copy()
        except Exception:
            return

    def _on_vision_info(self, msg: CameraInfo) -> None:
        with self._vis_lock:
            self._vis_info = msg

    def _clear_vision_buffers(self) -> None:
        with self._vis_lock:
            self._vis_bgr = None
            self._vis_depth = None
            self._vis_info = None

    def _ensure_vision_subscriptions_for_yolo(self) -> None:
        """
        Subscribe to camera topics only when a yolo_visual_center action starts.
        Avoids buffering stale images during earlier sequence motions after ros2 run.
        """
        if self._vision_subscriptions:
            return
        rgb_t = str(self.vision_cfg["rgb_topic"])
        depth_t = str(self.vision_cfg["depth_topic"])
        info_t = str(self.vision_cfg["camera_info_topic"])
        self._vision_subscriptions = [
            self.create_subscription(Image, rgb_t, self._on_vision_rgb, 1),
            self.create_subscription(Image, depth_t, self._on_vision_depth, 1),
            self.create_subscription(CameraInfo, info_t, self._on_vision_info, 1),
        ]
        self.get_logger().info(
            f"yolo_visual_center: vision subscriptions armed ({rgb_t}, {depth_t}, {info_t}) — detection uses frames from now on."
        )

    def _wait_vision_frames(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + float(timeout_sec)
        while time.monotonic() < deadline and rclpy.ok():
            self._raise_if_abort()
            with self._vis_lock:
                ok = self._vis_bgr is not None and self._vis_depth is not None and self._vis_info is not None
            if ok:
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        return False

    def _yolo_snapshot_and_detect(
        self,
        model: Any,
        *,
        target_class: str,
        min_conf: float,
        yolo_iou: float,
        roi: int,
        ray_kw: Dict[str, Any],
    ) -> Optional[tuple[int, int, float, float, float, float, float, float]]:
        if not self._wait_vision_frames(2.0):
            return None
        with self._vis_lock:
            bgr = self._vis_bgr.copy() if self._vis_bgr is not None else None
            depth = self._vis_depth.copy() if self._vis_depth is not None else None
            info = self._vis_info
        if bgr is None or depth is None or info is None:
            return None
        return self._yolo_detection_center_ray(
            model,
            bgr,
            depth,
            info,
            target_class=target_class,
            min_conf=min_conf,
            yolo_iou=yolo_iou,
            roi=roi,
            **ray_kw,
        )

    def _yolo_snapshot_track(
        self,
        det_track: _YoloDetectionTrack,
        model: Any,
        *,
        target_class: str,
        min_conf: float,
        yolo_iou: float,
        roi: int,
        ray_kw: Dict[str, Any],
        phase_tag: str,
    ) -> tuple[
        bool,
        Optional[tuple[int, int, float, float, float, float, float, float]],
        bool,
        Optional[tuple[int, int, float, float, float, float, float, float]],
        bool,
    ]:
        """
        One YOLO pass on the latest frame (no post-motion pause). Updates the temporal tracker.
        Returns (ok, filtered_det, keep_alive_used, raw_det, raw_frame_miss).
        raw_frame_miss is True when YOLO/depth returned nothing on this frame (not keep-alive).
        """
        det_raw = self._yolo_snapshot_and_detect(
            model,
            target_class=target_class,
            min_conf=min_conf,
            yolo_iou=yolo_iou,
            roi=roi,
            ray_kw=ray_kw,
        )
        raw_miss = det_raw is None
        ok, det, keep_alive = det_track.push(
            det_raw,
            log_fn=self.get_logger().info,
            phase_tag=phase_tag,
        )
        return ok, det, keep_alive, det_raw, raw_miss

    def _current_manipulator_joint_positions(self) -> Optional[list[float]]:
        if not self._has_complete_joint_state():
            return None
        js = self._latest_joint_state
        if js is None:
            return None
        pos_by_name = dict(zip(js.name, js.position))
        try:
            return [float(pos_by_name[name]) for name in self.joint_names]
        except KeyError:
            return None

    @staticmethod
    def _yolo_redetect_nudge_patterns(cfg: Dict[str, Any]) -> list[tuple[float, float, float]]:
        step = float(max(0.001, cfg.get("yolo_visual_center_redetect_nudge_step_m", 0.012)))
        s, s2 = step, 2.0 * step
        return [
            (s, 0.0, 0.0),
            (-s, 0.0, 0.0),
            (0.0, s, 0.0),
            (0.0, -s, 0.0),
            (s, s, 0.0),
            (-s, s, 0.0),
            (s, -s, 0.0),
            (-s, -s, 0.0),
            (0.0, 0.0, s),
            (0.0, 0.0, -s),
            (s2, 0.0, 0.0),
            (-s2, 0.0, 0.0),
            (0.0, s2, 0.0),
            (0.0, -s2, 0.0),
            (s2, s2, 0.0),
            (-s2, -s2, 0.0),
        ]

    def _yolo_redetect_at_last_pose_with_nudges(
        self,
        action: Dict[str, Any],
        cfg: Dict[str, Any],
        det_track: _YoloDetectionTrack,
        model: Any,
        *,
        last_joint_positions: list[float],
        target_class: str,
        min_conf: float,
        yolo_iou: float,
        roi: int,
        ray_kw: Dict[str, Any],
        optical: str,
        base: str,
        tool_frame: str,
        tf_timeout: float,
    ) -> bool:
        """Return to last pose where YOLO worked, then small optical-frame nudges until detection returns."""
        reloc_action = dict(action)
        vs = cfg.get("yolo_visual_center_redetect_velocity_scaling")
        if vs is not None:
            reloc_action["velocity_scaling"] = float(vs)

        self._yolo_moveit_servo_halt(cfg)
        self._clear_vision_buffers()

        def _try_detect(phase_tag: str) -> bool:
            det_raw = self._yolo_snapshot_and_detect(
                model,
                target_class=target_class,
                min_conf=min_conf,
                yolo_iou=yolo_iou,
                roi=roi,
                ray_kw=ray_kw,
            )
            if det_raw is None:
                return False
            ok, det, _ = det_track.push(
                det_raw,
                log_fn=self.get_logger().info,
                phase_tag=phase_tag,
            )
            return bool(ok and det is not None)

        skip_snap = bool(cfg.get("yolo_redetect_skip_snap_back_after_servo", True))
        home = list(last_joint_positions)
        current = self._current_manipulator_joint_positions()
        if _try_detect("yolo_visual_center redetect at current pose"):
            self.get_logger().info("yolo_visual_center: re-detected at current joint pose (no snap-back).")
            return True

        if not skip_snap:
            self.get_logger().warn(
                "yolo_visual_center: lost track — returning to last detection pose and nudging camera."
            )
            if current is not None:
                max_err = max(abs(a - b) for a, b in zip(current, home))
                if max_err < 0.02:
                    home = list(current)
            if not self._execute_joint_goal(reloc_action, home):
                if current is not None and _try_detect("yolo_visual_center redetect after failed snap"):
                    self.get_logger().warn(
                        "yolo_visual_center: snap-back MoveIt goal failed; continuing from current pose."
                    )
                else:
                    return False
            self._clear_vision_buffers()
            if _try_detect("yolo_visual_center redetect at last pose"):
                self.get_logger().info("yolo_visual_center: re-detected at last known joint pose.")
                return True
        else:
            self.get_logger().warn(
                "yolo_visual_center: lost track — nudging camera from current pose "
                "(servo snap-back disabled)."
            )

        max_trials = max(1, int(cfg.get("yolo_visual_center_redetect_nudge_max_trials", 16)))
        patterns = self._yolo_redetect_nudge_patterns(cfg)[:max_trials]
        for idx, (dx, dy, dz) in enumerate(patterns):
            self._raise_if_abort()
            pose = self._tool_pose_shift_optical_origin(
                dx,
                dy,
                dz,
                optical_frame=optical,
                base_frame=base,
                tool_frame=tool_frame,
                tf_timeout_sec=tf_timeout,
            )
            if pose is None:
                continue
            self.get_logger().info(
                f"yolo_visual_center redetect nudge {idx + 1}/{len(patterns)}: "
                f"optical Δ=({dx:.3f},{dy:.3f},{dz:.3f}) m"
            )
            if not self._move_cartesian_waypoint_then_ik(action, pose):
                continue
            if _try_detect(f"yolo_visual_center redetect nudge {idx + 1}"):
                self.get_logger().info(
                    f"yolo_visual_center: re-detected after optical nudge "
                    f"({dx:.3f},{dy:.3f},{dz:.3f}) m."
                )
                return True

        self.get_logger().warn(
            f"yolo_visual_center: no re-detection after {len(patterns)} camera nudge(s); "
            "restoring last known pose."
        )
        self._execute_joint_goal(reloc_action, home)
        self._clear_vision_buffers()
        return False

    def _yolo_miss_redetect_if_needed(
        self,
        action: Dict[str, Any],
        cfg: Dict[str, Any],
        det_track: _YoloDetectionTrack,
        model: Any,
        *,
        last_joint_positions: Optional[list[float]],
        consecutive_miss: int,
        redetect_sessions_left: int,
        redetect_miss_threshold: Optional[int] = None,
        target_class: str,
        min_conf: float,
        yolo_iou: float,
        roi: int,
        ray_kw: Dict[str, Any],
        optical: str,
        base: str,
        tool_frame: str,
        tf_timeout: float,
    ) -> tuple[int, int, bool]:
        """
        After consecutive_miss raw YOLO failures, return to last good pose and nudge camera.
        Returns (consecutive_miss, redetect_sessions_left, recovered_ok).
        """
        if redetect_miss_threshold is not None:
            threshold = max(1, int(redetect_miss_threshold))
        else:
            threshold = max(1, int(cfg.get("yolo_visual_center_redetect_after_misses", 10)))
        if consecutive_miss < threshold:
            return consecutive_miss, redetect_sessions_left, False
        if redetect_sessions_left <= 0:
            return consecutive_miss, redetect_sessions_left, False
        if last_joint_positions is None:
            return consecutive_miss, redetect_sessions_left, False

        recovered = self._yolo_redetect_at_last_pose_with_nudges(
            action,
            cfg,
            det_track,
            model,
            last_joint_positions=last_joint_positions,
            target_class=target_class,
            min_conf=min_conf,
            yolo_iou=yolo_iou,
            roi=roi,
            ray_kw=ray_kw,
            optical=optical,
            base=base,
            tool_frame=tool_frame,
            tf_timeout=tf_timeout,
        )
        det_track.reset()
        left = redetect_sessions_left - 1
        if recovered:
            self.get_logger().info(
                f"yolo_visual_center: redetect session succeeded ({left} session(s) remaining)."
            )
        else:
            self.get_logger().warn(
                f"yolo_visual_center: redetect session failed ({left} session(s) remaining)."
            )
        return 0, left, recovered

    def _get_yolo_model(self, path: str) -> Any:
        if YOLO is None:
            raise RuntimeError(
                "yolo_visual_center requires Python package 'ultralytics' (pip install ultralytics)."
            )
        p = os.path.expanduser(path)
        if not p:
            raise ValueError(
                "vision.yolo_model_path is empty; set it in ur3_action_sequence.yaml (vision: section)."
            )
        if not os.path.isfile(p):
            raise FileNotFoundError(f"YOLO weights not found: {p}")
        if self._yolo_model is None or self._yolo_loaded_path != p:
            self.get_logger().info(f"Loading YOLO weights: {p}")
            self._yolo_model = YOLO(p)
            self._yolo_loaded_path = p
        return self._yolo_model

    def _yolo_detection_center_ray(
        self,
        model: Any,
        bgr: np.ndarray,
        depth: np.ndarray,
        info: CameraInfo,
        *,
        target_class: str,
        min_conf: float,
        yolo_iou: float,
        roi: int,
        predict_conf_floor: float = 0.01,
        class_match_case_insensitive: bool = True,
    ) -> Optional[tuple[int, int, float, float, float, float, float, float]]:
        """
        Run YOLO and return bbox image center back-projected in camera optical coords:
        u, v, x_cam, y_cam, z, du/dv vs CameraInfo principal point (cx,cy)—same convention as ray—det_confidence.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        infer_conf = float(min(max(1e-6, predict_conf_floor), min_conf))
        results = model.predict(rgb, conf=infer_conf, iou=yolo_iou, verbose=False)[0]
        best_conf = -1.0
        best_box = None
        names = results.names
        boxes = results.boxes
        target_cmp = target_class.strip()
        if class_match_case_insensitive:
            target_cmp = target_cmp.lower()
        if boxes is not None and boxes.cls is not None:
            for i in range(len(boxes)):
                cid = int(boxes.cls[i].item())
                cf = float(boxes.conf[i].item())
                cname = str(names[cid]).strip()
                if class_match_case_insensitive:
                    if cname.lower() != target_cmp:
                        continue
                elif cname != target_class.strip():
                    continue
                if cf > best_conf:
                    best_conf = cf
                    best_box = boxes.xyxy[i].cpu().numpy()

        if best_box is None or best_conf < min_conf:
            return None

        x1, y1, x2, y2 = [int(round(v)) for v in best_box]
        h, w = bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2

        K = info.k
        fx, fy = float(K[0]), float(K[4])
        cx, cy = float(K[2]), float(K[5])

        patch = depth[max(0, v - roi) : min(h, v + roi + 1), max(0, u - roi) : min(w, u + roi + 1)]
        z = float(np.nanmedian(patch))
        if not math.isfinite(z) or z <= 0.05 or z > 10.0:
            return None

        x_cam = (float(u) - cx) * z / fx
        y_cam = (float(v) - cy) * z / fy
        du = float(u) - cx
        dv = float(v) - cy
        return u, v, float(x_cam), float(y_cam), float(z), du, dv, float(best_conf)

    def _lookup_tf_mat(self, target_frame: str, source_frame: str, timeout_sec: float) -> Optional[np.ndarray]:
        """
        TF transform converting a point from `source_frame` to `target_frame`:
        p_target = R @ p_source + t (4x4 stores R,t).
        """
        try:
            st = self._tf_buffer.lookup_transform(
                target_frame, source_frame, Time(), timeout=RclDuration(seconds=float(timeout_sec))
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"TF {target_frame} <- {source_frame} failed: {exc}")
            return None
        q = st.transform.rotation
        qv = np.array([float(q.x), float(q.y), float(q.z), float(q.w)], dtype=np.float64)
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = _quat_to_rot(qv)
        mat[0, 3] = float(st.transform.translation.x)
        mat[1, 3] = float(st.transform.translation.y)
        mat[2, 3] = float(st.transform.translation.z)
        return mat

    def _servo_trigger_service(self, service_name: str, timeout_sec: float = 8.0) -> bool:
        client = self._servo_trigger_clients.get(service_name)
        if client is None:
            client = self.create_client(Trigger, service_name)
            self._servo_trigger_clients[service_name] = client
        if not self._wait_for_service_abortable(client, timeout_sec):
            self.get_logger().error(f"MoveIt Servo service not available: {service_name}")
            return False
        future = client.call_async(Trigger.Request())
        self._spin_until_future_complete_abortable(future, timeout_sec + 2.0)
        if not future.done() or future.result() is None:
            self.get_logger().error(f"MoveIt Servo service call timed out: {service_name}")
            return False
        result = future.result()
        if not bool(result.success):
            self.get_logger().warn(
                f"MoveIt Servo service {service_name} returned success=false: {getattr(result, 'message', '')}"
            )
            return False
        return True

    def _ensure_servo_twist_publisher(self, cfg: Dict[str, Any]) -> None:
        topic = str(cfg.get("yolo_servo_twist_topic", "/servo_node/delta_twist_cmds"))
        if self._servo_twist_pub is None or self._servo_twist_topic != topic:
            self._servo_twist_pub = self.create_publisher(TwistStamped, topic, 10)
            self._servo_twist_topic = topic

    @staticmethod
    def _yolo_bbox_centered(du_px: float, dv_px: float, tol_px: float) -> bool:
        return abs(float(du_px)) <= float(tol_px) and abs(float(dv_px)) <= float(tol_px)

    @staticmethod
    def _yolo_bbox_offsets(
        det: tuple[int, int, float, float, float, float, float, float],
        det_raw: Optional[tuple[int, int, float, float, float, float, float, float]],
        *,
        prefer_raw: bool,
    ) -> tuple[float, float, float]:
        """Return (du, dv, z_ray) for centering; raw frame preferred when available."""
        if prefer_raw and det_raw is not None:
            return float(det_raw[5]), float(det_raw[6]), float(det_raw[4])
        return float(det[5]), float(det[6]), float(det[4])

    @staticmethod
    def _yolo_optical_nudge_m(
        du_px: float,
        dv_px: float,
        fx: float,
        fy: float,
        z_cam: float,
        *,
        step_gain: float,
        max_step_m: float,
    ) -> tuple[float, float]:
        """Map pixel offset to a bounded optical XY nudge (metres)."""
        if z_cam <= 1e-6 or fx <= 1e-6 or fy <= 1e-6:
            return 0.0, 0.0
        z = float(z_cam)
        du = float(du_px)
        dv = float(dv_px)
        # Pinhole lateral error in camera frame; negate so TF translation reduces |du|,|dv|.
        dx = -float(step_gain) * (du / float(fx)) * z
        dy = -float(step_gain) * (dv / float(fy)) * z
        cap = abs(float(max_step_m))
        dx = max(-cap, min(cap, dx))
        dy = max(-cap, min(cap, dy))
        return dx, dy

    @staticmethod
    def _pose_translation_m(a: Pose, b: Pose) -> float:
        return float(
            math.hypot(
                a.position.x - b.position.x,
                a.position.y - b.position.y,
                a.position.z - b.position.z,
            )
        )

    def _spin_vision_settle(self, frames: int = 8, period_sec: float = 0.05) -> None:
        n = max(1, int(frames))
        period = max(0.01, float(period_sec))
        for _ in range(n):
            self._raise_if_abort()
            rclpy.spin_once(self, timeout_sec=period)

    def _tool_pose_lateral_nudge_from_pixels(
        self,
        du_px: float,
        dv_px: float,
        z_cam: float,
        fx: float,
        fy: float,
        *,
        step_gain: float,
        max_step_m: float,
        nudge_sign: float,
        optical_frame: str,
        base_frame: str,
        tf_timeout_sec: float,
    ) -> Optional[Pose]:
        """Translate tool0 in base along camera X/Y (keeps current orientation)."""
        pose_now = self._lookup_tool0_pose()
        if pose_now is None:
            return None
        T_base_opt = self._lookup_tf_mat(base_frame, optical_frame, tf_timeout_sec)
        if T_base_opt is None:
            return None
        if z_cam <= 1e-6 or fx <= 1e-6 or fy <= 1e-6:
            return None
        R = T_base_opt[:3, :3]
        x_cam = (float(du_px) / float(fx)) * float(z_cam)
        y_cam = (float(dv_px) / float(fy)) * float(z_cam)
        sign = float(nudge_sign)
        delta_cam = np.array(
            [-sign * float(step_gain) * x_cam, -sign * float(step_gain) * y_cam, 0.0],
            dtype=np.float64,
        )
        xy_norm = float(np.linalg.norm(delta_cam[:2]))
        cap = abs(float(max_step_m))
        if xy_norm < 1e-9:
            return None
        if xy_norm > cap > 0.0:
            delta_cam[:2] *= cap / xy_norm
        delta_base = R @ delta_cam

        out = Pose()
        out.orientation = pose_now.orientation
        out.position.x = float(pose_now.position.x + delta_base[0])
        out.position.y = float(pose_now.position.y + delta_base[1])
        out.position.z = float(pose_now.position.z + delta_base[2])
        return out

    def _yolo_build_smooth_centering_waypoints(
        self,
        pose_target: Pose,
        *,
        segments: int,
    ) -> list[Pose]:
        """Interpolate tool0 position toward ``pose_target`` for one continuous Cartesian path."""
        pose_now = self._lookup_tool0_pose()
        if pose_now is None:
            return []
        n = max(1, int(segments))
        waypoints: list[Pose] = []
        for idx in range(1, n + 1):
            frac = float(idx) / float(n)
            wp = Pose()
            wp.orientation = pose_now.orientation
            wp.position.x = float(pose_now.position.x + frac * (pose_target.position.x - pose_now.position.x))
            wp.position.y = float(pose_now.position.y + frac * (pose_target.position.y - pose_now.position.y))
            wp.position.z = float(pose_now.position.z + frac * (pose_target.position.z - pose_now.position.z))
            waypoints.append(wp)
        return waypoints

    def _yolo_execute_centering_cartesian_stream(
        self,
        center_action: Dict[str, Any],
        waypoints: list[Pose],
        cfg: Dict[str, Any],
        *,
        min_move_m: float,
    ) -> tuple[bool, float]:
        """One continuous Cartesian trajectory (no per-step IK joint goals)."""
        if not waypoints:
            return False, 0.0
        before = self._lookup_tool0_pose()
        if before is None:
            return False, 0.0

        smooth = dict(center_action)
        smooth["velocity_scaling"] = float(
            cfg.get("yolo_center_velocity_scaling", center_action.get("velocity_scaling", 0.05))
        )
        smooth["acceleration_scaling"] = float(
            cfg.get("yolo_center_acceleration_scaling", center_action.get("acceleration_scaling", 0.05))
        )
        smooth["max_step_m"] = float(
            cfg.get("yolo_center_cartesian_max_step_m", self.motion_cfg.get("default_max_step_m", 0.002))
        )
        min_fraction = float(cfg.get("yolo_center_min_fraction", 0.75))

        for avoid in (bool(smooth.get("avoid_collisions", True)), False):
            trial = dict(smooth, avoid_collisions=avoid)
            cres = self._execute_cartesian_path(trial, waypoints, min_fraction)
            if cres:
                after = self._lookup_tool0_pose()
                if after is None:
                    return False, 0.0
                delta_m = self._pose_translation_m(before, after)
                if delta_m >= float(min_move_m):
                    return True, delta_m
                self.get_logger().warn(
                    f"yolo_visual_center: Cartesian stream finished but tool0 moved only "
                    f"{delta_m * 1000.0:.2f} mm."
                )
        return False, 0.0

    def _twist_linear_to_command_frame(
        self,
        v_base: np.ndarray,
        *,
        base_frame: str,
        command_frame: str,
        tf_timeout_sec: float,
    ) -> Optional[np.ndarray]:
        """Express a linear velocity (defined in ``base_frame`` axes) in ``command_frame`` axes."""
        if command_frame == base_frame:
            return v_base
        t_cmd_base = self._lookup_tf_mat(command_frame, base_frame, tf_timeout_sec)
        if t_cmd_base is None:
            return None
        return t_cmd_base[:3, :3] @ v_base

    def _yolo_servo_twist_optical_ibvs(
        self,
        du_px: float,
        dv_px: float,
        fx: float,
        fy: float,
        z_cam: float,
        *,
        cfg: Dict[str, Any],
        tol_px: float,
        period_sec: float,
        fine_pass: bool = False,
    ) -> Twist:
        """
        Pinhole IBVS velocity in the camera frame (same direction as Cartesian nudges).
        Command frame must match ``camera_optical_frame`` / MoveIt ``robot_link_command_frame``.
        """
        du = float(du_px)
        dv = float(dv_px)
        z = float(z_cam)
        nudge_sign = float(cfg.get("yolo_center_nudge_sign", 1.0))
        # MuJoCo rl_camera: global ibvs_sign cannot fix both axes; use per-axis signs.
        sign_u = nudge_sign * float(cfg.get("yolo_servo_ibvs_sign_u", cfg.get("yolo_servo_ibvs_sign", 1.0)))
        sign_v = nudge_sign * float(cfg.get("yolo_servo_ibvs_sign_v", cfg.get("yolo_servo_ibvs_sign", -1.0)))
        gain = float(cfg.get("yolo_servo_linear_gain", cfg.get("yolo_center_step_gain", 0.35)))
        gain *= float(cfg.get("yolo_servo_cartesian_speed_scale", 0.55))

        v_cam = np.array(
            [
                -sign_u * gain * (du / float(fx)) * z,
                -sign_v * gain * (dv / float(fy)) * z,
                0.0,
            ],
            dtype=np.float64,
        )
        v_cam /= max(float(period_sec), 1e-3)

        metric = abs(du) + abs(dv)
        slow_px = float(cfg.get("yolo_servo_slow_error_px", 100.0))
        if metric > slow_px > 0.0:
            v_cam[:2] *= slow_px / metric
        ramp_px = float(cfg.get("yolo_servo_near_center_ramp_px", max(3.0 * float(tol_px), 100.0)))
        if not fine_pass and ramp_px > 0.0 and metric < ramp_px:
            min_scale = float(cfg.get("yolo_servo_near_center_min_scale", 0.35))
            v_cam[:2] *= max(min_scale, (metric / ramp_px) ** 1.2)

        max_lin = float(cfg.get("yolo_servo_max_linear_m_s", 0.012))
        if fine_pass:
            max_lin = min(max_lin, float(cfg.get("yolo_servo_fine_pass_max_linear_m_s", 0.005)))
        v_norm = float(np.linalg.norm(v_cam))
        if v_norm > max_lin > 0.0:
            v_cam *= max_lin / v_norm

        twist = Twist()
        twist.linear.x = float(v_cam[0])
        twist.linear.y = float(v_cam[1])
        twist.linear.z = float(v_cam[2])
        return twist

    def _yolo_servo_twist_from_cartesian_nudge(
        self,
        du_px: float,
        dv_px: float,
        fx: float,
        fy: float,
        z_cam: float,
        *,
        cfg: Dict[str, Any],
        base_frame: str,
        optical_frame: str,
        tf_timeout_sec: float,
        tol_px: float,
        period_sec: float,
        fine_pass: bool = False,
    ) -> Optional[Twist]:
        """
        Servo velocity = (Cartesian optical nudge in base) / period (legacy tool0/base path).
        """
        if z_cam <= 1e-6 or fx <= 1e-6 or fy <= 1e-6:
            return None
        du = float(du_px)
        dv = float(dv_px)

        pose_now = self._lookup_tool0_pose()
        if pose_now is None:
            return None
        pose_goal = self._tool_pose_lateral_nudge_from_pixels(
            du,
            dv,
            z_cam,
            fx,
            fy,
            step_gain=float(cfg.get("yolo_center_step_gain", cfg.get("step_gain", 0.35))),
            max_step_m=float(cfg.get("yolo_center_max_step_m", 0.012)),
            nudge_sign=float(cfg.get("yolo_center_nudge_sign", 1.0)),
            optical_frame=optical_frame,
            base_frame=base_frame,
            tf_timeout_sec=tf_timeout_sec,
        )
        if pose_goal is None:
            return None

        delta_base = np.array(
            [
                float(pose_goal.position.x - pose_now.position.x),
                float(pose_goal.position.y - pose_now.position.y),
                float(pose_goal.position.z - pose_now.position.z),
            ],
            dtype=np.float64,
        )
        if float(np.linalg.norm(delta_base)) < 1e-9:
            return Twist()

        speed_scale = float(cfg.get("yolo_servo_cartesian_speed_scale", 0.55))
        v_opt = delta_base * speed_scale / max(float(period_sec), 1e-3)

        metric = abs(du) + abs(dv)
        slow_px = float(cfg.get("yolo_servo_slow_error_px", 100.0))
        if metric > slow_px > 0.0:
            v_opt *= slow_px / metric
        ramp_px = float(cfg.get("yolo_servo_near_center_ramp_px", max(3.0 * float(tol_px), 100.0)))
        if not fine_pass and ramp_px > 0.0 and metric < ramp_px:
            min_scale = float(cfg.get("yolo_servo_near_center_min_scale", 0.35))
            v_opt *= max(min_scale, (metric / ramp_px) ** 1.2)

        max_lin = float(cfg.get("yolo_servo_max_linear_m_s", 0.008))
        if fine_pass:
            max_lin = min(max_lin, float(cfg.get("yolo_servo_fine_pass_max_linear_m_s", 0.005)))
        v_norm = float(np.linalg.norm(v_opt))
        if v_norm > max_lin > 0.0:
            v_opt *= max_lin / v_norm

        # MoveIt Servo (joint_trajectory out) moves opposite to Cartesian position nudges unless flipped.
        ibvs_sign = float(cfg.get("yolo_servo_ibvs_sign", -1.0))
        v_opt *= ibvs_sign

        cmd_frame = str(cfg.get("yolo_servo_command_frame", "rl_camera_frame"))
        v_cmd = self._twist_linear_to_command_frame(
            v_opt,
            base_frame=base_frame,
            command_frame=cmd_frame,
            tf_timeout_sec=tf_timeout_sec,
        )
        if v_cmd is None:
            return None

        twist = Twist()
        twist.linear.x = float(v_cmd[0])
        twist.linear.y = float(v_cmd[1])
        twist.linear.z = float(v_cmd[2])
        return twist

    @staticmethod
    def _yolo_servo_smooth_offset(
        du: float,
        dv: float,
        state: Dict[str, Optional[float]],
        *,
        alpha: float,
    ) -> tuple[float, float]:
        """EMA on pixel error for Servo only (reduces bbox jitter shake)."""
        a = float(max(0.05, min(1.0, alpha)))
        if state.get("du") is None:
            state["du"] = float(du)
            state["dv"] = float(dv)
            return float(du), float(dv)
        du_s = a * float(du) + (1.0 - a) * float(state["du"])
        dv_s = a * float(dv) + (1.0 - a) * float(state["dv"])
        state["du"] = du_s
        state["dv"] = dv_s
        return du_s, dv_s

    def _yolo_servo_blend_twist(
        self,
        twist: Twist,
        state: Dict[str, Optional[np.ndarray]],
        *,
        alpha: float,
    ) -> Twist:
        """Low-pass filter commanded twist between frames."""
        v = np.array(
            [twist.linear.x, twist.linear.y, twist.linear.z],
            dtype=np.float64,
        )
        a = float(max(0.05, min(1.0, alpha)))
        prev = state.get("v")
        if prev is None:
            state["v"] = v
            return twist
        v_blend = a * v + (1.0 - a) * prev
        state["v"] = v_blend
        out = Twist()
        out.linear.x = float(v_blend[0])
        out.linear.y = float(v_blend[1])
        out.linear.z = float(v_blend[2])
        return out

    @staticmethod
    def _yolo_servo_slew_limit_twist(
        twist: Twist,
        state: Dict[str, Optional[np.ndarray]],
        *,
        max_delta_m: float,
    ) -> Twist:
        """Cap per-tick change in commanded linear velocity (removes high-frequency shake)."""
        cap = abs(float(max_delta_m))
        if cap <= 0.0:
            return twist
        v = np.array(
            [twist.linear.x, twist.linear.y, twist.linear.z],
            dtype=np.float64,
        )
        prev = state.get("slew")
        if prev is None:
            state["slew"] = v
            return twist
        delta = v - prev
        d_norm = float(np.linalg.norm(delta))
        if d_norm > cap:
            v = prev + delta * (cap / d_norm)
        state["slew"] = v
        out = Twist()
        out.linear.x = float(v[0])
        out.linear.y = float(v[1])
        out.linear.z = float(v[2])
        return out

    def _yolo_servo_one_cartesian_correction(
        self,
        action: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        du: float,
        dv: float,
        z_cam: float,
        fx: float,
        fy: float,
        optical: str,
        base: str,
        tf_timeout: float,
    ) -> bool:
        """Single streamed Cartesian nudge when Servo diverges (same geometry as centering mode)."""
        self._yolo_moveit_servo_halt(cfg)
        pose_goal = self._tool_pose_lateral_nudge_from_pixels(
            du,
            dv,
            z_cam,
            fx,
            fy,
            step_gain=float(cfg.get("yolo_center_step_gain", cfg.get("step_gain", 0.35))),
            max_step_m=float(cfg.get("yolo_center_max_step_m", 0.012)),
            nudge_sign=float(cfg.get("yolo_center_nudge_sign", 1.0)),
            optical_frame=optical,
            base_frame=base,
            tf_timeout_sec=tf_timeout,
        )
        if pose_goal is None:
            return False
        segments = max(4, int(cfg.get("yolo_center_path_segments", 20)))
        waypoints = self._yolo_build_smooth_centering_waypoints(pose_goal, segments=segments)
        ok, _ = self._yolo_execute_centering_cartesian_stream(
            action,
            waypoints,
            cfg,
            min_move_m=float(cfg.get("yolo_center_min_move_m", 0.0003)),
        )
        if ok:
            self.get_logger().warn(
                f"yolo_visual_center servo: divergence — applied one Cartesian correction "
                f"(offset_px=({du:.1f},{dv:.1f}))."
            )
        return ok

    def _publish_servo_twist(self, cfg: Dict[str, Any], twist: Twist) -> None:
        self._ensure_servo_twist_publisher(cfg)
        cmd_frame = str(cfg.get("yolo_servo_command_frame", "rl_camera_frame"))
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = cmd_frame
        msg.twist = twist
        self._servo_twist_pub.publish(msg)

    def _yolo_moveit_servo_publish_stop(self, cfg: Dict[str, Any], *, repeats: int = 4) -> None:
        """Zero twist without pausing Servo (keeps incoming_command_timeout alive)."""
        zero = Twist()
        for _ in range(max(1, int(repeats))):
            self._publish_servo_twist(cfg, zero)
            time.sleep(0.02)

    def _yolo_moveit_servo_halt(self, cfg: Dict[str, Any]) -> None:
        self._yolo_moveit_servo_publish_stop(cfg, repeats=6)
        pause_svc = str(cfg.get("yolo_servo_pause_service", "/servo_node/pause_servo"))
        self._servo_trigger_service(pause_svc, timeout_sec=3.0)

    def _yolo_moveit_servo_start(self, cfg: Dict[str, Any]) -> bool:
        start_svc = str(cfg.get("yolo_servo_start_service", "/servo_node/start_servo"))
        unpause_svc = str(cfg.get("yolo_servo_unpause_service", "/servo_node/unpause_servo"))
        if not self._servo_trigger_service(start_svc):
            return False
        if not self._servo_trigger_service(unpause_svc):
            return False
        self._ensure_servo_twist_publisher(cfg)
        self.get_logger().info(
            f"yolo_visual_center: MoveIt Servo active ({cfg.get('yolo_servo_twist_topic', '/servo_node/delta_twist_cmds')})."
        )
        return True

    def _yolo_moveit_servo_ibvs_loop(
        self,
        action: Dict[str, Any],
        cfg: Dict[str, Any],
        det_track: _YoloDetectionTrack,
        model: Any,
        *,
        target_class: str,
        min_conf: float,
        yolo_iou: float,
        roi: int,
        ray_kw: Dict[str, Any],
        optical: str,
        base: str,
        tf_timeout: float,
        tol_px: float,
        _track_frame: Callable[..., tuple],
        _on_raw_miss: Callable[[], bool],
        _save_last_detection_pose: Callable[[], None],
        fine_pass: bool,
        phase_tag: str,
    ) -> bool:
        """One Servo IBVS pass; convergence is judged on raw bbox offset (not EMA)."""
        use_raw_only = bool(cfg.get("yolo_servo_use_raw_only", True))

        rate_hz = max(5.0, float(cfg.get("yolo_servo_rate_hz", 30.0)))
        period = 1.0 / rate_hz
        duration = (
            float(cfg.get("yolo_servo_fine_pass_max_duration_sec", 20.0))
            if fine_pass
            else float(cfg.get("yolo_servo_max_duration_sec", 15.0))
        )
        deadline = time.monotonic() + duration
        stable_need = max(1, int(cfg.get("yolo_servo_stable_frames", 3)))
        stable = 0
        log_period = max(1, int(rate_hz * 0.5))
        tick = 0
        servo_miss = 0
        raw_streak = 0
        redetect_hold = max(
            5,
            int(cfg.get("yolo_servo_redetect_after_misses", cfg.get("yolo_center_redetect_after_misses", 12))),
        )
        warmup_need = max(1, int(cfg.get("yolo_servo_warmup_raw_frames", 3)))
        cross_brake_frames = max(0, int(cfg.get("yolo_servo_cross_brake_frames", 0)))
        offset_ema = float(cfg.get("yolo_servo_offset_ema_alpha", 0.45))
        vel_ema = float(cfg.get("yolo_servo_velocity_ema_alpha", 0.4))
        offset_state: Dict[str, Optional[float]] = {"du": None, "dv": None}
        vel_state: Dict[str, Optional[np.ndarray]] = {"v": None, "slew": None}
        use_smooth_twist = bool(cfg.get("yolo_servo_use_smoothed_for_twist", True))
        max_twist_delta = float(cfg.get("yolo_servo_max_twist_delta_m_s", 0.004)) * period
        last_metric: Optional[float] = None
        last_du_servo: Optional[float] = None
        last_dv_servo: Optional[float] = None
        brake_ticks = 0
        worsen_streak = 0
        diverge_need = max(0, int(cfg.get("yolo_servo_diverge_correct_after", 0)))
        diverge_min_metric = float(cfg.get("yolo_servo_diverge_min_metric_px", 120.0))
        ibvs_sign = float(cfg.get("yolo_servo_ibvs_sign", -1.0))
        use_optical_ibvs = bool(cfg.get("yolo_servo_use_optical_ibvs", True))
        cmd_on_keepalive = bool(cfg.get("yolo_servo_cmd_on_keepalive", True))
        diverge_abort_streak = max(0, int(cfg.get("yolo_servo_diverge_abort_streak", 0)))
        max_axis_offset = float(cfg.get("yolo_servo_max_axis_offset_px", 95.0))

        sign_u = float(cfg.get("yolo_servo_ibvs_sign_u", cfg.get("yolo_servo_ibvs_sign", 1.0)))
        sign_v = float(cfg.get("yolo_servo_ibvs_sign_v", cfg.get("yolo_servo_ibvs_sign", -1.0)))
        cmd_frame = str(cfg.get("yolo_servo_command_frame", "rl_camera_frame"))
        self.get_logger().info(
            f"yolo_visual_center: Servo {phase_tag} (rate={rate_hz:.0f} Hz, "
            f"max_v={cfg.get('yolo_servo_max_linear_m_s', 0.012)} m/s, "
            f"sign_u={sign_u:+.1f} sign_v={sign_v:+.1f}, optical_ibvs={use_optical_ibvs}, "
            f"raw_only={use_raw_only}, fov_guard={max_axis_offset:.0f}px, "
            f"vel_ema={vel_ema:.2f}, warmup_raw={warmup_need}, frame={cmd_frame}, tol={tol_px:.0f}px)."
        )
        if use_raw_only:
            self.get_logger().info(
                "yolo_visual_center servo: bypassing median/EMA tracker (raw YOLO per frame)."
            )
        cartesian_nudge_period = float(cfg.get("yolo_servo_cartesian_nudge_every_sec", 2.5))
        cartesian_nudge_min_metric = float(cfg.get("yolo_servo_cartesian_nudge_min_metric_px", 50.0))
        last_cartesian_nudge_t = time.monotonic()
        last_improve_metric_t = time.monotonic()

        while time.monotonic() < deadline and rclpy.ok():
            self._raise_if_abort()
            tick += 1
            keep_alive = False
            raw_miss = False
            if use_raw_only:
                det_raw = self._yolo_snapshot_and_detect(
                    model,
                    target_class=target_class,
                    min_conf=min_conf,
                    yolo_iou=yolo_iou,
                    roi=roi,
                    ray_kw=ray_kw,
                )
                if det_raw is None:
                    ok, det, raw_miss = False, None, True
                else:
                    ok, det, raw_miss = True, det_raw, False
            else:
                ok, det, keep_alive, det_raw, raw_miss = _track_frame(
                    f"yolo_visual_center servo {phase_tag}"
                )

            if not ok or det is None:
                self._yolo_moveit_servo_publish_stop(cfg)
                raw_streak = 0
                if not det_track.had_good:
                    self.get_logger().error(
                        f"No {target_class!r} detection for MoveIt Servo centering."
                    )
                    return False
                time.sleep(period)
                continue

            using_keepalive = False
            if det_raw is None:
                if cmd_on_keepalive and keep_alive and det is not None:
                    du_raw, dv_raw, z_ray = self._yolo_bbox_offsets(det, None, prefer_raw=False)
                    using_keepalive = True
                else:
                    self._yolo_moveit_servo_publish_stop(cfg)
                    raw_streak = 0
                    servo_miss += 1
                    if servo_miss < redetect_hold:
                        rclpy.spin_once(self, timeout_sec=0.0)
                        time.sleep(period)
                        continue
                    if _on_raw_miss():
                        stable = 0
                        servo_miss = 0
                        raw_streak = 0
                        self._yolo_moveit_servo_start(cfg)
                        continue
                    continue
            else:
                raw_streak += 1
                du_raw, dv_raw, z_ray = self._yolo_bbox_offsets(det, det_raw, prefer_raw=True)

            _save_last_detection_pose()
            servo_miss = 0
            du_raw_f = float(du_raw)
            dv_raw_f = float(dv_raw)
            du_smooth, dv_smooth = self._yolo_servo_smooth_offset(
                du_raw_f,
                dv_raw_f,
                offset_state,
                alpha=offset_ema,
            )
            du_cmd = du_smooth if use_smooth_twist else du_raw_f
            dv_cmd = dv_smooth if use_smooth_twist else dv_raw_f
            _, _, _, _, _, _, _, conf = det

            if raw_streak < warmup_need and not using_keepalive:
                self._yolo_moveit_servo_publish_stop(cfg)
                if tick % log_period == 0:
                    self.get_logger().info(
                        f"yolo_visual_center servo: warming up raw lock "
                        f"({raw_streak}/{warmup_need}) offset_px=({du_cmd:.1f},{dv_cmd:.1f})"
                    )
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(period)
                continue

            if brake_ticks > 0:
                self._yolo_moveit_servo_publish_stop(cfg)
                brake_ticks -= 1
                if tick % log_period == 0:
                    self.get_logger().info(
                        f"yolo_visual_center servo: cross-brake "
                        f"({brake_ticks} frames left) offset_px=({du_cmd:.1f},{dv_cmd:.1f})"
                    )
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(period)
                continue

            if last_du_servo is not None and last_dv_servo is not None:
                crossed_u = du_cmd * last_du_servo < 0.0 and abs(du_cmd) < abs(last_du_servo)
                crossed_v = dv_cmd * last_dv_servo < 0.0 and abs(dv_cmd) < abs(last_dv_servo)
                metric_now = abs(du_cmd) + abs(dv_cmd)
                if (crossed_u or crossed_v) and metric_now < max(
                    3.0 * float(tol_px), float(cfg.get("yolo_servo_near_center_ramp_px", 45.0))
                ):
                    brake_ticks = max(brake_ticks, cross_brake_frames)
                    self._yolo_moveit_servo_publish_stop(cfg)
                    if tick % log_period == 0:
                        self.get_logger().info(
                            "yolo_visual_center servo: zero-cross brake "
                            f"(Δu {last_du_servo:.0f}→{du_cmd:.0f}, Δv {last_dv_servo:.0f}→{dv_cmd:.0f})"
                        )
                    last_du_servo = du_cmd
                    last_dv_servo = dv_cmd
                    rclpy.spin_once(self, timeout_sec=0.0)
                    time.sleep(period)
                    continue
            last_du_servo = du_cmd
            last_dv_servo = dv_cmd

            metric_raw = abs(du_raw_f) + abs(dv_raw_f)
            if self._yolo_bbox_centered(du_raw_f, dv_raw_f, tol_px):
                stable += 1
                self._yolo_moveit_servo_publish_stop(cfg)
                if stable >= stable_need:
                    self.get_logger().info(
                        f"yolo_visual_center: centered via MoveIt Servo ({phase_tag}) "
                        f"raw=(Δu={du_raw_f:.1f}, Δv={dv_raw_f:.1f}) px, conf={conf:.3f}, "
                        f"stable_frames={stable_need}."
                    )
                    return True
            else:
                stable = 0
                with self._vis_lock:
                    cam_info = self._vis_info
                if cam_info is None:
                    time.sleep(period)
                    continue
                fx = float(cam_info.k[0])
                fy = float(cam_info.k[4])
                metric = metric_raw

                if abs(du_raw_f) > max_axis_offset or abs(dv_raw_f) > max_axis_offset:
                    self.get_logger().warn(
                        f"yolo_visual_center servo {phase_tag}: near FOV edge "
                        f"(Δu={du_raw_f:.1f}, Δv={dv_raw_f:.1f} px) — Cartesian recovery."
                    )
                    self._yolo_moveit_servo_publish_stop(cfg)
                    if self._yolo_servo_one_cartesian_correction(
                        action,
                        cfg,
                        du=float(du_cmd),
                        dv=float(dv_cmd),
                        z_cam=float(z_ray),
                        fx=fx,
                        fy=fy,
                        optical=optical,
                        base=base,
                        tf_timeout=tf_timeout,
                    ):
                        self._yolo_moveit_servo_start(cfg)
                        self._spin_vision_settle(12, 0.05)
                        last_cartesian_nudge_t = time.monotonic()
                        last_improve_metric_t = time.monotonic()
                        last_metric = None
                        offset_state["du"] = None
                        offset_state["dv"] = None
                        vel_state["v"] = None
                        vel_state["slew"] = None
                    continue

                if last_metric is not None and metric > last_metric + 12.0:
                    worsen_streak += 1
                else:
                    worsen_streak = max(0, worsen_streak - 1)

                if (
                    diverge_abort_streak > 0
                    and worsen_streak >= diverge_abort_streak
                    and metric >= max(80.0, 2.5 * float(tol_px))
                ):
                    self.get_logger().error(
                        "yolo_visual_center servo: pixel error growing — "
                        f"|Δ|={metric:.0f}px (was {last_metric:.0f}px). "
                        f"Check yolo_servo_ibvs_sign (current {ibvs_sign:+.1f}; try "
                        f"{-ibvs_sign:+.1f})."
                    )
                    return False

                if diverge_need > 0 and metric >= diverge_min_metric and worsen_streak >= diverge_need:
                    worsen_streak = 0
                    last_metric = None
                    offset_state["du"] = None
                    offset_state["dv"] = None
                    vel_state["v"] = None
                    if self._yolo_servo_one_cartesian_correction(
                        action,
                        cfg,
                        du=float(du_cmd),
                        dv=float(dv_cmd),
                        z_cam=float(z_ray),
                        fx=fx,
                        fy=fy,
                        optical=optical,
                        base=base,
                        tf_timeout=tf_timeout,
                    ):
                        self._yolo_moveit_servo_start(cfg)
                        self._spin_vision_settle(8, 0.04)
                        last_cartesian_nudge_t = time.monotonic()
                        last_improve_metric_t = time.monotonic()
                    continue

                now_t = time.monotonic()
                if last_metric is not None and metric < last_metric - 3.0:
                    last_improve_metric_t = now_t

                need_cartesian_nudge = (
                    cartesian_nudge_period > 0.0
                    and metric >= cartesian_nudge_min_metric
                    and (now_t - last_cartesian_nudge_t) >= cartesian_nudge_period
                    and (now_t - last_improve_metric_t) >= cartesian_nudge_period
                )
                if need_cartesian_nudge:
                    self.get_logger().info(
                        f"yolo_visual_center servo {phase_tag}: periodic Cartesian nudge "
                        f"(|Δ|={metric:.0f}px)."
                    )
                    if self._yolo_servo_one_cartesian_correction(
                        action,
                        cfg,
                        du=float(du_cmd),
                        dv=float(dv_cmd),
                        z_cam=float(z_ray),
                        fx=fx,
                        fy=fy,
                        optical=optical,
                        base=base,
                        tf_timeout=tf_timeout,
                    ):
                        self._yolo_moveit_servo_start(cfg)
                        self._spin_vision_settle(10, 0.05)
                        last_cartesian_nudge_t = now_t
                        last_improve_metric_t = now_t
                        last_metric = None
                        offset_state["du"] = None
                        offset_state["dv"] = None
                        vel_state["v"] = None
                    continue

                if use_optical_ibvs:
                    twist = self._yolo_servo_twist_optical_ibvs(
                        du_cmd,
                        dv_cmd,
                        fx,
                        fy,
                        float(z_ray),
                        cfg=cfg,
                        tol_px=tol_px,
                        period_sec=period,
                        fine_pass=fine_pass,
                    )
                else:
                    twist = self._yolo_servo_twist_from_cartesian_nudge(
                        du_cmd,
                        dv_cmd,
                        fx,
                        fy,
                        float(z_ray),
                        cfg=cfg,
                        base_frame=base,
                        optical_frame=optical,
                        tf_timeout_sec=tf_timeout,
                        tol_px=tol_px,
                        period_sec=period,
                        fine_pass=fine_pass,
                    )
                if twist is None:
                    self.get_logger().warn("yolo_visual_center servo: twist unavailable.")
                else:
                    twist = self._yolo_servo_blend_twist(twist, vel_state, alpha=vel_ema)
                    twist = self._yolo_servo_slew_limit_twist(
                        twist, vel_state, max_delta_m=max_twist_delta
                    )
                    self._publish_servo_twist(cfg, twist)
                    if tick % log_period == 0:
                        trend = ""
                        if last_metric is not None:
                            if metric < last_metric - 2.0:
                                trend = " ↓"
                            elif metric > last_metric + 2.0:
                                trend = " ↑"
                        last_metric = metric
                        ka = " keep-alive" if using_keepalive else ""
                        self.get_logger().info(
                            f"yolo_visual_center servo {phase_tag}{ka}: "
                            f"raw=({du_raw_f:.1f},{dv_raw_f:.1f}) cmd=({du_cmd:.1f},{dv_cmd:.1f}) "
                            f"|Δ|={metric:.0f}px{trend} "
                            f"v_{cmd_frame}=({twist.linear.x:.3f},{twist.linear.y:.3f},{twist.linear.z:.3f})"
                        )

            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

        self.get_logger().warn(
            f"yolo_visual_center: Servo {phase_tag} timed out after {duration:.1f}s."
        )
        return False

    def _yolo_moveit_servo_center_bbox(
        self,
        action: Dict[str, Any],
        cfg: Dict[str, Any],
        det_track: _YoloDetectionTrack,
        model: Any,
        *,
        target_class: str,
        min_conf: float,
        yolo_iou: float,
        roi: int,
        ray_kw: Dict[str, Any],
        optical: str,
        base: str,
        tool_frame: str,
        tf_timeout: float,
        tol_px: float,
        _track_frame: Callable[..., tuple],
        _on_raw_miss: Callable[[], bool],
        _save_last_detection_pose: Callable[[], None],
    ) -> bool:
        """IBVS centering via MoveIt Servo (coarse pass, optional fine pass)."""
        if not self._yolo_moveit_servo_start(cfg):
            return False
        det_track.reset()
        loop_kw = dict(
            action=action,
            cfg=cfg,
            det_track=det_track,
            model=model,
            target_class=target_class,
            min_conf=min_conf,
            yolo_iou=yolo_iou,
            roi=roi,
            ray_kw=ray_kw,
            optical=optical,
            base=base,
            tf_timeout=tf_timeout,
            tol_px=tol_px,
            _track_frame=_track_frame,
            _on_raw_miss=_on_raw_miss,
            _save_last_detection_pose=_save_last_detection_pose,
        )
        try:
            if self._yolo_moveit_servo_ibvs_loop(**loop_kw, fine_pass=False, phase_tag="coarse"):
                return True

            if bool(cfg.get("yolo_servo_cartesian_fallback", True)):
                det_chk = self._yolo_snapshot_and_detect(
                    model,
                    target_class=target_class,
                    min_conf=min_conf,
                    yolo_iou=yolo_iou,
                    roi=roi,
                    ray_kw=ray_kw,
                )
                if det_chk is not None:
                    du_c = float(det_chk[5])
                    dv_c = float(det_chk[6])
                    if self._yolo_bbox_centered(du_c, dv_c, tol_px):
                        self.get_logger().info(
                            f"yolo_visual_center: centered after coarse Servo "
                            f"(Δu={du_c:.1f}, Δv={dv_c:.1f} px)."
                        )
                        return True
                    metric_c = abs(du_c) + abs(dv_c)
                    if metric_c > 2.5 * float(tol_px):
                        self.get_logger().warn(
                            f"yolo_visual_center: coarse Servo stalled at |Δ|={metric_c:.0f}px — "
                            "Cartesian completion."
                        )
                        self._yolo_moveit_servo_halt(cfg)
                        if self._yolo_cartesian_center_bbox(
                            action,
                            cfg,
                            optical=optical,
                            base=base,
                            tool_frame=tool_frame,
                            tf_timeout=tf_timeout,
                            tol_px=tol_px,
                            _track_frame=_track_frame,
                            _on_raw_miss=_on_raw_miss,
                            _save_last_detection_pose=_save_last_detection_pose,
                        ):
                            return True

            if not bool(cfg.get("yolo_servo_fine_pass", True)):
                return False
            det_raw = self._yolo_snapshot_and_detect(
                model,
                target_class=target_class,
                min_conf=min_conf,
                yolo_iou=yolo_iou,
                roi=roi,
                ray_kw=ray_kw,
            )
            if det_raw is None:
                return False
            du = float(det_raw[5])
            dv = float(det_raw[6])
            metric = abs(du) + abs(dv)
            if self._yolo_bbox_centered(du, dv, tol_px) or metric > 3.0 * float(tol_px):
                return False
            self.get_logger().info(
                f"yolo_visual_center: Servo fine pass (raw |Δ|={metric:.0f}px, tol={tol_px:.0f}px)."
            )
            if not self._yolo_moveit_servo_start(cfg):
                return False
            if self._yolo_moveit_servo_ibvs_loop(**loop_kw, fine_pass=True, phase_tag="fine"):
                return True

            if bool(cfg.get("yolo_servo_cartesian_fallback", True)):
                self.get_logger().warn(
                    "yolo_visual_center: Servo IBVS did not converge — running Cartesian completion."
                )
                self._yolo_moveit_servo_halt(cfg)
                return self._yolo_cartesian_center_bbox(
                    action,
                    cfg,
                    optical=optical,
                    base=base,
                    tool_frame=tool_frame,
                    tf_timeout=tf_timeout,
                    tol_px=tol_px,
                    _track_frame=_track_frame,
                    _on_raw_miss=_on_raw_miss,
                    _save_last_detection_pose=_save_last_detection_pose,
                )
            return False
        finally:
            self._yolo_moveit_servo_halt(cfg)

    def _yolo_verify_centered_or_fail(
        self,
        cfg: Dict[str, Any],
        det_track: _YoloDetectionTrack,
        model: Any,
        *,
        target_class: str,
        min_conf: float,
        yolo_iou: float,
        roi: int,
        ray_kw: Dict[str, Any],
        tol_px: float,
        phase_tag: str,
    ) -> bool:
        """Final gate: require a live (non-keep-alive) detection within pixel tolerance."""
        if not bool(cfg.get("yolo_visual_center_require_centered_on_complete", True)):
            return True
        det_raw = self._yolo_snapshot_and_detect(
            model,
            target_class=target_class,
            min_conf=min_conf,
            yolo_iou=yolo_iou,
            roi=roi,
            ray_kw=ray_kw,
        )
        if det_raw is None:
            self.get_logger().error(
                f"yolo_visual_center: {phase_tag} — no live detection for final center check."
            )
            return False
        du = float(det_raw[5])
        dv = float(det_raw[6])
        if not self._yolo_bbox_centered(du, dv, tol_px):
            self.get_logger().error(
                f"yolo_visual_center: {phase_tag} — bbox not centered "
                f"(Δu={du:.1f}, Δv={dv:.1f} px, tolerance={tol_px:.1f})."
            )
            return False
        self.get_logger().info(
            f"yolo_visual_center: {phase_tag} — verified centered (Δu={du:.1f}, Δv={dv:.1f} px)."
        )
        return True

    def _yolo_cartesian_center_bbox(
        self,
        action: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        optical: str,
        base: str,
        tool_frame: str,
        tf_timeout: float,
        tol_px: float,
        _track_frame: Callable[..., tuple],
        _on_raw_miss: Callable[[], bool],
        _save_last_detection_pose: Callable[[], None],
    ) -> bool:
        """Center bbox with collision-aware Cartesian nudges in the camera optical frame."""
        max_iter = max(1, int(cfg.get("center_max_iterations") or cfg.get("max_iterations", 40)))
        step_gain = float(cfg.get("yolo_center_step_gain", cfg.get("step_gain", 0.5)))
        max_step = float(cfg.get("yolo_center_max_step_m", 0.025))
        prefer_raw = bool(cfg.get("yolo_center_use_raw_detection", True))
        deadline = time.monotonic() + float(cfg.get("yolo_center_max_duration_sec", 45.0))
        center_action = dict(action)
        center_action["avoid_collisions"] = bool(cfg.get("yolo_visual_center_avoid_collisions", True))

        self.get_logger().info(
            f"yolo_visual_center: streamed Cartesian centering (max_iter={max_iter}, "
            f"max_duration_sec={cfg.get('yolo_center_max_duration_sec', 60.0):.0f}, "
            f"step_gain={step_gain:.2f}, max_step_m={max_step:.3f}, "
            f"path_segments={int(cfg.get('yolo_center_path_segments', 4))}, "
            f"v_scale={float(cfg.get('yolo_center_velocity_scaling', 0.08)):.2f}, "
            f"avoid_collisions={center_action['avoid_collisions']})."
        )

        nudge_count = 0
        nudge_sign = float(cfg.get("yolo_center_nudge_sign", 1.0))
        min_move_m = float(cfg.get("yolo_center_min_move_m", 0.0003))
        path_segments = max(1, int(cfg.get("yolo_center_path_segments", 4)))
        center_misses = 0
        settle_frames = int(cfg.get("yolo_center_settle_frames", 25))
        settle_period = float(cfg.get("yolo_center_settle_period_sec", 0.06))
        while time.monotonic() < deadline and rclpy.ok():
            self._raise_if_abort()
            if nudge_count >= max_iter:
                break
            ok, det, keep_alive, det_raw, raw_miss = _track_frame("yolo_visual_center cartesian")
            if raw_miss:
                center_misses += 1
                if center_misses < max(5, int(cfg.get("yolo_center_redetect_after_misses", 30))):
                    self._spin_vision_settle(min(center_misses + 5, settle_frames), settle_period)
                    continue
                if _on_raw_miss():
                    center_misses = 0
                    self._spin_vision_settle(settle_frames, settle_period)
                    continue
                continue
            center_misses = 0
            if det_raw is not None:
                _save_last_detection_pose()
            if not ok or det is None:
                self.get_logger().error("yolo_visual_center: lost detection during Cartesian centering.")
                return False

            du, dv, z_ray = self._yolo_bbox_offsets(det, det_raw, prefer_raw=prefer_raw)
            _, _, _, _, _, _, _, conf = det
            if not keep_alive and self._yolo_bbox_centered(du, dv, tol_px):
                self.get_logger().info(
                    f"yolo_visual_center: centered via smooth Cartesian centering "
                    f"(nudges={nudge_count}, Δu={du:.1f}, Δv={dv:.1f} px, conf={conf:.3f})."
                )
                return True
            if keep_alive:
                self._spin_vision_settle(5, settle_period)
                continue

            with self._vis_lock:
                cam_info = self._vis_info
            if cam_info is None:
                continue
            fx = float(cam_info.k[0])
            fy = float(cam_info.k[4])
            pose_goal = self._tool_pose_lateral_nudge_from_pixels(
                du,
                dv,
                z_ray,
                fx,
                fy,
                step_gain=step_gain,
                max_step_m=max_step,
                nudge_sign=nudge_sign,
                optical_frame=optical,
                base_frame=base,
                tf_timeout_sec=tf_timeout,
            )
            if pose_goal is None:
                self.get_logger().warn("yolo_visual_center: lateral nudge pose unavailable (TF?).")
                continue
            waypoints = self._yolo_build_smooth_centering_waypoints(
                pose_goal, segments=path_segments
            )
            if not waypoints:
                continue
            pose_now = self._lookup_tool0_pose()
            planned_mm = 0.0
            if pose_now is not None:
                planned_mm = self._pose_translation_m(pose_now, waypoints[-1]) * 1000.0
            nudge_count += 1
            self.get_logger().info(
                f"yolo_visual_center cartesian stream {nudge_count}/{max_iter}: "
                f"offset_px=({du:.1f},{dv:.1f}) path={len(waypoints)} wp "
                f"total_shift≈{planned_mm:.1f} mm depth={z_ray:.3f}"
            )
            moved, delta_m = self._yolo_execute_centering_cartesian_stream(
                center_action, waypoints, cfg, min_move_m=min_move_m
            )
            if not moved:
                self.get_logger().warn(
                    f"yolo_visual_center: smooth centering nudge {nudge_count} failed "
                    "(Cartesian / IK)."
                )
                continue
            # Anchor redetect to the NEW pose so we never snap back (avoids shake).
            _save_last_detection_pose()
            self.get_logger().info(
                f"yolo_visual_center: cartesian stream moved {delta_m * 1000.0:.1f} mm — camera settle."
            )
            self._spin_vision_settle(settle_frames, settle_period)

        self.get_logger().error(
            f"yolo_visual_center: Cartesian centering did not converge "
            f"({nudge_count} nudge(s), tolerance={tol_px:.1f} px)."
        )
        return False

    def _tool_pose_lock_tcp_pick_orientation(
        self,
        lookat: np.ndarray,
        tcp_roll_rad: float,
        *,
        base_frame: str,
        tool_frame: str,
        tf_timeout_sec: float,
    ) -> Optional[Pose]:
        """Current tool0 position with orientation from pick-style lookat_vector + tcp_roll_rad."""
        T_base_tool = self._lookup_tf_mat(base_frame, tool_frame, tf_timeout_sec)
        if T_base_tool is None:
            return None
        T_base_tool = T_base_tool.copy()
        T_base_tool[:3, :3] = _tcp_rotation_from_lookat(lookat, float(tcp_roll_rad))
        return _pose_from_transform(T_base_tool)

    def _move_tool_pose_ik_joint(self, action: Dict[str, Any], pose: Pose) -> bool:
        """IK + joint trajectory only (no Cartesian path; keeps orientation goals exact)."""
        joint_positions = self._solve_ik(pose, action)
        if joint_positions is None:
            return False
        return self._execute_joint_goal(action, joint_positions)

    def _tool_pose_translate_along_optical_forward(
        self,
        delta_forward_m: float,
        *,
        optical_frame: str,
        base_frame: str,
        tool_frame: str,
        tf_timeout_sec: float,
    ) -> Optional[Pose]:
        """Translate camera optical origin along current optical +Z in base by delta_forward_m (meters)."""
        T_base_opt = self._lookup_tf_mat(base_frame, optical_frame, tf_timeout_sec)
        if T_base_opt is None:
            return None
        T_tool_opt = self._lookup_tf_mat(tool_frame, optical_frame, tf_timeout_sec)
        if T_tool_opt is None:
            return None
        R = T_base_opt[:3, :3]
        optical_z_in_base = R[:, 2]
        new_o = T_base_opt[:3, 3] + optical_z_in_base * float(delta_forward_m)
        T_goal = np.eye(4, dtype=np.float64)
        T_goal[:3, :3] = R
        T_goal[:3, 3] = new_o
        T_base_tool = T_goal @ np.linalg.inv(T_tool_opt)
        return _pose_from_transform(T_base_tool)

    def _tool_pose_shift_optical_origin(
        self,
        delta_x_cam: float,
        delta_y_cam: float,
        delta_z_cam: float,
        *,
        optical_frame: str,
        base_frame: str,
        tool_frame: str,
        tf_timeout_sec: float,
    ) -> Optional[Pose]:
        """Translate camera optical origin in optical axes (meters); orientation unchanged."""
        T_base_opt = self._lookup_tf_mat(base_frame, optical_frame, tf_timeout_sec)
        if T_base_opt is None:
            return None
        T_tool_opt = self._lookup_tf_mat(tool_frame, optical_frame, tf_timeout_sec)
        if T_tool_opt is None:
            return None
        R = T_base_opt[:3, :3]
        delta_base = R @ np.array([delta_x_cam, delta_y_cam, delta_z_cam], dtype=np.float64)
        T_goal = np.eye(4, dtype=np.float64)
        T_goal[:3, :3] = R
        T_goal[:3, 3] = T_base_opt[:3, 3] + delta_base
        T_base_tool = T_goal @ np.linalg.inv(T_tool_opt)
        return _pose_from_transform(T_base_tool)

    def _tool_poses_chain_optical_forward(
        self,
        forward_steps_m: Sequence[float],
        *,
        optical_frame: str,
        base_frame: str,
        tool_frame: str,
        tf_timeout_sec: float,
    ) -> list[Pose]:
        """Build tool0 poses for a chain of optical +Z moves from the current view (one continuous path)."""
        if not forward_steps_m:
            return []
        T_base_opt = self._lookup_tf_mat(base_frame, optical_frame, tf_timeout_sec)
        if T_base_opt is None:
            return []
        T_tool_opt = self._lookup_tf_mat(tool_frame, optical_frame, tf_timeout_sec)
        if T_tool_opt is None:
            return []
        R = T_base_opt[:3, :3]
        origin = T_base_opt[:3, 3].copy()
        optical_z = R[:, 2]
        inv_tool_opt = np.linalg.inv(T_tool_opt)
        poses: list[Pose] = []
        cumulative = 0.0
        for step_m in forward_steps_m:
            cumulative += float(step_m)
            T_goal = np.eye(4, dtype=np.float64)
            T_goal[:3, :3] = R
            T_goal[:3, 3] = origin + optical_z * cumulative
            poses.append(_pose_from_transform(T_goal @ inv_tool_opt))
        return poses

    @staticmethod
    def _yolo_plan_optical_forward_steps(
        z_cam: float,
        dist_target: float,
        tol_z: float,
        approach_clip: float,
        max_steps: int,
        *,
        single_motion: bool,
        single_trans_cap: float,
        min_det_z: float,
    ) -> list[float]:
        steps: list[float] = []
        z_est = float(z_cam)
        for _ in range(max(1, max_steps)):
            if abs(z_est - dist_target) <= tol_z:
                break
            step = float(z_est - dist_target)
            if min_det_z > 0.0 and z_est < min_det_z:
                retreat = float(min_det_z - z_est)
                if step < retreat:
                    step = retreat
            if single_motion:
                step = max(-single_trans_cap, min(single_trans_cap, step))
            elif abs(step) > approach_clip:
                step = math.copysign(approach_clip, step)
            if abs(step) < 1e-6:
                break
            steps.append(step)
            z_est -= step
        return steps

    def _move_cartesian_waypoints_then_ik(self, action: Dict[str, Any], waypoints: list[Pose]) -> bool:
        if not waypoints:
            return True
        min_fraction = float(action.get("min_fraction", self.motion_cfg["default_min_fraction"]))
        cres = self._execute_cartesian_path(action, waypoints, min_fraction)
        if cres:
            return True
        self.get_logger().warn(
            f"Cartesian path through {len(waypoints)} waypoint(s) failed — trying last waypoint only."
        )
        return self._move_cartesian_waypoint_then_ik(action, waypoints[-1])

    def _solve_ik_scaled_steps(self, pose: Pose, action: Dict[str, Any]) -> Optional[list[float]]:
        """Interpolate tool0 **position** toward ``pose`` while using full target orientation; retries with smaller steps."""
        step_scales = (1.0, 0.5, 0.25, 0.125)
        for idx, step_scale in enumerate(step_scales):
            pose_now = self._lookup_tool0_pose()
            if pose_now is None:
                return None
            tgt = Pose()
            tgt.orientation = pose.orientation
            tgt.position.x = float(pose_now.position.x + step_scale * (pose.position.x - pose_now.position.x))
            tgt.position.y = float(pose_now.position.y + step_scale * (pose.position.y - pose_now.position.y))
            tgt.position.z = float(pose_now.position.z + step_scale * (pose.position.z - pose_now.position.z))
            is_last = idx == len(step_scales) - 1
            joint_positions = self._solve_ik(tgt, action, silent_ik_failure=not is_last)
            if joint_positions is not None:
                if idx > 0:
                    self.get_logger().warn(f"IK succeeded with step_scale={step_scale}.")
                return joint_positions
        return None

    def _move_cartesian_waypoint_then_ik(self, action: Dict[str, Any], waypoint: Pose) -> bool:
        """Try MoveIt Cartesian path to a single waypoint, then IK + joint trajectory fallback."""
        min_fraction = float(action.get("min_fraction", self.motion_cfg["default_min_fraction"]))
        cres = self._execute_cartesian_path(action, [waypoint], min_fraction)
        if cres:
            return True
        self.get_logger().warn("Cartesian path for approach waypoint failed/incomplete — trying IK fallback.")
        joint_positions = self._solve_ik_scaled_steps(waypoint, action)
        if joint_positions is None:
            return False
        return self._execute_joint_goal(action, joint_positions)

    def _lookup_tool0_pose(self) -> Optional[Pose]:
        base = str(self.motion_cfg["base_frame"])
        try:
            t = self._tf_buffer.lookup_transform(base, "tool0", Time(), timeout=RclDuration(seconds=3.0))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"TF {base} -> tool0 failed: {exc}")
            return None
        pose = Pose()
        pose.position.x = float(t.transform.translation.x)
        pose.position.y = float(t.transform.translation.y)
        pose.position.z = float(t.transform.translation.z)
        pose.orientation = t.transform.rotation
        return pose

    def _execute_yolo_visual_center_action(self, action: Dict[str, Any]) -> bool:
        """
        YOLO + RGB-D: center bbox in image, then approach along optical +Z.

        Centering mode ``yolo_visual_center_centering_mode``:
        - ``cartesian`` (default): collision-aware optical-frame nudges via MoveIt Cartesian + IK.
        - ``servo``: linear IBVS twists on MoveIt Servo (``rl_camera_frame``, no wrist rotation).
        """
        cfg = {**self.vision_cfg}
        for key, val in action.items():
            if key in ("type", "name"):
                continue
            if val is not None:
                cfg[key] = val

        target_class = str(cfg.get("target_class") or cfg.get("default_target_class") or self.object_name).strip()
        if not target_class:
            raise ValueError(
                "yolo_visual_center requires target_class in the action, vision.default_target_class, "
                "or object_name."
            )

        min_conf = float(cfg["min_confidence"])
        dist_target = float(cfg["target_distance_m"])
        approach_iters = max(1, int(cfg["approach_max_iterations"]))
        approach_clip = float(cfg["approach_step_max_m"])
        single_motion = bool(cfg.get("yolo_visual_center_single_motion", False))
        single_trans_cap = float(max(0.01, cfg.get("yolo_visual_center_single_motion_max_translation_m", 2.0)))

        tol_px = float(cfg["center_tolerance_px"])
        tol_z = float(cfg["distance_tolerance_m"])
        roi = int(cfg["depth_roi_half_px"])
        yolo_iou = float(cfg["yolo_iou"])
        optical = str(cfg["camera_optical_frame"])
        wait_t = float(cfg["wait_image_timeout_sec"])
        tf_timeout = 2.0

        base = str(self.motion_cfg["base_frame"])
        tool_frame = str(self.motion_cfg["ik_link_name"])

        ap_cap = 1 if single_motion else approach_iters
        centering_mode = str(cfg.get("yolo_visual_center_centering_mode", "cartesian")).lower()
        self.get_logger().info(
            f"yolo_visual_center: class={target_class!r}, min_confidence={min_conf}, "
            f"target_distance_m={dist_target:.3f}, centering_mode={centering_mode}, "
            f"center_max_iter={int(cfg.get('center_max_iterations') or cfg.get('max_iterations', 20))}, "
            f"approach_iterations={'1 (single_motion)' if single_motion else approach_iters}"
        )
        if single_motion:
            self.get_logger().warn(
                "yolo_visual_center_single_motion: one centering move max, then one approach move "
                f"(|Δz_optical| cap {single_trans_cap:.2f} m). Not iterative centering."
            )

        mw = max(1, int(cfg["yolo_tracking_median_window"]))
        ema_a = float(cfg["yolo_tracking_ema_alpha"])
        ka_max = int(cfg["yolo_tracking_keep_alive_max_misses"])
        ka_log = int(cfg["yolo_tracking_log_keep_alive_every"])
        det_track = _YoloDetectionTrack(
            median_window=mw,
            ema_alpha=ema_a,
            max_keep_alive_misses=ka_max,
            log_every=ka_log,
        )
        ka_desc = "unlimited" if ka_max <= 0 else str(ka_max)
        self.get_logger().info(
            f"yolo_tracking: median_window={mw}, ema_alpha={ema_a:.3f}, "
            f"keep_alive_max_misses={ka_desc}, keep_alive_log_every={max(1, ka_log)}"
        )

        self._ensure_vision_subscriptions_for_yolo()

        if not self._wait_for_joint_state(action):
            return False
        if not self._wait_vision_frames(wait_t):
            self.get_logger().error("Timed out waiting for camera RGB/depth/camera_info.")
            return False

        model_path = str(cfg.get("yolo_model_path", ""))
        try:
            model = self._get_yolo_model(model_path)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(str(exc))
            return False

        p_floor = float(cfg.get("yolo_predict_conf_floor", 0.01))
        cls_ci = bool(cfg.get("yolo_class_match_case_insensitive", True))
        acquire_n = max(1, int(cfg.get("yolo_visual_center_acquire_max_attempts", 120)))
        acquire_spin = float(max(0.0, cfg.get("yolo_visual_center_acquire_spin_sec", 0.05)))
        acquire_conf = float(cfg.get("yolo_visual_center_acquire_min_confidence", min(min_conf, 0.25)))
        ray_kw: Dict[str, Any] = {
            "predict_conf_floor": p_floor,
            "class_match_case_insensitive": cls_ci,
        }
        redetect_sessions_left = max(0, int(cfg.get("yolo_visual_center_redetect_max_sessions", 3)))
        redetect_misses = max(1, int(cfg.get("yolo_visual_center_redetect_after_misses", 10)))
        consecutive_miss = 0
        last_detected_joints: Optional[list[float]] = None
        min_det_z = float(max(0.0, cfg.get("yolo_visual_center_min_detection_depth_m", 0.0)))
        if min_det_z > 0.0 and dist_target < min_det_z:
            self.get_logger().warn(
                f"yolo_visual_center: target_distance_m={dist_target:.3f} is below "
                f"min_detection_depth_m={min_det_z:.3f}; using min depth as approach floor."
            )
            dist_target = min_det_z
        self.get_logger().info(
            f"yolo_visual_center: redetect after {redetect_misses} raw miss(es) "
            f"(return to last pose + camera nudges), max_sessions={redetect_sessions_left}, "
            f"min_detection_depth_m={min_det_z:.3f}"
        )

        def _save_last_detection_pose() -> None:
            nonlocal last_detected_joints
            joints = self._current_manipulator_joint_positions()
            if joints is not None:
                last_detected_joints = joints

        def _track_frame(phase_tag: str) -> tuple[
            bool,
            Optional[tuple[int, int, float, float, float, float, float, float]],
            bool,
            Optional[tuple[int, int, float, float, float, float, float, float]],
            bool,
        ]:
            return self._yolo_snapshot_track(
                det_track,
                model,
                target_class=target_class,
                min_conf=min_conf,
                yolo_iou=yolo_iou,
                roi=roi,
                ray_kw=ray_kw,
                phase_tag=phase_tag,
            )

        def _on_raw_miss() -> bool:
            """True if redetect session recovered tracking; caller should retry detection."""
            nonlocal consecutive_miss, redetect_sessions_left
            consecutive_miss += 1
            consecutive_miss, redetect_sessions_left, recovered = self._yolo_miss_redetect_if_needed(
                action,
                cfg,
                det_track,
                model,
                last_joint_positions=last_detected_joints,
                consecutive_miss=consecutive_miss,
                redetect_sessions_left=redetect_sessions_left,
                target_class=target_class,
                min_conf=min_conf,
                yolo_iou=yolo_iou,
                roi=roi,
                ray_kw=ray_kw,
                optical=optical,
                base=base,
                tool_frame=tool_frame,
                tf_timeout=tf_timeout,
            )
            return recovered

        def _on_raw_miss_for_centering() -> bool:
            """Redetect during centering without snapping back to a stale pre-nudge pose."""
            nonlocal consecutive_miss, redetect_sessions_left
            if centering_mode in ("servo", "smooth", "moveit_servo", "ibvs", "velocity"):
                self._yolo_moveit_servo_publish_stop(cfg)
            center_thresh = max(1, int(cfg.get("yolo_center_redetect_after_misses", 30)))
            consecutive_miss = max(consecutive_miss + 1, center_thresh)
            consecutive_miss, redetect_sessions_left, recovered = self._yolo_miss_redetect_if_needed(
                action,
                cfg,
                det_track,
                model,
                last_joint_positions=last_detected_joints,
                consecutive_miss=consecutive_miss,
                redetect_sessions_left=redetect_sessions_left,
                redetect_miss_threshold=1,
                target_class=target_class,
                min_conf=min_conf,
                yolo_iou=yolo_iou,
                roi=roi,
                ray_kw=ray_kw,
                optical=optical,
                base=base,
                tool_frame=tool_frame,
                tf_timeout=tf_timeout,
            )
            if recovered:
                consecutive_miss = 0
            return recovered

        # --- Bootstrap: lock tracker (redetect at last pose after N raw misses).
        acquire_attempts = 0
        self.get_logger().info(
            f"yolo_visual_center: acquisition uses min_confidence={acquire_conf:.2f} "
            f"(tracking uses {min_conf:.2f})."
        )
        while acquire_attempts < acquire_n:
            self._raise_if_abort()
            rclpy.spin_once(self, timeout_sec=0.0)
            det_raw_acquire = self._yolo_snapshot_and_detect(
                model,
                target_class=target_class,
                min_conf=acquire_conf,
                yolo_iou=yolo_iou,
                roi=roi,
                ray_kw=ray_kw,
            )
            if det_raw_acquire is not None:
                ok, det, _ = det_track.push(
                    det_raw_acquire,
                    log_fn=self.get_logger().info,
                    phase_tag="yolo_visual_center acquire",
                )
                det_raw = det_raw_acquire
                raw_miss = False
            else:
                ok, det, _, det_raw, raw_miss = _track_frame("yolo_visual_center acquire")
            if det_raw is not None:
                _save_last_detection_pose()
            if raw_miss:
                if last_detected_joints is not None and _on_raw_miss():
                    acquire_attempts = 0
                    continue
                if consecutive_miss >= redetect_misses and redetect_sessions_left <= 0:
                    break
                acquire_attempts += 1
                if acquire_spin > 0.0:
                    time.sleep(acquire_spin)
                continue
            consecutive_miss = 0
            if ok and det is not None:
                _save_last_detection_pose()
                self.get_logger().info(
                    f"yolo_visual_center: detection lock acquired after {acquire_attempts + 1} frame(s)."
                )
                break
            acquire_attempts += 1
            if acquire_spin > 0.0:
                time.sleep(acquire_spin)
        else:
            self.get_logger().error(
                f"No {target_class!r} detection after {acquire_n} acquisition frame(s) "
                f"(redetect threshold={redetect_misses}, sessions left={redetect_sessions_left}) — "
                "check target_class, min_confidence, depth, or move farther (min_detection_depth_m)."
            )
            return False

        # --- Phase 1: center bbox in the image.
        if single_motion and centering_mode == "servo":
            self.get_logger().warn(
                "yolo_visual_center_single_motion: centering still runs full servo loop; "
                "only approach is capped to one step."
            )
        center_kw = dict(
            action=action,
            cfg=cfg,
            optical=optical,
            base=base,
            tool_frame=tool_frame,
            tf_timeout=tf_timeout,
            tol_px=tol_px,
            _track_frame=_track_frame,
            _on_raw_miss=_on_raw_miss_for_centering,
            _save_last_detection_pose=_save_last_detection_pose,
        )
        if centering_mode in ("servo", "smooth", "moveit_servo", "ibvs", "velocity"):
            centered = self._yolo_moveit_servo_center_bbox(
                action,
                cfg,
                det_track,
                model,
                target_class=target_class,
                min_conf=min_conf,
                yolo_iou=yolo_iou,
                roi=roi,
                ray_kw=ray_kw,
                optical=optical,
                base=base,
                tool_frame=tool_frame,
                tf_timeout=tf_timeout,
                tol_px=tol_px,
                _track_frame=_track_frame,
                _on_raw_miss=_on_raw_miss_for_centering,
                _save_last_detection_pose=_save_last_detection_pose,
            )
        elif centering_mode in ("cartesian", "moveit", "default", ""):
            centered = self._yolo_cartesian_center_bbox(**center_kw)
        else:
            raise ValueError(
                f"Unsupported yolo_visual_center_centering_mode: {centering_mode!r} "
                "(use 'servo'/'smooth', or 'cartesian')."
            )
        if not centered:
            return False
        if not self._yolo_verify_centered_or_fail(
            cfg,
            det_track,
            model,
            target_class=target_class,
            min_conf=min_conf,
            yolo_iou=yolo_iou,
            roi=roi,
            ray_kw=ray_kw,
            tol_px=tol_px,
            phase_tag="after centering",
        ):
            return False

        prefer_raw = bool(cfg.get("yolo_center_use_raw_detection", True))

        def _approach_offsets(
            det: tuple[int, int, float, float, float, float, float, float],
            det_raw: Optional[tuple[int, int, float, float, float, float, float, float]],
        ) -> tuple[float, float, float]:
            return self._yolo_bbox_offsets(det, det_raw, prefer_raw=prefer_raw)

        def _require_centered_for_approach(
            du: float,
            dv: float,
            *,
            keep_alive: bool,
            phase: str,
        ) -> bool:
            if keep_alive:
                self.get_logger().error(
                    f"yolo_visual_center: {phase} — cannot continue approach on keep-alive; need live detection."
                )
                return False
            if not self._yolo_bbox_centered(du, dv, tol_px):
                self.get_logger().error(
                    f"yolo_visual_center: {phase} — bbox not centered "
                    f"(Δu={du:.1f}, Δv={dv:.1f} px, tolerance={tol_px:.1f})."
                )
                return False
            return True

        if bool(cfg.get("yolo_visual_center_restore_pick_orientation_before_approach", False)):
            lookat_raw = action.get("lookat_vector", cfg.get("lookat_vector", [0.0, 0.0, -1.0]))
            lookat = _normalize_lookat_vector(_vector3(lookat_raw, "lookat_vector"))
            tcp_roll = float(action.get("tcp_roll_rad", cfg.get("tcp_roll_rad", 0.0)))
            pose_pick = self._tool_pose_lock_tcp_pick_orientation(
                lookat,
                tcp_roll,
                base_frame=base,
                tool_frame=tool_frame,
                tf_timeout_sec=tf_timeout,
            )
            if pose_pick is not None:
                self.get_logger().info(
                    "yolo_visual_center: restoring pick TCP orientation "
                    f"(lookat={lookat.tolist()}, tcp_roll_rad={tcp_roll:.3f}) before approach — "
                    "camera will leave the centered view."
                )
                if not self._move_tool_pose_ik_joint(action, pose_pick):
                    self.get_logger().warn(
                        "yolo_visual_center: pick-orientation restore failed; continuing approach at current pose."
                    )
        else:
            self.get_logger().info(
                "yolo_visual_center: keeping current TCP orientation after centering "
                "(restore_pick_orientation_before_approach is false)."
            )

        # --- Phase 2: approach along optical +Z to target_distance_m.
        use_smooth_approach = bool(cfg.get("yolo_visual_center_smooth_approach", True))

        def _approach_step_list(z_cam_in: float) -> list[float]:
            return self._yolo_plan_optical_forward_steps(
                z_cam_in,
                dist_target,
                tol_z,
                approach_clip,
                ap_cap,
                single_motion=single_motion,
                single_trans_cap=single_trans_cap,
                min_det_z=min_det_z,
            )

        if use_smooth_approach:
            ok, det, keep_alive, det_raw, raw_miss = _track_frame("yolo_visual_center approach")
            if det_raw is not None:
                _save_last_detection_pose()
            if raw_miss or not ok or det is None:
                self.get_logger().warn(
                    "yolo_visual_center: smooth approach — detection not ready; using stepped approach."
                )
            else:
                du0, dv0, z_cam0 = _approach_offsets(det, det_raw)
                _, _, _, _, _, _, _, conf0 = det
                if not _require_centered_for_approach(
                    du0, dv0, keep_alive=keep_alive, phase="smooth approach"
                ):
                    return False
                if abs(z_cam0 - dist_target) <= tol_z:
                    self.get_logger().info(
                        f"yolo_visual_center: depth already at target ({z_cam0:.3f} m), conf={conf0:.3f}"
                    )
                    return self._yolo_verify_centered_or_fail(
                        cfg,
                        det_track,
                        model,
                        target_class=target_class,
                        min_conf=min_conf,
                        yolo_iou=yolo_iou,
                        roi=roi,
                        ray_kw=ray_kw,
                        tol_px=tol_px,
                        phase_tag="depth at target",
                    )
                else:
                    fwd_steps = _approach_step_list(z_cam0)
                    poses = self._tool_poses_chain_optical_forward(
                        fwd_steps,
                        optical_frame=optical,
                        base_frame=base,
                        tool_frame=tool_frame,
                        tf_timeout_sec=tf_timeout,
                    )
                    if poses:
                        self.get_logger().info(
                            f"yolo_visual_center: smooth approach — {len(poses)} waypoint(s), "
                            f"total ΔZ_optical={sum(fwd_steps):.3f} m (one trajectory)."
                        )
                        if not self._move_cartesian_waypoints_then_ik(action, poses):
                            self.get_logger().warn(
                                "yolo_visual_center: smooth approach move failed; using stepped approach."
                            )
                        else:
                            ok2, det2, keep_alive2, det_raw2, raw_miss2 = _track_frame(
                                "yolo_visual_center approach verify"
                            )
                            if ok2 and det2 is not None:
                                du2, dv2, z2 = _approach_offsets(det2, det_raw2)
                                _, _, _, _, _, _, _, conf2 = det2
                                if abs(z2 - dist_target) <= tol_z:
                                    if not _require_centered_for_approach(
                                        du2, dv2, keep_alive=keep_alive2, phase="smooth approach verify"
                                    ):
                                        return False
                                    self.get_logger().info(
                                        f"yolo_visual_center: depth converged after smooth approach "
                                        f"(z={z2:.3f} m, conf={conf2:.3f})."
                                    )
                                    return self._yolo_verify_centered_or_fail(
                                        cfg,
                                        det_track,
                                        model,
                                        target_class=target_class,
                                        min_conf=min_conf,
                                        yolo_iou=yolo_iou,
                                        roi=roi,
                                        ray_kw=ray_kw,
                                        tol_px=tol_px,
                                        phase_tag="after smooth approach",
                                    )
                            if ok2 and det2 is not None:
                                z_fix = float(det_raw2[4]) if det_raw2 is not None else float(det2[4])
                                fix_steps = _approach_step_list(z_fix)
                                if fix_steps:
                                    pose_fix = self._tool_pose_translate_along_optical_forward(
                                        fix_steps[0],
                                        optical_frame=optical,
                                        base_frame=base,
                                        tool_frame=tool_frame,
                                        tf_timeout_sec=tf_timeout,
                                    )
                                    if pose_fix is not None and self._move_cartesian_waypoint_then_ik(
                                        action, pose_fix
                                    ):
                                        ok3, det3, _, dr3, _ = _track_frame(
                                            "yolo_visual_center approach verify"
                                        )
                                        if ok3 and det3 is not None:
                                            du3, dv3, z3 = _approach_offsets(det3, dr3)
                                            if abs(z3 - dist_target) <= tol_z:
                                                if not _require_centered_for_approach(
                                                    du3, dv3, keep_alive=False, phase="approach trim"
                                                ):
                                                    return False
                                                self.get_logger().info(
                                                    f"yolo_visual_center: depth converged after "
                                                    f"approach trim (z={z3:.3f} m)."
                                                )
                                                return self._yolo_verify_centered_or_fail(
                                                    cfg,
                                                    det_track,
                                                    model,
                                                    target_class=target_class,
                                                    min_conf=min_conf,
                                                    yolo_iou=yolo_iou,
                                                    roi=roi,
                                                    ray_kw=ray_kw,
                                                    tol_px=tol_px,
                                                    phase_tag="after approach trim",
                                                )
                            self.get_logger().warn(
                                "yolo_visual_center: smooth approach finished but depth not in tolerance; "
                                "using stepped approach."
                            )

        ap = 0
        while ap < ap_cap:
            self._raise_if_abort()
            ok, det, keep_alive, det_raw, raw_miss = _track_frame("yolo_visual_center approach")
            if det_raw is not None:
                _save_last_detection_pose()
            if raw_miss:
                if _on_raw_miss():
                    continue
                if consecutive_miss >= redetect_misses and redetect_sessions_left <= 0:
                    self.get_logger().error(
                        f"yolo_visual_center approach: no {target_class!r} after {consecutive_miss} raw miss(es) "
                        "and no redetect sessions remaining."
                    )
                    return False
                continue
            if not ok or det is None:
                if ka_max > 0 and det_track.miss_count > ka_max:
                    self.get_logger().error(
                        f"yolo_visual_center approach: lost detection for {det_track.miss_count} consecutive "
                        f"frames (> keep_alive_max_misses={ka_max})."
                    )
                elif not det_track.had_good:
                    self.get_logger().error(
                        f"yolo_visual_center approach: no prior lock on {target_class!r}; cannot continue."
                    )
                elif consecutive_miss >= redetect_misses and redetect_sessions_left <= 0:
                    self.get_logger().error(
                        f"yolo_visual_center approach: lost {target_class!r} after {consecutive_miss} raw miss(es) "
                        "and no redetect sessions remaining."
                    )
                else:
                    self.get_logger().error("yolo_visual_center approach: detection tracking failed.")
                return False
            consecutive_miss = 0
            _save_last_detection_pose()
            du, dv, z_cam = _approach_offsets(det, det_raw)
            _, _, _, _, _, _, _, last_det_conf = det

            if not _require_centered_for_approach(
                du, dv, keep_alive=keep_alive, phase="stepped approach"
            ):
                return False

            if abs(z_cam - dist_target) <= tol_z:
                self.get_logger().info(
                    f"yolo_visual_center: depth converged (~{dist_target:.2f} m ± {tol_z:.3f}), conf={last_det_conf:.3f}"
                )
                return self._yolo_verify_centered_or_fail(
                    cfg,
                    det_track,
                    model,
                    target_class=target_class,
                    min_conf=min_conf,
                    yolo_iou=yolo_iou,
                    roi=roi,
                    ray_kw=ray_kw,
                    tol_px=tol_px,
                    phase_tag="after stepped approach",
                )

            step = float(z_cam - dist_target)
            if min_det_z > 0.0 and z_cam < min_det_z:
                retreat = min_det_z - z_cam
                if step < retreat:
                    self.get_logger().info(
                        f"yolo_visual_center: depth {z_cam:.3f} m < min_detection {min_det_z:.3f} m — "
                        f"backing off {retreat:.3f} m before further approach."
                    )
                    step = retreat
            if single_motion:
                step = max(-single_trans_cap, min(single_trans_cap, step))
            elif abs(step) > approach_clip:
                step = math.copysign(approach_clip, step)

            pose_fwd = self._tool_pose_translate_along_optical_forward(
                step,
                optical_frame=optical,
                base_frame=base,
                tool_frame=tool_frame,
                tf_timeout_sec=tf_timeout,
            )
            if pose_fwd is None:
                return False

            ka_tag = " [keep-alive]" if keep_alive else ""
            self.get_logger().info(
                f"yolo_visual_center approach {ap + 1}/{ap_cap}{ka_tag}: "
                f"ΔZ_optical_step={step:.3f} m, depth={z_cam:.3f}"
            )
            if not self._move_cartesian_waypoint_then_ik(action, pose_fwd):
                return False
            ap += 1

        self.get_logger().error(
            f"yolo_visual_center: exceeded {ap_cap} approach iteration(s) without reaching depth "
            f"{dist_target:.2f} m ± {tol_z:.3f}."
        )
        return False

    def run(self) -> bool:
        self.get_logger().info(
            f"Starting sequence '{self.sequence_name}' from {self.config_file} "
            f"with {len(self.sequence)} actions."
        )
        all_ok = True
        for index, action in enumerate(self.sequence, start=1):
            self._raise_if_abort()
            name = str(action.get("name", f"action_{index}"))
            action_type = str(action.get("type", "")).lower()
            self.get_logger().info(f"[{index}/{len(self.sequence)}] Starting {name} ({action_type})")
            try:
                self._environment_before_action(action)
                ok = self._execute_action(action)
            except SequenceInterrupted:
                self.get_logger().warn(
                    f"Sequence stopped during '{name}' (Space, Ctrl+C, or stop_sequence service)."
                )
                return False
            except KeyboardInterrupt:
                self._abort_requested = True
                self.get_logger().warn(f"Sequence stopped during '{name}' (Ctrl+C).")
                return False
            except Exception as exc:  # noqa: BLE001 - report config/runtime failures cleanly.
                self.get_logger().error(f"Action '{name}' failed with exception: {exc}")
                ok = False

            if ok:
                self.get_logger().info(f"Action '{name}' complete.")
                self._environment_after_action(action, success=True)
                self._delay_after_action(action)
                self._publish_action_completed(name)
                continue

            self._environment_after_action(action, success=False)
            all_ok = False
            self.get_logger().error(f"Action '{name}' failed.")
            if not self.continue_on_failure:
                self.get_logger().error("Stopping sequence because continue_on_failure is false.")
                return False

        if all_ok:
            self.get_logger().info(f"Sequence '{self.sequence_name}' complete.")
        return all_ok

    def _select_sequence_names(self, config: Dict[str, Any], requested_sequence: str) -> list[str]:
        if self.prompt_for_object:
            object_name, sequence_names = self._prompt_for_object_sequence(config, self.object_name)
            self.get_logger().info(f"Selected sequence path '{' -> '.join(sequence_names)}' for object '{object_name}'.")
            return sequence_names

        if self.object_name:
            sequence_names = self._sequences_from_object(config, self.object_name)
            self.get_logger().info(f"Selected sequence path '{' -> '.join(sequence_names)}' for object '{self.object_name}'.")
            return sequence_names

        return [requested_sequence]

    def _prompt_for_object_sequence(self, config: Dict[str, Any], first_object_name: str = "") -> tuple[str, list[str]]:
        object_name = first_object_name.strip()
        while True:
            if not object_name:
                object_name = input("Target Object: ").strip()
            try:
                return object_name, self._sequences_from_object(config, object_name)
            except ValueError as exc:
                self.get_logger().warn(str(exc))
                object_name = ""

    @staticmethod
    def _sequences_from_object(config: Dict[str, Any], object_name: str) -> list[str]:
        object_sequences = config.get("object_sequences", {})
        if not isinstance(object_sequences, dict):
            raise ValueError("object_sequences must be a map from object name to sequence name(s)")
        try:
            sequence_names = object_sequences[object_name]
        except KeyError as exc:
            valid = ", ".join(sorted(str(name) for name in object_sequences)) or "<none>"
            raise ValueError(f"Unknown object '{object_name}'. Valid objects: {valid}") from exc
        if isinstance(sequence_names, str):
            return [sequence_names]
        if isinstance(sequence_names, list) and sequence_names:
            return [str(name) for name in sequence_names]
        raise ValueError(f"object_sequences entry for '{object_name}' must be a sequence name or non-empty list")

    @staticmethod
    def _load_named_preset_map(config: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
        raw = config.get(key)
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config key {key!r} must be a map of preset name -> fields")
        out: Dict[str, Dict[str, Any]] = {}
        for pname, val in raw.items():
            if not isinstance(val, dict):
                raise ValueError(f"Preset {key}/{pname!r} must be a map")
            out[str(pname)] = dict(val)
        return out

    def _environment_before_action(self, action: Dict[str, Any]) -> None:
        """
        Optional per-action MoveIt scene updates (independent of MuJoCo).
        Keys: detach_preset, attach_preset, keepout_preset (alias forbidden_zone).
        """
        settle = float(
            action.get(
                "planning_scene_settle_sec",
                self.motion_cfg.get("planning_scene_settle_sec", 0.12),
            )
        )
        dp = action.get("detach_preset")
        if dp:
            self._planning_scene_detach_from_preset(str(dp), settle_sec=settle)
        ap = action.get("attach_preset")
        if ap:
            self._planning_scene_attach_from_preset(str(ap), settle_sec=settle)
        kz = action.get("keepout_preset") or action.get("forbidden_zone")
        if kz:
            self._planning_scene_add_keepout_from_preset(str(kz), action, settle_sec=settle)

    def _environment_after_action(self, action: Dict[str, Any], *, success: bool) -> None:
        if not success:
            return
        if not bool(action.get("keepout_remove_after", True)):
            return
        kz = action.get("keepout_preset") or action.get("forbidden_zone")
        if not kz:
            return
        settle = float(
            action.get(
                "planning_scene_settle_sec",
                self.motion_cfg.get("planning_scene_settle_sec", 0.12),
            )
        )
        self._planning_scene_remove_keepout_from_preset(str(kz), settle_sec=settle)

    def _planning_scene_add_keepout_from_preset(
        self,
        preset_name: str,
        action: Dict[str, Any],
        *,
        settle_sec: float,
    ) -> None:
        preset = self._forbidden_zone_presets.get(preset_name)
        if not preset:
            valid = ", ".join(sorted(self._forbidden_zone_presets)) or "<empty>"
            raise ValueError(f"Unknown keepout_preset/forbidden_zone {preset_name!r}. Defined: {valid}")
        merged = dict(preset)
        for k in (
            "object_id",
            "frame_id",
            "x_min",
            "x_max",
            "y_min",
            "y_max",
            "z_min",
            "z_max",
        ):
            if k in action:
                merged[k] = action[k]
        if not self._planning_scene_add_keepout(merged, settle_sec=settle_sec):
            raise RuntimeError(f"keepout preset {preset_name!r}: add_keepout failed")

    def _planning_scene_remove_keepout_from_preset(self, preset_name: str, *, settle_sec: float) -> None:
        preset = self._forbidden_zone_presets.get(preset_name)
        if not preset:
            return
        oid = str(preset.get("object_id", "keepout_volume"))
        remove_action = {"object_id": oid}
        if not self._planning_scene_remove_keepout(remove_action, settle_sec=settle_sec):
            raise RuntimeError(f"keepout preset {preset_name!r}: remove_keepout failed")

    def _planning_scene_attach_from_preset(self, preset_name: str, *, settle_sec: float) -> None:
        preset = self._planning_attach_presets.get(preset_name)
        if not preset:
            valid = ", ".join(sorted(self._planning_attach_presets)) or "<empty>"
            raise ValueError(f"Unknown attach_preset {preset_name!r}. Defined: {valid}")
        if not self._planning_scene_attach_objects(dict(preset), settle_sec=settle_sec):
            raise RuntimeError(f"attach preset {preset_name!r} failed")

    def _planning_scene_detach_from_preset(self, preset_name: str, *, settle_sec: float) -> None:
        preset = self._planning_detach_presets.get(preset_name)
        if not preset:
            valid = ", ".join(sorted(self._planning_detach_presets)) or "<empty>"
            raise ValueError(f"Unknown detach_preset {preset_name!r}. Defined: {valid}")
        if not self._planning_scene_detach_objects(dict(preset), settle_sec=settle_sec):
            raise RuntimeError(f"detach preset {preset_name!r} failed")

    def _delay_after_action(self, action: Dict[str, Any]) -> None:
        delay_sec = float(action.get("delay_after_sec", action.get("delay_sec", 0.0)))
        if delay_sec < 0.0:
            raise ValueError(f"delay_after_sec must be >= 0.0, got {delay_sec}")
        if delay_sec <= 0.0:
            return
        self.get_logger().info(f"Waiting {delay_sec:.3f} seconds before next action.")
        self._sleep_interruptible(delay_sec)

    def _execute_action(self, action: Dict[str, Any]) -> bool:
        action_type = str(action.get("type", "")).lower()
        if action_type == "joint":
            return self._execute_joint_action(action)
        if action_type == "cartesian":
            return self._execute_cartesian_action(action)
        if action_type == "waypoint":
            return self._execute_waypoint_action(action)
        if action_type == "weld":
            return self._execute_weld_action(action)
        if action_type == "planning_scene":
            return self._execute_planning_scene_action(action)
        if action_type == "trigger":
            return self._execute_trigger_action(action)
        if action_type in ("yolo_visual_center", "visual_center_yolo", "yolo_visual_lookat_and_approach"):
            return self._execute_yolo_visual_center_action(action)
        raise ValueError(f"Unsupported action type: {action_type!r}")

    def _execute_trigger_action(self, action: Dict[str, Any]) -> bool:
        """Call std_srvs/Trigger once (e.g. MuJoCo plate table reset before welding)."""
        service_name = str(action.get("service_name", "")).strip()
        if not service_name:
            raise ValueError("trigger action requires non-empty service_name")
        wait_timeout = float(action.get("wait_timeout_sec", 10.0))
        call_timeout = float(action.get("call_timeout_sec", 15.0))

        client = self._trigger_service_client(service_name)
        if not self._wait_for_service_abortable(client, wait_timeout):
            self.get_logger().error(f"Trigger service not available: {service_name}")
            return False
        future = client.call_async(Trigger.Request())
        self._spin_until_future_complete_abortable(future, call_timeout)
        if not future.done():
            self.get_logger().error(f"Trigger service timed out: {service_name}")
            return False
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Trigger call failed ({service_name}): {exc}")
            return False
        if result is None or not result.success:
            detail = getattr(result, "message", "") if result is not None else ""
            self.get_logger().error(f"Trigger unsuccessful ({service_name}): {detail}")
            return False
        self.get_logger().info(f"Trigger OK ({service_name}): {getattr(result, 'message', '')}")
        return True

    def _weld_pose_publisher(self, topic: str) -> Any:
        if topic not in self._weld_pose_publishers:
            self._weld_pose_publishers[topic] = self.create_publisher(PoseStamped, topic, 10)
        return self._weld_pose_publishers[topic]

    def _trigger_service_client(self, service_name: str) -> Any:
        if service_name not in self._trigger_clients:
            self._trigger_clients[service_name] = self.create_client(Trigger, service_name)
        return self._trigger_clients[service_name]

    def _execute_weld_action(self, action_in: Dict[str, Any]) -> bool:
        """Publish a Case_Base-frame plate pose then call MuJoCo attach service (equality weld switch)."""
        action = dict(action_in)
        wkey = action.pop("mujoco_weld_preset", None)
        if wkey is not None:
            wname = str(wkey).strip()
            if not wname:
                raise ValueError("mujoco_weld_preset must be non-empty when set")
            base = dict(self._mujoco_weld_presets.get(wname, {}))
            if not base:
                valid = ", ".join(sorted(self._mujoco_weld_presets)) or "<empty>"
                raise ValueError(f"Unknown mujoco_weld_preset {wname!r}. Defined: {valid}")
            action = {**base, **action}

        topic = str(action.get("target_pose_topic", "/mujoco/weld/module_1_plate_case_offset"))
        service_name = str(action.get("service_name", "/mujoco/weld/module_1_plate_attach_case_base"))
        frame_id = str(action.get("target_frame", "Case_Base"))
        settle_sec = float(action.get("pose_settle_sec", 0.05))

        self.get_logger().info(
            "Weld step: switches MuJoCo constraints only — the arm receives no trajectory here, "
            "so visually it stays still until the next MoveIt motion."
        )

        xyz = action.get("target_xyz", [0.0, 0.0, 0.0])
        if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
            raise ValueError("weld action requires target_xyz as [x, y, z] in Case_Base frame")
        quat = action.get("target_quat_xyzw", [0.0, 0.0, 0.0, 1.0])
        if not isinstance(quat, (list, tuple)) or len(quat) != 4:
            raise ValueError("weld action requires target_quat_xyzw as [x, y, z, w]")

        msg = PoseStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x = float(xyz[0])
        msg.pose.position.y = float(xyz[1])
        msg.pose.position.z = float(xyz[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        pub = self._weld_pose_publisher(topic)
        pub.publish(msg)
        self.get_logger().info(
            f"Published weld target on {topic} (frame={frame_id!r}); waiting {settle_sec:.3f}s before service call."
        )
        if settle_sec > 0.0:
            self._sleep_interruptible(settle_sec)

        client = self._trigger_service_client(service_name)
        if not self._wait_for_service_abortable(client, 10.0):
            dom = ""
            try:
                import os

                dom = os.environ.get("ROS_DOMAIN_ID", "")
            except OSError:
                pass
            self.get_logger().error(
                f"Weld service not available: {service_name}. "
                f"If MuJoCo is running, ROS_DOMAIN_ID must match everywhere (your env: "
                f"ROS_DOMAIN_ID={dom or 'unset (defaults to 0)'}). "
                f"Launch and sequencer should both source the same workspace setup, e.g. "
                f"~/ur3_control/source_ws.sh. Check: ros2 service list | grep module_1_plate"
            )
            return False
        future = client.call_async(Trigger.Request())
        self._spin_until_future_complete_abortable(future, 15.0)
        if not future.done():
            self.get_logger().error(f"Weld service call timed out: {service_name}")
            return False
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Weld service call failed: {exc}")
            return False
        if result is None or not result.success:
            detail = getattr(result, "message", "") if result is not None else ""
            self.get_logger().error(f"Weld attach failed: {detail}")
            return False
        self.get_logger().info(f"Weld attach OK: {getattr(result, 'message', '')}")
        return True

    @staticmethod
    def _pose_from_xyz_rpy(xyz: Iterable[float], rpy: Iterable[float]) -> Pose:
        """Pose in parent frame; RPY intrinsic XYZ."""
        x, y, z = (float(v) for v in xyz)
        roll, pitch, yaw = (float(v) for v in rpy)
        q = _quat_from_rpy(roll, pitch, yaw)
        p = Pose()
        p.position.x, p.position.y, p.position.z = x, y, z
        p.orientation.x = float(q[0])
        p.orientation.y = float(q[1])
        p.orientation.z = float(q[2])
        p.orientation.w = float(q[3])
        return p

    def _publish_planning_scene_diff(self, scene: PlanningScene, *, settle_sec: float) -> None:
        scene.is_diff = True
        self._planning_scene_pub.publish(scene)
        for _ in range(5):
            rclpy.spin_once(self, timeout_sec=0.02)
        if settle_sec > 0.0:
            self._sleep_interruptible(settle_sec)

    def _execute_planning_scene_action(self, action_in: Dict[str, Any]) -> bool:
        """Update MoveIt planning scene: axis-aligned keepout box and/or tool0-attached collision bodies."""
        action = dict(action_in)
        preset_name = action.pop("preset", None)
        if preset_name is not None:
            pname = str(preset_name).strip()
            if not pname:
                raise ValueError("planning_scene preset must be non-empty when set")
            op = str(action.get("operation", action.get("command", ""))).lower().strip()
            if op in ("add_keepout", "keepout_add") or op in ("remove_keepout", "keepout_remove"):
                pmap = self._forbidden_zone_presets
            elif op == "attach":
                pmap = self._planning_attach_presets
            elif op == "detach":
                pmap = self._planning_detach_presets
            else:
                raise ValueError(
                    "planning_scene with preset= requires operation: "
                    "add_keepout | remove_keepout | attach | detach"
                )
            base = dict(pmap.get(pname, {}))
            if not base:
                valid = ", ".join(sorted(pmap)) or "<empty>"
                raise ValueError(f"Unknown planning_scene preset {pname!r} for {op!r}. Defined: {valid}")
            action = {**base, **action}

        op = str(action.get("operation", action.get("command", ""))).lower().strip()
        settle_sec = float(action.get("settle_sec", self.motion_cfg.get("planning_scene_settle_sec", 0.12)))
        if op in ("add_keepout", "keepout_add"):
            return self._planning_scene_add_keepout(action, settle_sec)
        if op in ("remove_keepout", "keepout_remove"):
            return self._planning_scene_remove_keepout(action, settle_sec)
        if op == "attach":
            return self._planning_scene_attach_objects(action, settle_sec)
        if op == "detach":
            return self._planning_scene_detach_objects(action, settle_sec)
        raise ValueError(
            "planning_scene requires operation: add_keepout | remove_keepout | attach | detach "
            f"(got {op!r})"
        )

    def _planning_scene_add_keepout(self, action: Dict[str, Any], settle_sec: float) -> bool:
        required = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
        for key in required:
            if key not in action:
                raise ValueError(f"add_keepout requires {key} in base/frame coordinates")
        xm, xM = float(action["x_min"]), float(action["x_max"])
        ym, yM = float(action["y_min"]), float(action["y_max"])
        zm, zM = float(action["z_min"]), float(action["z_max"])
        if xm >= xM or ym >= yM or zm >= zM:
            raise ValueError("add_keepout requires x_min < x_max (and same for y, z)")
        dx, dy, dz = xM - xm, yM - ym, zM - zm
        cx = 0.5 * (xm + xM)
        cy = 0.5 * (ym + yM)
        cz = 0.5 * (zm + zM)

        frame_id = str(action.get("frame_id", self.motion_cfg["base_frame"]))
        object_id = str(action.get("object_id", "keepout_volume"))
        co = CollisionObject()
        co.header.frame_id = frame_id
        co.header.stamp = self.get_clock().now().to_msg()
        co.id = object_id
        co.operation = CollisionObject.ADD
        co.pose = self._pose_from_xyz_rpy((cx, cy, cz), (0.0, 0.0, 0.0))
        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [dx, dy, dz]
        co.primitives = [prim]
        ident = Pose()
        ident.orientation.w = 1.0
        co.primitive_poses = [ident]

        ps = PlanningScene()
        ps.world.collision_objects = [co]
        self._publish_planning_scene_diff(ps, settle_sec=settle_sec)
        self.get_logger().info(
            f"Planning scene: ADD keepout '{object_id}' box in {frame_id!r} "
            f"center=({cx:.3f},{cy:.3f},{cz:.3f}) size=({dx:.3f},{dy:.3f},{dz:.3f})"
        )
        return True

    def _planning_scene_remove_keepout(self, action: Dict[str, Any], settle_sec: float) -> bool:
        object_id = str(action.get("object_id", "keepout_volume")).strip()
        if not object_id:
            raise ValueError("remove_keepout requires object_id")
        co = CollisionObject()
        co.id = object_id
        co.operation = CollisionObject.REMOVE
        ps = PlanningScene()
        ps.world.collision_objects = [co]
        self._publish_planning_scene_diff(ps, settle_sec=settle_sec)
        self.get_logger().info(f"Planning scene: REMOVE keepout '{object_id}'")
        return True

    def _planning_scene_attach_objects(self, action: Dict[str, Any], settle_sec: float) -> bool:
        """Attach collision primitives (box or cylinder) to a robot link (default tool0); poses in link frame."""
        link_name = str(action.get("link_name", str(self.motion_cfg.get("ik_link_name", "tool0"))))
        raw_objects = action.get("objects")
        if not isinstance(raw_objects, list) or not raw_objects:
            raise ValueError(
                "attach requires objects with id, dimensions, pose_xyz (optional pose_rpy). "
                "shape box (default): dimensions [dx,dy,dz]. "
                "shape cylinder: dimensions [height, radius] with axis along +Z of primitive pose."
            )
        touch_links = action.get("touch_links")
        touch_out: list[str] = []
        if isinstance(touch_links, list):
            touch_out = [str(x) for x in touch_links]

        attached: list[AttachedCollisionObject] = []
        for index, obj in enumerate(raw_objects):
            if not isinstance(obj, dict):
                raise ValueError(f"attach.objects[{index}] must be a map")
            oid = str(obj.get("id", "")).strip()
            if not oid:
                raise ValueError(f"attach.objects[{index}] requires non-empty id")
            shape = str(obj.get("shape", "box")).lower().strip()
            dims_any = obj.get("dimensions", obj.get("box_dimensions"))
            if not isinstance(dims_any, (list, tuple)) or len(dims_any) < 2:
                raise ValueError(f"attach.objects[{index}] requires dimensions (length depends on shape)")

            prim = SolidPrimitive()
            if shape in ("box", ""):
                if len(dims_any) != 3:
                    raise ValueError(f"attach.objects[{index}] box requires dimensions: [dx, dy, dz] in meters")
                prim.type = SolidPrimitive.BOX
                prim.dimensions = [float(dims_any[0]), float(dims_any[1]), float(dims_any[2])]
            elif shape == "cylinder":
                if len(dims_any) != 2:
                    raise ValueError(
                        f"attach.objects[{index}] cylinder requires dimensions: [height, radius] in meters"
                    )
                height, radius = float(dims_any[0]), float(dims_any[1])
                if height <= 0.0 or radius <= 0.0:
                    raise ValueError("cylinder height and radius must be positive")
                prim.type = SolidPrimitive.CYLINDER
                prim.dimensions = [height, radius]
            else:
                raise ValueError(f"attach.objects[{index}] unsupported shape {shape!r} (use 'box' or 'cylinder')")

            pxyz = obj.get("pose_xyz", [0.0, 0.0, 0.0])
            prpy = obj.get("pose_rpy", [0.0, 0.0, 0.0])
            if not isinstance(pxyz, (list, tuple)) or len(pxyz) != 3:
                raise ValueError(f"attach.objects[{index}] pose_xyz must be length-3")
            if not isinstance(prpy, (list, tuple)) or len(prpy) != 3:
                raise ValueError(f"attach.objects[{index}] pose_rpy must be length-3")

            aco = AttachedCollisionObject()
            aco.link_name = link_name
            if touch_out:
                aco.touch_links = touch_out
            aco.object.header.frame_id = link_name
            aco.object.header.stamp = self.get_clock().now().to_msg()
            aco.object.id = oid
            aco.object.operation = CollisionObject.ADD
            aco.object.pose = self._pose_from_xyz_rpy(pxyz, prpy)
            aco.object.primitives = [prim]
            ip = Pose()
            ip.orientation.w = 1.0
            aco.object.primitive_poses = [ip]
            attached.append(aco)

        rs = RobotState()
        rs.is_diff = True
        rs.attached_collision_objects = attached
        ps = PlanningScene()
        ps.robot_state = rs
        self._publish_planning_scene_diff(ps, settle_sec=settle_sec)
        self.get_logger().info(
            f"Planning scene: ATTACH {len(attached)} object(s) to link {link_name!r}: "
            f"{', '.join(a.object.id for a in attached)}"
        )
        return True

    def _planning_scene_detach_objects(self, action: Dict[str, Any], settle_sec: float) -> bool:
        link_name = str(action.get("link_name", str(self.motion_cfg.get("ik_link_name", "tool0"))))
        ids_any = action.get("object_ids")
        if not isinstance(ids_any, list) or not ids_any:
            single = action.get("object_id")
            if single:
                ids_any = [single]
            else:
                raise ValueError("detach requires object_ids: [ ... ] or object_id")

        remove_list: list[AttachedCollisionObject] = []
        for oid_raw in ids_any:
            oid = str(oid_raw).strip()
            if not oid:
                continue
            aco = AttachedCollisionObject()
            aco.link_name = link_name
            aco.object.id = oid
            aco.object.operation = CollisionObject.REMOVE
            remove_list.append(aco)

        rs = RobotState()
        rs.is_diff = True
        rs.attached_collision_objects = remove_list
        ps = PlanningScene()
        ps.robot_state = rs
        self._publish_planning_scene_diff(ps, settle_sec=settle_sec)
        self.get_logger().info(
            f"Planning scene: DETACH from {link_name!r}: {', '.join(str(o.object.id) for o in remove_list)}"
        )
        return True

    def _execute_joint_action(self, action: Dict[str, Any]) -> bool:
        positions = self._joint_target(action.get("target"))
        return self._execute_joint_goal(action, positions)

    def _execute_cartesian_action(self, action: Dict[str, Any]) -> bool:
        pose = self._tool_pose_from_tcp_action(action)
        mode = str(action.get("cartesian_mode", self.motion_cfg["default_cartesian_mode"])).lower()
        if mode in ("linear", "straight", "cartesian_path"):
            min_fraction = float(action.get("min_fraction", self.motion_cfg["default_min_fraction"]))
            cartesian_result = self._execute_cartesian_path(action, [pose], min_fraction)
            if cartesian_result is not None:
                return cartesian_result
            self.get_logger().warn("Cartesian path planning was incomplete; falling back to IK joint-goal planning.")
        elif mode not in ("moveit", "joint_goal", "planned"):
            raise ValueError(f"Unsupported cartesian_mode: {mode!r}")

        joint_positions = self._solve_ik(pose, action)
        if joint_positions is None:
            return False
        return self._execute_joint_goal(action, joint_positions)

    def _execute_waypoint_action(self, action: Dict[str, Any]) -> bool:
        waypoints = action.get("waypoints")
        if not isinstance(waypoints, list) or not waypoints:
            raise ValueError("waypoint action requires a non-empty waypoints list")

        blend_radius = float(action.get("blending_radius", 0.0))
        if blend_radius > 0.0 and self._execute_sequence_waypoints(action, waypoints, blend_radius):
            return True

        if not self._waypoints_are_all_cartesian(waypoints):
            self.get_logger().warn(
                "Mixed joint/Cartesian waypoints require MoveGroupSequence for blending; "
                "executing each waypoint sequentially without blending."
            )
            return self._execute_waypoints_sequentially(action, waypoints)

        poses = [self._tool_pose_from_tcp_action(wp) for wp in waypoints]
        min_fraction = float(action.get("min_fraction", self.motion_cfg["default_min_fraction"]))
        return self._execute_cartesian_path(action, poses, min_fraction)

    def _execute_sequence_waypoints(
        self,
        action: Dict[str, Any],
        waypoints: list[Dict[str, Any]],
        blend_radius: float,
    ) -> bool:
        if not self._wait_for_action_server_abortable(self._sequence_client, 1.0):
            self.get_logger().warn("MoveGroupSequence action is not available; using Cartesian path fallback.")
            return False

        items = []
        for index, waypoint in enumerate(waypoints):
            waypoint_action = self._waypoint_action(action, waypoint, index)
            joint_positions = self._waypoint_joint_positions(waypoint_action)
            if joint_positions is None:
                return False
            item = MotionSequenceItem()
            item.req = self._build_joint_motion_request(waypoint_action, joint_positions)
            item.blend_radius = blend_radius if index < len(waypoints) - 1 else 0.0
            items.append(item)

        goal = MoveGroupSequence.Goal()
        goal.request.items = items
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = int(action.get("replan_attempts", 1))

        timeout = self._planning_time(action) * max(len(items), 1) + 30.0
        self.get_logger().info(f"Sending blended sequence with radius {blend_radius:.3f} m.")
        send_future = self._sequence_client.send_goal_async(goal)
        self._spin_until_future_complete_abortable(send_future, 10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveGroupSequence goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        self._spin_until_future_complete_abortable(result_future, timeout, goal_handle)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("MoveGroupSequence planning/execution timed out.")
            return False

        result = result_future.result().result
        code = result.response.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"MoveGroupSequence failed: error_code={code}")
            return False
        return True

    def _execute_waypoints_sequentially(self, action: Dict[str, Any], waypoints: list[Dict[str, Any]]) -> bool:
        for index, waypoint in enumerate(waypoints):
            self._raise_if_abort()
            waypoint_action = self._waypoint_action(action, waypoint, index)
            joint_positions = self._waypoint_joint_positions(waypoint_action)
            if joint_positions is None:
                return False
            if not self._execute_joint_goal(waypoint_action, joint_positions):
                return False
        return True

    def _execute_joint_goal(self, action: Dict[str, Any], positions: list[float]) -> bool:
        if not self._wait_for_joint_state(action):
            return False
        if not self._wait_for_action_server_abortable(self._move_group_client, 10.0):
            self.get_logger().error(f"MoveGroup action is not available: {self.motion_cfg['move_group_action']}")
            return False

        goal = MoveGroup.Goal()
        goal.request = self._build_joint_motion_request(action, positions)
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = int(action.get("replan_attempts", 1))

        self.get_logger().info("Sending joint goal to MoveIt.")
        send_future = self._move_group_client.send_goal_async(goal)
        self._spin_until_future_complete_abortable(send_future, 10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveIt goal rejected.")
            return False

        timeout = self._planning_time(action) + 30.0
        result_future = goal_handle.get_result_async()
        self._spin_until_future_complete_abortable(result_future, timeout, goal_handle)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("MoveIt planning/execution timed out.")
            return False

        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"MoveIt failed: error_code={result.error_code.val}")
            return False
        return True

    def _execute_cartesian_path(self, action: Dict[str, Any], waypoints: list[Pose], min_fraction: float) -> Optional[bool]:
        if not self._wait_for_joint_state(action):
            return False
        if not self._wait_for_service_abortable(self._cartesian_client, 5.0):
            self.get_logger().error(
                f"Cartesian path service is not available: {self.motion_cfg['cartesian_path_service']}"
            )
            return None

        req = GetCartesianPath.Request()
        req.header.frame_id = str(self.motion_cfg["base_frame"])
        req.start_state.joint_state = self._latest_joint_state if self._latest_joint_state is not None else JointState()
        req.start_state.is_diff = False
        req.group_name = str(self.motion_cfg["planning_group"])
        req.link_name = str(self.motion_cfg["ik_link_name"])
        req.waypoints = waypoints
        req.max_step = float(action.get("max_step_m", self.motion_cfg["default_max_step_m"]))
        req.jump_threshold = float(action.get("jump_threshold", self.motion_cfg["default_jump_threshold"]))
        req.avoid_collisions = bool(action.get("avoid_collisions", self.motion_cfg["avoid_collisions"]))

        self.get_logger().info(f"Computing Cartesian path through {len(waypoints)} waypoint(s).")
        future = self._cartesian_client.call_async(req)
        self._spin_until_future_complete_abortable(future, self._planning_time(action) + 5.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Cartesian path service timed out or returned no result.")
            return None

        result = future.result()
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"Cartesian path failed: error_code={result.error_code.val}")
            return None
        if float(result.fraction) < min_fraction:
            self.get_logger().error(
                f"Cartesian path fraction {result.fraction:.3f} is below required {min_fraction:.3f}."
            )
            return None

        trajectory = result.solution
        self._retime_trajectory(trajectory, self._velocity(action))
        return self._execute_trajectory(trajectory, self._execution_timeout(action, trajectory))

    def _execute_trajectory(self, trajectory: RobotTrajectory, timeout: float) -> bool:
        if self._wait_for_action_server_abortable(self._execute_trajectory_client, 2.0):
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = trajectory
            send_future = self._execute_trajectory_client.send_goal_async(goal)
            self._spin_until_future_complete_abortable(send_future, 10.0)
            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                self.get_logger().error("ExecuteTrajectory goal rejected.")
                return False

            result_future = goal_handle.get_result_async()
            self._spin_until_future_complete_abortable(result_future, timeout, goal_handle)
            if not result_future.done() or result_future.result() is None:
                self.get_logger().error("ExecuteTrajectory timed out.")
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
                return False
            result = result_future.result().result
            if result.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(f"ExecuteTrajectory failed: error_code={result.error_code.val}")
                return False
            return True

        self.get_logger().warn("ExecuteTrajectory action is unavailable; using joint trajectory controller fallback.")
        return self._execute_joint_trajectory(trajectory.joint_trajectory, timeout)

    def _execute_joint_trajectory(self, trajectory: JointTrajectory, timeout: float) -> bool:
        if not self._wait_for_action_server_abortable(self._trajectory_client, 5.0):
            self.get_logger().error(f"Trajectory action is not available: {self.motion_cfg['trajectory_action']}")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._trajectory_client.send_goal_async(goal)
        self._spin_until_future_complete_abortable(send_future, 10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        self._spin_until_future_complete_abortable(result_future, timeout, goal_handle)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("Trajectory execution timed out.")
            cancel_future = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, cancel_future, timeout_sec=2.0)
            return False
        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(
                f"Trajectory execution failed: error_code={result.error_code}, message='{result.error_string}'"
            )
            return False
        return True

    def _solve_ik(
        self,
        tool_pose: Pose,
        action: Dict[str, Any],
        *,
        silent_ik_failure: bool = False,
    ) -> Optional[list[float]]:
        if not self._wait_for_joint_state(action):
            return None
        if not self._wait_for_service_abortable(self._ik_client, 10.0):
            self.get_logger().error("MoveIt IK service /compute_ik is not available.")
            return None

        req = GetPositionIK.Request()
        req.ik_request.group_name = str(self.motion_cfg["planning_group"])
        req.ik_request.ik_link_name = str(self.motion_cfg["ik_link_name"])
        req.ik_request.pose_stamped.header.frame_id = str(self.motion_cfg["base_frame"])
        req.ik_request.pose_stamped.pose = tool_pose
        req.ik_request.robot_state.joint_state = self._latest_joint_state if self._latest_joint_state is not None else JointState()
        req.ik_request.robot_state.is_diff = False
        req.ik_request.avoid_collisions = bool(action.get("avoid_collisions", self.motion_cfg["avoid_collisions"]))
        ik_timeout = float(action.get("ik_timeout_sec", self.motion_cfg["default_ik_timeout_sec"]))
        req.ik_request.timeout = _duration_from_seconds(ik_timeout)

        future = self._ik_client.call_async(req)
        self._spin_until_future_complete_abortable(future, ik_timeout + 3.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("IK service timed out or returned no result.")
            return None

        result = future.result()
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            if not silent_ik_failure:
                label = _moveit_error_label(int(result.error_code.val))
                suffix = f" ({label})" if label else ""
                self.get_logger().error(f"IK failed: error_code={result.error_code.val}{suffix}")
            return None

        joint_solution = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
        try:
            return [float(joint_solution[name]) for name in self.joint_names]
        except KeyError as exc:
            self.get_logger().error(f"IK result did not contain expected joint: {exc}")
            return None

    def _build_joint_motion_request(self, action: Dict[str, Any], positions: Iterable[float]) -> MotionPlanRequest:
        constraints = Constraints()
        constraints.name = str(action.get("name", "action_joint_goal"))
        tol = float(max(float(action.get("joint_tolerance_rad", self.motion_cfg["default_joint_tolerance_rad"])), 1e-4))
        for name, pos in zip(self.joint_names, positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = tol
            jc.tolerance_below = tol
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        req = MotionPlanRequest()
        req.workspace_parameters = self._workspace()
        req.group_name = str(self.motion_cfg["planning_group"])
        if self._latest_joint_state is not None:
            req.start_state.joint_state = self._latest_joint_state
            req.start_state.is_diff = False
        req.goal_constraints = [constraints]
        req.num_planning_attempts = int(action.get("planning_attempts", self.motion_cfg["default_planning_attempts"]))
        req.allowed_planning_time = self._planning_time(action)
        req.max_velocity_scaling_factor = self._velocity(action)
        req.max_acceleration_scaling_factor = self._acceleration(action)
        return req

    def _wait_for_joint_state(self, action: Dict[str, Any]) -> bool:
        timeout_sec = float(action.get("joint_state_timeout_sec", self.motion_cfg["default_joint_state_timeout_sec"]))
        deadline = time.time() + timeout_sec
        while rclpy.ok() and time.time() < deadline:
            self._raise_if_abort()
            if self._has_complete_joint_state():
                return True
            rclpy.spin_once(self, timeout_sec=0.05)
        self.get_logger().error(
            f"Timed out waiting for /joint_states with joints: {', '.join(self.joint_names)}"
        )
        return False

    def _has_complete_joint_state(self) -> bool:
        if self._latest_joint_state is None:
            return False
        names = set(self._latest_joint_state.name)
        return all(name in names for name in self.joint_names)

    def _waypoint_action(self, action: Dict[str, Any], waypoint: Dict[str, Any], index: int) -> Dict[str, Any]:
        if not isinstance(waypoint, dict):
            raise ValueError(f"waypoint {index + 1} must be a map")
        merged = dict(action)
        merged.pop("waypoints", None)
        merged.update(waypoint)
        merged.setdefault("name", f"{action.get('name', 'waypoint')}_{index + 1}")
        return merged

    def _waypoint_joint_positions(self, waypoint: Dict[str, Any]) -> Optional[list[float]]:
        waypoint_type = self._waypoint_type(waypoint)
        if waypoint_type == "joint":
            return self._joint_target(waypoint.get("target"))
        if waypoint_type == "cartesian":
            return self._solve_ik(self._tool_pose_from_tcp_action(waypoint), waypoint)
        raise ValueError(f"Unsupported waypoint type: {waypoint_type!r}")

    def _waypoints_are_all_cartesian(self, waypoints: list[Dict[str, Any]]) -> bool:
        return all(self._waypoint_type(waypoint) == "cartesian" for waypoint in waypoints)

    @staticmethod
    def _waypoint_type(waypoint: Dict[str, Any]) -> str:
        waypoint_type = str(waypoint.get("type", "")).lower()
        if waypoint_type in ("", "cartesian", "tcp", "pose"):
            if "target_xyz" in waypoint:
                return "cartesian"
        if waypoint_type in ("joint", "joints"):
            return "joint"
        if not waypoint_type and "target" in waypoint:
            return "joint"
        raise ValueError(
            "Waypoint requires either type: joint with target, or type: cartesian with target_xyz"
        )

    def _tool_pose_from_tcp_action(self, action: Dict[str, Any]) -> Pose:
        target_xyz = _vector3(action.get("target_xyz"), "target_xyz")
        lookat = _normalize_lookat_vector(
            _vector3(action.get("lookat_vector", [0.0, 0.0, -1.0]), "lookat_vector")
        )
        tcp_roll = float(action.get("tcp_roll_rad", 0.0))

        base_to_tcp = np.eye(4, dtype=np.float64)
        base_to_tcp[:3, :3] = _tcp_rotation_from_lookat(lookat, tcp_roll)
        base_to_tcp[:3, 3] = target_xyz
        base_to_tool = base_to_tcp @ np.linalg.inv(self.tool_to_tcp)
        return _pose_from_transform(base_to_tool)

    def _joint_target(self, target: Any) -> list[float]:
        if not isinstance(target, (list, tuple)) or len(target) != len(self.joint_names):
            raise ValueError(f"joint target must contain {len(self.joint_names)} values")
        return [float(v) for v in target]

    def _retime_trajectory(self, trajectory: RobotTrajectory, velocity_scaling: float) -> None:
        points = trajectory.joint_trajectory.points
        if not points:
            return

        velocity_scaling = max(float(velocity_scaling), 0.01)
        positions = [list(point.positions) for point in points]
        if any(not pos for pos in positions):
            return

        times = [0.0]
        previous = positions[0]
        min_segment_sec = 0.04
        for current in positions[1:]:
            deltas = [abs(float(a) - float(b)) for a, b in zip(current, previous)]
            # Keep Cartesian micro-waypoints moving continuously instead of stopping at every sample.
            times.append(times[-1] + max(max(deltas, default=0.0) / velocity_scaling, min_segment_sec))
            previous = current

        if len(points) == 1:
            points[0].time_from_start = _duration_from_seconds(0.2)
            points[0].velocities = [0.0] * len(positions[0])
            points[0].accelerations = [0.0] * len(positions[0])
            return

        joint_count = len(positions[0])
        for index, point in enumerate(points):
            point.time_from_start = _duration_from_seconds(times[index])
            velocities = []
            for joint_index in range(joint_count):
                if index == 0 or index == len(points) - 1:
                    velocities.append(0.0)
                    continue
                dt = max(times[index + 1] - times[index - 1], 1e-6)
                dq = float(positions[index + 1][joint_index]) - float(positions[index - 1][joint_index])
                velocities.append(dq / dt)
            point.velocities = velocities
            point.accelerations = [0.0] * joint_count

    def _workspace(self) -> WorkspaceParameters:
        workspace = WorkspaceParameters()
        workspace.header.frame_id = str(self.motion_cfg["base_frame"])
        workspace.min_corner.x = -2.0
        workspace.min_corner.y = -2.0
        workspace.min_corner.z = -0.5
        workspace.max_corner.x = 2.0
        workspace.max_corner.y = 2.0
        workspace.max_corner.z = 2.0
        return workspace

    def _planning_time(self, action: Dict[str, Any]) -> float:
        return float(action.get("planning_time_sec", self.motion_cfg["default_planning_time_sec"]))

    def _execution_timeout(self, action: Dict[str, Any], trajectory: RobotTrajectory) -> float:
        configured = action.get("execution_timeout_sec")
        if configured is not None:
            return float(configured)

        duration = self._trajectory_duration_sec(trajectory)
        margin = float(action.get("execution_timeout_margin_sec", self.motion_cfg["default_execution_timeout_margin_sec"]))
        return max(duration + margin, self._planning_time(action) + margin)

    @staticmethod
    def _trajectory_duration_sec(trajectory: RobotTrajectory) -> float:
        points = trajectory.joint_trajectory.points
        if not points:
            return 0.0
        last = points[-1].time_from_start
        return float(last.sec) + float(last.nanosec) * 1e-9

    def _velocity(self, action: Dict[str, Any]) -> float:
        return _scaling(action.get("velocity_scaling"), "velocity_scaling", self.motion_cfg["default_velocity_scaling"])

    def _acceleration(self, action: Dict[str, Any]) -> float:
        return _scaling(
            action.get("acceleration_scaling"),
            "acceleration_scaling",
            self.motion_cfg["default_acceleration_scaling"],
        )

    @staticmethod
    def _load_motion_config(config: Dict[str, Any]) -> Dict[str, Any]:
        motion = config.get("motion")
        if not isinstance(motion, dict):
            raise ValueError("Action config requires a top-level 'motion' map")

        out = dict(motion)
        out.setdefault("planning_group", "ur_manipulator")
        out.setdefault("base_frame", "base_link")
        out.setdefault("tcp_file", DEFAULT_TCP_FILE)
        out.setdefault("ik_link_name", "tool0")
        out.setdefault("move_group_action", "/move_action")
        out.setdefault("sequence_action", "/sequence_move_group")
        out.setdefault("cartesian_path_service", "/compute_cartesian_path")
        out.setdefault("execute_trajectory_action", "/execute_trajectory")
        out.setdefault("trajectory_action", "/joint_trajectory_controller/follow_joint_trajectory")
        out.setdefault("avoid_collisions", True)
        out.setdefault("default_cartesian_mode", "moveit")
        out.setdefault("default_ik_timeout_sec", 2.0)
        out.setdefault("default_planning_time_sec", 5.0)
        out.setdefault("default_planning_attempts", 10)
        out.setdefault("default_joint_state_timeout_sec", 5.0)
        out.setdefault("default_joint_tolerance_rad", 0.01)
        out.setdefault("default_velocity_scaling", 0.15)
        out.setdefault("default_acceleration_scaling", 0.15)
        out.setdefault("default_execution_timeout_margin_sec", 10.0)
        out.setdefault("default_max_step_m", 0.005)
        out.setdefault("default_jump_threshold", 0.0)
        out.setdefault("default_min_fraction", 0.95)
        out.setdefault("planning_scene_topic", "/planning_scene")
        out.setdefault("planning_scene_settle_sec", 0.12)
        return out

    @staticmethod
    def _load_sequence(config: Dict[str, Any], sequence_name: str) -> list[Dict[str, Any]]:
        sequences = config.get("sequences")
        if not isinstance(sequences, dict):
            raise ValueError("Action config requires a top-level 'sequences' map")
        sequence = sequences.get(sequence_name)
        if not isinstance(sequence, list) or not sequence:
            raise ValueError(f"Sequence '{sequence_name}' is missing or empty")
        for index, action in enumerate(sequence, start=1):
            if not isinstance(action, dict):
                raise ValueError(f"Sequence item {index} must be a map")
            if "type" not in action:
                raise ValueError(f"Sequence item {index} is missing 'type'")
        return sequence

    @classmethod
    def _load_sequences(cls, config: Dict[str, Any], sequence_names: list[str]) -> list[Dict[str, Any]]:
        merged: list[Dict[str, Any]] = []
        for sequence_name in sequence_names:
            merged.extend(cls._load_sequence(config, sequence_name))
        return merged


def _stdin_space_stop_listener(node: "MotionManager", shutdown: threading.Event) -> None:
    """TTY cbreak: Space alone (no Enter) sets stop flag; restores line discipline on exit."""
    try:
        import termios
        import tty
    except ImportError:
        return
    fd = sys.stdin.fileno()
    if fd < 0 or not sys.stdin.isatty() or not os.isatty(fd):
        return
    try:
        old_attrs = termios.tcgetattr(fd)
    except termios.error:
        return
    try:
        tty.setcbreak(fd)
        while not shutdown.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            chunk = sys.stdin.read(1)
            if chunk == " ":
                node._abort_requested = True
                print("\nStopping sequence (Space)...", file=sys.stderr, flush=True)
    finally:
        try:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        except termios.error:
            pass


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = MotionManager()

    stdin_shutdown = threading.Event()
    stdin_thread: Optional[threading.Thread] = None
    # Non-daemon: join in finally so tty settings are restored before process exit.
    if sys.stdin.isatty():
        stdin_thread = threading.Thread(
            target=_stdin_space_stop_listener,
            args=(node, stdin_shutdown),
            name="stdin-space-stop",
            daemon=False,
        )
        stdin_thread.start()

    def _on_sigint(_signum: int, _frame: Any) -> None:
        node._abort_requested = True
        print("\nStopping sequence (Ctrl+C)...", file=sys.stderr, flush=True)

    try:
        signal.siginterrupt(signal.SIGINT, True)
    except (AttributeError, ValueError):
        pass

    prev_sigint = signal.signal(signal.SIGINT, _on_sigint)
    exit_code = 0
    try:
        ok = node.run()
        if not ok:
            exit_code = 1
    except KeyboardInterrupt:
        node.get_logger().warn("KeyboardInterrupt — stopping sequence.")
        exit_code = 1
    finally:
        stdin_shutdown.set()
        if stdin_thread is not None:
            stdin_thread.join(timeout=2.0)
        signal.signal(signal.SIGINT, prev_sigint)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if exit_code != 0:
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv)
