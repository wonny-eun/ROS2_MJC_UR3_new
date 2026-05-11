#!/usr/bin/env python3
"""Config-driven action sequencer for UR3 MoveIt motions."""

from __future__ import annotations

import math
import os
import select
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import cv2
import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.action import ExecuteTrajectory, MoveGroup, MoveGroupSequence
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MotionSequenceItem,
    MoveItErrorCodes,
    PlanningOptions,
    RobotTrajectory,
    WorkspaceParameters,
)
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
        "center_tolerance_px": 35.0,
        "distance_tolerance_m": 0.025,
        "step_gain": 0.45,
        "depth_roi_half_px": 5,
        "yolo_iou": 0.5,
        "yolo_model_path": "",
        "wait_image_timeout_sec": 8.0,
        "default_target_class": "",
        # yolo_visual_center: wrist look-at phase then Cartesian approach along optical Z.
        "lookat_max_iterations": None,  # None → use max_iterations
        "approach_max_iterations": 8,
        "approach_step_max_m": 0.12,
        # Roll about view axis (+Z): match your RL camera vs tool0 mounting if needed (see Cartesian tcp_roll_rad).
        "camera_lookat_tcp_roll_rad": 0.0,
    }
    vis = config.get("vision")
    if isinstance(vis, dict):
        defaults.update(vis)
    return defaults


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
    ) -> Optional[tuple[int, int, float, float, float, float, float, float]]:
        """
        Run YOLO and return bbox image center back-projected in camera optical coords:
        u, v, x_cam, y_cam, z, du_to_image_center_px, dv_to_image_center_px, det_confidence.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = model.predict(rgb, conf=min_conf, iou=yolo_iou, verbose=False)[0]
        best_conf = -1.0
        best_box = None
        names = results.names
        boxes = results.boxes
        if boxes is not None and boxes.cls is not None:
            for i in range(len(boxes)):
                cid = int(boxes.cls[i].item())
                cf = float(boxes.conf[i].item())
                cname = str(names[cid])
                if cname != target_class:
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
        cx_img = 0.5 * float(w)
        cy_img = 0.5 * float(h)

        patch = depth[max(0, v - roi) : min(h, v + roi + 1), max(0, u - roi) : min(w, u + roi + 1)]
        z = float(np.nanmedian(patch))
        if not math.isfinite(z) or z <= 0.05 or z > 10.0:
            return None

        x_cam = (float(u) - cx) * z / fx
        y_cam = (float(v) - cy) * z / fy
        du = float(u) - cx_img
        dv = float(v) - cy_img
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

    def _tool_pose_optical_look_at_point(
        self,
        point_cam: np.ndarray,
        *,
        optical_frame: str,
        base_frame: str,
        tool_frame: str,
        roll_rad: float,
        tf_timeout_sec: float,
    ) -> Optional[Pose]:
        """Fix camera origin in base; rotate so optical +Z points from camera toward ``point_cam`` (optical coords)."""
        T_base_opt = self._lookup_tf_mat(base_frame, optical_frame, tf_timeout_sec)
        if T_base_opt is None:
            return None
        T_tool_opt = self._lookup_tf_mat(tool_frame, optical_frame, tf_timeout_sec)
        if T_tool_opt is None:
            return None
        pch = np.array(
            [float(point_cam[0]), float(point_cam[1]), float(point_cam[2]), 1.0],
            dtype=np.float64,
        )
        p_base_h = T_base_opt @ pch
        p_base = p_base_h[:3]
        o_base = T_base_opt[:3, 3].copy()
        toward = p_base - o_base
        nrm = float(np.linalg.norm(toward))
        if not math.isfinite(nrm) or nrm < 1e-4:
            return None
        d = toward / nrm
        R_opt_in_base = _tcp_rotation_from_lookat(d, roll_rad)
        T_goal_opt_in_base = np.eye(4, dtype=np.float64)
        T_goal_opt_in_base[:3, :3] = R_opt_in_base
        T_goal_opt_in_base[:3, 3] = o_base
        T_base_tool = T_goal_opt_in_base @ np.linalg.inv(T_tool_opt)
        return _pose_from_transform(T_base_tool)

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
        YOLO + RGB-D with two phases:
        (1) Wrist IK: rotate so camera optical +Z looks at the detection 3-D point while keeping lens position.
        (2) When bbox center lies within tolerance, Cartesian path (+ IK fallback): translate camera along +Z_optical in
            base until measured depth ~= ``target_distance_m``.
        Detection requires confidence >= ``min_confidence``.
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
        max_iters = max(1, int(cfg["max_iterations"]))
        lookat_lim = cfg.get("lookat_max_iterations")
        lookat_iters = max(1, int(lookat_lim if lookat_lim is not None else max_iters))
        approach_iters = max(1, int(cfg["approach_max_iterations"]))
        approach_clip = float(cfg["approach_step_max_m"])

        tol_px = float(cfg["center_tolerance_px"])
        tol_z = float(cfg["distance_tolerance_m"])
        roi = int(cfg["depth_roi_half_px"])
        yolo_iou = float(cfg["yolo_iou"])
        optical = str(cfg["camera_optical_frame"])
        wait_t = float(cfg["wait_image_timeout_sec"])
        cam_roll = float(cfg["camera_lookat_tcp_roll_rad"])
        tf_timeout = 2.0

        base = str(self.motion_cfg["base_frame"])
        tool_frame = str(self.motion_cfg["ik_link_name"])

        self.get_logger().info(
            f"yolo_visual_center: class={target_class!r}, min_confidence={min_conf}, "
            f"target_distance_m={dist_target:.3f}, lookat_iterations={lookat_iters}, "
            f"approach_iterations={approach_iters}"
        )

        self._ensure_vision_subscriptions_for_yolo()
        self._clear_vision_buffers()

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

        # --- Phase 1: wrist look-at toward detection until image center aligns (no distance servo yet).
        last_det_conf = -1.0
        for it in range(lookat_iters):
            self._raise_if_abort()
            if not self._wait_vision_frames(2.0):
                self.get_logger().warn("Lost camera stream.")
                return False
            with self._vis_lock:
                bgr = self._vis_bgr.copy() if self._vis_bgr is not None else None
                depth = self._vis_depth.copy() if self._vis_depth is not None else None
                info = self._vis_info

            if bgr is None or depth is None or info is None:
                return False

            det = self._yolo_detection_center_ray(
                model, bgr, depth, info, target_class=target_class, min_conf=min_conf, yolo_iou=yolo_iou, roi=roi
            )
            if det is None:
                self.get_logger().error(
                    f"No {target_class!r} detection (or invalid depth); cannot look at target. "
                    f"Last detection confidence logged may be stale."
                )
                return False
            _, _, x_cam, y_cam, z_ray, du, dv, last_det_conf = det

            if abs(du) <= tol_px and abs(dv) <= tol_px:
                self.get_logger().info(
                    f"yolo_visual_center: bbox centered within ±{tol_px:.0f}px (look-at round {it + 1}, "
                    f"conf={last_det_conf:.3f}). Proceeding to approach."
                )
                break

            pose_goal = self._tool_pose_optical_look_at_point(
                np.array([x_cam, y_cam, z_ray], dtype=np.float64),
                optical_frame=optical,
                base_frame=base,
                tool_frame=tool_frame,
                roll_rad=cam_roll,
                tf_timeout_sec=tf_timeout,
            )
            if pose_goal is None:
                self.get_logger().error("yolo_visual_center: failed to compute optical look-at tool pose.")
                return False

            joint_positions = self._solve_ik_scaled_steps(pose_goal, action)
            if joint_positions is None:
                self.get_logger().error(
                    "yolo_visual_center: no IK solution for wrist look-at (try Home pose closer or "
                    "camera_lookat_tcp_roll_rad to match mounting)."
                )
                return False
            self.get_logger().info(
                f"yolo_visual_center look-at {it + 1}/{lookat_iters}: conf={last_det_conf:.3f}, "
                f"center_offset_px=({du:.1f},{dv:.1f}), depth_m={z_ray:.3f}"
            )
            if not self._execute_joint_goal(action, joint_positions):
                return False
        else:
            self.get_logger().error(
                f"yolo_visual_center: bbox not centered within {lookat_iters} look-at iterations "
                f"(threshold ±{tol_px:.0f} px)."
            )
            return False

        # --- Phase 2: Cartesian + depth until ~target_distance_m along ray.
        for ap in range(approach_iters):
            self._raise_if_abort()
            if not self._wait_vision_frames(2.0):
                self.get_logger().warn("Lost camera stream during approach.")
                return False

            with self._vis_lock:
                bgr = self._vis_bgr.copy() if self._vis_bgr is not None else None
                depth = self._vis_depth.copy() if self._vis_depth is not None else None
                info = self._vis_info
            if bgr is None or depth is None or info is None:
                return False

            det = self._yolo_detection_center_ray(
                model, bgr, depth, info, target_class=target_class, min_conf=min_conf, yolo_iou=yolo_iou, roi=roi
            )
            if det is None:
                self.get_logger().error("yolo_visual_center: lost detection during approach.")
                return False
            _, _, _, _, z_cam, du, dv, last_det_conf = det

            if abs(du) > tol_px or abs(dv) > tol_px:
                self.get_logger().warn(
                    f"yolo_visual_center: bbox re-centered drift (Δu={du:.1f}, Δv={dv:.1f}); "
                    "re-run sequence or widen center_tolerance_px."
                )
                return False

            if abs(z_cam - dist_target) <= tol_z:
                self.get_logger().info(
                    f"yolo_visual_center: depth converged (~{dist_target:.2f} m ± {tol_z:.3f}), conf={last_det_conf:.3f}"
                )
                return True

            step = float(z_cam - dist_target)
            if abs(step) > approach_clip:
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

            self.get_logger().info(
                f"yolo_visual_center approach {ap + 1}/{approach_iters}: ΔZ_optical_step={step:.3f} m, depth={z_cam:.3f}"
            )
            if not self._move_cartesian_waypoint_then_ik(action, pose_fwd):
                return False

        self.get_logger().error(
            f"yolo_visual_center: exceeded {approach_iters} approach iterations without reaching depth "
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
                self._delay_after_action(action)
                self._publish_action_completed(name)
                continue

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

    def _execute_weld_action(self, action: Dict[str, Any]) -> bool:
        """Publish a Case_Base-frame plate pose then call MuJoCo attach service (equality weld switch)."""
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
        for current in positions[1:]:
            deltas = [abs(float(a) - float(b)) for a, b in zip(current, previous)]
            # Keep Cartesian micro-waypoints moving continuously instead of stopping at every sample.
            times.append(times[-1] + max(max(deltas, default=0.0) / velocity_scaling, 0.02))
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
