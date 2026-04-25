#!/usr/bin/env python3
"""
UR3 + MuJoCo + virtual D435i: terminal-driven pick sequence.

Uses OpenCV on /rl_camera/color + /rl_camera/depth, tf2 to world, and MoveIt 2's
MoveGroup action (ROS 2 Humble: typically /move_action). Initial view uses joint-space
goals (same effect as moveit_commander.set_joint_value_target). Grasp uses Cartesian
constraints on tool0. Adjust all caps at top.
"""

from __future__ import annotations

import math
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PointStamped, Pose, PoseStamped, Quaternion, TransformStamped, Vector3
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    BoundingVolume,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    OrientationConstraint,
    PlanningOptions,
    PositionConstraint,
    WorkspaceParameters,
)
from moveit_msgs.msg import MoveItErrorCodes
from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_point

# -----------------------------------------------------------------------------
# User-tunable parameters (edit here)
# -----------------------------------------------------------------------------

# Camera / perception
RGB_TOPIC = "/rl_camera/color"
DEPTH_TOPIC = "/rl_camera/depth"
CAMERA_INFO_TOPIC = "/rl_camera/camera_info"
CAMERA_OPTICAL_FRAME = "rl_camera_frame"  # depth/RGB frame_id from MuJoCo bridge
WORLD_FRAME = "world"

# MoveIt
PLANNING_GROUP = "ur_manipulator"
END_EFFECTOR_LINK = "tool0"
MOVE_GROUP_ACTION = "/move_action"
PLANNING_TIME_SEC = 15.0
PLANNING_ATTEMPTS = 8
GOAL_POSITION_TOLERANCE_M = 0.02  # sphere radius for position constraint
ORIENTATION_TOL_RAD = 0.35

# Motion scaling (0..1]
VEL_SCALE_NORMAL = 0.35
VEL_SCALE_APPROACH = 0.12
ACC_SCALE = 0.35

# Initial "look at workspace" — joint space (radians), same order as MANIPULATOR_JOINT_NAMES.
# Equivalent to move_group.set_joint_value_target(positions) before plan/execute.
ENABLE_INITIAL_VIEW_POSE = True
MANIPULATOR_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
# Tune this pose so the virtual camera sees the table objects (UR3 + your scene).
INITIAL_JOINT_POSITIONS = [
    -1.6406,
    -2.3038,
    1.4137,
    1.7104,
    1.6406,
    0.0,
]
# Planner goal tolerance per joint (rad) around each target angle.
INITIAL_JOINT_GOAL_TOLERANCE_RAD = 0.05
INITIAL_JOINT_MOVE_VEL_SCALE = 0.25
# After motion, wait until /joint_states is within this band (then optional extra sleep).
INITIAL_JOINT_SETTLE_TOL_RAD = 0.1
INITIAL_JOINT_SETTLE_TIMEOUT_SEC = 20.0
INITIAL_VIEW_SETTLE_SEC = 0.75  # extra dwell for MuJoCo/camera after joints settle

# Optional multi-view fusion scan (motion remains in this task script).
ENABLE_MULTI_VIEW_FUSION_SCAN = True
FUSION_RESET_SERVICE = "/multi_view_cloud_fusion/reset"
FUSION_CAPTURE_SERVICE = "/multi_view_cloud_fusion/capture"
FUSION_TARGET_TOPIC = "/multi_view_cloud_fusion/target_origin"
SCAN_RELATIVE_OFFSETS_XYZ: List[Tuple[float, float, float]] = [
    (0.10, 0.00, 0.16),
    (0.00, 0.10, 0.16),
    (-0.10, 0.00, 0.16),
]
SCAN_MOVE_VEL_SCALE = 0.22
SCAN_SETTLE_SEC = 0.55

# Offsets in world frame (meters): applied to detected object center
PRE_GRASP_OFFSET_XYZ = (0.0, 0.0, 0.12)  # e.g. above object
APPROACH_OFFSET_XYZ = (0.0, 0.0, 0.04)  # small extra retreat before final approach
FINAL_GRASP_OFFSET_XYZ = (0, 0, 0.05)  # exact grasp point tweak

# Grasp orientation: quaternion x,y,z,w for tool0 in WORLD_FRAME (tune in RViz)
GRASP_QUAT_XYZW = (0.64968, 0.75269, -0.022532, -0.1042)

# Timings
GRASP_PAUSE_SEC = 3.0
LIFT_DELTA_Z = 0.12
ACTION_TIMEOUT_SEC = 120.0
CAMERA_WAIT_TIMEOUT_SEC = 30.0
TF_TIMEOUT_SEC = 5.0

# OpenCV: HSV ranges per object name (lower_hsv, upper_hsv) — tune for your props / lighting
# Names are matched case-insensitively.
OBJECT_HSV_RANGES: Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = {
    "cylinder_1": ((35, 40, 40), (90, 255, 255)),   # green-ish
    "cylinder_2": ((90, 40, 40), (130, 255, 255)),  # blue-ish
    "cylinder_3": ((80, 20, 20), (100, 255, 255)),  # cyan-ish
    "square_1": ((0, 60, 40), (15, 255, 255)),      # red-ish
    "square_2": ((10, 60, 40), (30, 255, 255)),     # orange-ish
    "square_3": ((135, 40, 40), (170, 255, 255)),   # magenta-ish
    # Group aliases
    "cylinder": ((35, 20, 20), (130, 255, 255)),
    "square": ((0, 40, 30), (180, 255, 255)),
    "default": ((0, 0, 40), (180, 255, 255)),
}

# Minimum contour area (pixels) to accept as object
MIN_CONTOUR_AREA_PX = 400

# Depth sampling: half-size of square ROI around center (pixels)
DEPTH_ROI_HALF = 3

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _quat_normalize(q: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def classify_shape_type(contour: np.ndarray) -> str:
    """Very rough shape label from contour circularity."""
    area = float(cv2.contourArea(contour))
    peri = float(cv2.arcLength(contour, True))
    if peri < 1e-6 or area < 1e-6:
        return "unknown"
    circ = 4.0 * math.pi * area / (peri * peri)
    if circ > 0.78:
        return "sphere"
    if circ > 0.62:
        return "cylinder_like"
    return "box_like"


def build_pose_goal_constraints(
    pos_world: Tuple[float, float, float],
    quat_xyzw: Tuple[float, float, float, float],
    link_name: str = END_EFFECTOR_LINK,
    constraint_name: str = "pose_goal",
    position_tolerance_m: Optional[float] = None,
) -> Constraints:
    x, y, z = pos_world
    qx, qy, qz, qw = _quat_normalize(quat_xyzw)
    tol = float(GOAL_POSITION_TOLERANCE_M if position_tolerance_m is None else position_tolerance_m)

    sphere = SolidPrimitive()
    sphere.type = SolidPrimitive.SPHERE
    sphere.dimensions = [tol]

    region_pose = Pose()
    region_pose.position.x = x
    region_pose.position.y = y
    region_pose.position.z = z
    region_pose.orientation.w = 1.0

    vol = BoundingVolume()
    vol.primitives = [sphere]
    vol.primitive_poses = [region_pose]

    pos_c = PositionConstraint()
    pos_c.header = Header(frame_id=WORLD_FRAME)
    pos_c.link_name = link_name
    pos_c.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
    pos_c.constraint_region = vol
    pos_c.weight = 1.0

    ori_c = OrientationConstraint()
    ori_c.header = Header(frame_id=WORLD_FRAME)
    ori_c.link_name = link_name
    ori_c.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    ori_c.absolute_x_axis_tolerance = float(ORIENTATION_TOL_RAD)
    ori_c.absolute_y_axis_tolerance = float(ORIENTATION_TOL_RAD)
    ori_c.absolute_z_axis_tolerance = float(ORIENTATION_TOL_RAD)
    ori_c.weight = 1.0
    ori_c.parameterization = OrientationConstraint.XYZ_EULER_ANGLES

    c = Constraints()
    c.name = constraint_name
    c.position_constraints = [pos_c]
    c.orientation_constraints = [ori_c]
    return c


def build_joint_goal_constraints(
    joint_names: List[str],
    positions: List[float],
    tolerance_rad: float,
    constraint_name: str = "joint_goal",
) -> Constraints:
    if len(joint_names) != len(positions):
        raise ValueError("joint_names and positions must have same length")
    tol = float(max(tolerance_rad, 1e-4))
    jcs: List[JointConstraint] = []
    for jn, pos in zip(joint_names, positions):
        jc = JointConstraint()
        jc.joint_name = jn
        jc.position = float(pos)
        jc.tolerance_above = tol
        jc.tolerance_below = tol
        jc.weight = 1.0
        jcs.append(jc)
    c = Constraints()
    c.name = constraint_name
    c.joint_constraints = jcs
    return c


def build_motion_request(
    constraints: Constraints,
    vel_scale: float,
) -> MotionPlanRequest:
    wp = WorkspaceParameters()
    wp.header.frame_id = WORLD_FRAME
    wp.min_corner.x = -2.0
    wp.min_corner.y = -2.0
    wp.min_corner.z = -0.5
    wp.max_corner.x = 2.0
    wp.max_corner.y = 2.0
    wp.max_corner.z = 2.0

    req = MotionPlanRequest()
    req.workspace_parameters = wp
    req.start_state.joint_state.name = []
    req.start_state.joint_state.position = []
    req.start_state.is_diff = False
    req.goal_constraints = [constraints]
    req.group_name = PLANNING_GROUP
    req.num_planning_attempts = PLANNING_ATTEMPTS
    req.allowed_planning_time = float(PLANNING_TIME_SEC)
    req.max_velocity_scaling_factor = float(np.clip(vel_scale, 0.01, 1.0))
    req.max_acceleration_scaling_factor = float(np.clip(ACC_SCALE, 0.01, 1.0))
    return req


class PickTaskNode(Node):
    def __init__(self) -> None:
        super().__init__("ur3_pick_task")
        self._rgb: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._info: Optional[CameraInfo] = None
        self._lock = threading.Lock()
        self._js_lock = threading.Lock()
        self._latest_joint_pos: Dict[str, float] = {}

        self.create_subscription(Image, RGB_TOPIC, self._on_rgb, 1)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, 1)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._on_info, 1)
        self.create_subscription(JointState, "/joint_states", self._on_joint_states, 10)

        self._tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

        self._move_client = ActionClient(self, MoveGroup, MOVE_GROUP_ACTION)
        self._fusion_reset_client = self.create_client(Trigger, FUSION_RESET_SERVICE)
        self._fusion_capture_client = self.create_client(Trigger, FUSION_CAPTURE_SERVICE)
        self._fusion_target_pub = self.create_publisher(PoseStamped, FUSION_TARGET_TOPIC, 10)

    def _on_joint_states(self, msg: JointState) -> None:
        with self._js_lock:
            for n, p in zip(msg.name, msg.position):
                self._latest_joint_pos[n] = float(p)

    def _on_rgb(self, msg: Image) -> None:
        try:
            if msg.encoding not in ("rgb8", "bgr8"):
                self.get_logger().warn(f"Expected rgb8/bgr8, got {msg.encoding}")
                return
            if msg.height * msg.width * 3 != len(msg.data):
                self.get_logger().warn("RGB image size mismatch; dropping frame.")
                return
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
            bgr = arr.copy() if msg.encoding == "bgr8" else cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            with self._lock:
                self._rgb = bgr
        except Exception as e:
            self.get_logger().warn(f"RGB decode failed: {e}")

    def _on_depth(self, msg: Image) -> None:
        try:
            if msg.encoding != "32FC1":
                self.get_logger().warn(f"Expected depth 32FC1, got {msg.encoding}")
                return
            d = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            with self._lock:
                self._depth = d.copy()
        except Exception as e:
            self.get_logger().warn(f"Depth decode failed: {e}")

    def _on_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._info = msg

    def wait_for_camera(self, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline and rclpy.ok():
            with self._lock:
                ok = self._rgb is not None and self._depth is not None and self._info is not None
            if ok:
                return True
            time.sleep(0.05)
        return False

    def detect_object_center(
        self, object_name: str
    ) -> Tuple[Optional[Tuple[int, int]], str, str]:
        """
        Returns ((u, v) pixel center or None, matched_filter_key, shape_type).
        """
        key = object_name.strip().lower()
        hsv_range = None
        for name, span in OBJECT_HSV_RANGES.items():
            if name.lower() == key or key in name.lower() or name.lower() in key:
                hsv_range = span
                matched = name
                break
        if hsv_range is None:
            hsv_range = OBJECT_HSV_RANGES["default"]
            matched = "default"

        with self._lock:
            if self._rgb is None:
                return None, matched, "unknown"
            bgr = self._rgb.copy()

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_range[0]), np.array(hsv_range[1]))
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        best_area = 0
        for c in contours:
            a = cv2.contourArea(c)
            if a < MIN_CONTOUR_AREA_PX:
                continue
            if a > best_area:
                best_area = a
                best = c
        if best is None:
            return None, matched, "unknown"
        m = cv2.moments(best)
        if m["m00"] < 1e-6:
            return None, matched, "unknown"
        u = int(m["m10"] / m["m00"])
        v = int(m["m01"] / m["m00"])
        shape_type = classify_shape_type(best)
        return (u, v), matched, shape_type

    def pixel_to_world(self, u: int, v: int) -> Optional[Tuple[float, float, float]]:
        with self._lock:
            if self._depth is None or self._info is None:
                return None
            depth = self._depth.copy()
            info = self._info

        h, w = depth.shape
        u = int(np.clip(u, 0, w - 1))
        v = int(np.clip(v, 0, h - 1))
        r = DEPTH_ROI_HALF
        patch = depth[max(0, v - r) : min(h, v + r + 1), max(0, u - r) : min(w, u + r + 1)]
        z = float(np.nanmedian(patch))
        if not math.isfinite(z) or z <= 0.01 or z > 10.0:
            self.get_logger().error("Invalid depth at object center; check segmentation / camera view.")
            return None

        K = info.k
        fx, fy = float(K[0]), float(K[4])
        cx, cy = float(K[2]), float(K[5])
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy

        ps = PointStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = CAMERA_OPTICAL_FRAME
        ps.point.x = float(x)
        ps.point.y = float(y)
        ps.point.z = float(z)

        try:
            tf: TransformStamped = self._tf_buffer.lookup_transform(
                WORLD_FRAME,
                CAMERA_OPTICAL_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=TF_TIMEOUT_SEC),
            )
        except Exception as e:
            self.get_logger().error(f"TF lookup failed: {e}")
            return None

        out = do_transform_point(ps, tf)
        return (out.point.x, out.point.y, out.point.z)

    def _send_move_group_request(self, req: MotionPlanRequest, label: str, log_detail: str) -> bool:
        if not self._move_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error(f"MoveGroup action '{MOVE_GROUP_ACTION}' not available.")
            return False

        goal = MoveGroup.Goal()
        goal.request = req
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = False

        self.get_logger().info(f"MoveIt: {label} {log_detail}")
        send_future = self._move_client.send_goal_async(goal)
        start = time.time()
        while rclpy.ok() and not send_future.done() and time.time() - start < ACTION_TIMEOUT_SEC:
            time.sleep(0.02)
        if not send_future.done():
            self.get_logger().error("send_goal_async timed out.")
            return False
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveGroup goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        start = time.time()
        while rclpy.ok() and not result_future.done() and time.time() - start < ACTION_TIMEOUT_SEC:
            time.sleep(0.02)
        if not result_future.done():
            self.get_logger().error("MoveGroup result timed out.")
            return False

        res = result_future.result().result
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"MoveIt failed ({label}): error_code={res.error_code.val} "
                f"(SUCCESS={MoveItErrorCodes.SUCCESS})"
            )
            return False
        self.get_logger().info(f"MoveIt OK: {label}")
        return True

    def wait_until_joints_near(
        self,
        joint_names: List[str],
        targets: List[float],
        tol_rad: float,
        timeout_sec: float,
    ) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline and rclpy.ok():
            with self._js_lock:
                snap = dict(self._latest_joint_pos)
            ok = True
            for jn, tgt in zip(joint_names, targets):
                if jn not in snap:
                    ok = False
                    break
                if abs(snap[jn] - tgt) > tol_rad:
                    ok = False
                    break
            if ok and len(joint_names) > 0:
                return True
            time.sleep(0.05)
        self.get_logger().error(
            f"Timed out waiting for joints within {tol_rad} rad of target "
            f"(timeout {timeout_sec}s)."
        )
        return False

    def move_to_joint_positions(
        self,
        joint_names: List[str],
        positions: List[float],
        vel_scale: float,
        label: str,
    ) -> bool:
        """Joint-space goal via JointConstraint[] (ROS 2 equivalent of set_joint_value_target)."""
        if len(joint_names) != len(positions):
            self.get_logger().error("joint_names and positions length mismatch.")
            return False
        names = list(joint_names)
        cons = build_joint_goal_constraints(
            names,
            list(positions),
            INITIAL_JOINT_GOAL_TOLERANCE_RAD,
            constraint_name=f"{label}_joint_goal",
        )
        req = build_motion_request(cons, vel_scale)
        pairs = ", ".join(f"{n}={p:.4f}" for n, p in zip(names, positions))
        return self._send_move_group_request(req, label, f"joint target [{pairs}]")

    def move_to_world_pose(
        self,
        xyz: Tuple[float, float, float],
        vel_scale: float,
        label: str,
        quat_xyzw: Optional[Tuple[float, float, float, float]] = None,
        link_name: Optional[str] = None,
        constraint_name: Optional[str] = None,
        position_tolerance_m: Optional[float] = None,
    ) -> bool:
        q = quat_xyzw if quat_xyzw is not None else GRASP_QUAT_XYZW
        ln = link_name if link_name is not None else END_EFFECTOR_LINK
        cn = constraint_name if constraint_name is not None else label
        cons = build_pose_goal_constraints(
            xyz, q, link_name=ln, constraint_name=cn, position_tolerance_m=position_tolerance_m
        )
        req = build_motion_request(cons, vel_scale)
        detail = f"-> ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}) link={ln}"
        return self._send_move_group_request(req, label, detail)

    def call_trigger(self, client, name: str, timeout_sec: float = 5.0) -> bool:
        if not client.wait_for_service(timeout_sec=timeout_sec):
            self.get_logger().error(f"Service '{name}' unavailable.")
            return False
        fut = client.call_async(Trigger.Request())
        start = time.time()
        while rclpy.ok() and not fut.done() and time.time() - start < timeout_sec:
            time.sleep(0.02)
        if not fut.done():
            self.get_logger().error(f"Service call timeout: {name}")
            return False
        res = fut.result()
        if res is None or not bool(res.success):
            msg = "" if res is None else str(res.message)
            self.get_logger().error(f"Service '{name}' failed: {msg}")
            return False
        self.get_logger().info(f"Service '{name}': {res.message}")
        return True

    def publish_fusion_target_origin(self, xyz_world: Tuple[float, float, float]) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = WORLD_FRAME
        msg.pose.position.x = float(xyz_world[0])
        msg.pose.position.y = float(xyz_world[1])
        msg.pose.position.z = float(xyz_world[2])
        msg.pose.orientation.w = 1.0
        self._fusion_target_pub.publish(msg)


def _add_xyz(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def main(args: Optional[List[str]] = None) -> None:
    print("\n=== UR3 MuJoCo pick task ===")
    print("Ensure: ur3_mujoco_moveit.launch.py is running and use_sim_time is consistent.\n")
    name = input(
        "Enter object name (e.g. cylinder_1, cylinder_2, cylinder_3, "
        "square_1, square_2, square_3): "
    )
    if not name.strip():
        print("Empty name, exit.")
        return

    rclpy.init(args=args)
    node = PickTaskNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    try:
        if not node.wait_for_camera(CAMERA_WAIT_TIMEOUT_SEC):
            node.get_logger().error("Timed out waiting for camera topics.")
            return

        if ENABLE_INITIAL_VIEW_POSE:
            if len(MANIPULATOR_JOINT_NAMES) != len(INITIAL_JOINT_POSITIONS):
                node.get_logger().error(
                    "MANIPULATOR_JOINT_NAMES and INITIAL_JOINT_POSITIONS must have 6 entries each."
                )
                return
            node.get_logger().info(
                "Moving to initial joint-space view pose "
                f"(set_joint_value_target equivalent): {list(zip(MANIPULATOR_JOINT_NAMES, INITIAL_JOINT_POSITIONS))}"
            )
            if not node.move_to_joint_positions(
                MANIPULATOR_JOINT_NAMES,
                INITIAL_JOINT_POSITIONS,
                INITIAL_JOINT_MOVE_VEL_SCALE,
                "initial_view",
            ):
                return
            if not node.wait_until_joints_near(
                MANIPULATOR_JOINT_NAMES,
                INITIAL_JOINT_POSITIONS,
                INITIAL_JOINT_SETTLE_TOL_RAD,
                INITIAL_JOINT_SETTLE_TIMEOUT_SEC,
            ):
                return
            node.get_logger().info(
                f"Joints settled; waiting {INITIAL_VIEW_SETTLE_SEC:.2f}s for sim/camera before detection."
            )
            time.sleep(INITIAL_VIEW_SETTLE_SEC)

        uv, filter_key, shape_type = node.detect_object_center(name)
        if uv is None:
            node.get_logger().error(
                f"No contour found for '{name}' (filter='{filter_key}'). Tune OBJECT_HSV_RANGES / lighting."
            )
            return
        u, v = uv
        node.get_logger().info(
            f"Detection: pixel=({u},{v}), filter_key='{filter_key}', shape_type='{shape_type}'"
        )

        world = node.pixel_to_world(u, v)
        if world is None:
            return
        node.get_logger().info(f"Object center (world): {world}")

        pre = _add_xyz(world, PRE_GRASP_OFFSET_XYZ)
        pre = _add_xyz(pre, APPROACH_OFFSET_XYZ)
        grasp = _add_xyz(world, FINAL_GRASP_OFFSET_XYZ)

        # Move directly above identified target first.
        if not node.move_to_world_pose(pre, VEL_SCALE_NORMAL, "above_target"):
            return

        # Object-centric scan around target origin; fusion node does only capture+fusion.
        if ENABLE_MULTI_VIEW_FUSION_SCAN:
            node.publish_fusion_target_origin(world)
            if not node.call_trigger(node._fusion_reset_client, FUSION_RESET_SERVICE):
                return
            for i, rel in enumerate(SCAN_RELATIVE_OFFSETS_XYZ):
                scan_xyz = _add_xyz(world, rel)
                if not node.move_to_world_pose(
                    scan_xyz,
                    SCAN_MOVE_VEL_SCALE,
                    f"scan_view_{i+1}",
                    quat_xyzw=GRASP_QUAT_XYZW,
                    link_name=END_EFFECTOR_LINK,
                ):
                    return
                time.sleep(SCAN_SETTLE_SEC)
                # Refresh target stamp near capture time (same origin, current clock).
                node.publish_fusion_target_origin(world)
                if not node.call_trigger(node._fusion_capture_client, FUSION_CAPTURE_SERVICE):
                    return

        if not node.move_to_world_pose(pre, VEL_SCALE_NORMAL, "pre_grasp"):
            return
        if not node.move_to_world_pose(grasp, VEL_SCALE_APPROACH, "approach_grasp"):
            return

        node.get_logger().info(f"Simulated grasp: waiting {GRASP_PAUSE_SEC:.1f}s ...")
        time.sleep(GRASP_PAUSE_SEC)

        lifted = (grasp[0], grasp[1], grasp[2] + LIFT_DELTA_Z)
        if not node.move_to_world_pose(lifted, VEL_SCALE_NORMAL, "lift"):
            return

        node.get_logger().info("Sequence complete.")
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
