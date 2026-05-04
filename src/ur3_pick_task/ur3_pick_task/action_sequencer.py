#!/usr/bin/env python3
"""Config-driven action sequencer for UR3 MoveIt motions."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose
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
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


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


def _load_yaml(path: str) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a map: {path}")
    return data


def _duration_from_seconds(seconds: float) -> Duration:
    seconds = max(float(seconds), 0.0)
    sec = int(seconds)
    return Duration(sec=sec, nanosec=int((seconds - sec) * 1e9))


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
    return base_rot @ local_roll


def _pose_from_transform(tf: np.ndarray) -> Pose:
    q = _rot_to_quat(tf[:3, :3])
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
        self._latest_joint_state: Optional[JointState] = None

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

    def _on_joint_states(self, msg: JointState) -> None:
        if msg.name and msg.position:
            self._latest_joint_state = msg

    def run(self) -> bool:
        self.get_logger().info(
            f"Starting sequence '{self.sequence_name}' from {self.config_file} "
            f"with {len(self.sequence)} actions."
        )
        all_ok = True
        for index, action in enumerate(self.sequence, start=1):
            name = str(action.get("name", f"action_{index}"))
            action_type = str(action.get("type", "")).lower()
            self.get_logger().info(f"[{index}/{len(self.sequence)}] Starting {name} ({action_type})")
            try:
                ok = self._execute_action(action)
            except Exception as exc:  # noqa: BLE001 - report config/runtime failures cleanly.
                self.get_logger().error(f"Action '{name}' failed with exception: {exc}")
                ok = False

            if ok:
                self.get_logger().info(f"Action '{name}' complete.")
                self._delay_after_action(action)
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
        time.sleep(delay_sec)

    def _execute_action(self, action: Dict[str, Any]) -> bool:
        action_type = str(action.get("type", "")).lower()
        if action_type == "joint":
            return self._execute_joint_action(action)
        if action_type == "cartesian":
            return self._execute_cartesian_action(action)
        if action_type == "waypoint":
            return self._execute_waypoint_action(action)
        raise ValueError(f"Unsupported action type: {action_type!r}")

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
        if not self._sequence_client.wait_for_server(timeout_sec=1.0):
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
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveGroupSequence goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
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
        if not self._move_group_client.wait_for_server(timeout_sec=10.0):
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
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveIt goal rejected.")
            return False

        timeout = self._planning_time(action) + 30.0
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
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
        if not self._cartesian_client.wait_for_service(timeout_sec=5.0):
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
        rclpy.spin_until_future_complete(self, future, timeout_sec=self._planning_time(action) + 5.0)
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
        if self._execute_trajectory_client.wait_for_server(timeout_sec=2.0):
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = trajectory
            send_future = self._execute_trajectory_client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
            goal_handle = send_future.result()
            if goal_handle is None or not goal_handle.accepted:
                self.get_logger().error("ExecuteTrajectory goal rejected.")
                return False

            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
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
        if not self._trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Trajectory action is not available: {self.motion_cfg['trajectory_action']}")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory
        send_future = self._trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
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

    def _solve_ik(self, tool_pose: Pose, action: Dict[str, Any]) -> Optional[list[float]]:
        if not self._wait_for_joint_state(action):
            return None
        if not self._ik_client.wait_for_service(timeout_sec=10.0):
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
        rclpy.spin_until_future_complete(self, future, timeout_sec=ik_timeout + 3.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("IK service timed out or returned no result.")
            return None

        result = future.result()
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"IK failed: error_code={result.error_code.val}")
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
        lookat = _vector3(action.get("lookat_vector", [0.0, 0.0, -1.0]), "lookat_vector")
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


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = MotionManager()
    try:
        ok = node.run()
        if not ok:
            raise SystemExit(1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
