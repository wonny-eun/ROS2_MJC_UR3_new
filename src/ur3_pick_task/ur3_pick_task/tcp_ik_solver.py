#!/usr/bin/env python3
"""Interactive TCP inverse-kinematics solver using MoveIt /compute_ik."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import rclpy
import yaml
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import PoseStamped
from moveit_msgs.action import MoveGroup
from moveit_msgs.msg import (
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MoveItErrorCodes,
    PlanningOptions,
    WorkspaceParameters,
)
from moveit_msgs.srv import GetPositionIK
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


DEFAULT_TCP_FILE = (
    "/home/wonny/ur3_control/src/ROS2_MuJoCo_UR3/src/ur3_rl_bridge/"
    "config/tcp/gripper_tip.yaml"
)


def _load_yaml(path: str) -> Dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a map: {path}")
    return data


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
                [0.25 * s, (rot[0, 1] + rot[1, 0]) / s, (rot[0, 2] + rot[2, 0]) / s, (rot[2, 1] - rot[1, 2]) / s],
                dtype=np.float64,
            )
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            q = np.array(
                [(rot[0, 1] + rot[1, 0]) / s, 0.25 * s, (rot[1, 2] + rot[2, 1]) / s, (rot[0, 2] - rot[2, 0]) / s],
                dtype=np.float64,
            )
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            q = np.array(
                [(rot[0, 2] + rot[2, 0]) / s, (rot[1, 2] + rot[2, 1]) / s, 0.25 * s, (rot[1, 0] - rot[0, 1]) / s],
                dtype=np.float64,
            )
    return q / np.linalg.norm(q)


def _transform_from_yaml(data: Dict[str, Any]) -> np.ndarray:
    xyz = data.get("translation", {})
    if "rotation_xyzw" in data:
        q = data["rotation_xyzw"]
        quat = np.array([float(q.get("x", 0.0)), float(q.get("y", 0.0)), float(q.get("z", 0.0)), float(q.get("w", 1.0))])
    else:
        rpy = data.get("rotation_rpy", {})
        quat = _quat_from_rpy(float(rpy.get("roll", 0.0)), float(rpy.get("pitch", 0.0)), float(rpy.get("yaw", 0.0)))

    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = _quat_to_rot(quat)
    tf[:3, 3] = [float(xyz.get("x", 0.0)), float(xyz.get("y", 0.0)), float(xyz.get("z", 0.0))]
    return tf


def _parse_three_floats(text: str, label: str) -> np.ndarray:
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"{label} needs exactly 3 numbers")
    return np.array([float(v) for v in parts], dtype=np.float64)


def _param_vector3(value: Any, label: str) -> np.ndarray:
    if len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 numbers")
    return np.array([float(v) for v in value], dtype=np.float64)


def _normalize(v: np.ndarray, label: str) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        raise ValueError(f"{label} vector is zero")
    return v / n


def _tcp_rotation_from_lookat(lookat: np.ndarray, roll_rad: float = 0.0) -> np.ndarray:
    """Build orientation whose TCP local +Z axis points along lookat.

    roll_rad rotates the TCP frame around that local +Z approach axis.
    """
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
        [
            [cr, -sr, 0.0],
            [sr, cr, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return base_rot @ local_roll


def _pose_msg_from_transform(frame_id: str, tf: np.ndarray) -> PoseStamped:
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


class TcpIkSolver(Node):
    def __init__(self) -> None:
        super().__init__("tcp_ik_solver")
        self.declare_parameter("tcp_file", DEFAULT_TCP_FILE)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("planning_group", "ur_manipulator")
        self.declare_parameter("ik_link_name", "tool0")
        self.declare_parameter("avoid_collisions", True)
        self.declare_parameter("ik_timeout_sec", 2.0)
        self.declare_parameter("execute_motion", False)
        self.declare_parameter("execute_mode", "moveit")
        self.declare_parameter("move_group_action", "/move_action")
        self.declare_parameter("planning_time_sec", 15.0)
        self.declare_parameter("planning_attempts", 8)
        self.declare_parameter("joint_goal_tolerance_rad", 0.01)
        self.declare_parameter("velocity_scale", 0.15)
        self.declare_parameter("acceleration_scale", 0.20)
        self.declare_parameter("trajectory_action", "/joint_trajectory_controller/follow_joint_trajectory")
        self.declare_parameter("motion_duration_sec", 3.0)
        self.declare_parameter(
            "manipulator_joint_names",
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
        )
        self.declare_parameter("use_config_target", False)
        self.declare_parameter("target_tcp_xyz", [0.0, 0.0, 0.9])
        self.declare_parameter("lookat_vector", [0.0, 0.0, -1.0])
        self.declare_parameter("tcp_roll_rad", 0.0)

        self._ik_client = self.create_client(GetPositionIK, "/compute_ik")
        self._move_group_client = ActionClient(
            self,
            MoveGroup,
            str(self.get_parameter("move_group_action").value),
        )
        self._trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            str(self.get_parameter("trajectory_action").value),
        )

    def solve(self) -> bool:
        tcp_file = str(self.get_parameter("tcp_file").value)
        base_frame = str(self.get_parameter("base_frame").value)
        group_name = str(self.get_parameter("planning_group").value)
        ik_link_name = str(self.get_parameter("ik_link_name").value)
        avoid_collisions = bool(self.get_parameter("avoid_collisions").value)
        ik_timeout = float(self.get_parameter("ik_timeout_sec").value)
        execute_motion = bool(self.get_parameter("execute_motion").value)
        execute_mode = str(self.get_parameter("execute_mode").value).lower()
        use_config_target = bool(self.get_parameter("use_config_target").value)

        tcp_yaml = _load_yaml(tcp_file)
        tool_to_tcp = _transform_from_yaml(tcp_yaml)

        print("\n=== TCP IK Solver ===")
        print(f"Base frame: {base_frame}")
        print("TCP local +Z axis will point along the look-at vector.")
        if use_config_target:
            target_xyz = _param_vector3(self.get_parameter("target_tcp_xyz").value, "target_tcp_xyz")
            lookat = _param_vector3(self.get_parameter("lookat_vector").value, "lookat_vector")
            tcp_roll = float(self.get_parameter("tcp_roll_rad").value)
            print(f"Using config target_tcp_xyz: {target_xyz.tolist()}")
            print(f"Using config lookat_vector: {lookat.tolist()}")
            print(f"Using config tcp_roll_rad: {tcp_roll:.6f}")
        else:
            target_xyz = _parse_three_floats(input("Target TCP x y z: "), "target TCP")
            lookat = _parse_three_floats(input("Look-at vector x y z (example: 0 0 -1): "), "look-at")
            roll_text = input("TCP roll around look-at axis in radians (default 0): ").strip()
            tcp_roll = float(roll_text) if roll_text else 0.0

        base_to_tcp = np.eye(4, dtype=np.float64)
        base_to_tcp[:3, :3] = _tcp_rotation_from_lookat(lookat, tcp_roll)
        base_to_tcp[:3, 3] = target_xyz
        base_to_tool = base_to_tcp @ np.linalg.inv(tool_to_tcp)

        tcp_quat = _rot_to_quat(base_to_tcp[:3, :3])
        tool_quat = _rot_to_quat(base_to_tool[:3, :3])
        print(
            f"Target TCP quaternion xyzw: "
            f"{tcp_quat[0]:.6f} {tcp_quat[1]:.6f} {tcp_quat[2]:.6f} {tcp_quat[3]:.6f}"
        )
        print(
            f"Computed tool0 pose in {base_frame}: "
            f"x={base_to_tool[0, 3]:.6f}, y={base_to_tool[1, 3]:.6f}, z={base_to_tool[2, 3]:.6f}, "
            f"qx={tool_quat[0]:.6f}, qy={tool_quat[1]:.6f}, qz={tool_quat[2]:.6f}, qw={tool_quat[3]:.6f}"
        )

        if not self._ik_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("MoveIt IK service /compute_ik is not available.")
            return False

        req = GetPositionIK.Request()
        req.ik_request.group_name = group_name
        req.ik_request.ik_link_name = ik_link_name
        req.ik_request.pose_stamped = _pose_msg_from_transform(base_frame, base_to_tool)
        req.ik_request.avoid_collisions = avoid_collisions
        req.ik_request.timeout = Duration(sec=int(ik_timeout), nanosec=int((ik_timeout % 1.0) * 1e9))

        self.get_logger().info(
            f"Calling /compute_ik for {ik_link_name}, group={group_name}, avoid_collisions={avoid_collisions}"
        )
        future = self._ik_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=ik_timeout + 3.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("IK service timed out or returned no result.")
            return False

        result = future.result()
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"IK failed: error_code={result.error_code.val}. "
                "Try a closer target, different look-at vector, or avoid_collisions:=false."
            )
            return False

        joint_solution = dict(zip(result.solution.joint_state.name, result.solution.joint_state.position))
        joint_names = [str(name) for name in self.get_parameter("manipulator_joint_names").value]
        try:
            joint_positions = [float(joint_solution[name]) for name in joint_names]
        except KeyError as exc:
            self.get_logger().error(f"IK result did not contain expected joint: {exc}")
            return False

        print("\nIK joint solution:")
        for name, pos in zip(joint_names, joint_positions):
            print(f"  {name}: {pos:.6f}")

        if execute_motion:
            if execute_mode == "moveit":
                return self._execute_with_moveit(joint_names, joint_positions)
            if execute_mode == "controller":
                return self._execute_joint_trajectory(joint_names, joint_positions)
            self.get_logger().error("execute_mode must be 'moveit' or 'controller'")
            return False
        return True

    def _build_joint_motion_request(self, joint_names: list[str], joint_positions: list[float]) -> MotionPlanRequest:
        workspace = WorkspaceParameters()
        workspace.header.frame_id = str(self.get_parameter("base_frame").value)
        workspace.min_corner.x = -2.0
        workspace.min_corner.y = -2.0
        workspace.min_corner.z = -0.5
        workspace.max_corner.x = 2.0
        workspace.max_corner.y = 2.0
        workspace.max_corner.z = 2.0

        constraints = Constraints()
        constraints.name = "tcp_ik_joint_goal"
        tol = float(max(float(self.get_parameter("joint_goal_tolerance_rad").value), 1e-4))
        for name, pos in zip(joint_names, joint_positions):
            jc = JointConstraint()
            jc.joint_name = name
            jc.position = float(pos)
            jc.tolerance_above = tol
            jc.tolerance_below = tol
            jc.weight = 1.0
            constraints.joint_constraints.append(jc)

        req = MotionPlanRequest()
        req.workspace_parameters = workspace
        req.group_name = str(self.get_parameter("planning_group").value)
        req.goal_constraints = [constraints]
        req.num_planning_attempts = int(self.get_parameter("planning_attempts").value)
        req.allowed_planning_time = float(self.get_parameter("planning_time_sec").value)
        req.max_velocity_scaling_factor = float(np.clip(float(self.get_parameter("velocity_scale").value), 0.01, 1.0))
        req.max_acceleration_scaling_factor = float(np.clip(float(self.get_parameter("acceleration_scale").value), 0.01, 1.0))
        return req

    def _execute_with_moveit(self, joint_names: list[str], joint_positions: list[float]) -> bool:
        action_name = str(self.get_parameter("move_group_action").value)
        if not self._move_group_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"MoveGroup action server is not available: {action_name}")
            return False

        goal = MoveGroup.Goal()
        goal.request = self._build_joint_motion_request(joint_names, joint_positions)
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = False
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = 2

        self.get_logger().info("Sending IK joint solution to MoveIt for collision-aware route planning.")
        send_future = self._move_group_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveIt goal rejected.")
            return False

        timeout = float(self.get_parameter("planning_time_sec").value) + 30.0
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=timeout)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("MoveIt planning/execution timed out.")
            return False

        result = result_future.result().result
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"MoveIt planning/execution failed: error_code={result.error_code.val}"
            )
            return False
        self.get_logger().info("MoveIt collision-aware execution complete.")
        return True

    def _execute_joint_trajectory(self, joint_names: list[str], joint_positions: list[float]) -> bool:
        action_name = str(self.get_parameter("trajectory_action").value)
        duration = float(self.get_parameter("motion_duration_sec").value)
        if not self._trajectory_client.wait_for_server(timeout_sec=10.0):
            self.get_logger().error(f"Trajectory action server is not available: {action_name}")
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = joint_names
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.velocities = [0.0] * len(joint_names)
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1.0) * 1e9))
        goal.trajectory.points = [point]

        self.get_logger().info(f"Sending IK joint solution to MuJoCo controller: {action_name}")
        send_future = self._trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future, timeout_sec=10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected.")
            return False

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future, timeout_sec=duration + 10.0)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("Trajectory execution timed out.")
            return False

        result = result_future.result().result
        if result.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().error(
                f"Trajectory execution failed: error_code={result.error_code}, message='{result.error_string}'"
            )
            return False
        self.get_logger().info("MuJoCo trajectory execution complete.")
        return True


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = TcpIkSolver()
    try:
        node.solve()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
