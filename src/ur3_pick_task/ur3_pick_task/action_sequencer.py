#!/usr/bin/env python3
"""Config-driven action sequencer for UR3 MoveIt motions."""

from __future__ import annotations

import math
import os
import select
from copy import deepcopy
import signal
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import cv2
import numpy as np
import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped, Twist, TwistStamped, Vector3
from moveit_msgs.action import ExecuteTrajectory, MoveGroup, MoveGroupSequence
from moveit_msgs.msg import (
    AttachedCollisionObject,
    BoundingVolume,
    CollisionObject,
    Constraints,
    JointConstraint,
    MotionPlanRequest,
    MotionSequenceItem,
    MoveItErrorCodes,
    OrientationConstraint,
    PlanningOptions,
    PlanningScene,
    PositionConstraint,
    RobotState,
    RobotTrajectory,
    WorkspaceParameters,
)
from std_msgs.msg import Header
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.srv import GetCartesianPath, GetPositionIK
from rclpy.action import ActionClient
from rclpy.duration import Duration as RclDuration
from rclpy.node import Node
from rclpy.time import Time
from rcl_interfaces.msg import Parameter, ParameterDescriptor, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import CameraInfo, Image, JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore[misc, assignment]

from ur3_pick_task.foundation_pose_grasp import (
    FoundationPoseStackLauncher,
    apply_foundation_pose_grasp_overrides,
    compute_base_tcp_pose,
    grasp_spec_for_class,
    grasp_spec_with_standoff,
    merge_grasp_config,
    is_upside_down_in_base,
    normalize_T_base_object_for_grasp,
    object_yaw_for_grasp,
    object_z_tilt_from_vertical_deg,
    resolve_tcp_roll_for_grasp,
    run_hover_convergence_loop,
    symmetry_axes_for_class,
    wait_for_detection3d_output,
    wait_for_topic_publishers,
    wait_tf_stable,
    yaw_about_base_z,
)
from ur3_rl_bridge.yolo_scene_assign import assign_unique_scene_labels, pick_target_from_scene_assignments


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

# UR3 wrist_3 has no position limits (continuous rotation); real encoders accumulate 2π wraps.
DEFAULT_CONTINUOUS_JOINT_NAMES = frozenset({"wrist_3_joint"})


def _unwrap_joint_angle_to_current(target: float, current: float) -> float:
    """Return the 2π-equivalent of *target* that is closest to *current*."""
    two_pi = 2.0 * math.pi
    return float(target) + round((float(current) - float(target)) / two_pi) * two_pi


# Smooth Servo preset: same lateral geometry as Cartesian centering (base nudge → command frame),
# not raw pinhole v_cam — MoveIt Servo often uses inverse Jacobian and mis-realizes pure optical twists
# (symptom: |Δu| stuck while max_v saturates, |Δv| drifts from mask jitter).
_YOLO_SMOOTH_SERVO_PRESET: Dict[str, Any] = {
    "yolo_servo_use_optical_ibvs": False,
    # Must stay +1: -1 inverted Servo twists vs _tool_pose_lateral_nudge_from_pixels (↑ |Δpx| drift).
    "yolo_servo_ibvs_sign": 1.0,
    "yolo_servo_cartesian_fallback": False,
    "yolo_servo_stall_cartesian_sec": 0.0,
    "yolo_servo_cartesian_nudge_every_sec": 0.0,
    "yolo_servo_gain_scale": 0.14,
    "yolo_servo_linear_gain": 0.14,
    "yolo_servo_max_linear_m_s": 0.0055,
    "yolo_servo_offset_ema_alpha": 0.14,
    "yolo_servo_velocity_ema_alpha": 0.18,
    "yolo_servo_max_twist_delta_m_s": 0.0025,
    "yolo_servo_max_pixel_delta_u_px": 12.0,
    "yolo_servo_max_pixel_delta_v_px": 8.0,
    "yolo_servo_cmd_on_keepalive": False,
    "yolo_servo_keepalive_twist_scale": 0.22,
    "yolo_servo_cross_brake_frames": 0,
    "yolo_servo_stable_frames": 5,
    "yolo_servo_rate_hz": 22.0,
    "yolo_servo_fine_pass": False,
    "yolo_servo_enable_fov_cartesian_recovery": False,
    "yolo_servo_warmup_raw_frames": 1,
}


class SequenceInterrupted(Exception):
    """Raised when the operator stops the sequence (Ctrl+C or ~/stop_sequence service)."""


def _merge_vision_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Defaults + optional `vision:` block in ur3_action_sequence.yaml."""
    defaults: Dict[str, Any] = {
        "rgb_topic": "/rl_camera/color",
        "depth_topic": "/rl_camera/depth",
        "camera_info_topic": "/rl_camera/camera_info",
        # Depth + intrinsics match MuJoCo rl_camera; adjust if your hand-eye uses another optical frame.
        "camera_optical_frame": "camera_color_optical_frame",
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
        "approach_max_iterations": 2,
        "approach_step_max_m": 0.11,
        # Centering: "cartesian" (default) = stable optical nudges; "servo" = velocity IBVS (limit-cycles on noisy bbox);
        # "servo_then_cartesian" = short Servo coarse, then Cartesian only (shake-free finish).
        # "moveit_ik" = same YOLO pose goals but MoveIt IK + joint plan only (no straight-line Cartesian; safer vs self-collision).
        # "moveit_one_shot" = one YOLO sample after Look → tool0 goal (bbox center + target_distance_m) → single MoveGroup IK.
        "yolo_visual_center_centering_mode": "cartesian",
        "yolo_visual_center_use_moveit_ik": False,
        "yolo_moveit_one_shot_max_retries": 3,
        # Centering IK: full step only. On fail: separate recovery move (+Z, tool +X roll) → redetect → restart centering.
        "yolo_visual_center_ik_step_scales": [1.0],
        "yolo_visual_center_recovery_max_sessions": 3,
        "yolo_visual_center_recovery_lift_step_m": 0.04,
        "yolo_visual_center_recovery_roll_step_deg": 5.0,
        "yolo_visual_center_recovery_redetect_attempts": 40,
        "yolo_visual_center_recovery_settle_frames": 12,
        # cartesian_then_servo: Cartesian until |Δu|+|Δv| below handoff, then smooth Servo (continuous).
        "yolo_cart_then_servo_handoff_metric_px": 92.0,
        "yolo_cart_then_servo_handoff_max_px": 200.0,
        "yolo_smooth_servo_preset": True,
        "yolo_servo_hybrid_coarse_sec": 6.0,
        "center_max_iterations": 3,
        # Streamed Cartesian centering motions: min(plan ceiling, configured center_max_iterations).
        "yolo_center_max_plan_iterations": 3,
        "yolo_center_step_gain": 0.35,
        "yolo_center_max_step_m": 0.008,
        "yolo_center_max_duration_sec": 60.0,
        "yolo_center_path_segments": 28,
        # cosine: ease-in-out spacing along each nudge chord (softer than uniform linear samples).
        "yolo_center_waypoint_profile": "cosine",
        # Low-pass pixel error before each Cartesian nudge (0 = off); damps YOLO jitter between replans.
        "yolo_center_command_ema_alpha": 0.28,
        "yolo_center_cartesian_max_step_m": 0.002,
        "yolo_center_velocity_scaling": 0.045,
        "yolo_center_acceleration_scaling": 0.04,
        # Joint retiming after /compute_cartesian_path: softer motion for streamed centering paths.
        "yolo_center_retime_min_segment_sec": 0.052,
        "yolo_center_retime_duration_stretch": 1.14,
        # Strict then relaxed IK coverage; MoveIt often returns ~0.35–0.55 for long Cartesian chains near limits.
        "yolo_center_min_fraction": 0.72,
        "yolo_center_min_fraction_relaxed": 0.32,
        "yolo_center_redetect_after_misses": 30,
        "yolo_center_metric_adaptive": True,
        "yolo_center_adaptive_far_metric_px": 128.0,
        # First N streamed lateral centering motions use max "far" step size when bbox is still displaced;
        # afterward adaptive blending settles (two-step-coarse pattern).
        "yolo_center_coarse_nudge_max": 2,
        # Bump max lateral displacement cap toward pinhole-required chord ∝ |(Δ/f)·z| when |Δpix| large.
        "yolo_center_max_step_scale_by_pixel_error": True,
        "yolo_center_max_step_m_hard_abs": 0.075,
        "yolo_center_error_step_fraction": 1.0,
        # Pinhole one-shot: Δu,Δv+f_x,f_y,depth → lateral goal in one Cartesian stream (segments=1 typical).
        "yolo_center_pixel_direct": False,
        "yolo_center_direct_step_gain": 1.0,
        "yolo_center_direct_fraction": 1.0,
        "yolo_center_direct_path_segments": 1,
        "yolo_center_direct_velocity_scaling": 0.11,
        "yolo_center_direct_bypass_command_ema": True,
        "yolo_center_settle_frames": 25,
        "yolo_center_settle_period_sec": 0.06,
        # After a successful centering Cartesian stream: short spin only (heavy settle is for misses/redetect).
        "yolo_center_after_stream_spin_frames": 3,
        "yolo_center_after_stream_spin_period_sec": 0.022,
        "yolo_center_use_raw_detection": True,
        "yolo_center_nudge_sign": 1.0,
        "yolo_center_min_move_m": 0.0003,
        # When lateral centering fails (fixed tool0 orientation), rotate camera in optical frame.
        "yolo_center_enable_orient_fallback": True,
        "yolo_center_orient_max_attempts": 4,
        "yolo_center_orient_step_gain": 0.42,
        "yolo_center_orient_max_step_rad": 0.085,
        "yolo_center_orient_path_segments": 6,
        "yolo_center_orient_sign_u": -1.0,
        "yolo_center_orient_sign_v": -1.0,
        "yolo_center_min_orient_move_rad": 0.0025,
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
        "yolo_servo_ibvs_sign_u": -1.0,
        "yolo_servo_ibvs_sign_v": -1.0,
        "yolo_servo_max_axis_offset_px": 95.0,
        "yolo_servo_fov_pair_min_px": 30.0,
        "yolo_servo_fov_metric_px": 140.0,
        "yolo_servo_fov_recovery_cooldown_sec": 5.0,
        # False: no mid-servo Cartesian FOV moves (major shake source); end fallback still runs.
        "yolo_servo_enable_fov_cartesian_recovery": False,
        # Max |Δpx| per axis per tick on the *commanded* offset (after EMA). Stops mask flip hunting.
        "yolo_servo_max_pixel_delta_px": 14.0,
        "yolo_servo_max_pixel_delta_u_px": 24.0,
        "yolo_servo_max_pixel_delta_v_px": 9.0,
        # IBVS stalls (Jacobian/no motion): bump with Cartesian using proven geometry.
        "yolo_servo_stall_cartesian_sec": 2.8,
        "yolo_servo_stall_min_metric_px": 55.0,
        "yolo_servo_stall_improve_px": 12.0,
        "yolo_servo_stall_refresh_px": 5.0,
        "yolo_servo_stall_cartesian_cooldown_sec": 3.5,
        # Keep-alive replays stale bbox — damp servo so we don't drive the target off-screen.
        "yolo_servo_keepalive_twist_scale": 0.35,
        "yolo_servo_axis_ramp_px": 70.0,
        "yolo_servo_velocity_in_base": True,
        "yolo_servo_cartesian_fallback": True,
        "yolo_servo_rate_hz": 25.0,
        "yolo_servo_max_duration_sec": 90.0,
        "yolo_servo_use_raw_only": False,
        "yolo_servo_use_smoothed_for_twist": True,
        "yolo_servo_offset_ema_alpha": 0.12,
        "yolo_servo_velocity_ema_alpha": 0.12,
        "yolo_servo_max_twist_delta_m_s": 0.004,
        "yolo_servo_cartesian_speed_scale": 0.45,
        "yolo_servo_diverge_correct_after": 0,
        "yolo_servo_cartesian_nudge_every_sec": 0.0,
        "yolo_servo_diverge_min_metric_px": 120.0,
        "yolo_servo_match_cartesian_gain": False,
        "yolo_servo_gain_scale": 0.22,
        "yolo_servo_linear_gain": 0.35,
        # Applies to Cartesian-nudge→Servo twists only (+1 aligns with streamed Cartesian geometry).
        "yolo_servo_ibvs_sign": 1.0,
        "yolo_servo_cartesian_nudge_min_metric_px": 80.0,
        "yolo_servo_diverge_abort_streak": 0,
        "yolo_servo_cmd_on_keepalive": True,
        "yolo_servo_slow_error_px": 100.0,
        "yolo_servo_near_center_ramp_px": 45.0,
        "yolo_servo_near_center_min_scale": 0.12,
        "yolo_servo_settle_ramp_px": 0.0,
        "yolo_servo_settle_min_scale": 0.08,
        "yolo_servo_deadband_px": 10.0,
        "yolo_servo_center_hysteresis_px": 6.0,
        "yolo_servo_center_on_smoothed": True,
        "yolo_servo_fine_pass": True,
        "yolo_servo_fine_pass_max_duration_sec": 20.0,
        "yolo_servo_cross_brake_frames": 2,
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
        # After centering, interpret approach as "move up to this many metres closer along optical +Z"
        # (depth decreases). Goal depth = max(z_now - forward_m, floors). 0 = use target_distance_m only.
        "yolo_visual_center_approach_forward_m": 0.0,
        # With approach_forward_m: also enforce target_distance_m as minimum standoff (depth floor).
        "yolo_visual_center_approach_target_as_floor": True,
        # yolo_visual_center_approach_partial_ok: unset → true when approach_forward_m > 0, else false.
        # True: at most one look-at joint move, then one approach Cartesian move (no iterative servo stripes).
        "yolo_visual_center_single_motion": False,
        # Single-motion caps how far optical Z translates in one approach hop (metres).
        "yolo_visual_center_single_motion_max_translation_m": 2.0,
        # Ultralytics predict() conf= (lower keeps more boxes; we still require score >= min_confidence).
        "yolo_predict_conf_floor": 0.01,
        # Match YOLO class name to target_class case-insensitively (training export often differs).
        "yolo_class_match_case_insensitive": True,
        # If non-empty: only one of these classes is expected in frame; pick the highest-confidence box among them
        # (reduces wrong-label duplicates, e.g. Cylinder_1 vs Cylinder_2 on the same object).
        "yolo_exclusive_scene_classes": [],
        # When several boxes match target_class (label confusion): max_conf | closest_to_uv | scene_unique.
        "yolo_pick_among_strategy": "scene_unique",
        # One detection per class in scene; compare Cylinder_1 vs Cylinder_2 scores per object cluster.
        "yolo_scene_unique_classes": True,
        "yolo_scene_assign_rescore_crops": True,
        "yolo_scene_assign_cluster_iou": 0.45,
        # Per-class image hint [u, v] at Look pose; used with closest_to_uv when multiple target boxes.
        "yolo_pick_preferred_uv_by_class": {},
        "yolo_depth_valid_min_m": 0.03,
        "yolo_depth_valid_max_m": 12.0,
        "yolo_depth_min_bbox_pixels": 12,
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
        "yolo_tracking_median_window": 9,
        "yolo_tracking_ema_alpha": 0.35,
        "yolo_tracking_keep_alive_max_misses": 15,
        "yolo_tracking_log_keep_alive_every": 4,
    }
    vis = config.get("vision")
    if isinstance(vis, dict):
        defaults.update(vis)
    return defaults


def _merge_foundation_pose_config(config: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        # After yolo_visual_center: call bridge + wait for Isaac TF (camera→object typical).
        "bridge_trigger_service": "/foundation_pose_bridge/trigger",
        "bridge_set_parameters_service": "/foundation_pose_bridge/set_parameters",
        "mesh_package": "ur3_mujoco_sim",
        "mesh_relative_by_class": {
            "Cylinder_1": "meshes/foundationpose/cylinder_1.obj",
            "Cylinder_2": "meshes/foundationpose/cylinder_2.obj",
            "cylinder_1": "meshes/foundationpose/cylinder_1.obj",
            "cylinder_2": "meshes/foundationpose/cylinder_2.obj",
            "Box_1": "meshes/foundationpose/square_1.obj",
            "box_1": "meshes/foundationpose/square_1.obj",
        },
        # Isaac publishes this child frame relative to optical/camera frames (launch default fp_object).
        "object_tf_child_frame": "fp_object",
        # TF wait target (parent chain must exist: usually base→…→camera→fp_object).
        "tf_reference_frame": "base_link",
        "wait_tf_timeout_sec": 45.0,
        # Extra time after Isaac first publishes pose (cold TensorRT/graph bring-up).
        "tf_spin_period_sec": 0.03,
        "record_bbox_snapshot_on_yolo_complete": True,
        # If true (and YAML didn't add a follow-up cartesian), orient tool Z with world look-at for grasp prep.
        "include_grasp_prep_orientation": False,
        "grasp_prep_lookat_vector": [0.0, 0.0, -1.0],
        "grasp_prep_tcp_roll_rad": 0.0,
        # Isaac FoundationPose symmetry_axes (mesh frame). y_full = full rotation about mesh +Y.
        "symmetry_axes_by_class": {
            "Cylinder_1": ["y_full"],
            "Cylinder_2": ["y_full"],
            "cylinder_1": ["y_full"],
            "cylinder_2": ["y_full"],
            "Box_1": [],
            "box_1": [],
            "square_1": [],
        },
        # Map mesh/CAD names (square_1) to YOLO class labels (Box_1).
        "yolo_class_aliases": {"square_1": "Box_1", "box_1": "Box_1"},
        "bridge_trigger_min_confidence": 0.02,
        "sync_vision_camera_topics": True,
        "use_latched_bbox_on_trigger": True,
        # foundation_pose_grasp: auto-launch bridge + Isaac, wait for stable fp_object, MoveIt IK.
        "auto_launch_stack": True,
        "stack_launch_timeout_sec": 120.0,
        "bridge_post_launch_settle_sec": 2.0,
        "bridge_camera_warmup_timeout_sec": 12.0,
        "bridge_set_parameters_service_wait_sec": 25.0,
        "bridge_set_parameters_timeout_sec": 25.0,
        "bridge_set_parameters_timeout_fresh_sec": 60.0,
        "bridge_set_parameters_retries": 4,
        "bridge_set_parameters_retry_pause_sec": 3.0,
        "bridge_skip_yolo_model_path_push": True,
        "bridge_trigger_retry_period_sec": 0.25,
        "launch_isaac": True,
        "stack_launch_file": "foundation_pose_stack.launch.py",
        "grasp_config_file": "",
        "wait_tf_stable": True,
        "tf_stable_sample_count": 20,
        "tf_stable_sample_period_sec": 0.12,
        "tf_stable_max_position_std_m": 0.003,
        "tf_stable_max_rotation_deg": 1.5,
        "tf_stable_max_yaw_std_deg": 2.0,
        "tf_stable_warmup_sec": 3.0,
        "tf_stable_min_elapsed_sec": 6.0,
        "tf_stable_timeout_sec": 45.0,
        "fp_post_trigger_settle_sec": 4.0,
        # MoveIt IK link for foundation_pose_grasp (default: child_frame from gripper_tip.yaml).
        "ik_link_name": "",
        # Two-stage grasp: hover (large standoff) then Cartesian plunge (minimal standoff).
        "hover_approach_standoff_m": 0.12,
        "touch_approach_standoff_m": 0.01,
        "hover_ik_step_scales": [1.0, 0.5, 0.25],
        "hover_convergence_max_iterations": 5,
        "hover_convergence_settle_sec": 0.12,
        "plunge_min_fraction": 1.0,
        "plunge_max_step_m": 0.002,
        "plunge_cartesian_mode": "linear",
        # Stop Isaac GPU stack when foundation_pose / foundation_pose_grasp actions finish.
        "stop_stack_after_action": True,
        # Also stop on sequencer exit (Ctrl+C / sequence done) if we launched the stack.
        "stop_stack_on_sequencer_exit": True,
    }
    blk = config.get("foundation_pose")
    if isinstance(blk, dict):
        merged = dict(defaults)
        merged.update(blk)
        if isinstance(blk.get("mesh_relative_by_class"), dict):
            mm = dict(defaults["mesh_relative_by_class"])  # type: ignore[list-item]
            mm.update(dict(blk["mesh_relative_by_class"]))
            merged["mesh_relative_by_class"] = mm
        if isinstance(blk.get("symmetry_axes_by_class"), dict):
            sm = dict(defaults["symmetry_axes_by_class"])  # type: ignore[list-item]
            sm.update(dict(blk["symmetry_axes_by_class"]))
            merged["symmetry_axes_by_class"] = sm
        return merged
    return dict(defaults)


def _merge_gripper_config(config: Dict[str, Any]) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {
        "hybrid_close_service": "/mujoco/gripper/hybrid_close",
        "hybrid_open_service": "/mujoco/gripper/hybrid_open",
        "set_parameters_service": "/mujoco_node/set_parameters",
        "close_position_rad": 1.396,
        "hold_torque_nm": 2.5,
        "approach_ramp_rad_s": 0.35,
        "min_feedback_torque_nm": 2.0,
        "max_joint_speed_rad_s": 0.15,
        "contact_confirm_steps": 5,
        "hold_sec": 2.0,
        "timeout_sec": 45.0,
        "position_kp": 500.0,
        "position_kv": 30.0,
        "service_wait_timeout_sec": 10.0,
        "service_call_timeout_sec": 60.0,
        # Real UR3: XM540 on U2D2 (see ~/ur3_control/gripper/dynamixel_adaptive_grasp.py)
        "dynamixel_grasp_script": str(
            Path.home() / "ur3_control" / "gripper" / "dynamixel_adaptive_grasp.py"
        ),
        "dynamixel_device": (
            "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT891L5D-if00-port0"
        ),
        "dynamixel_baudrate": 1_000_000,
        "dynamixel_id": 1,
        "dynamixel_hold_time": 5.0,
        "dynamixel_torque_off_after_hold": False,
        "dynamixel_require_contact": False,
        "dynamixel_interactive_prompt": False,
        "dynamixel_pre_close_position": 0,
        "dynamixel_skip_pre_close_move": False,
        "dynamixel_open_position": 0,
        "dynamixel_profile_velocity": 30,
        "dynamixel_profile_acceleration": 10,
        "dynamixel_open_profile_velocity": 45,
        "dynamixel_open_profile_acceleration": 12,
        "dynamixel_move_timeout_sec": 10.0,
        "dynamixel_position_tolerance": 20,
        "dynamixel_torque_off_after_move": False,
    }
    blk = config.get("gripper")
    if isinstance(blk, dict):
        merged = dict(defaults)
        merged.update(blk)
        return merged
    return dict(defaults)


def _resolve_yolo_pick_among_kwargs(
    cfg: Dict[str, Any],
    target_class: str,
) -> Dict[str, Any]:
    """Resolve multi-box pick strategy from action/vision cfg (per-class UV hints supported)."""
    strategy = str(cfg.get("yolo_pick_among_strategy", "max_conf") or "max_conf").strip().lower()
    preferred_uv: Optional[tuple[float, float]] = None
    raw_uv = cfg.get("yolo_pick_preferred_uv")
    if isinstance(raw_uv, (list, tuple)) and len(raw_uv) >= 2:
        preferred_uv = (float(raw_uv[0]), float(raw_uv[1]))
    else:
        by_class = cfg.get("yolo_pick_preferred_uv_by_class") or {}
        if isinstance(by_class, dict):
            hint = by_class.get(target_class)
            if hint is None:
                hint = by_class.get(target_class.strip())
            if hint is None and target_class.strip():
                hint = by_class.get(target_class.strip().lower())
            if isinstance(hint, (list, tuple)) and len(hint) >= 2:
                preferred_uv = (float(hint[0]), float(hint[1]))
    return {"pick_among_strategy": strategy, "preferred_uv": preferred_uv}


def _resolve_yolo_scene_assign_kwargs(
    vision_cfg: Dict[str, Any],
    action_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Scene-unique assignment options + full class list (even when action filters to one target)."""
    merged = {**vision_cfg}
    if action_cfg:
        merged.update(action_cfg)
    strategy = str(merged.get("yolo_pick_among_strategy", "max_conf") or "max_conf").strip().lower()
    scene_unique = bool(merged.get("yolo_scene_unique_classes", False)) or strategy == "scene_unique"
    assign_classes: list[str] = []
    raw_assign = merged.get("yolo_scene_assign_classes")
    if isinstance(raw_assign, (list, tuple)):
        assign_classes = [str(x).strip() for x in raw_assign if str(x).strip()]
    if len(assign_classes) < 2:
        raw_v = vision_cfg.get("yolo_exclusive_scene_classes")
        if isinstance(raw_v, (list, tuple)):
            tmp = [str(x).strip() for x in raw_v if str(x).strip()]
            if len(tmp) >= 2:
                assign_classes = tmp
    if len(assign_classes) < 2:
        raw_excl = merged.get("yolo_exclusive_scene_classes")
        if isinstance(raw_excl, (list, tuple)):
            tmp = [str(x).strip() for x in raw_excl if str(x).strip()]
            if len(tmp) >= 2:
                assign_classes = tmp
    return {
        "scene_unique_classes": scene_unique,
        "scene_rescore_crops": bool(merged.get("yolo_scene_assign_rescore_crops", True)),
        "scene_cluster_iou": float(merged.get("yolo_scene_assign_cluster_iou", 0.45)),
        "scene_assign_exclusive_classes": assign_classes,
        "pick_among_strategy": strategy,
    }


def _yolo_bbox_center_uv(box_xyxy: Any) -> tuple[float, float]:
    bb = np.asarray(box_xyxy, dtype=np.float64).reshape(-1)
    return (float(bb[0] + bb[2]) * 0.5, float(bb[1] + bb[3]) * 0.5)


def _yolo_pick_target_match(
    boxes: Any,
    target_matches: list[tuple[float, int, str]],
    *,
    min_conf: float,
    pick_among_strategy: str = "max_conf",
    preferred_uv: Optional[tuple[float, float]] = None,
    log_pick_fn: Optional[Callable[[str], None]] = None,
) -> Optional[tuple[np.ndarray, float, str]]:
    """Pick one box among several that match ``target_class``."""
    viable = [(cf, i, cname) for cf, i, cname in target_matches if cf >= min_conf]
    if not viable:
        return None
    if len(viable) > 1 and log_pick_fn is not None:
        parts = [f"{cname}@{cf:.2f}" for cf, _, cname in sorted(viable, key=lambda x: -x[0])]
        log_pick_fn(
            f"yolo pick: {len(viable)} box(es) match target_class "
            f"(strategy={pick_among_strategy!r}): [{', '.join(parts)}]"
        )
    strategy = (pick_among_strategy or "max_conf").strip().lower()
    if len(viable) > 1 and strategy == "closest_to_uv" and preferred_uv is not None:
        pu, pv = preferred_uv

        def _dist2(entry: tuple[float, int, str]) -> float:
            cu, cv = _yolo_bbox_center_uv(boxes.xyxy[entry[1]].cpu().numpy())
            du, dv = cu - pu, cv - pv
            return du * du + dv * dv

        cf_w, i_w, cname_w = min(viable, key=_dist2)
    else:
        if len(viable) > 1 and strategy == "closest_to_uv" and preferred_uv is None and log_pick_fn is not None:
            log_pick_fn(
                "yolo pick: closest_to_uv requested but yolo_pick_preferred_uv unset; using max_conf."
            )
        cf_w, i_w, cname_w = max(viable, key=lambda x: x[0])
    return boxes.xyxy[i_w].cpu().numpy(), cf_w, cname_w


def _ultralytics_pick_bbox_xyxy_and_conf(
    results: Any,
    *,
    target_class: str,
    min_conf: float,
    class_match_case_insensitive: bool,
    exclusive_scene_classes: Optional[list[str]],
    pick_among_strategy: str = "max_conf",
    preferred_uv: Optional[tuple[float, float]] = None,
    scene_unique_classes: bool = False,
    scene_rescore_crops: bool = True,
    scene_cluster_iou: float = 0.45,
    scene_assign_exclusive_classes: Optional[list[str]] = None,
    model: Any = None,
    rgb: Optional[np.ndarray] = None,
    predict_conf_floor: float = 0.01,
    yolo_iou: float = 0.5,
    log_exclusive_fn: Optional[Callable[[str], None]] = None,
    log_pick_fn: Optional[Callable[[str], None]] = None,
) -> Optional[tuple[np.ndarray, float, str]]:
    """
    Match ``_yolo_detection_center_ray`` box selection semantics.
    Returns (xyxy ndarray shape (4,), conf, winning_class_label) or None.
    """
    exclusive_list: Optional[list[str]] = None
    if isinstance(exclusive_scene_classes, (list, tuple)):
        tmp = [str(x).strip() for x in exclusive_scene_classes if str(x).strip()]
        if tmp:
            exclusive_list = tmp

    strategy = (pick_among_strategy or "max_conf").strip().lower()
    use_scene_unique = bool(scene_unique_classes) or strategy == "scene_unique"
    assign_list = scene_assign_exclusive_classes if scene_assign_exclusive_classes else exclusive_list
    if use_scene_unique and assign_list and len(assign_list) >= 2 and rgb is not None:
        assigned = assign_unique_scene_labels(
            results,
            exclusive_scene_classes=assign_list,
            rgb=rgb,
            model=model,
            case_insensitive=class_match_case_insensitive,
            predict_conf_floor=float(predict_conf_floor),
            yolo_iou=float(yolo_iou),
            cluster_iou=float(scene_cluster_iou),
            rescore_crops=bool(scene_rescore_crops),
            log_fn=log_pick_fn,
        )
        picked_scene = pick_target_from_scene_assignments(
            assigned,
            target_class=target_class,
            min_conf=min_conf,
            case_insensitive=class_match_case_insensitive,
        )
        if picked_scene is not None:
            return picked_scene
        if log_exclusive_fn is not None:
            seen = sorted({d.class_name for d in assigned})
            log_exclusive_fn(
                "yolo scene unique: "
                f"target_class={target_class!r} not assigned among {seen!r} "
                f"(min_conf={min_conf:.3f})."
            )
        return None

    best_conf = -1.0
    best_box: Optional[np.ndarray] = None
    win_label = ""
    names = results.names
    boxes = results.boxes
    target_cmp = target_class.strip()
    if class_match_case_insensitive:
        target_cmp = target_cmp.lower()

    if class_match_case_insensitive:
        exclusive_norm = {x.lower() for x in exclusive_list} if exclusive_list else set()
    else:
        exclusive_norm = set(exclusive_list) if exclusive_list else set()

    if exclusive_norm and boxes is not None and boxes.cls is not None and len(boxes) > 0:
        candidates: list[tuple[float, int, str]] = []
        for i in range(len(boxes)):
            cid = int(boxes.cls[i].item())
            cf = float(boxes.conf[i].item())
            cname = str(names[cid]).strip()
            key = cname.lower() if class_match_case_insensitive else cname
            if key in exclusive_norm:
                candidates.append((cf, i, cname))
        if candidates:
            target_matches = [
                (cf, i, cname)
                for cf, i, cname in candidates
                if (cname.lower() if class_match_case_insensitive else cname) == target_cmp
            ]
            if target_matches:
                picked_tm = _yolo_pick_target_match(
                    boxes,
                    target_matches,
                    min_conf=min_conf,
                    pick_among_strategy=pick_among_strategy,
                    preferred_uv=preferred_uv,
                    log_pick_fn=log_pick_fn,
                )
                if picked_tm is not None:
                    best_box, best_conf, win_label = picked_tm
            elif log_exclusive_fn is not None:
                others = sorted({cname for _, _, cname in candidates})
                log_exclusive_fn(
                    "yolo exclusive scene: "
                    f"target_class={target_class!r} not among detections {others!r}; "
                    "not using other exclusive classes."
                )

    if best_box is None and boxes is not None and boxes.cls is not None:
        fallback_matches: list[tuple[float, int, str]] = []
        for i in range(len(boxes)):
            cid = int(boxes.cls[i].item())
            cf = float(boxes.conf[i].item())
            cname = str(names[cid]).strip()
            if class_match_case_insensitive:
                if cname.lower() != target_cmp:
                    continue
            elif cname != target_class.strip():
                continue
            fallback_matches.append((cf, i, cname))
        picked_fb = _yolo_pick_target_match(
            boxes,
            fallback_matches,
            min_conf=min_conf,
            pick_among_strategy=pick_among_strategy,
            preferred_uv=preferred_uv,
            log_pick_fn=log_pick_fn,
        )
        if picked_fb is not None:
            best_box, best_conf, win_label = picked_fb

    if best_box is None or best_conf < min_conf:
        return None
    return best_box, best_conf, win_label


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


def _duration_to_seconds(duration: DurationMsg) -> float:
    return float(duration.sec) + float(duration.nanosec) * 1e-9


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


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation; quaternions are [x, y, z, w]."""
    t = float(max(0.0, min(1.0, t)))
    q0 = np.asarray(q0, dtype=np.float64).reshape(4)
    q1 = np.asarray(q1, dtype=np.float64).reshape(4)
    q0 = q0 / max(float(np.linalg.norm(q0)), 1e-12)
    q1 = q1 / max(float(np.linalg.norm(q1)), 1e-12)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        out = q0 + t * (q1 - q0)
        return out / max(float(np.linalg.norm(out)), 1e-12)
    theta_0 = math.acos(dot)
    sin_theta_0 = math.sin(theta_0)
    theta = theta_0 * t
    s0 = math.sin(theta_0 - theta) / sin_theta_0
    s1 = math.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


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
        self.declare_parameter("sequence_name", "Cylinder_2")
        self.declare_parameter("object_name", "")
        self.declare_parameter("prompt_for_object", True)
        self.declare_parameter("continue_on_failure", False)
        self.declare_parameter("stop_sequence_service", "~/stop_sequence")
        self.declare_parameter("publish_action_completed_topic", "")
        self.declare_parameter("speed_scale", 1.0, ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("robot_backend", "", ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("foundation_pose_tcp_roll_deg", "", ParameterDescriptor(dynamic_typing=True))

        self.config_file = str(self.get_parameter("config_file").value)
        self.sequence_name = str(self.get_parameter("sequence_name").value)
        self.object_name = str(self.get_parameter("object_name").value).strip()
        self.prompt_for_object = bool(self.get_parameter("prompt_for_object").value)
        self.continue_on_failure = bool(self.get_parameter("continue_on_failure").value)

        self.config = _load_yaml(self.config_file)
        self.sequence_names = self._select_sequence_names(self.config, self.sequence_name)
        self.sequence_name = " -> ".join(self.sequence_names)
        self.motion_cfg = self._load_motion_config(self.config)
        backend_param = str(self.get_parameter("robot_backend").value or "").strip().lower()
        self.robot_backend = backend_param or str(self.motion_cfg.get("robot_backend", "sim")).strip().lower()
        if self.robot_backend not in {"sim", "real"}:
            raise ValueError("robot_backend must be 'sim' or 'real'")
        self.get_logger().info(f"Robot backend: {self.robot_backend}")
        speed_param = self.get_parameter("speed_scale").value
        motion_speed = self.motion_cfg.get("speed_scale", 1.0)
        self.speed_scale = self._normalize_speed_scale(
            speed_param if speed_param is not None else motion_speed
        )
        self.get_logger().info(
            f"Sequence speed_scale={self.speed_scale:.3f} "
            "(multiplies normal axis-position moves only; YOLO, FoundationPose, and gripper keep YAML speeds)."
        )
        self._forbidden_zone_presets = MotionManager._load_named_preset_map(self.config, "forbidden_zone_presets")
        self._planning_attach_presets = MotionManager._load_named_preset_map(self.config, "planning_attach_presets")
        self._planning_detach_presets = MotionManager._load_named_preset_map(self.config, "planning_detach_presets")
        self._mujoco_weld_presets = MotionManager._load_named_preset_map(self.config, "mujoco_weld_presets")
        self.sequence = self._load_sequences(self.config, self.sequence_names)
        self.joint_names = [str(name) for name in self.motion_cfg.get("joint_names", DEFAULT_JOINT_NAMES)]
        tcp_yaml = _load_yaml(str(self.motion_cfg["tcp_file"]))
        self.tool_to_tcp = _transform_from_yaml(tcp_yaml)
        self.tcp_frame_name = str(tcp_yaml.get("child_frame", "gripper_tip")).strip() or "gripper_tip"
        self.vision_cfg = _merge_vision_config(self.config)
        if self.robot_backend == "real":
            self.motion_cfg.setdefault("avoid_collisions", True)
            self.get_logger().info("Real backend: MoveIt avoid_collisions=true for all planning.")
            self.motion_cfg.setdefault("real_cartesian_retime_min_segment_sec", 0.08)
            self.motion_cfg.setdefault("real_cartesian_retime_duration_stretch", 2.0)
            self.motion_cfg.setdefault("cartesian_retime_max_joint_vel_rad_s", 0.35)
            real_vision = self.config.get("real_vision")
            if isinstance(real_vision, dict):
                self.vision_cfg.update(real_vision)
            self.get_logger().info(
                "Real vision topics: "
                f"rgb={self.vision_cfg.get('rgb_topic')}, "
                f"depth={self.vision_cfg.get('depth_topic')}, "
                f"camera_info={self.vision_cfg.get('camera_info_topic')}"
            )
        self.foundation_pose_cfg = _merge_foundation_pose_config(self.config)
        self._apply_cli_foundation_pose_tcp_roll()
        self.gripper_cfg = _merge_gripper_config(self.config)
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
        self._foundation_pose_param_clients: dict[str, Any] = {}
        self._fp_last_bbox_xyxy: Optional[tuple[float, float, float, float]] = None
        self._fp_last_detection_label: str = ""
        self._fp_last_z_ray_m: float = 0.0
        self._fp_stack = FoundationPoseStackLauncher(self.get_logger())
        self._grasp_by_class_cache: Dict[str, Any] = {}
        self._grasp_by_class_cache_key: str = ""

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

    def cleanup(self) -> None:
        if self._yolo_detection_is_armed():
            self._disarm_yolo_detection()
        fp_stack = getattr(self, "_fp_stack", None)
        if fp_stack is not None and bool(self.foundation_pose_cfg.get("stop_stack_on_sequencer_exit", True)):
            if fp_stack.stack_active():
                self.get_logger().info(
                    "foundation_pose: stopping stack on sequencer exit (stop_stack_on_sequencer_exit=true)."
                )
                fp_stack.stop_completely()
        listener = getattr(self, "_tf_listener", None)
        if listener is None:
            return
        executor = getattr(listener, "executor", None)
        if executor is not None:
            executor.shutdown()
        thread = getattr(listener, "dedicated_listener_thread", None)
        if thread is not None:
            thread.join(timeout=2.0)
        try:
            listener.unregister()
        except Exception:  # noqa: BLE001 - best effort during shutdown.
            pass
        self._tf_listener = None

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
            if msg.encoding == "32FC1":
                d = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            elif msg.encoding == "16UC1":
                # RealSense aligned depth (millimeters → meters).
                raw = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
                d = raw.astype(np.float32) * 0.001
            else:
                return
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

    def _yolo_detection_is_armed(self) -> bool:
        return bool(self._vision_subscriptions)

    def _arm_yolo_detection(self, *, preload_model: bool = True) -> None:
        """
        Subscribe to camera topics and optionally load YOLO weights.

        Use per-action YAML ``yolo_detection_enable`` (typically on Look, after the joint
        move completes and before ``delay_after_sec`` warmup) so detection is ready before
        ``yolo_visual_center`` without running inference for the whole sequence.
        """
        rgb_t = str(self.vision_cfg["rgb_topic"])
        depth_t = str(self.vision_cfg["depth_topic"])
        info_t = str(self.vision_cfg["camera_info_topic"])
        if not self._vision_subscriptions:
            self._vision_subscriptions = [
                self.create_subscription(Image, rgb_t, self._on_vision_rgb, 1),
                self.create_subscription(Image, depth_t, self._on_vision_depth, 1),
                self.create_subscription(CameraInfo, info_t, self._on_vision_info, 1),
            ]
            self.get_logger().info(
                f"YOLO detection armed ({rgb_t}, {depth_t}, {info_t})."
            )
        else:
            self.get_logger().debug("YOLO detection already armed.")
        self._clear_vision_buffers()
        if preload_model:
            model_path = str(self.vision_cfg.get("yolo_model_path", "")).strip()
            if model_path:
                self._get_yolo_model(model_path)

    def _disarm_yolo_detection(self) -> None:
        """Release camera subscriptions and unload YOLO weights (see ``yolo_detection_disable`` in YAML)."""
        if self._vision_subscriptions:
            for sub in self._vision_subscriptions:
                self.destroy_subscription(sub)
            self._vision_subscriptions = []
        self._clear_vision_buffers()
        if self._yolo_model is not None:
            self._yolo_model = None
            self._yolo_loaded_path = ""
        self.get_logger().info("YOLO detection disarmed (camera subscriptions released, model unloaded).")

    def _ensure_vision_subscriptions_for_yolo(self) -> None:
        """Legacy entry: arm vision if a yolo_visual_center action runs without prior Look enable."""
        self._arm_yolo_detection()

    def _yolo_detection_lifecycle_before_action(self, action: Dict[str, Any]) -> None:
        if bool(action.get("yolo_detection_enable_before", False)):
            self._arm_yolo_detection()
        if bool(action.get("yolo_detection_disable_before", False)):
            self._disarm_yolo_detection()

    def _yolo_detection_lifecycle_after_action(self, action: Dict[str, Any], *, success: bool) -> None:
        if bool(action.get("yolo_detection_enable", False)):
            if success or bool(action.get("yolo_detection_enable_on_failure", False)):
                self._arm_yolo_detection()
        if bool(action.get("yolo_detection_disable", False)):
            always = bool(action.get("yolo_detection_disable_always", True))
            if success or always:
                self._disarm_yolo_detection()

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

    def _foundation_pose_record_bbox_snapshot_after_yolo(
        self,
        model: Any,
        *,
        target_class: str,
        min_conf: float,
        yolo_iou: float,
        ray_kw: Dict[str, Any],
        depth_z_med: float,
    ) -> None:
        """Store last XYXY + label after successful yolo_visual_center (foundation_pose debug / parity with bridge)."""
        if YOLO is None or model is None:
            return
        if not bool(self.foundation_pose_cfg.get("record_bbox_snapshot_on_yolo_complete", True)):
            return
        if not self._wait_vision_frames(0.05):
            return
        predict_floor = float(ray_kw.get("predict_conf_floor", self.vision_cfg.get("yolo_predict_conf_floor", 0.01)))
        ci = bool(
            ray_kw.get(
                "class_match_case_insensitive",
                self.vision_cfg.get("yolo_class_match_case_insensitive", True),
            )
        )
        ex_raw = ray_kw.get("exclusive_scene_classes", self.vision_cfg.get("yolo_exclusive_scene_classes"))
        ex_list = None
        if isinstance(ex_raw, (list, tuple)):
            ex_list = [str(x).strip() for x in ex_raw if str(x).strip()]

        with self._vis_lock:
            bgr = self._vis_bgr.copy() if self._vis_bgr is not None else None
        if bgr is None:
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        infer_conf = float(min(max(1e-6, predict_floor), float(min_conf)))
        results = model.predict(rgb, conf=infer_conf, iou=float(yolo_iou), verbose=False)[0]
        picked = _ultralytics_pick_bbox_xyxy_and_conf(
            results,
            target_class=target_class,
            min_conf=float(min_conf),
            class_match_case_insensitive=ci,
            exclusive_scene_classes=ex_list,
            pick_among_strategy=str(
                ray_kw.get(
                    "pick_among_strategy",
                    self.vision_cfg.get("yolo_pick_among_strategy", "max_conf"),
                )
            ),
            preferred_uv=ray_kw.get("preferred_uv"),
            scene_unique_classes=bool(ray_kw.get("scene_unique_classes", False)),
            scene_rescore_crops=bool(ray_kw.get("scene_rescore_crops", True)),
            scene_cluster_iou=float(ray_kw.get("scene_cluster_iou", 0.45)),
            scene_assign_exclusive_classes=ray_kw.get("scene_assign_exclusive_classes"),
            model=model,
            rgb=rgb,
            predict_conf_floor=float(predict_floor),
            yolo_iou=float(yolo_iou),
            log_exclusive_fn=lambda m: self.get_logger().debug(m),
            log_pick_fn=lambda m: self.get_logger().info(m),
        )
        if picked is None:
            return
        best_box, _cf, lab = picked
        bb = np.asarray(best_box, dtype=np.float64).reshape(-1)
        self._fp_last_bbox_xyxy = (float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3]))
        self._fp_last_detection_label = str(lab)
        self._fp_last_z_ray_m = float(depth_z_med)
        self.get_logger().debug(
            f"foundation_pose snapshot: bbox={self._fp_last_bbox_xyxy} label={self._fp_last_detection_label!r} "
            f"z_med={self._fp_last_z_ray_m:.4f}"
        )

    def _foundation_pose_mesh_path_for_class(self, clazz: str) -> Optional[str]:
        key = str(clazz).strip()
        if not key:
            return None
        rel_map = self.foundation_pose_cfg.get("mesh_relative_by_class")
        pkg = str(self.foundation_pose_cfg.get("mesh_package", "")).strip()
        if not isinstance(rel_map, dict) or not pkg:
            return None
        rel = rel_map.get(key)
        if not rel:
            return None
        rel_s = str(rel).strip().lstrip("/")
        try:
            share = get_package_share_directory(pkg)
        except PackageNotFoundError:
            self.get_logger().warn(f"foundation_pose mesh package not found: {pkg!r}")
            return None
        return os.path.join(share, rel_s)

    def _foundation_pose_texture_path_for_mesh(self, mesh_path: Optional[str]) -> Optional[str]:
        if not mesh_path:
            return None
        base, _ext = os.path.splitext(mesh_path)
        for name in (f"{base}_texture.jpg", f"{base}_texture.png"):
            if os.path.isfile(name):
                return name
        return None

    def _foundation_pose_camera_optical_frame(self, fp_cfg: Dict[str, Any]) -> str:
        return str(
            fp_cfg.get("camera_optical_frame")
            or self.vision_cfg.get("yolo_servo_command_frame")
            or "camera_color_optical_frame"
        ).strip()

    def _foundation_pose_use_sim_time(self) -> bool:
        if self.robot_backend == "real":
            return False
        try:
            return bool(self.get_parameter("use_sim_time").value)
        except Exception:  # noqa: BLE001
            return True

    def _foundation_pose_stack_launch_file(self, fp_cfg: Dict[str, Any]) -> Optional[str]:
        name = str(fp_cfg.get("stack_launch_file", "")).strip()
        return name or None

    def _apply_cli_foundation_pose_tcp_roll(self) -> None:
        """CLI / launch param: extra FoundationPose grasp TCP roll offset in degrees."""
        val = self.get_parameter("foundation_pose_tcp_roll_deg").value
        if val is None or val == "":
            return
        if isinstance(val, (int, float)):
            roll_deg = float(val)
        else:
            raw = str(val).strip()
            if not raw:
                return
            try:
                roll_deg = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"foundation_pose_tcp_roll_deg must be a number (got {raw!r})") from exc
        if not math.isfinite(roll_deg):
            raise ValueError(f"foundation_pose_tcp_roll_deg must be finite (got {roll_deg!r})")
        roll_rad = math.radians(roll_deg)
        self.foundation_pose_cfg["tcp_roll_rad"] = roll_rad
        self.get_logger().info(
            f"FoundationPose grasp TCP roll offset from CLI: {roll_deg:.1f}° "
            f"(added to object yaw when sync_object_yaw_as_tcp_roll=true, tcp_roll_rad={roll_rad:.6f})"
        )

    def _foundation_pose_cfg_for_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        cfg_fp = dict(self.foundation_pose_cfg)
        if isinstance(action.get("foundation_pose"), dict):
            cfg_fp.update(dict(action["foundation_pose"]))
        if "stop_stack_after_action" in action:
            cfg_fp["stop_stack_after_action"] = action["stop_stack_after_action"]
        return cfg_fp

    def _foundation_pose_maybe_stop_stack_after_action(
        self, cfg_fp: Dict[str, Any], action: Dict[str, Any]
    ) -> None:
        if not bool(cfg_fp.get("stop_stack_after_action", False)):
            return
        fp_stack = getattr(self, "_fp_stack", None)
        if fp_stack is None:
            return
        name = str(action.get("name", action.get("type", "foundation_pose")))
        self.get_logger().info(
            f"foundation_pose: stopping Isaac stack after action {name!r} "
            "(stop_stack_after_action=true)."
        )
        fp_stack.stop_completely()

    def _foundation_pose_ensure_stack_running(
        self,
        cfg_fp: Dict[str, Any],
        tgt: str,
        *,
        force_relaunch: bool = False,
        tf_upright_in_base: Optional[bool] = None,
        tf_long_axis_in_object: Optional[Sequence[float]] = None,
    ) -> bool:
        if not bool(cfg_fp.get("auto_launch_stack", True)):
            return True
        trig = str(cfg_fp.get("bridge_trigger_service", "/foundation_pose_bridge/trigger")).strip()
        set_params = str(cfg_fp.get("bridge_set_parameters_service", "")).strip()
        ref = str(cfg_fp.get("tf_reference_frame", "base_link")).strip()
        mesh_path = self._foundation_pose_mesh_path_for_class(tgt)
        texture_path = self._foundation_pose_texture_path_for_mesh(mesh_path)
        symmetry_axes = symmetry_axes_for_class(tgt, cfg_fp)
        ok = self._fp_stack.ensure_running(
            self,
            bridge_trigger=trig,
            bridge_set_params=set_params,
            use_sim_time=self._foundation_pose_use_sim_time(),
            stack_launch_file=self._foundation_pose_stack_launch_file(cfg_fp),
            launch_timeout_sec=float(cfg_fp.get("stack_launch_timeout_sec", 120.0)),
            spin_cb=self._spin_cb,
            mesh_file_path=mesh_path,
            texture_file_path=texture_path,
            symmetry_axes=symmetry_axes,
            tf_reference_frame=ref,
            tf_upright_in_base=tf_upright_in_base,
            tf_long_axis_in_object=tf_long_axis_in_object,
            force_relaunch=force_relaunch,
            isaac_rgb_topic=str(cfg_fp.get("isaac_rgb_rect_topic", "/rgb/image_rect_color")),
            isaac_pipeline_ready_timeout_sec=float(
                cfg_fp.get("isaac_pipeline_ready_timeout_sec", 90.0)
            ),
        )
        if not ok:
            stack_file = self._foundation_pose_stack_launch_file(cfg_fp) or "foundation_pose_stack.launch.py"
            self.get_logger().error(
                f"foundation_pose: stack launch failed ({stack_file}) — "
                "Real robot needs RealSense running + Isaac env "
                "(~/isaac_ros_assets/setup_foundationpose_env.sh). "
                "See ~/.ros/log/foundation_pose_stack_latest.log"
            )
        return ok

    def _lookup_fp_object_mat_in_base(
        self,
        ref: str,
        child: str,
        *,
        optical_frame: str,
        lookup_timeout_sec: float = 0.25,
        log_error: bool = False,
    ) -> Optional[np.ndarray]:
        """Resolve object pose in base: direct TF or base←camera←fp_object chain."""
        mat = self._lookup_tf_mat(ref, child, lookup_timeout_sec, log_error=log_error)
        if mat is not None:
            return mat
        t_bo = self._lookup_tf_mat(ref, optical_frame, lookup_timeout_sec, log_error=log_error)
        t_oc = self._lookup_tf_mat(optical_frame, child, lookup_timeout_sec, log_error=log_error)
        if t_bo is not None and t_oc is not None:
            return t_bo @ t_oc
        return None

    def _foundation_pose_set_parameters_client(self, service_name: str) -> Any:
        svc = str(service_name).strip()
        if svc not in self._foundation_pose_param_clients:
            self._foundation_pose_param_clients[svc] = self.create_client(SetParameters, svc)
        return self._foundation_pose_param_clients[svc]

    @staticmethod
    def _foundation_pose_resolve_target_class(raw: str, fp_cfg: Dict[str, Any]) -> str:
        """Map mesh/CAD aliases (e.g. square_1) to YOLO class labels (Box_1)."""
        key = str(raw).strip()
        if not key:
            return key
        aliases = fp_cfg.get("yolo_class_aliases") or {}
        if not isinstance(aliases, dict):
            return key
        if key in aliases:
            return str(aliases[key]).strip()
        key_l = key.lower()
        for alias_key, alias_val in aliases.items():
            if str(alias_key).strip().lower() == key_l:
                return str(alias_val).strip()
        return key

    def _foundation_pose_bridge_push_detection_params(
        self,
        fp_cfg: Dict[str, Any],
        target_class: str,
        *,
        cold_stack: bool = False,
    ) -> bool:
        """Tell foundation_pose_bridge YOLO filter + confidence so Trigger matches centered object."""
        svc = str(fp_cfg.get("bridge_set_parameters_service", "")).strip()
        if not svc:
            self.get_logger().error("foundation_pose: bridge_set_parameters_service is empty")
            return False
        yolo_class = self._foundation_pose_resolve_target_class(target_class, fp_cfg)
        min_conf = float(
            fp_cfg.get(
                "bridge_trigger_min_confidence",
                self.vision_cfg.get(
                    "yolo_visual_center_acquire_min_confidence",
                    self.vision_cfg.get("min_confidence", 0.25),
                ),
            )
        )
        ex_raw = fp_cfg.get("yolo_exclusive_scene_classes", self.vision_cfg.get("yolo_exclusive_scene_classes"))
        ex_list = [str(x).strip() for x in ex_raw] if isinstance(ex_raw, (list, tuple)) else []
        # Never let bridge YOLO compete across classes (e.g. pick Box_1 when user chose Cylinder_1).
        if not ex_list:
            ex_list = [yolo_class]
        elif yolo_class not in ex_list and yolo_class.lower() not in {x.lower() for x in ex_list}:
            ex_list = [yolo_class] + ex_list

        plist: list[Parameter] = []
        pv_tc = ParameterValue()
        pv_tc.type = ParameterType.PARAMETER_STRING
        pv_tc.string_value = yolo_class
        plist.append(Parameter(name="target_class", value=pv_tc))

        pv_cf = ParameterValue()
        pv_cf.type = ParameterType.PARAMETER_DOUBLE
        pv_cf.double_value = float(min_conf)
        plist.append(Parameter(name="min_confidence", value=pv_cf))

        yolo_path = str(self.vision_cfg.get("yolo_model_path", "")).strip()
        if yolo_path and not bool(fp_cfg.get("bridge_skip_yolo_model_path_push", True)):
            pv_ym = ParameterValue()
            pv_ym.type = ParameterType.PARAMETER_STRING
            pv_ym.string_value = yolo_path
            plist.append(Parameter(name="yolo_model_path", value=pv_ym))

        if bool(fp_cfg.get("sync_vision_camera_topics", True)):
            for pname, vkey in (
                ("rgb_topic", "rgb_topic"),
                ("depth_topic", "depth_topic"),
                ("camera_info_topic", "camera_info_topic"),
            ):
                topic = str(self.vision_cfg.get(vkey, "")).strip()
                if topic:
                    pv_t = ParameterValue()
                    pv_t.type = ParameterType.PARAMETER_STRING
                    pv_t.string_value = topic
                    plist.append(Parameter(name=pname, value=pv_t))

        p_floor = float(self.vision_cfg.get("yolo_predict_conf_floor", 0.01))
        pv_pf = ParameterValue()
        pv_pf.type = ParameterType.PARAMETER_DOUBLE
        pv_pf.double_value = p_floor
        plist.append(Parameter(name="predict_conf_floor", value=pv_pf))

        if ex_list:
            pv_sa = ParameterValue()
            pv_sa.type = ParameterType.PARAMETER_STRING_ARRAY
            pv_sa.string_array_value = ex_list
            plist.append(Parameter(name="yolo_exclusive_scene_classes", value=pv_sa))

        use_latch = bool(fp_cfg.get("use_latched_bbox_on_trigger", True))
        latch_lbl = yolo_class
        pv_ul = ParameterValue()
        pv_ul.type = ParameterType.PARAMETER_BOOL
        pv_ul.bool_value = use_latch
        plist.append(Parameter(name="use_latched_bbox_on_trigger", value=pv_ul))
        if use_latch and self._fp_last_bbox_xyxy is not None:
            bb = [float(v) for v in self._fp_last_bbox_xyxy]
            pv_bb = ParameterValue()
            pv_bb.type = ParameterType.PARAMETER_DOUBLE_ARRAY
            pv_bb.double_array_value = bb
            plist.append(Parameter(name="latched_bbox_xyxy", value=pv_bb))
            pv_lbl = ParameterValue()
            pv_lbl.type = ParameterType.PARAMETER_STRING
            latch_lbl = str(self._fp_last_detection_label or yolo_class).strip()
            if latch_lbl.lower() != yolo_class.lower():
                self.get_logger().warn(
                    f"foundation_pose: latched YOLO label {latch_lbl!r} != target {yolo_class!r}; "
                    f"forcing latched_bbox_label={yolo_class!r} for bridge/Isaac."
                )
                latch_lbl = yolo_class
            pv_lbl.string_value = latch_lbl
            plist.append(Parameter(name="latched_bbox_label", value=pv_lbl))

        cli = self._foundation_pose_set_parameters_client(svc)
        wait_svc = float(fp_cfg.get("bridge_set_parameters_service_wait_sec", 20.0))
        if not self._wait_for_service_abortable(cli, wait_svc):
            self.get_logger().error(f"foundation_pose: SetParameters service not ready: {svc}")
            return False

        call_timeout = float(
            fp_cfg.get(
                "bridge_set_parameters_timeout_fresh_sec" if cold_stack else "bridge_set_parameters_timeout_sec",
                60.0 if cold_stack else 20.0,
            )
        )
        retries = max(1, int(fp_cfg.get("bridge_set_parameters_retries", 3)))
        retry_pause = float(fp_cfg.get("bridge_set_parameters_retry_pause_sec", 2.0))

        for attempt in range(1, retries + 1):
            if not cli.service_is_ready():
                if not self._wait_for_service_abortable(cli, wait_svc):
                    self.get_logger().error(f"foundation_pose: SetParameters service not ready: {svc}")
                    return False
            future = cli.call_async(SetParameters.Request(parameters=plist))
            self._spin_until_future_complete_abortable(future, call_timeout)
            if not future.done():
                self.get_logger().warn(
                    f"foundation_pose: SetParameters attempt {attempt}/{retries} timed out "
                    f"({call_timeout:.0f}s on {svc})."
                )
                if attempt < retries:
                    self._spin_cb(retry_pause)
                    continue
                self.get_logger().error(
                    f"foundation_pose: SetParameters failed after {retries} attempt(s) ({svc}). "
                    "Bridge may still be loading YOLO/Isaac — try pkill -f foundation_pose_stack."
                )
                return False
            res = future.result()
            if res is None:
                if attempt < retries:
                    self._spin_cb(retry_pause)
                    continue
                return False
            rejected = False
            for r in getattr(res, "results", []) or []:
                if hasattr(r, "successful") and not bool(r.successful):
                    msg = getattr(r, "reason", "")
                    self.get_logger().error(f"foundation_pose: parameter rejected ({svc}): {msg}")
                    rejected = True
            if rejected:
                return False
            break
        latch_note = ""
        if use_latch and self._fp_last_bbox_xyxy is not None:
            latch_note = (
                f", latched_bbox={self._fp_last_bbox_xyxy}, latched_label={latch_lbl!r}"
            )
        self.get_logger().info(
            f"foundation_pose: bridge params → target_class={yolo_class!r} "
            f"(requested {target_class!r}), exclusive={ex_list!r}, min_confidence={min_conf:.3f}{latch_note}"
        )
        return True

    def _foundation_pose_wait_tf_object(
        self,
        *,
        fp_cfg: Dict[str, Any],
        timeout_sec: float,
        trig_action: Optional[Dict[str, Any]] = None,
        bridge_trigger_service: str = "",
    ) -> bool:
        ref = str(fp_cfg.get("tf_reference_frame") or self.motion_cfg.get("base_link", "base_link")).strip()
        child = str(fp_cfg.get("object_tf_child_frame", "fp_object")).strip()
        optical = self._foundation_pose_camera_optical_frame(fp_cfg)
        output_topic = str(fp_cfg.get("isaac_output_topic", "/output")).strip()
        retrigger_period = float(fp_cfg.get("bridge_retrigger_period_sec", 8.0))
        deadline = time.monotonic() + max(0.1, float(timeout_sec))
        period = float(fp_cfg.get("tf_spin_period_sec", 0.03))
        last_retrigger = time.monotonic()
        saw_output = False
        from vision_msgs.msg import Detection3DArray

        def _on_output(msg: Detection3DArray) -> None:
            nonlocal saw_output
            if len(msg.detections) > 0:
                saw_output = True

        out_sub = self.create_subscription(Detection3DArray, output_topic, _on_output, 10)
        self.get_logger().info(
            f"foundation_pose: waiting for pose pipeline (TF {ref} <- {child} or via {optical!r}, "
            f"/output on {output_topic!r}, timeout={timeout_sec:.1f}s)."
        )
        try:
            while time.monotonic() < deadline and rclpy.ok():
                self._raise_if_abort()
                fp_mat = self._lookup_fp_object_mat_in_base(
                    ref, child, optical_frame=optical, lookup_timeout_sec=0.2
                )
                if fp_mat is not None:
                    self.get_logger().info(f"foundation_pose: object pose available in {ref!r}.")
                    return True
                if saw_output:
                    self.get_logger().info(
                        f"foundation_pose: Isaac publishing {output_topic!r}; waiting for TF…",
                        throttle_duration_sec=15.0,
                    )
                if (
                    trig_action is not None
                    and bridge_trigger_service
                    and (time.monotonic() - last_retrigger) >= retrigger_period
                ):
                    self.get_logger().info("foundation_pose: re-triggering bridge (waiting for Isaac pose)…")
                    self._foundation_pose_bridge_trigger_with_warmup(
                        trig_action, fp_cfg, bridge_trigger_service
                    )
                    last_retrigger = time.monotonic()
                rclpy.spin_once(self, timeout_sec=period)
        finally:
            self.destroy_subscription(out_sub)
        hint = (
            f"timed out waiting for TF {ref} <- {child} (and {optical!r} <- {child}). "
            "Check Isaac GPU/TensorRT, mesh matches target class, and /output publishes."
        )
        if not saw_output:
            hint += f" No messages on {output_topic!r} — Isaac may still be loading engines or mesh is wrong."
        self.get_logger().error(f"foundation_pose: {hint}")
        return False

    def _spin_cb(self, timeout_sec: float) -> None:
        rclpy.spin_once(self, timeout_sec=float(timeout_sec))

    def _foundation_pose_grasp_config_path(self, fp_cfg: Dict[str, Any]) -> str:
        explicit = os.path.expanduser(str(fp_cfg.get("grasp_config_file", "")).strip())
        if explicit:
            return explicit
        try:
            share = get_package_share_directory("ur3_pick_task")
            candidate = os.path.join(share, "config", "foundation_pose_grasp.yaml")
            if os.path.isfile(candidate):
                return candidate
        except PackageNotFoundError:
            pass
        dev = Path(__file__).resolve().parents[1] / "config" / "foundation_pose_grasp.yaml"
        return str(dev)

    def _load_grasp_by_class(self, fp_cfg: Dict[str, Any]) -> Dict[str, Any]:
        path = self._foundation_pose_grasp_config_path(fp_cfg)
        if self._grasp_by_class_cache_key == path and self._grasp_by_class_cache:
            return self._grasp_by_class_cache
        merged = merge_grasp_config(fp_cfg, grasp_config_file=path)
        self._grasp_by_class_cache_key = path
        self._grasp_by_class_cache = merged
        return merged

    def _foundation_pose_object_translation_quat(
        self,
        ref: str,
        child: str,
        timeout_sec: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        optical = self._foundation_pose_camera_optical_frame(self.foundation_pose_cfg)
        mat = self._lookup_fp_object_mat_in_base(ref, child, optical_frame=optical, lookup_timeout_sec=timeout_sec)
        if mat is None:
            raise RuntimeError(f"TF {ref} <- {child} unavailable")
        return mat[:3, 3].copy(), _rot_to_quat(mat[:3, :3])

    def _execute_foundation_pose_grasp_action(self, action: Dict[str, Any]) -> bool:
        """
        Auto-launch FoundationPose stack (bridge + YOLO + Isaac), wait for stable fp_object,
        apply per-class grasp offsets/axis alignment, MoveIt IK to gripper_tip (TCP).
        """
        cfg_fp = dict(self.foundation_pose_cfg)
        if isinstance(action.get("foundation_pose"), dict):
            cfg_fp.update(dict(action["foundation_pose"]))
        for top_key in (
            "auto_launch_stack",
            "stack_launch_timeout_sec",
            "wait_tf_timeout_sec",
            "wait_tf_stable",
            "tf_stable_sample_count",
            "tf_stable_max_position_std_m",
            "tf_stable_max_rotation_deg",
            "tf_stable_timeout_sec",
            "yolo_exclusive_scene_classes",
            "bridge_trigger_min_confidence",
            "use_latched_bbox_on_trigger",
        ):
            if top_key in action:
                cfg_fp[top_key] = action[top_key]

        tgt_raw = str(action.get("target_class") or "").strip()
        if not tgt_raw:
            tgt_raw = str(self.object_name or "").strip()
        if not tgt_raw:
            tgt_raw = str(self.vision_cfg.get("default_target_class", "") or "").strip()
        if not tgt_raw:
            self.get_logger().error(
                "foundation_pose_grasp: missing target_class (action, object_name, or vision.default_target_class)."
            )
            return False
        tgt = self._foundation_pose_resolve_target_class(tgt_raw, cfg_fp)

        trig = str(cfg_fp.get("bridge_trigger_service", "/foundation_pose_bridge/trigger")).strip()
        set_params = str(cfg_fp.get("bridge_set_parameters_service", "")).strip()
        tf_timeout = float(action.get("wait_tf_timeout_sec", cfg_fp.get("wait_tf_timeout_sec", 45.0)))
        ref = str(cfg_fp.get("tf_reference_frame", "base_link")).strip()
        child = str(cfg_fp.get("object_tf_child_frame", "fp_object")).strip()

        mesh_path = self._foundation_pose_mesh_path_for_class(tgt)
        texture_path = self._foundation_pose_texture_path_for_mesh(mesh_path)
        symmetry_axes = symmetry_axes_for_class(tgt, cfg_fp)
        try:
            grasp_by_class = self._load_grasp_by_class(cfg_fp)
            grasp_spec_pre = grasp_spec_for_class(grasp_by_class, tgt)
            if isinstance(action.get("grasp"), dict):
                grasp_spec_pre = dict(grasp_spec_pre)
                grasp_spec_pre.update(dict(action["grasp"]))
        except ValueError:
            grasp_spec_pre = {}
        tf_long_axis = grasp_spec_pre.get("long_axis_in_object")
        tf_upright = bool(
            grasp_spec_pre.get("upright_in_base", grasp_spec_pre.get("upright_z_in_base", False))
        )
        if mesh_path:
            self.get_logger().info(
                f"foundation_pose_grasp: Isaac mesh {mesh_path!r}"
                + (f" texture {texture_path!r}" if texture_path else "")
                + f" symmetry_axes={symmetry_axes}"
            )
        else:
            self.get_logger().warn(
                f"foundation_pose_grasp: no mesh path for class {tgt!r}; Isaac may use default FOUNDATION_POSE_MESH."
            )

        if bool(cfg_fp.get("auto_launch_stack", True)):
            long_axis_arg = tf_long_axis if tf_long_axis is not None and len(tf_long_axis) == 3 else None
            if not self._foundation_pose_ensure_stack_running(
                cfg_fp,
                tgt,
                force_relaunch=bool(cfg_fp.get("force_relaunch_stack_on_grasp", True)),
                tf_upright_in_base=tf_upright,
                tf_long_axis_in_object=long_axis_arg,
            ):
                return False

        self._ensure_vision_subscriptions_for_yolo()

        trig_action = dict(action)
        trig_action["service_name"] = trig
        trig_action.setdefault("wait_timeout_sec", 8.0)
        trig_action.setdefault("call_timeout_sec", 25.0)
        self._foundation_pose_post_stack_settle(cfg_fp, action)
        if getattr(self._fp_stack, "started_fresh", True):
            isaac_settle = float(cfg_fp.get("isaac_post_launch_settle_sec", 15.0))
        else:
            isaac_settle = float(cfg_fp.get("isaac_post_launch_settle_if_reused_sec", 0.0))
        if isaac_settle > 0.0:
            self.get_logger().info(
                f"foundation_pose_grasp: settling {isaac_settle:.0f}s "
                f"({'cold stack' if getattr(self._fp_stack, 'started_fresh', True) else 'stack reused'})…"
            )
            settle_end = time.monotonic() + isaac_settle
            while time.monotonic() < settle_end and rclpy.ok():
                self._raise_if_abort()
                self._spin_cb(0.1)

        if not self._foundation_pose_bridge_push_detection_params(
            cfg_fp,
            tgt,
            cold_stack=bool(getattr(self._fp_stack, "started_fresh", True)),
        ):
            return False

        if not self._foundation_pose_bridge_trigger_with_warmup(trig_action, cfg_fp, trig):
            self.get_logger().error(f"foundation_pose_grasp: bridge Trigger failed ({trig}).")
            return False

        post_trig = float(
            action.get("fp_post_trigger_settle_sec", cfg_fp.get("fp_post_trigger_settle_sec", 4.0))
        )
        if post_trig > 0.0:
            self.get_logger().info(
                f"foundation_pose_grasp: post-trigger settle {post_trig:.1f}s "
                "(FoundationPose refine before pose lock)…"
            )
            end = time.monotonic() + post_trig
            while time.monotonic() < end and rclpy.ok():
                self._raise_if_abort()
                self._spin_cb(0.1)

        rgb_rect = str(cfg_fp.get("isaac_rgb_rect_topic", "/rgb/image_rect_color")).strip()
        depth_rect = str(cfg_fp.get("isaac_depth_rect_topic", "/depth_registered/image_rect")).strip()
        isaac_input_wait = float(cfg_fp.get("isaac_input_topics_timeout_sec", 20.0))
        if isaac_input_wait > 0.0:
            if not wait_for_topic_publishers(
                self, rgb_rect, timeout_sec=isaac_input_wait, spin_cb=self._spin_cb
            ):
                self.get_logger().error(
                    f"foundation_pose_grasp: Isaac RGB bridge not publishing {rgb_rect!r} — "
                    "never reached TF wait (fp_object needs Isaac /output). "
                    f"Ensure RealSense or sim camera is running (backend={self.robot_backend!r}). "
                    "Stale stack: pkill -f foundation_pose_stack then retry. "
                    "See ~/.ros/log/foundation_pose_stack_latest.log"
                )
                tail = self._fp_stack.log_tail()
                if tail:
                    self.get_logger().error(f"FoundationPose stack log (tail):\n{tail}")
                return False
            if not wait_for_topic_publishers(
                self, depth_rect, timeout_sec=isaac_input_wait, spin_cb=self._spin_cb
            ):
                self.get_logger().warn(
                    f"foundation_pose_grasp: {depth_rect!r} not publishing yet; continuing…"
                )

        output_topic = str(cfg_fp.get("isaac_output_topic", "/output")).strip()
        fresh = bool(getattr(self._fp_stack, "started_fresh", True))
        output_wait = float(
            cfg_fp.get(
                "isaac_output_wait_timeout_sec",
                90.0 if fresh else 25.0,
            )
        )
        if output_wait > 0.0:
            self.get_logger().info(
                f"foundation_pose_grasp: waiting up to {output_wait:.0f}s for Isaac {output_topic!r} "
                f"(mesh={mesh_path or 'default'!r})…"
            )
            if not wait_for_detection3d_output(
                self,
                topic=output_topic,
                timeout_sec=output_wait,
                spin_cb=self._spin_cb,
            ):
                tail = self._fp_stack.log_tail()
                self.get_logger().error(
                    f"foundation_pose_grasp: Isaac never published {output_topic!r} "
                    f"(waited {output_wait:.0f}s). "
                    "Isaac composable may still be loading TensorRT, or camera/detection sync failed. "
                    "See ~/.ros/log/foundation_pose_stack_latest.log"
                )
                if tail:
                    self.get_logger().error(f"FoundationPose stack log (tail):\n{tail}")
                return False

        if not self._foundation_pose_wait_tf_object(
            fp_cfg=cfg_fp,
            timeout_sec=tf_timeout,
            trig_action=trig_action,
            bridge_trigger_service=trig,
        ):
            return False

        if bool(cfg_fp.get("wait_tf_stable", True)):
            stable_timeout = float(
                action.get("tf_stable_timeout_sec", cfg_fp.get("tf_stable_timeout_sec", tf_timeout))
            )
            ok, msg = wait_tf_stable(
                lambda: self._foundation_pose_object_translation_quat(ref, child, 1.0),
                sample_count=int(cfg_fp.get("tf_stable_sample_count", 12)),
                sample_period_sec=float(cfg_fp.get("tf_stable_sample_period_sec", 0.08)),
                max_position_std_m=float(cfg_fp.get("tf_stable_max_position_std_m", 0.003)),
                max_rotation_deg=float(cfg_fp.get("tf_stable_max_rotation_deg", 1.5)),
                max_yaw_std_deg=float(cfg_fp.get("tf_stable_max_yaw_std_deg", 2.0)),
                warmup_sec=float(
                    action.get("tf_stable_warmup_sec", cfg_fp.get("tf_stable_warmup_sec", 3.0))
                ),
                min_elapsed_sec=float(
                    action.get("tf_stable_min_elapsed_sec", cfg_fp.get("tf_stable_min_elapsed_sec", 6.0))
                ),
                timeout_sec=stable_timeout,
                spin_cb=self._spin_cb,
            )
            if not ok:
                self.get_logger().error(f"foundation_pose_grasp: pose not stable — {msg}")
                return False
            self.get_logger().info(f"foundation_pose_grasp: fp_object locked — {msg}")

        try:
            grasp_by_class = self._load_grasp_by_class(cfg_fp)
            grasp_spec = grasp_spec_for_class(grasp_by_class, tgt)
            grasp_spec = apply_foundation_pose_grasp_overrides(grasp_spec, cfg_fp, action)
        except ValueError as exc:
            self.get_logger().error(f"foundation_pose_grasp: {exc}")
            return False

        optical = self._foundation_pose_camera_optical_frame(cfg_fp)
        T_base_object = self._lookup_fp_object_mat_in_base(
            ref, child, optical_frame=optical, lookup_timeout_sec=tf_timeout, log_error=True
        )
        if T_base_object is None:
            self.get_logger().error(f"foundation_pose_grasp: could not resolve {ref} <- {child} for MoveIt.")
            return False

        R_raw = T_base_object[:3, :3].copy()
        tilt_raw = object_z_tilt_from_vertical_deg(R_raw)
        fp_inverted = is_upside_down_in_base(R_raw)
        T_base_object = normalize_T_base_object_for_grasp(T_base_object, grasp_spec)
        if bool(grasp_spec.get("upright_in_base", grasp_spec.get("upright_z_in_base", False))):
            tilt_up = object_z_tilt_from_vertical_deg(T_base_object[:3, :3])
            self.get_logger().info(
                f"foundation_pose_grasp: upright_in_base — object Z tilt "
                f"{tilt_raw:.2f}° → {tilt_up:.2f}° "
                f"(yaw={math.degrees(object_yaw_for_grasp(T_base_object[:3, :3], grasp_spec)):.1f}°, "
                f"fp_inverted={fp_inverted})"
            )

        if not self._foundation_pose_execute_two_stage_grasp(
            action, cfg_fp, T_base_object, grasp_spec, tgt
        ):
            return False
        return True

    def _foundation_pose_log_tcp_goals(
        self,
        grasp_spec: Dict[str, Any],
        tgt: str,
        tcp_link: str,
        pose_hover: Pose,
        pose_touch: Pose,
        hover_standoff: float,
        touch_standoff: float,
        T_base_object: np.ndarray,
    ) -> None:
        offset = grasp_spec.get(
            "grip_point_offset_m",
            grasp_spec.get("position_offset_m", [0, 0, 0]),
        )
        offset_frame = str(
            grasp_spec.get("grip_point_offset_frame", grasp_spec.get("grip_offset_frame", "object"))
        )
        lookat = grasp_spec.get("lookat_vector", [0.0, 0.0, -1.0])
        _, tcp_roll, sync_yaw = resolve_tcp_roll_for_grasp(T_base_object, grasp_spec)
        roll_off = float(grasp_spec.get("tcp_roll_rad", 0.0))
        roll_mode = "yaw+offset" if sync_yaw else "offset_only"
        pos_mode = "fp_pose"
        base = str(self.motion_cfg.get("base_frame", "base_link"))
        self.get_logger().info(
            f"foundation_pose_grasp: class={tgt!r} grip_offset_m={offset} "
            f"grip_offset_frame={offset_frame!r} "
            f"lookat_base={lookat} hover_standoff_m={hover_standoff:.3f} "
            f"touch_standoff_m={touch_standoff:.3f} "
            f"position_mode={pos_mode} "
            f"tcp_roll={math.degrees(tcp_roll):.1f}° "
            f"(roll_mode={roll_mode}, sync_yaw={sync_yaw}, tcp_roll_rad={math.degrees(roll_off):.1f}°) "
            f"→ two-stage MoveIt on {tcp_link!r}"
        )
        self.get_logger().info(
            f"foundation_pose_grasp: HOVER TCP in {base!r}: "
            f"xyz=({pose_hover.position.x:.3f}, {pose_hover.position.y:.3f}, {pose_hover.position.z:.3f})"
        )
        self.get_logger().info(
            f"foundation_pose_grasp: PLUNGE TCP in {base!r}: "
            f"xyz=({pose_touch.position.x:.3f}, {pose_touch.position.y:.3f}, {pose_touch.position.z:.3f})"
        )

    def _foundation_pose_move_with_collisions_retry(
        self,
        move_fn: Callable[[Dict[str, Any]], bool],
        action: Dict[str, Any],
        stage_label: str,
    ) -> bool:
        trial = dict(action)
        trial["avoid_collisions"] = self._moveit_avoid_collisions(action)
        return move_fn(trial)

    def _foundation_pose_run_hover_convergence(
        self,
        action: Dict[str, Any],
        pose_hover: Pose,
        step_scales: Sequence[float],
        cfg_fp: Dict[str, Any],
    ) -> bool:
        """Stage 1: iterative IK homing to absolute HOVER before plunge."""
        max_iter = max(
            1,
            int(
                action.get(
                    "hover_convergence_max_iterations",
                    cfg_fp.get("hover_convergence_max_iterations", 5),
                )
            ),
        )
        settle_sec = float(
            action.get(
                "hover_convergence_settle_sec",
                cfg_fp.get("hover_convergence_settle_sec", 0.12),
            )
        )
        scales = tuple(float(s) for s in step_scales)

        def _solve_at_scale(step_scale: float) -> Optional[list[float]]:
            result = self._solve_ik_with_step_scales_result(pose_hover, action, (step_scale,))
            if result is None:
                return None
            return result[0]

        def _execute_at_scale(joints: list[float], _step_scale: float) -> bool:
            return self._execute_joint_goal(action, joints)

        def _settle() -> None:
            if settle_sec > 0.0:
                time.sleep(settle_sec)

        return run_hover_convergence_loop(
            max_iterations=max_iter,
            step_scales=scales,
            solve_at_scale=_solve_at_scale,
            execute_at_scale=_execute_at_scale,
            settle_after_partial=_settle,
            log_info=self.get_logger().info,
            log_warn=self.get_logger().warn,
            log_error=self.get_logger().error,
        )

    def _foundation_pose_execute_two_stage_grasp(
        self,
        action: Dict[str, Any],
        cfg_fp: Dict[str, Any],
        T_base_object: np.ndarray,
        grasp_spec: Dict[str, Any],
        tgt: str,
    ) -> bool:
        """
        Stage 1 (hover): iterative IK homing (partial steps retried until step_scale=1.0).
        Stage 2 (plunge): straight-line Cartesian at min_fraction=1.0; IK fallback only at step_scale=1.0.
        """
        hover_standoff = float(
            action.get(
                "hover_standoff_m",
                action.get("hover_approach_standoff_m", cfg_fp.get("hover_approach_standoff_m", 0.12)),
            )
        )
        touch_standoff = float(
            action.get(
                "touch_standoff_m",
                action.get("touch_approach_standoff_m", cfg_fp.get("touch_approach_standoff_m", 0.01)),
            )
        )
        tcp_link = str(action.get("ik_link_name") or cfg_fp.get("ik_link_name") or self.tcp_frame_name).strip()
        hover_scales_raw = action.get("hover_ik_step_scales", cfg_fp.get("hover_ik_step_scales", [1.0, 0.5, 0.25]))
        hover_scales = tuple(float(x) for x in hover_scales_raw)

        spec_hover = grasp_spec_with_standoff(grasp_spec, hover_standoff)
        spec_touch = grasp_spec_with_standoff(grasp_spec, touch_standoff)
        try:
            pose_hover = _pose_from_transform(compute_base_tcp_pose(T_base_object, spec_hover))
            pose_touch = _pose_from_transform(compute_base_tcp_pose(T_base_object, spec_touch))
        except ValueError as exc:
            self.get_logger().error(f"foundation_pose_grasp: invalid grasp spec — {exc}")
            return False

        self._foundation_pose_log_tcp_goals(
            grasp_spec, tgt, tcp_link, pose_hover, pose_touch, hover_standoff, touch_standoff, T_base_object
        )

        base_name = str(action.get("name", "foundation_pose_grasp"))
        base_move = dict(action)
        base_move["ik_link_name"] = tcp_link

        max_hover_iter = int(
            action.get(
                "hover_convergence_max_iterations",
                cfg_fp.get("hover_convergence_max_iterations", 5),
            )
        )
        self.get_logger().info(
            f"foundation_pose_grasp stage 1/2 HOVER: standoff={hover_standoff:.3f}m "
            f"ik_scales={hover_scales} convergence_max_iter={max_hover_iter}"
        )
        hover_action = dict(base_move)
        hover_action["name"] = f"{base_name}_hover"
        if not self._foundation_pose_move_with_collisions_retry(
            lambda a: self._foundation_pose_run_hover_convergence(a, pose_hover, hover_scales, cfg_fp),
            hover_action,
            "hover",
        ):
            self.get_logger().error(
                "foundation_pose_grasp: stage 1 HOVER failed "
                "(could not reach full hover pose; plunge aborted)."
            )
            return False

        plunge_min = float(action.get("plunge_min_fraction", cfg_fp.get("plunge_min_fraction", 1.0)))
        plunge_max_step = float(action.get("plunge_max_step_m", cfg_fp.get("plunge_max_step_m", 0.002)))
        cart_mode = str(action.get("plunge_cartesian_mode", cfg_fp.get("plunge_cartesian_mode", "linear")))
        self.get_logger().info(
            f"foundation_pose_grasp stage 2/2 PLUNGE: standoff={touch_standoff:.3f}m "
            f"cartesian_mode={cart_mode!r} min_fraction={plunge_min:.3f} max_step_m={plunge_max_step:.4f}"
        )
        plunge_action = dict(base_move)
        plunge_action["name"] = f"{base_name}_plunge"
        plunge_action["cartesian_mode"] = cart_mode
        plunge_action["min_fraction"] = plunge_min
        plunge_action["max_step_m"] = plunge_max_step
        if not self._foundation_pose_move_with_collisions_retry(
            lambda a: self._move_tool_pose_plunge_full(a, pose_touch),
            plunge_action,
            "plunge",
        ):
            self.get_logger().error(
                "foundation_pose_grasp: stage 2 PLUNGE failed "
                "(require 100% Cartesian path or IK at step_scale=1.0)."
            )
            return False
        return True

    def _execute_foundation_pose_action(self, action: Dict[str, Any]) -> bool:
        """Trigger YOLO→Detection bridge, wait Isaac TF pose, optional IK look-down on current TCP XYZ."""
        cfg_fp = dict(self.foundation_pose_cfg)
        if isinstance(action.get("foundation_pose"), dict):
            cfg_fp.update(dict(action["foundation_pose"]))
        for top_key in (
            "include_grasp_prep_orientation",
            "wait_tf_timeout_sec",
            "log_mesh_hint",
            "yolo_exclusive_scene_classes",
            "bridge_trigger_min_confidence",
            "use_latched_bbox_on_trigger",
        ):
            if top_key in action:
                cfg_fp[top_key] = action[top_key]

        tgt_raw = str(action.get("target_class") or "").strip()
        if not tgt_raw:
            tgt_raw = str(self.object_name or "").strip()
        if not tgt_raw:
            tgt_raw = str(self.vision_cfg.get("default_target_class", "") or "").strip()
        if not tgt_raw:
            self.get_logger().error(
                "foundation_pose: missing target_class (set on action, CLI object_name, or vision.default_target_class)."
            )
            return False
        tgt = self._foundation_pose_resolve_target_class(tgt_raw, cfg_fp)
        if tgt != tgt_raw:
            self.get_logger().info(
                f"foundation_pose: target_class {tgt_raw!r} → YOLO label {tgt!r} (yolo_class_aliases)."
            )

        mesh_hint = self._foundation_pose_mesh_path_for_class(tgt)
        sym_axes = symmetry_axes_for_class(tgt, cfg_fp)
        if mesh_hint:
            hint = (
                f"foundation_pose CAD (Isaac mesh_file_path): {mesh_hint} — "
                f"symmetry_axes={sym_axes} (mesh frame, via foundation_pose.symmetry_axes_by_class)."
            )
            if bool(action.get("log_mesh_hint", True)):
                self.get_logger().info(hint)

        trig = str(cfg_fp.get("bridge_trigger_service", "/foundation_pose_bridge/trigger")).strip()
        tf_timeout = float(action.get("wait_tf_timeout_sec", cfg_fp.get("wait_tf_timeout_sec", 45.0)))

        if bool(cfg_fp.get("auto_launch_stack", True)):
            if not self._foundation_pose_ensure_stack_running(cfg_fp, tgt, force_relaunch=False):
                return False

        self._ensure_vision_subscriptions_for_yolo()
        if self._fp_last_bbox_xyxy is not None:
            self.get_logger().debug(
                f"foundation_pose: last yolo_xyxy snapshot {self._fp_last_bbox_xyxy} label={self._fp_last_detection_label!r}"
            )

        trig_action = dict(action)
        trig_action["service_name"] = trig
        trig_action.setdefault("wait_timeout_sec", 8.0)
        trig_action.setdefault("call_timeout_sec", 25.0)
        self._foundation_pose_post_stack_settle(cfg_fp, action)
        if not self._foundation_pose_bridge_push_detection_params(
            cfg_fp,
            tgt,
            cold_stack=bool(getattr(self._fp_stack, "started_fresh", False)),
        ):
            return False
        if not self._foundation_pose_bridge_trigger_with_warmup(trig_action, cfg_fp, trig):
            self.get_logger().error(f"foundation_pose: bridge Trigger failed ({trig}).")
            return False

        if not self._foundation_pose_wait_tf_object(fp_cfg=cfg_fp, timeout_sec=tf_timeout):
            self.get_logger().error(
                "foundation_pose: TF wait failed — ensure Isaac FoundationPose is running "
                f"(stack: {self._foundation_pose_stack_launch_file(cfg_fp) or 'foundation_pose_stack.launch.py'}) "
                f"and check `ros2 run tf2_ros tf2_echo base_link {cfg_fp.get('object_tf_child_frame', 'fp_object')}`."
            )
            return False

        child = str(cfg_fp.get("object_tf_child_frame", "fp_object")).strip()
        ref = str(cfg_fp.get("tf_reference_frame", "base_link")).strip()
        try:
            t = self._tf_buffer.lookup_transform(ref, child, Time(), timeout=RclDuration(seconds=1.0))
            p = t.transform.translation
            self.get_logger().info(
                f"foundation_pose: OK — TF {ref} <- {child} at "
                f"({p.x:.3f}, {p.y:.3f}, {p.z:.3f}) m."
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(f"foundation_pose: TF echo after wait failed: {exc}")

        if bool(cfg_fp.get("include_grasp_prep_orientation", False)):
            look_raw = action.get("lookat_vector", cfg_fp.get("grasp_prep_lookat_vector", [0.0, 0.0, -1.0]))
            lookat = _normalize_lookat_vector(_vector3(look_raw, "lookat_vector"))
            tcp_roll = float(action.get("tcp_roll_rad", cfg_fp.get("grasp_prep_tcp_roll_rad", 0.0)))
            base_frame = str(self.motion_cfg.get("base_frame", "base_link")).strip()
            tool_frame = str(self.motion_cfg.get("ik_link_name", "tool0")).strip()
            ik_timeout = float(action.get("tf_timeout_sec", self.motion_cfg.get("default_joint_state_timeout_sec", 5.0)))
            pose_pick = self._tool_pose_lock_tcp_pick_orientation(
                lookat,
                tcp_roll,
                base_frame=base_frame,
                tool_frame=tool_frame,
                tf_timeout_sec=ik_timeout,
            )
            if pose_pick is None:
                self.get_logger().error("foundation_pose: grasp_prep_orientation IK TF lookup failed.")
                return False
            self.get_logger().info(f"foundation_pose: grasp_prep TCP lookat_vector={lookat.tolist()}, tcp_roll={tcp_roll:.3f}")
            return self._move_tool_pose_ik_joint(action, pose_pick)

        return True

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

    def _continuous_joint_names(self) -> frozenset[str]:
        raw = self.motion_cfg.get("continuous_joint_names")
        if raw is None:
            return DEFAULT_CONTINUOUS_JOINT_NAMES
        if isinstance(raw, str):
            names = {raw.strip()} if raw.strip() else set()
        elif isinstance(raw, (list, tuple, set, frozenset)):
            names = {str(name).strip() for name in raw if str(name).strip()}
        else:
            names = set()
        return frozenset(names) if names else DEFAULT_CONTINUOUS_JOINT_NAMES

    def _align_continuous_joint_targets(self, positions: Sequence[float]) -> list[float]:
        """Align YAML/IK targets with live encoder wrap on continuous joints (e.g. wrist_3).

        Real UR wrists accumulate revolutions (e.g. -9.42 rad vs YAML -3.14 rad). MoveIt and
        the trajectory controller reject goals when the numeric gap is ~2π even though the
        physical orientation matches. Pick the nearest 2π-equivalent before planning.
        """
        aligned = [float(v) for v in positions]
        if len(aligned) != len(self.joint_names):
            return aligned

        current = self._current_manipulator_joint_positions()
        if current is None:
            return aligned

        continuous = self._continuous_joint_names()
        for idx, name in enumerate(self.joint_names):
            if name not in continuous:
                continue
            raw_target = aligned[idx]
            adjusted = _unwrap_joint_angle_to_current(raw_target, current[idx])
            if abs(adjusted - raw_target) > 1e-4:
                self.get_logger().info(
                    f"Unwrapped {name} target for MoveIt: "
                    f"{raw_target:.4f} -> {adjusted:.4f} rad "
                    f"(current={current[idx]:.4f})"
                )
            aligned[idx] = adjusted
        return aligned

    def _unwrap_continuous_joints_in_joint_trajectory(
        self,
        joint_traj: JointTrajectory,
        *,
        reference_by_name: Optional[Dict[str, float]] = None,
    ) -> JointTrajectory:
        """Align every trajectory waypoint to the live/previous wrap on continuous joints."""
        if not joint_traj.points:
            return joint_traj

        continuous = self._continuous_joint_names()
        traj_idx_by_name = {name: idx for idx, name in enumerate(joint_traj.joint_names)}
        to_unwrap: list[tuple[int, str]] = []
        for name in self.joint_names:
            if name not in continuous:
                continue
            traj_idx = traj_idx_by_name.get(name)
            if traj_idx is not None:
                to_unwrap.append((traj_idx, name))
        if not to_unwrap:
            return joint_traj

        if reference_by_name is None:
            current = self._current_manipulator_joint_positions()
            if current is None:
                return joint_traj
            reference_by_name = dict(zip(self.joint_names, current))

        prev: Dict[str, float] = {}
        for _, name in to_unwrap:
            if name not in reference_by_name:
                return joint_traj
            prev[name] = float(reference_by_name[name])

        max_traj_idx = max(idx for idx, _ in to_unwrap)
        adjusted_any = False
        for point in joint_traj.points:
            if len(point.positions) <= max_traj_idx:
                continue
            positions = list(point.positions)
            for traj_idx, name in to_unwrap:
                raw = float(positions[traj_idx])
                adjusted = _unwrap_joint_angle_to_current(raw, prev[name])
                if abs(adjusted - raw) > 1e-4:
                    adjusted_any = True
                positions[traj_idx] = adjusted
                prev[name] = adjusted
            point.positions = positions

        if adjusted_any:
            unwrapped = ", ".join(name for _, name in to_unwrap)
            self.get_logger().info(
                f"Unwrapped continuous joint trajectory waypoints ({unwrapped}) "
                "to match live encoder wrap before controller execution."
            )
        return joint_traj

    def _unwrap_continuous_joints_in_robot_trajectory(self, trajectory: RobotTrajectory) -> RobotTrajectory:
        self._unwrap_continuous_joints_in_joint_trajectory(trajectory.joint_trajectory)
        return trajectory

    def _execute_planned_move_group_result(
        self,
        action: Dict[str, Any],
        result: MoveGroup.Result,
        *,
        log_label: str,
    ) -> bool:
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"{log_label} planning failed: error_code={result.error_code.val}")
            return False

        planned = result.planned_trajectory
        if not planned.joint_trajectory.points:
            self.get_logger().error(f"{log_label} plan returned an empty trajectory.")
            return False

        timeout = float(action.get("execution_timeout_sec", self._planning_time(action) + 30.0))
        return self._execute_trajectory(planned, timeout)

    def _moveit_start_joint_state(self) -> JointState:
        """JointState ordered like the MoveIt group (not /joint_states topic order)."""
        positions = self._current_manipulator_joint_positions()
        if positions is None:
            return self._latest_joint_state if self._latest_joint_state is not None else JointState()

        js = JointState()
        js.name = list(self.joint_names)
        js.position = positions
        latest = self._latest_joint_state
        if latest is not None and latest.velocity and len(latest.velocity) == len(latest.name):
            vel_by_name = dict(zip(latest.name, latest.velocity))
            try:
                js.velocity = [float(vel_by_name[name]) for name in self.joint_names]
            except KeyError:
                pass
        return js

    def _moveit_avoid_collisions(self, action: Optional[Dict[str, Any]] = None) -> bool:
        if action is not None and "avoid_collisions" in action:
            return bool(action["avoid_collisions"])
        return bool(self.motion_cfg.get("avoid_collisions", True))

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

    def _yolo_acquire_debug_banner(
        self,
        model: Any,
        *,
        target_class: str,
        acquire_conf: float,
        yolo_iou: float,
        ray_kw: Dict[str, Any],
    ) -> None:
        """Emit one diagnostic when bootstrap cannot lock YOLO+depth after many frames."""
        if not self._wait_vision_frames(1.0):
            self.get_logger().warn("yolo_visual_center acquisition debug: timed out waiting for camera frames.")
            return
        with self._vis_lock:
            bgr = self._vis_bgr.copy() if self._vis_bgr is not None else None
            depth_buf = self._vis_depth.copy() if self._vis_depth is not None else None
            info = self._vis_info
        if bgr is None:
            self.get_logger().warn("yolo_visual_center acquisition debug: RGB buffer empty.")
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            results = model.predict(rgb, conf=0.004, iou=yolo_iou, verbose=False)[0]
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"yolo_visual_center acquisition debug: ultralytics.predict failed: {exc}")
            return
        names = results.names
        parts: list[str] = []
        boxes = results.boxes
        if boxes is None or boxes.cls is None or len(boxes) == 0:
            parts.append("(no_detection_boxes)")
        else:
            for i in range(len(boxes)):
                cid = int(boxes.cls[i].item())
                cf = float(boxes.conf[i].item())
                cname = str(names[cid]).strip()
                parts.append(f"{cname}:{cf:.2f}")
                if len(parts) >= 28:
                    break
        ci = bool(ray_kw.get("class_match_case_insensitive", True))
        tgt_strip = target_class.strip()
        hint = ""
        for p in parts:
            if p.startswith("(no_detection"):
                continue
            name_part, sep, conf_part = p.partition(":")
            if not sep:
                continue
            cname_line = name_part.strip()
            if (ci and cname_line.lower() == tgt_strip.lower()) or (
                not ci and cname_line == tgt_strip
            ):
                hint = (
                    f" — class name matches `{cname_line}` @ {conf_part} (threshold {acquire_conf:.2f}); "
                    "if still failing, depth inside bbox likely invalid."
                )
                break
        intrinsics_ok = info is not None and hasattr(info, "k") and len(info.k) >= 5
        self.get_logger().warn(
            "yolo_visual_center acquisition debug (after repeated misses): "
            f"camera_info.fx≈{'%.1f' % float(info.k[0]) if intrinsics_ok else 'n/a'}; "
            f"YOLO detections(conf≥0.004): [{', '.join(parts)}]{hint}"
        )
        if depth_buf is None:
            self.get_logger().warn(
                "yolo_visual_center acquisition debug: depth buffer empty — "
                "depth Image must publish sensor_msgs/Image with encoding "
                "`32FC1` (meters) or `16UC1` (millimeters, e.g. RealSense)."
            )
            return
        finite = depth_buf[np.isfinite(depth_buf)]
        if finite.size <= 0:
            self.get_logger().warn("yolo_visual_center acquisition debug: depth has no finite values.")
            return
        self.get_logger().warn(
            "yolo_visual_center acquisition debug: depth finite pixels "
            f"median={float(np.median(finite)):.3f} m, "
            f"range=[{float(np.min(finite)):.3f}, {float(np.max(finite)):.3f}]"
        )
        if depth_buf is None or info is None:
            return
        weak_conf = float(max(0.01, min(acquire_conf, 0.12)))
        ray_probe = dict(ray_kw)
        ray_probe.setdefault("depth_min_bbox_pixels", 8)
        det_probe = self._yolo_detection_center_ray(
            model,
            bgr,
            depth_buf,
            info,
            target_class=target_class,
            min_conf=weak_conf,
            yolo_iou=yolo_iou,
            roi=int(self.vision_cfg.get("depth_roi_half_px", 5)),
            predict_conf_floor=float(ray_kw.get("predict_conf_floor", 0.01)),
            class_match_case_insensitive=ci,
            exclusive_scene_classes=ray_kw.get("exclusive_scene_classes"),
        )
        if det_probe is None:
            self.get_logger().warn(
                f"yolo_visual_center acquisition debug: probe at min_conf={weak_conf:.2f} "
                f"for {target_class!r} returned no valid ray (conf too low or depth inside bbox invalid)."
            )
        else:
            self.get_logger().warn(
                f"yolo_visual_center acquisition debug: probe OK at min_conf={weak_conf:.2f} "
                f"(z={det_probe[4]:.3f} m, du={det_probe[5]:.1f}, dv={det_probe[6]:.1f} px) — "
                f"lower yolo_visual_center_acquire_min_confidence if bootstrap still fails."
            )

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
        depth_valid_min_m: float = 0.03,
        depth_valid_max_m: float = 12.0,
        depth_min_bbox_pixels: int = 12,
        exclusive_scene_classes: Optional[list[str]] = None,
        pick_among_strategy: str = "max_conf",
        preferred_uv: Optional[tuple[float, float]] = None,
        scene_unique_classes: bool = False,
        scene_rescore_crops: bool = True,
        scene_cluster_iou: float = 0.45,
        scene_assign_exclusive_classes: Optional[list[str]] = None,
    ) -> Optional[tuple[int, int, float, float, float, float, float, float]]:
        """
        Run YOLO and return bbox image center back-projected in camera optical coords:
        u, v, x_cam, y_cam, z, du/dv vs CameraInfo principal point (cx,cy)—same convention as ray—det_confidence.

        If ``exclusive_scene_classes`` is set: only detections matching ``target_class`` are used
        (highest conf if several). Other exclusive classes are ignored so Box_1 is never replaced
        by a cylinder. Without a target match, selection falls through to the strict filter below.
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        infer_conf = float(min(max(1e-6, predict_conf_floor), min_conf))
        results = model.predict(rgb, conf=infer_conf, iou=yolo_iou, verbose=False)[0]
        exclusive_list_fp: Optional[list[str]] = None
        if isinstance(exclusive_scene_classes, (list, tuple)):
            tmp_ex = [str(x).strip() for x in exclusive_scene_classes if str(x).strip()]
            if tmp_ex:
                exclusive_list_fp = tmp_ex
        picked = _ultralytics_pick_bbox_xyxy_and_conf(
            results,
            target_class=target_class,
            min_conf=min_conf,
            class_match_case_insensitive=class_match_case_insensitive,
            exclusive_scene_classes=exclusive_list_fp,
            pick_among_strategy=pick_among_strategy,
            preferred_uv=preferred_uv,
            scene_unique_classes=scene_unique_classes,
            scene_rescore_crops=scene_rescore_crops,
            scene_cluster_iou=scene_cluster_iou,
            scene_assign_exclusive_classes=scene_assign_exclusive_classes,
            model=model,
            rgb=rgb,
            predict_conf_floor=float(predict_conf_floor),
            yolo_iou=float(yolo_iou),
            log_exclusive_fn=lambda m: self.get_logger().debug(m),
            log_pick_fn=lambda m: self.get_logger().info(m),
        )
        if picked is None:
            return None
        best_box, best_conf, _win_label = picked
        x1, y1, x2, y2 = [int(round(float(v))) for v in np.asarray(best_box, dtype=np.float64).reshape(-1).tolist()]
        h, w = bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        u = (x1 + x2) // 2
        v = (y1 + y2) // 2

        K = info.k
        fx, fy = float(K[0]), float(K[4])
        cx, cy = float(K[2]), float(K[5])

        z_lo = float(depth_valid_min_m)
        z_hi = float(depth_valid_max_m)
        min_px = max(1, int(depth_min_bbox_pixels))
        roi_half = max(3, int(roi))

        def _median_depth_valid(arr: np.ndarray) -> Optional[float]:
            flat = arr.reshape(-1)
            finite = flat[np.isfinite(flat)]
            valid = finite[(finite > z_lo) & (finite < z_hi)]
            if valid.size < 1:
                return None
            return float(np.median(valid))

        crop = depth[y1 : y2 + 1, x1 : x2 + 1]
        z = _median_depth_valid(crop)
        if z is None or int(crop.size) < min_px:
            patch = depth[
                max(0, v - roi_half) : min(h, v + roi_half + 1),
                max(0, u - roi_half) : min(w, u + roi_half + 1),
            ]
            z = _median_depth_valid(patch)
        if z is None:
            return None

        x_cam = (float(u) - cx) * z / fx
        y_cam = (float(v) - cy) * z / fy
        du = float(u) - cx
        dv = float(v) - cy
        return u, v, float(x_cam), float(y_cam), float(z), du, dv, float(best_conf)

    def _lookup_tf_mat(
        self,
        target_frame: str,
        source_frame: str,
        timeout_sec: float,
        *,
        log_error: bool = True,
    ) -> Optional[np.ndarray]:
        """
        TF transform converting a point from `source_frame` to `target_frame`:
        p_target = R @ p_source + t (4x4 stores R,t).
        """
        try:
            st = self._tf_buffer.lookup_transform(
                target_frame, source_frame, Time(), timeout=RclDuration(seconds=float(timeout_sec))
            )
        except Exception as exc:  # noqa: BLE001
            if log_error:
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
    def _yolo_bbox_centered(
        du_px: float,
        dv_px: float,
        tol_px: float,
        *,
        metric: str = "per_axis",
    ) -> bool:
        """Return True when bbox offset is within tolerance.

        ``per_axis``: |Δu|≤tol and |Δv|≤tol (default).
        ``l1``: |Δu|+|Δv| ≤ 2·tol (stricter for diagonal offset — better visual center).
        ``linf``: max(|Δu|,|Δv|) ≤ tol.
        """
        du = abs(float(du_px))
        dv = abs(float(dv_px))
        t = float(tol_px)
        mode = (metric or "per_axis").lower().strip()
        if mode == "l1":
            return (du + dv) <= (2.0 * t)
        if mode in ("linf", "chebyshev", "max"):
            return max(du, dv) <= t
        return du <= t and dv <= t

    @staticmethod
    def _yolo_center_tolerance_metric(cfg: Dict[str, Any]) -> str:
        return str(cfg.get("yolo_center_tolerance_metric", "per_axis")).lower().strip()

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
    def _yolo_center_adaptive_stream_params(
        metric_px: float,
        tol_px: float,
        cfg: Dict[str, Any],
    ) -> tuple[float, float, float]:
        """
        Scale streamed Cartesian centering by L1 pixel error |Δu|+|Δv|.

        Far from center: larger optical step gain, larger per-nudge lateral cap, faster path velocity.
        Near center: smaller moves to settle, with a configured floor so motion does not crawl.
        """
        if not bool(cfg.get("yolo_center_metric_adaptive", True)):
            sg = float(cfg.get("yolo_center_step_gain", cfg.get("step_gain", 0.35)))
            ms = float(cfg.get("yolo_center_max_step_m", 0.012))
            vs = float(cfg.get("yolo_center_velocity_scaling", 0.045))
            return sg, ms, vs

        tol = float(max(1.0, tol_px))
        m = float(max(0.0, abs(metric_px)))

        near_cfg = cfg.get("yolo_center_adaptive_near_metric_px", None)
        if near_cfg is None:
            near_m = max(36.0, 2.6 * tol)
        else:
            near_m = float(near_cfg)
        far_m = float(cfg.get("yolo_center_adaptive_far_metric_px", 128.0))
        if far_m <= near_m + 5.0:
            far_m = near_m + 48.0

        base_sg = float(cfg.get("yolo_center_step_gain", cfg.get("step_gain", 0.35)))
        sg_lo = float(cfg.get("yolo_center_step_gain_near", base_sg * 0.86))
        sg_hi = float(cfg.get("yolo_center_step_gain_far", max(base_sg * 1.5, sg_lo + 0.18)))
        sg_lo = float(min(sg_lo, sg_hi))

        base_ms = float(cfg.get("yolo_center_max_step_m", 0.012))
        ms_lo = float(
            cfg.get(
                "yolo_center_max_step_m_near",
                max(0.0065, min(0.010, base_ms * 0.75)),
            )
        )
        ms_hi = float(cfg.get("yolo_center_max_step_m_far", max(base_ms * 1.55, 0.018)))
        ms_hi = float(max(ms_hi, ms_lo + 0.004))

        base_vs = float(cfg.get("yolo_center_velocity_scaling", 0.045))
        vs_lo = float(cfg.get("yolo_center_velocity_scaling_near", base_vs))
        vs_hi = float(cfg.get("yolo_center_velocity_scaling_far", min(0.12, base_vs + 0.058)))
        vs_hi = float(max(vs_hi, vs_lo + 0.018))

        if m <= near_m:
            t = 0.0
        elif m >= far_m:
            t = 1.0
        else:
            t = (m - near_m) / (far_m - near_m)

        sg = sg_lo + t * (sg_hi - sg_lo)
        ms = ms_lo + t * (ms_hi - ms_lo)
        vs = vs_lo + t * (vs_hi - vs_lo)
        return float(sg), float(ms), float(vs)

    @staticmethod
    def _yolo_relaxed_center_tolerance_px(cfg: Dict[str, Any], tol_px: float) -> float:
        """Optional looser tolerance only for soft-complete after nudges (never for early exit)."""
        relaxed = cfg.get("yolo_single_motion_center_tolerance_px")
        if relaxed is None:
            return float(tol_px)
        return float(max(float(tol_px), float(relaxed)))

    @staticmethod
    def _yolo_center_error_based_max_step_m(
        du_px: float,
        dv_px: float,
        z_cam: float,
        fx: float,
        fy: float,
        *,
        step_gain: float,
        adaptive_cap: float,
        cfg: Dict[str, Any],
        fraction: Optional[float] = None,
    ) -> float:
        """
        Scale lateral cap with optical error chord: ~|| (step_gain·Δu/fx·z , step_gain·Δv/fy·z ) ||₂.

        Large pixel error ⇒ larger allowable per-nudge displacement (within hard safety cap).
        """
        fx_abs = abs(float(fx))
        fy_abs = abs(float(fy))
        z_abs = abs(float(z_cam))
        sg_abs = abs(float(step_gain))
        if fraction is None:
            frac_raw = cfg.get(
                "yolo_center_error_step_fraction",
                cfg.get("yolo_center_error_scale_fraction", 1.0),
            )
            frac = float(max(0.02, min(1.35, float(frac_raw))))
        else:
            frac = float(max(0.02, min(1.35, float(fraction))))
        if (
            fx_abs < 1e-9
            or fy_abs < 1e-9
            or z_abs < 1e-9
            or not bool(cfg.get("yolo_center_max_step_scale_by_pixel_error", True))
        ):
            return float(max(0.0, adaptive_cap))

        ms_far = float(
            cfg.get("yolo_center_max_step_m_far", cfg.get("yolo_center_max_step_m", 0.012))
        )
        ms_base = float(cfg.get("yolo_center_max_step_m", 0.012))
        hard_default = float(max(ms_far * 3.8, ms_base * 6.5, 0.055))
        hard = float(cfg.get("yolo_center_max_step_m_hard_abs", hard_default))

        gx = sg_abs * abs(float(du_px)) / fx_abs * z_abs
        gy = sg_abs * abs(float(dv_px)) / fy_abs * z_abs
        chord_m = frac * math.hypot(float(gx), float(gy))
        out = float(max(float(adaptive_cap), chord_m))
        return float(min(hard, out))

    @staticmethod
    def _pose_translation_m(a: Pose, b: Pose) -> float:
        return float(
            math.hypot(
                a.position.x - b.position.x,
                a.position.y - b.position.y,
                a.position.z - b.position.z,
            )
        )

    @staticmethod
    def _pose_rotation_rad(a: Pose, b: Pose) -> float:
        qa = np.array(
            [a.orientation.x, a.orientation.y, a.orientation.z, a.orientation.w],
            dtype=np.float64,
        )
        qb = np.array(
            [b.orientation.x, b.orientation.y, b.orientation.z, b.orientation.w],
            dtype=np.float64,
        )
        qa = qa / max(float(np.linalg.norm(qa)), 1e-12)
        qb = qb / max(float(np.linalg.norm(qb)), 1e-12)
        dot = float(np.clip(abs(float(np.dot(qa, qb))), -1.0, 1.0))
        return float(2.0 * math.acos(dot))

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
        sign_u: float = -1.0,
        sign_v: float = -1.0,
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
        su = float(sign_u)
        sv = float(sign_v)
        delta_cam = np.array(
            [-sign * su * float(step_gain) * x_cam, -sign * sv * float(step_gain) * y_cam, 0.0],
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

    def _tool_pose_orient_nudge_from_pixels(
        self,
        du_px: float,
        dv_px: float,
        *,
        fx: float,
        fy: float,
        cfg: Dict[str, Any],
        nudge_sign: float,
        optical_frame: str,
        base_frame: str,
        tf_timeout_sec: float,
    ) -> Optional[Pose]:
        """
        Small camera re-aim: rotate tool0 in base while keeping TCP position fixed.

        Pixel offsets map to optical-frame pitch (X) / yaw (Y), same sign convention as IBVS.
        """
        pose_now = self._lookup_tool0_pose()
        if pose_now is None:
            return None
        T_base_opt = self._lookup_tf_mat(base_frame, optical_frame, tf_timeout_sec)
        if T_base_opt is None:
            return None
        if fx <= 1e-6 or fy <= 1e-6:
            return None

        sign_u = float(nudge_sign) * float(
            cfg.get("yolo_center_orient_sign_u", cfg.get("yolo_servo_ibvs_sign_u", 1.0))
        )
        sign_v = float(nudge_sign) * float(
            cfg.get("yolo_center_orient_sign_v", cfg.get("yolo_servo_ibvs_sign_v", -1.0))
        )
        gain = float(cfg.get("yolo_center_orient_step_gain", cfg.get("yolo_center_step_gain", 0.35)))
        theta_y = -sign_u * gain * (float(du_px) / float(fx))
        theta_x = -sign_v * gain * (float(dv_px) / float(fy))
        cap = abs(float(cfg.get("yolo_center_orient_max_step_rad", 0.085)))
        ang = math.hypot(theta_x, theta_y)
        if ang < 1e-9:
            return None
        if cap > 0.0 and ang > cap:
            scale = cap / ang
            theta_x *= scale
            theta_y *= scale

        cx, sx = math.cos(theta_x), math.sin(theta_x)
        cy, sy = math.cos(theta_y), math.sin(theta_y)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
        ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
        r_delta_opt = rx @ ry

        r_base_opt = T_base_opt[:3, :3]
        q_now = np.array(
            [
                pose_now.orientation.x,
                pose_now.orientation.y,
                pose_now.orientation.z,
                pose_now.orientation.w,
            ],
            dtype=np.float64,
        )
        r_tool = _quat_to_rot(q_now)
        r_tool_new = _orthonormalize_rotation(r_base_opt @ r_delta_opt @ r_base_opt.T @ r_tool)
        q_new = _rot_to_quat(r_tool_new)

        out = Pose()
        out.position.x = float(pose_now.position.x)
        out.position.y = float(pose_now.position.y)
        out.position.z = float(pose_now.position.z)
        out.orientation.x = float(q_new[0])
        out.orientation.y = float(q_new[1])
        out.orientation.z = float(q_new[2])
        out.orientation.w = float(q_new[3])
        return out

    def _yolo_build_smooth_orient_centering_waypoints(
        self,
        pose_target: Pose,
        *,
        segments: int,
        profile: str = "cosine",
    ) -> list[Pose]:
        """Interpolate tool0 orientation toward ``pose_target``; position stays at the current TCP."""
        pose_now = self._lookup_tool0_pose()
        if pose_now is None:
            return []
        q0 = np.array(
            [
                pose_now.orientation.x,
                pose_now.orientation.y,
                pose_now.orientation.z,
                pose_now.orientation.w,
            ],
            dtype=np.float64,
        )
        q1 = np.array(
            [
                pose_target.orientation.x,
                pose_target.orientation.y,
                pose_target.orientation.z,
                pose_target.orientation.w,
            ],
            dtype=np.float64,
        )
        n = max(1, int(segments))
        prof = str(profile).lower().strip()
        waypoints: list[Pose] = []
        for idx in range(1, n + 1):
            frac = self._yolo_center_waypoint_fraction(idx, n, profile=prof)
            qt = _quat_slerp(q0, q1, frac)
            wp = Pose()
            wp.position.x = float(pose_now.position.x)
            wp.position.y = float(pose_now.position.y)
            wp.position.z = float(pose_now.position.z)
            wp.orientation.x = float(qt[0])
            wp.orientation.y = float(qt[1])
            wp.orientation.z = float(qt[2])
            wp.orientation.w = float(qt[3])
            waypoints.append(wp)
        return waypoints

    @staticmethod
    def _yolo_center_waypoint_fraction(idx: int, n: int, *, profile: str) -> float:
        """Chord parameter in (0,1] for waypoint ``idx`` of ``n`` (end at 1)."""
        n = max(1, int(n))
        idx = max(1, min(n, int(idx)))
        t = float(idx) / float(n)
        prof = (profile or "cosine").lower().strip()
        if prof in ("linear", "uniform", "even"):
            return t
        # Ease-in-out: smaller steps near start/end → less abrupt Cartesian / joint reshaping per segment.
        return 0.5 * (1.0 - math.cos(math.pi * t))

    def _yolo_build_smooth_centering_waypoints(
        self,
        pose_target: Pose,
        *,
        segments: int,
        profile: str = "cosine",
    ) -> list[Pose]:
        """Interpolate tool0 position toward ``pose_target`` for one continuous Cartesian path."""
        pose_now = self._lookup_tool0_pose()
        if pose_now is None:
            return []
        n = max(1, int(segments))
        prof = str(profile).lower().strip()
        waypoints: list[Pose] = []
        for idx in range(1, n + 1):
            frac = self._yolo_center_waypoint_fraction(idx, n, profile=prof)
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
        smooth["cartesian_retime_min_segment_sec"] = float(
            cfg.get("yolo_center_retime_min_segment_sec", 0.04)
        )
        smooth["cartesian_retime_duration_stretch"] = float(
            cfg.get("yolo_center_retime_duration_stretch", 1.0)
        )
        strict_frac = float(max(0.01, min(1.0, cfg.get("yolo_center_min_fraction", 0.72))))
        relaxed_req = cfg.get("yolo_center_min_fraction_relaxed", None)
        if relaxed_req is None:
            relaxed_frac = max(0.2, strict_frac * 0.42)
        else:
            relaxed_frac = float(relaxed_req)
        relaxed_frac = float(max(0.05, min(1.0, relaxed_frac)))
        fraction_tiers = [strict_frac]
        if abs(relaxed_frac - strict_frac) > 1e-9:
            fraction_tiers.append(relaxed_frac)

        for tier_idx, min_fraction in enumerate(fraction_tiers):
            if tier_idx > 0:
                self.get_logger().warn(
                    "yolo_visual_center: Cartesian path coverage low — retrying execution with "
                    f"relaxed min_fraction={min_fraction:.2f} (partial segment motion is OK during centering)."
                )
            avoid_order = (self._moveit_avoid_collisions(smooth),)
            for avoid in avoid_order:
                trial = dict(smooth, avoid_collisions=avoid)
                cres = self._execute_cartesian_path(trial, waypoints, min_fraction)
                if not cres:
                    continue
                after = self._lookup_tool0_pose()
                if after is None:
                    return False, 0.0
                delta_m = self._pose_translation_m(before, after)
                delta_rad = self._pose_rotation_rad(before, after)
                min_orient_rad = float(cfg.get("yolo_center_min_orient_move_rad", 0.0025))
                if delta_m >= float(min_move_m) or delta_rad >= min_orient_rad:
                    return True, delta_m
                self.get_logger().warn(
                    f"yolo_visual_center: Cartesian stream finished but tool0 moved only "
                    f"{delta_m * 1000.0:.2f} mm (Δθ={math.degrees(delta_rad):.2f}°)."
                )

        ik_frac = float(
            cfg.get(
                "yolo_center_ik_fallback_min_fraction",
                cfg.get("yolo_center_min_fraction_relaxed", 0.05),
            )
        )
        ik_trial = dict(
            smooth,
            avoid_collisions=self._moveit_avoid_collisions(smooth),
            min_fraction=ik_frac,
        )
        self.get_logger().warn(
            "yolo_visual_center: Cartesian centering path blocked or incomplete — trying IK fallback."
        )
        min_orient_rad = float(cfg.get("yolo_center_min_orient_move_rad", 0.0025))

        def _stream_moved_enough(after_pose: Pose) -> tuple[bool, float]:
            delta_m = self._pose_translation_m(before, after_pose)
            delta_rad = self._pose_rotation_rad(before, after_pose)
            ok_move = delta_m >= float(min_move_m) or delta_rad >= min_orient_rad
            return ok_move, delta_m

        if self._move_cartesian_waypoint_then_ik(ik_trial, waypoints[-1]):
            after = self._lookup_tool0_pose()
            if after is not None:
                ok_move, delta_m = _stream_moved_enough(after)
                if ok_move:
                    return True, delta_m
        self.get_logger().warn(
            "yolo_visual_center: Cartesian+IK waypoint failed — trying joint-space IK."
        )
        joint_positions = self._solve_ik_scaled_steps(waypoints[-1], ik_trial)
        if joint_positions is not None and self._execute_joint_goal(ik_trial, joint_positions):
            after = self._lookup_tool0_pose()
            if after is not None:
                ok_move, delta_m = _stream_moved_enough(after)
                if ok_move:
                    return True, delta_m
        return False, 0.0

    def _yolo_execute_centering_moveit_ik(
        self,
        center_action: Dict[str, Any],
        goal_pose: Pose,
        cfg: Dict[str, Any],
        *,
        min_move_m: float,
    ) -> tuple[bool, float]:
        """Reach YOLO centering goal via /compute_ik + MoveGroup joint plan (no Cartesian straight segment)."""
        before = self._lookup_tool0_pose()
        if before is None:
            return False, 0.0
        min_orient_rad = float(cfg.get("yolo_center_min_orient_move_rad", 0.0025))
        trial = dict(center_action)
        trial["avoid_collisions"] = self._moveit_avoid_collisions(center_action)
        lift_m, roll_deg = self._yolo_recovery_offsets(trial, cfg)
        candidates = self._yolo_relaxed_pose_candidates_from_goal(
            goal_pose, cfg, recovery_lift_m=lift_m, recovery_roll_deg=roll_deg
        )
        moved = False
        for _label, pose in candidates:
            if self._move_yolo_center_ik(trial, cfg, pose):
                moved = True
                break
        if moved:
            after = self._lookup_tool0_pose()
            if after is None:
                return False, 0.0
            delta_m = self._pose_translation_m(before, after)
            delta_rad = self._pose_rotation_rad(before, after)
            if delta_m >= float(min_move_m) or delta_rad >= min_orient_rad:
                return True, delta_m
            self.get_logger().warn(
                f"yolo_visual_center: MoveIt IK move finished but tool0 shifted only "
                f"{delta_m * 1000.0:.2f} mm (Δθ={math.degrees(delta_rad):.2f}°)."
            )
        return False, 0.0

    def _move_tool_poses_moveit_ik(self, action: Dict[str, Any], poses: list[Pose]) -> bool:
        """Approach/center via sequential IK + joint goals (collision-aware MoveIt planning)."""
        if not poses:
            return True
        for idx, pose in enumerate(poses):
            if not self._move_tool_pose_ik_joint(action, pose):
                self.get_logger().warn(
                    f"yolo_visual_center: MoveIt IK step {idx + 1}/{len(poses)} failed."
                )
                return False
        return True

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

    @staticmethod
    def _yolo_servo_apply_pixel_deadband(
        du_px: float, dv_px: float, deadband_px: float
    ) -> tuple[float, float]:
        """Soft deadband: no twist inside band; linear ramp outside (reduces YOLO jitter shake)."""
        db = max(0.0, float(deadband_px))
        if db <= 0.0:
            return float(du_px), float(dv_px)

        def _soft(v: float) -> float:
            av = abs(float(v))
            if av <= db:
                return 0.0
            return math.copysign(av - db, float(v))

        return _soft(du_px), _soft(dv_px)

    @staticmethod
    def _yolo_servo_slew_limit_pixels(
        du_px: float,
        dv_px: float,
        state: Dict[str, Optional[float]],
        max_delta_u_px: float,
        max_delta_v_px: float,
    ) -> tuple[float, float]:
        """Cap per-tick change in commanded pixel error (asymmetric u/v reduces vertical mask hunts)."""
        mdu = abs(float(max_delta_u_px))
        mdv = abs(float(max_delta_v_px))
        du0, dv0 = float(du_px), float(dv_px)
        if mdu <= 0.0 and mdv <= 0.0:
            state["du"], state["dv"] = du0, dv0
            return du0, dv0
        if state.get("du") is None or state.get("dv") is None:
            state["du"], state["dv"] = du0, dv0
            return du0, dv0

        def _step(new: float, old: float, md: float) -> float:
            if md <= 0.0:
                return new
            d = new - old
            if d > md:
                return old + md
            if d < -md:
                return old - md
            return new

        ndu = _step(du0, float(state["du"]), mdu)
        ndv = _step(dv0, float(state["dv"]), mdv)
        state["du"], state["dv"] = ndu, ndv
        return ndu, ndv

    @staticmethod
    def _yolo_servo_near_center_velocity_scale(
        metric_px: float,
        *,
        tol_px: float,
        cfg: Dict[str, Any],
        fine_pass: bool,
    ) -> float:
        settle_ramp = float(cfg.get("yolo_servo_settle_ramp_px", 0.0))
        ramp_px = settle_ramp if settle_ramp > 0.0 else float(
            cfg.get("yolo_servo_near_center_ramp_px", max(2.5 * float(tol_px), 45.0))
        )
        if fine_pass:
            ramp_px = min(ramp_px, max(2.0 * float(tol_px), 30.0))
        if ramp_px <= 0.0:
            return 1.0
        metric = float(metric_px)
        if metric >= ramp_px:
            return 1.0
        min_scale = float(
            cfg.get(
                "yolo_servo_settle_min_scale",
                cfg.get("yolo_servo_near_center_min_scale", 0.12),
            )
        )
        t = max(0.0, min(1.0, metric / ramp_px))
        smooth = t * t * (3.0 - 2.0 * t)
        return max(min_scale, smooth)

    @staticmethod
    def _yolo_servo_needs_fov_recovery(
        du_raw: float,
        dv_raw: float,
        metric_raw: float,
        cfg: Dict[str, Any],
    ) -> bool:
        """
        Cartesian FOV recovery only when the object is actually near the image edge on
        both axes (or total error is huge). Avoids vertical-only spikes (|dv|>95, |du|~0)
        that were causing repeated 3s Cartesian moves and up/down shake.
        """
        fov = float(cfg.get("yolo_servo_max_axis_offset_px", 95.0))
        min_pair = float(cfg.get("yolo_servo_fov_pair_min_px", 30.0))
        metric_limit = float(cfg.get("yolo_servo_fov_metric_px", 140.0))
        adu = abs(float(du_raw))
        adv = abs(float(dv_raw))
        if adu > fov and adv > min_pair:
            return True
        if adv > fov and adu > min_pair:
            return True
        return float(metric_raw) > metric_limit

    @staticmethod
    def _yolo_servo_axis_velocity_scale(
        err_px: float,
        *,
        tol_px: float,
        cfg: Dict[str, Any],
        fine_pass: bool,
    ) -> float:
        """Per-axis speed scale (higher ramp than L1 metric so |dv|~50 does not saturate)."""
        ramp_px = float(
            cfg.get("yolo_servo_axis_ramp_px", max(4.5 * float(tol_px), 70.0))
        )
        if fine_pass:
            ramp_px = min(ramp_px, max(3.0 * float(tol_px), 50.0))
        if ramp_px <= 0.0:
            return 1.0
        err = abs(float(err_px))
        if err >= ramp_px:
            return 1.0
        min_scale = float(
            cfg.get(
                "yolo_servo_settle_min_scale",
                cfg.get("yolo_servo_near_center_min_scale", 0.12),
            )
        )
        t = max(0.0, min(1.0, err / ramp_px))
        smooth = t * t * (3.0 - 2.0 * t)
        return max(min_scale, smooth)

    @staticmethod
    def _yolo_servo_clip_linear_xy_per_axis(
        v_xy: np.ndarray,
        du_px: float,
        dv_px: float,
        *,
        max_lin: float,
        tol_px: float,
        cfg: Dict[str, Any],
        fine_pass: bool,
    ) -> np.ndarray:
        """Per-axis velocity cap from pixel error (stops v_y saturating when only dv is moderate)."""
        out = np.asarray(v_xy, dtype=np.float64).copy()
        max_lin = abs(float(max_lin))
        if max_lin <= 0.0:
            return out
        for i, err in enumerate((float(du_px), float(dv_px))):
            cap = max_lin * MotionManager._yolo_servo_axis_velocity_scale(
                err, tol_px=tol_px, cfg=cfg, fine_pass=fine_pass
            )
            out[i] = float(np.clip(out[i], -cap, cap))
        return out

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
        deadband = float(cfg.get("yolo_servo_deadband_px", 10.0))
        du, dv = self._yolo_servo_apply_pixel_deadband(du_px, dv_px, deadband)
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
        if metric < 1e-6:
            return Twist()
        slow_px = float(cfg.get("yolo_servo_slow_error_px", 100.0))
        if metric > slow_px > 0.0:
            v_cam[:2] *= slow_px / metric
        max_lin = float(cfg.get("yolo_servo_max_linear_m_s", 0.012))
        if fine_pass:
            max_lin = min(max_lin, float(cfg.get("yolo_servo_fine_pass_max_linear_m_s", 0.005)))
        v_cam[:2] = self._yolo_servo_clip_linear_xy_per_axis(
            v_cam[:2],
            du,
            dv,
            max_lin=max_lin,
            tol_px=tol_px,
            cfg=cfg,
            fine_pass=fine_pass,
        )

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
        deadband = float(cfg.get("yolo_servo_deadband_px", 10.0))
        du, dv = self._yolo_servo_apply_pixel_deadband(du_px, dv_px, deadband)
        if abs(du) + abs(dv) < 1e-6:
            return Twist()

        pose_now = self._lookup_tool0_pose()
        if pose_now is None:
            return None
        sign_u = float(cfg.get("yolo_center_orient_sign_u", cfg.get("yolo_servo_ibvs_sign_u", -1.0)))
        sign_v = float(cfg.get("yolo_center_orient_sign_v", cfg.get("yolo_servo_ibvs_sign_v", -1.0)))
        pose_goal = self._tool_pose_lateral_nudge_from_pixels(
            du,
            dv,
            z_cam,
            fx,
            fy,
            step_gain=float(cfg.get("yolo_center_step_gain", cfg.get("step_gain", 0.35))),
            max_step_m=float(cfg.get("yolo_center_max_step_m", 0.012)),
            nudge_sign=float(cfg.get("yolo_center_nudge_sign", 1.0)),
            sign_u=sign_u,
            sign_v=sign_v,
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
        v_opt *= self._yolo_servo_near_center_velocity_scale(
            metric, tol_px=tol_px, cfg=cfg, fine_pass=fine_pass
        )

        max_lin = float(cfg.get("yolo_servo_max_linear_m_s", 0.008))
        if fine_pass:
            max_lin = min(max_lin, float(cfg.get("yolo_servo_fine_pass_max_linear_m_s", 0.005)))
        v_norm = float(np.linalg.norm(v_opt))
        if v_norm > max_lin > 0.0:
            v_opt *= max_lin / v_norm
        if v_norm < 1e-9:
            return Twist()

        # Rare setups invert Servo vs planner (+1 matches Cartesian streamed nudge direction).
        ibvs_sign = float(cfg.get("yolo_servo_ibvs_sign", 1.0))
        v_opt *= ibvs_sign

        cmd_frame = str(cfg.get("yolo_servo_command_frame", "camera_color_optical_frame"))
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
        wp_profile = str(cfg.get("yolo_center_waypoint_profile", "cosine")).lower().strip()
        sign_u = float(cfg.get("yolo_center_orient_sign_u", cfg.get("yolo_servo_ibvs_sign_u", -1.0)))
        sign_v = float(cfg.get("yolo_center_orient_sign_v", cfg.get("yolo_servo_ibvs_sign_v", -1.0)))
        pose_goal = self._tool_pose_lateral_nudge_from_pixels(
            du,
            dv,
            z_cam,
            fx,
            fy,
            step_gain=float(cfg.get("yolo_center_step_gain", cfg.get("step_gain", 0.35))),
            max_step_m=float(cfg.get("yolo_center_max_step_m", 0.012)),
            nudge_sign=float(cfg.get("yolo_center_nudge_sign", 1.0)),
            sign_u=sign_u,
            sign_v=sign_v,
            optical_frame=optical,
            base_frame=base,
            tf_timeout_sec=tf_timeout,
        )
        if pose_goal is None:
            return False
        segments = max(4, int(cfg.get("yolo_center_path_segments", 20)))
        waypoints = self._yolo_build_smooth_centering_waypoints(
            pose_goal, segments=segments, profile=wp_profile
        )
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

    def _yolo_cfg_for_smooth_servo_phase(self, cfg: Dict[str, Any], tol_px: float) -> Dict[str, Any]:
        """Merge conservative Servo gains/filters (``servo_smooth`` / ``cartesian_then_servo`` phase 2)."""
        out = dict(cfg)
        if bool(cfg.get("yolo_smooth_servo_preset", True)):
            out.update(_YOLO_SMOOTH_SERVO_PRESET)
        tol = float(max(1.0, tol_px))
        out.setdefault("yolo_servo_deadband_px", float(max(3.0, min(14.0, 0.38 * tol))))
        out.setdefault("yolo_servo_center_hysteresis_px", float(max(2.0, min(8.0, 0.28 * tol))))
        ov_prefix = "yolo_smooth_servo_override_"
        for key, val in cfg.items():
            if not isinstance(key, str) or val is None:
                continue
            if key.startswith(ov_prefix):
                out["yolo_servo_" + key[len(ov_prefix) :]] = val
        return out

    def _publish_servo_twist(self, cfg: Dict[str, Any], twist: Twist) -> None:
        self._ensure_servo_twist_publisher(cfg)
        cmd_frame = str(cfg.get("yolo_servo_command_frame", "camera_color_optical_frame"))
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
        """One Servo IBVS pass; convergence uses smoothed offset + hysteresis when configured."""
        use_raw_only = bool(cfg.get("yolo_servo_use_raw_only", True))
        center_on_smoothed = bool(cfg.get("yolo_servo_center_on_smoothed", True))
        center_hyst = max(0.0, float(cfg.get("yolo_servo_center_hysteresis_px", 6.0)))
        deadband_px = float(cfg.get("yolo_servo_deadband_px", 10.0))
        in_center_zone = False

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
        ibvs_sign = float(cfg.get("yolo_servo_ibvs_sign", 1.0))
        use_optical_ibvs = bool(cfg.get("yolo_servo_use_optical_ibvs", True))
        cmd_on_keepalive = bool(cfg.get("yolo_servo_cmd_on_keepalive", True))
        diverge_abort_streak = max(0, int(cfg.get("yolo_servo_diverge_abort_streak", 0)))
        max_axis_offset = float(cfg.get("yolo_servo_max_axis_offset_px", 95.0))
        fov_cooldown = float(cfg.get("yolo_servo_fov_recovery_cooldown_sec", 5.0))
        last_fov_recovery_t = 0.0
        enable_fov_recovery = bool(cfg.get("yolo_servo_enable_fov_cartesian_recovery", False))
        max_pixel_delta_u = float(
            cfg.get("yolo_servo_max_pixel_delta_u_px", cfg.get("yolo_servo_max_pixel_delta_px", 14.0))
        )
        max_pixel_delta_v = float(
            cfg.get("yolo_servo_max_pixel_delta_v_px", cfg.get("yolo_servo_max_pixel_delta_px", 14.0))
        )
        pixel_slew_state: Dict[str, Optional[float]] = {"du": None, "dv": None}
        stall_sec = float(cfg.get("yolo_servo_stall_cartesian_sec", 2.8))
        stall_min_metric = float(cfg.get("yolo_servo_stall_min_metric_px", 55.0))
        stall_refresh_px = float(cfg.get("yolo_servo_stall_refresh_px", 5.0))
        stall_cooldown = float(cfg.get("yolo_servo_stall_cartesian_cooldown_sec", 3.5))
        last_stall_cartesian_t = 0.0
        best_metric_seen: Optional[float] = None
        last_metric_improve_t = time.monotonic()

        sign_u = float(cfg.get("yolo_servo_ibvs_sign_u", cfg.get("yolo_servo_ibvs_sign", 1.0)))
        sign_v = float(cfg.get("yolo_servo_ibvs_sign_v", cfg.get("yolo_servo_ibvs_sign", -1.0)))
        cmd_frame = str(cfg.get("yolo_servo_command_frame", "camera_color_optical_frame"))
        self.get_logger().info(
            f"yolo_visual_center: Servo {phase_tag} (rate={rate_hz:.0f} Hz, "
            f"max_v={cfg.get('yolo_servo_max_linear_m_s', 0.012)} m/s, "
            f"sign_u={sign_u:+.1f} sign_v={sign_v:+.1f}, optical_ibvs={use_optical_ibvs}, "
            f"raw_only={use_raw_only}, fov_cartesian={enable_fov_recovery}, fov_guard={max_axis_offset:.0f}px, "
            f"px_slew_u≤{max_pixel_delta_u:.0f}px v≤{max_pixel_delta_v:.0f}px, "
            f"stall_cart={stall_sec:.1f}s, "
            f"vel_ema={vel_ema:.2f}, deadband={deadband_px:.0f}px, hyst={center_hyst:.0f}px, "
            f"warmup_raw={warmup_need}, frame={cmd_frame}, tol={tol_px:.0f}px)."
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
            du_chk = du_smooth if center_on_smoothed else du_raw_f
            dv_chk = dv_smooth if center_on_smoothed else dv_raw_f
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

            du_cmd, dv_cmd = self._yolo_servo_slew_limit_pixels(
                du_cmd, dv_cmd, pixel_slew_state, max_pixel_delta_u, max_pixel_delta_v
            )

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
                    3.0 * float(tol_px), 2.0 * deadband_px
                ):
                    brake_ticks = max(brake_ticks, cross_brake_frames)
                    self._yolo_moveit_servo_publish_stop(cfg)
                    if tick % log_period == 0:
                        self.get_logger().info(
                            "yolo_visual_center servo: zero-cross brake "
                            f"(Δu {last_du_servo:.0f}→{du_cmd:.0f}, Δv {last_dv_servo:.0f}→{dv_cmd:.0f})"
                        )
                    pixel_slew_state["du"] = None
                    pixel_slew_state["dv"] = None
                    last_du_servo = du_cmd
                    last_dv_servo = dv_cmd
                    rclpy.spin_once(self, timeout_sec=0.0)
                    time.sleep(period)
                    continue
            last_du_servo = du_cmd
            last_dv_servo = dv_cmd

            metric_raw = abs(du_raw_f) + abs(dv_raw_f)
            now_st = time.monotonic()
            if stall_sec > 0.0 and best_metric_seen is None:
                best_metric_seen = metric_raw
                last_metric_improve_t = now_st
            elif stall_sec > 0.0 and best_metric_seen is not None:
                prev_best = best_metric_seen
                best_metric_seen = min(best_metric_seen, metric_raw)
                if metric_raw <= prev_best - stall_refresh_px:
                    last_metric_improve_t = now_st

            if in_center_zone:
                still_centered = self._yolo_bbox_centered(
                    du_chk, dv_chk, tol_px + center_hyst
                )
            else:
                still_centered = self._yolo_bbox_centered(du_chk, dv_chk, tol_px)
                if still_centered:
                    in_center_zone = True

            if still_centered:
                stable += 1
                if stable >= stable_need:
                    vel_state["v"] = None
                    vel_state["slew"] = None
                    pixel_slew_state["du"] = None
                    pixel_slew_state["dv"] = None
                    self._yolo_moveit_servo_publish_stop(cfg, repeats=6)
                    self.get_logger().info(
                        f"yolo_visual_center: centered via MoveIt Servo ({phase_tag}) "
                        f"raw=(Δu={du_raw_f:.1f}, Δv={dv_raw_f:.1f}) "
                        f"chk=({du_chk:.1f},{dv_chk:.1f}) px, conf={conf:.3f}, "
                        f"stable_frames={stable_need}."
                    )
                    return True
                # Keep feeding Servo lightly so incoming_command_timeout does not stall mid-settle.
                self._yolo_moveit_servo_publish_stop(cfg, repeats=1)
                rclpy.spin_once(self, timeout_sec=0.0)
                time.sleep(period)
                continue
            else:
                in_center_zone = False
                stable = 0
                with self._vis_lock:
                    cam_info = self._vis_info
                if cam_info is None:
                    time.sleep(period)
                    continue
                fx = float(cam_info.k[0])
                fy = float(cam_info.k[4])
                metric = metric_raw

                now_t = time.monotonic()
                stall_recovery = (
                    stall_sec > 0.0
                    and best_metric_seen is not None
                    and metric_raw >= stall_min_metric
                    and (now_t - last_metric_improve_t) >= stall_sec
                    and (now_t - last_stall_cartesian_t) >= stall_cooldown
                )
                if stall_recovery:
                    last_stall_cartesian_t = now_t
                    self.get_logger().warn(
                        f"yolo_visual_center servo {phase_tag}: pixel error stalled "
                        f"(|Δ|≈{metric_raw:.0f}px ≥ {stall_min_metric:.0f}px for {stall_sec:.1f}s "
                        f"without −{stall_refresh_px:.0f}px drop from prior best) "
                        f"— Cartesian correction."
                    )
                    self._yolo_moveit_servo_publish_stop(cfg)
                    if self._yolo_servo_one_cartesian_correction(
                        action,
                        cfg,
                        du=float(du_raw_f),
                        dv=float(dv_raw_f),
                        z_cam=float(z_ray),
                        fx=fx,
                        fy=fy,
                        optical=optical,
                        base=base,
                        tf_timeout=tf_timeout,
                    ):
                        self._yolo_moveit_servo_start(cfg)
                        self._spin_vision_settle(14, 0.05)
                        last_cartesian_nudge_t = time.monotonic()
                        last_improve_metric_t = time.monotonic()
                        last_metric = None
                        offset_state["du"] = None
                        offset_state["dv"] = None
                        vel_state["v"] = None
                        vel_state["slew"] = None
                        pixel_slew_state["du"] = None
                        pixel_slew_state["dv"] = None
                        best_metric_seen = None
                        last_metric_improve_t = time.monotonic()
                    continue

                need_fov = self._yolo_servo_needs_fov_recovery(
                    du_raw_f, dv_raw_f, metric_raw, cfg
                )
                if (
                    enable_fov_recovery
                    and need_fov
                    and (now_t - last_fov_recovery_t) >= fov_cooldown
                ):
                    last_fov_recovery_t = now_t
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
                        pixel_slew_state["du"] = None
                        pixel_slew_state["dv"] = None
                        best_metric_seen = None
                        last_metric_improve_t = time.monotonic()
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
                    vel_state["slew"] = None
                    pixel_slew_state["du"] = None
                    pixel_slew_state["dv"] = None
                    best_metric_seen = None
                    last_metric_improve_t = time.monotonic()
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
                        vel_state["slew"] = None
                        pixel_slew_state["du"] = None
                        pixel_slew_state["dv"] = None
                        best_metric_seen = None
                        last_metric_improve_t = time.monotonic()
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
                    if using_keepalive:
                        ka_scale = float(cfg.get("yolo_servo_keepalive_twist_scale", 0.35))
                        twist.linear.x *= ka_scale
                        twist.linear.y *= ka_scale
                        twist.linear.z *= ka_scale
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
        tol_metric = self._yolo_center_tolerance_metric(cfg)
        if not self._yolo_bbox_centered(du, dv, tol_px, metric=tol_metric):
            self.get_logger().error(
                f"yolo_visual_center: {phase_tag} — bbox not centered "
                f"(Δu={du:.1f}, Δv={dv:.1f} px, tolerance={tol_px:.1f}, metric={tol_metric})."
            )
            return False
        try:
            self._foundation_pose_record_bbox_snapshot_after_yolo(
                model=model,
                target_class=target_class,
                min_conf=min_conf,
                yolo_iou=float(cfg.get("yolo_iou", self.vision_cfg.get("yolo_iou", 0.5))),
                ray_kw=ray_kw,
                depth_z_med=float(det_raw[4]),
            )
        except Exception as exc:  # noqa: BLE001 — debug aid only.
            self.get_logger().debug(f"foundation_pose bbox snapshot skipped: {exc}")
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
        handoff_metric_px: Optional[float] = None,
        ik_only: bool = False,
        _track_frame: Callable[..., tuple],
        _on_raw_miss: Callable[[], bool],
        _save_last_detection_pose: Callable[[], None],
    ) -> bool:
        """Center bbox via streamed lateral nudges (Cartesian path or MoveIt IK-only when ``ik_only``)."""
        tol_metric = self._yolo_center_tolerance_metric(cfg)
        plan_ceiling = max(1, int(cfg.get("yolo_center_max_plan_iterations", 3)))
        configured = max(
            1,
            int(cfg.get("center_max_iterations") or cfg.get("max_iterations", plan_ceiling)),
        )
        max_iter = min(plan_ceiling, configured)
        pixel_direct_mode = bool(cfg.get("yolo_center_pixel_direct", False))
        if bool(cfg.get("yolo_visual_center_single_motion", False)) and pixel_direct_mode:
            sm_nudges = max(1, int(cfg.get("yolo_single_motion_center_max_nudges", 2)))
            max_iter = min(plan_ceiling, max(max_iter, sm_nudges))
        prefer_raw = bool(cfg.get("yolo_center_use_raw_detection", True))
        deadline = time.monotonic() + float(cfg.get("yolo_center_max_duration_sec", 45.0))
        center_action = dict(action)
        center_action["avoid_collisions"] = bool(cfg.get("yolo_visual_center_avoid_collisions", True))

        settle_frames = int(cfg.get("yolo_center_settle_frames", 25))
        settle_period = float(cfg.get("yolo_center_settle_period_sec", 0.06))
        stream_spin_frames = max(0, int(cfg.get("yolo_center_after_stream_spin_frames", 3)))
        stream_spin_period = float(cfg.get("yolo_center_after_stream_spin_period_sec", 0.022))

        wp_prof = str(cfg.get("yolo_center_waypoint_profile", "cosine")).lower().strip()
        cmd_ema = float(cfg.get("yolo_center_command_ema_alpha", 0.0))
        stretch = float(cfg.get("yolo_center_retime_duration_stretch", 1.0))
        retime_min_seg = float(cfg.get("yolo_center_retime_min_segment_sec", 0.04))
        adapt = bool(cfg.get("yolo_center_metric_adaptive", True))
        path_seg_log = (
            int(cfg.get("yolo_center_direct_path_segments", 1))
            if pixel_direct_mode
            else int(cfg.get("yolo_center_path_segments", 4))
        )
        orient_fallback = bool(cfg.get("yolo_center_enable_orient_fallback", True))
        orient_max = max(0, int(cfg.get("yolo_center_orient_max_attempts", 4)))
        motion_kind = "MoveIt IK (joint plan)" if ik_only else "Cartesian stream"
        self.get_logger().info(
            f"yolo_visual_center: {motion_kind} centering (max_iter={max_iter}, "
            f"max_duration_sec={cfg.get('yolo_center_max_duration_sec', 60.0):.0f}, "
            f"metric_adaptive={adapt}, pixel_direct={pixel_direct_mode}, "
            f"path_segments={path_seg_log}, tolerance_px={tol_px:.1f} ({tol_metric}), "
            f"waypoints={wp_prof!r}, cmd_ema_alpha={'off' if cmd_ema <= 0.0 else f'{cmd_ema:.2f}'}, "
            f"retime_stretch={stretch:.2f}, retime_dt≥{retime_min_seg:.3f}s, "
            f"after_stream_spin={stream_spin_frames}×{stream_spin_period:.3f}s, "
            f"miss_spin={settle_frames}×{settle_period:.3f}s, "
            f"handoff_metric_px={handoff_metric_px if handoff_metric_px is not None else 'none'}, "
            f"orient_fallback={orient_fallback} (max={orient_max}), "
            f"avoid_collisions={center_action['avoid_collisions']})."
        )

        cmd_ema_state: Dict[str, Optional[float]] = {"du": None, "dv": None}

        nudge_count = 0
        orient_count = 0
        nudge_sign = float(cfg.get("yolo_center_nudge_sign", 1.0))
        sign_u = float(cfg.get("yolo_center_orient_sign_u", cfg.get("yolo_servo_ibvs_sign_u", -1.0)))
        sign_v = float(cfg.get("yolo_center_orient_sign_v", cfg.get("yolo_servo_ibvs_sign_v", -1.0)))
        min_move_m = float(cfg.get("yolo_center_min_move_m", 0.0003))
        path_segments = max(1, int(cfg.get("yolo_center_path_segments", 4)))
        center_misses = 0
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
            if cmd_ema > 0.0 and not (
                pixel_direct_mode and bool(cfg.get("yolo_center_direct_bypass_command_ema", True))
            ):
                aema = float(max(0.02, min(1.0, cmd_ema)))
                if cmd_ema_state["du"] is None:
                    cmd_ema_state["du"] = float(du)
                    cmd_ema_state["dv"] = float(dv)
                else:
                    cmd_ema_state["du"] = aema * float(du) + (1.0 - aema) * float(cmd_ema_state["du"])
                    cmd_ema_state["dv"] = aema * float(dv) + (1.0 - aema) * float(cmd_ema_state["dv"])
                du = float(cmd_ema_state["du"])
                dv = float(cmd_ema_state["dv"])
            _, _, _, _, _, _, _, conf = det
            metric = abs(float(du)) + abs(float(dv))

            coarse_phase = False
            direct_frac_kw: Optional[float] = None

            if pixel_direct_mode:
                step_gain = float(cfg.get("yolo_center_direct_step_gain", 1.0))
                v_scale = float(
                    cfg.get(
                        "yolo_center_direct_velocity_scaling",
                        cfg.get("yolo_center_velocity_scaling_far", 0.1),
                    )
                )
                max_step = 0.0
                direct_frac_kw = float(cfg.get("yolo_center_direct_fraction", 1.0))
            else:
                coarse_n_max = max(0, int(cfg.get("yolo_center_coarse_nudge_max", 0)))
                coarse_n_max = min(coarse_n_max, max_iter)
                coarse_m_cfg = cfg.get("yolo_center_coarse_min_metric_px", None)
                coarse_m_px = (
                    float(coarse_m_cfg)
                    if coarse_m_cfg is not None
                    else max(40.0, 2.8 * float(tol_px))
                )
                coarse_phase = (
                    coarse_n_max > 0
                    and nudge_count < coarse_n_max
                    and metric >= coarse_m_px
                )
                if coarse_phase:
                    step_gain, max_step, v_scale = self._yolo_center_adaptive_stream_params(
                        1.0e9, tol_px, cfg
                    )
                else:
                    step_gain, max_step, v_scale = self._yolo_center_adaptive_stream_params(
                        metric, tol_px, cfg
                    )
            if (
                handoff_metric_px is not None
                and not keep_alive
                and not self._yolo_bbox_centered(du, dv, tol_px, metric=tol_metric)
                and metric <= float(handoff_metric_px)
            ):
                self.get_logger().info(
                    f"yolo_visual_center: Cartesian coarse done for Servo handoff "
                    f"(|Δ|={metric:.1f}px ≤ {float(handoff_metric_px):.0f}px, nudges={nudge_count}, "
                    f"conf={conf:.3f})."
                )
                return True
            if not keep_alive and self._yolo_bbox_centered(du, dv, tol_px, metric=tol_metric):
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
            cfg_cap: Dict[str, Any] = cfg
            frac_pass: Optional[float] = None
            if pixel_direct_mode:
                cfg_cap = dict(cfg)
                cfg_cap["yolo_center_max_step_scale_by_pixel_error"] = True
                dh = cfg.get("yolo_center_direct_max_step_m_hard_abs")
                if dh is not None:
                    cfg_cap["yolo_center_max_step_m_hard_abs"] = float(dh)
                frac_pass = direct_frac_kw

            ms_adaptive = 0.0 if pixel_direct_mode else float(max_step)
            max_step = self._yolo_center_error_based_max_step_m(
                du,
                dv,
                z_ray,
                fx,
                fy,
                step_gain=step_gain,
                adaptive_cap=ms_adaptive,
                cfg=cfg_cap,
                fraction=frac_pass,
            )
            pose_goal = self._tool_pose_lateral_nudge_from_pixels(
                du,
                dv,
                z_ray,
                fx,
                fy,
                step_gain=step_gain,
                max_step_m=max_step,
                nudge_sign=nudge_sign,
                sign_u=sign_u,
                sign_v=sign_v,
                optical_frame=optical,
                base_frame=base,
                tf_timeout_sec=tf_timeout,
            )
            if pose_goal is None:
                self.get_logger().warn("yolo_visual_center: lateral nudge pose unavailable (TF?).")
                continue
            waypoints = self._yolo_build_smooth_centering_waypoints(
                pose_goal,
                segments=max(
                    1,
                    (
                        int(cfg.get("yolo_center_direct_path_segments", 1))
                        if pixel_direct_mode
                        else path_segments
                    ),
                ),
                profile=wp_prof,
            )
            if not waypoints:
                continue
            pose_now = self._lookup_tool0_pose()
            planned_mm = 0.0
            if pose_now is not None:
                planned_mm = self._pose_translation_m(pose_now, waypoints[-1]) * 1000.0
            nudge_count += 1
            stream_cfg = dict(cfg)
            stream_cfg["yolo_center_velocity_scaling"] = float(v_scale)
            phase_tag_a = ""
            if pixel_direct_mode:
                phase_tag_a = " direct"
            elif coarse_phase:
                phase_tag_a = " coarse"
            cap_floor = (
                f" floor_cap={ms_adaptive:.3f}"
                if abs(max_step - ms_adaptive) > 0.00075
                else ""
            )
            stream_tag = "moveit_ik" if ik_only else "cartesian stream"
            self.get_logger().info(
                f"yolo_visual_center {stream_tag} {nudge_count}/{max_iter}: "
                f"offset_px=({du:.1f},{dv:.1f}) |Δ|={metric:.0f}px{phase_tag_a} "
                f"gain={step_gain:.2f} cap={max_step:.3f}m{cap_floor} v_scale={v_scale:.3f} "
                f"path={len(waypoints)} wp total_shift≈{planned_mm:.1f} mm depth={z_ray:.3f}"
            )
            used_orient_fallback = False
            goal_pose = waypoints[-1]
            if ik_only:
                moved, delta_m = self._yolo_execute_centering_moveit_ik(
                    center_action, goal_pose, stream_cfg, min_move_m=min_move_m
                )
            else:
                moved, delta_m = self._yolo_execute_centering_cartesian_stream(
                    center_action, waypoints, stream_cfg, min_move_m=min_move_m
                )
            if not moved and orient_fallback and orient_count < orient_max:
                pose_orient = self._tool_pose_orient_nudge_from_pixels(
                    du,
                    dv,
                    fx=fx,
                    fy=fy,
                    cfg=cfg,
                    nudge_sign=nudge_sign,
                    optical_frame=optical,
                    base_frame=base,
                    tf_timeout_sec=tf_timeout,
                )
                if pose_orient is not None:
                    orient_count += 1
                    o_segments = max(1, int(cfg.get("yolo_center_orient_path_segments", 6)))
                    o_waypoints = self._yolo_build_smooth_orient_centering_waypoints(
                        pose_orient,
                        segments=o_segments,
                        profile=wp_prof,
                    )
                    if o_waypoints:
                        pose_now_o = self._lookup_tool0_pose()
                        planned_deg = 0.0
                        if pose_now_o is not None:
                            planned_deg = math.degrees(
                                self._pose_rotation_rad(pose_now_o, o_waypoints[-1])
                            )
                        self.get_logger().warn(
                            f"yolo_visual_center: lateral nudge {nudge_count} blocked — "
                            f"optical orient fallback {orient_count}/{orient_max} "
                            f"(offset_px=({du:.1f},{dv:.1f}), planned≈{planned_deg:.2f}°)."
                        )
                        if ik_only:
                            moved, delta_m = self._yolo_execute_centering_moveit_ik(
                                center_action, o_waypoints[-1], stream_cfg, min_move_m=min_move_m
                            )
                        else:
                            moved, delta_m = self._yolo_execute_centering_cartesian_stream(
                                center_action, o_waypoints, stream_cfg, min_move_m=min_move_m
                            )
                        used_orient_fallback = moved
            if not moved:
                fail_modes = "MoveIt IK / orient" if ik_only else "Cartesian / IK / orient"
                self.get_logger().warn(
                    f"yolo_visual_center: smooth centering nudge {nudge_count} failed ({fail_modes})."
                )
                continue
            # Anchor redetect to the NEW pose so we never snap back (avoids shake).
            _save_last_detection_pose()
            move_tag = "orient" if used_orient_fallback else ("moveit_ik" if ik_only else "cartesian")
            if used_orient_fallback and pose_now is not None:
                after_move = self._lookup_tool0_pose()
                if after_move is not None:
                    dtheta = math.degrees(self._pose_rotation_rad(pose_now, after_move))
                    move_tag = f"orient(Δθ≈{dtheta:.2f}°)"
            self.get_logger().info(
                f"yolo_visual_center: {move_tag} stream moved {delta_m * 1000.0:.1f} mm "
                f"— refresh vision ({stream_spin_frames}×{stream_spin_period:.3f}s)."
            )
            if stream_spin_frames > 0:
                self._spin_vision_settle(stream_spin_frames, stream_spin_period)
            else:
                rclpy.spin_once(self, timeout_sec=0.0)

        if nudge_count > 0 and bool(cfg.get("yolo_center_allow_soft_complete", False)):
            ok_sc, det_sc, keep_sc, dr_sc, miss_sc = _track_frame(
                "yolo_visual_center cartesian soft-complete"
            )
            if ok_sc and det_sc is not None and not keep_sc and not miss_sc:
                du_sc, dv_sc, _ = self._yolo_bbox_offsets(det_sc, dr_sc, prefer_raw=prefer_raw)
                tol_soft = self._yolo_relaxed_center_tolerance_px(cfg, tol_px)
                if self._yolo_bbox_centered(du_sc, dv_sc, tol_soft, metric=tol_metric):
                    self.get_logger().warn(
                        f"yolo_visual_center: soft-complete after {nudge_count} nudge(s) "
                        f"(Δu={du_sc:.1f}, Δv={dv_sc:.1f} px, relaxed tolerance={tol_soft:.1f})."
                    )
                    return True

        fail_label = "MoveIt IK centering" if ik_only else "Cartesian centering"
        self.get_logger().error(
            f"yolo_visual_center: {fail_label} did not converge "
            f"({nudge_count} nudge(s), tolerance={tol_px:.1f} px, metric={tol_metric})."
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
        """IK + MoveGroup joint plan (no straight-line Cartesian path)."""
        joint_positions = self._solve_ik_scaled_steps(pose, action)
        if joint_positions is None:
            return False
        return self._execute_joint_goal(action, joint_positions)

    def _move_tool_pose_ik_joint_scaled(
        self,
        action: Dict[str, Any],
        pose: Pose,
        step_scales: Sequence[float],
    ) -> bool:
        """IK + joint plan using explicit position-interpolation scales."""
        joint_positions = self._solve_ik_with_step_scales(pose, action, step_scales)
        if joint_positions is None:
            return False
        return self._execute_joint_goal(action, joint_positions)

    def _move_tool_pose_plunge_full(self, action: Dict[str, Any], pose: Pose) -> bool:
        """Straight-line Cartesian plunge; IK fallback only at step_scale=1.0 (no partial IK)."""
        min_fraction = float(action.get("min_fraction", 1.0))
        cres = self._execute_cartesian_path(action, [pose], min_fraction)
        if cres:
            return True
        self.get_logger().warn(
            "Plunge Cartesian path incomplete or failed — trying IK at step_scale=1.0 only."
        )
        joint_positions = self._solve_ik_with_step_scales(pose, action, (1.0,))
        if joint_positions is None:
            self.get_logger().error("Plunge IK failed: no solution at full step (step_scale=1.0).")
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

    def _yolo_lookat_hint_vector(self, cfg: Dict[str, Any]) -> np.ndarray:
        raw = cfg.get("yolo_visual_center_lookat_hint", [0.0, 0.0, -1.0])
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
        return _normalize(
            np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float64),
            "yolo_visual_center_lookat_hint",
        )

    @staticmethod
    def _rotation_matrix_local_x(angle_rad: float) -> np.ndarray:
        """Rotation about the tool / TCP +X axis (positive angle → camera looks slightly lower when raised)."""
        c = math.cos(float(angle_rad))
        s = math.sin(float(angle_rad))
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)

    def _yolo_recovery_offsets(self, _action: Dict[str, Any], _cfg: Dict[str, Any]) -> tuple[float, float]:
        """Centering IK never bakes in recovery; use _yolo_execute_recovery_adjustment between sessions."""
        return 0.0, 0.0

    def _yolo_recovery_step_sizes(self, action: Dict[str, Any], cfg: Dict[str, Any]) -> tuple[float, float]:
        """Per recovery session: ΔZ (base +Z) and Δroll (tool +X), applied only in recovery motion (not in centering IK)."""
        lift_step = float(cfg.get("yolo_visual_center_recovery_lift_step_m", 0.04))
        roll_step = float(cfg.get("yolo_visual_center_recovery_roll_step_deg", 5.0))
        if "yolo_visual_center_recovery_lift_step_m" in action:
            lift_step = float(action["yolo_visual_center_recovery_lift_step_m"])
        roll_key = "yolo_visual_center_recovery_roll_step_deg"
        if roll_key not in action and "yolo_visual_center_recovery_roll_add_deg" in action:
            roll_key = "yolo_visual_center_recovery_roll_add_deg"
        if roll_key in action:
            roll_step = float(action[roll_key])
        return lift_step, roll_step

    def _yolo_tool_pose_apply_recovery_delta(
        self, pose_now: Pose, *, lift_m: float, roll_deg: float
    ) -> Pose:
        """Current tool0 pose + base +Z lift + roll about tool +X (recovery-only motion)."""
        out = Pose()
        out.position.x = float(pose_now.position.x)
        out.position.y = float(pose_now.position.y)
        out.position.z = float(pose_now.position.z + float(lift_m))
        q = np.array(
            [
                pose_now.orientation.x,
                pose_now.orientation.y,
                pose_now.orientation.z,
                pose_now.orientation.w,
            ],
            dtype=np.float64,
        )
        R = _quat_to_rot(q)
        if abs(float(roll_deg)) > 1e-6:
            R = _orthonormalize_rotation(
                R @ self._rotation_matrix_local_x(math.radians(float(roll_deg)))
            )
        qn = _rot_to_quat(R)
        out.orientation.x = float(qn[0])
        out.orientation.y = float(qn[1])
        out.orientation.z = float(qn[2])
        out.orientation.w = float(qn[3])
        return out

    def _yolo_execute_recovery_adjustment(
        self,
        action: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        lift_m: float,
        roll_deg: float,
        tool_frame: str,
        tf_timeout: float,
    ) -> bool:
        """Move arm higher (+Z) and roll about tool +X; does not include bbox centering."""
        pose_now = self._lookup_tool0_pose()
        if pose_now is None:
            self.get_logger().error("yolo_visual_center recovery: could not read current tool0 pose.")
            return False
        goal = self._yolo_tool_pose_apply_recovery_delta(pose_now, lift_m=lift_m, roll_deg=roll_deg)
        move_action = dict(action)
        move_action["avoid_collisions"] = bool(cfg.get("yolo_visual_center_avoid_collisions", True))
        self.get_logger().info(
            "yolo_visual_center: recovery motion (no centering) — "
            f"base +Z {float(lift_m) * 1000.0:.1f} mm, tool +X roll {float(roll_deg):+.1f}°."
        )
        if not self._move_yolo_center_ik(move_action, cfg, goal):
            self.get_logger().error("yolo_visual_center recovery: MoveIt IK failed for recovery motion.")
            return False
        settle_n = max(0, int(cfg.get("yolo_visual_center_recovery_settle_frames", 12)))
        settle_p = float(cfg.get("yolo_center_after_stream_spin_period_sec", 0.04))
        if settle_n > 0:
            self._spin_vision_settle(settle_n, settle_p)
        return True

    def _yolo_redetect_after_recovery(
        self,
        model: Any,
        det_track: _YoloDetectionTrack,
        *,
        target_class: str,
        min_conf: float,
        yolo_iou: float,
        roi: int,
        ray_kw: Dict[str, Any],
        cfg: Dict[str, Any],
    ) -> Optional[tuple[Any, Any]]:
        """Fresh YOLO detection after recovery motion (restart centering from new view)."""
        attempts = max(5, int(cfg.get("yolo_visual_center_recovery_redetect_attempts", 40)))
        spin_sec = float(cfg.get("yolo_visual_center_acquire_spin_sec", 0.05))
        for idx in range(attempts):
            self._raise_if_abort()
            rclpy.spin_once(self, timeout_sec=spin_sec)
            det_raw = self._yolo_snapshot_and_detect(
                model,
                target_class=target_class,
                min_conf=min_conf,
                yolo_iou=yolo_iou,
                roi=roi,
                ray_kw=ray_kw,
            )
            if det_raw is None:
                continue
            ok, det, _ = det_track.push(
                det_raw,
                log_fn=self.get_logger().info,
                phase_tag="yolo_visual_center after recovery",
            )
            if ok and det is not None:
                self.get_logger().info(
                    f"yolo_visual_center: redetected {target_class!r} after recovery "
                    f"(attempt {idx + 1}/{attempts})."
                )
                return det_raw, det
        self.get_logger().error(
            f"yolo_visual_center: no {target_class!r} detection after recovery ({attempts} frames)."
        )
        return None

    def _tool_pose_center_for_yolo(
        self,
        x_cam: float,
        y_cam: float,
        z_cam: float,
        z_goal_m: float,
        *,
        recovery_lift_m: float = 0.0,
        recovery_roll_deg: float = 0.0,
        nudge_sign: float = 1.0,
        sign_u: float = -1.0,
        sign_v: float = -1.0,
        optical_frame: str,
        base_frame: str,
        tool_frame: str,
        tf_timeout_sec: float,
    ) -> Optional[Pose]:
        """
        Center bbox + depth keeping current tool0 orientation; optional base +Z lift and +X roll.

        Does not impose base_link [0,0,-1] lookat — preserves Look pose RPY, adds roll in tool frame.
        """
        pose = self._tool_pose_center_object_at_depth(
            x_cam,
            y_cam,
            z_cam,
            z_goal_m,
            nudge_sign=nudge_sign,
            sign_u=sign_u,
            sign_v=sign_v,
            optical_frame=optical_frame,
            base_frame=base_frame,
            tool_frame=tool_frame,
            tf_timeout_sec=tf_timeout_sec,
        )
        if pose is None:
            return None
        if float(recovery_lift_m) != 0.0:
            pose.position.z = float(pose.position.z + float(recovery_lift_m))
        if abs(float(recovery_roll_deg)) > 1e-6:
            q = np.array(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                dtype=np.float64,
            )
            R = _quat_to_rot(q)
            R = _orthonormalize_rotation(R @ self._rotation_matrix_local_x(math.radians(float(recovery_roll_deg))))
            qn = _rot_to_quat(R)
            pose.orientation.x = float(qn[0])
            pose.orientation.y = float(qn[1])
            pose.orientation.z = float(qn[2])
            pose.orientation.w = float(qn[3])
        return pose

    def _yolo_relaxed_pose_candidates_from_goal(
        self,
        goal_pose: Pose,
        cfg: Dict[str, Any],
        *,
        recovery_lift_m: float = 0.0,
        recovery_roll_deg: float = 0.0,
    ) -> list[tuple[str, Pose]]:
        """Stream centering: goal pose plus optional lift / tool +X roll on current orientation."""
        candidates: list[tuple[str, Pose]] = [("stream_goal", goal_pose)]
        if abs(recovery_lift_m) < 1e-6 and abs(recovery_roll_deg) < 1e-6:
            return candidates
        alt = Pose()
        alt.position.x = float(goal_pose.position.x)
        alt.position.y = float(goal_pose.position.y)
        alt.position.z = float(goal_pose.position.z + float(recovery_lift_m))
        alt.orientation = goal_pose.orientation
        if abs(recovery_roll_deg) > 1e-6:
            q = np.array(
                [
                    alt.orientation.x,
                    alt.orientation.y,
                    alt.orientation.z,
                    alt.orientation.w,
                ],
                dtype=np.float64,
            )
            R = _quat_to_rot(q)
            R = _orthonormalize_rotation(
                R @ self._rotation_matrix_local_x(math.radians(float(recovery_roll_deg)))
            )
            qn = _rot_to_quat(R)
            alt.orientation.x = float(qn[0])
            alt.orientation.y = float(qn[1])
            alt.orientation.z = float(qn[2])
            alt.orientation.w = float(qn[3])
        candidates.append(
            (
                f"recovery_lift_{recovery_lift_m * 1000.0:.0f}mm_roll_{recovery_roll_deg:.1f}deg",
                alt,
            )
        )
        return candidates

    def _move_yolo_center_ik(
        self,
        action: Dict[str, Any],
        cfg: Dict[str, Any],
        pose: Pose,
    ) -> bool:
        scales_raw = cfg.get("yolo_visual_center_ik_step_scales", [1.0])
        scales = tuple(float(s) for s in scales_raw) if isinstance(scales_raw, (list, tuple)) else (1.0,)
        return self._move_tool_pose_ik_joint_scaled(action, pose, scales)

    def _tool_pose_center_object_at_depth(
        self,
        x_cam: float,
        y_cam: float,
        z_cam: float,
        z_goal_m: float,
        *,
        nudge_sign: float = 1.0,
        sign_u: float = -1.0,
        sign_v: float = -1.0,
        optical_frame: str,
        base_frame: str,
        tool_frame: str,
        tf_timeout_sec: float,
    ) -> Optional[Pose]:
        """
        Single tool0 goal: center object in image and set optical depth to ``z_goal_m``.

        Uses the same camera-frame translation convention as ``_tool_pose_lateral_nudge_from_pixels``
        (translate tool0 in base, keep orientation) plus optical +Z for depth — NOT
        ``_tool_pose_shift_optical_origin``, which inverts lateral direction for this rig.
        """
        pose_now = self._lookup_tool0_pose()
        if pose_now is None or float(z_cam) <= 1e-6:
            return None
        T_base_opt = self._lookup_tf_mat(base_frame, optical_frame, tf_timeout_sec)
        if T_base_opt is None:
            return None
        R = T_base_opt[:3, :3]
        sign = float(nudge_sign)
        su = float(sign_u)
        sv = float(sign_v)
        delta_cam = np.array(
            [
                -sign * su * float(x_cam),
                -sign * sv * float(y_cam),
                float(z_cam) - float(z_goal_m),
            ],
            dtype=np.float64,
        )
        delta_base = R @ delta_cam
        out = Pose()
        out.orientation = pose_now.orientation
        out.position.x = float(pose_now.position.x + delta_base[0])
        out.position.y = float(pose_now.position.y + delta_base[1])
        out.position.z = float(pose_now.position.z + delta_base[2])
        return out

    def _yolo_moveit_one_shot_center_and_standoff(
        self,
        action: Dict[str, Any],
        cfg: Dict[str, Any],
        model: Any,
        *,
        det: tuple[int, int, float, float, float, float, float, float],
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
        z_goal_m: float,
        tol_z_m: float,
    ) -> bool:
        """One MoveGroup plan: center bbox at depth, keep current tool0 orientation (no recovery offsets)."""
        _u, _v, x_cam, y_cam, z_cam, du, dv, conf = det
        nudge_sign = float(cfg.get("yolo_center_nudge_sign", 1.0))
        sign_u = float(cfg.get("yolo_center_orient_sign_u", cfg.get("yolo_servo_ibvs_sign_u", -1.0)))
        sign_v = float(cfg.get("yolo_center_orient_sign_v", cfg.get("yolo_servo_ibvs_sign_v", -1.0)))
        pose_goal = self._tool_pose_center_for_yolo(
            x_cam,
            y_cam,
            z_cam,
            z_goal_m,
            recovery_lift_m=0.0,
            recovery_roll_deg=0.0,
            nudge_sign=nudge_sign,
            sign_u=sign_u,
            sign_v=sign_v,
            optical_frame=optical,
            base_frame=base,
            tool_frame=tool_frame,
            tf_timeout_sec=tf_timeout,
        )
        if pose_goal is None:
            self.get_logger().error("yolo_visual_center moveit_one_shot: goal pose TF failed.")
            return False

        delta_cam = np.array(
            [
                -nudge_sign * sign_u * float(x_cam),
                -nudge_sign * sign_v * float(y_cam),
                float(z_cam) - float(z_goal_m),
            ],
            dtype=np.float64,
        )
        shift_mm = 1000.0 * float(np.linalg.norm(delta_cam))
        move_action = dict(action)
        move_action["avoid_collisions"] = bool(cfg.get("yolo_visual_center_avoid_collisions", True))
        self.get_logger().info(
            "yolo_visual_center moveit_one_shot: MoveGroup IK (Look orientation, no recovery offsets). "
            f"object optical=({x_cam:.4f}, {y_cam:.4f}, {z_cam:.3f}) m, "
            f"Δu={du:.1f} Δv={dv:.1f} px, goal depth={z_goal_m:.3f} m, "
            f"tool0 Δ_cam=({delta_cam[0]:.4f}, {delta_cam[1]:.4f}, {delta_cam[2]:.4f}) m "
            f"(|Δ|≈{shift_mm:.1f} mm), conf={conf:.3f}."
        )
        if not self._move_yolo_center_ik(move_action, cfg, pose_goal):
            self.get_logger().error("yolo_visual_center moveit_one_shot: MoveIt IK / joint plan failed.")
            return False

        settle_n = max(0, int(cfg.get("yolo_center_after_stream_spin_frames", 8)))
        settle_p = float(cfg.get("yolo_center_after_stream_spin_period_sec", 0.04))
        if settle_n > 0:
            self._spin_vision_settle(settle_n, settle_p)

        tol_metric = self._yolo_center_tolerance_metric(cfg)
        det_after = self._yolo_snapshot_and_detect(
            model,
            target_class=target_class,
            min_conf=min_conf,
            yolo_iou=yolo_iou,
            roi=roi,
            ray_kw=ray_kw,
        )
        if det_after is None:
            self.get_logger().error("yolo_visual_center moveit_one_shot: no detection after move.")
            return False
        _u2, _v2, _xc, _yc, z2, du2, dv2, conf2 = det_after
        centered = self._yolo_bbox_centered(du2, dv2, tol_px, metric=tol_metric)
        depth_ok = abs(float(z2) - float(z_goal_m)) <= float(tol_z_m)
        if centered and depth_ok:
            self.get_logger().info(
                f"yolo_visual_center moveit_one_shot: success (Δu={du2:.1f}, Δv={dv2:.1f} px, "
                f"z={z2:.3f} m, conf={conf2:.3f})."
            )
            return True
        self.get_logger().error(
            f"yolo_visual_center moveit_one_shot: verify failed "
            f"(centered={centered}, depth_ok={depth_ok}, Δu={du2:.1f}, Δv={dv2:.1f}, "
            f"z={z2:.3f} m vs goal {z_goal_m:.3f}±{tol_z_m:.3f})."
        )
        return False

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
            if min_det_z > 0.0 and z_est < min_det_z and step > 0.0:
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
        """Interpolate IK link **position** toward ``pose``; retries with smaller steps."""
        custom = action.get("ik_step_scales")
        if custom is not None:
            return self._solve_ik_with_step_scales(pose, action, tuple(float(x) for x in custom))
        return self._solve_ik_with_step_scales(pose, action, (1.0, 0.5, 0.25, 0.125))

    def _solve_ik_with_step_scales_result(
        self,
        pose: Pose,
        action: Dict[str, Any],
        step_scales: Sequence[float],
    ) -> Optional[tuple[list[float], float]]:
        """Interpolate IK link position toward absolute ``pose``; return (joints, step_scale) or None."""
        ik_link = str(action.get("ik_link_name", self.motion_cfg["ik_link_name"])).strip()
        scales = tuple(float(s) for s in step_scales)
        for idx, step_scale in enumerate(scales):
            pose_now = self._lookup_link_pose(ik_link)
            if pose_now is None:
                return None
            tgt = Pose()
            tgt.orientation = pose.orientation
            tgt.position.x = float(pose_now.position.x + step_scale * (pose.position.x - pose_now.position.x))
            tgt.position.y = float(pose_now.position.y + step_scale * (pose.position.y - pose_now.position.y))
            tgt.position.z = float(pose_now.position.z + step_scale * (pose.position.z - pose_now.position.z))
            is_last = idx == len(scales) - 1
            joint_positions = self._solve_ik(tgt, action, silent_ik_failure=not is_last)
            if joint_positions is not None:
                return joint_positions, float(step_scale)
        return None

    def _solve_ik_with_step_scales(
        self,
        pose: Pose,
        action: Dict[str, Any],
        step_scales: Sequence[float],
    ) -> Optional[list[float]]:
        """Single-shot IK toward ``pose`` (first successful scale only — prefer hover convergence for HOVER)."""
        result = self._solve_ik_with_step_scales_result(pose, action, step_scales)
        if result is None:
            return None
        joint_positions, step_scale = result
        if step_scale < 1.0 - 1e-6:
            self.get_logger().warn(f"IK succeeded with step_scale={step_scale}.")
        return joint_positions

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

    def _lookup_link_pose(self, link_name: str) -> Optional[Pose]:
        base = str(self.motion_cfg["base_frame"])
        link = str(link_name).strip()
        try:
            t = self._tf_buffer.lookup_transform(base, link, Time(), timeout=RclDuration(seconds=3.0))
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"TF {base} -> {link} failed: {exc}")
            return None
        pose = Pose()
        pose.position.x = float(t.transform.translation.x)
        pose.position.y = float(t.transform.translation.y)
        pose.position.z = float(t.transform.translation.z)
        pose.orientation = t.transform.rotation
        return pose

    def _lookup_tool0_pose(self) -> Optional[Pose]:
        return self._lookup_link_pose("tool0")

    def _execute_yolo_visual_center_action(self, action: Dict[str, Any]) -> bool:
        """
        YOLO + RGB-D: center bbox in image, then approach along optical +Z.

        Approach depth goal is ``dist_approach``: normally ``target_distance_m``, or when
        ``yolo_visual_center_approach_forward_m`` > 0, ``max(z_now - forward_m, floor)`` where
        ``floor`` uses ``min_detection_depth_m`` and optionally ``target_distance_m``
        (``yolo_visual_center_approach_target_as_floor``). If Cartesian/IK cannot finish the motion,
        ``yolo_visual_center_approach_partial_ok`` (default: same as forward mode active) accepts
        the last reachable pose.

        Centering mode ``yolo_visual_center_centering_mode``:
        - ``cartesian`` (default): streamed Cartesian optical nudges (discrete streamed paths).
        - ``servo`` / ``smooth``: MoveIt Servo IBVS twists (legacy tuning in YAML).
        - ``servo_smooth`` / ``pure_servo`` / ``servo_continuous``: Servo-only with continuous preset (one IBVS
          phase; no Cartesian nudges, no coarse/fine servo split, no cross-brake frames by default).
        - ``cartesian_then_servo`` / ``cart_servo``: Cartesian until |Δu|+|Δv| below handoff, then ``servo_smooth`` preset — continuous fine correction without Cartesian replans.
        - ``servo_then_cartesian`` / ``hybrid`` / ``blend``: time-capped coarse Servo (no stall
          Cartesian inside Servo loop), then Cartesian finish.
        - ``moveit_ik`` / ``ik_joint``: same YOLO goal poses; reach them with /compute_ik +
          MoveGroup joint planning only (no straight-line Cartesian — reduces arm self-collision).
        - ``moveit_one_shot`` / ``one_shot``: after Look, one YOLO sample → tool0 goal with object at
          image center and ``target_distance_m`` depth → **one** MoveGroup IK plan (no Cartesian, no approach loop).
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
        fwd_m = float(max(0.0, float(cfg.get("yolo_visual_center_approach_forward_m", 0.0) or 0.0)))
        partial_raw = cfg.get("yolo_visual_center_approach_partial_ok")
        approach_partial_ok = (fwd_m > 1e-6) if partial_raw is None else bool(partial_raw)
        approach_iters = max(1, int(cfg["approach_max_iterations"]))
        approach_clip = float(cfg["approach_step_max_m"])
        single_motion = bool(cfg.get("yolo_visual_center_single_motion", False))
        single_trans_cap = float(max(0.01, cfg.get("yolo_visual_center_single_motion_max_translation_m", 2.0)))

        tol_px = float(cfg["center_tolerance_px"])
        tol_metric = self._yolo_center_tolerance_metric(cfg)
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
        use_moveit_ik = bool(cfg.get("yolo_visual_center_use_moveit_ik", False)) or centering_mode in (
            "moveit_ik",
            "ik_joint",
            "ik",
            "moveit_joint",
            "joint_ik",
        )
        if use_moveit_ik:
            cfg["yolo_visual_center_use_moveit_ik"] = True
            if "yolo_visual_center_avoid_collisions" not in action:
                cfg["yolo_visual_center_avoid_collisions"] = True
        servo_like_centering = centering_mode in (
            "servo",
            "smooth",
            "moveit_servo",
            "ibvs",
            "velocity",
            "servo_then_cartesian",
            "hybrid",
            "blend",
            "cartesian_then_servo",
            "cart_servo",
            "servo_smooth",
            "pure_servo",
            "servo_continuous",
        )
        plan_cap = max(1, int(cfg.get("yolo_center_max_plan_iterations", 3)))
        cen_req = max(1, int(cfg.get("center_max_iterations") or cfg.get("max_iterations", plan_cap)))
        self.get_logger().info(
            f"yolo_visual_center: class={target_class!r}, min_confidence={min_conf}, "
            f"target_distance_m={dist_target:.3f}, approach_forward_m={fwd_m:.3f}, "
            f"approach_partial_ok={approach_partial_ok}, centering_mode={centering_mode}, "
            f"center_streams_cap={plan_cap}, center_requested={cen_req}, center_effective={min(plan_cap, cen_req)}, "
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
            "depth_valid_min_m": float(cfg.get("yolo_depth_valid_min_m", 0.03)),
            "depth_valid_max_m": float(cfg.get("yolo_depth_valid_max_m", 12.0)),
            "depth_min_bbox_pixels": int(cfg.get("yolo_depth_min_bbox_pixels", 12)),
        }
        excl = cfg.get("yolo_exclusive_scene_classes")
        if isinstance(excl, (list, tuple)) and len(excl) > 0:
            ray_kw["exclusive_scene_classes"] = [str(x).strip() for x in excl if str(x).strip()]
        pick_kw = _resolve_yolo_pick_among_kwargs(cfg, target_class)
        ray_kw["pick_among_strategy"] = pick_kw["pick_among_strategy"]
        if pick_kw["preferred_uv"] is not None:
            ray_kw["preferred_uv"] = pick_kw["preferred_uv"]
        scene_kw = _resolve_yolo_scene_assign_kwargs(self.vision_cfg, cfg)
        ray_kw.update(scene_kw)
        if scene_kw.get("scene_unique_classes") and scene_kw.get("scene_assign_exclusive_classes"):
            self.get_logger().info(
                "yolo scene unique: assign "
                f"{scene_kw['scene_assign_exclusive_classes']!r} → pick {target_class!r}"
            )
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
            if servo_like_centering:
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
        acquire_diag_done = False
        self.get_logger().info(
            f"yolo_visual_center: acquisition uses min_confidence={acquire_conf:.2f} "
            f"(tracking uses {min_conf:.2f})."
        )
        while acquire_attempts < acquire_n:
            self._raise_if_abort()
            rclpy.spin_once(self, timeout_sec=0.05)
            det_raw_acquire = self._yolo_snapshot_and_detect(
                model,
                target_class=target_class,
                min_conf=acquire_conf,
                yolo_iou=yolo_iou,
                roi=roi,
                ray_kw=ray_kw,
            )
            if (
                det_raw_acquire is None
                and acquire_attempts >= 30
                and not acquire_diag_done
            ):
                acquire_diag_done = True
                self._yolo_acquire_debug_banner(
                    model,
                    target_class=target_class,
                    acquire_conf=acquire_conf,
                    yolo_iou=yolo_iou,
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

        one_shot_mode = centering_mode in (
            "moveit_one_shot",
            "single_moveit",
            "moveit_single",
            "one_shot",
        )
        if one_shot_mode:
            if det is None:
                self.get_logger().error("yolo_visual_center moveit_one_shot: no detection after acquire.")
                return False
            prefer_raw_os = bool(cfg.get("yolo_center_use_raw_detection", True))
            z_goal = float(dist_target)
            max_retries = max(0, int(cfg.get("yolo_moveit_one_shot_max_retries", 3)))
            max_recovery_sessions = max(
                0, int(cfg.get("yolo_visual_center_recovery_max_sessions", 3))
            )
            lift_step, roll_step = self._yolo_recovery_step_sizes(action, cfg)
            self.get_logger().info(
                "yolo_visual_center moveit_one_shot: center from Look (recovery offsets=0); "
                f"on fail → recovery motion (+Z {lift_step * 1000:.1f} mm, tool +X roll {roll_step:+.1f}°) "
                f"→ redetect → restart centering (max {max_recovery_sessions} recovery session(s))."
            )
            for session in range(max_recovery_sessions + 1):
                self._raise_if_abort()
                if session > 0:
                    self.get_logger().info(
                        f"yolo_visual_center: recovery session {session}/{max_recovery_sessions} "
                        "— restarting full centering from fresh detection."
                    )
                centered_ok = False
                for attempt in range(max_retries + 1):
                    self._raise_if_abort()
                    sample = det_raw if prefer_raw_os and det_raw is not None else det
                    if attempt > 0:
                        ok_r, det_r, ka_r, dr_r, miss_r = _track_frame(
                            f"moveit_one_shot retry {attempt} (session {session})"
                        )
                        if miss_r or not ok_r or det_r is None or ka_r:
                            self.get_logger().warn(
                                f"yolo_visual_center moveit_one_shot: session {session} "
                                f"retry {attempt} — no live detection."
                            )
                            continue
                        sample = dr_r if prefer_raw_os and dr_r is not None else det_r
                        det = det_r
                        if dr_r is not None:
                            det_raw = dr_r
                    if self._yolo_moveit_one_shot_center_and_standoff(
                        action,
                        cfg,
                        model,
                        det=sample,
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
                        z_goal_m=z_goal,
                        tol_z_m=tol_z,
                    ):
                        centered_ok = True
                        break
                if centered_ok:
                    try:
                        self._foundation_pose_record_bbox_snapshot_after_yolo(
                            model=model,
                            target_class=target_class,
                            min_conf=min_conf,
                            yolo_iou=yolo_iou,
                            ray_kw=ray_kw,
                            depth_z_med=float(sample[4]),
                        )
                    except Exception as exc:  # noqa: BLE001
                        self.get_logger().debug(f"foundation_pose bbox snapshot skipped: {exc}")
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
                        phase_tag="after moveit_one_shot",
                    )
                if session >= max_recovery_sessions:
                    break
                if not self._yolo_execute_recovery_adjustment(
                    action,
                    cfg,
                    lift_m=lift_step,
                    roll_deg=roll_step,
                    tool_frame=tool_frame,
                    tf_timeout=tf_timeout,
                ):
                    self.get_logger().warn(
                        "yolo_visual_center: recovery motion failed — will still try redetect."
                    )
                redetect = self._yolo_redetect_after_recovery(
                    model,
                    det_track,
                    target_class=target_class,
                    min_conf=min_conf,
                    yolo_iou=yolo_iou,
                    roi=roi,
                    ray_kw=ray_kw,
                    cfg=cfg,
                )
                if redetect is None:
                    self.get_logger().error(
                        "yolo_visual_center: redetect after recovery failed — stopping."
                    )
                    return False
                det_raw, det = redetect
                _save_last_detection_pose()
            self.get_logger().error(
                "yolo_visual_center moveit_one_shot failed after "
                f"{max_recovery_sessions + 1} centering session(s) "
                f"({max_retries + 1} IK attempt(s) per session)."
            )
            return False

        # --- Phase 1: center bbox in the image.
        if single_motion and centering_mode in (
            "servo",
            "smooth",
            "moveit_servo",
            "ibvs",
            "velocity",
            "servo_then_cartesian",
            "hybrid",
            "blend",
            "servo_smooth",
            "pure_servo",
            "servo_continuous",
            "cartesian_then_servo",
            "cart_servo",
        ):
            self.get_logger().warn(
                "yolo_visual_center_single_motion: centering still runs servo (or hybrid servo phase); "
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
            ik_only=use_moveit_ik,
            _track_frame=_track_frame,
            _on_raw_miss=_on_raw_miss_for_centering,
            _save_last_detection_pose=_save_last_detection_pose,
        )
        if centering_mode in ("servo_then_cartesian", "hybrid", "blend"):
            hybrid_sec = float(max(0.5, cfg.get("yolo_servo_hybrid_coarse_sec", 6.0)))
            cfg_coarse = {
                **cfg,
                "yolo_servo_max_duration_sec": hybrid_sec,
                "yolo_servo_fine_pass": False,
                "yolo_servo_cartesian_fallback": False,
                # Mid-loop stall Cartesian bumps alternate Servo vs discrete moves → perceived shake.
                "yolo_servo_stall_cartesian_sec": 0.0,
            }
            self.get_logger().info(
                "yolo_visual_center: hybrid centering — coarse Servo "
                f"(≤{hybrid_sec:.1f}s, no stall Cartesian inside Servo), then Cartesian finish."
            )
            self._yolo_moveit_servo_center_bbox(
                action,
                cfg_coarse,
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
            centered = self._yolo_cartesian_center_bbox(**center_kw)
        elif centering_mode in ("cartesian_then_servo", "cart_servo"):
            handoff = float(cfg.get("yolo_cart_then_servo_handoff_metric_px", 92.0))
            handoff_max = float(cfg.get("yolo_cart_then_servo_handoff_max_px", 200.0))
            handoff = min(max(handoff, float(tol_px) * 2.5 + 10.0), max(handoff_max, float(tol_px) * 4.0))
            self.get_logger().info(
                "yolo_visual_center: Cartesian→Servo hybrid — Cartesian until coarse below "
                f"{handoff:.0f}px (|Δu|+|Δv|), then smooth Servo to final tolerance {tol_px:.0f}px."
            )
            coarse_kw = dict(center_kw, handoff_metric_px=handoff)
            if not self._yolo_cartesian_center_bbox(**coarse_kw):
                centered = False
            else:
                det_gate = self._yolo_snapshot_and_detect(
                    model,
                    target_class=target_class,
                    min_conf=min_conf,
                    yolo_iou=yolo_iou,
                    roi=roi,
                    ray_kw=ray_kw,
                )
                need_servo = True
                if det_gate is not None:
                    need_servo = not self._yolo_bbox_centered(
                        float(det_gate[5]), float(det_gate[6]), tol_px, metric=tol_metric
                    )
                if need_servo:
                    cfg_servo = self._yolo_cfg_for_smooth_servo_phase(cfg, tol_px)
                    centered = self._yolo_moveit_servo_center_bbox(
                        action,
                        cfg_servo,
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
                else:
                    self.get_logger().info(
                        "yolo_visual_center: Cartesian already satisfied pixel tolerance — skipping Servo."
                    )
                    centered = True
        elif centering_mode in ("servo_smooth", "pure_servo", "servo_continuous"):
            cfg_servo = self._yolo_cfg_for_smooth_servo_phase(cfg, tol_px)
            centered = self._yolo_moveit_servo_center_bbox(
                action,
                cfg_servo,
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
        elif centering_mode in ("servo", "smooth", "moveit_servo", "ibvs", "velocity"):
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
        elif centering_mode in ("moveit_ik", "ik_joint", "ik", "moveit_joint", "joint_ik"):
            centered = self._yolo_cartesian_center_bbox(**center_kw)
        elif centering_mode in ("cartesian", "default", ""):
            centered = self._yolo_cartesian_center_bbox(**center_kw)
        else:
            raise ValueError(
                f"Unsupported yolo_visual_center_centering_mode: {centering_mode!r} "
                "(use 'cartesian', 'moveit_ik', 'cartesian_then_servo', "
                "'servo_smooth'/pure_servo/servo_continuous, "
                "'servo'/'smooth', or 'servo_then_cartesian'/'hybrid')."
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

        def _approach_move_to_pose(pose_goal: Pose) -> bool:
            if bool(cfg.get("yolo_visual_center_use_moveit_ik", False)):
                lift_m, roll_deg = self._yolo_recovery_offsets(action, cfg)
                for _label, pose in self._yolo_relaxed_pose_candidates_from_goal(
                    pose_goal, cfg, recovery_lift_m=lift_m, recovery_roll_deg=roll_deg
                ):
                    if self._move_yolo_center_ik(action, cfg, pose):
                        return True
                return False
            return self._move_cartesian_waypoint_then_ik(action, pose_goal)

        def _approach_move_poses(poses: list[Pose]) -> bool:
            if bool(cfg.get("yolo_visual_center_use_moveit_ik", False)):
                return self._move_tool_poses_moveit_ik(action, poses)
            return self._move_cartesian_waypoints_then_ik(action, poses)

        def _approach_offsets(
            det: tuple[int, int, float, float, float, float, float, float],
            det_raw: Optional[tuple[int, int, float, float, float, float, float, float]],
        ) -> tuple[float, float, float]:
            return self._yolo_bbox_offsets(det, det_raw, prefer_raw=prefer_raw)

        def _recenter_at_depth_if_enabled(phase_tag: str) -> bool:
            if not bool(cfg.get("yolo_visual_center_recenter_at_depth", True)):
                return False
            rc_n = max(1, int(cfg.get("yolo_visual_center_recenter_at_depth_max_nudges", 3)))
            rc_cfg = dict(cfg)
            rc_cfg["yolo_center_max_plan_iterations"] = rc_n
            rc_cfg["max_iterations"] = rc_n
            self.get_logger().info(
                f"yolo_visual_center: {phase_tag} — micro-recenter at depth (≤{rc_n} nudges)."
            )
            return self._yolo_cartesian_center_bbox(
                **{**center_kw, "cfg": rc_cfg, "tol_px": tol_px}
            )

        def _require_centered_for_approach(
            du: float,
            dv: float,
            *,
            keep_alive: bool,
            phase: str,
            det: Optional[tuple[int, int, float, float, float, float, float, float]] = None,
            det_raw: Optional[tuple[int, int, float, float, float, float, float, float]] = None,
        ) -> bool:
            if keep_alive:
                self.get_logger().error(
                    f"yolo_visual_center: {phase} — cannot continue approach on keep-alive; need live detection."
                )
                return False
            if self._yolo_bbox_centered(du, dv, tol_px, metric=tol_metric):
                return True
            if bool(cfg.get("yolo_visual_center_recenter_on_approach_drift", True)):
                if _recenter_at_depth_if_enabled(f"{phase} drift recovery"):
                    ok_rc, det_rc, ka_rc, dr_rc, miss_rc = _track_frame(f"{phase} after recenter")
                    if ok_rc and det_rc is not None and not ka_rc and not miss_rc:
                        du_rc, dv_rc, _ = _approach_offsets(det_rc, dr_rc)
                        if self._yolo_bbox_centered(du_rc, dv_rc, tol_px, metric=tol_metric):
                            return True
            self.get_logger().error(
                f"yolo_visual_center: {phase} — bbox not centered "
                f"(Δu={du:.1f}, Δv={dv:.1f} px, tolerance={tol_px:.1f}, metric={tol_metric})."
            )
            return False

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

        dist_approach = float(dist_target)
        if fwd_m > 1e-6:
            ok_g, det_g, _, dr_g, miss_g = _track_frame("yolo_visual_center approach depth goal")
            if ok_g and det_g is not None and not miss_g:
                _, _, z_here = _approach_offsets(det_g, dr_g)
                floors: list[float] = [min_det_z]
                if bool(cfg.get("yolo_visual_center_approach_target_as_floor", True)):
                    floors.append(dist_target)
                z_floor = max(floors) if floors else 0.0
                dist_approach = max(z_here - fwd_m, z_floor)
                self.get_logger().info(
                    "yolo_visual_center: approach from measured depth "
                    f"z={z_here:.3f} m by up to {fwd_m:.3f} m → depth goal {dist_approach:.3f} m "
                    f"(floor max={z_floor:.3f} m)."
                )
            else:
                self.get_logger().warn(
                    "yolo_visual_center: approach_forward_m is set but depth goal sample failed — "
                    f"using absolute target_distance_m={dist_target:.3f} m."
                )

        # --- Phase 2: approach along optical +Z to dist_approach (absolute depth) or legacy target_distance_m.
        use_smooth_approach = bool(cfg.get("yolo_visual_center_smooth_approach", True))

        def _approach_step_list(z_cam_in: float) -> list[float]:
            return self._yolo_plan_optical_forward_steps(
                z_cam_in,
                dist_approach,
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
                if abs(z_cam0 - dist_approach) <= tol_z:
                    self.get_logger().info(
                        f"yolo_visual_center: depth already at goal ({z_cam0:.3f} m), conf={conf0:.3f}"
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
                        approach_label = "MoveIt IK" if cfg.get("yolo_visual_center_use_moveit_ik") else "Cartesian"
                        if not _approach_move_poses(poses):
                            self.get_logger().warn(
                                f"yolo_visual_center: smooth approach ({approach_label}) failed; "
                                "using stepped approach."
                            )
                        else:
                            ok2, det2, keep_alive2, det_raw2, raw_miss2 = _track_frame(
                                "yolo_visual_center approach verify"
                            )
                            if ok2 and det2 is not None:
                                du2, dv2, z2 = _approach_offsets(det2, det_raw2)
                                _, _, _, _, _, _, _, conf2 = det2
                                if abs(z2 - dist_approach) <= tol_z:
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
                                    if pose_fix is not None and _approach_move_to_pose(pose_fix):
                                        ok3, det3, _, dr3, _ = _track_frame(
                                            "yolo_visual_center approach verify"
                                        )
                                        if ok3 and det3 is not None:
                                            du3, dv3, z3 = _approach_offsets(det3, dr3)
                                            if abs(z3 - dist_approach) <= tol_z:
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

            if abs(z_cam - dist_approach) <= tol_z:
                self.get_logger().info(
                    f"yolo_visual_center: depth converged (~{dist_approach:.2f} m ± {tol_z:.3f}), conf={last_det_conf:.3f}"
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

            step = float(z_cam - dist_approach)
            # Only clamp when the planned step moves closer (ΔZ_optical > 0). If we are too close
            # to the object (z < target) step is negative (move away); forcing retreat>0 would
            # step the wrong way and depth diverges (see terminal: 0.338 → 0.324 m).
            if min_det_z > 0.0 and z_cam < min_det_z and step > 0.0:
                retreat = min_det_z - z_cam
                if step < retreat:
                    self.get_logger().info(
                        f"yolo_visual_center: depth {z_cam:.3f} m < min_detection {min_det_z:.3f} m — "
                        f"limiting forward approach step to {retreat:.3f} m (was {step:.3f} m)."
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
                if approach_partial_ok:
                    self.get_logger().warn(
                        "yolo_visual_center: approach TF/IK unavailable — partial_ok, stopping approach."
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
                        phase_tag="after partial approach",
                    )
                return False

            ka_tag = " [keep-alive]" if keep_alive else ""
            self.get_logger().info(
                f"yolo_visual_center approach {ap + 1}/{ap_cap}{ka_tag}: "
                f"ΔZ_optical_step={step:.3f} m, depth={z_cam:.3f}"
            )
            if not _approach_move_to_pose(pose_fwd):
                if approach_partial_ok:
                    fail_lbl = "MoveIt IK" if cfg.get("yolo_visual_center_use_moveit_ik") else "Cartesian/IK"
                    self.get_logger().warn(
                        f"yolo_visual_center: approach step failed ({fail_lbl}) — "
                        "partial_ok, accepting pose after partial travel."
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
                        phase_tag="after partial approach",
                    )
                return False
            ap += 1

        if approach_partial_ok:
            self.get_logger().warn(
                f"yolo_visual_center: approach ended before depth goal "
                f"{dist_approach:.2f} m ± {tol_z:.3f} (partial_ok — accepting current pose)."
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
                phase_tag="after partial approach",
            )
        self.get_logger().error(
            f"yolo_visual_center: exceeded {ap_cap} approach iteration(s) without reaching depth "
            f"{dist_approach:.2f} m ± {tol_z:.3f}."
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
        object_sequences = config.get("object_sequences", {})
        valid = ", ".join(sorted(str(name) for name in object_sequences)) if isinstance(object_sequences, dict) else ""
        object_name = first_object_name.strip()
        while True:
            if not object_name:
                if not sys.stdin.isatty():
                    raise RuntimeError(
                        "Interactive object prompt needs a real terminal (TTY). "
                        "Use one of:\n"
                        "  bash ~/ur3_control/run_action_sequencer.sh\n"
                        "  bash ~/ur3_control/run_action_sequencer.sh Box_1\n"
                        "  ros2 run ur3_pick_task action_sequencer --ros-args "
                        "-p config_file:=... -p object_name:=Box_1"
                    )
                prompt = f"Target Object ({valid}): " if valid else "Target Object: "
                try:
                    object_name = input(prompt).strip()
                except EOFError as exc:
                    raise RuntimeError(
                        "Could not read object name (stdin closed). "
                        "Pass -p object_name:=Box_1 or use run_action_sequencer.sh."
                    ) from exc
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
        YOLO lifecycle: yolo_detection_enable_before, yolo_detection_disable_before.
        """
        self._yolo_detection_lifecycle_before_action(action)
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
        self._yolo_detection_lifecycle_after_action(action, success=success)
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
        if self._should_skip_mujoco_action(action):
            self.get_logger().info(
                f"Skipping MuJoCo-only action '{action.get('name', action_type)}' "
                f"because robot_backend={self.robot_backend}."
            )
            return True
        if action_type == "joint":
            return self._execute_joint_action(action)
        if action_type == "cartesian":
            return self._execute_cartesian_action(action)
        if action_type in ("tool_delta", "cartesian_delta", "lift"):
            return self._execute_tool_delta_action(action)
        if action_type == "waypoint":
            return self._execute_waypoint_action(action)
        if action_type == "weld":
            return self._execute_weld_action(action)
        if action_type == "planning_scene":
            return self._execute_planning_scene_action(action)
        if action_type == "trigger":
            return self._execute_trigger_action(action)
        if action_type in ("foundation_pose", "foundation_pose_wait"):
            cfg_fp = self._foundation_pose_cfg_for_action(action)
            try:
                return self._execute_foundation_pose_action(action)
            finally:
                self._foundation_pose_maybe_stop_stack_after_action(cfg_fp, action)
        if action_type in ("foundation_pose_grasp", "foundation_pose_grasp_move"):
            cfg_fp = self._foundation_pose_cfg_for_action(action)
            try:
                return self._execute_foundation_pose_grasp_action(action)
            finally:
                self._foundation_pose_maybe_stop_stack_after_action(cfg_fp, action)
        if action_type in ("yolo_visual_center", "visual_center_yolo", "yolo_visual_lookat_and_approach"):
            return self._execute_yolo_visual_center_action(action)
        if action_type in ("gripper_hybrid_close", "gripper_close_hybrid"):
            return self._execute_gripper_hybrid_close_action(action)
        if action_type in ("dynamixel_move", "dynamixel_position", "dynamixel_goal"):
            if self.robot_backend != "real":
                self.get_logger().info(
                    f"Skipping Dynamixel action '{action.get('name', action_type)}' "
                    f"(robot_backend={self.robot_backend})."
                )
                return True
            return self._execute_dynamixel_move_action(action)
        if action_type in ("gripper_hybrid_open", "gripper_open"):
            return self._execute_gripper_hybrid_open_action(action)
        raise ValueError(f"Unsupported action type: {action_type!r}")

    def _should_skip_mujoco_action(self, action: Dict[str, Any]) -> bool:
        if self.robot_backend != "real":
            return False
        action_type = str(action.get("type", "")).lower()
        if action_type == "weld" or action.get("mujoco_weld_preset"):
            return True
        if action_type != "trigger":
            return False
        service_name = str(action.get("service_name", "")).strip()
        return service_name.startswith("/mujoco/weld/")

    def _execute_tool_delta_action(self, action: Dict[str, Any]) -> bool:
        """
        Relative tool move in base frame using current TF(tool0) as start.

        Intended for post-grasp lift motions that should preserve current orientation.

        YAML example:
          - name: Lift
            type: tool_delta
            delta_xyz: [0.0, 0.0, 0.10]   # +Z 10 cm in base frame
            cartesian_mode: linear
        """
        delta_any = action.get("delta_xyz", action.get("delta", action.get("delta_base_xyz")))
        if not isinstance(delta_any, (list, tuple)) or len(delta_any) != 3:
            raise ValueError("tool_delta requires delta_xyz: [dx, dy, dz] in base frame (meters)")
        dx, dy, dz = float(delta_any[0]), float(delta_any[1]), float(delta_any[2])

        link = str(action.get("ik_link_name") or self.motion_cfg.get("ik_link_name", "tool0")).strip() or "tool0"
        pose_now = self._lookup_link_pose(link)
        if pose_now is None:
            return False
        pose_goal = Pose()
        pose_goal.orientation = pose_now.orientation
        pose_goal.position.x = float(pose_now.position.x + dx)
        pose_goal.position.y = float(pose_now.position.y + dy)
        pose_goal.position.z = float(pose_now.position.z + dz)

        mode = str(action.get("cartesian_mode", self.motion_cfg["default_cartesian_mode"])).lower()
        if mode in ("linear", "straight", "cartesian_path"):
            min_fraction = float(action.get("min_fraction", self.motion_cfg["default_min_fraction"]))
            cartesian_result = self._execute_cartesian_path(action, [pose_goal], min_fraction)
            if cartesian_result is not None:
                return cartesian_result
            self.get_logger().warn("tool_delta Cartesian planning incomplete; falling back to IK joint-goal.")
        elif mode not in ("moveit", "joint_goal", "planned"):
            raise ValueError(f"tool_delta: unsupported cartesian_mode: {mode!r}")

        joint_positions = self._solve_ik(pose_goal, action)
        if joint_positions is None:
            return False
        return self._execute_joint_goal(action, joint_positions)

    def _set_mujoco_tip_object_param(self, tip_object: str) -> bool:
        """Select which Module_2 tip bodies/welds to use (Box_1, Cylinder_1, Cylinder_2)."""
        tip = str(tip_object).strip()
        if not tip:
            return True
        client = self._mujoco_set_params_client("/mujoco_node/set_parameters")
        if not self._wait_for_service_abortable(client, 3.0):
            self.get_logger().error(
                "Cannot set module2_tip_object — /mujoco_node/set_parameters unavailable "
                "(is MuJoCo sim running?)."
            )
            return False
        from rcl_interfaces.msg import Parameter as RosParam
        from rcl_interfaces.msg import ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters

        req = SetParameters.Request()
        pv = ParameterValue()
        pv.type = ParameterType.PARAMETER_STRING
        pv.string_value = tip
        req.parameters = [RosParam(name="module2_tip_object", value=pv)]
        future = client.call_async(req)
        self._spin_until_future_complete_abortable(future, 5.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("set_parameters(module2_tip_object) timed out.")
            return False
        results = getattr(future.result(), "results", []) or []
        for r in results:
            if hasattr(r, "successful") and not bool(r.successful):
                self.get_logger().error(
                    f"set_parameters(module2_tip_object) rejected: {getattr(r, 'reason', '')}"
                )
                return False
        self.get_logger().info(f"MuJoCo module2_tip_object set to {tip!r}.")
        return True

    def _call_trigger_service(self, service_name: str, call_timeout: float) -> tuple[bool, str]:
        """Call std_srvs/Trigger once; return (success, message)."""
        client = self._trigger_service_client(service_name)
        future = client.call_async(Trigger.Request())
        self._spin_until_future_complete_abortable(future, call_timeout)
        if not future.done():
            return False, "service call timed out"
        try:
            result = future.result()
        except Exception as exc:  # noqa: BLE001
            return False, str(exc)
        if result is None:
            return False, "no result"
        return bool(result.success), str(getattr(result, "message", "") or "")

    def _foundation_pose_post_stack_settle(self, cfg_fp: Dict[str, Any], action: Dict[str, Any]) -> None:
        """Brief spin after stack launch so bridge/Isaac subscriptions receive sim camera frames."""
        settle = float(
            action.get("bridge_post_launch_settle_sec", cfg_fp.get("bridge_post_launch_settle_sec", 2.0))
        )
        if settle <= 0.0:
            return
        deadline = time.monotonic() + settle
        while time.monotonic() < deadline and rclpy.ok():
            self._raise_if_abort()
            self._spin_cb(min(0.1, settle))

    def _foundation_pose_bridge_trigger_with_warmup(
        self,
        action: Dict[str, Any],
        cfg_fp: Dict[str, Any],
        service_name: str,
    ) -> bool:
        """
        Retry foundation_pose_bridge/trigger until rgb+depth+camera_info arrive.

        The bridge node may register its service before the first camera callback.
        """
        wait_timeout = float(action.get("wait_timeout_sec", 8.0))
        call_timeout = float(action.get("call_timeout_sec", 25.0))
        warmup = float(
            action.get("bridge_camera_warmup_timeout_sec", cfg_fp.get("bridge_camera_warmup_timeout_sec", 12.0))
        )
        period = float(
            action.get("bridge_trigger_retry_period_sec", cfg_fp.get("bridge_trigger_retry_period_sec", 0.25))
        )
        client = self._trigger_service_client(service_name)
        if not self._wait_for_service_abortable(client, wait_timeout):
            self.get_logger().error(f"Trigger service not available: {service_name}")
            return False

        self._ensure_vision_subscriptions_for_yolo()
        deadline = time.monotonic() + max(1.0, warmup)
        last_msg = ""
        attempt = 0
        while time.monotonic() < deadline and rclpy.ok():
            self._raise_if_abort()
            attempt += 1
            with self._vis_lock:
                have_cam = (
                    self._vis_bgr is not None
                    and self._vis_depth is not None
                    and self._vis_info is not None
                )
            if not have_cam:
                rclpy.spin_once(self, timeout_sec=period)
                continue
            rclpy.spin_once(self, timeout_sec=period)
            ok, msg = self._call_trigger_service(service_name, call_timeout)
            if ok:
                if msg:
                    self.get_logger().info(f"Trigger OK ({service_name}): {msg}")
                else:
                    self.get_logger().info(f"Trigger OK ({service_name})")
                return True
            last_msg = msg
            transient = (
                "Missing rgb" in msg
                or "camera_info" in msg
                or "subscription data" in msg
            )
            if not transient:
                break
            if attempt == 1 or attempt % 8 == 0:
                self.get_logger().info(
                    f"foundation_pose: bridge warming up ({msg}) — retrying…"
                )
            rclpy.spin_once(self, timeout_sec=period)

        self.get_logger().error(
            f"Trigger unsuccessful ({service_name}): {last_msg or 'timeout waiting for camera frames'}"
        )
        return False

    def _execute_trigger_action(self, action: Dict[str, Any]) -> bool:
        """Call std_srvs/Trigger once (e.g. MuJoCo plate table reset before welding)."""
        service_name = str(action.get("service_name", "")).strip()
        if not service_name:
            raise ValueError("trigger action requires non-empty service_name")
        wait_timeout = float(action.get("wait_timeout_sec", 10.0))
        call_timeout = float(action.get("call_timeout_sec", 15.0))

        tip_object = str(
            action.get("mujoco_tip_object", action.get("tip_object", self.object_name or ""))
        ).strip()
        if tip_object and "module_2_tips" in service_name:
            if not self._set_mujoco_tip_object_param(tip_object):
                return False

        client = self._trigger_service_client(service_name)
        if not self._wait_for_service_abortable(client, wait_timeout):
            self.get_logger().error(f"Trigger service not available: {service_name}")
            return False
        ok, msg = self._call_trigger_service(service_name, call_timeout)
        if not ok:
            self.get_logger().error(f"Trigger unsuccessful ({service_name}): {msg}")
            return False
        if msg:
            self.get_logger().info(f"Trigger OK ({service_name}): {msg}")
        else:
            self.get_logger().info(f"Trigger OK ({service_name})")
        return True

    def _mujoco_set_params_client(self, service_name: str) -> Any:
        if not hasattr(self, "_mujoco_set_params_clients"):
            from rcl_interfaces.srv import SetParameters

            self._mujoco_set_params_clients: dict[str, Any] = {}
        if service_name not in self._mujoco_set_params_clients:
            from rcl_interfaces.srv import SetParameters

            self._mujoco_set_params_clients[service_name] = self.create_client(
                SetParameters, service_name
            )
        return self._mujoco_set_params_clients[service_name]

    def _push_mujoco_gripper_hybrid_params(self, action: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
        from rcl_interfaces.msg import Parameter as RosParam
        from rcl_interfaces.msg import ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters

        svc = str(cfg.get("set_parameters_service", "/mujoco_node/set_parameters")).strip()
        if not svc:
            self.get_logger().error("gripper: set_parameters_service is empty")
            return False

        def _fval(key: str, default_key: str) -> float:
            if key in action:
                return float(action[key])
            return float(cfg.get(default_key, 0.0))

        def _ival(key: str, default_key: str) -> int:
            if key in action:
                return int(action[key])
            return int(cfg.get(default_key, 0))

        fields = [
            ("gripper_hybrid_close_position_rad", _fval("close_position_rad", "close_position_rad"), ParameterType.PARAMETER_DOUBLE),
            ("gripper_hybrid_hold_torque_nm", _fval("hold_torque_nm", "hold_torque_nm"), ParameterType.PARAMETER_DOUBLE),
            ("gripper_hybrid_approach_ramp_rad_s", _fval("approach_ramp_rad_s", "approach_ramp_rad_s"), ParameterType.PARAMETER_DOUBLE),
            ("gripper_hybrid_min_feedback_torque_nm", _fval("min_feedback_torque_nm", "min_feedback_torque_nm"), ParameterType.PARAMETER_DOUBLE),
            ("gripper_hybrid_max_joint_speed_rad_s", _fval("max_joint_speed_rad_s", "max_joint_speed_rad_s"), ParameterType.PARAMETER_DOUBLE),
            ("gripper_hybrid_contact_confirm_steps", _ival("contact_confirm_steps", "contact_confirm_steps"), ParameterType.PARAMETER_INTEGER),
            ("gripper_hybrid_hold_sec", _fval("hold_sec", "hold_sec"), ParameterType.PARAMETER_DOUBLE),
            ("gripper_hybrid_timeout_sec", _fval("timeout_sec", "timeout_sec"), ParameterType.PARAMETER_DOUBLE),
            ("gripper_hybrid_position_kp", _fval("position_kp", "position_kp"), ParameterType.PARAMETER_DOUBLE),
            ("gripper_hybrid_position_kv", _fval("position_kv", "position_kv"), ParameterType.PARAMETER_DOUBLE),
        ]

        req = SetParameters.Request()
        req.parameters = []
        for name, value, ptype in fields:
            pv = ParameterValue()
            pv.type = ptype
            if ptype == ParameterType.PARAMETER_INTEGER:
                pv.integer_value = int(value)
            else:
                pv.double_value = float(value)
            req.parameters.append(RosParam(name=name, value=pv))

        cli = self._mujoco_set_params_client(svc)
        wait_timeout = float(action.get("wait_timeout_sec", cfg.get("service_wait_timeout_sec", 10.0)))
        if not self._wait_for_service_abortable(cli, wait_timeout):
            self.get_logger().error(f"gripper: SetParameters service not ready: {svc}")
            return False
        future = cli.call_async(req)
        self._spin_until_future_complete_abortable(future, wait_timeout)
        if not future.done() or future.result() is None:
            self.get_logger().error(f"gripper: SetParameters timed out ({svc})")
            return False
        for r in getattr(future.result(), "results", []) or []:
            if hasattr(r, "successful") and not bool(r.successful):
                self.get_logger().error(
                    f"gripper: SetParameters rejected ({svc}): {getattr(r, 'reason', '')}"
                )
                return False
        self.get_logger().info(
            f"gripper: MuJoCo hybrid params → close={fields[0][1]:.3f} rad hold={fields[1][1]:.3f} N·m"
        )
        return True

    def _dynamixel_cfg_for_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        cfg = dict(self.gripper_cfg)
        if isinstance(action.get("gripper"), dict):
            cfg.update(dict(action["gripper"]))
        for key, value in action.items():
            if str(key).startswith("dynamixel_"):
                cfg[key] = value
        return cfg

    def _dynamixel_goal_position_from_action(
        self, action: Dict[str, Any], cfg: Dict[str, Any], *, for_pre_close: bool = False
    ) -> int:
        if for_pre_close:
            if "dynamixel_pre_close_position" in action:
                return int(action["dynamixel_pre_close_position"])
            return int(cfg.get("dynamixel_pre_close_position", 0))
        for key in ("goal_position", "dynamixel_goal_position", "position"):
            if key in action:
                return int(action[key])
        return int(cfg.get("dynamixel_pre_close_position", 0))

    def _dynamixel_open_position_from_action(
        self, action: Dict[str, Any], cfg: Dict[str, Any]
    ) -> int:
        for key in ("goal_position", "dynamixel_open_position", "dynamixel_goal_position", "position"):
            if key in action:
                return int(action[key])
        return int(cfg.get("dynamixel_open_position", 0))

    def _dynamixel_move_profiles_from_action(
        self,
        action: Dict[str, Any],
        cfg: Dict[str, Any],
        *,
        for_open: bool = False,
        for_close: bool = False,
    ) -> tuple[int, int]:
        if for_open:
            vel = action.get(
                "dynamixel_open_profile_velocity",
                cfg.get("dynamixel_open_profile_velocity", 45),
            )
            acc = action.get(
                "dynamixel_open_profile_acceleration",
                cfg.get("dynamixel_open_profile_acceleration", 12),
            )
        elif for_close:
            vel = action.get(
                "dynamixel_close_profile_velocity",
                action.get(
                    "dynamixel_profile_velocity",
                    cfg.get("dynamixel_close_profile_velocity", cfg.get("dynamixel_profile_velocity", 30)),
                ),
            )
            acc = action.get(
                "dynamixel_close_profile_acceleration",
                action.get(
                    "dynamixel_profile_acceleration",
                    cfg.get(
                        "dynamixel_close_profile_acceleration",
                        cfg.get("dynamixel_profile_acceleration", 10),
                    ),
                ),
            )
        else:
            vel = action.get("dynamixel_profile_velocity", cfg.get("dynamixel_profile_velocity", 30))
            acc = action.get(
                "dynamixel_profile_acceleration", cfg.get("dynamixel_profile_acceleration", 10)
            )
        return int(vel), int(acc)

    def _load_dynamixel_grasp_module(self, script_path: Path):
        import importlib.util
        import sys

        mod_name = "ur3_dynamixel_adaptive_grasp"
        spec = importlib.util.spec_from_file_location(mod_name, script_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    def _execute_dynamixel_move_action(
        self,
        action: Dict[str, Any],
        *,
        for_open: bool = False,
        for_close: bool = False,
    ) -> bool:
        """Real robot: move XM540 to absolute encoder goal (position control)."""
        cfg = self._dynamixel_cfg_for_action(action)
        script_path = Path(str(cfg.get("dynamixel_grasp_script", "")).strip()).expanduser()
        if not script_path.is_file():
            self.get_logger().error(f"dynamixel_move: script not found: {script_path}")
            return False

        module = self._load_dynamixel_grasp_module(script_path)
        if module is None:
            self.get_logger().error(f"dynamixel_move: cannot load {script_path}")
            return False

        grasp_cfg_cls = getattr(module, "GraspConfig", None)
        run_move_fn = getattr(module, "run_move_to_position", None)
        if grasp_cfg_cls is None or run_move_fn is None:
            self.get_logger().error(
                f"dynamixel_move: {script_path} missing GraspConfig / run_move_to_position"
            )
            return False

        grasp_cfg = grasp_cfg_cls.from_mapping(cfg)
        device = Path(grasp_cfg.device)
        if not device.exists():
            self.get_logger().error(
                f"dynamixel_move: serial device missing: {grasp_cfg.device} (plug in U2D2)"
            )
            return False

        if for_open:
            goal = self._dynamixel_open_position_from_action(action, cfg)
        elif for_close:
            goal = self._dynamixel_goal_position_from_action(action, cfg, for_pre_close=True)
        else:
            goal = self._dynamixel_goal_position_from_action(action, cfg)
        prof_vel, prof_acc = self._dynamixel_move_profiles_from_action(
            action, cfg, for_open=for_open, for_close=for_close
        )
        leave_torque_on = action.get("dynamixel_leave_torque_on")
        if leave_torque_on is None:
            leave_torque_on = not bool(
                action.get("dynamixel_torque_off_after_move", grasp_cfg.torque_off_after_move)
            )

        label = "dynamixel_open" if for_open else "dynamixel_move"
        self.get_logger().info(
            f"{label}: goal_position={goal} profile_vel={prof_vel} profile_acc={prof_acc} "
            f"id={grasp_cfg.dxl_id} device={grasp_cfg.device}"
        )

        def _log(msg: str) -> None:
            self.get_logger().info(str(msg))

        try:
            run_move_fn(
                grasp_cfg,
                goal,
                profile_velocity=prof_vel,
                profile_acceleration=prof_acc,
                log_print=_log,
                leave_torque_on=bool(leave_torque_on),
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"{label} failed: {exc}")
            return False

        self.get_logger().info(f"{label}: OK (goal_position={goal})")
        return True

    def _execute_dynamixel_gripper_open_action(self, action: Dict[str, Any]) -> bool:
        """Real robot: open gripper — position move to open_position with open profile."""
        open_action = dict(action)
        open_action.setdefault("dynamixel_leave_torque_on", True)
        return self._execute_dynamixel_move_action(open_action, for_open=True)

    def _execute_dynamixel_gripper_close_action(self, action: Dict[str, Any]) -> bool:
        """Real robot: XM540 adaptive grasp via ~/ur3_control/gripper/dynamixel_adaptive_grasp.py."""
        cfg = self._dynamixel_cfg_for_action(action)
        script_path = Path(str(cfg.get("dynamixel_grasp_script", "")).strip()).expanduser()
        if not script_path.is_file():
            self.get_logger().error(
                f"gripper_hybrid_close (real): dynamixel script not found: {script_path}"
            )
            return False

        module = self._load_dynamixel_grasp_module(script_path)
        if module is None:
            self.get_logger().error(f"gripper_hybrid_close (real): cannot load {script_path}")
            return False

        grasp_cfg_cls = getattr(module, "GraspConfig", None)
        run_fn = getattr(module, "run_adaptive_grasp", None)
        if grasp_cfg_cls is None or run_fn is None:
            self.get_logger().error(
                f"gripper_hybrid_close (real): {script_path} missing GraspConfig / run_adaptive_grasp"
            )
            return False

        grasp_cfg = grasp_cfg_cls.from_mapping(cfg)
        device = Path(grasp_cfg.device)
        if not device.exists():
            self.get_logger().error(
                f"gripper_hybrid_close (real): serial device missing: {grasp_cfg.device} "
                "(plug in U2D2, run ls-serial)"
            )
            return False

        self.get_logger().info(
            "gripper_hybrid_close (real): Dynamixel adaptive grasp "
            f"id={grasp_cfg.dxl_id} device={grasp_cfg.device} "
            f"script={script_path}"
        )

        def _log(msg: str) -> None:
            self.get_logger().info(str(msg))

        try:
            run_fn(grasp_cfg, log_print=_log)
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"gripper_hybrid_close (real) failed: {exc}")
            return False

        self.get_logger().info("gripper_hybrid_close (real): OK")
        return True

    def _execute_gripper_hybrid_close_action(self, action: Dict[str, Any]) -> bool:
        """Sim: MuJoCo hybrid close. Real: Dynamixel pre-open move then adaptive grasp."""
        if self.robot_backend == "real":
            cfg = self._dynamixel_cfg_for_action(action)
            skip_pre = bool(
                action.get(
                    "dynamixel_skip_pre_close_move",
                    cfg.get("dynamixel_skip_pre_close_move", False),
                )
            )
            if not skip_pre:
                pre_goal = self._dynamixel_goal_position_from_action(
                    action, cfg, for_pre_close=True
                )
                pre_action = dict(action)
                pre_action["goal_position"] = pre_goal
                pre_action["dynamixel_leave_torque_on"] = True
                self.get_logger().info(
                    f"gripper_hybrid_close (real): moving to pre-close position {pre_goal} "
                    f"(close profile)…"
                )
                if not self._execute_dynamixel_move_action(pre_action, for_close=True):
                    return False
            return self._execute_dynamixel_gripper_close_action(action)

        cfg = dict(self.gripper_cfg)
        if isinstance(action.get("gripper"), dict):
            cfg.update(dict(action["gripper"]))

        service_name = str(action.get("service_name", cfg.get("hybrid_close_service", ""))).strip()
        if not service_name:
            raise ValueError("gripper_hybrid_close requires hybrid_close_service in gripper config")

        if not self._push_mujoco_gripper_hybrid_params(action, cfg):
            return False

        wait_timeout = float(action.get("wait_timeout_sec", cfg.get("service_wait_timeout_sec", 10.0)))
        call_timeout = float(action.get("call_timeout_sec", cfg.get("service_call_timeout_sec", 60.0)))
        client = self._trigger_service_client(service_name)
        if not self._wait_for_service_abortable(client, wait_timeout):
            self.get_logger().error(f"gripper_hybrid_close: service not available: {service_name}")
            return False

        self.get_logger().info(
            "gripper_hybrid_close: closing via position→torque (contact from actuator feedback only)…"
        )
        ok, msg = self._call_trigger_service(service_name, call_timeout)
        if not ok:
            self.get_logger().error(f"gripper_hybrid_close failed ({service_name}): {msg}")
            return False
        self.get_logger().info(f"gripper_hybrid_close OK: {msg or service_name}")
        return True

    def _execute_gripper_hybrid_open_action(self, action: Dict[str, Any]) -> bool:
        """Sim: MuJoCo hybrid open. Real: Dynamixel position move to open_position."""
        if self.robot_backend == "real":
            return self._execute_dynamixel_gripper_open_action(action)

        cfg = dict(self.gripper_cfg)
        service_name = str(
            action.get("service_name", cfg.get("hybrid_open_service", "/mujoco/gripper/hybrid_open"))
        ).strip()
        if not service_name:
            raise ValueError("gripper_hybrid_open requires hybrid_open_service in gripper config")
        wait_timeout = float(action.get("wait_timeout_sec", cfg.get("service_wait_timeout_sec", 10.0)))
        call_timeout = float(action.get("call_timeout_sec", 15.0))
        client = self._trigger_service_client(service_name)
        if not self._wait_for_service_abortable(client, wait_timeout):
            self.get_logger().error(f"gripper_hybrid_open: service not available: {service_name}")
            return False
        ok, msg = self._call_trigger_service(service_name, call_timeout)
        if not ok:
            self.get_logger().error(f"gripper_hybrid_open failed ({service_name}): {msg}")
            return False
        self.get_logger().info(f"gripper_hybrid_open OK: {msg or service_name}")
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
        """Publish optional weld offset pose, then call MuJoCo attach Trigger service."""
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
        publish_pose = bool(action.get("publish_target_pose", True))
        wait_timeout = float(action.get("wait_timeout_sec", 10.0))
        call_timeout = float(action.get("call_timeout_sec", 15.0))

        self.get_logger().info(
            "Weld step: switches MuJoCo constraints only — the arm receives no trajectory here, "
            "so visually it stays still until the next MoveIt motion."
        )

        if publish_pose:
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
                f"Published weld target on {topic} (frame={frame_id!r}); "
                f"waiting {settle_sec:.3f}s before service call."
            )
            if settle_sec > 0.0:
                self._sleep_interruptible(settle_sec)
        else:
            self.get_logger().info(
                f"Weld step: publish_target_pose=false — calling {service_name} without pose publish."
            )

        client = self._trigger_service_client(service_name)
        if not self._wait_for_service_abortable(client, wait_timeout):
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
        self._spin_until_future_complete_abortable(future, call_timeout)
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

        world_remove_list: list[CollisionObject] = []
        for remove in remove_list:
            # Detaching can materialize a world object at the current tool pose. MuJoCo owns
            # the real body in this task, so remove any MoveIt proxy in a second diff.
            co = CollisionObject()
            co.id = remove.object.id
            co.operation = CollisionObject.REMOVE
            world_remove_list.append(co)

        ps = PlanningScene()
        ps.world.collision_objects = world_remove_list
        self._publish_planning_scene_diff(ps, settle_sec=settle_sec)
        self.get_logger().info(
            f"Planning scene: DETACH/REMOVE from {link_name!r}: {', '.join(str(o.object.id) for o in remove_list)}"
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
            if action.get("cartesian_linear_fallback") is False:
                self.get_logger().error(
                    "Cartesian linear path failed and cartesian_linear_fallback is disabled."
                )
                return False
            self.get_logger().warn(
                "Cartesian path planning was incomplete; falling back to IK joint-goal planning."
            )
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
        goal.planning_options.plan_only = True
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
            self.get_logger().error("MoveGroupSequence planning timed out.")
            return False

        result = result_future.result().result
        code = result.response.error_code.val
        if code != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"MoveGroupSequence failed: error_code={code}")
            return False

        planned = list(result.response.planned_trajectories)
        if not planned:
            self.get_logger().error("MoveGroupSequence plan returned no trajectories.")
            return False

        exec_timeout = float(action.get("execution_timeout_sec", timeout))
        for index, trajectory in enumerate(planned):
            if not self._execute_trajectory(trajectory, exec_timeout):
                self.get_logger().error(f"MoveGroupSequence segment {index + 1}/{len(planned)} failed.")
                return False
        return True

    def _execute_waypoints_sequentially(self, action: Dict[str, Any], waypoints: list[Dict[str, Any]]) -> bool:
        for index, waypoint in enumerate(waypoints):
            self._raise_if_abort()
            waypoint_action = self._waypoint_action(action, waypoint, index)
            waypoint_type = self._waypoint_type(waypoint)
            if waypoint_type == "joint":
                if not self._execute_joint_action(waypoint_action):
                    return False
            elif waypoint_type == "cartesian":
                if not self._execute_cartesian_action(waypoint_action):
                    return False
            else:
                raise ValueError(f"Unsupported waypoint type: {waypoint_type!r}")
        return True

    def _execute_cartesian_moveit_fallback(self, action: Dict[str, Any], pose: Pose) -> bool:
        if self._execute_pose_goal(action, pose):
            return True
        self.get_logger().warn("Pose goal planning failed; falling back to IK joint-goal planning.")
        joint_positions = self._solve_ik(pose, action)
        if joint_positions is None:
            return False
        return self._execute_joint_goal(action, joint_positions)

    def _build_pose_goal_constraints(self, action: Dict[str, Any], tool_pose: Pose) -> Constraints:
        link_name = str(action.get("ik_link_name", self.motion_cfg["ik_link_name"])).strip() or "tool0"
        base_frame = str(self.motion_cfg["base_frame"])
        pos_tol = float(action.get("position_tolerance_m", 0.005))
        ori_tol = float(action.get("orientation_tolerance_rad", 0.05))

        sphere = SolidPrimitive()
        sphere.type = SolidPrimitive.SPHERE
        sphere.dimensions = [pos_tol]

        region_pose = Pose()
        region_pose.position = tool_pose.position
        region_pose.orientation.w = 1.0

        vol = BoundingVolume()
        vol.primitives = [sphere]
        vol.primitive_poses = [region_pose]

        pos_c = PositionConstraint()
        pos_c.header = Header(frame_id=base_frame)
        pos_c.link_name = link_name
        pos_c.target_point_offset = Vector3(x=0.0, y=0.0, z=0.0)
        pos_c.constraint_region = vol
        pos_c.weight = 1.0

        ori_c = OrientationConstraint()
        ori_c.header = Header(frame_id=base_frame)
        ori_c.link_name = link_name
        ori_c.orientation = tool_pose.orientation
        ori_c.absolute_x_axis_tolerance = ori_tol
        ori_c.absolute_y_axis_tolerance = ori_tol
        ori_c.absolute_z_axis_tolerance = ori_tol
        ori_c.weight = 1.0
        ori_c.parameterization = OrientationConstraint.XYZ_EULER_ANGLES

        constraints = Constraints()
        constraints.name = str(action.get("name", "pose_goal"))
        constraints.position_constraints = [pos_c]
        constraints.orientation_constraints = [ori_c]
        return constraints

    def _build_motion_request_from_constraints(
        self, action: Dict[str, Any], constraints: Constraints
    ) -> MotionPlanRequest:
        req = MotionPlanRequest()
        req.workspace_parameters = self._workspace()
        req.group_name = str(self.motion_cfg["planning_group"])
        req.start_state.joint_state = self._moveit_start_joint_state()
        req.start_state.is_diff = False
        req.goal_constraints = [constraints]
        req.num_planning_attempts = int(action.get("planning_attempts", self.motion_cfg["default_planning_attempts"]))
        req.allowed_planning_time = self._planning_time(action)
        req.max_velocity_scaling_factor = self._velocity(action)
        req.max_acceleration_scaling_factor = self._acceleration(action)
        return req

    def _execute_pose_goal(self, action: Dict[str, Any], tool_pose: Pose) -> bool:
        if not self._wait_for_joint_state(action):
            return False
        self._wait_for_joint_settle(action)
        if not self._wait_for_action_server_abortable(self._move_group_client, 10.0):
            self.get_logger().error(f"MoveGroup action is not available: {self.motion_cfg['move_group_action']}")
            return False

        goal = MoveGroup.Goal()
        goal.request = self._build_motion_request_from_constraints(
            action, self._build_pose_goal_constraints(action, tool_pose)
        )
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True
        goal.planning_options.look_around = False
        goal.planning_options.replan = True
        goal.planning_options.replan_attempts = int(action.get("replan_attempts", 1))

        self.get_logger().info("Sending pose goal to MoveIt.")
        send_future = self._move_group_client.send_goal_async(goal)
        self._spin_until_future_complete_abortable(send_future, 10.0)
        goal_handle = send_future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("MoveIt pose goal rejected.")
            return False

        timeout = float(action.get("execution_timeout_sec", self._planning_time(action) + 30.0))
        result_future = goal_handle.get_result_async()
        self._spin_until_future_complete_abortable(result_future, timeout, goal_handle)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("MoveIt pose goal planning timed out.")
            return False

        result = result_future.result().result
        return self._execute_planned_move_group_result(action, result, log_label="MoveIt pose goal")

    def _execute_joint_goal(self, action: Dict[str, Any], positions: list[float]) -> bool:
        if not self._wait_for_joint_state(action):
            return False
        self._wait_for_joint_settle(action)
        if not self._wait_for_action_server_abortable(self._move_group_client, 10.0):
            self.get_logger().error(f"MoveGroup action is not available: {self.motion_cfg['move_group_action']}")
            return False

        goal = MoveGroup.Goal()
        goal.request = self._build_joint_motion_request(action, positions)
        goal.planning_options = PlanningOptions()
        goal.planning_options.plan_only = True
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

        timeout = float(action.get("execution_timeout_sec", self._planning_time(action) + 30.0))
        result_future = goal_handle.get_result_async()
        self._spin_until_future_complete_abortable(result_future, timeout, goal_handle)
        if not result_future.done() or result_future.result() is None:
            self.get_logger().error("MoveIt joint planning timed out.")
            return False

        result = result_future.result().result
        return self._execute_planned_move_group_result(action, result, log_label="MoveIt joint goal")

    def _execute_cartesian_path(self, action: Dict[str, Any], waypoints: list[Pose], min_fraction: float) -> Optional[bool]:
        if not self._wait_for_joint_state(action):
            return False
        self._wait_for_joint_settle(action)
        if not self._wait_for_service_abortable(self._cartesian_client, 5.0):
            self.get_logger().error(
                f"Cartesian path service is not available: {self.motion_cfg['cartesian_path_service']}"
            )
            return None

        req = GetCartesianPath.Request()
        req.header.frame_id = str(self.motion_cfg["base_frame"])
        req.start_state.joint_state = self._moveit_start_joint_state()
        req.start_state.is_diff = False
        req.group_name = str(self.motion_cfg["planning_group"])
        req.link_name = str(action.get("ik_link_name") or self.motion_cfg["ik_link_name"]).strip()
        req.waypoints = waypoints
        req.max_step = float(action.get("max_step_m", self.motion_cfg["default_max_step_m"]))
        req.jump_threshold = float(action.get("jump_threshold", self.motion_cfg["default_jump_threshold"]))
        req.avoid_collisions = self._moveit_avoid_collisions(action)

        self.get_logger().info(
            f"Computing Cartesian path through {len(waypoints)} waypoint(s) "
            f"(avoid_collisions={req.avoid_collisions})."
        )
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
        if "cartesian_retime_min_segment_sec" in action:
            min_seg = float(action["cartesian_retime_min_segment_sec"])
        elif self.robot_backend == "real":
            min_seg = float(self.motion_cfg.get("real_cartesian_retime_min_segment_sec", 0.08))
        else:
            min_seg = 0.04

        if "cartesian_retime_duration_stretch" in action:
            dur_stretch = float(action["cartesian_retime_duration_stretch"])
        elif self.robot_backend == "real":
            dur_stretch = float(self.motion_cfg.get("real_cartesian_retime_duration_stretch", 2.0))
        else:
            dur_stretch = 1.0

        self._retime_trajectory(
            trajectory,
            self._velocity(action),
            min_segment_sec=min_seg,
            duration_stretch=dur_stretch,
        )
        return self._execute_trajectory(trajectory, self._execution_timeout(action, trajectory))

    def _execute_trajectory(self, trajectory: RobotTrajectory, timeout: float) -> bool:
        self._unwrap_continuous_joints_in_robot_trajectory(trajectory)
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
        self._unwrap_continuous_joints_in_joint_trajectory(trajectory)
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
        req.ik_request.ik_link_name = str(action.get("ik_link_name", self.motion_cfg["ik_link_name"])).strip()
        req.ik_request.pose_stamped.header.frame_id = str(self.motion_cfg["base_frame"])
        req.ik_request.pose_stamped.pose = tool_pose
        req.ik_request.robot_state.joint_state = self._moveit_start_joint_state()
        req.ik_request.robot_state.is_diff = False
        req.ik_request.avoid_collisions = self._moveit_avoid_collisions(action)
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
        # Align continuous-joint YAML/IK targets to live /joint_states wrap before MoveIt plans.
        positions = self._align_continuous_joint_targets(list(positions))
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
        req.start_state.joint_state = self._moveit_start_joint_state()
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

    def _wait_for_joint_settle(self, action: Dict[str, Any]) -> None:
        if not bool(action.get("wait_for_joint_settle", self.motion_cfg["wait_for_joint_settle"])):
            return
        timeout_sec = float(action.get("joint_settle_timeout_sec", self.motion_cfg["joint_settle_timeout_sec"]))
        max_velocity = float(action.get("joint_settle_max_velocity_rad_s", self.motion_cfg["joint_settle_max_velocity_rad_s"]))
        required_samples = max(
            1,
            int(action.get("joint_settle_required_samples", self.motion_cfg["joint_settle_required_samples"])),
        )
        deadline = time.monotonic() + max(0.0, timeout_sec)
        settled_samples = 0
        last_max_velocity = 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            self._raise_if_abort()
            js = self._latest_joint_state
            if js is not None and js.name and js.velocity and len(js.velocity) == len(js.name):
                vel_by_name = dict(zip(js.name, js.velocity))
                try:
                    last_max_velocity = max(abs(float(vel_by_name[name])) for name in self.joint_names)
                except KeyError:
                    last_max_velocity = float("inf")
                if last_max_velocity <= max_velocity:
                    settled_samples += 1
                    if settled_samples >= required_samples:
                        return
                else:
                    settled_samples = 0
            rclpy.spin_once(self, timeout_sec=0.02)
        if timeout_sec > 0.0:
            self.get_logger().warn(
                "Joint settle wait timed out before motion "
                f"(max |qd|={last_max_velocity:.4f} rad/s > {max_velocity:.4f}); continuing."
            )

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

    def _retime_trajectory(
        self,
        trajectory: RobotTrajectory,
        velocity_scaling: float,
        *,
        min_segment_sec: float = 0.04,
        duration_stretch: float = 1.0,
        max_joint_vel_rad_s: float = 0.5,
    ) -> None:
        points = trajectory.joint_trajectory.points
        if not points:
            return

        velocity_scaling = max(float(velocity_scaling), 0.01)
        min_seg = max(0.01, float(min_segment_sec))
        stretch = max(0.5, float(duration_stretch))
        max_joint_vel = max(0.05, float(max_joint_vel_rad_s))
        positions = [list(point.positions) for point in points]
        if any(not pos for pos in positions):
            return

        times = [0.0]
        previous = positions[0]
        for current in positions[1:]:
            deltas = [abs(float(a) - float(b)) for a, b in zip(current, previous)]
            # Keep Cartesian micro-waypoints moving continuously instead of stopping at every sample.
            dt = max(max(deltas, default=0.0) / velocity_scaling, min_seg) * stretch
            times.append(times[-1] + dt)
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
                if index >= len(points) - 1:
                    velocities.append(0.0)
                    continue
                dt = max(times[index + 1] - times[index], 1e-6)
                dq = float(positions[index + 1][joint_index]) - float(positions[index][joint_index])
                vel = dq / dt
                velocities.append(float(max(-max_joint_vel, min(max_joint_vel, vel))))
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

    @staticmethod
    def _normalize_speed_scale(value: Any) -> float:
        scale = float(value)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError(f"speed_scale must be a finite number > 0, got {value!r}")
        return min(scale, 5.0)

    def _scale_motion_factor(self, value: Any, label: str, default: float) -> float:
        base = _scaling(value, label, default)
        scaled = min(1.0, base * self.speed_scale)
        return max(scaled, 1e-4)

    @staticmethod
    def _uses_sequence_speed_scale(action: Dict[str, Any]) -> bool:
        action_type = str(action.get("type", "")).lower()
        return action_type in {"joint", "cartesian", "tool_delta", "cartesian_delta", "lift", "waypoint"}

    def _velocity(self, action: Dict[str, Any]) -> float:
        if not self._uses_sequence_speed_scale(action):
            return _scaling(
                action.get("velocity_scaling"),
                "velocity_scaling",
                self.motion_cfg["default_velocity_scaling"],
            )
        return self._scale_motion_factor(
            action.get("velocity_scaling"),
            "velocity_scaling",
            self.motion_cfg["default_velocity_scaling"],
        )

    def _acceleration(self, action: Dict[str, Any]) -> float:
        if not self._uses_sequence_speed_scale(action):
            return _scaling(
                action.get("acceleration_scaling"),
                "acceleration_scaling",
                self.motion_cfg["default_acceleration_scaling"],
            )
        return self._scale_motion_factor(
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
        out.setdefault("wait_for_joint_settle", True)
        out.setdefault("joint_settle_timeout_sec", 1.0)
        out.setdefault("joint_settle_max_velocity_rad_s", 0.02)
        out.setdefault("joint_settle_required_samples", 3)
        out.setdefault("default_joint_tolerance_rad", 0.01)
        out.setdefault("speed_scale", 1.0)
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
    node: Optional[MotionManager] = None
    stdin_shutdown = threading.Event()
    stdin_thread: Optional[threading.Thread] = None

    try:
        node = MotionManager()
    except (RuntimeError, ValueError, EOFError, KeyboardInterrupt, Exception) as exc:
        if exc.__class__.__name__ == "ParserError" and exc.__class__.__module__.startswith("yaml"):
            print(f"action_sequencer: invalid YAML in config_file — {exc}", file=sys.stderr, flush=True)
        else:
            print(f"action_sequencer: {exc}", file=sys.stderr, flush=True)
        if rclpy.ok():
            rclpy.shutdown()
        return

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
        if node is not None:
            node.cleanup()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if exit_code != 0:
        print("Action sequence stopped or failed — see log output above.", file=sys.stderr, flush=True)
    # Non-zero exit lets run_action_sequencer.sh detect failure; that script always keeps the tab open.
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv)
