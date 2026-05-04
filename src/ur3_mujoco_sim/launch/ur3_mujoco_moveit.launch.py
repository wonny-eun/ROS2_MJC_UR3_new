#!/usr/bin/env python3

import math
import os
import random
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue, ParameterFile
from launch_ros.substitutions import FindPackageShare

from ur_moveit_config.launch_common import load_yaml


def _sample_non_overlapping_xy(
    n: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    min_dist_m: float,
    max_tries: int = 4000,
):
    points = []
    for _ in range(max_tries):
        if len(points) >= n:
            break
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        ok = True
        for px, py in points:
            if math.hypot(x - px, y - py) < min_dist_m:
                ok = False
                break
        if ok:
            points.append((x, y))
    if len(points) != n:
        raise RuntimeError(
            f"Could not sample {n} non-overlapping points in bounds "
            f"x[{x_min},{x_max}], y[{y_min},{y_max}] with min_dist={min_dist_m}."
        )
    return points


def _write_randomized_scene_copy(base_scene_path: str):
    """
    Create a temporary MJCF copy with randomized object poses.
    Drops all objects from the same spawn Z (default 0.9 m) while randomizing XY + continuous roll/pitch/yaw.
    """
    tree = ET.parse(base_scene_path)
    root = tree.getroot()

    # Tunables: update here if you want different spawn area.
    x_min, x_max = -0.45, -0.15
    y_min, y_max = -0.45, -0.15
    same_z = 0.9
    min_dist_m = 0.07

    object_body_names = [
        "cylinder_1_obj",
        "cylinder_2_obj",
        "square_1_obj",
    ]
    sampled_xy = _sample_non_overlapping_xy(
        len(object_body_names), x_min, x_max, y_min, y_max, min_dist_m
    )
    body_to_xy = {name: xy for name, xy in zip(object_body_names, sampled_xy)}
    object_pose = {}
    for body_name, (x, y) in body_to_xy.items():
        object_name = body_name.replace("_obj", "")
        roll = random.uniform(-math.pi, math.pi)
        pitch = random.uniform(-math.pi, math.pi)
        yaw = random.uniform(-math.pi, math.pi)
        object_pose[object_name] = {
            "x": x,
            "y": y,
            "z": same_z,
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
        }

    found = set()
    for body in root.iter("body"):
        name = body.get("name")
        if name in body_to_xy:
            p = object_pose[name.replace("_obj", "")]
            body.set("pos", f"{p['x']:.4f} {p['y']:.4f} {p['z']:.4f}")
            # Apply initial orientation on geoms (visual+collision). For freejoint bodies this
            # yields consistent startup orientation in MuJoCo.
            for geom in body.findall("geom"):
                geom.set("euler", f"{p['roll']:.6f} {p['pitch']:.6f} {p['yaw']:.6f}")
            found.add(name)

    missing = sorted(set(object_body_names) - found)
    if missing:
        raise RuntimeError(
            "Could not find object bodies in MJCF for startup randomization: "
            + ", ".join(missing)
        )

    # Keep randomized copy in the same directory as the base scene so relative
    # <include file="UR3_RG2.xml"/> and mesh paths continue to resolve.
    out_dir = os.path.dirname(base_scene_path)
    out_path = os.path.join(out_dir, f"ur3_scene_randomized_{os.getpid()}.xml")
    tree.write(out_path, encoding="utf-8", xml_declaration=False)
    return out_path, object_pose


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    mujoco_model = LaunchConfiguration("mujoco_model")
    enable_virtual_perception = LaunchConfiguration("enable_virtual_perception")
    enable_object_manager = LaunchConfiguration("enable_object_manager")
    enable_mujoco_rl_camera_preview = LaunchConfiguration("enable_mujoco_rl_camera_preview")
    enable_camera_noise = LaunchConfiguration("enable_camera_noise")
    enable_yolo_object_preview = LaunchConfiguration("enable_yolo_object_preview")
    enable_rviz = LaunchConfiguration("enable_rviz")
    enable_handeye_tf = LaunchConfiguration("enable_handeye_tf")
    handeye_file = LaunchConfiguration("handeye_file")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_target_class = LaunchConfiguration("yolo_target_class")
    camera_rgb_topic = "/rl_camera/noisy/color"
    camera_depth_topic = "/rl_camera/noisy/depth"
    camera_publish_rate = LaunchConfiguration("camera_publish_rate")
    lidar_publish_rate = LaunchConfiguration("lidar_publish_rate")
    sim_speed_factor = LaunchConfiguration("sim_speed_factor")

    base_scene = os.path.join(get_package_share_directory("ur3_mujoco_sim"), "mjcf", "ur3_scene_table.xml")
    default_mujoco_scene, _object_pose = _write_randomized_scene_copy(base_scene)

    # Robot description (UR3 official model + MuJoCo ros2_control)
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("ur3_mujoco_sim"), "urdf", "ur3_mujoco_fixed.urdf.xacro"]
            ),
            " ",
            "name:=ur",
            " ",
            "ur_type:=ur3",
            " ",
            "tf_prefix:=",
            " ",
            "mujoco_model:=",
            mujoco_model,
            " ",
            "headless:=",
            headless,
            " ",
            "camera_publish_rate:=",
            camera_publish_rate,
            " ",
            "lidar_publish_rate:=",
            lidar_publish_rate,
            " ",
            "sim_speed_factor:=",
            sim_speed_factor,
            " ",
        ]
    )
    robot_description = {"robot_description": ParameterValue(robot_description_content, value_type=str)}

    # Semantic description (SRDF)
    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "srdf", "ur.srdf.xacro"]),
            " ",
            "name:=ur",
            " ",
            "prefix:=",
            " ",
        ]
    )
    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(robot_description_semantic_content, value_type=str)
    }

    # MoveIt planning + kinematics
    # kinematics.yaml uses ROS 2 param-file layout (/**/ros__parameters/...); it must be loaded
    # as a ParameterFile, not load_yaml() wrapped in {"robot_description_kinematics": ...}, or
    # MoveIt never sees ur_manipulator and pose goals fail with "Unable to construct goal representation".
    robot_description_kinematics = ParameterFile(
        PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "config", "kinematics.yaml"]),
        allow_substs=True,
    )
    robot_description_planning = load_yaml("ur_moveit_config", "config/joint_limits.yaml")

    ompl_planning_yaml = load_yaml("ur_moveit_config", "config/ompl_planning.yaml")
    ompl_planning_pipeline_config = {
        "move_group": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": (
                "default_planner_request_adapters/AddTimeOptimalParameterization "
                "default_planner_request_adapters/FixWorkspaceBounds "
                "default_planner_request_adapters/FixStartStateBounds "
                "default_planner_request_adapters/FixStartStateCollision "
                "default_planner_request_adapters/FixStartStatePathConstraints"
            ),
            "start_state_max_bounds_error": 0.1,
        }
    }
    if ompl_planning_yaml:
        ompl_planning_pipeline_config["move_group"].update(ompl_planning_yaml)

    # MoveIt controller manager (make it talk to ros2_control JTC)
    # We keep the name 'joint_trajectory_controller' to match ur_moveit_config.
    controllers_yaml = load_yaml("ur_moveit_config", "config/controllers.yaml")
    if controllers_yaml:
        # Ensure the non-scaled controller is default for simulation
        if "scaled_joint_trajectory_controller" in controllers_yaml:
            controllers_yaml["scaled_joint_trajectory_controller"]["default"] = False
        if "joint_trajectory_controller" in controllers_yaml:
            controllers_yaml["joint_trajectory_controller"]["default"] = True

    moveit_controllers = {
        "moveit_simple_controller_manager": controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
        "trajectory_execution.execution_duration_monitoring": False,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {"robot_description_planning": robot_description_planning},
            ompl_planning_pipeline_config,
            moveit_controllers,
            trajectory_execution,
            planning_scene_monitor_parameters,
            {"use_sim_time": use_sim_time},
        ],
    )

    # ROS 2 control (MuJoCo hardware) + controllers
    controller_parameters = ParameterFile(
        PathJoinSubstitution([FindPackageShare("ur3_mujoco_sim"), "config", "ur3_controllers.yaml"]),
    )

    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}, robot_description, controller_parameters],
        remappings=[("~/robot_description", "/robot_description")],
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    spawn_jsb = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    spawn_jtc = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--controller-manager", "/controller_manager"],
        output="screen",
    )

    rviz_config = PathJoinSubstitution([FindPackageShare("ur_moveit_config"), "rviz", "view_robot.rviz"])
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        condition=IfCondition(enable_rviz),
        arguments=["-d", rviz_config],
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            {"robot_description_planning": robot_description_planning},
            ompl_planning_pipeline_config,
            {"use_sim_time": use_sim_time},
        ],
    )

    virtual_perception_node = Node(
        package="ur3_rl_bridge",
        executable="virtual_perception_node",
        output="screen",
        condition=IfCondition(enable_virtual_perception),
        parameters=[
            {"use_sim_time": use_sim_time},
            {"publish_rate_hz": 30.0},
            {"camera_frame": "camera_color_optical_frame"},
            {"object_frame": "target_object"},
        ],
    )

    object_manager_node = Node(
        package="ur3_rl_bridge",
        executable="mujoco_object_manager_node",
        output="screen",
        condition=IfCondition(enable_object_manager),
        parameters=[
            {"use_sim_time": use_sim_time},
            {"target_pose_topic": "/target_pose"},
            {"object_frame": "target_object"},
            {"reset_pose_bounds_xyz": [0.20, 0.45, -0.20, 0.20, 0.05, 0.35]},
        ],
    )

    noisy_camera_node = Node(
        package="ur3_rl_bridge",
        executable="noisy_camera_node",
        output="screen",
        condition=IfCondition(enable_camera_noise),
        parameters=[
            {"use_sim_time": use_sim_time},
            {"rgb_in_topic": "/rl_camera/color"},
            {"depth_in_topic": "/rl_camera/depth"},
            {"rgb_out_topic": camera_rgb_topic},
            {"depth_out_topic": camera_depth_topic},
            {"rgb_noise_sigma": 1.0},
            {"rgb_salt_pepper_prob": 0.0001},
            {"rgb_brightness_jitter": 1.0},
            {"rgb_contrast_jitter": 0.01},
            {"depth_noise_sigma_m": 0.001},
            {"depth_quant_step_m": 0.001},
            {"depth_dropout_prob": 0.001},
            {"depth_edge_dropout_prob": 0.005},
            {"depth_edge_threshold_m": 0.02},
            {"depth_shadow_enable": True},
            {"depth_shadow_direction": "left"},
            {"depth_shadow_k_px_m": 3.0},
            {"depth_shadow_min_radius_px": 1},
            {"depth_shadow_max_radius_px": 8},
            {"depth_shadow_edge_threshold_m": 0.015},
            {"depth_shadow_max_depth_m": 4.0},
        ],
    )

    # OpenCV windows: noisy camera topics derived from mujoco_node publishers.
    # Runs even when MuJoCo is headless (no GLFW sim window) as long as DISPLAY is set for OpenCV.
    rl_camera_preview_node = Node(
        package="ur3_rl_bridge",
        executable="mujoco_rl_camera_preview",
        output="screen",
        condition=IfCondition(enable_mujoco_rl_camera_preview),
        parameters=[
            {"use_sim_time": use_sim_time},
            {"rgb_topic": camera_rgb_topic},
            {"depth_topic": camera_depth_topic},
            {"show_rgb": False},
            {"show_depth": True},
            {"display_hz": 30.0},
        ],
    )

    yolo_object_preview_node = Node(
        package="ur3_rl_bridge",
        executable="yolo_object_preview",
        output="screen",
        condition=IfCondition(enable_yolo_object_preview),
        parameters=[
            {"use_sim_time": use_sim_time},
            {"rgb_topic": camera_rgb_topic},
            {"model_path": yolo_model_path},
            {"target_class": yolo_target_class},
            {"display_hz": 10.0},
            {"conf": 0.35},
            {"iou": 0.5},
            {"show_window": True},
            {"publish_annotated": True},
            {"annotated_topic": "/yolo/annotated"},
        ],
    )

    handeye_tf_publisher_node = Node(
        package="ur3_rl_bridge",
        executable="handeye_tf_publisher",
        output="screen",
        condition=IfCondition(enable_handeye_tf),
        parameters=[
            {"use_sim_time": use_sim_time},
            {"calibration_file": handeye_file},
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument(
                "mujoco_model",
                default_value=default_mujoco_scene,
                description="Absolute path to MJCF loaded by mujoco_ros2_simulation (scene or bare robot).",
            ),
            DeclareLaunchArgument("enable_virtual_perception", default_value="true"),
            DeclareLaunchArgument("enable_object_manager", default_value="true"),
            DeclareLaunchArgument(
                "enable_mujoco_rl_camera_preview",
                default_value="true",
                description="If true, open OpenCV depth preview for rl_camera (requires DISPLAY).",
            ),
            DeclareLaunchArgument(
                "enable_camera_noise",
                default_value="true",
                description="If true, publish noisy /rl_camera/noisy/color and /rl_camera/noisy/depth topics.",
            ),
            DeclareLaunchArgument(
                "enable_yolo_object_preview",
                default_value="true",
                description="If true, run YOLO on noisy color camera and open an annotated OpenCV window.",
            ),
            DeclareLaunchArgument(
                "enable_rviz",
                default_value="true",
                description="If true, open RViz with the MoveIt robot view.",
            ),
            DeclareLaunchArgument(
                "enable_handeye_tf",
                default_value="false",
                description="If true, publish tool0 -> camera optical frame from the selected hand-eye YAML.",
            ),
            DeclareLaunchArgument(
                "handeye_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("ur3_rl_bridge"), "config", "handeye", "d435i_on_gripper.yaml"]
                ),
                description="YAML file containing the hand-eye transform from tool0 to the camera optical frame.",
            ),
            DeclareLaunchArgument(
                "yolo_model_path",
                default_value="/home/wonny/ur3_control/runs/segment/ur3_yolo_multi_seg3/weights/best.pt",
                description="Path to trained YOLO segmentation model (best.pt).",
            ),
            DeclareLaunchArgument(
                "yolo_target_class",
                default_value="",
                description="Optional class filter, e.g. Cylinder_2. Empty means show all classes.",
            ),
            DeclareLaunchArgument(
                "camera_publish_rate",
                default_value="30.0",
                description="MuJoCo camera publish rate (Hz) for /rl_camera/*. Lower if CPU-limited.",
            ),
            DeclareLaunchArgument(
                "lidar_publish_rate",
                default_value="5.0",
                description="MuJoCo lidar thread rate (Hz).",
            ),
            DeclareLaunchArgument(
                "sim_speed_factor",
                default_value="1.0",
                description="MuJoCo wall-clock scale (1.0 = real-time factor 1).",
            ),
            robot_state_publisher_node,
            ros2_control_node,
            spawn_jsb,
            spawn_jtc,
            move_group_node,
            rviz_node,
            virtual_perception_node,
            object_manager_node,
            handeye_tf_publisher_node,
            noisy_camera_node,
            rl_camera_preview_node,
            yolo_object_preview_node,
        ]
    )

