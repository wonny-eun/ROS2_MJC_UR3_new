#!/usr/bin/env python3

import os

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution, LaunchConfiguration
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue, ParameterFile
from launch_ros.substitutions import FindPackageShare

from ur_moveit_config.launch_common import load_yaml


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    headless = LaunchConfiguration("headless")
    mujoco_model = LaunchConfiguration("mujoco_model")
    enable_virtual_perception = LaunchConfiguration("enable_virtual_perception")
    enable_object_manager = LaunchConfiguration("enable_object_manager")
    enable_mujoco_rl_camera_preview = LaunchConfiguration("enable_mujoco_rl_camera_preview")
    camera_publish_rate = LaunchConfiguration("camera_publish_rate")
    lidar_publish_rate = LaunchConfiguration("lidar_publish_rate")
    sim_speed_factor = LaunchConfiguration("sim_speed_factor")

    default_mujoco_scene = PathJoinSubstitution(
        [FindPackageShare("ur3_mujoco_sim"), "mjcf", "ur3_scene_table.xml"]
    )

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
    robot_description_kinematics = load_yaml("ur_moveit_config", "config/kinematics.yaml")
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
            {"robot_description_kinematics": robot_description_kinematics},
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
        arguments=["-d", rviz_config],
        parameters=[robot_description, robot_description_semantic, {"use_sim_time": use_sim_time}],
    )

    virtual_perception_node = Node(
        package="ur3_rl_bridge",
        executable="virtual_perception_node",
        output="screen",
        condition=IfCondition(enable_virtual_perception),
        parameters=[
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
            {"target_pose_topic": "/target_pose"},
            {"object_frame": "target_object"},
            {"reset_pose_bounds_xyz": [0.20, 0.45, -0.20, 0.20, 0.05, 0.35]},
        ],
    )

    # OpenCV windows: /rl_camera/color (+ depth) from mujoco_node publishers.
    # Runs even when MuJoCo is headless (no GLFW sim window) as long as DISPLAY is set for OpenCV.
    rl_camera_preview_node = Node(
        package="ur3_rl_bridge",
        executable="mujoco_rl_camera_preview",
        output="screen",
        condition=IfCondition(enable_mujoco_rl_camera_preview),
        parameters=[
            {"rgb_topic": "/rl_camera/color"},
            {"depth_topic": "/rl_camera/depth"},
            {"show_depth": True},
            {"display_hz": 30.0},
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
                description="If true, open OpenCV windows for rl_camera (requires DISPLAY).",
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
            rl_camera_preview_node,
        ]
    )

