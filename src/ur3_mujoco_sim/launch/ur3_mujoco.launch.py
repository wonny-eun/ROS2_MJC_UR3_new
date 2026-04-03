#!/usr/bin/env python3

from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue, ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("ur3_mujoco_sim"), "urdf", "ur3_mujoco.urdf"]),
        ]
    )
    robot_description = {"robot_description": ParameterValue(value=robot_description_content, value_type=str)}

    controller_parameters = ParameterFile(
        PathJoinSubstitution([FindPackageShare("ur3_mujoco_sim"), "config", "controllers.yaml"]),
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": True}],
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        output="both",
        parameters=[{"use_sim_time": True}, controller_parameters],
        remappings=[("~/robot_description", "/robot_description")],
    )

    spawn_joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        name="spawn_joint_state_broadcaster",
        arguments=["joint_state_broadcaster"],
        output="both",
    )

    spawn_position_controller = Node(
        package="controller_manager",
        executable="spawner",
        name="spawn_position_controller",
        arguments=["position_controller"],
        output="both",
    )

    return LaunchDescription(
        [
            robot_state_publisher_node,
            control_node,
            spawn_joint_state_broadcaster,
            spawn_position_controller,
        ]
    )

