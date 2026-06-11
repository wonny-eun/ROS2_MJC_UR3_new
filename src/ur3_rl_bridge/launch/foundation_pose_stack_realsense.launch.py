#!/usr/bin/env python3
"""FoundationPose stack for real UR3 + Intel RealSense D435i (bridge + Isaac FP + fp_object TF)."""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import SetParameter
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    use_sim = LaunchConfiguration("use_sim_time")
    launch_isaac = LaunchConfiguration("launch_isaac")

    bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ur3_rl_bridge"), "/launch/foundation_pose_bridge_realsense.launch.py"]
        ),
    )

    isaac_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("ur3_rl_bridge"), "/launch/isaac_foundation_pose_realsense.launch.py"]
        ),
        launch_arguments={"launch_rviz": "false"}.items(),
        condition=IfCondition(launch_isaac),
    )

    fastdds_transports = os.environ.get("FASTDDS_BUILTIN_TRANSPORTS", "").strip()

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("launch_isaac", default_value="true"),
            *(
                []
                if fastdds_transports
                else [SetEnvironmentVariable(name="FASTDDS_BUILTIN_TRANSPORTS", value="UDPv4")]
            ),
            SetParameter(name="use_sim_time", value=use_sim),
            bridge_launch,
            isaac_launch,
        ]
    )
