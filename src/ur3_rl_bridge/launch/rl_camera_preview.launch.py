#!/usr/bin/env python3
"""Standalone OpenCV preview for MuJoCo rl_camera topics (run while sim is up)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                'rgb_topic',
                default_value='/rl_camera/color',
                description='sensor_msgs/Image RGB topic from mujoco_ros2_simulation',
            ),
            DeclareLaunchArgument(
                'depth_topic',
                default_value='/rl_camera/depth',
                description='sensor_msgs/Image depth topic (32FC1)',
            ),
            Node(
                package='ur3_rl_bridge',
                executable='mujoco_rl_camera_preview',
                output='screen',
                parameters=[
                    {'rgb_topic': LaunchConfiguration('rgb_topic')},
                    {'depth_topic': LaunchConfiguration('depth_topic')},
                    {'show_depth': True},
                    {'display_hz': 30.0},
                ],
            ),
        ]
    )
