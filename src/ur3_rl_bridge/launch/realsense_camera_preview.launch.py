#!/usr/bin/env python3
"""OpenCV RGB + depth preview for Intel RealSense D435i (run while realsense2_camera is up)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rgb_topic",
                default_value="/camera/camera/color/image_raw",
                description="RealSense color image topic",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera/camera/aligned_depth_to_color/image_raw",
                description="RealSense aligned depth topic (16UC1, mm)",
            ),
            DeclareLaunchArgument("display_hz", default_value="30.0"),
            DeclareLaunchArgument("depth_max_m", default_value="2.0"),
            Node(
                package="ur3_rl_bridge",
                executable="mujoco_rl_camera_preview",
                output="screen",
                parameters=[
                    {"rgb_topic": LaunchConfiguration("rgb_topic")},
                    {"depth_topic": LaunchConfiguration("depth_topic")},
                    {"show_rgb": True},
                    {"show_depth": True},
                    {"display_hz": LaunchConfiguration("display_hz")},
                    {"depth_max_m": LaunchConfiguration("depth_max_m")},
                    {"rgb_window_title": "D435i RGB"},
                    {"depth_window_title": "D435i depth"},
                ],
            ),
        ]
    )
