#!/usr/bin/env python3
"""
Bring up Intel RealSense with RGB + depth for RL / perception.

Requires: sudo apt install ros-${ROS_DISTRO}-realsense2-camera
  (and udev rules / RealSense SDK as per Intel docs)

Typical topics (default namespace /camera, camera_name camera):
  - Color: /camera/camera/color/image_raw
  - Depth (aligned to color): /camera/camera/aligned_depth_to_color/image_raw
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    camera_namespace = LaunchConfiguration("camera_namespace")
    camera_name = LaunchConfiguration("camera_name")
    serial_no = LaunchConfiguration("serial_no")

    rs_launch = PathJoinSubstitution(
        [FindPackageShare("realsense2_camera"), "launch", "rs_launch.py"]
    )

    included = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(rs_launch),
        launch_arguments={
            "camera_name": camera_name,
            "camera_namespace": camera_namespace,
            "serial_no": serial_no,
            "enable_color": "true",
            "enable_depth": "true",
            "enable_infra1": "false",
            "enable_infra2": "false",
            "rgb_camera.color_profile": "640x480x30",
            "depth_module.depth_profile": "640x480x30",
            "enable_sync": "true",
            "align_depth.enable": "true",
            "publish_tf": "true",
            "pointcloud.enable": "false",
            "log_level": "info",
        }.items(),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "camera_namespace",
                default_value="camera",
                description="ROS namespace for the RealSense stack.",
            ),
            DeclareLaunchArgument(
                "camera_name",
                default_value="camera",
                description="camera_name parameter passed to realsense2_camera.",
            ),
            DeclareLaunchArgument(
                "serial_no",
                default_value="''",
                description="Device serial number, or empty for first device.",
            ),
            included,
        ]
    )
