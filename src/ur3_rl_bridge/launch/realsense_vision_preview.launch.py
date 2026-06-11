#!/usr/bin/env python3
"""Real D435i OpenCV RGB/depth + YOLO preview (mirrors sim enable_* preview flags)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

_DEFAULT_YOLO_MODEL = (
    "/home/wonny/ur3_control/runs/segment/ur3_multi_sim_real_ultra/weights/best.pt"
)


def generate_launch_description():
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    yolo_model_path = LaunchConfiguration("yolo_model_path")
    yolo_target_class = LaunchConfiguration("yolo_target_class")
    enable_camera_preview = LaunchConfiguration("enable_camera_preview")
    enable_yolo_preview = LaunchConfiguration("enable_yolo_preview")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rgb_topic",
                default_value="/camera/camera/color/image_raw",
                description="RealSense color image topic (same stream as action_sequencer vision)",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera/camera/aligned_depth_to_color/image_raw",
                description="RealSense aligned depth topic (16UC1, mm)",
            ),
            DeclareLaunchArgument("display_hz", default_value="30.0"),
            DeclareLaunchArgument("depth_max_m", default_value="2.0"),
            DeclareLaunchArgument(
                "enable_camera_preview",
                default_value="true",
                description="OpenCV D435i RGB + depth windows (like enable_mujoco_rl_camera_preview in sim).",
            ),
            DeclareLaunchArgument(
                "enable_yolo_preview",
                default_value="true",
                description="YOLO annotated OpenCV window + /yolo/annotated (like enable_yolo_object_preview in sim).",
            ),
            DeclareLaunchArgument(
                "yolo_model_path",
                default_value=_DEFAULT_YOLO_MODEL,
                description="Path to trained YOLO segmentation model (best.pt).",
            ),
            DeclareLaunchArgument(
                "yolo_target_class",
                default_value="",
                description="Optional class filter, e.g. Cylinder_2. Empty shows all classes.",
            ),
            DeclareLaunchArgument(
                "yolo_scene_unique_classes",
                default_value="true",
                description="Assign one unique label per physical object (Box_1, Cylinder_1, Cylinder_2).",
            ),
            Node(
                package="ur3_rl_bridge",
                executable="mujoco_rl_camera_preview",
                output="screen",
                condition=IfCondition(enable_camera_preview),
                parameters=[
                    {"rgb_topic": rgb_topic},
                    {"depth_topic": depth_topic},
                    {"show_rgb": True},
                    {"show_depth": True},
                    {"display_hz": LaunchConfiguration("display_hz")},
                    {"depth_max_m": LaunchConfiguration("depth_max_m")},
                    {"rgb_window_title": "D435i RGB"},
                    {"depth_window_title": "D435i depth"},
                ],
            ),
            Node(
                package="ur3_rl_bridge",
                executable="yolo_object_preview",
                output="screen",
                condition=IfCondition(enable_yolo_preview),
                parameters=[
                    {"rgb_topic": rgb_topic},
                    {"model_path": yolo_model_path},
                    {"target_class": yolo_target_class},
                    {"display_hz": 10.0},
                    {"conf": 0.35},
                    {"predict_conf_floor": 0.01},
                    {"iou": 0.5},
                    {"show_window": True},
                    {"publish_annotated": True},
                    {"annotated_topic": "/yolo/annotated"},
                    {"scene_unique_classes": LaunchConfiguration("yolo_scene_unique_classes")},
                    {"scene_assign_rescore_crops": True},
                    {"scene_assign_cluster_iou": 0.45},
                    {
                        "exclusive_scene_classes": [
                            "Box_1",
                            "Cylinder_1",
                            "Cylinder_2",
                        ]
                    },
                ],
            ),
        ]
    )
