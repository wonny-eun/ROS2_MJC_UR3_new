#!/usr/bin/env python3
"""Start YOLO→Detection2DArray + object point cloud for Isaac FoundationPose + RViz."""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    # Prefer same YOLO path as ur3_pick_task YAML; override via launch arg in a wrapper if needed.
    default_yolo = "/home/wonny/ur3_control/runs/segment/ur3_multi_sim_real_ultra/weights/best.pt"
    yolo_path = os.environ.get("YOLO_MODEL_PATH", default_yolo)

    return LaunchDescription(
        [
            Node(
                package="ur3_rl_bridge",
                executable="foundation_pose_bridge_node",
                name="foundation_pose_bridge",
                output="screen",
                parameters=[
                    {"use_sim_time": True},
                    {
                        "yolo_model_path": yolo_path,
                        "rgb_topic": "/rl_camera/noisy/color",
                        "depth_topic": "/rl_camera/noisy/depth",
                        "camera_info_topic": "/rl_camera/camera_info",
                        "camera_frame_fallback": "camera_color_optical_frame",
                        "detection_topic": "/foundation_pose/yolo_detection2_d_array",
                        "object_cloud_topic": "/foundation_pose/object_cloud",
                        "yolo_exclusive_scene_classes": ["Box_1", "Cylinder_1", "Cylinder_2"],
                        "target_class": "Box_1",
                        "min_confidence": 0.02,
                        "predict_conf_floor": 0.01,
                        "use_latched_bbox_on_trigger": False,
                    }
                ],
            ),
        ]
    )
