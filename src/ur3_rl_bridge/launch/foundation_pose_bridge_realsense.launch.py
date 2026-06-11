#!/usr/bin/env python3
"""YOLO→Detection + segmentation for Isaac FoundationPose using Intel RealSense D435i topics."""

import os

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
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
                    {"use_sim_time": False},
                    {
                        "yolo_model_path": yolo_path,
                        "rgb_topic": "/camera/camera/color/image_raw",
                        "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
                        "camera_info_topic": "/camera/camera/color/camera_info",
                        "camera_frame_fallback": "camera_color_optical_frame",
                        "detection_topic": "/foundation_pose/yolo_detection2_d_array",
                        "object_cloud_topic": "/foundation_pose/object_cloud",
                        "target_class": "Cylinder_1",
                        "min_confidence": 0.02,
                        "predict_conf_floor": 0.01,
                        "use_latched_bbox_on_trigger": True,
                        "use_yolo_segmentation_mask": True,
                        "segmentation_topic": "/segmentation",
                        "segmentation_mask_width": 640,
                        "segmentation_mask_height": 480,
                    },
                ],
            ),
        ]
    )
