#!/usr/bin/env python3
"""
Isaac ROS FoundationPose for UR3 real robot + Intel RealSense D435i.

Requires RealSense driver:
  bash ~/ur3_control/launch_realsense_d435i_640x480.sh

Same env as sim (~/isaac_ros_assets/setup_foundationpose_env.sh):
  FOUNDATION_POSE_MESH, FOUNDATION_POSE_REFINE_ENGINE, FOUNDATION_POSE_SCORE_ENGINE
"""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node, SetParameter
from launch_ros.descriptions import ComposableNode

_CUDART_SEARCH_DIRS = (
    "/usr/local/cuda-12.6/targets/x86_64-linux/lib",
    "/usr/local/cuda-12.6/lib64",
    "/usr/local/cuda/targets/x86_64-linux/lib",
    "/usr/local/cuda-12/targets/x86_64-linux/lib",
)


def _find_libcudart_dir() -> str | None:
    for directory in _CUDART_SEARCH_DIRS:
        if os.path.isfile(os.path.join(directory, "libcudart.so.12")):
            return directory
    return None


def _find_libcvcuda_dir() -> str | None:
    for directory in ("/opt/nvidia/cvcuda0/lib", "/usr/lib/x86_64-linux-gnu", "/usr/local/cvcuda/lib"):
        if os.path.isfile(os.path.join(directory, "libcvcuda.so.0")):
            return directory
    return None


def _find_npp_lib_dir() -> str | None:
    home = os.path.expanduser("~")
    for directory in (
        f"{home}/isaac_ros_assets/npp_extract/usr/local/cuda-12.6/targets/x86_64-linux/lib",
        "/usr/local/cuda-12.6/targets/x86_64-linux/lib",
    ):
        if os.path.isfile(os.path.join(directory, "libnppial.so.12")):
            return directory
    return None


def _cuda_ld_library_path() -> str:
    parts: list[str] = []
    for directory in (_find_libcudart_dir(), _find_libcvcuda_dir(), _find_npp_lib_dir()):
        if directory and directory not in parts:
            parts.append(directory)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    for directory in reversed(parts):
        if directory not in existing.split(":"):
            existing = f"{directory}:{existing}" if existing else directory
    return existing


def _preflight_errors() -> list[str]:
    errors: list[str] = []
    try:
        from ament_index_python.packages import get_package_share_directory

        get_package_share_directory("isaac_ros_foundationpose")
    except Exception:
        errors.append(
            "ROS package 'isaac_ros_foundationpose' not found. Install:\n"
            "  sudo apt install ros-humble-isaac-ros-foundationpose"
        )
    for pkg in ("isaac_ros_image_proc", "isaac_ros_depth_image_proc"):
        try:
            from ament_index_python.packages import get_package_share_directory

            get_package_share_directory(pkg)
        except Exception:
            errors.append(
                f"ROS package '{pkg}' not found. Install:\n"
                f"  sudo apt install ros-humble-{pkg.replace('_', '-')}"
            )
    if _find_npp_lib_dir() is None:
        errors.append(
            "libnppial.so.12 missing (rgb Nitros bridge needs NPP). Install or extract:\n"
            "  apt-get download libnpp-12-6 && dpkg-deb -x libnpp-12-6_*.deb ~/isaac_ros_assets/npp_extract"
        )
    import subprocess

    ld = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=False)
    if "libnvToolsExt.so" not in (ld.stdout or ""):
        errors.append(
            "NVIDIA library libnvToolsExt.so.1 missing. Install:\n"
            "  sudo apt install libnvtoolsext1"
        )
    if _find_libcudart_dir() is None:
        errors.append(
            "libcudart.so.12 not found. Install:\n"
            "  sudo apt install -y cuda-cudart-12-6"
        )
    if _find_libcvcuda_dir() is None:
        errors.append(
            "libcvcuda.so.0 (CV-CUDA) not found. Install:\n"
            "  bash ~/isaac_ros_assets/install_cvcuda.sh"
        )
    for label, path in (
        ("FOUNDATION_POSE_REFINE_ENGINE", os.environ.get("FOUNDATION_POSE_REFINE_ENGINE", "")),
        ("FOUNDATION_POSE_SCORE_ENGINE", os.environ.get("FOUNDATION_POSE_SCORE_ENGINE", "")),
    ):
        if path and not os.path.isfile(path):
            errors.append(f"{label} file not found: {path}")
    return errors


def generate_launch_description() -> LaunchDescription:
    try:
        from ament_index_python.packages import get_package_share_directory
    except ImportError:  # pragma: no cover
        return LaunchDescription(
            [LogInfo(msg="ament_index_python missing — cannot resolve Isaac package share")]
        )

    preflight = _preflight_errors()
    if preflight:
        raise RuntimeError(
            "isaac_foundation_pose_realsense.launch.py: cannot start — fix:\n\n" + "\n\n".join(preflight)
        )

    isaac_share = get_package_share_directory("isaac_ros_foundationpose")
    try:
        mujoco_share = get_package_share_directory("ur3_mujoco_sim")
    except Exception:
        mujoco_share = ""

    fp_mesh_dir = os.path.join(mujoco_share, "meshes", "foundationpose") if mujoco_share else ""
    mesh_default = os.path.join(fp_mesh_dir, "square_1.obj") if fp_mesh_dir else ""
    texture_default = os.path.join(fp_mesh_dir, "square_1_texture.jpg") if fp_mesh_dir else ""
    mesh = os.environ.get("FOUNDATION_POSE_MESH") or mesh_default or "/tmp/missing_mesh.obj"
    refine = os.environ.get("FOUNDATION_POSE_REFINE_ENGINE", "/tmp/refine_trt_engine.plan")
    score = os.environ.get("FOUNDATION_POSE_SCORE_ENGINE", "/tmp/score_trt_engine.plan")
    texture = os.environ.get("FOUNDATION_POSE_TEXTURE") or texture_default or "/tmp/texture_map.png"
    if not os.path.isfile(texture):
        raise RuntimeError(
            f"FoundationPose texture missing: {texture}\n"
            "  Set FOUNDATION_POSE_TEXTURE or rebuild ur3_mujoco_sim meshes."
        )

    mask_w = int(os.environ.get("FOUNDATION_POSE_MASK_W", "640"))
    mask_h = int(os.environ.get("FOUNDATION_POSE_MASK_H", "480"))
    symmetry_raw = os.environ.get("FOUNDATION_POSE_SYMMETRY_AXES", "").strip()
    symmetry_axes = [part.strip() for part in symmetry_raw.split(",") if part.strip()]
    tf_reference_frame = os.environ.get("FOUNDATION_POSE_TF_REFERENCE_FRAME", "base_link").strip()
    tf_upright_raw = os.environ.get("FOUNDATION_POSE_TF_UPRIGHT_IN_BASE", "1").strip().lower()
    tf_upright_in_base = tf_upright_raw not in ("0", "false", "no", "off")
    long_axis_raw = os.environ.get("FOUNDATION_POSE_LONG_AXIS", "").strip()
    long_axis_in_object = [float(x) for x in long_axis_raw.split(",") if x.strip()] if long_axis_raw else []
    use_yolo_seg_raw = os.environ.get("FOUNDATION_POSE_USE_YOLO_SEG_MASK", "1").strip().lower()
    use_yolo_seg_mask = use_yolo_seg_raw not in ("0", "false", "no", "off")
    launch_rviz = LaunchConfiguration("launch_rviz")
    rgb_topic = LaunchConfiguration("rgb_topic")
    depth_topic = LaunchConfiguration("depth_topic")
    camera_info_topic = LaunchConfiguration("camera_info_topic")

    detections_topic = "/foundation_pose/yolo_detection2_d_array"
    foundationpose_params: dict = {
        "mesh_file_path": mesh,
        "texture_path": texture,
        "refine_engine_file_path": refine,
        "score_engine_file_path": score,
        "refine_input_tensor_names": ["input_tensor1", "input_tensor2"],
        "refine_input_binding_names": ["input1", "input2"],
        "refine_output_tensor_names": ["output_tensor1", "output_tensor2"],
        "refine_output_binding_names": ["output1", "output2"],
        "score_input_tensor_names": ["input_tensor1", "input_tensor2"],
        "score_input_binding_names": ["input1", "input2"],
        "score_output_tensor_names": ["output_tensor"],
        "score_output_binding_names": ["output1"],
    }
    if symmetry_axes:
        foundationpose_params["symmetry_axes"] = symmetry_axes

    foundationpose_node = ComposableNode(
        name="foundationpose",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::FoundationPoseNode",
        parameters=[foundationpose_params],
        remappings=[
            ("pose_estimation/depth_image", "/depth_registered/image_rect"),
            ("pose_estimation/image", "/rgb/image_rect_color"),
            ("pose_estimation/camera_info", "/rgb/camera_info"),
            ("pose_estimation/segmentation", "/segmentation"),
            ("pose_estimation/output", "/output"),
        ],
    )

    detection2_d_to_mask_node = ComposableNode(
        name="detection2_d_to_mask",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::Detection2DToMask",
        parameters=[{"mask_width": mask_w, "mask_height": mask_h}],
        remappings=[("detection2_d_array", detections_topic)],
    )

    rgb_nitros_bridge = ComposableNode(
        package="isaac_ros_image_proc",
        plugin="nvidia::isaac_ros::image_proc::ResizeNode",
        name="rgb_nitros_bridge",
        parameters=[
            {
                "output_width": mask_w,
                "output_height": mask_h,
                "keep_aspect_ratio": False,
                "encoding_desired": "rgb8",
            }
        ],
        remappings=[
            ("image", rgb_topic),
            ("camera_info", camera_info_topic),
            ("resize/image", "rgb/image_rect_color"),
            ("resize/camera_info", "rgb/camera_info"),
        ],
    )

    # RealSense aligned depth is 16UC1 (mm) — ConvertMetricNode accepts mono16 directly.
    depth_nitros_bridge = ComposableNode(
        package="isaac_ros_depth_image_proc",
        plugin="nvidia::isaac_ros::depth_image_proc::ConvertMetricNode",
        name="depth_nitros_bridge",
        remappings=[
            ("image_raw", depth_topic),
            ("image", "depth_registered/image_rect"),
        ],
    )

    fp_components = [foundationpose_node, rgb_nitros_bridge, depth_nitros_bridge]
    if not use_yolo_seg_mask:
        fp_components.insert(1, detection2_d_to_mask_node)

    fp_container = ComposableNodeContainer(
        name="foundationpose_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=fp_components,
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(isaac_share, "rviz", "foundationpose.rviz")],
        condition=IfCondition(launch_rviz),
    )

    cuda_ld = _cuda_ld_library_path()
    env_actions = []
    if cuda_ld:
        env_actions.append(SetEnvironmentVariable(name="LD_LIBRARY_PATH", value=cuda_ld))

    output_tf_params: dict = {
        "output_topic": "/output",
        "child_frame": "fp_object",
        "reference_frame": tf_reference_frame,
        "upright_in_base": tf_upright_in_base,
        "short_axis_in_object": [0.0, 1.0, 0.0],
        "tf_filter_enable": True,
        "tf_filter_window": 15,
        "tf_filter_require_full_window": True,
        "tf_filter_use_yaw_median": True,
        "tf_filter_lock_on_stable": True,
        "tf_filter_lock_stable_frames": 15,
        "tf_filter_lock_min_samples": 40,
        "tf_filter_lock_pos_tol_m": 0.003,
        "tf_filter_lock_rot_tol_deg": 1.0,
    }
    if len(long_axis_in_object) == 3:
        output_tf_params["long_axis_in_object"] = long_axis_in_object

    return LaunchDescription(
        [
            DeclareLaunchArgument("launch_rviz", default_value="false"),
            DeclareLaunchArgument(
                "rgb_topic",
                default_value="/camera/camera/color/image_raw",
            ),
            DeclareLaunchArgument(
                "depth_topic",
                default_value="/camera/camera/aligned_depth_to_color/image_raw",
            ),
            DeclareLaunchArgument(
                "camera_info_topic",
                default_value="/camera/camera/color/camera_info",
            ),
            SetParameter(name="use_sim_time", value=False),
            *env_actions,
            LogInfo(msg=f"Isaac FoundationPose (RealSense D435i): mesh={mesh}"),
            LogInfo(msg="  RealSense driver must be running (launch_realsense_d435i_640x480.sh)"),
            fp_container,
            rviz_node,
            Node(
                package="ur3_rl_bridge",
                executable="foundation_pose_output_tf_node",
                name="foundation_pose_output_tf",
                output="screen",
                parameters=[output_tf_params],
            ),
        ]
    )
