#!/usr/bin/env python3
"""
Bring up NVIDIA isaac_ros_foundationpose for the UR3 MuJoCo rl_camera stack.

Requires (e.g. from ~/.bashrc or setup_foundationpose_env.sh):

  FOUNDATION_POSE_MESH
  FOUNDATION_POSE_REFINE_ENGINE
  FOUNDATION_POSE_SCORE_ENGINE

Optional: FOUNDATION_POSE_MASK_W/H (default 640x480), FOUNDATION_POSE_TEXTURE (optional texture map).
"""

from __future__ import annotations

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node, SetParameter
from launch_ros.descriptions import ComposableNode

# libcudart from cuda-cudart-12-6 lives under cuda-12.6/, but default ldconfig only lists cuda-12.9 paths.
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
    """Return human-readable missing-dependency messages (empty if OK)."""
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
    _npp = _find_npp_lib_dir()
    if _npp is None:
        errors.append(
            "libnppial.so.12 missing (rgb Nitros bridge needs NPP). Install or extract:\n"
            "  apt-get download libnpp-12-6 && dpkg-deb -x libnpp-12-6_*.deb ~/isaac_ros_assets/npp_extract\n"
            "  (setup_foundationpose_env.sh adds it to LD_LIBRARY_PATH)"
        )
    import subprocess

    ld = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, check=False)
    if "libnvToolsExt.so" not in (ld.stdout or ""):
        errors.append(
            "NVIDIA library libnvToolsExt.so.1 missing (FoundationPose node will not load). Install:\n"
            "  sudo apt install libnvtoolsext1"
        )
    if _find_libcudart_dir() is None:
        errors.append(
            "libcudart.so.12 not found on disk. Install:\n"
            "  sudo apt install -y cuda-cudart-12-6\n"
            "Optional (system-wide): bash ~/isaac_ros_assets/fix_cuda_ldconfig.sh"
        )

    if _find_libcvcuda_dir() is None:
        errors.append(
            "libcvcuda.so.0 (CV-CUDA) not found. Isaac FoundationPose GXF will fail. Install:\n"
            "  bash ~/isaac_ros_assets/install_cvcuda.sh"
        )
    for label, path in (
        ("FOUNDATION_POSE_REFINE_ENGINE", os.environ.get("FOUNDATION_POSE_REFINE_ENGINE", "")),
        ("FOUNDATION_POSE_SCORE_ENGINE", os.environ.get("FOUNDATION_POSE_SCORE_ENGINE", "")),
    ):
        if path and not os.path.isfile(path):
            errors.append(f"{label} file not found: {path}\n  bash ~/isaac_ros_assets/build_foundationpose_engines.sh")
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
            "isaac_foundation_pose_rlcamera.launch.py: cannot start — fix:\n\n" + "\n\n".join(preflight)
        )

    try:
        isaac_share = get_package_share_directory("isaac_ros_foundationpose")
    except Exception as exc:
        raise RuntimeError(
            "isaac_ros_foundationpose not found — install ros-humble-isaac-ros-foundationpose"
        ) from exc

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
            "  Rebuild ur3_mujoco_sim after adding Blender textures, or set FOUNDATION_POSE_TEXTURE."
        )

    mask_w = int(os.environ.get("FOUNDATION_POSE_MASK_W", "640"))
    mask_h = int(os.environ.get("FOUNDATION_POSE_MASK_H", "480"))
    launch_rviz = LaunchConfiguration("launch_rviz")

    # Single composable container at root namespace (matches isaac_ros_foundationpose_realsense).
    detections_topic = "/foundation_pose/yolo_detection2_d_array"

    foundationpose_node = ComposableNode(
        name="foundationpose",
        package="isaac_ros_foundationpose",
        plugin="nvidia::isaac_ros::foundationpose::FoundationPoseNode",
        parameters=[
            {
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
        ],
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
            ("image", "/rl_camera/noisy/color"),
            ("camera_info", "/rl_camera/camera_info"),
            ("resize/image", "rgb/image_rect_color"),
            ("resize/camera_info", "rgb/camera_info"),
        ],
    )

    depth_nitros_bridge = ComposableNode(
        package="isaac_ros_depth_image_proc",
        plugin="nvidia::isaac_ros::depth_image_proc::ConvertMetricNode",
        name="depth_nitros_bridge",
        remappings=[
            ("image_raw", "/fp_bridge/depth_mono16"),
            ("image", "depth_registered/image_rect"),
        ],
    )

    fp_container = ComposableNodeContainer(
        name="foundationpose_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        composable_node_descriptions=[
            foundationpose_node,
            detection2_d_to_mask_node,
            rgb_nitros_bridge,
            depth_nitros_bridge,
        ],
        output="screen",
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=[
            "-d",
            os.path.join(isaac_share, "rviz", "foundationpose.rviz"),
        ],
        condition=IfCondition(launch_rviz),
    )

    cuda_ld = _cuda_ld_library_path()
    env_actions = []
    npp_dir = _find_npp_lib_dir()
    if cuda_ld:
        env_actions.append(SetEnvironmentVariable(name="LD_LIBRARY_PATH", value=cuda_ld))
        env_actions.append(
            LogInfo(
                msg=(
                    f"LD_LIBRARY_PATH: cuda={_find_libcudart_dir()} "
                    f"cvcuda={_find_libcvcuda_dir()} npp={npp_dir}"
                )
            )
        )

    depth_mono16_node = Node(
        package="ur3_rl_bridge",
        executable="foundation_pose_depth_mono16_node",
        name="foundation_pose_depth_mono16",
        output="screen",
        parameters=[
            {
                "input_topic": "/rl_camera/noisy/depth",
                "output_topic": "/fp_bridge/depth_mono16",
            }
        ],
    )

    return LaunchDescription(
        [
            SetParameter(name="use_sim_time", value=True),
            *env_actions,
            DeclareLaunchArgument("launch_rviz", default_value="false"),
            LogInfo(msg=f"Isaac FoundationPose: mesh={mesh} texture={texture}"),
            LogInfo(msg=f"  refine={refine}"),
            LogInfo(msg=f"  score={score}"),
            LogInfo(
                msg="FoundationPose: single Nitros container (FP + mask + rgb/depth bridges)"
            ),
            depth_mono16_node,
            fp_container,
            rviz_node,
            Node(
                package="ur3_rl_bridge",
                executable="foundation_pose_output_tf_node",
                name="foundation_pose_output_tf",
                output="screen",
                parameters=[
                    {
                        "output_topic": "/output",
                        "child_frame": "fp_object",
                        "tf_filter_enable": True,
                        "tf_filter_window": 10,
                        "tf_filter_require_full_window": False,
                        "tf_filter_lock_on_stable": True,
                        "tf_filter_lock_stable_frames": 10,
                        "tf_filter_lock_pos_tol_m": 0.003,
                        "tf_filter_lock_rot_tol_deg": 1.0,
                    }
                ],
            ),
        ]
    )
