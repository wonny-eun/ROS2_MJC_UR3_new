#!/usr/bin/env bash
# RealSense D435i OpenCV RGB/depth + YOLO preview (same roles as sim UR3_GUI_FULL previews).
# Start after: bash ~/ur3_control/launch_realsense_d435i_640x480.sh (~7 s until streams are up).

set +e

source "${HOME}/ur3_control/ur3_display_env.sh"
source "${HOME}/ur3_control/source_ws.sh"

RGB_TOPIC="${UR3_RGB_TOPIC:-/camera/camera/color/image_raw}"
DEPTH_TOPIC="${UR3_DEPTH_TOPIC:-/camera/camera/aligned_depth_to_color/image_raw}"
YOLO_MODEL="${UR3_YOLO_MODEL:-/home/wonny/ur3_control/runs/segment/ur3_multi_sim_real_ultra/weights/best.pt}"
YOLO_TARGET="${UR3_YOLO_TARGET_CLASS:-}"
ENABLE_CAM="${UR3_CAMERA_PREVIEW:-true}"
ENABLE_YOLO="${UR3_YOLO_PREVIEW:-true}"

if [[ "${ENABLE_YOLO}" == "true" && ! -f "${YOLO_MODEL}" ]]; then
  echo "WARNING: YOLO model not found: ${YOLO_MODEL} — disabling YOLO preview."
  ENABLE_YOLO="false"
fi

echo "RealSense vision preview (RealSense driver must be running):"
echo "  rgb:   ${RGB_TOPIC}"
echo "  depth: ${DEPTH_TOPIC}"
echo "  camera_preview: ${ENABLE_CAM}"
echo "  yolo_preview:   ${ENABLE_YOLO}"
echo "  yolo_model:     ${YOLO_MODEL}"

VISION_LAUNCH_ARGS=(
  "rgb_topic:=${RGB_TOPIC}"
  "depth_topic:=${DEPTH_TOPIC}"
  "enable_camera_preview:=${ENABLE_CAM}"
  "enable_yolo_preview:=${ENABLE_YOLO}"
  "yolo_model_path:=${YOLO_MODEL}"
)
if [[ -n "${YOLO_TARGET}" ]]; then
  VISION_LAUNCH_ARGS+=("yolo_target_class:=${YOLO_TARGET}")
fi

ros2 launch ur3_rl_bridge realsense_vision_preview.launch.py "${VISION_LAUNCH_ARGS[@]}"

echo ""
echo "Vision preview exited. Shell stays open."
exec "${SHELL:-/bin/bash}" -i
