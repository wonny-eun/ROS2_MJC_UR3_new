#!/usr/bin/env bash
# Launch MoveIt for the real UR3 from /opt/ros only.
# Start this after launch_real_ur3_driver_clean.sh is running.
#
# Optional RealSense vision previews (same roles as sim UR3_GUI_FULL=1):
#   UR3_CAMERA_PREVIEW=true   (default) — OpenCV D435i RGB + depth windows
#   UR3_YOLO_PREVIEW=true     (default) — YOLO annotated window + /yolo/annotated
#   UR3_YOLO_MODEL=.../best.pt (default matches simulation)
#   Requires RealSense driver: bash ~/ur3_control/launch_realsense_d435i_640x480.sh
#
# RViz is on by default. Disable only with: UR3_MOVEIT_RVIZ=false

set +e

UR_TYPE="${UR3_TYPE:-ur3}"
case "${UR3_MOVEIT_RVIZ:-true}" in
  false|False|FALSE|0|no|No|NO) LAUNCH_RVIZ="false" ;;
  *) LAUNCH_RVIZ="true" ;;
esac
CAMERA_PREVIEW="${UR3_CAMERA_PREVIEW:-true}"
YOLO_PREVIEW="${UR3_YOLO_PREVIEW:-true}"
YOLO_MODEL="${UR3_YOLO_MODEL:-/home/wonny/ur3_control/runs/segment/ur3_multi_sim_real_ultra/weights/best.pt}"
YOLO_TARGET="${UR3_YOLO_TARGET_CLASS:-}"
REAL_SCENE_XACRO="${UR3_REAL_SCENE_XACRO:-/home/wonny/ur3_control/real_urdf/ur3_real_scene.urdf.xacro}"
MOVEIT_LAUNCH_FILE="/tmp/ur_moveit_real_scene_${USER}.launch.py"
RGB_TOPIC="${UR3_RGB_TOPIC:-/camera/camera/color/image_raw}"
DEPTH_TOPIC="${UR3_DEPTH_TOPIC:-/camera/camera/aligned_depth_to_color/image_raw}"

unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset ROS_PACKAGE_PATH

# shellcheck source=/dev/null
source "${HOME}/ur3_control/ur3_display_env.sh"
source /opt/ros/humble/setup.bash
# Do NOT source the workspace overlay here — it breaks MoveIt (robot_description_semantic).
# Camera preview runs in a subshell with the workspace sourced separately.

PREVIEW_PID=""
stop_camera_preview() {
  if [[ -n "${PREVIEW_PID}" ]] && kill -0 "${PREVIEW_PID}" 2>/dev/null; then
    kill -INT "${PREVIEW_PID}" 2>/dev/null
    wait "${PREVIEW_PID}" 2>/dev/null
  fi
}
trap stop_camera_preview EXIT INT TERM

if [[ "${YOLO_PREVIEW}" == "true" && ! -f "${YOLO_MODEL}" ]]; then
  echo "WARNING: YOLO model not found (${YOLO_MODEL}); disabling YOLO preview."
  YOLO_PREVIEW="false"
fi

echo "Launching real UR MoveIt:"
echo "  ur_type: ${UR_TYPE}"
echo "  launch_rviz: ${LAUNCH_RVIZ}"
echo "  display: ${DISPLAY:-unset}"
echo "  camera_preview: ${CAMERA_PREVIEW}"
echo "  yolo_preview: ${YOLO_PREVIEW}"
echo "  yolo_model: ${YOLO_MODEL}"
echo "  real_scene_xacro: ${REAL_SCENE_XACRO}"

VISION_LOG="${HOME}/.ros/ur3_realsense_vision_preview_${USER}.log"

if [[ "${CAMERA_PREVIEW}" == "true" || "${YOLO_PREVIEW}" == "true" ]]; then
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "WARNING: DISPLAY is unset — skipping OpenCV/YOLO preview."
    echo "  Try: source ~/ur3_control/ur3_display_env.sh"
  else
    echo "Starting RealSense vision preview (rgb=${RGB_TOPIC}, DISPLAY=${DISPLAY})"
    echo "  RealSense driver must be running: bash ~/ur3_control/launch_realsense_d435i_640x480.sh"
    echo "  Log: ${VISION_LOG}"
    VISION_LAUNCH_ARGS=(
      "rgb_topic:=${RGB_TOPIC}"
      "depth_topic:=${DEPTH_TOPIC}"
      "enable_camera_preview:=${CAMERA_PREVIEW}"
      "enable_yolo_preview:=${YOLO_PREVIEW}"
      "yolo_model_path:=${YOLO_MODEL}"
    )
    if [[ -n "${YOLO_TARGET}" ]]; then
      VISION_LAUNCH_ARGS+=("yolo_target_class:=${YOLO_TARGET}")
    fi
    bash -c "
      source \"${HOME}/ur3_control/source_ws.sh\"
      exec ros2 launch ur3_rl_bridge realsense_vision_preview.launch.py ${VISION_LAUNCH_ARGS[*]}
    " >"${VISION_LOG}" 2>&1 &
    PREVIEW_PID=$!
    sleep 1
    if ! kill -0 "${PREVIEW_PID}" 2>/dev/null; then
      echo "ERROR: Vision preview exited immediately. Last log lines:"
      tail -20 "${VISION_LOG}" 2>/dev/null || true
    fi
  fi
fi

python3 - "${MOVEIT_LAUNCH_FILE}" "${REAL_SCENE_XACRO}" <<'PY'
import sys
from pathlib import Path

src = Path("/opt/ros/humble/share/ur_moveit_config/launch/ur_moveit.launch.py")
dst = Path(sys.argv[1])
real_scene_xacro = sys.argv[2]
text = src.read_text()
old_desc = '''            PathJoinSubstitution([FindPackageShare(description_package), "urdf", description_file]),'''
new_desc = f'''            "{real_scene_xacro}",'''
if old_desc not in text:
    raise SystemExit("Could not patch ur_moveit.launch.py description xacro path")
patched = text.replace(old_desc, new_desc)
srdf_marker = 'FindPackageShare(moveit_config_package), "srdf", moveit_config_file'
if srdf_marker not in patched or "/u/r/./s/r/d/f" in patched:
    raise SystemExit("Refusing to write launch file: SRDF xacro path looks corrupted")
dst.write_text(patched)
PY

ros2 launch "${MOVEIT_LAUNCH_FILE}" \
  ur_type:="${UR_TYPE}" \
  use_sim_time:=false \
  launch_rviz:="${LAUNCH_RVIZ}"

echo ""
echo "MoveIt launch exited. Shell stays open."
exec "${SHELL:-/bin/bash}" -i
