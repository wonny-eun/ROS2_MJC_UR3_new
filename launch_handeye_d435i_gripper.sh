#!/usr/bin/env bash
# Publish tool0 -> camera_color_optical_frame from hand-eye calibration YAML.
#
# Use with:
#   - URDF use_handeye_camera_tf:=true (default) — mount/rl_camera stay for MoveIt collision;
#     optical frame TF comes from this node, not URDF.
#   - RealSense publish_tf:=false
#
# Start after the UR driver (tool0 must exist in TF).

set +e

HANDEYE_YAML="${UR3_HANDEYE_FILE:-${HOME}/ur3_control/src/ROS2_MuJoCo_UR3/src/ur3_rl_bridge/config/handeye/d435i_on_gripper.yaml}"

if [ ! -f "${HANDEYE_YAML}" ]; then
  echo "ERROR: hand-eye file not found: ${HANDEYE_YAML}" >&2
  exec "${SHELL:-/bin/bash}" -i
fi

# shellcheck source=/dev/null
source "${HOME}/ur3_control/source_ws.sh"

echo "Hand-eye TF publisher:"
echo "  calibration: ${HANDEYE_YAML}"
echo "  publishes:   tool0 -> camera_color_optical_frame (static)"
echo "  URDF keeps:  tool0 -> gripper -> d435i_mount -> rl_camera_frame (collision mesh)"

ros2 run ur3_rl_bridge handeye_tf_publisher --ros-args \
  -p "calibration_file:=${HANDEYE_YAML}"

echo ""
echo "Hand-eye TF publisher exited. Shell stays open."
exec "${SHELL:-/bin/bash}" -i
