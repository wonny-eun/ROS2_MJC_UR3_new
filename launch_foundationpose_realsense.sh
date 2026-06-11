#!/usr/bin/env bash
# FoundationPose stack for real UR3 + Intel RealSense D435i.
# Start RealSense first: bash ~/ur3_control/launch_realsense_d435i_640x480.sh

set +e
source "${HOME}/ur3_control/source_ws.sh"

FP_ENV="${HOME}/isaac_ros_assets/setup_foundationpose_env.sh"
if [[ -f "${FP_ENV}" ]]; then
  # shellcheck source=/dev/null
  source "${FP_ENV}"
fi

echo "Launching FoundationPose (RealSense D435i)…"
echo "  Requires: RealSense driver + Isaac TensorRT engines"
ros2 launch ur3_rl_bridge foundation_pose_stack_realsense.launch.py "$@"

echo ""
echo "FoundationPose stack exited. Shell stays open."
exec "${SHELL:-/bin/bash}" -i
