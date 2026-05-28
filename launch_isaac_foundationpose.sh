#!/usr/bin/env bash
# Isaac FoundationPose for MuJoCo rl_camera (run after sim + bridge trigger).
#
# Run (do NOT source):
#   bash ~/ur3_control/launch_isaac_foundationpose.sh
#
# Keep this terminal open while Isaac runs (Ctrl+C to stop).

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "ERROR: Do not 'source' this script." >&2
  echo "  Run: bash ~/ur3_control/launch_isaac_foundationpose.sh" >&2
  return 2 2>/dev/null || exit 2
fi

UR3_ROOT="${HOME}/ur3_control"
LOG_DIR="${UR3_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/launch_isaac_foundationpose.log"

exec > >(tee -a "${LOG_FILE}") 2>&1
echo "=== $(date -Is) launch_isaac_foundationpose.sh ==="

# No 'exec' — keep bash alive so Cursor terminals stay open on failure.
set -eo pipefail

source "${UR3_ROOT}/source_ws.sh"
export FOUNDATION_POSE_OBJECT="${FOUNDATION_POSE_OBJECT:-Box_1}"
source "${HOME}/isaac_ros_assets/setup_foundationpose_env.sh"
# Ensure GXF tensorops can load libnppial (ResizeNode) in composable containers.
export LD_LIBRARY_PATH="${NPP_LD_LIBRARY_PATH:-}${NPP_LD_LIBRARY_PATH:+:}${LD_LIBRARY_PATH}"

echo "Log file: ${LOG_FILE}"
echo "Launching Isaac FoundationPose (Ctrl+C to stop)..."

_launch_code=0
ros2 launch ur3_rl_bridge isaac_foundation_pose_rlcamera.launch.py "$@" || _launch_code=$?

echo ""
if [ "${_launch_code}" -eq 0 ]; then
  echo "Launch ended normally (exit 0)."
else
  echo "Launch exited with code ${_launch_code}."
  echo "See last 80 lines: tail -80 ${LOG_FILE}"
  tail -80 "${LOG_FILE}" 2>/dev/null || true
fi

if [ "${_launch_code}" -ne 0 ] && [ -t 0 ]; then
  read -r -p "Launch failed — press Enter to close this terminal..." _ || true
fi
exit "${_launch_code}"
