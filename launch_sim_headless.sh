#!/usr/bin/env bash
# Stable sim: no MuJoCo GLFW / RViz / OpenCV windows (camera topics still publish).
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "ERROR: Run with: bash ~/ur3_control/launch_sim_headless.sh" >&2
  return 2 2>/dev/null || exit 2
fi

UR3_ROOT="${HOME}/ur3_control"
LOG_DIR="${UR3_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/launch_sim_headless.log"
exec > >(tee -a "${LOG_FILE}") 2>&1
echo "=== $(date -Is) launch_sim_headless.sh ==="

set -eo pipefail
# shellcheck source=/dev/null
source "${UR3_ROOT}/ur3_display_env.sh"

set +u
# shellcheck source=/dev/null
source "${UR3_ROOT}/source_ws.sh"
set -e

echo "Headless MuJoCo + MoveIt (no GUI). Log: ${LOG_FILE}"
_launch_code=0
ros2 launch ur3_mujoco_sim ur3_mujoco_moveit.launch.py \
  headless:=true \
  enable_rviz:=false \
  enable_mujoco_rl_camera_preview:=false \
  enable_yolo_object_preview:=false || _launch_code=$?

echo "Exit code: ${_launch_code}"
if [ "${_launch_code}" -ne 0 ] && [ -t 0 ]; then
  read -r -p "Launch failed — press Enter to close..." _ || true
fi
exit "${_launch_code}"
