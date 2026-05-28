#!/usr/bin/env bash
# MuJoCo + RViz + optional camera/YOLO previews.
#
# Run (do NOT source — sourcing would run exit in your interactive shell):
#   bash ~/ur3_control/launch_sim_gui.sh
#
# All preview windows (MuJoCo + RViz + OpenCV + YOLO):
#   UR3_GUI_FULL=1 bash ~/ur3_control/launch_sim_gui.sh
#
# Default: MuJoCo + RViz only (safer in Cursor — fewer OpenGL windows).
#
# Physics only (no GUI windows):
#   bash ~/ur3_control/launch_sim_headless.sh

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "ERROR: Do not 'source' this script." >&2
  echo "  Run: bash ~/ur3_control/launch_sim_gui.sh" >&2
  return 2 2>/dev/null || exit 2
fi

UR3_ROOT="${HOME}/ur3_control"
LOG_DIR="${UR3_ROOT}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/launch_sim_gui.log"

# Avoid `exec > >(tee …)` — Ctrl+C can SIGPIPE/kill the shell and close Cursor's terminal tab.
_log() {
  echo "$@" | tee -a "${LOG_FILE}"
}

_stay_in_shell() {
  echo ""
  echo "Shell ready (type commands here; 'exit' closes this terminal tab)."
  exec "${SHELL:-/bin/bash}" -i
}

_log "=== $(date -Is) launch_sim_gui.sh ==="

set -e

unset UR3_FORCE_MUJOCO_HEADLESS
unset LIBGL_ALWAYS_SOFTWARE

# shellcheck source=/dev/null
source "${UR3_ROOT}/ur3_display_env.sh"

if ! "${UR3_ROOT}/ur3_check_gpu.sh" | tee -a "${LOG_FILE}"; then
  echo ""
  _log "GPU/display preflight failed. Log: ${LOG_FILE}"
  _log "Headless fallback: bash ~/ur3_control/launch_sim_headless.sh"
  read -r -p "Press Enter to return to shell..." _ || true
  _stay_in_shell
fi

set +u
# shellcheck source=/dev/null
source "${UR3_ROOT}/source_ws.sh"
set -e

if [ -z "${DISPLAY:-}" ]; then
  _log "ERROR: DISPLAY is not set."
  read -r -p "Press Enter to return to shell..." _ || true
  _stay_in_shell
fi
if [ -z "${XAUTHORITY:-}" ]; then
  _log "ERROR: XAUTHORITY is not set — RViz/MuJoCo GLX will fail."
  read -r -p "Press Enter to return to shell..." _ || true
  _stay_in_shell
fi

YOLO_MODEL="/home/wonny/ur3_control/runs/segment/ur3_multi_sim_real_ultra/weights/best.pt"
ENABLE_YOLO="false"
ENABLE_CAM_PREVIEW="false"
if [ "${UR3_GUI_FULL:-0}" = "1" ]; then
  ENABLE_YOLO="true"
  ENABLE_CAM_PREVIEW="true"
  if [ ! -f "${YOLO_MODEL}" ]; then
    ENABLE_YOLO="false"
    _log "WARN: YOLO model missing (${YOLO_MODEL}); yolo preview disabled."
  fi
  _log "UR3_GUI_FULL=1: MuJoCo + RViz + OpenCV depth + YOLO preview."
else
  _log "Default: MuJoCo + RViz only. For all previews: UR3_GUI_FULL=1 bash $0"
fi

if [ "${UR3_LAUNCH_DRY_RUN:-0}" = "1" ]; then
  _log "Dry run OK (DISPLAY=${DISPLAY:-unset}, workspace sourced)."
  exit 0
fi

_log "Using DISPLAY=${DISPLAY} XAUTHORITY=${XAUTHORITY}"
_log "Log file: ${LOG_FILE}"
_log "Launching MuJoCo viewer + RViz (Ctrl+C to stop)..."

_user_stop=0
on_int() {
  _user_stop=1
}
trap on_int INT

_launch_code=0
set +e
ros2 launch ur3_mujoco_sim ur3_mujoco_moveit.launch.py \
  headless:=false \
  enable_rviz:=true \
  enable_mujoco_rl_camera_preview:="${ENABLE_CAM_PREVIEW}" \
  enable_yolo_object_preview:="${ENABLE_YOLO}"
_launch_code=$?
set -e

trap - INT

# Ctrl+C (or SIGTERM): keep this terminal tab open with an interactive shell.
if [ "${_user_stop}" -eq 1 ] || [ "${_launch_code}" -eq 130 ] || [ "${_launch_code}" -eq 143 ]; then
  _log ""
  _log "Stopped (Ctrl+C)."
  _stay_in_shell
fi

echo ""
if [ "${_launch_code}" -eq 0 ]; then
  _log "Launch ended normally (exit 0)."
else
  _log "Launch exited with code ${_launch_code}."
  _log "See last 80 lines: tail -80 ${LOG_FILE}"
  tail -80 "${LOG_FILE}" 2>/dev/null || true
  _stay_in_shell
fi

exit 0
