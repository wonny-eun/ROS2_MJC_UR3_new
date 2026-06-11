#!/usr/bin/env bash
# Run ur3_pick_task action_sequencer without closing the Cursor terminal.
#
# Usage:
#   bash ~/ur3_control/run_action_sequencer.sh                # asks object + arm move speed
#   bash ~/ur3_control/run_action_sequencer.sh Box_1            # arm move speed scale 1.0
#   bash ~/ur3_control/run_action_sequencer.sh Cylinder_1 0.5   # half speed/accel for normal arm moves
#   bash ~/ur3_control/run_action_sequencer.sh UR3_Connect_Test 90 0.3
#       # FoundationPose grasp: absolute TCP roll 90° after FP locks grip point; arm speed 0.3
#   UR3_FP_TCP_ROLL_DEG=90 bash ~/ur3_control/run_action_sequencer.sh UR3_Connect_Test 0.3
#   UR3_SPEED_SCALE=0.3 bash ~/ur3_control/run_action_sequencer.sh Cylinder_1
#   UR3_ROBOT_BACKEND=real UR3_USE_SIM_TIME=false bash ~/ur3_control/run_action_sequencer.sh Cylinder_1 0.2
#   bash ~/ur3_control/run_real_action_sequencer.sh Cylinder_1 0.8   # uses ur3_action_sequence_real.yaml
#
# Do NOT: source this file (kills the shell if you have set -e).

if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
  echo "ERROR: Do not 'source' this script." >&2
  echo "  Run: bash ~/ur3_control/run_action_sequencer.sh [object_name]" >&2
  return 0 2>/dev/null || exit 0
fi

UR3_ROOT="${HOME}/ur3_control"

_stay_in_shell() {
  echo ""
  echo "Shell ready (sequencer stopped; type commands here)."
  # Keep this terminal tab open in Cursor/VS Code (do not bare-exit the script).
  local sh="${SHELL:-/bin/bash}"
  if [ -r /dev/tty ] && [ -w /dev/tty ]; then
    exec "${sh}" -i </dev/tty >/dev/tty 2>/dev/tty
  fi
  if [ -t 0 ]; then
    exec "${sh}" -i
  fi
  # Last resort: block until Enter so the tab does not vanish instantly.
  read -r -p "Press Enter to return to shell... " _ || true
  exec "${sh}" -i
}

# Never use set -e: a failed source/ros2 run must reach _stay_in_shell, not kill the tab.
set +e
set +u

# shellcheck source=/dev/null
source "${UR3_ROOT}/source_ws.sh"
_ws_source_rc=$?
if [ "${_ws_source_rc}" -ne 0 ]; then
  echo "ERROR: failed to source ${UR3_ROOT}/source_ws.sh (exit ${_ws_source_rc})" >&2
  _stay_in_shell
fi

CONFIG_FILE="${UR3_ACTION_CONFIG:-}"
if [ -z "${CONFIG_FILE}" ]; then
  _pkg_prefix="$(ros2 pkg prefix ur3_pick_task 2>/dev/null)"
  if [ "${UR3_ROBOT_BACKEND}" = "real" ]; then
    CONFIG_FILE="${_pkg_prefix}/share/ur3_pick_task/config/actions/ur3_action_sequence_real.yaml"
  else
    CONFIG_FILE="${_pkg_prefix}/share/ur3_pick_task/config/actions/ur3_action_sequence.yaml"
  fi
fi
if [ -z "${CONFIG_FILE}" ] || [ ! -f "${CONFIG_FILE}" ]; then
  echo "ERROR: ur3_pick_task not in workspace — build and source install/setup.bash" >&2
  _stay_in_shell
fi
UR3_USE_SIM_TIME="${UR3_USE_SIM_TIME:-true}"
UR3_ROBOT_BACKEND="${UR3_ROBOT_BACKEND:-sim}"

if ! python3 -c "import yaml; yaml.safe_load(open('${CONFIG_FILE}'))" 2>/dev/null; then
  echo "ERROR: Invalid YAML in config file:" >&2
  echo "  ${CONFIG_FILE}" >&2
  python3 -c "import yaml; yaml.safe_load(open('${CONFIG_FILE}'))" 2>&1 | head -5 >&2
  _stay_in_shell
fi

PROMPT_FOR_OBJECT="false"
if [ -n "${1:-}" ]; then
  OBJECT_NAME="$1"
else
  echo "Valid objects: Box_1, Cylinder_1, Cylinder_2, Gripping_Test, UR3_Connect_Test, ..."
  read -rp "Target Object: " OBJECT_NAME
  if [ -z "${OBJECT_NAME}" ]; then
    echo "No object name entered." >&2
    _stay_in_shell
  fi
fi

_is_fp_tcp_roll_deg() {
  python3 -c "import math; v=float('${1}'); assert math.isfinite(v); raise SystemExit(0 if abs(v) >= 5.0 else 1)" 2>/dev/null
}

TCP_ROLL_DEG="${UR3_FP_TCP_ROLL_DEG:-}"
if [ -n "${3:-}" ]; then
  TCP_ROLL_DEG="${2}"
  SPEED_SCALE="${3}"
elif [ -n "${2:-}" ]; then
  if _is_fp_tcp_roll_deg "${2}"; then
    TCP_ROLL_DEG="${2}"
    SPEED_SCALE="${UR3_SPEED_SCALE:-}"
  else
    SPEED_SCALE="${2}"
  fi
else
  SPEED_SCALE="${UR3_SPEED_SCALE:-}"
fi

if [ -n "${TCP_ROLL_DEG}" ]; then
  if ! python3 -c "import math; v=float('${TCP_ROLL_DEG}'); assert math.isfinite(v)" 2>/dev/null; then
    echo "ERROR: FoundationPose TCP roll must be a number in degrees (got '${TCP_ROLL_DEG}')" >&2
    _stay_in_shell
  fi
  TCP_ROLL_DEG="$(python3 -c "print(float('${TCP_ROLL_DEG}'))")"
fi

if [ -z "${SPEED_SCALE}" ]; then
  read -rp "Arm move speed scale (1.0 = normal, 0.5 = half speed): " SPEED_SCALE
fi
SPEED_SCALE="${SPEED_SCALE:-1.0}"
if ! python3 -c "import math; s=float('${SPEED_SCALE}'); assert math.isfinite(s) and s > 0" 2>/dev/null; then
  echo "ERROR: speed scale must be a number > 0 (got '${SPEED_SCALE}')" >&2
  _stay_in_shell
fi
SPEED_SCALE="$(python3 -c "print(float('${SPEED_SCALE}'))")"

echo "Running action_sequencer (object_name=${OBJECT_NAME}, arm_move_speed_scale=${SPEED_SCALE}, backend=${UR3_ROBOT_BACKEND}, use_sim_time=${UR3_USE_SIM_TIME})..."
if [ -n "${TCP_ROLL_DEG}" ]; then
  echo "FoundationPose grasp TCP roll (CLI): ${TCP_ROLL_DEG}° absolute about look-at axis"
fi
echo "Config: ${CONFIG_FILE}"
echo "Ctrl+C stops the sequence; this terminal tab stays open."

_user_stop=0
on_int() {
  _user_stop=1
}
trap on_int INT

_run_code=0
_FP_ROLL_ARGS=()
if [ -n "${TCP_ROLL_DEG}" ]; then
  _FP_ROLL_ARGS=(-p "foundation_pose_tcp_roll_deg:=${TCP_ROLL_DEG}")
fi
ros2 run ur3_pick_task action_sequencer --ros-args \
  -p "use_sim_time:=${UR3_USE_SIM_TIME}" \
  -p "config_file:=${CONFIG_FILE}" \
  -p "object_name:=${OBJECT_NAME}" \
  -p "prompt_for_object:=${PROMPT_FOR_OBJECT}" \
  -p "robot_backend:=${UR3_ROBOT_BACKEND}" \
  -p "speed_scale:=${SPEED_SCALE}" \
  "${_FP_ROLL_ARGS[@]}"
_run_code=$?
trap - INT

if [ "${_user_stop}" -eq 1 ] || [ "${_run_code}" -eq 130 ] || [ "${_run_code}" -eq 143 ]; then
  echo ""
  echo "Stopped (Ctrl+C)."
  _stay_in_shell
fi

if [ "${_run_code}" -ne 0 ]; then
  echo ""
  echo "action_sequencer exited with code ${_run_code}."
  _stay_in_shell
fi

echo "Action sequence finished."
_stay_in_shell
