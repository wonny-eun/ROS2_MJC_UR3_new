#!/usr/bin/env bash
# Stop UR ROS driver processes without broad `pkill -f` patterns.
# Broad patterns can match the shell/terminal command line and close Cursor terminals.

set +e

_stay_open() {
  echo ""
  echo "Shell stays open. Type exit to close, or Ctrl+C if this is a temporary terminal."
  local sh="${SHELL:-/bin/bash}"
  if [ -r /dev/tty ] && [ -w /dev/tty ]; then
    exec "${sh}" -i </dev/tty >/dev/tty 2>&1
  fi
  while true; do
    sleep 3600
  done
}

echo "Stopping UR driver nodes by exact process name..."

for name in \
  ur_ros2_control_node \
  dashboard_client \
  robot_state_helper \
  controller_stopper_node \
  urscript_interface \
  trajectory_until_node \
  spawner
do
  if pgrep -x "${name}" >/dev/null 2>&1; then
    echo "Stopping ${name}"
    pkill -TERM -x "${name}" 2>/dev/null || true
  fi
done

sleep 1

for name in \
  ur_ros2_control_node \
  dashboard_client \
  robot_state_helper \
  controller_stopper_node \
  urscript_interface \
  trajectory_until_node \
  spawner
do
  if pgrep -x "${name}" >/dev/null 2>&1; then
    echo "Force stopping ${name}"
    pkill -KILL -x "${name}" 2>/dev/null || true
  fi
done

echo "Remaining matching UR processes:"
ps -eo pid,comm,args | awk '/ur_ros2_control_node|dashboard_client|robot_state_helper|controller_stopper_node|urscript_interface|trajectory_until_node/ && !/awk/ {print}'

echo ""
echo "Done."
_stay_open
