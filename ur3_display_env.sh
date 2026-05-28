#!/usr/bin/env bash
# Source from launch scripts to set DISPLAY / XAUTHORITY (safe with set -u).
#   source ~/ur3_control/ur3_display_env.sh

if command -v who >/dev/null 2>&1; then
  _ur3_disp=$(who 2>/dev/null | awk -v u="${USER:-$(whoami)}" '$1==u && $2 ~ /^:/{print $2; exit}')
  if [ -n "$_ur3_disp" ]; then
    export DISPLAY="$_ur3_disp"
  fi
fi
if [ -z "${DISPLAY:-}" ]; then
  if [ -S /tmp/.X11-unix/X1 ]; then
    export DISPLAY=:1
  elif [ -S /tmp/.X11-unix/X0 ]; then
    export DISPLAY=:0
  fi
fi
if [ -z "${XAUTHORITY:-}" ]; then
  _uid=$(id -u)
  if [ -f "/run/user/${_uid}/gdm/Xauthority" ]; then
    export XAUTHORITY="/run/user/${_uid}/gdm/Xauthority"
  elif [ -f "${HOME}/.Xauthority" ]; then
    export XAUTHORITY="${HOME}/.Xauthority"
  fi
fi
export QT_X11_NO_MITSHM=1
if command -v xhost >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
  xhost +local: >/dev/null 2>&1 || true
fi
