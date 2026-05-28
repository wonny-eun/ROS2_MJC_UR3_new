#!/usr/bin/env bash
# Preflight for MuJoCo viewer + RViz (need working GLX). Run: ~/ur3_control/ur3_check_gpu.sh
set -euo pipefail

_ok=0
_warn=0
_fail=0

note() { echo "$*"; }
ok() { note "OK: $*"; _ok=$((_ok + 1)); }
warn() { note "WARN: $*"; _warn=$((_warn + 1)); }
fail() { note "FAIL: $*"; _fail=$((_fail + 1)); }

# --- DISPLAY / XAUTHORITY ---
# shellcheck source=/dev/null
source "${HOME}/ur3_control/ur3_display_env.sh"

note "=== UR3 display / GPU preflight ==="
note "DISPLAY=${DISPLAY:-<unset>}  XAUTHORITY=${XAUTHORITY:-<unset>}"

if [ -z "${DISPLAY:-}" ]; then
  fail "DISPLAY is not set — log in on the Ubuntu desktop (session :1), not pure SSH."
else
  if xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; then
    ok "X11 display $DISPLAY responds (xdpyinfo)."
  else
    fail "xdpyinfo failed on $DISPLAY — wrong DISPLAY or missing X cookie."
  fi
fi

if [ -z "${XAUTHORITY:-}" ]; then
  warn "XAUTHORITY unset — RViz often fails GLX with BadValue. Run: source ~/ur3_control/source_ws.sh"
else
  ok "XAUTHORITY is set."
fi

# --- NVIDIA driver mismatch (main cause of GLXContext / BadValue on this machine) ---
_kver=""
_uver=""
if [ -r /proc/driver/nvidia/version ]; then
  _kver=$(sed -n 's/^NVRM version: NVIDIA UNIX x86_64 Kernel Module  \([^ ]*\).*/\1/p' /proc/driver/nvidia/version)
fi
if command -v modinfo >/dev/null 2>&1; then
  _uver=$(modinfo -F version nvidia 2>/dev/null || true)
fi

if [ -n "$_kver" ] && [ -n "$_uver" ] && [ "$_kver" != "$_uver" ]; then
  fail "NVIDIA kernel module ($_kver) != userspace driver ($_uver)."
  note "       This breaks OpenGL/GLX (RViz: Unable to create GLXContext / BadValue)."
  note "       Fix: reboot the PC so the new driver module loads (after apt upgrade)."
  note "       Until reboot, use headless sim: headless:=true enable_rviz:=false"
elif command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    ok "nvidia-smi works."
  elif nvidia-smi 2>&1 | grep -q "Driver/library version mismatch"; then
    fail "nvidia-smi: Driver/library version mismatch — reboot required."
  else
    warn "nvidia-smi failed (see above). GLX windows may not work."
  fi
elif lspci 2>/dev/null | grep -qi nvidia; then
  warn "NVIDIA GPU present but nvidia module/smi not available."
fi

# --- Optional GLX probe ---
if command -v glxinfo >/dev/null 2>&1 && [ -n "${DISPLAY:-}" ]; then
  if glxinfo -B 2>/dev/null | grep -q "OpenGL"; then
    ok "glxinfo reports an OpenGL renderer."
  else
    fail "glxinfo could not get OpenGL on $DISPLAY."
  fi
else
  note "Tip: sudo apt install mesa-utils  # adds glxinfo for deeper GLX checks"
fi

note "=== Summary: $_ok ok, $_warn warn, $_fail fail ==="
if [ "$_fail" -gt 0 ]; then
  note ""
  note "GUI launch blocked until GPU/X11 issues are fixed."
  note "  Reboot (recommended): sudo reboot"
  note "  Then: source ~/ur3_control/source_ws.sh && ~/ur3_control/launch_sim_gui.sh"
  note "  Headless (physics only): ros2 launch ur3_mujoco_sim ur3_mujoco_moveit.launch.py headless:=true enable_rviz:=false"
  exit 1
fi
exit 0
