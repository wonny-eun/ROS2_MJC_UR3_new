# Source this file after building, before ros2 launch AND before ros2 run (same machine):
#   source ~/ur3_control/source_ws.sh
#
# IMPORTANT: Every terminal that should see the same topics/services must share ROS_DOMAIN_ID.
# The common `humble` alias sets ROS_DOMAIN_ID=30 via `ros_domain`; this file defaults to 30 so
# launches that only `source source_ws.sh` still match `ros2 run` sessions that used `humble` first.
if [ -z "${ROS_DOMAIN_ID+x}" ]; then
  export ROS_DOMAIN_ID=30
elif [ -z "$ROS_DOMAIN_ID" ]; then
  export ROS_DOMAIN_ID=30
fi
#
# Build (run these as separate lines or a script — do not paste without newlines):
#   export MUJOCO_DIR=/opt/ros/humble/opt/mujoco_vendor
#   source /opt/ros/humble/setup.bash
#   cd ~/ur3_control/src/ROS2_MuJoCo_UR3
#   colcon build --symlink-install --allow-overriding ur_controllers ur_dashboard_msgs ur_description
#
# MuJoCo + RViz need DISPLAY and XAUTHORITY (Cursor terminals often omit XAUTHORITY).
set +u 2>/dev/null || true
# shellcheck source=/dev/null
source "${HOME}/ur3_control/ur3_display_env.sh"
# Clear stale forced-headless from older setups (otherwise launch stays headless with no windows).
unset UR3_FORCE_MUJOCO_HEADLESS 2>/dev/null || true
#
# If GLFW still fails in this terminal only, use headless (physics OK, no windows):
#   export UR3_FORCE_MUJOCO_HEADLESS=1
#   ros2 launch ur3_mujoco_sim ur3_mujoco_moveit.launch.py headless:=true enable_rviz:=false
# Or run from Ubuntu's desktop terminal: ~/ur3_control/launch_sim_gui.sh
#
# Preflight (NVIDIA mismatch / GLX): ~/ur3_control/ur3_check_gpu.sh
if [ -r /proc/driver/nvidia/version ] && command -v modinfo >/dev/null 2>&1; then
  _ur3_kver=$(sed -n 's/^NVRM version: NVIDIA UNIX x86_64 Kernel Module  \([^ ]*\).*/\1/p' /proc/driver/nvidia/version)
  _ur3_uver=$(modinfo -F version nvidia 2>/dev/null || true)
  if [ -n "$_ur3_kver" ] && [ -n "$_ur3_uver" ] && [ "$_ur3_kver" != "$_ur3_uver" ]; then
    echo "WARNING: NVIDIA driver mismatch (kernel $_ur3_kver vs userspace $_ur3_uver)." >&2
    echo "  RViz/MuJoCo windows will not open until you reboot. Run: ~/ur3_control/ur3_check_gpu.sh" >&2
  fi
fi
#
export MUJOCO_DIR=/opt/ros/humble/opt/mujoco_vendor

set +u 2>/dev/null || true
# shellcheck source=/dev/null
source /opt/ros/humble/setup.bash
# shellcheck source=/dev/null
source "$HOME/ur3_control/src/ROS2_MuJoCo_UR3/install/setup.bash"
