#!/usr/bin/env bash
# Launch the real UR3 driver from /opt/ros only.
# This avoids mixing /opt/ros ur_robot_driver with older workspace UR message libraries.

set +e

ROBOT_IP="${UR3_ROBOT_IP:-172.168.2.101}"
CALIBRATION_FILE="${UR3_CALIBRATION_FILE:-/home/wonny/my_ur3_calibration.yaml}"
UR_TYPE="${UR3_TYPE:-ur3}"
ARM_ONLY_RECIPE="${UR3_ARM_ONLY_RECIPE:-true}"
ARM_ONLY_RECIPE_FILE="${HOME}/ur3_control/rtde_input_recipe_arm_only.txt"
ARM_ONLY_LAUNCH_FILE="/tmp/ur_control_arm_only_${USER}.launch.py"
REAL_SCENE_XACRO="${UR3_REAL_SCENE_XACRO:-/home/wonny/ur3_control/real_urdf/ur3_real_scene.urdf.xacro}"

echo "Launching real UR driver with clean ROS environment:"
echo "  ur_type: ${UR_TYPE}"
echo "  robot_ip: ${ROBOT_IP}"
echo "  calibration: ${CALIBRATION_FILE}"
echo "  arm_only_recipe: ${ARM_ONLY_RECIPE}"
echo "  real_scene_xacro: ${REAL_SCENE_XACRO}"
echo "  camera mount/collision: URDF (gripper -> d435i_mount -> rl_camera_frame)"
echo "  camera optical TF: hand-eye YAML — after driver is up, run:"
echo "    bash ~/ur3_control/launch_handeye_d435i_gripper.sh"
echo ""

# Stop only known node executable names. Avoid broad pkill -f patterns.
for name in \
  ur_ros2_control_node \
  dashboard_client \
  robot_state_helper \
  controller_stopper_node \
  urscript_interface \
  trajectory_until_node \
  spawner
do
  pkill -TERM -x "${name}" 2>/dev/null || true
done
sleep 1

# Remove workspace overlay variables that can make /opt/ros driver load old local libraries.
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset ROS_PACKAGE_PATH

source /opt/ros/humble/setup.bash

LAUNCH_TARGET="ur_robot_driver ur_control.launch.py"
if [ "${ARM_ONLY_RECIPE}" = "true" ]; then
  python3 - "${ARM_ONLY_LAUNCH_FILE}" "${REAL_SCENE_XACRO}" <<'PY'
import sys
from pathlib import Path

src = Path("/opt/ros/humble/share/ur_robot_driver/launch/ur_control.launch.py")
dst = Path(sys.argv[1])
real_scene_xacro = sys.argv[2]
text = src.read_text()
old_desc = '''            PathJoinSubstitution([FindPackageShare(description_package), "urdf", description_file]),'''
new_desc = f'''            "{real_scene_xacro}",'''
if old_desc not in text:
    raise SystemExit("Could not patch ur_control.launch.py description xacro path")
text = text.replace(old_desc, new_desc)
old = '''    input_recipe_filename = PathJoinSubstitution(
        [FindPackageShare("ur_robot_driver"), "resources", "rtde_input_recipe.txt"]
    )
'''
new = '''    # Arm-only recipe: do not claim digital/tool/analog outputs over RTDE.
    # This avoids fieldbus/PLC conflicts while testing UR arm motion only.
    input_recipe_filename = "/home/wonny/ur3_control/rtde_input_recipe_arm_only.txt"
'''
if old not in text:
    raise SystemExit("Could not patch ur_control.launch.py input recipe block")
dst.write_text(text.replace(old, new))
PY
  LAUNCH_TARGET="${ARM_ONLY_LAUNCH_FILE}"
fi

ros2 launch ${LAUNCH_TARGET} \
  ur_type:="${UR_TYPE}" \
  robot_ip:="${ROBOT_IP}" \
  kinematics_params_file:="${CALIBRATION_FILE}" \
  launch_rviz:=false

echo ""
echo "UR driver launch exited. Shell stays open."
exec "${SHELL:-/bin/bash}" -i
