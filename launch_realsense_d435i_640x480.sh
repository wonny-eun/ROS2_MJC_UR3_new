#!/usr/bin/env bash
# Launch Intel RealSense D435i RGB + aligned depth (default 640x480x15).
# Override: UR3_REALSENSE_PROFILE=640x480x30 bash ~/ur3_control/launch_realsense_d435i_640x480.sh
# Keep this terminal running while using camera topics.

set +e

REALSENSE_PROFILE="${UR3_REALSENSE_PROFILE:-640x480x15}"

unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset ROS_PACKAGE_PATH

source /opt/ros/humble/setup.bash

echo "Launching RealSense D435i at ${REALSENSE_PROFILE} RGB + aligned depth..."
echo "  publish_tf:=false — do NOT use RealSense camera TF"
echo "  optical TF: bash ~/ur3_control/launch_handeye_d435i_gripper.sh"

ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  enable_infra1:=false \
  enable_infra2:=false \
  enable_gyro:=false \
  enable_accel:=false \
  rgb_camera.color_profile:=${REALSENSE_PROFILE} \
  depth_module.depth_profile:=${REALSENSE_PROFILE} \
  align_depth.enable:=true \
  enable_sync:=true \
  pointcloud.enable:=false \
  publish_tf:=false \
  initial_reset:=true

echo ""
echo "RealSense launch exited. Shell stays open."
exec "${SHELL:-/bin/bash}" -i
