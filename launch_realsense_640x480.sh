#!/usr/bin/env bash
# Launch Intel RealSense D435i at 640x480 RGB + aligned depth (default 15 FPS).
# Override: UR3_REALSENSE_PROFILE=640x480x30 bash ~/ur3_control/launch_realsense_640x480.sh

set +e

REALSENSE_PROFILE="${UR3_REALSENSE_PROFILE:-640x480x15}"

unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset ROS_PACKAGE_PATH

source /opt/ros/humble/setup.bash

echo "Launching RealSense D435i: color=${REALSENSE_PROFILE}, depth=${REALSENSE_PROFILE}, aligned depth enabled."

ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  rgb_camera.color_profile:=${REALSENSE_PROFILE} \
  depth_module.depth_profile:=${REALSENSE_PROFILE} \
  align_depth.enable:=true \
  enable_sync:=true \
  pointcloud.enable:=false

echo ""
echo "RealSense launch exited. Shell stays open."
exec "${SHELL:-/bin/bash}" -i
