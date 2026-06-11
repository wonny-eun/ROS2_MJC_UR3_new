#!/usr/bin/env bash
# Start an interactive ROS 2 Humble shell without any workspace overlay.
# Use this for the real UR driver so old local UR packages do not shadow /opt/ros packages.

set +e

unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset LD_LIBRARY_PATH
unset PYTHONPATH
unset ROS_PACKAGE_PATH

source /opt/ros/humble/setup.bash

echo "ROS sourced cleanly:"
echo "  ROS_DISTRO=${ROS_DISTRO}"
echo "  AMENT_PREFIX_PATH=${AMENT_PREFIX_PATH}"
echo ""
echo "Now run real UR commands in this shell."

exec "${SHELL:-/bin/bash}" -i
