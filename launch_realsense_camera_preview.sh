#!/usr/bin/env bash
# OpenCV RGB + depth only (no YOLO). For full stack use launch_realsense_vision_preview.sh.

set +e
UR3_YOLO_PREVIEW=false exec bash "${HOME}/ur3_control/launch_realsense_vision_preview.sh"
