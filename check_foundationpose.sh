#!/usr/bin/env bash
# Quick health check while sim + bridge + Isaac are running.
source "${HOME}/ur3_control/source_ws.sh"
echo "=== Topics (camera ~30 Hz target, ~10 Hz detections/mask, >0 Hz FP output) ==="
timeout 4 ros2 topic hz /rl_camera/noisy/color 2>&1 | tail -3 || true
timeout 4 ros2 topic hz /foundation_pose/yolo_detection2_d_array 2>&1 | tail -3 || true
timeout 4 ros2 topic hz /segmentation 2>&1 | tail -3 || true
timeout 4 ros2 topic hz /fp_bridge/depth_mono16 2>&1 | tail -3 || true
timeout 4 ros2 topic hz /rgb/image_rect_color 2>&1 | tail -3 || true
timeout 4 ros2 topic hz /depth_registered/image_rect 2>&1 | tail -3 || true
timeout 4 ros2 topic hz /output 2>&1 | tail -3 || true
echo ""
echo "=== TF (use sim time) ==="
timeout 5 ros2 run tf2_ros tf2_echo rl_camera_frame fp_object --ros-args -p use_sim_time:=true 2>&1 | tail -5 || true
timeout 5 ros2 run tf2_ros tf2_echo base_link fp_object --ros-args -p use_sim_time:=true 2>&1 | tail -5 || true
timeout 5 ros2 run tf2_ros tf2_echo base_link mujoco_square_1 --ros-args -p use_sim_time:=true 2>&1 | tail -5 || true
