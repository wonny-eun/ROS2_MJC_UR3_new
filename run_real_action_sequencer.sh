#!/usr/bin/env bash
# Run the action sequencer against a real UR3 ROS stack.
#
# MoveIt collision checking stays enabled (avoid_collisions=true).
# Skips MuJoCo-only weld/detach Trigger services.
# Uses ur3_action_sequence_real.yaml (not the sim config).
#
# Usage:
#   export ROS_DOMAIN_ID=30
#   bash ~/ur3_control/run_real_action_sequencer.sh UR3_Connect_Test 90 0.3
#     # FoundationPose grasp: absolute TCP roll 90°; arm move speed scale 0.3
#   bash ~/ur3_control/run_real_action_sequencer.sh Cylinder_1 0.3   # speed only (unchanged)

export UR3_ROBOT_BACKEND=real
export UR3_USE_SIM_TIME=false
export UR3_ACTION_CONFIG="${UR3_ACTION_CONFIG:-${HOME}/ur3_control/src/ROS2_MuJoCo_UR3/src/ur3_pick_task/config/actions/ur3_action_sequence_real.yaml}"

exec "${HOME}/ur3_control/run_action_sequencer.sh" "$@"
