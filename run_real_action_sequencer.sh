#!/usr/bin/env bash
# Run the action sequencer against a real UR3 ROS stack.
#
# Real mode keeps MoveIt/RViz planning-scene attach/detach for obstacle avoidance,
# but skips MuJoCo-only weld/detach Trigger services.

export UR3_ROBOT_BACKEND=real
export UR3_USE_SIM_TIME=false

exec "${HOME}/ur3_control/run_action_sequencer.sh" "$@"
