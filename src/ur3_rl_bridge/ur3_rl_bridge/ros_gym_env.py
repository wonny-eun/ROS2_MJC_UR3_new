#!/usr/bin/env python3

from typing import Optional

import numpy as np
import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

try:
    import gymnasium as gym
except ImportError as exc:  # pragma: no cover
    raise ImportError("Please install gymnasium to use UR3RosGymEnv") from exc


class UR3RosGymEnv(gym.Env):
    """
    Gymnasium wrapper for ROS2-based UR3 simulation.

    Action path: RL action -> FollowJointTrajectory -> ros2_control JTC -> MuJoCo.
    """

    metadata = {"render_modes": []}

    def __init__(self, node_name: str = "ur3_ros_gym_env") -> None:
        super().__init__()
        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = Node(node_name)

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6 + 6 + 7,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(6,), dtype=np.float32)

        self._last_joint_state: Optional[JointState] = None
        self.node.create_subscription(JointState, "/joint_states", self._on_joint_state, 10)

        self._reset_cli = self.node.create_client(Trigger, "/episode_reset")
        self._traj_ac = ActionClient(
            self.node, FollowJointTrajectory, "/joint_trajectory_controller/follow_joint_trajectory"
        )

    def _on_joint_state(self, msg: JointState) -> None:
        self._last_joint_state = msg

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._reset_cli.wait_for_service(timeout_sec=1.0):
            self._reset_cli.call_async(Trigger.Request())

        while rclpy.ok() and self._last_joint_state is None:
            rclpy.spin_once(self.node, timeout_sec=0.1)

        return self._get_obs(), {}

    def step(self, action):
        q = self._get_q()
        q_cmd = q + np.asarray(action, dtype=np.float64)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory()
        goal.trajectory.joint_names = self.joint_names

        pt = JointTrajectoryPoint()
        pt.positions = q_cmd.tolist()
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = int(0.25 * 1e9)
        goal.trajectory.points = [pt]

        if not self._traj_ac.wait_for_server(timeout_sec=1.5):
            raise RuntimeError("FollowJointTrajectory action server is unavailable")
        self._traj_ac.send_goal_async(goal)

        for _ in range(5):
            rclpy.spin_once(self.node, timeout_sec=0.05)

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def _get_q(self) -> np.ndarray:
        msg = self._last_joint_state
        if msg is None:
            return np.zeros(6, dtype=np.float64)
        name_to_idx = {name: i for i, name in enumerate(msg.name)}
        return np.array([msg.position[name_to_idx[j]] for j in self.joint_names], dtype=np.float64)

    def _get_obs(self) -> np.ndarray:
        q = self._get_q()
        dq = np.zeros(6, dtype=np.float64)
        obj_pose = np.array([0.30, 0.00, 0.25, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        return np.concatenate([q, dq, obj_pose]).astype(np.float32)

    def _compute_reward(self, _obs: np.ndarray) -> float:
        # Replace with task reward (distance, contact, grasp success, etc.)
        return 0.0

    def close(self):
        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

