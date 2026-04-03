#!/usr/bin/env python3

import random

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_srvs.srv import Trigger


class MujocoObjectManagerNode(Node):
    """
    Sim-side object pose manager.

    This node provides a stable ROS API for episode resets and future
    "initialize-and-let-go" experiments. It currently stores/publishes a target pose
    command but does not directly mutate MuJoCo state (left as backend integration TODO).
    """

    def __init__(self) -> None:
        super().__init__("mujoco_object_manager_node")
        self.declare_parameter("target_pose_topic", "/target_pose")
        self.declare_parameter("object_frame", "target_object")
        self.declare_parameter("reset_pose_bounds_xyz", [0.20, 0.45, -0.20, 0.20, 0.05, 0.35])

        topic = str(self.get_parameter("target_pose_topic").value)
        self._sub = self.create_subscription(PoseStamped, topic, self._on_target_pose, 10)
        self._pub = self.create_publisher(PoseStamped, topic, 10)
        self._srv = self.create_service(Trigger, "/episode_reset", self._on_episode_reset)
        self._last_pose = PoseStamped()
        self.get_logger().info(f"Mujoco object manager ready. Listening on {topic}")

    def _on_target_pose(self, msg: PoseStamped) -> None:
        self._last_pose = msg
        # TODO(sim-backend): write msg pose into MuJoCo freejoint qpos and call mj_forward.

    def _on_episode_reset(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        b = list(self.get_parameter("reset_pose_bounds_xyz").value)
        if len(b) != 6:
            resp.success = False
            resp.message = "reset_pose_bounds_xyz must be [xmin,xmax,ymin,ymax,zmin,zmax]"
            return resp

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = random.uniform(float(b[0]), float(b[1]))
        pose.pose.position.y = random.uniform(float(b[2]), float(b[3]))
        pose.pose.position.z = random.uniform(float(b[4]), float(b[5]))
        pose.pose.orientation.w = 1.0
        self._pub.publish(pose)
        self._last_pose = pose

        # TODO(sim-backend): when MuJoCo API hook is added, directly set freejoint qpos here.
        resp.success = True
        resp.message = "Published randomized target pose for episode reset"
        return resp


def main() -> None:
    rclpy.init()
    node = MujocoObjectManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

