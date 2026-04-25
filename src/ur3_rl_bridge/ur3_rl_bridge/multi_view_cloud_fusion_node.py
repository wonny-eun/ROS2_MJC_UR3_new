#!/usr/bin/env python3

import math
import struct
import threading
import time
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener


def _transform_to_matrix(tf_msg) -> np.ndarray:
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        w = 1.0
        x = y = z = 0.0
    else:
        x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    r = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r
    out[0, 3] = float(t.x)
    out[1, 3] = float(t.y)
    out[2, 3] = float(t.z)
    return out


def _cloud_xyz_rgb(points_xyz: np.ndarray, rgb: Tuple[int, int, int], frame_id: str, stamp) -> PointCloud2:
    cloud = PointCloud2()
    cloud.header.frame_id = frame_id
    cloud.header.stamp = stamp
    cloud.height = 1
    cloud.width = int(points_xyz.shape[0])
    cloud.is_bigendian = False
    cloud.is_dense = True
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    cloud.point_step = 16
    cloud.row_step = cloud.point_step * cloud.width

    r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
    rgb_u32 = (r << 16) | (g << 8) | b
    buff = bytearray(cloud.row_step)
    for i, p in enumerate(points_xyz):
        struct.pack_into("<fffI", buff, i * 16, float(p[0]), float(p[1]), float(p[2]), rgb_u32)
    cloud.data = bytes(buff)
    return cloud


class MultiViewCloudFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("multi_view_cloud_fusion_node")
        self.declare_parameter("rgb_topic", "/rl_camera/color")
        self.declare_parameter("depth_topic", "/rl_camera/depth")
        self.declare_parameter("camera_info_topic", "/rl_camera/camera_info")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("ee_frame", "tool0")
        self.declare_parameter("sample_stride_px", 4)
        self.declare_parameter("depth_min_m", 0.10)
        self.declare_parameter("depth_max_m", 2.50)
        self.declare_parameter("target_origin_topic", "/multi_view_cloud_fusion/target_origin")
        self.declare_parameter("object_crop_radius_m", 0.10)
        self.declare_parameter("object_crop_z_margin_m", 0.12)
        self.declare_parameter(
            "handeye_t_cam_ee",
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        )

        self._rgb: Optional[np.ndarray] = None
        self._depth: Optional[np.ndarray] = None
        self._info: Optional[CameraInfo] = None
        self._target_origin_msg: Optional[PoseStamped] = None
        self._lock = threading.Lock()

        self.create_subscription(Image, str(self.get_parameter("rgb_topic").value), self._on_rgb, 1)
        self.create_subscription(Image, str(self.get_parameter("depth_topic").value), self._on_depth, 1)
        self.create_subscription(
            CameraInfo, str(self.get_parameter("camera_info_topic").value), self._on_info, 1
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("target_origin_topic").value),
            self._on_target_origin,
            10,
        )

        self._pub_pose1 = self.create_publisher(PointCloud2, "/fused_cloud_pose1", 1)
        self._pub_pose2 = self.create_publisher(PointCloud2, "/fused_cloud_pose2", 1)
        self._pub_pose3 = self.create_publisher(PointCloud2, "/fused_cloud_pose3", 1)
        self._capture_srv = self.create_service(
            Trigger, "/multi_view_cloud_fusion/capture", self._on_capture
        )
        self._reset_srv = self.create_service(Trigger, "/multi_view_cloud_fusion/reset", self._on_reset)
        self._capture_index = 0

        self._tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=30.0))
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=True)

    def _on_rgb(self, msg: Image) -> None:
        if msg.encoding != "rgb8":
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        with self._lock:
            self._rgb = arr.copy()

    def _on_depth(self, msg: Image) -> None:
        if msg.encoding != "32FC1":
            return
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
        with self._lock:
            self._depth = arr.copy()

    def _on_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._info = msg

    def _on_target_origin(self, msg: PoseStamped) -> None:
        with self._lock:
            self._target_origin_msg = msg

    def _wait_for_images(self, timeout_sec: float = 20.0) -> bool:
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < timeout_sec:
            with self._lock:
                ok = self._rgb is not None and self._depth is not None and self._info is not None
            if ok:
                return True
            time.sleep(0.05)
        return False

    def _capture_cloud_in_base(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._depth is None or self._info is None:
                return None
            depth = self._depth.copy()
            info = self._info

        stride = int(self.get_parameter("sample_stride_px").value)
        zmin = float(self.get_parameter("depth_min_m").value)
        zmax = float(self.get_parameter("depth_max_m").value)
        crop_r = float(self.get_parameter("object_crop_radius_m").value)
        crop_z = float(self.get_parameter("object_crop_z_margin_m").value)
        base_frame = str(self.get_parameter("base_frame").value)
        ee_frame = str(self.get_parameter("ee_frame").value)
        with self._lock:
            target_msg = self._target_origin_msg

        # T_cam_ee: camera -> end-effector
        t_cam_ee_raw = list(self.get_parameter("handeye_t_cam_ee").value)
        if len(t_cam_ee_raw) != 16:
            self.get_logger().error("handeye_t_cam_ee must have 16 values.")
            return None
        t_cam_ee = np.array([float(v) for v in t_cam_ee_raw], dtype=np.float64).reshape((4, 4))

        try:
            tf_be = self._tf_buffer.lookup_transform(
                base_frame,
                ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            )
        except Exception as e:
            self.get_logger().error(f"TF lookup {base_frame}<-{ee_frame} failed: {e}")
            return None
        t_base_ee = _transform_to_matrix(tf_be)
        t_base_cam = t_base_ee @ t_cam_ee

        target_base = None
        if target_msg is not None:
            try:
                tf_bt = self._tf_buffer.lookup_transform(
                    base_frame,
                    target_msg.header.frame_id,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=1.0),
                )
                t_base_target = _transform_to_matrix(tf_bt)
                p_target = np.array(
                    [
                        float(target_msg.pose.position.x),
                        float(target_msg.pose.position.y),
                        float(target_msg.pose.position.z),
                        1.0,
                    ],
                    dtype=np.float64,
                )
                p_base = t_base_target @ p_target
                target_base = p_base[:3]
            except Exception as e:
                self.get_logger().warn(f"Failed to transform target origin into {base_frame}: {e}")

        k = info.k
        fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
        h, w = depth.shape

        pts = []
        for v in range(0, h, max(1, stride)):
            for u in range(0, w, max(1, stride)):
                z = float(depth[v, u])
                if not math.isfinite(z) or z < zmin or z > zmax:
                    continue
                x = (float(u) - cx) * z / fx
                y = (float(v) - cy) * z / fy
                p_cam = np.array([x, y, z, 1.0], dtype=np.float64)
                p_base = t_base_cam @ p_cam
                if target_base is not None:
                    dx = float(p_base[0] - target_base[0])
                    dy = float(p_base[1] - target_base[1])
                    dz = float(p_base[2] - target_base[2])
                    if (dx * dx + dy * dy) > crop_r * crop_r:
                        continue
                    if abs(dz) > crop_z:
                        continue
                pts.append([p_base[0], p_base[1], p_base[2]])
        if not pts:
            return None
        return np.asarray(pts, dtype=np.float32)

    def _on_capture(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        if not self._wait_for_images():
            resp.success = False
            resp.message = "Timed out waiting for RGB-D topics."
            return resp

        pubs = [self._pub_pose1, self._pub_pose2, self._pub_pose3]
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        frame = str(self.get_parameter("base_frame").value)
        idx = self._capture_index % 3

        pts_base = self._capture_cloud_in_base()
        if pts_base is None:
            resp.success = False
            resp.message = f"Failed to capture cloud at slot {idx+1}."
            return resp
        msg = _cloud_xyz_rgb(pts_base, colors[idx], frame, self.get_clock().now().to_msg())
        pubs[idx].publish(msg)
        self._capture_index += 1
        resp.success = True
        resp.message = f"Published fused cloud slot {idx+1} with {pts_base.shape[0]} points."
        return resp

    def _on_reset(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        self._capture_index = 0
        resp.success = True
        resp.message = "Fusion capture index reset to slot 1."
        return resp


def main() -> None:
    rclpy.init()
    node = MultiViewCloudFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

