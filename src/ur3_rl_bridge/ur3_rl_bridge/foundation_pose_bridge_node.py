#!/usr/bin/env python3
"""Bridge MuJoCo rl_camera + YOLO to Isaac FoundationPose and RViz.

Publishes on trigger (``~/trigger``):
  - ``detection_topic`` — ``vision_msgs/Detection2DArray`` (debug / legacy).
  - ``segmentation_topic`` — ``sensor_msgs/Image`` mono8 mask for FoundationPose (640×480).
  - ``object_cloud_topic`` — cropped depth → ``sensor_msgs/PointCloud2`` in the camera frame.

When ``use_yolo_segmentation_mask`` is true (default), YOLO-seg instance masks are published
directly to ``/segmentation`` and Isaac ``Detection2DToMask`` should be disabled in launch.
"""

from __future__ import annotations

import copy
import math
import os
import struct
import threading
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField
from std_srvs.srv import Trigger
from vision_msgs.msg import (
    BoundingBox2D,
    Detection2D,
    Detection2DArray,
    ObjectHypothesisWithPose,
    Point2D,
    Pose2D,
)

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None  # type: ignore[misc, assignment]

BBox_xyxy_cf = Tuple[float, float, float, float, str, float]


@dataclass(frozen=True)
class _YoloPick:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    score: float
    det_index: int
    mask_full: Optional[np.ndarray]  # uint8 H×W, 255=object, same size as RGB


def _image_rgb_and_header(msg: Image) -> Tuple[np.ndarray, Any, str]:
    if msg.encoding == "rgb8":
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        return rgb, msg.header.stamp, msg.header.frame_id
    if msg.encoding == "bgr8":
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb, msg.header.stamp, msg.header.frame_id
    raise ValueError(f"Unsupported RGB encoding '{msg.encoding}'")


def _depth_image_to_meters(msg: Image) -> np.ndarray:
    """Decode depth Image to float32 meters (32FC1 or RealSense 16UC1 mm)."""
    if msg.encoding == "32FC1":
        return np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
    if msg.encoding in ("16UC1", "mono16"):
        raw = np.frombuffer(msg.data, dtype=np.uint16).reshape((msg.height, msg.width))
        return raw.astype(np.float32) * 0.001
    raise ValueError(f"Unsupported depth encoding '{msg.encoding}' (expected 32FC1 or 16UC1)")


def _bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = (float(a[0]), float(a[1]), float(a[2]), float(a[3]))
    bx1, by1, bx2, by2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 1e-12 else 0.0


def _class_name_matches(
    cname: str,
    target_class: str,
    *,
    case_insensitive: bool,
) -> bool:
    if case_insensitive:
        return cname.strip().lower() == target_class.strip().lower()
    return cname.strip() == target_class.strip()


def _yolo_pick_detection(
    model: Any,
    rgb: np.ndarray,
    *,
    target_class: str,
    min_conf: float,
    yolo_iou: float,
    predict_conf_floor: float,
    exclusive_scene_classes: Optional[Sequence[str]],
    class_match_case_insensitive: bool,
    want_mask: bool,
    latch_xyxy: Optional[Sequence[float]] = None,
) -> Optional[_YoloPick]:
    infer_conf = float(min(max(1e-6, predict_conf_floor), min_conf))
    results = model.predict(
        rgb,
        conf=infer_conf,
        iou=yolo_iou,
        retina_masks=want_mask,
        verbose=False,
    )[0]
    names = results.names
    boxes = results.boxes
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return None

    target_cmp = target_class.strip()
    if class_match_case_insensitive:
        target_cmp = target_cmp.lower()

    exclusive_list: Optional[list[str]] = None
    if isinstance(exclusive_scene_classes, (list, tuple)):
        tmp = [str(x).strip() for x in exclusive_scene_classes if str(x).strip()]
        if tmp:
            exclusive_list = tmp
    if class_match_case_insensitive:
        exclusive_norm = {x.lower() for x in exclusive_list} if exclusive_list else set()
    else:
        exclusive_norm = set(exclusive_list) if exclusive_list else set()

    candidates: list[tuple[float, int, str]] = []
    for i in range(len(boxes)):
        cid = int(boxes.cls[i].item())
        cf = float(boxes.conf[i].item())
        cname = str(names[cid]).strip()
        key = cname.lower() if class_match_case_insensitive else cname
        if exclusive_norm and key not in exclusive_norm:
            continue
        if not _class_name_matches(cname, target_class, case_insensitive=class_match_case_insensitive):
            continue
        if cf < min_conf:
            continue
        candidates.append((cf, i, cname))

    if not candidates:
        return None

    if latch_xyxy is not None and len(latch_xyxy) >= 4:
        latch = [float(latch_xyxy[0]), float(latch_xyxy[1]), float(latch_xyxy[2]), float(latch_xyxy[3])]
        best_iou = -1.0
        best: Optional[tuple[float, int, str]] = None
        for cf, i, cname in candidates:
            xyxy = boxes.xyxy[i].cpu().numpy()
            iou = _bbox_iou(latch, [float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])])
            if iou > best_iou:
                best_iou = iou
                best = (cf, i, cname)
        if best is None or best_iou < 0.05:
            return None
        cf_w, i_w, cname_w = best
    else:
        cf_w, i_w, cname_w = max(candidates, key=lambda x: x[0])

    xyxy = boxes.xyxy[i_w].cpu().numpy()
    x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

    mask_full: Optional[np.ndarray] = None
    if want_mask and results.masks is not None and len(results.masks) > i_w:
        mh, mw = rgb.shape[:2]
        mask_full = np.zeros((mh, mw), dtype=np.uint8)
        mdata = results.masks.data[i_w].cpu().numpy()
        if mdata.shape[0] != mh or mdata.shape[1] != mw:
            mdata = cv2.resize(mdata.astype(np.float32), (mw, mh), interpolation=cv2.INTER_LINEAR)
        mask_full[mdata > 0.5] = 255

    return _YoloPick(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        label=cname_w,
        score=float(cf_w),
        det_index=int(i_w),
        mask_full=mask_full,
    )


def _yolo_pick_bbox_xyxy(
    model: Any,
    rgb: np.ndarray,
    *,
    target_class: str,
    min_conf: float,
    yolo_iou: float,
    predict_conf_floor: float,
    exclusive_scene_classes: Optional[Sequence[str]],
    class_match_case_insensitive: bool,
) -> Optional[BBox_xyxy_cf]:
    pick = _yolo_pick_detection(
        model,
        rgb,
        target_class=target_class,
        min_conf=min_conf,
        yolo_iou=yolo_iou,
        predict_conf_floor=predict_conf_floor,
        exclusive_scene_classes=exclusive_scene_classes,
        class_match_case_insensitive=class_match_case_insensitive,
        want_mask=False,
    )
    if pick is None:
        return None
    return (pick.x1, pick.y1, pick.x2, pick.y2, pick.label, pick.score)


def _bbox_binary_mask(h: int, w: int, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w - 1, x2), min(h - 1, y2)
    if x2c > x1c and y2c > y1c:
        mask[y1c : y2c + 1, x1c : x2c + 1] = 255
    return mask


def _resize_mask_nearest(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[1] == width and mask.shape[0] == height:
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _mask_to_image_msg(mask_u8: np.ndarray, stamp: Any, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(mask_u8.shape[0])
    msg.width = int(mask_u8.shape[1])
    msg.encoding = "mono8"
    msg.is_bigendian = False
    msg.step = int(mask_u8.shape[1])
    msg.data = np.ascontiguousarray(mask_u8).tobytes()
    return msg


def _bbox_to_pointcloud(
    depth_m: np.ndarray,
    info: CameraInfo,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    stride_px: int,
    z_min: float,
    z_max: float,
    stamp: Any,
    frame_id: str,
    rgb_uint8: Optional[np.ndarray] = None,
) -> PointCloud2:
    """Back-project cropped depth strip to xyz in camera frame (pinhole, z forward)."""
    h, w = depth_m.shape[:2]
    x1c, y1c = max(0, x1), max(0, y1)
    x2c, y2c = min(w - 1, x2), min(h - 1, y2)
    if x2c <= x1c or y2c <= y1c:
        cloud = PointCloud2()
        cloud.header.stamp = stamp
        cloud.header.frame_id = frame_id
        cloud.height = 1
        cloud.width = 0
        cloud.is_dense = True
        cloud.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        cloud.point_step = 12
        cloud.row_step = 0
        cloud.data = b""
        return cloud

    fx = float(info.k[0])
    fy = float(info.k[4])
    cx = float(info.k[2])
    cy = float(info.k[5])
    st = max(1, int(stride_px))

    pts: list[Tuple[float, float, float]] = []
    for v in range(y1c, y2c + 1, st):
        row = depth_m[v]
        for u in range(x1c, x2c + 1, st):
            zm = float(row[u])
            if not math.isfinite(zm) or zm <= z_min or zm >= z_max:
                continue
            x = (float(u) - cx) * zm / fx
            y = (float(v) - cy) * zm / fy
            pts.append((x, y, zm))

    cloud = PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = frame_id
    cloud.height = 1
    cloud.width = int(len(pts))
    cloud.is_bigendian = False
    cloud.is_dense = True
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    if rgb_uint8 is not None:
        cloud.fields.append(PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1))
        cloud.point_step = 16
    else:
        cloud.point_step = 12

    cloud.row_step = cloud.point_step * cloud.width
    buff = bytearray(cloud.row_step)
    rgba_u32_fn = rgb_uint8 is not None
    for i, (xx, yy, zz) in enumerate(pts):
        off = i * cloud.point_step
        if rgba_u32_fn:
            uc = max(x1c, min(x2c, int(round((xx * fx / zz) + cx))))
            vc = max(y1c, min(y2c, int(round((yy * fy / zz) + cy))))
            r, g, b = rgb_uint8[vc, uc].tolist()
            rgb_packed = (int(r) << 16) | (int(g) << 8) | int(b)
            struct.pack_into("<fffI", buff, off, float(xx), float(yy), float(zz), rgb_packed)
        else:
            struct.pack_into("<fff", buff, off, float(xx), float(yy), float(zz))
    cloud.data = bytes(buff)
    return cloud


def _mask_to_pointcloud(
    depth_m: np.ndarray,
    info: CameraInfo,
    mask_u8: np.ndarray,
    *,
    stride_px: int,
    z_min: float,
    z_max: float,
    stamp: Any,
    frame_id: str,
    rgb_uint8: Optional[np.ndarray] = None,
) -> PointCloud2:
    """Back-project depth where ``mask_u8`` is non-zero."""
    h, w = depth_m.shape[:2]
    if mask_u8.shape[0] != h or mask_u8.shape[1] != w:
        mask_u8 = cv2.resize(mask_u8, (w, h), interpolation=cv2.INTER_NEAREST)

    fx = float(info.k[0])
    fy = float(info.k[4])
    cx = float(info.k[2])
    cy = float(info.k[5])
    st = max(1, int(stride_px))

    pts: list[Tuple[float, float, float]] = []
    for v in range(0, h, st):
        row = depth_m[v]
        mrow = mask_u8[v]
        for u in range(0, w, st):
            if mrow[u] == 0:
                continue
            zm = float(row[u])
            if not math.isfinite(zm) or zm <= z_min or zm >= z_max:
                continue
            x = (float(u) - cx) * zm / fx
            y = (float(v) - cy) * zm / fy
            pts.append((x, y, zm))

    cloud = PointCloud2()
    cloud.header.stamp = stamp
    cloud.header.frame_id = frame_id
    cloud.height = 1
    cloud.width = int(len(pts))
    cloud.is_bigendian = False
    cloud.is_dense = True
    cloud.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    if rgb_uint8 is not None:
        cloud.fields.append(PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1))
        cloud.point_step = 16
    else:
        cloud.point_step = 12

    cloud.row_step = cloud.point_step * cloud.width
    buff = bytearray(cloud.row_step)
    rgba_u32_fn = rgb_uint8 is not None
    for i, (xx, yy, zz) in enumerate(pts):
        off = i * cloud.point_step
        if rgba_u32_fn:
            uc = max(0, min(w - 1, int(round((xx * fx / zz) + cx))))
            vc = max(0, min(h - 1, int(round((yy * fy / zz) + cy))))
            r, g, b = rgb_uint8[vc, uc].tolist()
            rgb_packed = (int(r) << 16) | (int(g) << 8) | int(b)
            struct.pack_into("<fffI", buff, off, float(xx), float(yy), float(zz), rgb_packed)
        else:
            struct.pack_into("<fff", buff, off, float(xx), float(yy), float(zz))
    cloud.data = bytes(buff)
    return cloud


def _detection_msg(
    *,
    stamp: Any,
    frame_id: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_id: str,
    score: float,
) -> Detection2D:
    det = Detection2D()
    det.header.stamp = stamp
    det.header.frame_id = frame_id
    hyp = ObjectHypothesisWithPose()
    hyp.hypothesis.class_id = str(class_id)
    hyp.hypothesis.score = float(score)
    hyp.pose.pose.orientation.w = 1.0
    det.results.append(hyp)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bb = BoundingBox2D()
    # vision_msgs/Pose2D (not geometry_msgs): position + theta
    bb.center = Pose2D(position=Point2D(x=float(cx), y=float(cy)), theta=0.0)
    bb.size_x = float(max(1.0, x2 - x1))
    bb.size_y = float(max(1.0, y2 - y1))
    det.bbox = bb
    return det


class FoundationPoseBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("foundation_pose_bridge")

        self.declare_parameter("rgb_topic", "/rl_camera/color")
        self.declare_parameter("depth_topic", "/rl_camera/depth")
        self.declare_parameter("camera_info_topic", "/rl_camera/camera_info")
        self.declare_parameter("camera_frame_fallback", "camera_color_optical_frame")
        self.declare_parameter("yolo_model_path", "")
        self.declare_parameter("min_confidence", 0.4)
        self.declare_parameter("yolo_iou", 0.5)
        self.declare_parameter("target_class", "Box_1")
        self.declare_parameter("yolo_exclusive_scene_classes", ["Box_1", "Cylinder_1", "Cylinder_2"])
        self.declare_parameter("class_match_case_insensitive", True)
        self.declare_parameter("predict_conf_floor", 0.01)
        self.declare_parameter("detection_topic", "/foundation_pose/yolo_detection2_d_array")
        self.declare_parameter("object_cloud_topic", "/foundation_pose/object_cloud")
        self.declare_parameter("cloud_stride_px", 2)
        self.declare_parameter("cloud_depth_min_m", 0.05)
        self.declare_parameter("cloud_depth_max_m", 3.0)
        self.declare_parameter("include_rgb_in_cloud", True)
        self.declare_parameter("use_latched_bbox_on_trigger", False)
        self.declare_parameter("latched_bbox_xyxy", [0.0, 0.0, 0.0, 0.0])
        self.declare_parameter("latched_bbox_label", "")
        self.declare_parameter("use_yolo_segmentation_mask", True)
        self.declare_parameter("segmentation_topic", "/segmentation")
        self.declare_parameter("segmentation_mask_width", 640)
        self.declare_parameter("segmentation_mask_height", 480)
        # Re-publish detections with fresh image stamps so Isaac GXF sync matches rgb/depth/camera_info.
        self.declare_parameter("keepalive_republish_hz", 10.0)

        model_path = str(self.get_parameter("yolo_model_path").value).strip()
        expanded = os.path.expanduser(model_path)
        if not expanded or not os.path.isfile(expanded):
            raise RuntimeError(
                "Parameter 'yolo_model_path' must point to an existing YOLO weights file "
                f"(got {model_path!r})."
            )
        if YOLO is None:
            raise RuntimeError("Install ultralytics: pip install ultralytics")
        self._model = YOLO(expanded)
        self._use_seg_mask = bool(self.get_parameter("use_yolo_segmentation_mask").value)
        self._seg_w = max(1, int(self.get_parameter("segmentation_mask_width").value))
        self._seg_h = max(1, int(self.get_parameter("segmentation_mask_height").value))

        self._det_topic = str(self.get_parameter("detection_topic").value)
        self._seg_topic = str(self.get_parameter("segmentation_topic").value)
        self._cloud_topic = str(self.get_parameter("object_cloud_topic").value)
        qos_latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._pub_det = self.create_publisher(Detection2DArray, self._det_topic, qos_latched)
        self._pub_seg = (
            self.create_publisher(Image, self._seg_topic, qos_latched) if self._use_seg_mask else None
        )
        self._pub_cloud = self.create_publisher(PointCloud2, self._cloud_topic, qos_latched)

        self._rgb_msg: Optional[Image] = None
        self._depth_msg: Optional[Image] = None
        self._info: Optional[CameraInfo] = None
        self._last_det: Optional[Detection2DArray] = None
        self._last_seg: Optional[Image] = None
        self._lock = threading.Lock()

        keepalive_hz = float(self.get_parameter("keepalive_republish_hz").value)
        if keepalive_hz > 0.0:
            self.create_timer(1.0 / keepalive_hz, self._on_keepalive)

        self.create_subscription(Image, str(self.get_parameter("rgb_topic").value), self._on_rgb, 1)
        self.create_subscription(Image, str(self.get_parameter("depth_topic").value), self._on_depth, 1)
        self.create_subscription(CameraInfo, str(self.get_parameter("camera_info_topic").value), self._on_info, 1)

        self.create_service(Trigger, "~/trigger", self._on_trigger)
        seg_txt = (
            f"seg_mask '{self._seg_topic}' ({self._seg_w}x{self._seg_h})"
            if self._use_seg_mask
            else "bbox→Detection2DToMask (legacy)"
        )
        self.get_logger().info(
            f"FoundationPose bridge: detections '{self._det_topic}', {seg_txt}, "
            f"cloud '{self._cloud_topic}', trigger '~/trigger'"
        )

    def _on_rgb(self, msg: Image) -> None:
        with self._lock:
            self._rgb_msg = msg

    def _on_depth(self, msg: Image) -> None:
        with self._lock:
            self._depth_msg = msg

    def _on_info(self, msg: CameraInfo) -> None:
        with self._lock:
            self._info = msg

    def _on_keepalive(self) -> None:
        with self._lock:
            det = self._last_det
            seg = self._last_seg
            rgb_msg = self._rgb_msg
        if rgb_msg is None:
            return
        stamp = rgb_msg.header.stamp
        frame_id = rgb_msg.header.frame_id

        if self._use_seg_mask and seg is not None and self._pub_seg is not None:
            stamped_seg = copy.deepcopy(seg)
            stamped_seg.header.stamp = stamp
            stamped_seg.header.frame_id = frame_id or stamped_seg.header.frame_id
            with self._lock:
                self._last_seg = stamped_seg
            self._pub_seg.publish(stamped_seg)

        if det is None:
            return
        # Legacy bbox path: keep detection stamps fresh for Detection2DToMask sync.
        if not self._use_seg_mask:
            stamped = copy.deepcopy(det)
            stamped.header.stamp = stamp
            stamped.header.frame_id = frame_id or stamped.header.frame_id
            for item in stamped.detections:
                item.header.stamp = stamp
                item.header.frame_id = stamped.header.frame_id
            with self._lock:
                self._last_det = stamped
            self._pub_det.publish(stamped)

    def _on_trigger(self, _req: Trigger.Request, resp: Trigger.Response) -> Trigger.Response:
        tgt = str(self.get_parameter("target_class").value).strip()
        min_conf = float(self.get_parameter("min_confidence").value)
        yolo_iou = float(self.get_parameter("yolo_iou").value)
        stride = int(self.get_parameter("cloud_stride_px").value)
        zmin = float(self.get_parameter("cloud_depth_min_m").value)
        zmax = float(self.get_parameter("cloud_depth_max_m").value)
        excl_pv = self.get_parameter("yolo_exclusive_scene_classes").get_parameter_value()
        excl_list = list(excl_pv.string_array_value) if excl_pv.string_array_value else None

        ci = bool(self.get_parameter("class_match_case_insensitive").value)
        p_floor = float(self.get_parameter("predict_conf_floor").value)

        frame_fb = str(self.get_parameter("camera_frame_fallback").value)

        with self._lock:
            rgb_msg = self._rgb_msg
            depth_msg = self._depth_msg
            cam_info = self._info

        if rgb_msg is None or depth_msg is None or cam_info is None:
            resp.success = False
            resp.message = "Missing rgb, depth, or camera_info subscription data."
            return resp
        try:
            rgb, stamp, fid = _image_rgb_and_header(rgb_msg)
        except ValueError as exc:
            resp.success = False
            resp.message = str(exc)
            return resp

        try:
            depth = _depth_image_to_meters(depth_msg)
        except ValueError as exc:
            resp.success = False
            resp.message = str(exc)
            return resp

        label = tgt or "Box_1"
        score = 1.0
        use_latch = bool(self.get_parameter("use_latched_bbox_on_trigger").value)
        latch_xyxy = list(self.get_parameter("latched_bbox_xyxy").value)
        latch_for_pick: Optional[list[float]] = None
        if use_latch and len(latch_xyxy) >= 4:
            x1, y1, x2, y2 = (float(latch_xyxy[0]), float(latch_xyxy[1]), float(latch_xyxy[2]), float(latch_xyxy[3]))
            if x2 > x1 + 2.0 and y2 > y1 + 2.0:
                label = str(self.get_parameter("latched_bbox_label").value).strip() or label
                latch_for_pick = [x1, y1, x2, y2]
                self.get_logger().info(
                    f"Trigger: latched bbox ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) label={label!r} "
                    f"→ YOLO-seg match"
                )
            else:
                use_latch = False

        pick = _yolo_pick_detection(
            self._model,
            rgb,
            target_class=label if use_latch else (tgt or "Box_1"),
            min_conf=min_conf,
            yolo_iou=yolo_iou,
            predict_conf_floor=p_floor,
            exclusive_scene_classes=excl_list,
            class_match_case_insensitive=ci,
            want_mask=self._use_seg_mask,
            latch_xyxy=latch_for_pick,
        )
        if pick is None:
            resp.success = False
            resp.message = (
                f"No YOLO detection for target_class={tgt!r} (conf≥{min_conf}). "
                "Check target_class matches YOLO label (e.g. Box_1 not square_1)."
            )
            return resp

        x1, y1, x2, y2, label, score = pick.x1, pick.y1, pick.x2, pick.y2, pick.label, pick.score
        ix1 = int(round(x1))
        iy1 = int(round(y1))
        ix2 = int(round(x2))
        iy2 = int(round(y2))

        cframe = rgb_msg.header.frame_id or fid or frame_fb
        img_h, img_w = rgb.shape[:2]
        mask_full = pick.mask_full
        if mask_full is None and self._use_seg_mask:
            self.get_logger().warn(
                "YOLO model returned no instance mask — falling back to bbox rectangle for segmentation."
            )
            mask_full = _bbox_binary_mask(img_h, img_w, ix1, iy1, ix2, iy2)
        elif mask_full is None:
            mask_full = _bbox_binary_mask(img_h, img_w, ix1, iy1, ix2, iy2)

        arr = Detection2DArray()
        arr.header.stamp = stamp
        arr.header.frame_id = cframe
        arr.detections.append(
            _detection_msg(
                stamp=stamp,
                frame_id=cframe,
                x1=float(ix1),
                y1=float(iy1),
                x2=float(ix2),
                y2=float(iy2),
                class_id=label,
                score=score,
            )
        )
        self._pub_det.publish(arr)
        with self._lock:
            self._last_det = arr

        if self._use_seg_mask and self._pub_seg is not None:
            mask_fp = _resize_mask_nearest(mask_full, self._seg_w, self._seg_h)
            seg_msg = _mask_to_image_msg(mask_fp, stamp, cframe)
            self._pub_seg.publish(seg_msg)
            with self._lock:
                self._last_seg = seg_msg
            fg_px = int(np.count_nonzero(mask_fp))
            self.get_logger().info(
                f"Trigger: YOLO-seg mask {self._seg_w}x{self._seg_h}, "
                f"foreground={fg_px} px, det={label!r} conf={score:.3f}"
            )

        use_rgb_cloud = bool(self.get_parameter("include_rgb_in_cloud").value)
        cloud_rgb = rgb if use_rgb_cloud else None
        cloud = _mask_to_pointcloud(
            depth,
            cam_info,
            mask_full,
            stride_px=stride,
            z_min=zmin,
            z_max=zmax,
            stamp=stamp,
            frame_id=cframe,
            rgb_uint8=cloud_rgb,
        )
        self._pub_cloud.publish(cloud)

        resp.success = True
        seg_note = f" seg→{self._seg_topic}" if self._use_seg_mask else ""
        resp.message = f"Latched {label} conf={score:.3f}; det + cloud{seg_note} published."
        return resp


def main() -> None:
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init()
    node = FoundationPoseBridgeNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
