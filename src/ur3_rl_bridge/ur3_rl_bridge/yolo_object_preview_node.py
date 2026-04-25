#!/usr/bin/env python3
"""Run YOLO segmentation on the rl_camera RGB stream and show class names."""

import os
import threading
from typing import Dict, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - reported at runtime in the node.
    YOLO = None


OBJECT_MAP: Dict[str, Tuple[str, str]] = {
    "Cylinder_1": ("Cylindrical_Object", "Cylindrical_Tip"),
    "Cylinder_2": ("Cylindrical_Object", "Cylindrical_Tip"),
    "Box_1": ("Box_Object", "Box_Tip"),
}


def _image_msg_to_rgb(msg: Image) -> np.ndarray:
    if msg.encoding == "rgb8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3)).copy()
    if msg.encoding == "bgr8":
        bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape((msg.height, msg.width, 3))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raise ValueError(f"Unsupported image encoding '{msg.encoding}'. Expected rgb8 or bgr8.")


def _bgr_to_image_msg(bgr: np.ndarray, stamp, frame_id: str) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(bgr.shape[0])
    msg.width = int(bgr.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = int(bgr.shape[1] * 3)
    msg.data = bgr.tobytes()
    return msg


class YoloObjectPreview(Node):
    def __init__(self):
        super().__init__("yolo_object_preview")
        self.declare_parameter("rgb_topic", "/rl_camera/color")
        self.declare_parameter("model_path", "")
        self.declare_parameter("display_hz", 10.0)
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("target_class", "")
        self.declare_parameter("show_window", True)
        self.declare_parameter("publish_annotated", True)
        self.declare_parameter("annotated_topic", "/yolo/annotated")

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        if not model_path:
            raise RuntimeError("Parameter 'model_path' is required. Set it to your YOLO best.pt file.")
        if YOLO is None:
            raise RuntimeError("Python package 'ultralytics' is not installed. Install it with: pip install ultralytics")

        self._model = YOLO(os.path.expanduser(model_path))
        self._conf = float(self.get_parameter("conf").get_parameter_value().double_value)
        self._iou = float(self.get_parameter("iou").get_parameter_value().double_value)
        self._target_class = self.get_parameter("target_class").get_parameter_value().string_value.strip()
        self._show_window = self.get_parameter("show_window").get_parameter_value().bool_value
        self._latest_rgb = None
        self._latest_stamp = None
        self._latest_frame_id = ""
        self._lock = threading.Lock()

        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        self.create_subscription(Image, rgb_topic, self._on_rgb, 1)

        self._publisher = None
        if self.get_parameter("publish_annotated").get_parameter_value().bool_value:
            annotated_topic = self.get_parameter("annotated_topic").get_parameter_value().string_value
            self._publisher = self.create_publisher(Image, annotated_topic, 1)
            self.get_logger().info(f"Publishing annotated image: {annotated_topic}")

        self._gui_ok = bool(os.environ.get("DISPLAY")) and self._show_window
        if self._gui_ok:
            cv2.namedWindow("YOLO object preview", cv2.WINDOW_NORMAL)
        elif self._show_window:
            self.get_logger().warn("DISPLAY is not set; annotated image window will not open.")

        hz = float(self.get_parameter("display_hz").get_parameter_value().double_value)
        self.create_timer(1.0 / max(hz, 1.0), self._tick)
        self.get_logger().info(f"YOLO preview subscribed to {rgb_topic}")

    def _on_rgb(self, msg: Image) -> None:
        try:
            rgb = _image_msg_to_rgb(msg)
        except Exception as exc:
            self.get_logger().warn(str(exc))
            return
        with self._lock:
            self._latest_rgb = rgb
            self._latest_stamp = msg.header.stamp
            self._latest_frame_id = msg.header.frame_id

    def _tick(self) -> None:
        with self._lock:
            if self._latest_rgb is None:
                return
            rgb = self._latest_rgb.copy()
            stamp = self._latest_stamp
            frame_id = self._latest_frame_id

        result = self._model.predict(rgb, conf=self._conf, iou=self._iou, verbose=False)[0]
        names = result.names

        detected = []
        if result.boxes is not None and result.boxes.cls is not None:
            for cls_id, conf in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                class_name = str(names[int(cls_id)])
                if self._target_class and class_name != self._target_class:
                    continue
                object_type, gripper = OBJECT_MAP.get(class_name, ("Unknown_Object", "Unknown_Tip"))
                detected.append((class_name, float(conf), object_type, gripper))

        annotated_bgr = result.plot()
        if self._target_class:
            cv2.putText(
                annotated_bgr,
                f"Target: {self._target_class}",
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for i, (class_name, conf, object_type, gripper) in enumerate(detected[:4]):
            text = f"{class_name} ({conf:.2f}) -> {object_type}, {gripper}"
            cv2.putText(
                annotated_bgr,
                text,
                (12, 64 + 28 * i),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if self._publisher is not None:
            self._publisher.publish(_bgr_to_image_msg(annotated_bgr, stamp, frame_id))
        if self._gui_ok:
            cv2.imshow("YOLO object preview", annotated_bgr)
            cv2.waitKey(1)


def main():
    rclpy.init()
    node = None
    try:
        node = YoloObjectPreview()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
