#!/usr/bin/env python3
"""Local HTTP site comparing TF tool0 vs Case_Base in base_link after each sequencer action."""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


def _share_html_dir() -> Path:
    """Resolved at runtime from install/share (ament_index); falls back beside source."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("ur3_pick_task")) / "web" / "pose_compare"
    except Exception:  # noqa: BLE001
        return Path(__file__).resolve().parent.parent / "web" / "pose_compare"


def _quat_dot(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return abs(sum(float(x) * float(y) for x, y in zip(a, b)))


def _quat_sep_deg(qa: tuple[float, ...], qb: tuple[float, ...]) -> float:
    d = min(1.0, max(0.0, _quat_dot(qa, qb)))
    ang = 2.0 * math.acos(d)
    return math.degrees(ang)


_HTTP_NODE: PoseCompareSite | None = None


class _PoseCompareHTTPHandler(BaseHTTPRequestHandler):
    def log_message(self, _format: str, *_args: Any) -> None:
        return

    def do_GET(self) -> None:
        share = _share_html_dir()
        if self.path.startswith("/api/data"):
            node = _HTTP_NODE
            if node is None:
                body = json.dumps({"rows": []}).encode("utf-8")
            else:
                with node._lock:
                    body = json.dumps({"rows": list(node.rows)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            p = share / "index.html"
            if not p.is_file():
                self.send_error(404, "Missing index.html in share — did you install ur3_pick_task?")
                return
            txt = p.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(txt)))
            self.end_headers()
            self.wfile.write(txt)
            return
        self.send_error(404)


class PoseCompareSite(Node):
    def __init__(self) -> None:
        super().__init__("pose_compare_site")

        self.declare_parameter("listen_action_topic", "/action_compare/action_name")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tool_frame", "tool0")
        self.declare_parameter("case_frame", "Case_Base")
        self.declare_parameter("tf_timeout_sec", 0.5)
        self.declare_parameter("http_host", "127.0.0.1")
        self.declare_parameter("http_port", 8766)

        self._listen = str(self.get_parameter("listen_action_topic").value).strip()
        self._base = str(self.get_parameter("base_frame").value).strip()
        self._tool = str(self.get_parameter("tool_frame").value).strip()
        self._case = str(self.get_parameter("case_frame").value).strip()
        self._tf_sec = float(self.get_parameter("tf_timeout_sec").value)
        host = str(self.get_parameter("http_host").value)
        port = int(self.get_parameter("http_port").value)

        self._tf = Buffer(cache_time=Duration(seconds=60.0))
        self._tf_listener = TransformListener(self._tf, self)

        self.rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

        if self._listen:
            self.create_subscription(String, self._listen, self._on_action, 50)
            self.get_logger().info(
                f'Listening for completed action names on "{self._listen}". '
                f"Comparing TF {self._base}→{self._tool} vs {self._base}→{self._case}."
            )
        else:
            self.get_logger().warn("listen_action_topic is empty — no automatic rows.")

        global _HTTP_NODE

        self._srv = ThreadingHTTPServer((host, port), _PoseCompareHTTPHandler)
        self._srv_thread = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self._srv_thread.start()
        _HTTP_NODE = self
        self.get_logger().info(f"Open http://{host}:{port}/ for pose comparison table.")

    def destroy_node(self) -> bool:
        global _HTTP_NODE
        _HTTP_NODE = None
        self._srv.shutdown()
        self._srv_thread.join(timeout=2.0)
        return super().destroy_node()

    def _lookup(self, child: str, note: str) -> tuple[list[float], list[float], str]:
        ok = ""
        try:
            t = self._tf.lookup_transform(self._base, child, Time(), timeout=Duration(seconds=self._tf_sec))
            tr = t.transform.translation
            q = t.transform.rotation
            p = [float(tr.x), float(tr.y), float(tr.z)]
            quat = (float(q.x), float(q.y), float(q.z), float(q.w))
            return p, list(quat), ok
        except Exception as exc:  # noqa: BLE001
            return ([0.0, 0.0, 0.0], (0.0, 0.0, 0.0, 1.0), f"{note}: {exc}")

    def _on_action(self, msg: String) -> None:
        name = str(msg.data).strip() or "(unnamed)"
        pt, qt, et = self._lookup(self._tool, "tool0")
        pc, qc, ec = self._lookup(self._case, "Case_Base")
        dp = [pt[i] - pc[i] for i in range(3)]
        dist = math.sqrt(sum(x * x for x in dp))
        dtheta = _quat_sep_deg(tuple(qt), tuple(qc))
        notes = "; ".join(s for s in (et.strip(), ec.strip()) if s)
        if not notes:
            notes = "TF ok"

        row = {
            "action": name,
            "dx_mm": round(dp[0] * 1000.0, 3),
            "dy_mm": round(dp[1] * 1000.0, 3),
            "dz_mm": round(dp[2] * 1000.0, 3),
            "dist_mm": round(dist * 1000.0, 3),
            "dtheta_deg": round(dtheta, 3),
            "note": notes,
        }
        with self._lock:
            self.rows.append(row)
        self.get_logger().info(
            f"Recorded after '{name}': |Δp|={row['dist_mm']} mm, Δθ={row['dtheta_deg']}° ({notes[:80]})"
        )


def main() -> None:
    rclpy.init()
    node = PoseCompareSite()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
