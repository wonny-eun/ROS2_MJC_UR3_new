"""FoundationPose stack helpers: auto-launch, stable TF wait, grasp pose from config."""

from __future__ import annotations

import math
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import rclpy

import numpy as np
import yaml

try:
    from ur3_rl_bridge.fp_pose_utils import (
        correct_table_top_orientation,
        is_upside_down_in_base,
        rotation_for_table_top_upright,
    )
except ImportError:  # pragma: no cover — fallback when pick_task runs without ur3_rl_bridge on path.
    _FLIP_RX_PI = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)

    def is_upside_down_in_base(R_base_object: np.ndarray) -> bool:
        R = np.asarray(R_base_object, dtype=np.float64).reshape(3, 3)
        return float(R[2, 2]) < 0.0

    def correct_table_top_orientation(R_base_object: np.ndarray) -> np.ndarray:
        R = np.asarray(R_base_object, dtype=np.float64).reshape(3, 3)
        if is_upside_down_in_base(R):
            return R @ _FLIP_RX_PI
        return R

    def rotation_for_table_top_upright(
        R_base_object: np.ndarray,
        *,
        long_axis_object: Optional[Sequence[float]] = None,
        short_axis_object: Sequence[float] = (0.0, 1.0, 0.0),
    ) -> np.ndarray:
        R = correct_table_top_orientation(R_base_object)
        if long_axis_object is not None:
            yaw = yaw_long_axis_in_base(R, long_axis_object, short_axis_object)
        else:
            yaw = yaw_about_base_z(R)
        return rotation_upright_z_from_yaw(yaw)

_AXIS_VEC: Dict[str, np.ndarray] = {
    "+X": np.array([1.0, 0.0, 0.0]),
    "-X": np.array([-1.0, 0.0, 0.0]),
    "+Y": np.array([0.0, 1.0, 0.0]),
    "-Y": np.array([0.0, -1.0, 0.0]),
    "+Z": np.array([0.0, 0.0, 1.0]),
    "-Z": np.array([0.0, 0.0, -1.0]),
}


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_grasp_config(
    foundation_pose_cfg: Dict[str, Any],
    *,
    grasp_config_file: str = "",
) -> Dict[str, Any]:
    """Merge optional grasp YAML + inline foundation_pose.grasp_by_class."""
    out: Dict[str, Any] = {}
    path = str(grasp_config_file or foundation_pose_cfg.get("grasp_config_file", "")).strip()
    if path and os.path.isfile(path):
        data = load_yaml(path)
        blk = data.get("grasp_by_class", data)
        if isinstance(blk, dict):
            out.update(blk)
    inline = foundation_pose_cfg.get("grasp_by_class")
    if isinstance(inline, dict):
        out.update(inline)
    return out


def _normalize_vec(vec: np.ndarray, label: str) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError(f"{label} is zero")
    return v / n


def yaw_about_base_z(R_base_object: np.ndarray) -> float:
    """Rotation about base_link +Z (yaw) from fp_object orientation in base (radians)."""
    R = np.asarray(R_base_object, dtype=np.float64).reshape(3, 3)
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def yaw_long_axis_in_base(
    R_base_object: np.ndarray,
    long_axis_object: Sequence[float],
    short_axis_object: Sequence[float] = (0.0, 1.0, 0.0),
) -> float:
    """In-plane yaw from mesh long axis (+X for Box_1), resolving 90° FP swaps."""
    R = np.asarray(R_base_object, dtype=np.float64).reshape(3, 3)

    def _norm(v: Sequence[float]) -> np.ndarray:
        a = np.asarray(v, dtype=np.float64).reshape(3)
        n = float(np.linalg.norm(a))
        return a / n if n > 1e-12 else np.array([1.0, 0.0, 0.0], dtype=np.float64)

    long_v = R @ _norm(long_axis_object)
    short_v = R @ _norm(short_axis_object)
    long_h = long_v[:2]
    short_h = short_v[:2]
    ln = float(np.linalg.norm(long_h))
    sn = float(np.linalg.norm(short_h))
    if ln < 1e-9 and sn < 1e-9:
        return yaw_about_base_z(R)
    if sn > ln:
        h = short_h / sn
        return float((math.atan2(float(h[1]), float(h[0])) + math.pi * 0.5 + math.pi) % (2.0 * math.pi) - math.pi)
    h = long_h / ln
    return math.atan2(float(h[1]), float(h[0]))


def object_yaw_for_grasp(R_base_object: np.ndarray, grasp_spec: Dict[str, Any]) -> float:
    """Yaw synced to TCP roll: long-axis yaw when configured, else base +Z yaw."""
    R = correct_table_top_orientation(R_base_object)
    long_axis = grasp_spec.get("long_axis_in_object")
    if long_axis is not None:
        short_axis = grasp_spec.get("short_axis_in_object", [0.0, 1.0, 0.0])
        return yaw_long_axis_in_base(R, long_axis, short_axis)
    return yaw_about_base_z(R)


def rotation_upright_z_from_yaw(yaw_rad: float) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_upright_z_in_base(R_base_object: np.ndarray) -> np.ndarray:
    """
    Force object +Z parallel to base +Z (table-top upright).

    Keeps in-plane yaw from FoundationPose; removes roll/pitch tilt.
    """
    return rotation_upright_z_from_yaw(yaw_about_base_z(R_base_object))


def normalize_T_base_object_for_grasp(T_base_object: np.ndarray, grasp_spec: Dict[str, Any]) -> np.ndarray:
    """
    Apply grasp-spec constraints to fp_object in base_link.

    ``upright_in_base: true`` (Box_1 on table): object +Z aligned with base +Z, yaw only.
    """
    upright = bool(
        grasp_spec.get("upright_in_base", grasp_spec.get("upright_z_in_base", False))
    )
    if not upright:
        return np.asarray(T_base_object, dtype=np.float64)
    T = np.asarray(T_base_object, dtype=np.float64).copy()
    long_axis = grasp_spec.get("long_axis_in_object")
    short_axis = grasp_spec.get("short_axis_in_object", [0.0, 1.0, 0.0])
    T[:3, :3] = rotation_for_table_top_upright(
        T[:3, :3],
        long_axis_object=long_axis,
        short_axis_object=short_axis,
    )
    return T


def object_z_tilt_from_vertical_deg(R_base_object: np.ndarray) -> float:
    """Angle between object +Z (R[:,2]) and base +Z — 0° when upright."""
    R = np.asarray(R_base_object, dtype=np.float64).reshape(3, 3)
    z_obj = R[:, 2]
    z_obj = z_obj / max(float(np.linalg.norm(z_obj)), 1e-12)
    dot = min(1.0, max(-1.0, float(z_obj[2])))
    return float(math.degrees(math.acos(dot)))


def rotation_tcp_from_lookat(lookat: np.ndarray, roll_rad: float = 0.0) -> np.ndarray:
    """
    TCP orientation: gripper_tip +Z aligns with ``lookat`` (unit vector in base_link).

    Matches pick-task / action_sequencer lookat convention (e.g. [0, 0, -1] → downward in base).
    """
    z_axis = _normalize_vec(lookat, "lookat_vector")
    up_hint = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(z_axis, up_hint))) > 0.98:
        up_hint = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = np.cross(up_hint, z_axis)
    xn = float(np.linalg.norm(x_axis))
    if xn < 1e-12:
        raise ValueError("lookat_vector is parallel to world up; use a different look-at.")
    x_axis = x_axis / xn
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    base_rot = np.column_stack([x_axis, y_axis, z_axis])
    cr = math.cos(float(roll_rad))
    sr = math.sin(float(roll_rad))
    local_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return base_rot @ local_roll


def parse_axis_label(label: str) -> np.ndarray:
    key = str(label).strip().upper()
    if key not in _AXIS_VEC:
        raise ValueError(f"Invalid axis label {label!r}; use one of {sorted(_AXIS_VEC)}")
    return _AXIS_VEC[key].copy()


def rotation_tcp_to_object(tcp_axis_to_object_axis: Dict[str, Any]) -> np.ndarray:
    """
    Build R_object_tcp (columns = gripper_tip / TCP basis axes in object frame).

    ``tcp_axis_to_object_axis`` maps TCP axis names x,y,z to object axis labels, e.g.
    ``x: "+Y"`` means gripper_tip +X is parallel to object +Y.
    """
    if not isinstance(tcp_axis_to_object_axis, dict):
        raise ValueError("tcp_axis_to_object_axis must be a map with keys x, y, z")

    def _col(axis_name: str) -> np.ndarray:
        if axis_name not in tcp_axis_to_object_axis:
            raise ValueError(f"tcp_axis_to_object_axis missing key {axis_name!r}")
        return parse_axis_label(str(tcp_axis_to_object_axis[axis_name]))

    x = _col("x")
    y_raw = _col("y")
    z_hint = _col("z")

    x = x / max(np.linalg.norm(x), 1e-12)
    y = y_raw - x * float(np.dot(y_raw, x))
    yn = np.linalg.norm(y)
    if yn < 1e-6:
        raise ValueError("tcp_axis_to_object_axis: TCP x and y are parallel; pick independent axes.")
    y = y / yn
    z = np.cross(x, y)
    zn = np.linalg.norm(z)
    if zn < 1e-6:
        raise ValueError("tcp_axis_to_object_axis: degenerate frame from x/y.")
    z = z / zn
    if float(np.dot(z, z_hint)) < 0.0:
        z = -z
        y = -y
    return np.column_stack([x, y, z])


def _grip_point_offset_m(grasp_spec: Dict[str, Any]) -> np.ndarray:
    """Grip offset [x, y, z] metres; frame set by ``grip_point_offset_frame`` (default object)."""
    offset = grasp_spec.get(
        "grip_point_offset_m",
        grasp_spec.get("position_offset_m", grasp_spec.get("offset_xyz", [0.0, 0.0, 0.0])),
    )
    if not isinstance(offset, (list, tuple)) or len(offset) != 3:
        raise ValueError("grip_point_offset_m / position_offset_m must be [x, y, z] (metres).")
    return np.array([float(offset[0]), float(offset[1]), float(offset[2])], dtype=np.float64)


def _grip_point_offset_frame(grasp_spec: Dict[str, Any]) -> str:
    """
    ``object``: offset rotates with fp_object (legacy).
    ``base_link``: offset along fixed base_link X/Y/Z after FP object origin is placed.
    """
    raw = str(
        grasp_spec.get(
            "grip_point_offset_frame",
            grasp_spec.get("grip_offset_frame", "object"),
        )
    ).strip().lower()
    if raw in ("base", "base_link", "global", "world"):
        return "base_link"
    if raw in ("object", "fp_object", "mesh"):
        return "object"
    raise ValueError(
        f"grip_point_offset_frame must be 'object' or 'base_link' (got {raw!r})"
    )


def grip_point_position_in_base(
    T_base_object: np.ndarray,
    grasp_spec: Dict[str, Any],
) -> np.ndarray:
    """
    Grip point XYZ in base_link: FP object origin plus configured offset.

    With ``grip_point_offset_frame: base_link``, offset is added in base_link axes
    (e.g. [0, 0, 0.02] = 2 cm up in the world regardless of object yaw).
    """
    T_bo = normalize_T_base_object_for_grasp(T_base_object, grasp_spec)
    offset = _grip_point_offset_m(grasp_spec)
    if _grip_point_offset_frame(grasp_spec) == "base_link":
        return T_bo[:3, 3] + offset
    return T_bo[:3, :3] @ offset + T_bo[:3, 3]


def _lookat_in_base(grasp_spec: Dict[str, Any]) -> np.ndarray:
    """Unit look-at in base_link (default [0, 0, -1] = vertical downward)."""
    raw = grasp_spec.get(
        "lookat_vector_in_base",
        grasp_spec.get("lookat_vector", [0.0, 0.0, -1.0]),
    )
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("lookat_vector / lookat_vector_in_base must be [x, y, z] in base_link.")
    return _normalize_vec(
        np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float64),
        "lookat_vector",
    )


def _lookat_in_object(grasp_spec: Dict[str, Any]) -> Optional[np.ndarray]:
    """Legacy object-frame look-at (used only by ``transform_object_to_tcp``)."""
    raw = grasp_spec.get("lookat_vector_in_object")
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("lookat_vector_in_object must be [x, y, z] in object frame.")
    return np.array([float(raw[0]), float(raw[1]), float(raw[2])], dtype=np.float64)


def _tcp_position_in_object(grasp_spec: Dict[str, Any], lookat: np.ndarray) -> np.ndarray:
    """Grip point offset plus optional retreat along -lookat (approach standoff)."""
    t_obj = _grip_point_offset_m(grasp_spec)
    standoff = float(grasp_spec.get("approach_standoff_m", 0.0))
    if standoff > 0.0:
        lv = _normalize_vec(lookat, "lookat_vector_in_object")
        t_obj = t_obj - lv * standoff
    return t_obj


def transform_object_to_tcp(grasp_spec: Dict[str, Any]) -> np.ndarray:
    """
    4x4 T_object_tcp: TCP pose in fp_object frame.

    - Position: grip point = object origin (0,0,0) + ``grip_point_offset_m`` / ``position_offset_m``,
      minus ``approach_standoff_m`` along the unit look-at vector (retreat before final approach).
    - Orientation: prefer ``lookat_vector_in_object`` (TCP +Z along that unit vector, default [0,0,-1]),
      else ``tcp_axis_to_object_axis`` (legacy axis map).
    """
    lookat = _lookat_in_object(grasp_spec)
    axis_map = grasp_spec.get("tcp_axis_to_object_axis")
    roll = float(grasp_spec.get("tcp_roll_rad", 0.0))

    T = np.eye(4, dtype=np.float64)

    if lookat is not None:
        T[:3, :3] = rotation_tcp_from_lookat(lookat, roll)
        T[:3, 3] = _tcp_position_in_object(grasp_spec, lookat)
        return T

    if isinstance(axis_map, dict):
        T[:3, :3] = rotation_tcp_to_object(axis_map)
        T[:3, 3] = _grip_point_offset_m(grasp_spec)
        return T

    default_look = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    T[:3, :3] = rotation_tcp_from_lookat(default_look, roll)
    T[:3, 3] = _tcp_position_in_object(grasp_spec, default_look)
    return T


def transform_object_to_grasp(grasp_spec: Dict[str, Any]) -> np.ndarray:
    """Alias for ``transform_object_to_tcp``."""
    return transform_object_to_tcp(grasp_spec)


def rotation_tool_to_object(tcp_axis_to_object_axis: Dict[str, Any]) -> np.ndarray:
    """Backward-compatible alias for ``rotation_tcp_to_object``."""
    return rotation_tcp_to_object(tcp_axis_to_object_axis)


def apply_foundation_pose_grasp_overrides(
    grasp_spec: Dict[str, Any],
    cfg_fp: Dict[str, Any],
    action: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply global / per-action grasp options for ``foundation_pose_grasp``."""
    out = dict(grasp_spec)
    act = action or {}
    grasp_blk = act.get("grasp") if isinstance(act.get("grasp"), dict) else {}

    for src in (cfg_fp, act, grasp_blk):
        if not isinstance(src, dict):
            continue
        if "sync_object_yaw_as_tcp_roll" in src:
            out["sync_object_yaw_as_tcp_roll"] = bool(src["sync_object_yaw_as_tcp_roll"])
        if "tcp_roll_rad" in src:
            out["tcp_roll_rad"] = float(src["tcp_roll_rad"])
    return out


def resolve_tcp_roll_for_grasp(
    T_base_object: np.ndarray,
    grasp_spec: Dict[str, Any],
) -> tuple[float, float, bool]:
    """Return (object_yaw_rad, tcp_roll_rad, sync_yaw) for logging / grasp IK."""
    T_bo = normalize_T_base_object_for_grasp(T_base_object, grasp_spec)
    sync_yaw = bool(grasp_spec.get("sync_object_yaw_as_tcp_roll", True))
    obj_yaw = object_yaw_for_grasp(T_bo[:3, :3], grasp_spec) if sync_yaw else 0.0
    roll_off = float(grasp_spec.get("tcp_roll_rad", 0.0))
    tcp_roll = obj_yaw + roll_off if sync_yaw else roll_off
    return obj_yaw, tcp_roll, sync_yaw


def compute_base_tcp_pose(
    T_base_object: np.ndarray,
    grasp_spec: Dict[str, Any],
) -> np.ndarray:
    """
    T_base_gripper_tip for MoveIt IK (gripper_tip).

    - Position: FP object origin + ``grip_point_offset_m`` (object or base_link frame).
    - Orientation: base lookat_vector + object yaw (optional) as TCP roll.
    """
    lookat = _lookat_in_base(grasp_spec)
    _, tcp_roll, _ = resolve_tcp_roll_for_grasp(T_base_object, grasp_spec)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rotation_tcp_from_lookat(lookat, tcp_roll)
    p = grip_point_position_in_base(T_base_object, grasp_spec)
    standoff = float(grasp_spec.get("approach_standoff_m", 0.0))
    if standoff > 0.0:
        p = p - lookat * standoff
    T[:3, 3] = p
    return T


def normalize_symmetry_axes(value: Any) -> List[str]:
    """Parse Isaac ``symmetry_axes`` entries (e.g. ``y_full``, ``x_180``)."""
    if value is None:
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        out: List[str] = []
        for item in value:
            out.extend(normalize_symmetry_axes(item))
        return out
    return []


def symmetry_axes_for_class(class_name: str, fp_cfg: Dict[str, Any]) -> List[str]:
    """
    Rotational symmetries for Isaac FoundationPose (mesh frame).

    Defaults to ``y_full`` (cylinder/box exports: geometric axis along mesh +Y).
    Configure via ``symmetry_axes_by_class`` in ``foundation_pose`` YAML.
    """
    mapping = fp_cfg.get("symmetry_axes_by_class")
    if not isinstance(mapping, dict):
        return ["y_full"]

    name = str(class_name).strip()
    if name in mapping:
        return normalize_symmetry_axes(mapping[name])

    for pattern, raw in mapping.items():
        key = str(pattern).strip()
        if key in ("default", "_default"):
            continue
        if fnmatch(name, key):
            return normalize_symmetry_axes(raw)

    if "default" in mapping:
        axes = normalize_symmetry_axes(mapping["default"])
        return axes if axes else ["y_full"]
    if "_default" in mapping:
        axes = normalize_symmetry_axes(mapping["_default"])
        return axes if axes else ["y_full"]

    return ["y_full"]


def grasp_spec_with_standoff(grasp_spec: Dict[str, Any], standoff_m: float) -> Dict[str, Any]:
    """Copy grasp spec with ``approach_standoff_m`` overridden (hover vs touch stages)."""
    out = dict(grasp_spec)
    out["approach_standoff_m"] = float(standoff_m)
    return out


def compute_base_tool0_pose(
    T_base_object: np.ndarray,
    grasp_spec: Dict[str, Any],
    T_tool_tcp: np.ndarray,
) -> np.ndarray:
    """Deprecated: converts TCP goal to tool0. Prefer ``compute_base_tcp_pose`` + IK on gripper_tip."""
    T_base_tcp = compute_base_tcp_pose(T_base_object, grasp_spec)
    return T_base_tcp @ np.linalg.inv(T_tool_tcp)


@dataclass(frozen=True)
class HoverIkAttempt:
    """Successful IK toward the absolute HOVER pose from the current configuration."""

    joint_positions: List[float]
    step_scale: float


HoverSolveFn = Callable[[], Optional[HoverIkAttempt]]
HoverScaleSolveFn = Callable[[float], Optional[List[float]]]
HoverExecuteAtScaleFn = Callable[[List[float], float], bool]
HoverSettleFn = Callable[[], None]


def run_hover_convergence_loop(
    *,
    max_iterations: int,
    step_scales: Sequence[float],
    solve_at_scale: HoverScaleSolveFn,
    execute_at_scale: HoverExecuteAtScaleFn,
    settle_after_partial: HoverSettleFn,
    log_info: Callable[[str], None],
    log_warn: Callable[[str], None],
    log_error: Callable[[str], None],
    full_step_tolerance: float = 1e-6,
) -> bool:
    """
    Iteratively home the arm to the absolute HOVER pose before plunge.

    Each outer iteration re-solves from the current configuration. Within an iteration,
    step scales are tried from largest to smallest until MoveIt planning+execution succeeds
    (prefer a full jump; fall back to partial steps only when larger scales fail). Only
    ``step_scale == 1.0`` completes stage 1.
    """
    max_iter = max(1, int(max_iterations))
    unique_scales = sorted({float(s) for s in step_scales}, reverse=True)
    if not unique_scales:
        unique_scales = [1.0]

    for iteration in range(1, max_iter + 1):
        executed_scale: Optional[float] = None
        for scale in unique_scales:
            joints = solve_at_scale(scale)
            if joints is None:
                continue
            if execute_at_scale(joints, scale):
                executed_scale = scale
                break
            log_warn(
                f"foundation_pose_grasp HOVER: iteration {iteration}/{max_iter} — "
                f"planning/execution failed at step_scale={scale:.3f}; trying next scale."
            )

        if executed_scale is None:
            log_error(
                f"foundation_pose_grasp HOVER: iteration {iteration}/{max_iter} — "
                "no IK+plan succeeded at any step scale."
            )
            return False

        if executed_scale >= 1.0 - float(full_step_tolerance):
            log_info(
                f"foundation_pose_grasp HOVER: reached full target on iteration "
                f"{iteration}/{max_iter} (step_scale={executed_scale:.3f})."
            )
            return True

        log_warn(
            f"foundation_pose_grasp HOVER: iteration {iteration}/{max_iter} — "
            f"partial step_scale={executed_scale:.3f}; settling and re-approaching from new pose."
        )
        settle_after_partial()

    log_error(
        f"foundation_pose_grasp HOVER: failed after {max_iter} iteration(s) — "
        "could not achieve step_scale=1.0 (target blocked or unreachable). Aborting before plunge."
    )
    return False


def grasp_spec_for_class(grasp_by_class: Dict[str, Any], target_class: str) -> Dict[str, Any]:
    if target_class in grasp_by_class:
        return dict(grasp_by_class[target_class])
    lower = {str(k).lower(): v for k, v in grasp_by_class.items()}
    key = str(target_class).lower()
    if key in lower:
        return dict(lower[key])
    valid = ", ".join(sorted(str(k) for k in grasp_by_class)) or "<none>"
    raise ValueError(f"No grasp spec for class {target_class!r}. Defined: {valid}")


class FoundationPoseStackLauncher:
    """Start bridge + Isaac FoundationPose launch in background if not already running."""

    def __init__(self, logger: Any) -> None:
        self._log = logger
        self._proc: Optional[subprocess.Popen] = None
        self._proc_log_file: Any = None
        self._stack_log_path: str = ""
        self._last_mesh: Optional[str] = None
        self._last_symmetry_axes: tuple[str, ...] = ()
        # True when this call launched or force-relaunched the stack (cold Isaac / TensorRT).
        self.started_fresh: bool = False

    @staticmethod
    def _resolve_stack_launch_file() -> Optional[str]:
        """Installed share path, then source-tree fallback (before colcon reinstall)."""
        try:
            from ament_index_python.packages import get_package_share_directory

            share = get_package_share_directory("ur3_rl_bridge")
            installed = os.path.join(share, "launch", "foundation_pose_stack.launch.py")
            if os.path.isfile(installed):
                return installed
        except Exception:  # noqa: BLE001
            pass
        here = os.path.dirname(os.path.abspath(__file__))
        for rel in (
            os.path.join(here, "..", "..", "..", "ur3_rl_bridge", "launch", "foundation_pose_stack.launch.py"),
            "/home/wonny/ur3_control/src/ROS2_MuJoCo_UR3/src/ur3_rl_bridge/launch/foundation_pose_stack.launch.py",
        ):
            path = os.path.normpath(rel)
            if os.path.isfile(path):
                return path
        return None

    @staticmethod
    def _stack_launch_command(*, use_sim_time: bool, stack_launch_file: str | None = None) -> list[str]:
        launch_name = (stack_launch_file or "").strip()
        if launch_name:
            return ["ros2", "launch", "ur3_rl_bridge", launch_name, f"use_sim_time:={'true' if use_sim_time else 'false'}"]
        launch_file = FoundationPoseStackLauncher._resolve_stack_launch_file()
        if launch_file:
            cmd = ["ros2", "launch", launch_file]
        else:
            cmd = ["ros2", "launch", "ur3_rl_bridge", "foundation_pose_stack.launch.py"]
        if use_sim_time:
            cmd.append("use_sim_time:=true")
        else:
            cmd.append("use_sim_time:=false")
        return cmd

    @staticmethod
    def service_ready(node: Any, service_name: str, timeout_sec: float = 0.5) -> bool:
        from rclpy.node import Node

        if not isinstance(node, Node):
            return False
        names = node.get_service_names_and_types()
        return any(s == service_name for s, _ in names)

    def stack_active(self) -> bool:
        """True if this launcher started a stack subprocess or bridge service is up."""
        if self._proc is not None and self._proc.poll() is None:
            return True
        return False

    def stop_completely(self) -> None:
        """Terminate our launch subprocess and any orphan foundation_pose_stack processes."""
        self._stop_stack_completely()

    def _stop_stack_completely(self) -> None:
        """Terminate our launch subprocess and any orphan foundation_pose_stack processes."""
        self.shutdown()
        for pattern in (
            "foundation_pose_stack.launch.py",
            "foundation_pose_stack_realsense.launch.py",
            "isaac_foundation_pose_rlcamera.launch.py",
            "isaac_foundation_pose_realsense.launch.py",
            "foundation_pose_bridge_node",
            "foundation_pose_depth_mono16",
            "foundation_pose_output_tf",
            "foundationpose_container",
            "component_container_mt",
        ):
            try:
                subprocess.run(["pkill", "-f", pattern], check=False, timeout=5.0)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(3.5)

    def ensure_running(
        self,
        node: Any,
        *,
        bridge_trigger: str,
        bridge_set_params: str,
        use_sim_time: bool = True,
        launch_timeout_sec: float = 120.0,
        spin_cb: Optional[Callable[[float], None]] = None,
        mesh_file_path: Optional[str] = None,
        texture_file_path: Optional[str] = None,
        symmetry_axes: Optional[Sequence[str]] = None,
        tf_reference_frame: Optional[str] = None,
        tf_upright_in_base: Optional[bool] = None,
        tf_long_axis_in_object: Optional[Sequence[float]] = None,
        force_relaunch: bool = False,
        stack_launch_file: str | None = None,
        isaac_rgb_topic: str = "/rgb/image_rect_color",
        isaac_pipeline_ready_timeout_sec: float = 90.0,
    ) -> bool:
        mesh = (mesh_file_path or os.environ.get("FOUNDATION_POSE_MESH") or "").strip() or None
        texture = (texture_file_path or os.environ.get("FOUNDATION_POSE_TEXTURE") or "").strip() or None
        sym_list = tuple(normalize_symmetry_axes(symmetry_axes)) if symmetry_axes is not None else ()
        stack_up = self.service_ready(node, bridge_trigger, 0.2)
        if stack_up and (mesh or sym_list):
            if force_relaunch:
                self._log.warn(
                    f"FoundationPose: force_relaunch — stopping existing stack to apply mesh {mesh!r} "
                    f"symmetry_axes={list(sym_list) or list(self._last_symmetry_axes)}."
                )
                self._stop_stack_completely()
                stack_up = False
            elif mesh and self._last_mesh and mesh != self._last_mesh:
                self._log.warn(
                    f"FoundationPose target mesh changed ({self._last_mesh!r} → {mesh!r}); restarting stack."
                )
                self._stop_stack_completely()
                stack_up = False
            elif sym_list and sym_list != self._last_symmetry_axes:
                self._log.warn(
                    f"FoundationPose symmetry_axes changed ({list(self._last_symmetry_axes)} → "
                    f"{list(sym_list)}); restarting stack."
                )
                self._stop_stack_completely()
                stack_up = False

        if stack_up:
            self.started_fresh = False
            if mesh:
                self._log.info(
                    f"FoundationPose stack already running (bridge service present), mesh {mesh!r} "
                    f"symmetry_axes={list(self._last_symmetry_axes)}."
                )
            else:
                self._log.info("FoundationPose stack already running (bridge service present).")
            if mesh:
                self._last_mesh = mesh
            return True

        self.started_fresh = True
        if self._proc is not None and self._proc.poll() is None:
            self._log.info("FoundationPose stack launch already in progress…")
        else:
            cmd = self._stack_launch_command(
                use_sim_time=use_sim_time,
                stack_launch_file=stack_launch_file,
            )
            inner = " ".join(cmd)
            parts: list[str] = []
            ws_sh = os.path.expanduser("~/ur3_control/source_ws.sh")
            if os.path.isfile(ws_sh):
                parts.append(f"source {ws_sh}")
            fp_env = os.path.expanduser("~/isaac_ros_assets/setup_foundationpose_env.sh")
            if os.path.isfile(fp_env):
                parts.append(f"source {fp_env}")
            if mesh:
                parts.append(f"export FOUNDATION_POSE_MESH={shlex.quote(mesh)}")
            if texture:
                parts.append(f"export FOUNDATION_POSE_TEXTURE={shlex.quote(texture)}")
            if sym_list:
                parts.append(
                    f"export FOUNDATION_POSE_SYMMETRY_AXES={shlex.quote(','.join(sym_list))}"
                )
            ref_tf = (tf_reference_frame or os.environ.get("FOUNDATION_POSE_TF_REFERENCE_FRAME") or "").strip()
            if ref_tf:
                parts.append(f"export FOUNDATION_POSE_TF_REFERENCE_FRAME={shlex.quote(ref_tf)}")
            if tf_upright_in_base is not None:
                parts.append(
                    f"export FOUNDATION_POSE_TF_UPRIGHT_IN_BASE={'1' if tf_upright_in_base else '0'}"
                )
            long_axis = tf_long_axis_in_object
            if long_axis is not None and len(long_axis) == 3:
                la = ",".join(str(float(v)) for v in long_axis)
                parts.append(f"export FOUNDATION_POSE_LONG_AXIS={shlex.quote(la)}")
            parts.append("export FOUNDATION_POSE_USE_YOLO_SEG_MASK=1")
            parts.append(f"exec {inner}")
            cmd = ["bash", "-lc", " && ".join(parts)]
            sym_txt = ",".join(sym_list) if sym_list else "(none)"
            self._log.info(
                f"Launching FoundationPose stack (symmetry_axes={sym_txt}): {' '.join(cmd)}"
            )
            log_path = os.path.expanduser("~/.ros/log/foundation_pose_stack_latest.log")
            try:
                log_fp = open(log_path, "w", encoding="utf-8")
                log_fp.write(f"# launch: {' '.join(cmd)}\n")
                log_fp.flush()
            except OSError:
                log_fp = None
            self._stack_log_path = log_path
            self._proc = subprocess.Popen(
                cmd,
                stdout=log_fp if log_fp is not None else subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
                start_new_session=True,
            )
            if log_fp is not None:
                self._proc_log_file = log_fp
            else:
                self._proc_log_file = None

        deadline = time.monotonic() + max(5.0, float(launch_timeout_sec))
        while time.monotonic() < deadline:
            if spin_cb is not None:
                spin_cb(0.05)
            trig_ok = self.service_ready(node, bridge_trigger, 0.5)
            params_ok = (not bridge_set_params) or self.service_ready(node, bridge_set_params, 0.5)
            if trig_ok and params_ok:
                self._log.info("FoundationPose stack is ready (bridge trigger + set_parameters up).")
                if self.started_fresh and isaac_pipeline_ready_timeout_sec > 0.0:
                    rgb_topic = str(isaac_rgb_topic or "/rgb/image_rect_color").strip()
                    self._log.info(
                        f"FoundationPose: waiting up to {isaac_pipeline_ready_timeout_sec:.0f}s "
                        f"for Isaac RGB bridge {rgb_topic!r}…"
                    )
                    if not wait_for_topic_publishers(
                        node,
                        rgb_topic,
                        timeout_sec=float(isaac_pipeline_ready_timeout_sec),
                        spin_cb=spin_cb,
                    ):
                        tail = self.log_tail()
                        self._log.error(
                            f"FoundationPose: Isaac pipeline not ready — no publisher on {rgb_topic!r}. "
                            "Is sim running (/rl_camera/noisy/color + /clock)? "
                            "Stale GPU stack processes can block Nitros — retry after "
                            "pkill -f foundation_pose_stack."
                        )
                        if tail:
                            self._log.error(f"FoundationPose stack log (tail):\n{tail}")
                        return False
                    self._log.info(f"FoundationPose: Isaac RGB bridge publishing {rgb_topic!r}.")
                if mesh:
                    self._last_mesh = mesh
                if sym_list:
                    self._last_symmetry_axes = sym_list
                return True
            if self._proc is not None and self._proc.poll() is not None:
                err = self.log_tail(max_chars=3000)
                if not err and self._proc.stdout is not None:
                    err = self._proc.stdout.read().decode("utf-8", errors="replace")
                self._log.error(
                    f"FoundationPose stack launch exited ({self._proc.returncode}): "
                    f"{err[-1200:] if err else '<no output — see ~/.ros/log/foundation_pose_stack_latest.log>'}"
                )
                return False
            time.sleep(0.25)

        self._log.error("FoundationPose stack launch timed out waiting for bridge service.")
        return False

    def log_tail(self, max_chars: int = 2000) -> str:
        if self._stack_log_path and os.path.isfile(self._stack_log_path):
            try:
                with open(self._stack_log_path, encoding="utf-8", errors="replace") as f:
                    data = f.read()
                return data[-max_chars:] if data else ""
            except OSError:
                pass
        return read_stack_launch_log_tail(self, max_chars=max_chars)

    def shutdown(self) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._proc_log_file is not None:
            try:
                self._proc_log_file.close()
            except Exception:  # noqa: BLE001
                pass
            self._proc_log_file = None
        self._proc = None


def wait_tf_stable(
    lookup_fn: Callable[[], Tuple[np.ndarray, np.ndarray]],
    *,
    sample_count: int = 12,
    sample_period_sec: float = 0.08,
    max_position_std_m: float = 0.002,
    max_rotation_deg: float = 1.5,
    max_yaw_std_deg: float = 2.0,
    warmup_sec: float = 0.0,
    min_elapsed_sec: float = 0.0,
    timeout_sec: float = 45.0,
    spin_cb: Optional[Callable[[float], None]] = None,
) -> Tuple[bool, str]:
    """
    Wait until fp_object TF stops moving (filtered/locked pose).

    ``warmup_sec`` / ``min_elapsed_sec`` defer lock until FoundationPose has refined.
    ``max_yaw_std_deg`` checks in-plane yaw spread (unwrap + std), not only quat delta.
    """
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    first_sample_at: Optional[float] = None
    last_err = ""

    while time.monotonic() < deadline:
        if spin_cb is not None:
            spin_cb(float(sample_period_sec))
        try:
            t, q = lookup_fn()
            samples.append((np.asarray(t, dtype=np.float64).reshape(3), np.asarray(q, dtype=np.float64).reshape(4)))
            if first_sample_at is None:
                first_sample_at = time.monotonic()
            last_err = ""
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            time.sleep(float(sample_period_sec))
            continue

        need = max(3, int(sample_count))
        if len(samples) < need:
            time.sleep(float(sample_period_sec))
            continue

        elapsed = time.monotonic() - (first_sample_at or time.monotonic())
        if elapsed < max(0.0, float(warmup_sec)):
            time.sleep(float(sample_period_sec))
            continue
        if elapsed < max(0.0, float(min_elapsed_sec)):
            time.sleep(float(sample_period_sec))
            continue

        window = samples[-need:]
        trans = np.stack([s[0] for s in window], axis=0)
        t_std = float(np.max(np.std(trans, axis=0)))

        yaws = np.array(
            [
                yaw_about_base_z(correct_table_top_orientation(_quat_xyzw_to_rot(q)))
                for _t, q in window
            ],
            dtype=np.float64,
        )
        yaw_std = float(np.std(np.unwrap(yaws)))

        q0 = window[0][1]
        q0 = q0 / max(np.linalg.norm(q0), 1e-12)
        ang_max = 0.0
        for _t, q in window[1:]:
            qn = q / max(np.linalg.norm(q), 1e-12)
            d = min(1.0, max(-1.0, abs(float(np.dot(q0, qn)))))
            ang_max = max(ang_max, float(np.degrees(2.0 * np.arccos(d))))

        yaw_ok = yaw_std <= math.radians(float(max_yaw_std_deg))
        if t_std <= float(max_position_std_m) and ang_max <= float(max_rotation_deg) and yaw_ok:
            return True, (
                f"stable (σ_pos={t_std*1000:.2f} mm, Δθ={ang_max:.2f}°, "
                f"σ_yaw={math.degrees(yaw_std):.2f}° over {need} samples, elapsed={elapsed:.1f}s)"
            )

        time.sleep(float(sample_period_sec))

    return False, last_err or "timeout waiting for stable object TF"


def _quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
    """3×3 rotation from quaternion (x, y, z, w)."""
    x, y, z, w = (float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3]))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def wait_for_topic_publishers(
    node: Any,
    topic: str,
    *,
    timeout_sec: float = 30.0,
    spin_cb: Optional[Callable[[float], None]] = None,
) -> bool:
    """Return True when at least one publisher exists on ``topic``."""
    deadline = time.monotonic() + max(0.5, float(timeout_sec))
    while time.monotonic() < deadline and rclpy.ok():
        try:
            if node.get_publishers_info_by_topic(topic):
                return True
        except Exception:  # noqa: BLE001
            pass
        if spin_cb is not None:
            spin_cb(0.1)
        else:
            rclpy.spin_once(node, timeout_sec=0.1)
    return False


def read_stack_launch_log_tail(launcher: "FoundationPoseStackLauncher", max_chars: int = 2000) -> str:
    """Best-effort tail of background ``ros2 launch`` stdout for diagnostics."""
    proc = getattr(launcher, "_proc", None)
    if proc is None or proc.stdout is None:
        return ""
    try:
        import select

        chunks: list[bytes] = []
        while select.select([proc.stdout], [], [], 0)[0]:
            chunk = proc.stdout.read(4096)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        if not raw and proc.poll() is not None:
            raw = proc.stdout.read() or b""
        return raw.decode("utf-8", errors="replace")[-max_chars:]
    except Exception:  # noqa: BLE001
        return ""


def wait_for_detection3d_output(
    node: Any,
    *,
    topic: str = "/output",
    timeout_sec: float = 120.0,
    spin_cb: Optional[Callable[[float], None]] = None,
) -> bool:
    """Block until Isaac FoundationPose publishes at least one Detection3DArray detection."""
    from vision_msgs.msg import Detection3DArray

    got = {"ok": False}

    def _cb(msg: Detection3DArray) -> None:
        if len(msg.detections) > 0:
            got["ok"] = True

    sub = node.create_subscription(Detection3DArray, topic, _cb, 10)
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    try:
        while time.monotonic() < deadline and rclpy.ok():
            if spin_cb is not None:
                spin_cb(0.05)
            else:
                rclpy.spin_once(node, timeout_sec=0.05)
            if got["ok"]:
                return True
    finally:
        node.destroy_subscription(sub)
    return False
