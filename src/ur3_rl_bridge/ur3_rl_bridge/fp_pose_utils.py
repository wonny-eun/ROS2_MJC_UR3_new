"""Shared FoundationPose orientation helpers (base_link upright + long-axis yaw)."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

PoseSample = Tuple[np.ndarray, np.ndarray]  # translation (3,), quaternion xyzw (4,)

# 180° about mesh +X (table-top box): maps upside-down FP pose to object +Z up in base.
_FLIP_RX_PI = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64)


def wrap_to_pi(angle_rad: float) -> float:
    return float((float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi)


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / n


def quat_xyzw_to_rot(q_xyzw: np.ndarray) -> np.ndarray:
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


def rot_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(R[0, 0] + R[1, 1] + R[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return normalize_quaternion(np.array([x, y, z, w], dtype=np.float64))


def pose_to_matrix(t: np.ndarray, q_xyzw: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quat_xyzw_to_rot(q_xyzw)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def matrix_to_pose(T: np.ndarray) -> PoseSample:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return T[:3, 3].copy(), rot_to_quat_xyzw(T[:3, :3])


def yaw_about_base_z(R_base_object: np.ndarray) -> float:
    R = np.asarray(R_base_object, dtype=np.float64).reshape(3, 3)
    return math.atan2(float(R[1, 0]), float(R[0, 0]))


def _normalize_axis(axis: Sequence[float]) -> np.ndarray:
    v = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return v / n


def yaw_long_axis_in_base(
    R_base_object: np.ndarray,
    long_axis_object: Sequence[float],
    short_axis_object: Sequence[float] = (0.0, 1.0, 0.0),
) -> float:
    """In-plane yaw from mesh long axis (+X for Box_1), resolving 90° FP swaps."""
    R = np.asarray(R_base_object, dtype=np.float64).reshape(3, 3)
    long_v = R @ _normalize_axis(long_axis_object)
    short_v = R @ _normalize_axis(short_axis_object)
    long_h = long_v[:2]
    short_h = short_v[:2]
    ln = float(np.linalg.norm(long_h))
    sn = float(np.linalg.norm(short_h))
    if ln < 1e-9 and sn < 1e-9:
        return yaw_about_base_z(R)
    if sn > ln:
        h = short_h / sn
        return wrap_to_pi(math.atan2(float(h[1]), float(h[0])) + math.pi * 0.5)
    h = long_h / ln
    return math.atan2(float(h[1]), float(h[0]))


def rotation_upright_z_from_yaw(yaw_rad: float) -> np.ndarray:
    c = math.cos(float(yaw_rad))
    s = math.sin(float(yaw_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rotation_upright_z_in_base(R_base_object: np.ndarray) -> np.ndarray:
    return rotation_upright_z_from_yaw(yaw_about_base_z(R_base_object))


def object_z_tilt_from_vertical_deg(R_base_object: np.ndarray) -> float:
    R = np.asarray(R_base_object, dtype=np.float64).reshape(3, 3)
    z_obj = R[:, 2]
    z_obj = z_obj / max(float(np.linalg.norm(z_obj)), 1e-12)
    dot = min(1.0, max(-1.0, float(z_obj[2])))
    return float(math.degrees(math.acos(dot)))


def is_upside_down_in_base(R_base_object: np.ndarray) -> bool:
    """True when object +Z (R[:,2]) points opposite to base +Z (FP inverted pose)."""
    R = np.asarray(R_base_object, dtype=np.float64).reshape(3, 3)
    return float(R[2, 2]) < 0.0


def correct_table_top_orientation(R_base_object: np.ndarray) -> np.ndarray:
    """
    Fix FoundationPose upside-down estimates before yaw extraction.

    Box on table: mesh +Z should align with base +Z. When FP returns inverted pose
    (tilt ≈ 175°), apply 180° about object +X so long-axis yaw is meaningful.
    """
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
    """Table-top upright R: flip if inverted, then yaw-only about base +Z."""
    R = correct_table_top_orientation(R_base_object)
    yaw = yaw_for_upright_pose(
        R,
        long_axis_object=long_axis_object,
        short_axis_object=short_axis_object,
    )
    return rotation_upright_z_from_yaw(yaw)


def yaw_for_upright_pose(
    R_base_object: np.ndarray,
    *,
    long_axis_object: Optional[Sequence[float]] = None,
    short_axis_object: Sequence[float] = (0.0, 1.0, 0.0),
) -> float:
    R = correct_table_top_orientation(R_base_object)
    if long_axis_object is not None:
        return yaw_long_axis_in_base(R, long_axis_object, short_axis_object)
    return yaw_about_base_z(R)


def upright_pose_in_base(
    t_base: np.ndarray,
    R_base_object: np.ndarray,
    *,
    long_axis_object: Optional[Sequence[float]] = None,
    short_axis_object: Sequence[float] = (0.0, 1.0, 0.0),
) -> PoseSample:
    """Force object +Z parallel to base +Z; keep in-plane yaw (long-axis when configured)."""
    R_up = rotation_for_table_top_upright(
        R_base_object,
        long_axis_object=long_axis_object,
        short_axis_object=short_axis_object,
    )
    return np.asarray(t_base, dtype=np.float64).reshape(3).copy(), rot_to_quat_xyzw(R_up)


def average_poses_in_base(
    samples: list[PoseSample],
    *,
    use_yaw_median: bool,
    long_axis_object: Optional[Sequence[float]] = None,
    short_axis_object: Sequence[float] = (0.0, 1.0, 0.0),
) -> PoseSample:
    if not samples:
        raise ValueError("samples must be non-empty")
    translations = np.stack([s[0] for s in samples], axis=0)
    t_mean = np.mean(translations, axis=0)
    rotations = [correct_table_top_orientation(quat_xyzw_to_rot(s[1])) for s in samples]

    if use_yaw_median:
        yaws = np.array(
            [
                yaw_for_upright_pose(
                    R,
                    long_axis_object=long_axis_object,
                    short_axis_object=short_axis_object,
                )
                for R in rotations
            ],
            dtype=np.float64,
        )
        yaw_med = float(np.median(np.unwrap(yaws)))
        return t_mean, rot_to_quat_xyzw(rotation_upright_z_from_yaw(yaw_med))

    if len(samples) == 1:
        yaw = yaw_for_upright_pose(
            rotations[0],
            long_axis_object=long_axis_object,
            short_axis_object=short_axis_object,
        )
        return t_mean, rot_to_quat_xyzw(rotation_upright_z_from_yaw(yaw))

    ref = normalize_quaternion(samples[0][1])
    acc = np.zeros(4, dtype=np.float64)
    for _t, q in samples:
        qn = normalize_quaternion(q)
        if float(np.dot(ref, qn)) < 0.0:
            qn = -qn
        acc += qn
    q_mean = normalize_quaternion(acc)
    R_mean = quat_xyzw_to_rot(q_mean)
    return upright_pose_in_base(
        t_mean,
        R_mean,
        long_axis_object=long_axis_object,
        short_axis_object=short_axis_object,
    )
