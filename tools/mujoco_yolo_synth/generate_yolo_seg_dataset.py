#!/usr/bin/env python3
"""
Generate synthetic training data for YOLOv8-seg from MuJoCo.

Outputs:
  <out_dir>/
    images/{train,val}/XXXXXX.jpg
    labels/{train,val}/XXXXXX.txt     (YOLOv8-seg polygon format)
    dataset.yaml

YOLOv8-seg label format (one object per line):
  <class_id> x1 y1 x2 y2 ... xN yN
where polygon points are normalized to [0,1] by image width/height.

Notes:
- Default workflow: spawn the free object at ``--pos-z`` (e.g. 0.9 m), run physics for ``--settle-time-s``
  (default 5 s of simulation time), then render so the object has landed near the table (~0.79 m).
- Match your MJCF tabletop height with ``--pos-z`` for the initial drop height.
- For Intel RealSense–style capture, keep the camera at least ~0.3 m from the target geoms; use
  ``--cam-min-dist-m`` (default 0.30) and optionally ``--cam-track-target`` so the free camera
  look-at follows the object centroid while sampling azimuth/elevation/distance.
- Use ``--cam-frame-placement mixed`` (default) so objects appear in the image center *and* near
  corners/margins. Set ``center`` only if you want the old always-centered training bias.
- "Texture randomization" at runtime is limited in MuJoCo; this script supports:
    - per-geom RGBA randomization
    - swapping geom material IDs among a provided list (if your XML defines multiple materials)
  If you want true texture swaps, define multiple materials/textures in the XML and
  enable --material-ids / --geom-material-map.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import mujoco
import numpy as np


@dataclasses.dataclass(frozen=True)
class NoiseConfig:
    rgb_gaussian_sigma: float  # in pixel intensity units (0..255)
    rgb_salt_pepper_prob: float
    rgb_brightness_jitter: float  # additive pixel intensity jitter
    rgb_contrast_jitter: float  # multiplicative contrast jitter around 1.0
    depth_gaussian_sigma_m: float  # meters
    depth_quantization_step_m: float  # meters (e.g. 0.001 for 1mm)
    depth_dropout_prob: float
    depth_edge_dropout_prob: float
    depth_edge_threshold_m: float
    depth_shadow_enable: bool
    depth_shadow_direction: str
    depth_shadow_k_px_m: float
    depth_shadow_min_radius_px: int
    depth_shadow_max_radius_px: int
    depth_shadow_edge_threshold_m: float
    depth_shadow_max_depth_m: float


def _mkdir_clean(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def _apply_rgb_noise(rgb_u8: np.ndarray, cfg: NoiseConfig) -> np.ndarray:
    out = rgb_u8.astype(np.float32)
    if cfg.rgb_contrast_jitter > 0:
        contrast = _rand_uniform(1.0 - cfg.rgb_contrast_jitter, 1.0 + cfg.rgb_contrast_jitter)
        out = (out - 127.5) * contrast + 127.5
    if cfg.rgb_brightness_jitter > 0:
        out += _rand_uniform(-cfg.rgb_brightness_jitter, cfg.rgb_brightness_jitter)
    if cfg.rgb_gaussian_sigma > 0:
        out += np.random.normal(loc=0.0, scale=cfg.rgb_gaussian_sigma, size=out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if cfg.rgb_salt_pepper_prob > 0:
        salt = np.random.random(out.shape[:2]) < (cfg.rgb_salt_pepper_prob * 0.5)
        pepper = np.random.random(out.shape[:2]) < (cfg.rgb_salt_pepper_prob * 0.5)
        out[salt, :] = 255
        out[pepper, :] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_depth_shadow(depth_m: np.ndarray, cfg: NoiseConfig) -> np.ndarray:
    if (
        not cfg.depth_shadow_enable
        or cfg.depth_shadow_k_px_m <= 0
        or cfg.depth_shadow_max_radius_px <= 0
        or cfg.depth_shadow_edge_threshold_m <= 0
    ):
        return depth_m

    out = depth_m.astype(np.float32).copy()
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if depth_m.shape[1] < 2:
        return out

    left = depth_m[:, :-1]
    right = depth_m[:, 1:]
    valid_pair = valid[:, :-1] & valid[:, 1:]
    min_radius = int(min(cfg.depth_shadow_min_radius_px, cfg.depth_shadow_max_radius_px))
    max_radius = int(max(cfg.depth_shadow_min_radius_px, cfg.depth_shadow_max_radius_px))
    direction = cfg.depth_shadow_direction.lower()

    if direction == "left":
        # Aligned D435i depth can leave a horizontal shadow on one side of foreground edges.
        edge = valid_pair & ((left - right) > cfg.depth_shadow_edge_threshold_m)
        edge_rows, edge_cols = np.nonzero(edge)
        fg_depth = right[edge_rows, edge_cols]
        for row, col, z in zip(edge_rows.tolist(), edge_cols.tolist(), fg_depth.tolist()):
            if cfg.depth_shadow_max_depth_m > 0 and float(z) > cfg.depth_shadow_max_depth_m:
                continue
            radius = int(np.clip(round(cfg.depth_shadow_k_px_m / max(float(z), 1e-6)), min_radius, max_radius))
            x0 = max(0, col - radius + 1)
            out[row, x0 : col + 1] = 0.0
    elif direction == "right":
        edge = valid_pair & ((right - left) > cfg.depth_shadow_edge_threshold_m)
        edge_rows, edge_cols = np.nonzero(edge)
        fg_depth = left[edge_rows, edge_cols]
        width = depth_m.shape[1]
        for row, col, z in zip(edge_rows.tolist(), edge_cols.tolist(), fg_depth.tolist()):
            if cfg.depth_shadow_max_depth_m > 0 and float(z) > cfg.depth_shadow_max_depth_m:
                continue
            radius = int(np.clip(round(cfg.depth_shadow_k_px_m / max(float(z), 1e-6)), min_radius, max_radius))
            x0 = col + 1
            x1 = min(width, x0 + radius)
            out[row, x0:x1] = 0.0

    return out


def _apply_depth_noise_and_quantization(depth_m: np.ndarray, cfg: NoiseConfig) -> np.ndarray:
    out = depth_m.astype(np.float32).copy()
    valid = np.isfinite(out) & (out > 0)
    if cfg.depth_gaussian_sigma_m > 0:
        out[valid] += np.random.normal(0.0, cfg.depth_gaussian_sigma_m, size=int(valid.sum())).astype(np.float32)
    if cfg.depth_quantization_step_m > 0:
        step = float(cfg.depth_quantization_step_m)
        out[valid] = np.round(out[valid] / step) * step
    if cfg.depth_dropout_prob > 0:
        drop = np.random.random(out.shape) < cfg.depth_dropout_prob
        out[valid & drop] = 0.0
    if cfg.depth_edge_dropout_prob > 0:
        edge_valid = np.isfinite(out) & (out > 0)
        edge = np.zeros(out.shape, dtype=bool)
        dz_x = np.abs(out[:, 1:] - out[:, :-1])
        dz_y = np.abs(out[1:, :] - out[:-1, :])
        valid_x = edge_valid[:, 1:] & edge_valid[:, :-1]
        valid_y = edge_valid[1:, :] & edge_valid[:-1, :]
        edge[:, 1:] |= valid_x & (dz_x > cfg.depth_edge_threshold_m)
        edge[:, :-1] |= valid_x & (dz_x > cfg.depth_edge_threshold_m)
        edge[1:, :] |= valid_y & (dz_y > cfg.depth_edge_threshold_m)
        edge[:-1, :] |= valid_y & (dz_y > cfg.depth_edge_threshold_m)
        drop_edge = np.random.random(out.shape) < cfg.depth_edge_dropout_prob
        out[edge_valid & edge & drop_edge] = 0.0
    out = _apply_depth_shadow(out, cfg)
    # keep non-positive depths as-is (background can be 0 depending on API)
    out[out < 0] = 0.0
    return out


def _mask_to_polygons_yolo(
    mask_u8: np.ndarray,
    class_id: int,
    *,
    min_area_px: int,
    approx_epsilon_px: float,
    max_polys: int,
) -> List[str]:
    """
    Convert a binary mask to one or more YOLOv8-seg polygon lines.
    """
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    h, w = mask_u8.shape[:2]
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: List[str] = []

    # sort by area, largest first; optionally keep multiple disconnected components
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[: max_polys]
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < float(min_area_px):
            continue
        eps = float(approx_epsilon_px)
        poly = cv2.approxPolyDP(c, epsilon=eps, closed=True)
        # poly shape (N,1,2)
        pts = poly.reshape(-1, 2).astype(np.float32)
        if pts.shape[0] < 3:
            continue

        pts[:, 0] = pts[:, 0] / float(w)
        pts[:, 1] = pts[:, 1] / float(h)
        pts = _clamp01(pts)
        coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts.tolist())
        lines.append(f"{int(class_id)} {coords}")

    return lines


def _rand_uniform(a: float, b: float) -> float:
    return float(a + (b - a) * random.random())


def _rand_from_range(values: Sequence[float]) -> float:
    if len(values) != 2:
        raise ValueError(f"Expected a two-value range, got {values}")
    return _rand_uniform(float(values[0]), float(values[1]))


def _rand_int_from_range(values: Sequence[int]) -> int:
    if len(values) != 2:
        raise ValueError(f"Expected a two-value range, got {values}")
    lo = int(values[0])
    hi = int(values[1])
    if lo > hi:
        lo, hi = hi, lo
    return random.randint(lo, hi)


def _sample_noise_config(args) -> NoiseConfig:
    return NoiseConfig(
        rgb_gaussian_sigma=_rand_from_range(args.rgb_noise_sigma_range),
        rgb_salt_pepper_prob=_rand_from_range(args.rgb_salt_pepper_prob_range),
        rgb_brightness_jitter=_rand_from_range(args.rgb_brightness_jitter_range),
        rgb_contrast_jitter=_rand_from_range(args.rgb_contrast_jitter_range),
        depth_gaussian_sigma_m=_rand_from_range(args.depth_noise_sigma_m_range),
        depth_quantization_step_m=_rand_from_range(args.depth_quant_step_m_range),
        depth_dropout_prob=_rand_from_range(args.depth_dropout_prob_range),
        depth_edge_dropout_prob=_rand_from_range(args.depth_edge_dropout_prob_range),
        depth_edge_threshold_m=_rand_from_range(args.depth_edge_threshold_m_range),
        depth_shadow_enable=bool(args.depth_shadow_enable),
        depth_shadow_direction=str(args.depth_shadow_direction),
        depth_shadow_k_px_m=_rand_from_range(args.depth_shadow_k_px_m_range),
        depth_shadow_min_radius_px=_rand_int_from_range(args.depth_shadow_min_radius_px_range),
        depth_shadow_max_radius_px=_rand_int_from_range(args.depth_shadow_max_radius_px_range),
        depth_shadow_edge_threshold_m=_rand_from_range(args.depth_shadow_edge_threshold_m_range),
        depth_shadow_max_depth_m=_rand_from_range(args.depth_shadow_max_depth_m_range),
    )


def _body_is_descendant_of(model: mujoco.MjModel, body_id: int, ancestor_name: str) -> bool:
    ancestor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ancestor_name)
    if ancestor_id < 0:
        return False
    b = int(body_id)
    while b >= 0:
        if b == ancestor_id:
            return True
        parent = int(model.body_parentid[b])
        if parent == b:
            break
        b = parent
    return False


def _hide_robot_geoms_for_rgb(model: mujoco.MjModel, *, robot_root_body: str, keep_geom_ids: Sequence[int]) -> None:
    keep = set(int(g) for g in keep_geom_ids)
    for gid in range(int(model.ngeom)):
        if gid in keep:
            continue
        body_id = int(model.geom_bodyid[gid])
        if _body_is_descendant_of(model, body_id, robot_root_body):
            model.geom_rgba[gid, 3] = 0.0


def _assign_robot_geoms_to_hidden_group(
    model: mujoco.MjModel,
    *,
    robot_root_body: str,
    keep_geom_ids: Sequence[int],
    hidden_group_id: int = 5,
) -> int:
    if hidden_group_id < 0 or hidden_group_id >= int(mujoco.mjNGROUP):
        raise ValueError(
            f"hidden_group_id must be in [0, {int(mujoco.mjNGROUP) - 1}], got {hidden_group_id}."
        )
    keep = set(int(g) for g in keep_geom_ids)
    changed = 0
    for gid in range(int(model.ngeom)):
        if gid in keep:
            continue
        body_id = int(model.geom_bodyid[gid])
        if _body_is_descendant_of(model, body_id, robot_root_body):
            model.geom_group[gid] = hidden_group_id
            changed += 1
    return changed


def _sample_non_overlapping_xy(
    n: int,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    min_dist_m: float,
    max_tries: int = 4000,
) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for _ in range(max_tries):
        if len(points) >= n:
            break
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        ok = True
        for px, py in points:
            if math.hypot(x - px, y - py) < min_dist_m:
                ok = False
                break
        if ok:
            points.append((x, y))
    if len(points) != n:
        raise RuntimeError(
            f"Could not sample {n} non-overlapping points in bounds "
            f"x[{x_min},{x_max}], y[{y_min},{y_max}] with min_dist={min_dist_m}."
        )
    return points


def _rand_color_rgba(alpha: float = 1.0) -> np.ndarray:
    # Slight bias toward non-dark colors (helps segmentation/learning)
    r = _rand_uniform(0.15, 1.0)
    g = _rand_uniform(0.15, 1.0)
    b = _rand_uniform(0.15, 1.0)
    return np.array([r, g, b, float(alpha)], dtype=np.float32)


def _set_freejoint_qpos(data: mujoco.MjData, body_id: int, pos_xyz: Sequence[float], yaw_rad: float) -> None:
    """
    Place a free-jointed body by writing qpos for the joint attached to that body.
    Assumes body has exactly one free joint as its first joint (common for movable props).
    """
    jadr = data.joint(body_id).qposadr  # type: ignore[attr-defined]
    # The above accessor might not exist depending on mujoco Python version.
    # So we use the robust low-level arrays below.
    # Find first joint belonging to body.
    j0 = mujoco.mj_bodyid2jid(data.model, body_id)  # type: ignore[attr-defined]
    _ = jadr  # keep linters quiet
    raise RuntimeError(
        "This MuJoCo Python build does not expose body->joint helper. "
        "Use --freejoint-joint-name to explicitly name the free joint for the object."
    )


def _set_freejoint_by_name(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    pos_xyz,
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> None:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint '{joint_name}' not found.")
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Joint '{joint_name}' is not a FREE joint.")
    _set_freejoint_by_joint_id(model, data, jid, pos_xyz, roll_rad, pitch_rad, yaw_rad)


def _set_freejoint_by_body_name(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    pos_xyz: Sequence[float],
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> None:
    """Set the first joint of ``body_name`` (expects a single FREE joint, unnamed in MJCF is ok)."""
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body '{body_name}' not found.")
    jid = int(model.body_jntadr[bid])
    if jid < 0:
        raise ValueError(f"Body '{body_name}' has no joint.")
    if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Body '{body_name}' first joint is not FREE (type={model.jnt_type[jid]}).")
    _set_freejoint_by_joint_id(model, data, jid, pos_xyz, roll_rad, pitch_rad, yaw_rad)


def _set_freejoint_by_joint_id(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    jid: int,
    pos_xyz: Sequence[float],
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float,
) -> None:
    qadr = int(model.jnt_qposadr[jid])
    x, y, z = [float(v) for v in pos_xyz]
    euler = np.array([float(roll_rad), float(pitch_rad), float(yaw_rad)], dtype=np.float64)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_euler2Quat(quat, euler, "xyz")
    data.qpos[qadr : qadr + 7] = np.array([x, y, z, quat[0], quat[1], quat[2], quat[3]], dtype=np.float64)
    vadr = int(model.jnt_dofadr[jid])
    data.qvel[vadr : vadr + 6] = 0.0


def _simulate_settling(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    seconds: float,
    max_steps: int,
) -> None:
    """
    Integrate with mj_step until simulation time advances by ``seconds``.
    Call after mj_forward so contacts are warm-started.
    """
    if seconds <= 0.0:
        mujoco.mj_forward(model, data)
        return
    t_end = float(data.time) + float(seconds)
    steps = 0
    while float(data.time) < t_end - 1e-9:
        if steps >= max_steps:
            raise RuntimeError(
                f"Settling exceeded --settle-max-steps ({max_steps}) before reaching "
                f"{seconds} s of sim time (current time={data.time:.4g}, target={t_end:.4g}). "
                "Increase --settle-max-steps or check model timestep / instabilities."
            )
        mujoco.mj_step(model, data)
        steps += 1
    mujoco.mj_forward(model, data)


def _randomize_lights(
    model: mujoco.MjModel,
    *,
    diffuse_range: Tuple[float, float],
    ambient_range: Tuple[float, float],
    specular_range: Tuple[float, float],
    z_range: Tuple[float, float],
    headlight_scale: float,
) -> None:
    # MJCF lights are not added here; existing lights are overwritten with
    # dataset-specific ranges so the same MJCF can stay bright for simulation.
    model.vis.headlight.diffuse[:] = np.asarray(model.vis.headlight.diffuse) * float(headlight_scale)
    model.vis.headlight.ambient[:] = np.asarray(model.vis.headlight.ambient) * float(headlight_scale)
    model.vis.headlight.specular[:] = np.asarray(model.vis.headlight.specular) * float(headlight_scale)

    if model.nlight <= 0:
        return
    for i in range(model.nlight):
        # position in world (if light is not attached) or in body frame (if attached)
        model.light_pos[i, 0] = _rand_uniform(-1.0, 1.0)
        model.light_pos[i, 1] = _rand_uniform(-1.0, 1.0)
        model.light_pos[i, 2] = _rand_uniform(*z_range)
        model.light_dir[i, 0] = _rand_uniform(-0.5, 0.5)
        model.light_dir[i, 1] = _rand_uniform(-0.5, 0.5)
        model.light_dir[i, 2] = _rand_uniform(-1.0, -0.2)
        model.light_diffuse[i, :3] = np.array([_rand_uniform(*diffuse_range)] * 3, dtype=np.float32)
        model.light_ambient[i, :3] = np.array([_rand_uniform(*ambient_range)] * 3, dtype=np.float32)
        model.light_specular[i, :3] = np.array([_rand_uniform(*specular_range)] * 3, dtype=np.float32)


def _randomize_geom_appearance(
    model: mujoco.MjModel,
    geom_ids: Sequence[int],
    *,
    randomize_color: bool,
    material_ids: Optional[Sequence[int]],
    randomize_surface: bool,
    mat_specular_range: Tuple[float, float],
    mat_shininess_range: Tuple[float, float],
    mat_reflectance_range: Tuple[float, float],
) -> None:
    randomized_material_ids = set()
    for gid in geom_ids:
        if randomize_color:
            model.geom_rgba[gid, :] = _rand_color_rgba(alpha=1.0)
        if material_ids:
            model.geom_matid[gid] = int(random.choice(list(material_ids)))
        if randomize_surface:
            matid = int(model.geom_matid[gid])
            if matid >= 0 and matid not in randomized_material_ids:
                model.mat_specular[matid] = _rand_uniform(*mat_specular_range)
                model.mat_shininess[matid] = _rand_uniform(*mat_shininess_range)
                model.mat_reflectance[matid] = _rand_uniform(*mat_reflectance_range)
                randomized_material_ids.add(matid)


def _make_free_camera(
    lookat: Sequence[float],
    distance: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = np.array(lookat, dtype=np.float64)
    cam.distance = float(distance)
    cam.azimuth = float(azimuth_deg)
    cam.elevation = float(elevation_deg)
    return cam


def _camera_head_position(data: mujoco.MjData, cam: mujoco.MjvCamera) -> np.ndarray:
    """World position of the free camera (matches MuJoCo viewer / Renderer)."""
    head = np.zeros(3, dtype=np.float64)
    forward = np.zeros(3, dtype=np.float64)
    up = np.zeros(3, dtype=np.float64)
    right = np.zeros(3, dtype=np.float64)
    mujoco.mjv_cameraFrame(head, forward, up, right, data, cam)
    return head.copy()


def _min_dist_camera_to_geoms(data: mujoco.MjData, cam: mujoco.MjvCamera, geom_ids: Sequence[int]) -> float:
    p = _camera_head_position(data, cam)
    best = float("inf")
    for gid in geom_ids:
        g = data.geom_xpos[int(gid), :3]
        dx = float(g[0] - p[0])
        dy = float(g[1] - p[1])
        dz = float(g[2] - p[2])
        best = min(best, math.sqrt(dx * dx + dy * dy + dz * dz))
    return best


def _sample_camera_meeting_min_dist(
    data: mujoco.MjData,
    geom_ids: Sequence[int],
    *,
    lookat: np.ndarray,
    cam_distance: Tuple[float, float],
    cam_azimuth_deg: Tuple[float, float],
    cam_elevation_deg: Tuple[float, float],
    min_dist_m: float,
    max_tries: int,
) -> mujoco.MjvCamera:
    if min_dist_m <= 0:
        return _make_free_camera(
            lookat,
            _rand_uniform(*cam_distance),
            _rand_uniform(*cam_azimuth_deg),
            _rand_uniform(*cam_elevation_deg),
        )
    for _ in range(max_tries):
        cam = _make_free_camera(
            lookat,
            _rand_uniform(*cam_distance),
            _rand_uniform(*cam_azimuth_deg),
            _rand_uniform(*cam_elevation_deg),
        )
        if _min_dist_camera_to_geoms(data, cam, geom_ids) >= min_dist_m - 1e-9:
            return cam
    raise RuntimeError(
        f"Could not sample a free camera with distance >= {min_dist_m} m to all target geoms "
        f"after {max_tries} tries. Widen --cam-distance / angles, lower --cam-min-dist-m, or "
        "move the object with --pos-x/--pos-y/--pos-z."
    )


def _sample_lookat_with_frame_placement(
    object_centroid: np.ndarray,
    *,
    placement: str,
    offcenter_frac: float,
    corner_frac: float,
    offset_xy_m: float,
) -> np.ndarray:
    """
    Choose free-camera look-at. Tracking the object centroid centers it in the image; a tangential
    world XY offset moves the object toward margins and corners (training diversity).
    """
    centroid = np.asarray(object_centroid, dtype=np.float64).copy()
    mode = str(placement).strip().lower()
    if mode == "center":
        return centroid

    max_off = float(max(0.0, offset_xy_m))
    if max_off <= 0.0:
        return centroid

    use_offcenter = mode == "corners"
    if mode == "mixed":
        use_offcenter = random.random() < float(max(0.0, min(1.0, offcenter_frac)))
    if not use_offcenter:
        return centroid

    use_corner = mode == "corners"
    if mode == "mixed":
        use_corner = random.random() < float(max(0.0, min(1.0, corner_frac)))

    if use_corner:
        sx = random.choice([-1.0, 1.0])
        sy = random.choice([-1.0, 1.0])
        if random.random() < 0.2:
            if random.random() < 0.5:
                sx = 0.0
            else:
                sy = 0.0
        mag = random.uniform(0.55, 1.0)
        offset = np.array([sx * mag * max_off, sy * mag * max_off, 0.0], dtype=np.float64)
    else:
        offset = np.array(
            [
                random.uniform(-max_off, max_off),
                random.uniform(-max_off, max_off),
                0.0,
            ],
            dtype=np.float64,
        )
    return centroid + offset


def _render(
    renderer: "mujoco.Renderer",
    data: mujoco.MjData,
    cam: mujoco.MjvCamera,
    *,
    want_depth: bool,
    want_seg: bool,
    scene_option: Optional[mujoco.MjvOption] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (rgb_u8, depth_m, seg) where seg is int32 ID image if supported.
    """
    renderer.update_scene(data, camera=cam, scene_option=scene_option)

    rgb = renderer.render()
    # renderer.render() returns uint8 RGB by default
    rgb_u8 = np.asarray(rgb, dtype=np.uint8)

    depth_m = None
    if want_depth:
        # Support both newer API (render(depth=True)) and older API
        # (enable_depth_rendering()/render()/disable_depth_rendering()).
        try:
            depth = renderer.render(depth=True)
        except TypeError:
            enable_depth = getattr(renderer, "enable_depth_rendering", None)
            disable_depth = getattr(renderer, "disable_depth_rendering", None)
            if callable(enable_depth):
                enable_depth()
                try:
                    depth = renderer.render()
                finally:
                    if callable(disable_depth):
                        disable_depth()
            else:
                raise RuntimeError(
                    "Your mujoco Python Renderer does not support depth capture with "
                    "render(depth=True) or enable_depth_rendering()."
                )
        depth_m = np.asarray(depth, dtype=np.float32)

    seg = None
    if want_seg:
        try:
            seg_img = renderer.render(segmentation=True)
            seg = np.asarray(seg_img, dtype=np.int32)
        except TypeError:
            enable_seg = getattr(renderer, "enable_segmentation_rendering", None)
            disable_seg = getattr(renderer, "disable_segmentation_rendering", None)
            if callable(enable_seg):
                enable_seg()
                try:
                    seg_img = renderer.render()
                finally:
                    if callable(disable_seg):
                        disable_seg()
                seg = np.asarray(seg_img, dtype=np.int32)
            else:
                raise RuntimeError(
                    "Your mujoco Python Renderer does not support segmentation=True "
                    "or enable_segmentation_rendering(). Upgrade mujoco Python or "
                    "use a custom ID render pass."
                )

    return rgb_u8, depth_m, seg


def _save_depth_png_16u(depth_m: np.ndarray, path: Path, scale: float = 1000.0) -> None:
    """
    Save depth in millimeters as 16-bit PNG.
    """
    d = depth_m.copy()
    d_mm = np.clip(d * float(scale), 0, 65535).astype(np.uint16)
    cv2.imwrite(str(path), d_mm)


def _write_dataset_yaml(out_dir: Path, class_names: List[str]) -> None:
    dataset_root = out_dir.expanduser().resolve()
    yaml = (
        f"path: {dataset_root}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(class_names)}\n"
        f"names: {json.dumps(class_names)}\n"
    )
    (out_dir / "dataset.yaml").write_text(yaml)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to MJCF/XML model file.")
    ap.add_argument("--out", required=True, help="Output dataset directory.")
    ap.add_argument("--n", type=int, default=2000, help="Total images to generate.")
    ap.add_argument("--val-split", type=float, default=0.1, help="Fraction for validation.")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)

    ap.add_argument("--object-geom-names", nargs="+", required=True, help="Geom names to segment (object).")
    ap.add_argument("--class-name", default="object", help="Single class name for YOLO (used when --class-map is omitted).")
    ap.add_argument(
        "--class-map",
        nargs="*",
        default=None,
        help=(
            "Optional per-geom class mapping entries: '<geom_name>:<class_name>'. "
            "Example: cylinder_1_vis:Cylinder_1 cylinder_2_vis:Cylinder_2 square_1_vis:Box_1"
        ),
    )

    ap.add_argument(
        "--freejoint-joint-name",
        default=None,
        help="Name of FREE joint to reposition object (mutually exclusive with --freejoint-body-name).",
    )
    ap.add_argument(
        "--freejoint-body-name",
        default=None,
        help="Body name whose first joint is FREE (use for ur3_scene_table props; joints are often unnamed).",
    )
    ap.add_argument(
        "--freejoint-body-names",
        nargs="*",
        default=None,
        help=(
            "Optional list of FREE body names to randomize together in one scene "
            "(e.g. cylinder_1_obj cylinder_2_obj square_1_obj)."
        ),
    )
    ap.add_argument(
        "--pos-x",
        type=float,
        nargs=2,
        default=[-0.45, -0.15],
        help="Uniform range for x (UR3 table workspace; override for other scenes).",
    )
    ap.add_argument(
        "--pos-y",
        type=float,
        nargs=2,
        default=[-0.45, -0.15],
        help="Uniform range for y (UR3 table workspace; override for other scenes).",
    )
    ap.add_argument(
        "--pos-z",
        type=float,
        nargs=2,
        default=[0.9, 0.9],
        help="Initial spawn z (world) for the free object before gravity settles (e.g. 0.9 m above table).",
    )
    ap.add_argument(
        "--settle-time-s",
        type=float,
        default=5.0,
        help="Simulated seconds (mj_step) to run after placing the object before rendering. Use 0 to skip.",
    )
    ap.add_argument(
        "--settle-max-steps",
        type=int,
        default=500_000,
        help="Safety cap on mj_step calls during settling (prevents infinite loops if unstable).",
    )
    ap.add_argument(
        "--min-body-spacing-m",
        type=float,
        default=0.07,
        help="Minimum XY spacing between random poses when using --freejoint-body-names.",
    )
    ap.add_argument("--roll", type=float, nargs=2, default=[-math.pi, math.pi], help="Uniform range for roll (rad), euler xyz.")
    ap.add_argument("--pitch", type=float, nargs=2, default=[-math.pi, math.pi], help="Uniform range for pitch (rad), euler xyz.")
    ap.add_argument("--yaw", type=float, nargs=2, default=[-math.pi, math.pi], help="Uniform range for yaw (rad), euler xyz.")

    ap.add_argument("--randomize-color", action="store_true", help="Randomize object geom RGBA.")
    ap.add_argument("--material-ids", nargs="*", type=int, default=None, help="Optional list of material IDs to sample.")
    ap.add_argument(
        "--randomize-surface",
        action="store_true",
        help="Randomize material surface properties (specular, shininess, reflectance) for target geoms.",
    )
    ap.add_argument(
        "--mat-specular-range",
        type=float,
        nargs=2,
        default=[0.6, 1.0],
        help="Uniform range for MuJoCo material specular term.",
    )
    ap.add_argument(
        "--mat-shininess-range",
        type=float,
        nargs=2,
        default=[0.4, 1.0],
        help="Uniform range for MuJoCo material shininess term.",
    )
    ap.add_argument(
        "--mat-reflectance-range",
        type=float,
        nargs=2,
        default=[0.4, 0.9],
        help="Uniform range for MuJoCo material reflectance term.",
    )
    ap.add_argument("--randomize-lighting", action="store_true", help="Overwrite/randomize existing MuJoCo lights.")
    ap.add_argument(
        "--light-diffuse-range",
        type=float,
        nargs=2,
        default=[0.15, 0.9],
        help="Uniform range for existing light diffuse intensity. Lower values create darker images.",
    )
    ap.add_argument(
        "--light-ambient-range",
        type=float,
        nargs=2,
        default=[0.0, 0.18],
        help="Uniform range for existing light ambient intensity.",
    )
    ap.add_argument(
        "--light-specular-range",
        type=float,
        nargs=2,
        default=[0.0, 0.45],
        help="Uniform range for existing light specular intensity.",
    )
    ap.add_argument(
        "--light-z-range",
        type=float,
        nargs=2,
        default=[0.4, 1.6],
        help="Uniform range for existing light z positions.",
    )
    ap.add_argument(
        "--headlight-scale",
        type=float,
        default=0.25,
        help="Scale MuJoCo visual headlight during dataset rendering. Use 0 to disable it.",
    )

    ap.add_argument(
        "--cam-lookat",
        type=float,
        nargs=3,
        default=[-0.36, -0.28, 0.79],
        help="Free-camera look-at (world) when --cam-fixed-lookat is set; otherwise ignored if tracking target.",
    )
    ap.add_argument(
        "--cam-distance",
        type=float,
        nargs=2,
        default=[0.35, 1.15],
        help="Distance from look-at to camera (m); keep min >= --cam-min-dist-m when tracking the target.",
    )
    ap.add_argument("--cam-azimuth-deg", type=float, nargs=2, default=[-180.0, 180.0])
    ap.add_argument("--cam-elevation-deg", type=float, nargs=2, default=[-55.0, -15.0])
    ap.add_argument(
        "--cam-min-dist-m",
        type=float,
        default=0.30,
        help="Minimum straight-line distance (m) from camera to each target geom center (RealSense ~0.3 m). Use 0 to disable.",
    )
    ap.add_argument(
        "--cam-sample-max-tries",
        type=int,
        default=400,
        help="Max random camera samples per image when enforcing --cam-min-dist-m.",
    )
    ap.add_argument(
        "--cam-fixed-lookat",
        action="store_true",
        help="If set, use --cam-lookat instead of the centroid of --object-geom-names.",
    )
    ap.add_argument(
        "--cam-frame-placement",
        choices=["center", "mixed", "corners"],
        default="mixed",
        help=(
            "Image framing: center=always look at object centroid; mixed=~half centered and half "
            "off-center; corners=strong corner/margin bias. Ignored when --cam-fixed-lookat is set."
        ),
    )
    ap.add_argument(
        "--cam-frame-offcenter-frac",
        type=float,
        default=0.55,
        help="For mixed: fraction of frames with decentered look-at (object not in image center).",
    )
    ap.add_argument(
        "--cam-frame-corner-frac",
        type=float,
        default=0.45,
        help="For mixed off-center frames: fraction biased toward image corners/edges.",
    )
    ap.add_argument(
        "--cam-lookat-offset-xy-m",
        type=float,
        default=0.10,
        help="Max world-frame XY look-at offset from object centroid (larger => nearer image corners).",
    )

    ap.add_argument(
        "--rgb-noise-sigma-range",
        type=float,
        nargs=2,
        default=[0.0, 6.0],
        help="Per-image random Gaussian RGB sigma range in [0..255] space.",
    )
    ap.add_argument(
        "--rgb-salt-pepper-prob-range",
        type=float,
        nargs=2,
        default=[0.0, 0.006],
        help="Per-image random probability for RGB salt/pepper pixels.",
    )
    ap.add_argument(
        "--rgb-brightness-jitter-range",
        type=float,
        nargs=2,
        default=[0.0, 10.0],
        help="Per-image random max additive brightness jitter in [0..255] space.",
    )
    ap.add_argument(
        "--rgb-contrast-jitter-range",
        type=float,
        nargs=2,
        default=[0.0, 0.10],
        help="Per-image random max contrast jitter around 1.0.",
    )
    ap.add_argument("--depth-noise-sigma-m-range", type=float, nargs=2, default=[0.0, 0.004])
    ap.add_argument("--depth-quant-step-m-range", type=float, nargs=2, default=[0.0005, 0.002])
    ap.add_argument("--depth-dropout-prob-range", type=float, nargs=2, default=[0.0, 0.04])
    ap.add_argument(
        "--depth-edge-dropout-prob-range",
        type=float,
        nargs=2,
        default=[0.03, 0.18],
        help="Per-image random probability of invalidating pixels near depth discontinuities.",
    )
    ap.add_argument(
        "--depth-edge-threshold-m-range",
        type=float,
        nargs=2,
        default=[0.006, 0.018],
        help="Per-image random depth jump threshold used to find boundary pixels.",
    )
    ap.add_argument(
        "--depth-shadow-enable",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply D435i-style horizontal depth occlusion shadows.",
    )
    ap.add_argument(
        "--depth-shadow-direction",
        choices=["left", "right"],
        default="left",
        help="Horizontal side where aligned-depth occlusion shadows are invalidated.",
    )
    ap.add_argument(
        "--depth-shadow-k-px-m-range",
        type=float,
        nargs=2,
        default=[0.8, 2.2],
        help="Per-image random K in radius_px = K / depth_m for depth shadows.",
    )
    ap.add_argument(
        "--depth-shadow-min-radius-px-range",
        type=int,
        nargs=2,
        default=[1, 2],
        help="Per-image random minimum depth shadow radius in pixels.",
    )
    ap.add_argument(
        "--depth-shadow-max-radius-px-range",
        type=int,
        nargs=2,
        default=[5, 10],
        help="Per-image random maximum depth shadow radius in pixels.",
    )
    ap.add_argument(
        "--depth-shadow-edge-threshold-m-range",
        type=float,
        nargs=2,
        default=[0.01, 0.025],
        help="Per-image random horizontal depth jump threshold for shadow edges.",
    )
    ap.add_argument(
        "--depth-shadow-max-depth-m-range",
        type=float,
        nargs=2,
        default=[4.0, 4.0],
        help="Per-image random maximum foreground depth that can create a depth shadow. Use <=0 to disable.",
    )
    # Backward-compatible fixed-value aliases. If used, they become [value, value] ranges.
    ap.add_argument("--rgb-noise-sigma", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--depth-noise-sigma-m", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--depth-quant-step-m", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--save-depth", action="store_true", help="Also save depth PNGs (16-bit mm).")
    ap.add_argument(
        "--hide-robot-in-rgb",
        action="store_true",
        help="Hide UR3 meshes in rendered RGB images for dataset generation (physics unchanged).",
    )
    ap.add_argument(
        "--hide-robot-in-all-renders",
        action="store_true",
        help=(
            "Exclude UR3 geoms from scene rendering (RGB + depth + segmentation) "
            "using a hidden geom group; physics unchanged."
        ),
    )
    ap.add_argument(
        "--robot-root-body",
        default="base_link",
        help="Robot root body name used when --hide-robot-in-rgb is enabled.",
    )

    ap.add_argument("--min-mask-area-px", type=int, default=200)
    ap.add_argument("--poly-eps-px", type=float, default=2.0, help="Contour approx epsilon (pixels).")
    ap.add_argument("--max-polys", type=int, default=2, help="Keep up to N disconnected components per image.")

    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.rgb_noise_sigma is not None:
        args.rgb_noise_sigma_range = [float(args.rgb_noise_sigma), float(args.rgb_noise_sigma)]
    if args.depth_noise_sigma_m is not None:
        args.depth_noise_sigma_m_range = [float(args.depth_noise_sigma_m), float(args.depth_noise_sigma_m)]
    if args.depth_quant_step_m is not None:
        args.depth_quant_step_m_range = [float(args.depth_quant_step_m), float(args.depth_quant_step_m)]

    freejoint_mode_count = int(bool(args.freejoint_joint_name)) + int(bool(args.freejoint_body_name)) + int(
        bool(args.freejoint_body_names)
    )
    if freejoint_mode_count > 1:
        raise SystemExit(
            "Use only one of --freejoint-joint-name, --freejoint-body-name, or --freejoint-body-names."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)

    model_path = Path(args.model).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()

    if out_dir.exists():
        _mkdir_clean(out_dir)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    # Create folder structure
    for split in ["train", "val"]:
        _mkdir_clean(out_dir / "images" / split)
        _mkdir_clean(out_dir / "labels" / split)
        if args.save_depth:
            _mkdir_clean(out_dir / "depth" / split)

    # Load model
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Resolve object geom IDs
    geom_ids: List[int] = []
    for gn in args.object_geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gn)
        if gid < 0:
            raise ValueError(f"Geom '{gn}' not found in model.")
        geom_ids.append(int(gid))

    if args.hide_robot_in_rgb:
        _hide_robot_geoms_for_rgb(
            model,
            robot_root_body=str(args.robot_root_body),
            keep_geom_ids=geom_ids,
        )

    render_option: Optional[mujoco.MjvOption] = None
    if args.hide_robot_in_all_renders:
        hidden_group_id = 5
        hidden_count = _assign_robot_geoms_to_hidden_group(
            model,
            robot_root_body=str(args.robot_root_body),
            keep_geom_ids=geom_ids,
            hidden_group_id=hidden_group_id,
        )
        render_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(render_option)
        render_option.geomgroup[:] = 1
        render_option.geomgroup[hidden_group_id] = 0
        if hidden_count == 0:
            print(
                f"Warning: --hide-robot-in-all-renders enabled but no geoms found under "
                f"robot root body '{args.robot_root_body}'."
            )

    # Build geom -> class mapping.
    geom_to_class_name: Dict[str, str] = {}
    if args.class_map:
        for entry in args.class_map:
            if ":" not in entry:
                raise ValueError(
                    f"Invalid --class-map entry '{entry}'. Expected format '<geom_name>:<class_name>'."
                )
            geom_name, class_name = entry.split(":", 1)
            geom_name = geom_name.strip()
            class_name = class_name.strip()
            if not geom_name or not class_name:
                raise ValueError(f"Invalid --class-map entry '{entry}': empty geom or class.")
            geom_to_class_name[geom_name] = class_name

        missing = [gn for gn in args.object_geom_names if gn not in geom_to_class_name]
        extra = [gn for gn in geom_to_class_name if gn not in args.object_geom_names]
        if missing:
            raise ValueError(
                "Missing class mapping for geoms in --object-geom-names: " + ", ".join(sorted(missing))
            )
        if extra:
            raise ValueError(
                "Found --class-map geoms not present in --object-geom-names: " + ", ".join(sorted(extra))
            )
    else:
        for gn in args.object_geom_names:
            geom_to_class_name[gn] = str(args.class_name)

    # Preserve appearance order from object-geom-names.
    class_names: List[str] = []
    class_name_to_id: Dict[str, int] = {}
    geom_id_to_class_id: Dict[int, int] = {}
    for gn, gid in zip(args.object_geom_names, geom_ids):
        cname = geom_to_class_name[gn]
        if cname not in class_name_to_id:
            class_name_to_id[cname] = len(class_names)
            class_names.append(cname)
        geom_id_to_class_id[gid] = class_name_to_id[cname]

    # Renderer
    renderer = mujoco.Renderer(model, height=int(args.height), width=int(args.width))

    _write_dataset_yaml(out_dir, class_names)

    n_total = int(args.n)
    n_val = int(round(n_total * float(args.val_split)))
    n_train = n_total - n_val
    split_for_idx = ["train"] * n_train + ["val"] * n_val
    random.shuffle(split_for_idx)

    meta = {
        "model": str(model_path),
        "n_total": n_total,
        "splits": {"train": n_train, "val": n_val},
        "object_geom_names": args.object_geom_names,
        "class_names": class_names,
        "geom_to_class_name": geom_to_class_name,
        "noise_ranges": {
            "rgb_noise_sigma_range": list(args.rgb_noise_sigma_range),
            "rgb_salt_pepper_prob_range": list(args.rgb_salt_pepper_prob_range),
            "rgb_brightness_jitter_range": list(args.rgb_brightness_jitter_range),
            "rgb_contrast_jitter_range": list(args.rgb_contrast_jitter_range),
            "depth_noise_sigma_m_range": list(args.depth_noise_sigma_m_range),
            "depth_quant_step_m_range": list(args.depth_quant_step_m_range),
            "depth_dropout_prob_range": list(args.depth_dropout_prob_range),
            "depth_edge_dropout_prob_range": list(args.depth_edge_dropout_prob_range),
            "depth_edge_threshold_m_range": list(args.depth_edge_threshold_m_range),
            "depth_shadow_enable": bool(args.depth_shadow_enable),
            "depth_shadow_direction": str(args.depth_shadow_direction),
            "depth_shadow_k_px_m_range": list(args.depth_shadow_k_px_m_range),
            "depth_shadow_min_radius_px_range": list(args.depth_shadow_min_radius_px_range),
            "depth_shadow_max_radius_px_range": list(args.depth_shadow_max_radius_px_range),
            "depth_shadow_edge_threshold_m_range": list(args.depth_shadow_edge_threshold_m_range),
            "depth_shadow_max_depth_m_range": list(args.depth_shadow_max_depth_m_range),
        },
        "seed": int(args.seed),
        "camera": {
            "min_dist_m": float(args.cam_min_dist_m),
            "sample_max_tries": int(args.cam_sample_max_tries),
            "fixed_lookat": bool(args.cam_fixed_lookat),
            "lookat_default_xyz": list(args.cam_lookat),
            "distance_range": [float(args.cam_distance[0]), float(args.cam_distance[1])],
            "frame_placement": str(args.cam_frame_placement),
            "frame_offcenter_frac": float(args.cam_frame_offcenter_frac),
            "frame_corner_frac": float(args.cam_frame_corner_frac),
            "lookat_offset_xy_m": float(args.cam_lookat_offset_xy_m),
        },
        "settle": {
            "time_s": float(args.settle_time_s),
            "max_steps": int(args.settle_max_steps),
            "timestep_s": float(model.opt.timestep),
        },
    }
    (out_dir / "generation_meta.json").write_text(json.dumps(meta, indent=2))

    sample_noise_meta = []
    for idx in range(n_total):
        split = split_for_idx[idx]
        noise_cfg = _sample_noise_config(args)

        # Reset state each sample so scenes do not depend on previous rollouts.
        mujoco.mj_resetData(model, data)

        # Randomize object pose (free joint)
        randomized_freejoint = False
        if args.freejoint_joint_name:
            x = _rand_uniform(*args.pos_x)
            y = _rand_uniform(*args.pos_y)
            z = float(args.pos_z[0])
            roll = _rand_uniform(*args.roll)
            pitch = _rand_uniform(*args.pitch)
            yaw = _rand_uniform(*args.yaw)
            _set_freejoint_by_name(model, data, args.freejoint_joint_name, (x, y, z), roll, pitch, yaw)
            randomized_freejoint = True
        elif args.freejoint_body_name:
            x = _rand_uniform(*args.pos_x)
            y = _rand_uniform(*args.pos_y)
            z = float(args.pos_z[0])
            roll = _rand_uniform(*args.roll)
            pitch = _rand_uniform(*args.pitch)
            yaw = _rand_uniform(*args.yaw)
            _set_freejoint_by_body_name(model, data, args.freejoint_body_name, (x, y, z), roll, pitch, yaw)
            randomized_freejoint = True
        elif args.freejoint_body_names:
            body_names = [str(bn) for bn in args.freejoint_body_names]
            sampled_xy = _sample_non_overlapping_xy(
                n=len(body_names),
                x_min=float(args.pos_x[0]),
                x_max=float(args.pos_x[1]),
                y_min=float(args.pos_y[0]),
                y_max=float(args.pos_y[1]),
                min_dist_m=float(args.min_body_spacing_m),
            )
            for body_name, (x, y) in zip(body_names, sampled_xy):
                z = float(args.pos_z[0])
                roll = _rand_uniform(*args.roll)
                pitch = _rand_uniform(*args.pitch)
                yaw = _rand_uniform(*args.yaw)
                _set_freejoint_by_body_name(model, data, body_name, (x, y, z), roll, pitch, yaw)
            randomized_freejoint = True

        # Randomize appearance + lighting
        if args.randomize_color or args.material_ids or args.randomize_surface:
            _randomize_geom_appearance(
                model,
                geom_ids,
                randomize_color=bool(args.randomize_color),
                material_ids=list(args.material_ids) if args.material_ids else None,
                randomize_surface=bool(args.randomize_surface),
                mat_specular_range=(float(args.mat_specular_range[0]), float(args.mat_specular_range[1])),
                mat_shininess_range=(float(args.mat_shininess_range[0]), float(args.mat_shininess_range[1])),
                mat_reflectance_range=(float(args.mat_reflectance_range[0]), float(args.mat_reflectance_range[1])),
            )
        if args.randomize_lighting:
            _randomize_lights(
                model,
                diffuse_range=(float(args.light_diffuse_range[0]), float(args.light_diffuse_range[1])),
                ambient_range=(float(args.light_ambient_range[0]), float(args.light_ambient_range[1])),
                specular_range=(float(args.light_specular_range[0]), float(args.light_specular_range[1])),
                z_range=(float(args.light_z_range[0]), float(args.light_z_range[1])),
                headlight_scale=float(args.headlight_scale),
            )

        # Warm-start contacts, then let the object fall and settle before rendering.
        mujoco.mj_forward(model, data)
        if float(args.settle_time_s) > 0.0:
            if randomized_freejoint:
                _simulate_settling(
                    model,
                    data,
                    seconds=float(args.settle_time_s),
                    max_steps=int(args.settle_max_steps),
                )
            elif idx == 0:
                print(
                    "Warning: --settle-time-s > 0 but no --freejoint-joint-name / --freejoint-body-name; "
                    "skipping physics settling (nothing is being dropped each frame)."
                )

        if args.cam_fixed_lookat:
            lookat_arr = np.array(args.cam_lookat, dtype=np.float64)
        else:
            object_centroid = np.mean(
                np.stack([data.geom_xpos[gid, :3].copy() for gid in geom_ids], axis=0),
                axis=0,
            )
            lookat_arr = _sample_lookat_with_frame_placement(
                object_centroid,
                placement=str(args.cam_frame_placement),
                offcenter_frac=float(args.cam_frame_offcenter_frac),
                corner_frac=float(args.cam_frame_corner_frac),
                offset_xy_m=float(args.cam_lookat_offset_xy_m),
            )

        cam = _sample_camera_meeting_min_dist(
            data,
            geom_ids,
            lookat=lookat_arr,
            cam_distance=(float(args.cam_distance[0]), float(args.cam_distance[1])),
            cam_azimuth_deg=(float(args.cam_azimuth_deg[0]), float(args.cam_azimuth_deg[1])),
            cam_elevation_deg=(float(args.cam_elevation_deg[0]), float(args.cam_elevation_deg[1])),
            min_dist_m=float(args.cam_min_dist_m),
            max_tries=int(args.cam_sample_max_tries),
        )

        rgb_u8, depth_m, seg = _render(
            renderer,
            data,
            cam,
            want_depth=args.save_depth,
            want_seg=True,
            scene_option=render_option,
        )

        # Create masks by selecting requested geom IDs in seg map.
        # MuJoCo segmentation encoding differs across versions; for mujoco.Renderer(segmentation=True),
        # seg usually encodes (objtype, objid). Here we handle common cases:
        # - If seg has shape (H,W,2): [objtype, objid]
        # - If seg has shape (H,W): packed int (not handled here)
        mask_by_class_id: Dict[int, np.ndarray] = {}
        if seg.ndim == 3 and seg.shape[2] >= 2:
            # MuJoCo Renderer segmentation channel order differs by version:
            # some versions return [objtype, objid], others return [objid, objtype].
            geom_objtype = int(mujoco.mjtObj.mjOBJ_GEOM)
            ch0 = seg[..., 0]
            ch1 = seg[..., 1]
            target_geom_ids = np.array(geom_ids, dtype=np.int32)
            if np.any((ch0 == geom_objtype) & np.isin(ch1, target_geom_ids)):
                objtype = ch0
                objid = ch1
            elif np.any((ch1 == geom_objtype) & np.isin(ch0, target_geom_ids)):
                objtype = ch1
                objid = ch0
            else:
                objtype = ch0
                objid = ch1

            is_geom = objtype == geom_objtype
            for gid in geom_ids:
                class_id = geom_id_to_class_id[gid]
                is_this_geom = is_geom & (objid == int(gid))
                if class_id not in mask_by_class_id:
                    mask_by_class_id[class_id] = is_this_geom.astype(np.uint8) * 255
                else:
                    # Merge geoms that belong to the same class.
                    mask_by_class_id[class_id] = np.maximum(
                        mask_by_class_id[class_id], is_this_geom.astype(np.uint8) * 255
                    )
        else:
            raise RuntimeError(
                f"Unsupported segmentation array shape: {seg.shape}. "
                "Expected (H,W,2) with [objtype,objid]."
            )

        # Add per-image sensor noise after segmentation labels are rendered.
        rgb_u8 = _apply_rgb_noise(rgb_u8, noise_cfg)
        if depth_m is not None:
            depth_m = _apply_depth_noise_and_quantization(depth_m, noise_cfg)

        # Write image
        stem = f"{idx:06d}"
        img_path = out_dir / "images" / split / f"{stem}.jpg"
        lbl_path = out_dir / "labels" / split / f"{stem}.txt"
        # cv2 expects BGR
        bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        # Label from per-class masks.
        lines: List[str] = []
        for class_id, class_mask in sorted(mask_by_class_id.items(), key=lambda kv: kv[0]):
            lines.extend(
                _mask_to_polygons_yolo(
                    class_mask,
                    class_id=int(class_id),
                    min_area_px=int(args.min_mask_area_px),
                    approx_epsilon_px=float(args.poly_eps_px),
                    max_polys=int(args.max_polys),
                )
            )
        lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        # Optional depth
        if args.save_depth and depth_m is not None:
            _save_depth_png_16u(depth_m, out_dir / "depth" / split / f"{stem}.png", scale=1000.0)

        sample_noise_meta.append(
            {
                "image": str((Path("images") / split / f"{stem}.jpg").as_posix()),
                "noise": dataclasses.asdict(noise_cfg),
            }
        )

        if (idx + 1) % 100 == 0 or idx == 0:
            print(f"[{idx+1}/{n_total}] wrote {img_path.relative_to(out_dir)}")

    print("\nDone.")
    (out_dir / "sample_noise_meta.json").write_text(json.dumps(sample_noise_meta, indent=2))
    print(f"Dataset written to: {out_dir}")
    print(f"Dataset YAML: {out_dir / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

