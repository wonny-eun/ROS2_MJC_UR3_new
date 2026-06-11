"""Assign unique YOLO class labels per physical object in a multi-object scene.

When Cylinder_1 and Cylinder_2 look similar, raw YOLO may mislabel or duplicate a class.
This module clusters detections by bbox overlap, scores each cluster against every
exclusive scene class (full-frame detections + optional crop re-score), then picks a
one-to-one assignment so each class appears at most once.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, Callable, Optional

import numpy as np


@dataclass(frozen=True)
class SceneAssignedDetection:
    """One physical object after unique scene assignment."""

    xyxy: np.ndarray
    class_name: str
    confidence: float
    class_scores: dict[str, float]
    raw_yolo_class: str


def _norm_class(name: str, *, case_insensitive: bool) -> str:
    s = str(name).strip()
    return s.lower() if case_insensitive else s


def _resolve_exclusive_classes(
    exclusive_scene_classes: Optional[list[str]],
    *,
    case_insensitive: bool,
) -> list[str]:
    if not isinstance(exclusive_scene_classes, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in exclusive_scene_classes:
        label = str(raw).strip()
        if not label:
            continue
        key = _norm_class(label, case_insensitive=case_insensitive)
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
    return out


def bbox_iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in np.asarray(a, dtype=np.float64).reshape(-1)[:4]]
    bx1, by1, bx2, by2 = [float(v) for v in np.asarray(b, dtype=np.float64).reshape(-1)[:4]]
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def union_bbox_xyxy(boxes: list[np.ndarray]) -> np.ndarray:
    arr = np.stack([np.asarray(b, dtype=np.float64).reshape(-1)[:4] for b in boxes], axis=0)
    return np.array([arr[:, 0].min(), arr[:, 1].min(), arr[:, 2].max(), arr[:, 3].max()], dtype=np.float64)


def _gather_raw_detections(
    results: Any,
    exclusive_classes: list[str],
    *,
    case_insensitive: bool,
    conf_floor: float,
) -> list[tuple[np.ndarray, str, float]]:
    names = results.names
    boxes = results.boxes
    if boxes is None or boxes.cls is None or len(boxes) == 0:
        return []
    ex_norm = {_norm_class(c, case_insensitive=case_insensitive): c for c in exclusive_classes}
    out: list[tuple[np.ndarray, str, float]] = []
    for i in range(len(boxes)):
        cf = float(boxes.conf[i].item())
        if cf < conf_floor:
            continue
        cid = int(boxes.cls[i].item())
        cname = str(names[cid]).strip()
        key = _norm_class(cname, case_insensitive=case_insensitive)
        if key not in ex_norm:
            continue
        xyxy = boxes.xyxy[i].cpu().numpy()
        out.append((xyxy, cname, cf))
    return out


def _cluster_detections(
    dets: list[tuple[np.ndarray, str, float]],
    *,
    cluster_iou: float,
) -> list[list[int]]:
    if not dets:
        return []
    order = sorted(range(len(dets)), key=lambda i: -dets[i][2])
    clusters: list[list[int]] = []
    for idx in order:
        xyxy = dets[idx][0]
        placed = False
        for cl in clusters:
            if any(bbox_iou_xyxy(xyxy, dets[j][0]) >= cluster_iou for j in cl):
                cl.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    return clusters


def _crop_rgb_for_bbox(
    rgb: np.ndarray,
    xyxy: np.ndarray,
    *,
    pad_px: int = 12,
) -> np.ndarray:
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in np.asarray(xyxy, dtype=np.float64).reshape(-1)[:4]]
    x1 = max(0, x1 - pad_px)
    y1 = max(0, y1 - pad_px)
    x2 = min(w - 1, x2 + pad_px)
    y2 = min(h - 1, y2 + pad_px)
    if x2 <= x1 or y2 <= y1:
        return rgb
    return rgb[y1 : y2 + 1, x1 : x2 + 1]


def _scores_from_crop_predict(
    model: Any,
    rgb: np.ndarray,
    xyxy: np.ndarray,
    exclusive_classes: list[str],
    *,
    case_insensitive: bool,
    predict_conf_floor: float,
    yolo_iou: float,
) -> dict[str, float]:
    scores = {c: 0.0 for c in exclusive_classes}
    if model is None or rgb is None:
        return scores
    crop = _crop_rgb_for_bbox(rgb, xyxy)
    if crop.size <= 0:
        return scores
    try:
        pred = model.predict(
            crop,
            conf=float(max(1e-6, predict_conf_floor)),
            iou=float(yolo_iou),
            verbose=False,
        )[0]
    except Exception:  # noqa: BLE001
        return scores
    names = pred.names
    boxes = pred.boxes
    if boxes is None or boxes.cls is None:
        return scores
    canon = {_norm_class(c, case_insensitive=case_insensitive): c for c in exclusive_classes}
    for i in range(len(boxes)):
        cid = int(boxes.cls[i].item())
        cf = float(boxes.conf[i].item())
        cname = str(names[cid]).strip()
        key = _norm_class(cname, case_insensitive=case_insensitive)
        if key not in canon:
            continue
        label = canon[key]
        scores[label] = max(scores[label], cf)
    return scores


def _cluster_class_scores(
    cluster_indices: list[int],
    dets: list[tuple[np.ndarray, str, float]],
    exclusive_classes: list[str],
    *,
    case_insensitive: bool,
    model: Optional[Any],
    rgb: Optional[np.ndarray],
    rescore_crops: bool,
    predict_conf_floor: float,
    yolo_iou: float,
) -> tuple[dict[str, float], str, np.ndarray]:
    boxes = [dets[i][0] for i in cluster_indices]
    union = union_bbox_xyxy(boxes)
    scores = {c: 0.0 for c in exclusive_classes}
    raw_best = ""
    raw_best_conf = -1.0
    canon = {_norm_class(c, case_insensitive=case_insensitive): c for c in exclusive_classes}
    for i in cluster_indices:
        _xyxy, cname, cf = dets[i]
        key = _norm_class(cname, case_insensitive=case_insensitive)
        if key in canon:
            label = canon[key]
            scores[label] = max(scores[label], cf)
        if cf > raw_best_conf:
            raw_best_conf = cf
            raw_best = cname
    if rescore_crops and model is not None and rgb is not None:
        crop_scores = _scores_from_crop_predict(
            model,
            rgb,
            union,
            exclusive_classes,
            case_insensitive=case_insensitive,
            predict_conf_floor=predict_conf_floor,
            yolo_iou=yolo_iou,
        )
        for c in exclusive_classes:
            scores[c] = max(scores[c], crop_scores[c])
    return scores, raw_best, union


def _best_unique_assignment(
    score_rows: list[dict[str, float]],
    class_names: list[str],
) -> list[str]:
    n = len(score_rows)
    if n == 0 or not class_names:
        return []
    best_labels: list[str] = []
    best_total = -1.0
    for class_perm in permutations(class_names, n):
        total = sum(float(score_rows[i].get(class_perm[i], 0.0)) for i in range(n))
        if total > best_total:
            best_total = total
            best_labels = list(class_perm)
    return best_labels


def assign_unique_scene_labels(
    results: Any,
    *,
    exclusive_scene_classes: list[str],
    rgb: Optional[np.ndarray] = None,
    model: Optional[Any] = None,
    case_insensitive: bool = True,
    predict_conf_floor: float = 0.01,
    yolo_iou: float = 0.5,
    cluster_iou: float = 0.45,
    rescore_crops: bool = True,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[SceneAssignedDetection]:
    """Cluster detections, score each object against all scene classes, assign unique labels."""
    exclusive = _resolve_exclusive_classes(exclusive_scene_classes, case_insensitive=case_insensitive)
    if not exclusive:
        return []

    dets = _gather_raw_detections(
        results,
        exclusive,
        case_insensitive=case_insensitive,
        conf_floor=float(predict_conf_floor),
    )
    if not dets:
        return []

    clusters = _cluster_detections(dets, cluster_iou=float(cluster_iou))
    score_rows: list[dict[str, float]] = []
    unions: list[np.ndarray] = []
    raw_labels: list[str] = []
    for cl in clusters:
        scores, raw_label, union = _cluster_class_scores(
            cl,
            dets,
            exclusive,
            case_insensitive=case_insensitive,
            model=model,
            rgb=rgb,
            rescore_crops=bool(rescore_crops),
            predict_conf_floor=float(predict_conf_floor),
            yolo_iou=float(yolo_iou),
        )
        score_rows.append(scores)
        unions.append(union)
        raw_labels.append(raw_label)

    if len(score_rows) > len(exclusive):
        ranked = sorted(
            range(len(score_rows)),
            key=lambda i: max(score_rows[i].values()) if score_rows[i] else 0.0,
            reverse=True,
        )[: len(exclusive)]
        score_rows = [score_rows[i] for i in ranked]
        unions = [unions[i] for i in ranked]
        raw_labels = [raw_labels[i] for i in ranked]

    assigned_classes = _best_unique_assignment(score_rows, exclusive)
    out: list[SceneAssignedDetection] = []
    for i, class_name in enumerate(assigned_classes):
        conf = float(score_rows[i].get(class_name, 0.0))
        out.append(
            SceneAssignedDetection(
                xyxy=unions[i],
                class_name=class_name,
                confidence=conf,
                class_scores=dict(score_rows[i]),
                raw_yolo_class=raw_labels[i],
            )
        )
        if log_fn is not None:
            score_txt = ", ".join(f"{k}:{score_rows[i].get(k, 0.0):.2f}" for k in exclusive)
            fix_note = ""
            if raw_labels[i] and _norm_class(raw_labels[i], case_insensitive=case_insensitive) != _norm_class(
                class_name, case_insensitive=case_insensitive
            ):
                fix_note = f", relabel {raw_labels[i]!r}->{class_name!r}"
            log_fn(
                f"yolo scene assign: object {i + 1}/{len(assigned_classes)} "
                f"scores=[{score_txt}] -> {class_name!r}@{conf:.2f}{fix_note}"
            )
    return out


def pick_target_from_scene_assignments(
    assigned: list[SceneAssignedDetection],
    *,
    target_class: str,
    min_conf: float,
    case_insensitive: bool = True,
) -> Optional[tuple[np.ndarray, float, str]]:
    """Return (xyxy, conf, class_name) for ``target_class`` after unique scene assignment."""
    tgt = _norm_class(target_class, case_insensitive=case_insensitive)
    best: Optional[tuple[np.ndarray, float, str]] = None
    for det in assigned:
        if _norm_class(det.class_name, case_insensitive=case_insensitive) != tgt:
            continue
        if det.confidence < min_conf:
            continue
        if best is None or det.confidence > best[1]:
            best = (det.xyxy, det.confidence, det.class_name)
    return best


def format_cylinder_score_hint(det: SceneAssignedDetection) -> str:
    """Short Cylinder_1 vs Cylinder_2 comparison for overlay text."""
    c1 = det.class_scores.get("Cylinder_1", det.class_scores.get("cylinder_1", 0.0))
    c2 = det.class_scores.get("Cylinder_2", det.class_scores.get("cylinder_2", 0.0))
    if c1 <= 0.0 and c2 <= 0.0:
        return ""
    return f"C1:{c1:.2f} C2:{c2:.2f}"
