#!/usr/bin/env python3
"""Convert Labelme JSON in yolo_real_data into YOLO-seg layout for Ultralytics.

Reads flat ``<dataset>/images/*.json`` (+ matching ``imagePath`` RGB), writes
``images/{train,val}/`` (symlinks to parent RGB) and ``labels/{train,val}/*.txt``.

Class order matches ``data/yolo_multi_drop/dataset.yaml``:
  0 Cylinder_1, 1 Cylinder_2, 2 Box_1

Example::

  python3 tools/yolo_combine_labelme_real/prepare_yolo_real_for_training.py \\
    --dataset /home/wonny/ur3_control/data/yolo_real_data --val-split 0.1 --seed 0

Then train with ``data/yolo_multi_drop_plus_real.yaml`` (see repo ``data/``).

For real RGB capture, vary arm poses so objects appear in the **center and corners** of the
image (do not run look-at before saving). See ``collection_poses.yaml`` in this directory.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path

# Must match yolo_multi_drop/dataset.yaml
CLASS_TO_ID = {
    "Cylinder_1": 0,
    "Cylinder_2": 1,
    "Box_1": 2,
}


def polygon_to_yolo_line(
    cls_id: int, points: list, img_w: float, img_h: float
) -> str:
    parts: list[str] = [str(cls_id)]
    for xy in points:
        x = float(xy[0]) / img_w
        y = float(xy[1]) / img_h
        x = min(1.0, max(0.0, x))
        y = min(1.0, max(0.0, y))
        parts.append(f"{x:.6f}")
        parts.append(f"{y:.6f}")
    return " ".join(parts)


def load_labelme(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def write_labels_for_json(
    json_path: Path,
    out_txt: Path,
    skip_unknown: bool,
) -> tuple[int, int]:
    """Returns (num_shapes_written, num_shapes_skipped)."""
    data = load_labelme(json_path)
    w = float(data.get("imageWidth") or 0)
    h = float(data.get("imageHeight") or 0)
    if w <= 0 or h <= 0:
        raise ValueError(f"{json_path}: missing or invalid imageWidth/imageHeight")

    lines: list[str] = []
    skipped = 0
    for shape in data.get("shapes") or []:
        if shape.get("shape_type") != "polygon":
            skipped += 1
            continue
        label = shape.get("label")
        if label not in CLASS_TO_ID:
            if skip_unknown:
                skipped += 1
                continue
            raise ValueError(f"{json_path}: unknown label {label!r}; expected one of {sorted(CLASS_TO_ID)}")
        pts = shape.get("points") or []
        if len(pts) < 3:
            skipped += 1
            continue
        cls_id = CLASS_TO_ID[label]
        lines.append(polygon_to_yolo_line(cls_id, pts, w, h))

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines), skipped


def ensure_symlink(link: Path, target: Path, force: bool) -> None:
    """Create link -> target (relative target if under same tree)."""
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if not force:
            return
        link.unlink()

    try:
        rel = os.path.relpath(target, start=link.parent)
        link.symlink_to(rel)
    except OSError:
        link.symlink_to(target.resolve())


def copy_rgb(dst: Path, src: Path, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "yolo_real_data",
        help="Root of yolo_real_data (contains images/ with Labelme json + rgb).",
    )
    ap.add_argument("--val-split", type=float, default=0.1, help="Fraction of frames for val (0..1).")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for train/val split.")
    ap.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy RGB into images/train|val instead of symlinks.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing symlinks / labels / copied images.",
    )
    ap.add_argument(
        "--skip-unknown",
        action="store_true",
        help="Skip polygons with labels not in CLASS_TO_ID instead of failing.",
    )
    args = ap.parse_args()

    root: Path = args.dataset.resolve()
    images_root = root / "images"
    labels_root = root / "labels"

    if not images_root.is_dir():
        print(f"Missing images dir: {images_root}", file=sys.stderr)
        return 1

    json_files = sorted(p for p in images_root.glob("*.json") if p.is_file())
    if not json_files:
        print(f"No Labelme *.json under {images_root} (top level only).", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    stems = [p.stem for p in json_files]
    rng.shuffle(stems)
    n_val = int(round(len(stems) * float(args.val_split)))
    n_val = max(0, min(len(stems), n_val))
    val_stems = set(stems[:n_val])

    total_shapes = 0
    total_skip = 0
    for jp in json_files:
        stem = jp.stem
        split = "val" if stem in val_stems else "train"

        data = load_labelme(jp)
        im_rel = data.get("imagePath") or f"{stem}.jpg"
        src_im = (jp.parent / im_rel).resolve()
        if not src_im.is_file():
            print(f"Missing image for {jp.name}: expected {src_im}", file=sys.stderr)
            return 1

        dst_im = images_root / split / src_im.name
        if args.copy_images:
            copy_rgb(dst_im, src_im, args.force)
        else:
            ensure_symlink(dst_im, src_im, args.force)

        out_txt = labels_root / split / f"{stem}.txt"
        n_ok, n_sk = write_labels_for_json(jp, out_txt, args.skip_unknown)
        total_shapes += n_ok
        total_skip += n_sk

    print(f"Dataset root: {root}")
    print(f"JSON files: {len(json_files)}  (train={len(json_files) - n_val}, val={n_val})")
    print(f"YOLO lines written (polygons): {total_shapes}  skipped: {total_skip}")
    print(f"RGB layout: {'copy' if args.copy_images else 'symlink'} -> {images_root}/train|val/")
    print(f"Labels: {labels_root}/train|val/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
