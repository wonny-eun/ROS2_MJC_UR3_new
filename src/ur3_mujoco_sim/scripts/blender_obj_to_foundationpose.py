#!/usr/bin/env python3
"""Prepare Blender OBJ+MTL exports for Isaac ROS FoundationPose.

- Scale vertices (default 0.001: mm -> meters, matching MuJoCo STL convention).
- Translate mesh so centroid is at origin (FoundationPose expectation).
- Copy texture image next to OBJ; rewrite MTL map_Kd to local filename.

Usage:
  python3 scripts/blender_obj_to_foundationpose.py \\
    --input-dir ~/Downloads \\
    --output-dir meshes/foundationpose \\
    square_1 cylinder_1 cylinder_2
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path


def _write_assimp_mtl(out_mtl: Path, mat_name: str, texture_name: str) -> None:
    """Isaac FoundationPose uses Assimp; Blender PBR keys (aniso, Pr, ...) break mesh load."""
    out_mtl.write_text(
        "\n".join(
            [
                "# Assimp-compatible MTL for Isaac ROS FoundationPose",
                f"newmtl {mat_name}",
                "Ka 0.200000 0.200000 0.200000",
                "Kd 0.800000 0.800000 0.800000",
                "Ks 0.500000 0.500000 0.500000",
                "Ns 32.000000",
                "d 1.000000",
                "illum 2",
                f"map_Kd {texture_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _parse_map_kd(mtl_path: Path) -> str | None:
    if not mtl_path.is_file():
        return None
    for line in mtl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.lower().startswith("map_kd "):
            return line.split(maxsplit=1)[1].strip()
    return None


def _process_obj(
    src_obj: Path,
    src_mtl: Path | None,
    out_dir: Path,
    name: str,
    scale: float,
) -> tuple[Path, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    text = src_obj.read_text(encoding="utf-8", errors="replace").splitlines()

    verts: list[tuple[float, float, float]] = []
    for line in text:
        if line.startswith("v "):
            parts = line.split()
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))

    if not verts:
        raise ValueError(f"No vertices in {src_obj}")

    n = len(verts)
    cx = sum(p[0] for p in verts) / n
    cy = sum(p[1] for p in verts) / n
    cz = sum(p[2] for p in verts) / n

    texture_out: Path | None = None
    map_kd = _parse_map_kd(src_mtl) if src_mtl else None
    if map_kd:
        src_tex = (src_obj.parent / map_kd).resolve()
        if src_tex.is_file():
            ext = src_tex.suffix.lower() or ".jpg"
            texture_out = out_dir / f"{name}_texture{ext}"
            shutil.copy2(src_tex, texture_out)

    out_obj = out_dir / f"{name}.obj"
    out_mtl = out_dir / f"{name}.mtl"

    out_lines: list[str] = [
        f"# FoundationPose asset from {src_obj.name}",
        f"# scale={scale} pre_center_m=({cx * scale},{cy * scale},{cz * scale})",
        f"mtllib {name}.mtl",
    ]

    for line in text:
        if line.startswith("v "):
            parts = line.split()
            x = float(parts[1]) * scale - cx * scale
            y = float(parts[2]) * scale - cy * scale
            z = float(parts[3]) * scale - cz * scale
            out_lines.append(f"v {x:.9f} {y:.9f} {z:.9f}")
        elif line.startswith(("vt ", "vn ", "f ", "usemtl ", "o ", "g ", "s ")):
            out_lines.append(line)
        elif line.startswith("mtllib "):
            continue

    out_obj.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    if texture_out is not None:
        _write_assimp_mtl(out_mtl, f"{name}_mat", texture_out.name)
    else:
        out_mtl = None

    return out_obj, texture_out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path.home() / "Downloads")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "meshes" / "foundationpose",
    )
    parser.add_argument("--scale", type=float, default=0.001, help="mm->m uses 0.001")
    parser.add_argument("names", nargs="+", default=["square_1", "cylinder_1", "cylinder_2"])
    args = parser.parse_args()

    manifest: list[str] = ["# YOLO class -> mesh + texture (for Isaac FoundationPose)", ""]
    class_map = {
        "square_1": "Box_1",
        "cylinder_1": "Cylinder_1",
        "cylinder_2": "Cylinder_2",
    }

    for name in args.names:
        src_obj = args.input_dir / f"{name}.obj"
        src_mtl = args.input_dir / f"{name}.mtl"
        if not src_obj.is_file():
            raise SystemExit(f"Missing {src_obj}")
        out_obj, texture = _process_obj(src_obj, src_mtl, args.output_dir, name, args.scale)
        yolo = class_map.get(name, name)
        manifest.append(f"{yolo}:")
        manifest.append(f"  mesh: {out_obj.name}")
        if texture is not None:
            manifest.append(f"  texture: {texture.name}")
        manifest.append("")

        print(f"Wrote {out_obj}")
        if texture is not None:
            print(f"  texture {texture}")
        else:
            print("  WARN: no texture copied — Isaac may fail mesh render")

    manifest_path = args.output_dir / "mesh_texture_map.yaml"
    yaml_lines = ["# Auto-generated — set FOUNDATION_POSE_OBJECT or export per class", ""]
    for name in args.names:
        yolo = class_map.get(name, name)
        tex = args.output_dir / f"{name}_texture.jpg"
        if not tex.is_file():
            for p in args.output_dir.glob(f"{name}_texture.*"):
                tex = p
                break
        yaml_lines.append(f"{yolo}:")
        yaml_lines.append(f"  mesh: meshes/foundationpose/{name}.obj")
        if tex.is_file():
            yaml_lines.append(f"  texture: meshes/foundationpose/{tex.name}")
        yaml_lines.append("")
    manifest_path.write_text("\n".join(yaml_lines), encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
