#!/usr/bin/env python3
"""Binary STL to Wavefront OBJ: scale verts + center mesh at centroid.

MJCF meshes use ``scale=\"0.001 0.001 0.001\"`` on STL in mm units; apply the same ``--scale``.
FoundationPose prefers OBJ with origin near centroid (see NVIDIA mesh-center tutorials).

Usage (from ``ur3_mujoco_sim`` package root):

  python3 scripts/stl_to_centered_obj.py meshes/cylinder_1.stl meshes/foundationpose/cylinder_1.obj
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def load_binary_stl_triangles(stl_path: Path) -> list[tuple[tuple[float, float, float], ...]]:
    data = stl_path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"STL too short: {stl_path}")
    n = struct.unpack_from("<I", data, 80)[0]
    offset = 84
    facet_sz = 50
    expected = offset + facet_sz * int(n)
    if len(data) < expected:
        raise ValueError(f"STL truncated (claims {n} facets): {stl_path}")

    out: list[tuple[tuple[float, float, float], ...]] = []
    for i in range(int(n)):
        base = offset + i * facet_sz
        v1 = struct.unpack_from("<fff", data, base + 12)
        v2 = struct.unpack_from("<fff", data, base + 24)
        v3 = struct.unpack_from("<fff", data, base + 36)
        out.append((v1, v2, v3))
    return out


def write_centered_obj(out_path: Path, triangles: list[tuple[tuple[float, float, float], ...]], scale: float) -> None:
    verts_scaled: list[tuple[float, float, float]] = []
    for t in triangles:
        for vx, vy, vz in t:
            verts_scaled.append((vx * scale, vy * scale, vz * scale))

    nvert = len(verts_scaled)
    cx = sum(p[0] for p in verts_scaled) / nvert
    cy = sum(p[1] for p in verts_scaled) / nvert
    cz = sum(p[2] for p in verts_scaled) / nvert

    coords: dict[tuple[int, int, int], int] = {}

    def vertex_index(x: float, y: float, z: float) -> int:
        k = (
            int(round(x * 1_000_000)),
            int(round(y * 1_000_000)),
            int(round(z * 1_000_000)),
        )
        idx = coords.get(k)
        if idx is not None:
            return idx
        idx = len(coords)
        coords[k] = idx
        return idx

    indexed_tris: list[tuple[int, int, int]] = []
    for t in triangles:
        idxs = []
        for vx, vy, vz in t:
            x = vx * scale - cx
            y = vy * scale - cy
            z = vz * scale - cz
            idxs.append(vertex_index(x, y, z))
        indexed_tris.append((idxs[0], idxs[1], idxs[2]))

    inv_list: list[tuple[float, float, float]] = [(-1e9, -1e9, -1e9)] * len(coords)

    # Fill vertex coordinates from centroid-centered positions
    for tri in triangles:
        for vx, vy, vz in tri:
            x = vx * scale - cx
            y = vy * scale - cy
            z = vz * scale - cz
            k = (
                int(round(x * 1_000_000)),
                int(round(y * 1_000_000)),
                int(round(z * 1_000_000)),
            )
            oid = coords[k]
            inv_list[oid] = (x, y, z)

    buf: list[str] = [
        "# centroid-centered OBJ (STL scale applied)",
        f"# scale={scale:.12g} pre_center_m=({cx:.12g},{cy:.12g},{cz:.12g})",
    ]
    for vx, vy, vz in inv_list:
        buf.append(f"v {vx:.9g} {vy:.9g} {vz:.9g}")

    off = 1
    for a, b, c in indexed_tris:
        buf.append(f"f {a + off} {b + off} {c + off}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(buf) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_stl", type=Path)
    ap.add_argument("output_obj", type=Path)
    ap.add_argument("--scale", type=float, default=0.001, help="STL unit scale → meters (MuJoCo default 0.001)")
    args = ap.parse_args()
    tri = load_binary_stl_triangles(args.input_stl)
    write_centered_obj(args.output_obj, tri, args.scale)
    print(f"Wrote {args.output_obj}")


if __name__ == "__main__":
    main()
