from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import open3d as o3d


PLY_FOLDER = Path(r"..\dataset\8i")
OUT_FOLDER = Path(r"hpr_output_static")

CAMERA_SCALE = 1.0

DELTA_X = 500.0

HPR_RADIUS_FACTOR = 10000

MAX_FRAMES = None

WRITE_ASCII = True

SIDE_AXIS = np.asarray(
    [1.0, 0.0, 0.0],
    dtype=np.float64,
)

def get_bbox_center_and_diameter(
    pcd: o3d.geometry.PointCloud,
):
    pts = np.asarray(pcd.points)

    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)

    center = (pmin + pmax) / 2.0
    diameter = np.linalg.norm(pmax - pmin)

    return center, diameter

def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_ply(folder: Path):
    files = sorted(folder.glob("*.ply"), key=natural_key)

    if not files:
        raise FileNotFoundError(
            f"No ply files found in {folder}"
        )

    return files


def read_pcd(path: Path):
    pcd = o3d.io.read_point_cloud(str(path))

    if pcd.is_empty():
        raise RuntimeError(
            f"Cannot read {path}"
        )

    return pcd


def write_ascii(path: Path, pcd):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    o3d.io.write_point_cloud(
        str(path),
        pcd,
        write_ascii=True,
        compressed=False,
    )


def estimate_radius(
    pcd,
    camera_position,
):
    points = np.asarray(pcd.points)

    max_dist = np.linalg.norm(
        points - camera_position,
        axis=1,
    ).max()

    return max_dist * HPR_RADIUS_FACTOR


def hpr_indices(
    pcd,
    camera_position,
):
    radius = estimate_radius(
        pcd,
        camera_position,
    )

    _, idx = pcd.hidden_point_removal(
        camera_position,
        radius,
    )

    return np.asarray(idx)


def hpr_union(
    pcd,
    c1,
    c2,
    c3,
):
    idx1 = hpr_indices(pcd, c1)
    idx2 = hpr_indices(pcd, c2)
    idx3 = hpr_indices(pcd, c3)

    return np.unique(
        np.concatenate(
            [idx1, idx2, idx3]
        )
    )


def main():

    OUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = list_ply(
        PLY_FOLDER
    )

    if MAX_FRAMES is not None:
        files = files[:MAX_FRAMES]

    center, diameter = get_bbox_center_and_diameter(
        pcd
    )

    c1 = center + np.array(
        [0.0, 0.0, diameter * CAMERA_SCALE]
    )

    c2 = c1 + np.array(
        [DELTA_X, 0.0, 0.0]
    )

    c3 = c1 - np.array(
        [DELTA_X, 0.0, 0.0]
    )

    print(
        f"Processing {len(files)} frames"
    )

    for i, ply_path in enumerate(files, start=1):

        pcd = read_pcd(ply_path)

        visible_idx = hpr_union(
            pcd,
            c1,
            c2,
            c3,
        )

        pcd_out = pcd.select_by_index(
            visible_idx.tolist()
        )

        out_path = (
            OUT_FOLDER
            / f"{ply_path.stem}_hpr.ply"
        )

        write_ascii(
            out_path,
            pcd_out,
        )

        print(
            f"[{i}/{len(files)}] "
            f"{ply_path.name} "
            f"{len(pcd_out.points)}/{len(pcd.points)}"
        )

    print("Done")


if __name__ == "__main__":
    main()