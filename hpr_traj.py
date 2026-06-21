from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import open3d as o3d

PLY_FOLDER = Path(r"..\dataset\8i")
CAM_JSON_PATH = Path(r"campath_0.json")
OUT_FOLDER = Path(r"hpr_output\longdress")

DELTA_X = 500.0
MAX_FRAMES = 1
ALLOW_TRUNCATE = True

HPR_RADIUS_FACTOR = 10000

WRITE_ASCII = True

SIDE_AXIS = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)

def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_ply_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.ply"), key=natural_key)

    if not files:
        raise FileNotFoundError(f"No .ply files found in folder: {folder}")

    return files


def load_camera_trajectory(json_path: Path) -> list[dict]:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        trajectory = data["camera"]["trajectory"]
    except KeyError as exc:
        raise ValueError("File json khong co truong camera.trajectory") from exc

    if not isinstance(trajectory, list) or len(trajectory) == 0:
        raise ValueError("camera.trajectory rong hoac khong hop le")

    return trajectory


def read_point_cloud(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))

    if pcd.is_empty():
        raise ValueError(f"Khong doc duoc hoac point cloud rong: {path}")

    return pcd


def write_point_cloud_ascii(path: Path, pcd: o3d.geometry.PointCloud) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    ok = o3d.io.write_point_cloud(
        str(path),
        pcd,
        write_ascii=True,
        compressed=False,
        print_progress=False,
    )

    if not ok:
        raise RuntimeError(f"Ghi file that bai: {path}")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first_lines = [next(f).strip() for _ in range(3)]

    if "format ascii 1.0" not in first_lines:
        raise RuntimeError(
            f"File da ghi nhung header khong phai ascii 1.0: {path}"
        )


def get_pcd_center(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    points = np.asarray(pcd.points, dtype=np.float64)

    if points.size == 0:
        raise ValueError("Point cloud khong co diem")

    return (points.min(axis=0) + points.max(axis=0)) / 2.0


def get_camera_position(pose: dict) -> np.ndarray:
    if "position" not in pose:
        raise ValueError("Camera pose khong co truong position")

    c = np.asarray(pose["position"], dtype=np.float64)

    if c.shape != (3,):
        raise ValueError(f"Camera position khong hop le: shape={c.shape}")

    return c


def ism24_viewpoints(
    c1: np.ndarray,
    delta_x: float,
    side_axis: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    side_axis = np.asarray(side_axis, dtype=np.float64)

    norm = np.linalg.norm(side_axis)
    if norm < 1e-9:
        raise ValueError("SIDE_AXIS khong hop le")

    side_axis = side_axis / norm
    offset = delta_x * side_axis

    c2 = c1 + offset
    c3 = c1 - offset

    return c1, c2, c3


def estimate_hpr_radius(
    pcd: o3d.geometry.PointCloud,
    camera_position: np.ndarray,
    radius_factor: float,
) -> float:
    points = np.asarray(pcd.points, dtype=np.float64)

    if points.size == 0:
        raise ValueError("Point cloud khong co diem")

    camera_position = np.asarray(camera_position, dtype=np.float64)
    max_dist = np.linalg.norm(points - camera_position[None, :], axis=1).max()

    radius = float(max_dist * radius_factor)

    if not math.isfinite(radius) or radius <= 0:
        raise ValueError(f"HPR radius khong hop le: {radius}")

    return radius


def hidden_point_indices(
    pcd: o3d.geometry.PointCloud,
    camera_position: np.ndarray,
    radius_factor: float,
) -> np.ndarray:
    radius = estimate_hpr_radius(
        pcd=pcd,
        camera_position=camera_position,
        radius_factor=radius_factor,
    )

    _, indices = pcd.hidden_point_removal(
        camera_location=np.asarray(camera_position, dtype=np.float64),
        radius=radius,
    )

    return np.asarray(indices, dtype=np.int64)


def hpr_union_ism24(
    pcd: o3d.geometry.PointCloud,
    c1: np.ndarray,
    c2: np.ndarray,
    c3: np.ndarray,
    radius_factor: float,
) -> np.ndarray:
    idx1 = hidden_point_indices(pcd, c1, radius_factor)
    idx2 = hidden_point_indices(pcd, c2, radius_factor)
    idx3 = hidden_point_indices(pcd, c3, radius_factor)

    visible_idx = np.unique(np.concatenate([idx1, idx2, idx3]))

    return visible_idx


def select_points(
    pcd: o3d.geometry.PointCloud,
    indices: np.ndarray,
) -> o3d.geometry.PointCloud:
    return pcd.select_by_index(indices.astype(np.int64).tolist())


# def write_hpr_log(log_path: Path, rows: list[HprFrameLog]) -> None:
#     log_path.parent.mkdir(parents=True, exist_ok=True)

#     with log_path.open("w", newline="", encoding="utf-8") as f:
#         writer = csv.DictWriter(
#             f,
#             fieldnames=[
#                 "frame_idx",
#                 "input_ply",
#                 "output_ply",

#                 "raw_C1_x",
#                 "raw_C1_y",
#                 "raw_C1_z",

#                 "center_x",
#                 "center_y",
#                 "center_z",

#                 "C1_x",
#                 "C1_y",
#                 "C1_z",
#                 "C2_x",
#                 "C2_y",
#                 "C2_z",
#                 "C3_x",
#                 "C3_y",
#                 "C3_z",

#                 "input_points",
#                 "output_points",
#                 "reduction_percent",
#                 "radius_factor",
#             ],
#         )

#         writer.writeheader()

#         for row in rows:
#             writer.writerow(
#                 {
#                     "frame_idx": row.frame_idx,
#                     "input_ply": row.input_ply,
#                     "output_ply": row.output_ply,

#                     "raw_C1_x": row.raw_c1[0],
#                     "raw_C1_y": row.raw_c1[1],
#                     "raw_C1_z": row.raw_c1[2],

#                     "center_x": row.center[0],
#                     "center_y": row.center[1],
#                     "center_z": row.center[2],

#                     "C1_x": row.c1[0],
#                     "C1_y": row.c1[1],
#                     "C1_z": row.c1[2],
#                     "C2_x": row.c2[0],
#                     "C2_y": row.c2[1],
#                     "C2_z": row.c2[2],
#                     "C3_x": row.c3[0],
#                     "C3_y": row.c3[1],
#                     "C3_z": row.c3[2],

#                     "input_points": row.input_points,
#                     "output_points": row.output_points,
#                     "reduction_percent": row.reduction_percent,
#                     "radius_factor": row.radius_factor,
#                 }
#             )

def resolve_num_pairs(
    num_ply: int,
    num_cam: int,
    max_frames: int | None,
    allow_truncate: bool,
) -> int:
    if num_ply != num_cam:
        msg = f"Number of PLY files ({num_ply}) != number of camera poses ({num_cam})."

        if allow_truncate:
            print(f"[WARN] {msg} Processing only matched pairs.")
        else:
            raise ValueError(msg)

    num_pairs = min(num_ply, num_cam)

    if max_frames is not None:
        num_pairs = min(num_pairs, max_frames)

    return num_pairs


def process_framewise() -> None:
    OUT_FOLDER.mkdir(parents=True, exist_ok=True)

    ply_files = list_ply_files(PLY_FOLDER)
    trajectory = load_camera_trajectory(CAM_JSON_PATH)

    num_ply = len(ply_files)
    num_cam = len(trajectory)

    num_pairs = resolve_num_pairs(
        num_ply=num_ply,
        num_cam=num_cam,
        max_frames=MAX_FRAMES,
        allow_truncate=ALLOW_TRUNCATE,
    )

    print(f"Found {num_ply} PLY files.")
    print(f"Loaded {num_cam} camera poses from JSON.")
    print(f"Processing {num_pairs} frame-camera pairs.")
    print(f"deltaX = {DELTA_X:.3f}")
    print(f"HPR_RADIUS_FACTOR = {HPR_RADIUS_FACTOR:.3f}")
    print(f"WRITE_ASCII = {WRITE_ASCII}")
    print()

    for i in range(num_pairs):
        frame_idx = i + 1

        ply_path = ply_files[i]
        ply_base_name = ply_path.stem

        print(f"[{frame_idx}/{num_pairs}] {ply_path.name} -> camera {frame_idx}")

        pcd = read_point_cloud(ply_path)
        pose = trajectory[i]

        center = get_pcd_center(pcd)

        c1, c2, c3 = ism24_viewpoints(
            c1=get_camera_position(pose),
            delta_x=DELTA_X,
            side_axis=SIDE_AXIS,
        )

        visible_idx = hpr_union_ism24(
            pcd=pcd,
            c1=c1,
            c2=c2,
            c3=c3,
            radius_factor=HPR_RADIUS_FACTOR,
        )

        pcd_out = select_points(pcd, visible_idx)

        out_name = f"{ply_base_name}_hpr_cam{frame_idx:04d}.ply"
        out_path = OUT_FOLDER / out_name

        if WRITE_ASCII:
            write_point_cloud_ascii(out_path, pcd_out)
        else:
            ok = o3d.io.write_point_cloud(
                str(out_path),
                pcd_out,
                write_ascii=False,
                compressed=False,
                print_progress=False,
            )
            if not ok:
                raise RuntimeError(f"Ghi file that bai: {out_path}")

        input_points = len(pcd.points)
        output_points = len(pcd_out.points)
        reduction_percent = 100.0 * (1.0 - output_points / max(input_points, 1))

        print(f"  raw C1 = [{c1[0]:.3f} {c1[1]:.3f} {c1[2]:.3f}]")
        print(f"  center = [{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}]")
        print(f"  C1     = [{c1[0]:.3f} {c1[1]:.3f} {c1[2]:.3f}]")
        print(f"  C2     = [{c2[0]:.3f} {c2[1]:.3f} {c2[2]:.3f}]")
        print(f"  C3     = [{c3[0]:.3f} {c3[1]:.3f} {c3[2]:.3f}]")
        print(f"  visible = {output_points} / {input_points} points")
        print(f"  reduction = {reduction_percent:.1f}%")
        print(f"  saved ascii 1.0: {out_name}")
        print()


    print("Done.")

def main() -> None:
    process_framewise()


if __name__ == "__main__":
    main()