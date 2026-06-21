from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont

ORIGINAL_DIR = Path(r"..\dataset\8i")
HPR_DIR = Path(r"hpr_output_ism24_framewise_ascii\longdress")

OUT_DIR = Path(r"demo_compare_longdress")
OUTPUT_VIDEO = Path(r"demo_compare_longdress.mp4")

WIDTH = 1000
HEIGHT = 1000
FPS = 30

CAMERA_JSON = Path(r"campath_0.json")

FOVY = 50.0

POINT_SIZE = 2.0
MAX_FRAMES: int | None = 300

VISUALIZER_ZOOM = 0.7

BG_COLOR = (1.0, 1.0, 1.0)

@dataclass(frozen=True)

def load_camera_trajectory(json_path: Path) -> list[dict]:
    if not json_path.exists():
        raise FileNotFoundError(f"Khong tim thay camera json: {json_path}")

    import json

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        trajectory = data["camera"]["trajectory"]
    except KeyError as exc:
        raise ValueError("File json khong co truong camera.trajectory") from exc

    if not isinstance(trajectory, list) or len(trajectory) == 0:
        raise ValueError("camera.trajectory rong hoac khong hop le")

    return trajectory


def get_camera_position_from_json_pose(pose: dict) -> np.ndarray:
    if "position" not in pose:
        raise ValueError("Camera pose khong co truong position")

    c = np.asarray(pose["position"], dtype=np.float64)

    if c.shape != (3,):
        raise ValueError(f"Camera position khong hop le: shape={c.shape}")

    return c

def natural_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def index_ply_files(folder: Path) -> dict[str, Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Khong tim thay folder: {folder}")

    files = sorted(folder.rglob("*.ply"), key=natural_key)

    if not files:
        raise FileNotFoundError(f"Khong tim thay file .ply trong folder: {folder}")

    index: dict[str, Path] = {}

    for path in files:
        if path.name in index:
            raise ValueError(
                f"Bi trung ten file .ply: {path.name}. "
                f"Hay dung folder chi chua mot sequence hoac doi ten file."
            )
        index[path.name] = path

    return index


def read_pcd(path: Path) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(str(path))

    if pcd.is_empty():
        raise ValueError(f"Khong doc duoc hoac point cloud rong: {path}")

    return pcd


def get_pcd_center(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    points = np.asarray(pcd.points, dtype=np.float64)

    if points.size == 0:
        raise ValueError("Point cloud khong co diem")

    return (points.min(axis=0) + points.max(axis=0)) / 2.0


def render_single_with_view_setup(
    pcd: o3d.geometry.PointCloud,
    camera_position: np.ndarray,
    target_center: np.ndarray,
    width: int,
    height: int,
    point_size: float,
    bg_color: tuple[float, float, float],
    zoom: float,
) -> tuple[Image.Image, o3d.camera.PinholeCameraParameters]:

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg_color, dtype=np.float64)
    opt.point_size = point_size

    eye = np.asarray(camera_position, dtype=np.float64)
    target = np.asarray(target_center+[0.0,100,0.0], dtype=np.float64)
    up = np.asarray([0.0, 100, 0.0], dtype=np.float64)

    front = target - eye
    if np.linalg.norm(front) < 1e-9:
        front = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    else:
        front = front / np.linalg.norm(front)

    ctr = vis.get_view_control()
    ctr.set_lookat(target)
    ctr.set_front(front)
    ctr.set_up(up)
    ctr.set_zoom(zoom)

    vis.poll_events()
    vis.update_renderer()

    params = ctr.convert_to_pinhole_camera_parameters()

    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()

    arr = np.asarray(img)
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB"), params


def render_single_with_existing_params(
    pcd: o3d.geometry.PointCloud,
    params: o3d.camera.PinholeCameraParameters,
    width: int,
    height: int,
    point_size: float,
    bg_color: tuple[float, float, float],
) -> Image.Image:

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray(bg_color, dtype=np.float64)
    opt.point_size = point_size

    ctr = vis.get_view_control()

    ctr.convert_from_pinhole_camera_parameters(
        params,
        allow_arbitrary=True,
    )

    vis.poll_events()
    vis.update_renderer()

    img = vis.capture_screen_float_buffer(False)
    vis.destroy_window()

    arr = np.asarray(img)
    if arr.dtype in (np.float32, np.float64):
        arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(arr).convert("RGB")


def render_original_and_hpr_same_view(
    original_pcd: o3d.geometry.PointCloud,
    hpr_pcd: o3d.geometry.PointCloud,
    camera_position: np.ndarray,
    target_center: np.ndarray,
) -> tuple[Image.Image, Image.Image]:

    original_img, params = render_single_with_view_setup(
        pcd=original_pcd,
        camera_position=camera_position,
        target_center=target_center,
        width=WIDTH,
        height=HEIGHT,
        point_size=POINT_SIZE,
        bg_color=BG_COLOR,
        zoom=VISUALIZER_ZOOM,
    )

    hpr_img = render_single_with_existing_params(
        pcd=hpr_pcd,
        params=params,
        width=WIDTH,
        height=HEIGHT,
        point_size=POINT_SIZE,
        bg_color=BG_COLOR,
    )

    return original_img, hpr_img


def load_font(size: int):
    candidates = [
        "arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue

    return ImageFont.load_default()


def add_label(
    image: Image.Image,
    title: str,
    subtitle: str,
    point_text: str,
) -> Image.Image:
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    font_title = load_font(30)
    font_text = load_font(19)

    pad = 18
    box_h = 88

    draw.rectangle([0, 0, canvas.width, box_h], fill=(255, 255, 255))

    draw.text((pad, 8), title, fill=(20, 20, 20), font=font_title)
    draw.text((pad, 43), subtitle, fill=(80, 80, 80), font=font_text)
    draw.text((pad, 65), point_text, fill=(80, 80, 80), font=font_text)

    return canvas


def compose_side_by_side(
    left: Image.Image,
    right: Image.Image,
    frame_idx: int,
    total_frames: int,
) -> Image.Image:
    gap = 12
    top_h = 70

    out_w = left.width + right.width + gap
    out_h = top_h + left.height

    canvas = Image.new("RGB", (out_w, out_h), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)

    font_title = load_font(32)
    font_text = load_font(21)

    title = "Viewport playback along HPR camera trajectory"
    info = f"Frame {frame_idx:04d}/{total_frames:04d}"

    draw.text((24, 13), title, fill=(20, 30, 45), font=font_title)

    info_bbox = draw.textbbox((0, 0), info, font=font_text)
    info_w = info_bbox[2] - info_bbox[0]

    draw.text((out_w - info_w - 24, 24), info, fill=(80, 80, 80), font=font_text)

    canvas.paste(left, (0, top_h))
    canvas.paste(right, (left.width + gap, top_h))

    draw.rectangle(
        [left.width, top_h, left.width + gap - 1, out_h],
        fill=(230, 230, 230),
    )

    return canvas

def make_video_ffmpeg(frames_dir: Path, output_video: Path, fps: int) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]

    subprocess.run(cmd, check=True)


def main() -> None:
    original_index = index_ply_files(ORIGINAL_DIR)
    hpr_index = index_ply_files(HPR_DIR)
    trajectory = load_camera_trajectory(CAMERA_JSON)

    if len(trajectory) < len(hpr_log):
        raise ValueError(
            f"Camera JSON chi co {len(trajectory)} poses, "
            f"nhung HPR log co {len(hpr_log)} frames."
        )

    if MAX_FRAMES is not None:
        hpr_log = hpr_log[:MAX_FRAMES]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames_dir = OUT_DIR / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    total_frames = len(hpr_log)

    print(f"Camera JSON     : {CAMERA_JSON}")
    print(f"System          : {platform.system()}")
    print(f"Original dir    : {ORIGINAL_DIR}")
    print(f"HPR dir         : {HPR_DIR}")
    print(f"Frames          : {total_frames}")
    print(f"Render backend  : Open3D Visualizer")
    print(f"FOVY note       : {FOVY}")
    print(f"Point size      : {POINT_SIZE}")
    print(f"Zoom            : {VISUALIZER_ZOOM}")
    print()

    for out_i, row in enumerate(hpr_log, start=1):
        if row.input_ply not in original_index:
            raise FileNotFoundError(f"Khong tim thay original ply: {row.input_ply}")

        if row.output_ply not in hpr_index:
            raise FileNotFoundError(f"Khong tim thay HPR ply: {row.output_ply}")

        original_path = original_index[row.input_ply]
        hpr_path = hpr_index[row.output_ply]

        original_pcd = read_pcd(original_path)
        hpr_pcd = read_pcd(hpr_path)


        json_pose = trajectory[row.frame_idx - 1]
        camera_position = get_camera_position_from_json_pose(json_pose)
        target_center = get_pcd_center(original_pcd)

        original_img, hpr_img = render_original_and_hpr_same_view(
            original_pcd=original_pcd,
            hpr_pcd=hpr_pcd,
            camera_position=camera_position,
            target_center=target_center,
        )

        n_original = len(original_pcd.points)
        n_hpr = len(hpr_pcd.points)
        reduction = 100.0 * (1.0 - n_hpr / max(n_original, 1))

        left = add_label(
            image=original_img,
            title="Original",
            subtitle="No hidden point removal",
            point_text=f"Points: {n_original:,}",
        )

        right = add_label(
            image=hpr_img,
            title="After HPR",
            subtitle="Hidden points removed",
            point_text=f"Points: {n_hpr:,} | Reduction: {reduction:.1f}%",
        )

        combined = compose_side_by_side(
            left=left,
            right=right,
            frame_idx=row.frame_idx,
            total_frames=total_frames,
        )

        out_frame = frames_dir / f"frame_{out_i:04d}.png"
        combined.save(out_frame)

        print(
            f"[{out_i:04d}/{total_frames:04d}] "
            f"{row.input_ply} | {row.output_ply} | "
            f"C1=[{camera_position[0]:.3f}, {camera_position[1]:.3f}, {camera_position[2]:.3f}] | "
            f"reduction={reduction:.1f}%"
        )

    if OUTPUT_VIDEO is not None:
        make_video_ffmpeg(frames_dir, OUTPUT_VIDEO, FPS)
        print()
        print(f"Saved video: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()