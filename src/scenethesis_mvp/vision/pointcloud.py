from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from scenethesis_mvp.schemas.depth import DepthResult
from scenethesis_mvp.schemas.scene_graph_3d import Object3DBoundingBox, ObjectPointCloudSpec, Pose3DSpec, SceneGraph3D
from scenethesis_mvp.schemas.segmentation import SegmentationResult
from scenethesis_mvp.utils.io import write_json


def build_pointcloud_scene_graph(
    segmentation: SegmentationResult,
    depth: DepthResult,
    out_dir: str | Path,
    max_points_per_object: int = 5000,
    min_mask_pixels: int = 128,
) -> SceneGraph3D:
    if segmentation.missing_object_ids:
        raise RuntimeError("Cannot project point clouds because segmentation is incomplete.")
    target_dir = Path(out_dir)
    points_dir = target_dir / "object_pointclouds"
    points_dir.mkdir(parents=True, exist_ok=True)
    depth_map = np.load(depth.depth_path).astype("float32")
    if depth_map.shape != (depth.intrinsics.height, depth.intrinsics.width):
        raise RuntimeError("Depth map shape does not match recorded camera intrinsics.")

    records: list[ObjectPointCloudSpec] = []
    poses: list[Pose3DSpec] = []
    missing: list[str] = []
    for detection in segmentation.detections:
        if not detection.object_id:
            continue
        mask = load_mask(detection.mask_path, target_shape=depth_map.shape)
        if int(mask.sum()) < min_mask_pixels:
            missing.append(detection.object_id)
            continue
        points = mask_depth_to_points(mask, depth_map, depth)
        points = points[np.isfinite(points).all(axis=1)]
        points = points[points[:, 2] > 0]
        if len(points) < min_mask_pixels:
            missing.append(detection.object_id)
            continue
        points = deterministic_sample(points, max_points=max_points_per_object)
        bbox = estimate_bbox(points)
        point_path = points_dir / f"{detection.object_id}.ply"
        write_ascii_ply(point_path, points)
        records.append(
            ObjectPointCloudSpec(
                object_id=detection.object_id,
                phrase=detection.phrase,
                points_path=str(point_path),
                point_count=len(points),
                bbox=bbox,
            )
        )
        poses.append(
            Pose3DSpec(
                object_id=detection.object_id,
                x=bbox.center[0],
                y=bbox.center[2],
                z=max(0.0, -bbox.center[1]),
                yaw_deg=bbox.yaw_deg,
                scale=1.0,
            )
        )
    if missing:
        raise RuntimeError("Depth projection failed for required objects: " + ", ".join(sorted(set(missing))))
    graph = SceneGraph3D(pointclouds=records, poses=poses, missing_object_ids=[])
    write_json(target_dir / "scene_graph_3d.json", graph)
    write_json(target_dir / "initial_3dbb.json", [record.model_dump(mode="json") for record in records])
    write_json(target_dir / "pose_init_depth.json", [pose.model_dump(mode="json") for pose in poses])
    return graph


def load_mask(mask_path: str | Path, target_shape: tuple[int, int]) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    target_height, target_width = target_shape
    if mask.size != (target_width, target_height):
        mask = mask.resize((target_width, target_height), resample=Image.Resampling.NEAREST)
    return np.asarray(mask) > 127


def mask_depth_to_points(mask: np.ndarray, depth_map: np.ndarray, depth: DepthResult) -> np.ndarray:
    v, u = np.nonzero(mask)
    z = depth_map[v, u]
    x = (u.astype("float32") - depth.intrinsics.cx) * z / depth.intrinsics.fx
    y = (v.astype("float32") - depth.intrinsics.cy) * z / depth.intrinsics.fy
    return np.column_stack([x, y, z]).astype("float32")


def deterministic_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points
    indices = np.linspace(0, len(points) - 1, num=max_points, dtype=int)
    return points[indices]


def estimate_bbox(points: np.ndarray) -> Object3DBoundingBox:
    center = points.mean(axis=0)
    centered_xz = points[:, [0, 2]] - center[[0, 2]]
    if len(centered_xz) >= 3:
        covariance = np.cov(centered_xz.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        principal = eigenvectors[:, int(np.argmax(eigenvalues))]
        yaw = math.degrees(math.atan2(float(principal[1]), float(principal[0])))
    else:
        yaw = 0.0
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    size = np.maximum(max_xyz - min_xyz, 1e-4)
    return Object3DBoundingBox(
        center=[round(float(value), 6) for value in center.tolist()],
        size=[round(float(value), 6) for value in size.tolist()],
        yaw_deg=round(float(yaw % 360.0), 6),
    )


def write_ascii_ply(path: str | Path, points: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="ascii") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("end_header\n")
        for point in points:
            handle.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
