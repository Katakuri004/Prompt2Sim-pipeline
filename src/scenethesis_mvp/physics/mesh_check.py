from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.collision import object_aabb, should_skip_collision
from scenethesis_mvp.layout.stability import center_inside_support, mounted_support_kind, support_z_candidates
from scenethesis_mvp.schemas.mesh_metrics import (
    MeshCollisionRecord,
    MeshMetrics,
    MeshObjectRecord,
    MeshSupportRecord,
)
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec


@dataclass(frozen=True)
class ObjectPointCloud:
    object_id: str
    asset_id: str
    source: str
    mesh_path: Path | None
    points: np.ndarray


def compute_mesh_metrics(
    scene: SceneSpec,
    registry: AssetRegistry,
    sample_points: int = 256,
    collision_distance: float = 0.035,
    support_tolerance: float = 0.04,
    require_meshes: bool = False,
) -> tuple[MeshMetrics, dict[str, Any]]:
    clouds = {
        obj.id: object_point_cloud(obj, registry, sample_points=sample_points, require_mesh=require_meshes)
        for obj in scene.objects
    }
    object_map = {obj.id: obj for obj in scene.objects}
    object_records = [
        MeshObjectRecord(
            object_id=cloud.object_id,
            asset_id=cloud.asset_id,
            source=cloud.source,
            mesh_path=str(cloud.mesh_path) if cloud.mesh_path else None,
            sampled_points=len(cloud.points),
        )
        for cloud in clouds.values()
    ]
    mesh_count = sum(1 for record in object_records if record.source == "mesh")
    proxy_count = len(object_records) - mesh_count

    collisions: list[MeshCollisionRecord] = []
    closest_pairs: list[dict[str, Any]] = []
    broad_phase_count = 0
    narrow_phase_count = 0
    boxes = {obj.id: object_aabb(obj, registry) for obj in scene.objects}
    for left_id, right_id in combinations(boxes, 2):
        left_obj = object_map[left_id]
        right_obj = object_map[right_id]
        if should_skip_collision(left_obj, right_obj):
            continue
        left_box = boxes[left_id]
        right_box = boxes[right_id]
        if not left_box.overlaps(right_box, tolerance=0.0):
            continue
        broad_phase_count += 1
        narrow_phase_count += 1
        min_distance, point_a, point_b = nearest_points(clouds[left_id].points, clouds[right_id].points)
        penetration = left_box.penetration(right_box)
        closest_pairs.append(
            {
                "object_a": left_id,
                "object_b": right_id,
                "min_distance": round(min_distance, 6),
                "aabb_penetration": round(penetration, 6),
                "point_a": rounded_point(point_a),
                "point_b": rounded_point(point_b),
            }
        )
        if min_distance <= collision_distance:
            collisions.append(
                MeshCollisionRecord(
                    object_a=left_id,
                    object_b=right_id,
                    min_distance=round(min_distance, 6),
                    aabb_penetration=round(penetration, 6),
                    point_a=rounded_point(point_a),
                    point_b=rounded_point(point_b),
                )
            )

    supports = [
        support_record(obj, scene, registry, clouds[obj.id], support_tolerance)
        for obj in scene.objects
    ]
    failures = [record for record in supports if not record.supported]
    collision_penalty = round(sum(max(0.001, collision_distance - item.min_distance) for item in collisions), 6)
    support_penalty = round(sum(max(0.001, item.contact_distance - support_tolerance) for item in failures), 6)
    metrics = MeshMetrics(
        object_count=len(scene.objects),
        mesh_object_count=mesh_count,
        proxy_object_count=proxy_count,
        broad_phase_pair_count=broad_phase_count,
        narrow_phase_pair_count=narrow_phase_count,
        mesh_collision_count=len(collisions),
        support_failure_count=len(failures),
        mesh_collisions=collisions,
        supports=supports,
        objects=object_records,
        collision_penalty=collision_penalty,
        support_penalty=support_penalty,
        total_penalty=round(collision_penalty * 10.0 + support_penalty * 8.0, 6),
    )
    samples = {
        "collision_distance": collision_distance,
        "support_tolerance": support_tolerance,
        "closest_pairs": closest_pairs,
        "collisions": [collision.model_dump(mode="json") for collision in collisions],
    }
    return metrics, samples


def refine_mesh_layout(
    scene: SceneSpec,
    registry: AssetRegistry,
    max_iters: int = 6,
    sample_points: int = 256,
    collision_distance: float = 0.035,
    support_tolerance: float = 0.04,
    require_meshes: bool = False,
) -> tuple[SceneSpec, MeshMetrics, dict[str, Any]]:
    current = deepcopy(scene)
    history: list[dict[str, Any]] = []
    metrics, samples = compute_mesh_metrics(
        current,
        registry,
        sample_points=sample_points,
        collision_distance=collision_distance,
        support_tolerance=support_tolerance,
        require_meshes=require_meshes,
    )
    for index in range(max(0, max_iters)):
        if metrics.mesh_collision_count == 0:
            break
        moved_any = False
        for collision in metrics.mesh_collisions:
            moved_any = separate_mesh_collision(current, collision)
        history.append({"iteration": index + 1, "collision_count": metrics.mesh_collision_count, "moved": moved_any})
        if not moved_any:
            break
        metrics, samples = compute_mesh_metrics(
            current,
            registry,
            sample_points=sample_points,
            collision_distance=collision_distance,
            support_tolerance=support_tolerance,
            require_meshes=require_meshes,
        )
    samples["refinement_history"] = history
    return SceneSpec.model_validate(current.model_dump()), metrics, samples


def separate_mesh_collision(scene: SceneSpec, collision: MeshCollisionRecord) -> bool:
    left = scene.object_by_id(collision.object_a)
    right = scene.object_by_id(collision.object_b)
    move = movable_object(left, right)
    if move.role == "anchor" or move.parent_id:
        return False
    other = right if move.id == left.id else left
    dx = move.placement.x - other.placement.x
    dy = move.placement.y - other.placement.y
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        dx, dy, norm = 1.0, 0.0, 1.0
    step = max(0.04, collision.aabb_penetration + 0.04)
    move.placement.x += dx / norm * step
    move.placement.y += dy / norm * step
    return True


def movable_object(left: ObjectSpec, right: ObjectSpec) -> ObjectSpec:
    priority = {"anchor": 0, "parent": 1, "child": 2}
    if priority[left.role] > priority[right.role]:
        return left
    if priority[right.role] > priority[left.role]:
        return right
    return right if right.id > left.id else left


def object_point_cloud(
    obj: ObjectSpec,
    registry: AssetRegistry,
    sample_points: int = 256,
    require_mesh: bool = False,
) -> ObjectPointCloud:
    if not obj.asset_id:
        raise ValueError(f"object {obj.id} has no asset_id")
    asset = registry.get(obj.asset_id)
    mesh_path = asset.resolved_mesh_path(registry.base_dir)
    if mesh_path:
        if not mesh_path.is_file():
            raise RuntimeError(f"asset {asset.id} references missing mesh: {mesh_path}")
        mesh = load_mesh(mesh_path)
        local_points = deterministic_points(mesh, max_points=sample_points)
        points = transform_points(local_points, obj, registry, mesh.bounds)
        return ObjectPointCloud(obj.id, asset.id, "mesh", mesh_path, points)
    if require_mesh:
        raise RuntimeError(f"asset {asset.id} has no local mesh path for mesh physics")
    points = proxy_box_points(asset.scaled_dimensions(obj.placement.scale), obj.placement.yaw_deg)
    points[:, 0] += obj.placement.x
    points[:, 1] += obj.placement.y
    points[:, 2] += obj.placement.z
    return ObjectPointCloud(obj.id, asset.id, "aabb_proxy", None, points)


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(str(path), force="scene")
    if isinstance(loaded, trimesh.Scene):
        if hasattr(loaded, "to_geometry"):
            mesh = loaded.to_geometry()
        else:
            mesh = loaded.dump(concatenate=True)
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise RuntimeError(f"unsupported mesh load result for {path}: {type(loaded)}")
    if mesh.vertices.size == 0:
        raise RuntimeError(f"mesh has no vertices: {path}")
    return mesh


def deterministic_points(mesh: trimesh.Trimesh, max_points: int) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    bounds = np.asarray(mesh.bounds, dtype=float)
    corners = bbox_corners(bounds[0], bounds[1])
    if len(mesh.faces):
        triangles = vertices[np.asarray(mesh.faces)]
        centers = triangles.mean(axis=1)
        candidates = np.vstack([vertices, centers])
    else:
        candidates = vertices
    points = np.vstack([corners, candidates])
    if len(points) <= max_points:
        return points
    if max_points <= len(corners):
        return corners[:max_points]
    remaining = max_points - len(corners)
    indices = np.linspace(0, len(candidates) - 1, num=remaining, dtype=int)
    return np.vstack([corners, candidates[indices]])


def transform_points(
    local_points: np.ndarray,
    obj: ObjectSpec,
    registry: AssetRegistry,
    source_bounds: np.ndarray | None = None,
) -> np.ndarray:
    if not obj.asset_id:
        raise ValueError(f"object {obj.id} has no asset_id")
    asset = registry.get(obj.asset_id)
    target_dims = np.asarray(asset.scaled_dimensions(obj.placement.scale), dtype=float)
    if source_bounds is None:
        source_min = local_points.min(axis=0)
        source_max = local_points.max(axis=0)
    else:
        source_min = np.asarray(source_bounds[0], dtype=float)
        source_max = np.asarray(source_bounds[1], dtype=float)
    source_size = np.maximum(source_max - source_min, 1e-6)
    source_center = (source_min + source_max) * 0.5
    scaled = (local_points - source_center) * (target_dims / source_size)
    yaw = math.radians(obj.placement.yaw_deg)
    rotation = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = scaled @ rotation.T
    return rotated + np.asarray([obj.placement.x, obj.placement.y, obj.placement.z], dtype=float)


def bbox_corners(source_min: np.ndarray, source_max: np.ndarray) -> np.ndarray:
    min_x, min_y, min_z = source_min
    max_x, max_y, max_z = source_max
    return np.asarray(
        [
            [min_x, min_y, min_z],
            [max_x, min_y, min_z],
            [min_x, max_y, min_z],
            [max_x, max_y, min_z],
            [min_x, min_y, max_z],
            [max_x, min_y, max_z],
            [min_x, max_y, max_z],
            [max_x, max_y, max_z],
        ],
        dtype=float,
    )


def proxy_box_points(dimensions: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
    dx, dy, dz = dimensions
    corners = np.asarray(
        [
            [-dx / 2, -dy / 2, -dz / 2],
            [dx / 2, -dy / 2, -dz / 2],
            [-dx / 2, dy / 2, -dz / 2],
            [dx / 2, dy / 2, -dz / 2],
            [-dx / 2, -dy / 2, dz / 2],
            [dx / 2, -dy / 2, dz / 2],
            [-dx / 2, dy / 2, dz / 2],
            [dx / 2, dy / 2, dz / 2],
            [0, 0, -dz / 2],
            [0, 0, dz / 2],
        ],
        dtype=float,
    )
    yaw = math.radians(yaw_deg)
    rotation = np.array(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return corners @ rotation.T


def nearest_points(left: np.ndarray, right: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    best_distance = float("inf")
    best_left = left[0]
    best_right = right[0]
    chunk_size = 128
    for start in range(0, len(left), chunk_size):
        chunk = left[start : start + chunk_size]
        deltas = chunk[:, None, :] - right[None, :, :]
        distances = np.einsum("ijk,ijk->ij", deltas, deltas)
        flat_index = int(np.argmin(distances))
        distance = float(distances.reshape(-1)[flat_index])
        if distance < best_distance:
            row, column = np.unravel_index(flat_index, distances.shape)
            best_distance = distance
            best_left = chunk[row]
            best_right = right[column]
    return math.sqrt(best_distance), best_left, best_right


def support_record(
    obj: ObjectSpec,
    scene: SceneSpec,
    registry: AssetRegistry,
    cloud: ObjectPointCloud,
    support_tolerance: float,
) -> MeshSupportRecord:
    bottom_z = float(cloud.points[:, 2].min())
    mount_kind = mounted_support_kind(obj, registry)
    if mount_kind and mount_kind != "ground_and_wall":
        return MeshSupportRecord(
            object_id=obj.id,
            support_id=mount_kind,
            relation=obj.relation,
            bottom_z=round(bottom_z, 6),
            support_z=round(bottom_z, 6),
            contact_distance=0.0,
            supported=True,
        )
    candidates = support_z_candidates(obj, scene, registry)
    support_z = min(candidates, key=lambda value: abs(value - bottom_z))
    contact_distance = abs(bottom_z - support_z)
    footprint_ok = True
    support_id = obj.parent_id
    if obj.parent_id and obj.relation in {"on", "inside"}:
        footprint_ok = center_inside_support(obj, scene.object_by_id(obj.parent_id), registry)
    supported = contact_distance <= support_tolerance and footprint_ok
    return MeshSupportRecord(
        object_id=obj.id,
        support_id=support_id,
        relation=obj.relation,
        bottom_z=round(bottom_z, 6),
        support_z=round(support_z, 6),
        contact_distance=round(contact_distance, 6),
        supported=supported,
    )


def rounded_point(point: np.ndarray) -> list[float]:
    return [round(float(value), 6) for value in point.tolist()]
