from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.stability import snap_all_to_support
from scenethesis_mvp.schemas.scene_graph_3d import SceneGraph3D
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import write_json


@dataclass(frozen=True)
class DepthPoseRefinementConfig:
    enabled: bool = True
    apply_scale: bool = True
    apply_yaw: bool = True
    min_extent_m: float = 0.025
    min_yaw_anisotropy: float = 1.25
    max_scale_delta_fraction: float = 0.12
    min_scale: float = 0.20
    max_scale: float = 1.60
    max_scale_estimate_ratio: float = 1.65
    max_yaw_delta_deg: float = 18.0


def apply_depth_pose_refinement(
    scene: SceneSpec,
    graph: SceneGraph3D,
    registry: AssetRegistry,
    out_dir: str | Path,
    cfg: dict[str, Any] | DepthPoseRefinementConfig | None = None,
    artifact_name: str = "depth_pose_refinement.json",
) -> tuple[SceneSpec, dict[str, Any]]:
    config = cfg if isinstance(cfg, DepthPoseRefinementConfig) else DepthPoseRefinementConfig(**(cfg or {}))
    if not config.enabled:
        raise RuntimeError("Depth pose refinement is disabled in faithful config.")

    object_ids = {obj.id for obj in scene.objects}
    pointclouds = {item.object_id: item for item in graph.pointclouds}
    missing = sorted(object_ids - set(pointclouds))
    if missing:
        raise RuntimeError("Depth pose refinement is missing graph point clouds for: " + ", ".join(missing))

    updated = scene.model_copy(deep=True)
    records: list[dict[str, Any]] = []
    applied_scale_updates = 0
    applied_yaw_updates = 0
    for obj in updated.objects:
        if not obj.asset_id:
            raise RuntimeError(f"Depth pose refinement requires an asset_id for {obj.id}.")
        asset = registry.get(obj.asset_id)
        pointcloud = pointclouds[obj.id]
        bbox_size = np.asarray(pointcloud.bbox.size, dtype=np.float64)
        if bbox_size.shape != (3,) or not np.isfinite(bbox_size).all():
            raise RuntimeError(f"Depth pose refinement received invalid bbox size for {obj.id}: {pointcloud.bbox.size}")

        before = obj.placement.model_dump(mode="json")
        scale_record = _depth_scale_update(obj.placement.scale, bbox_size, asset.dimensions, config)
        yaw_record = _depth_yaw_update(obj.placement.yaw_deg, pointcloud.bbox.yaw_deg, bbox_size, config)
        if config.apply_scale and scale_record["applied"]:
            obj.placement.scale = float(scale_record["new_scale"])
            applied_scale_updates += 1
        if config.apply_yaw and yaw_record["applied"]:
            obj.placement.yaw_deg = float(yaw_record["new_yaw_deg"])
            applied_yaw_updates += 1

        records.append(
            {
                "object_id": obj.id,
                "asset_id": asset.id,
                "bbox_size_m": [round(float(value), 6) for value in bbox_size.tolist()],
                "bbox_yaw_deg": round(float(pointcloud.bbox.yaw_deg), 6),
                "before": before,
                "scale_update": scale_record,
                "yaw_update": yaw_record,
                "after": obj.placement.model_dump(mode="json"),
            }
        )

    refined = snap_all_to_support(SceneSpec.model_validate(updated.model_dump()), registry)
    report = {
        "ok": True,
        "provider": "depth_pro_mask_projection",
        "source": "scene_graph_3d.pointclouds[].bbox",
        "applied_scale_updates": applied_scale_updates,
        "applied_yaw_updates": applied_yaw_updates,
        "objects": records,
    }
    write_json(Path(out_dir) / artifact_name, report)
    return refined, report


def _depth_scale_update(
    current_scale: float,
    bbox_size: np.ndarray,
    asset_dimensions: list[float],
    config: DepthPoseRefinementConfig,
) -> dict[str, Any]:
    asset_dims = np.asarray(asset_dimensions, dtype=np.float64)
    estimates: list[dict[str, float | str]] = []
    if bbox_size[1] >= config.min_extent_m and asset_dims[2] > 0:
        estimates.append({"axis": "height", "scale": float(bbox_size[1] / asset_dims[2])})
    depth_footprint = max(float(bbox_size[0]), float(bbox_size[2]))
    asset_footprint = max(float(asset_dims[0]), float(asset_dims[1]))
    if depth_footprint >= config.min_extent_m and asset_footprint > 0:
        estimates.append({"axis": "footprint", "scale": float(depth_footprint / asset_footprint)})
    valid = [float(item["scale"]) for item in estimates if np.isfinite(float(item["scale"])) and float(item["scale"]) > 0]
    if not valid:
        return {
            "applied": False,
            "reason": "no_valid_depth_scale_estimate",
            "estimates": estimates,
            "current_scale": round(float(current_scale), 6),
        }
    raw_target = float(np.median(valid))
    estimate_ratio = max(valid) / max(min(valid), 1e-8)
    if estimate_ratio > config.max_scale_estimate_ratio:
        return {
            "applied": False,
            "reason": "inconsistent_depth_scale_estimates",
            "estimates": estimates,
            "estimate_ratio": round(float(estimate_ratio), 6),
            "raw_target_scale": round(raw_target, 6),
            "current_scale": round(float(current_scale), 6),
        }
    if raw_target < config.min_scale or raw_target > config.max_scale:
        return {
            "applied": False,
            "reason": "raw_depth_scale_outside_bounds",
            "estimates": estimates,
            "estimate_ratio": round(float(estimate_ratio), 6),
            "raw_target_scale": round(raw_target, 6),
            "scale_bounds": [round(float(config.min_scale), 6), round(float(config.max_scale), 6)],
            "current_scale": round(float(current_scale), 6),
        }
    bounded_target = raw_target
    max_delta = max(0.0, float(current_scale) * config.max_scale_delta_fraction)
    delta = float(np.clip(bounded_target - current_scale, -max_delta, max_delta))
    new_scale = float(current_scale + delta)
    return {
        "applied": abs(delta) > 1e-6,
        "reason": "bounded_depth_bbox_scale",
        "estimates": estimates,
        "estimate_ratio": round(float(estimate_ratio), 6),
        "raw_target_scale": round(raw_target, 6),
        "bounded_target_scale": round(bounded_target, 6),
        "delta": round(delta, 6),
        "current_scale": round(float(current_scale), 6),
        "new_scale": round(new_scale, 6),
    }


def _depth_yaw_update(
    current_yaw_deg: float,
    depth_yaw_deg: float,
    bbox_size: np.ndarray,
    config: DepthPoseRefinementConfig,
) -> dict[str, Any]:
    horizontal = sorted([max(float(bbox_size[0]), config.min_extent_m), max(float(bbox_size[2]), config.min_extent_m)])
    anisotropy = horizontal[1] / horizontal[0]
    if anisotropy < config.min_yaw_anisotropy:
        return {
            "applied": False,
            "reason": "depth_bbox_not_directional",
            "anisotropy": round(float(anisotropy), 6),
            "current_yaw_deg": round(float(current_yaw_deg), 6),
            "depth_yaw_deg": round(float(depth_yaw_deg), 6),
        }
    delta = angle_delta_deg(current_yaw_deg, depth_yaw_deg)
    clipped_delta = float(np.clip(delta, -config.max_yaw_delta_deg, config.max_yaw_delta_deg))
    new_yaw = (float(current_yaw_deg) + clipped_delta) % 360.0
    return {
        "applied": abs(clipped_delta) > 1e-6,
        "reason": "bounded_depth_bbox_yaw",
        "anisotropy": round(float(anisotropy), 6),
        "raw_delta_deg": round(float(delta), 6),
        "delta_deg": round(float(clipped_delta), 6),
        "current_yaw_deg": round(float(current_yaw_deg), 6),
        "depth_yaw_deg": round(float(depth_yaw_deg), 6),
        "new_yaw_deg": round(float(new_yaw), 6),
    }


def angle_delta_deg(current: float, target: float) -> float:
    return float((target - current + 180.0) % 360.0 - 180.0)
