from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.stability import mounted_support_kind, snap_all_to_support
from scenethesis_mvp.schemas.scene_graph_3d import SceneGraph3D
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec
from scenethesis_mvp.utils.io import read_json, write_json
from scenethesis_mvp.vision.depth_pose_refinement import DepthPoseRefinementConfig, _depth_scale_update, angle_delta_deg


@dataclass(frozen=True)
class JointPoseOptimizerConfig:
    enabled: bool = True
    provider: str = "depth_roma_joint"
    max_iters: int = 8
    learning_rate: float = 0.45
    max_translation_step_m: float = 0.04
    max_scale_step_fraction: float = 0.05
    max_yaw_step_deg: float = 5.0
    min_yaw_anisotropy: float = 1.25
    min_correspondences: int = 12
    min_mean_confidence: float = 0.60
    depth_position_weight: float = 0.35
    depth_scale_weight: float = 0.85
    depth_yaw_weight: float = 0.45
    roma_yaw_weight: float = 0.65
    movable_scene_margin_fraction: float = 0.14


@dataclass(frozen=True)
class ObjectTargets:
    depth_xy: tuple[float, float] | None
    depth_scale: float | None
    depth_yaw_deg: float | None
    roma_yaw_deg: float | None
    roma_yaw_delta_deg: float | None
    correspondence_path: Path
    match_count: int
    mean_confidence: float


def run_joint_pose_optimizer(
    scene: SceneSpec,
    graph: SceneGraph3D,
    registry: AssetRegistry,
    out_dir: str | Path,
    cfg: dict[str, Any] | JointPoseOptimizerConfig | None = None,
) -> tuple[SceneSpec, dict[str, Any]]:
    config = cfg if isinstance(cfg, JointPoseOptimizerConfig) else JointPoseOptimizerConfig(**(cfg or {}))
    if not config.enabled:
        raise RuntimeError("Joint pose optimizer is disabled in faithful config.")
    if config.provider != "depth_roma_joint":
        raise RuntimeError(f"Unsupported joint pose optimizer provider: {config.provider}")

    target_dir = Path(out_dir)
    updated = scene.model_copy(deep=True)
    targets = build_object_targets(updated, graph, registry, target_dir, config)
    history: list[dict[str, Any]] = []
    object_records: dict[str, dict[str, Any]] = {
        obj.id: {
            "object_id": obj.id,
            "asset_id": obj.asset_id,
            "initial": obj.placement.model_dump(mode="json"),
            "targets": serialize_targets(targets[obj.id]),
            "accepted_updates": [],
            "rejected_updates": [],
        }
        for obj in updated.objects
    }

    initial_loss = scene_loss(updated, targets, registry, config)
    applied_translation_updates = 0
    applied_scale_updates = 0
    applied_yaw_updates = 0
    for iteration in range(config.max_iters):
        before_total = scene_loss(updated, targets, registry, config)
        iteration_record: dict[str, Any] = {
            "iteration": iteration,
            "loss_before": round(before_total["total_loss"], 8),
            "objects": [],
        }
        changed = False
        for obj in updated.objects:
            before_loss = object_loss(obj, targets[obj.id], registry, config)
            proposal = propose_object_update(obj, targets[obj.id], updated, registry, config)
            if not proposal["has_update"]:
                continue
            candidate = updated.model_copy(deep=True)
            candidate_obj = candidate.object_by_id(obj.id)
            apply_proposal(candidate_obj, proposal)
            clamp_to_scene(candidate_obj, candidate, registry)
            candidate = snap_all_to_support(candidate, registry)
            candidate_obj = candidate.object_by_id(obj.id)
            after_loss = object_loss(candidate_obj, targets[obj.id], registry, config)
            event = {
                "object_id": obj.id,
                "loss_before": round(before_loss["total_loss"], 8),
                "loss_after": round(after_loss["total_loss"], 8),
                "proposal": proposal,
            }
            if after_loss["total_loss"] <= before_loss["total_loss"] - 1e-9:
                updated = candidate
                obj = updated.object_by_id(obj.id)
                changed = True
                applied_translation_updates += int(bool(proposal.get("translation_delta_m")))
                applied_scale_updates += int(abs(float(proposal.get("scale_delta", 0.0))) > 1e-9)
                applied_yaw_updates += int(abs(float(proposal.get("yaw_delta_deg", 0.0))) > 1e-9)
                object_records[obj.id]["accepted_updates"].append(event)
                event["accepted"] = True
            else:
                object_records[obj.id]["rejected_updates"].append(event)
                event["accepted"] = False
            iteration_record["objects"].append(event)
        after_total = scene_loss(updated, targets, registry, config)
        iteration_record["loss_after"] = round(after_total["total_loss"], 8)
        history.append(iteration_record)
        if not changed or abs(before_total["total_loss"] - after_total["total_loss"]) < 1e-8:
            break

    final_loss = scene_loss(updated, targets, registry, config)
    for obj in updated.objects:
        object_records[obj.id]["final"] = obj.placement.model_dump(mode="json")
        object_records[obj.id]["final_loss"] = object_loss(obj, targets[obj.id], registry, config)

    report = {
        "ok": final_loss["total_loss"] <= initial_loss["total_loss"] + 1e-8,
        "provider": config.provider,
        "method": "bounded SGD-style 5-DoF updates from Depth Pro graph boxes and RoMa correspondence residuals",
        "max_iters": config.max_iters,
        "learning_rate": config.learning_rate,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "applied_updates": applied_translation_updates + applied_scale_updates + applied_yaw_updates,
        "applied_translation_updates": applied_translation_updates,
        "applied_scale_updates": applied_scale_updates,
        "applied_yaw_updates": applied_yaw_updates,
        "objects": list(object_records.values()),
    }
    write_json(target_dir / "joint_pose_optimizer.json", report)
    write_json(
        target_dir / "pose_loss_history.json",
        {
            "provider": config.provider,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "iterations": history,
        },
    )
    if not report["ok"]:
        raise RuntimeError("Joint pose optimizer increased total loss; refusing to continue.")
    return SceneSpec.model_validate(updated.model_dump()), report


def build_object_targets(
    scene: SceneSpec,
    graph: SceneGraph3D,
    registry: AssetRegistry,
    out_dir: Path,
    config: JointPoseOptimizerConfig,
) -> dict[str, ObjectTargets]:
    object_ids = {obj.id for obj in scene.objects}
    pointclouds = {item.object_id: item for item in graph.pointclouds}
    missing_graph = sorted(object_ids - set(pointclouds))
    if missing_graph:
        raise RuntimeError("Joint pose optimizer is missing graph point clouds for: " + ", ".join(missing_graph))
    correspondence_records = load_correspondence_records(scene, out_dir, config)
    xy_targets = depth_xy_targets(scene, graph, registry, config)
    targets: dict[str, ObjectTargets] = {}
    for obj in scene.objects:
        if not obj.asset_id:
            raise RuntimeError(f"Joint pose optimizer requires an asset_id for {obj.id}.")
        pointcloud = pointclouds[obj.id]
        bbox_size = np.asarray(pointcloud.bbox.size, dtype=np.float64)
        if bbox_size.shape != (3,) or not np.isfinite(bbox_size).all():
            raise RuntimeError(f"Joint pose optimizer received invalid bbox for {obj.id}: {pointcloud.bbox.size}")
        asset = registry.get(obj.asset_id)
        scale_target = depth_scale_target(obj.placement.scale, bbox_size, asset.dimensions, config)
        yaw_target = depth_yaw_target(pointcloud.bbox.yaw_deg, bbox_size, config)
        correspondence = correspondence_records[obj.id]
        targets[obj.id] = ObjectTargets(
            depth_xy=xy_targets.get(obj.id),
            depth_scale=scale_target,
            depth_yaw_deg=yaw_target,
            roma_yaw_deg=(obj.placement.yaw_deg + correspondence["yaw_delta_deg"]) % 360.0,
            roma_yaw_delta_deg=correspondence["yaw_delta_deg"],
            correspondence_path=correspondence["path"],
            match_count=correspondence["match_count"],
            mean_confidence=correspondence["mean_confidence"],
        )
    return targets


def load_correspondence_records(
    scene: SceneSpec,
    out_dir: Path,
    config: JointPoseOptimizerConfig,
) -> dict[str, dict[str, Any]]:
    diagnostics_path = out_dir / "correspondence_diagnostics.json"
    if not diagnostics_path.is_file():
        raise RuntimeError(f"Joint pose optimizer requires RoMa diagnostics: {diagnostics_path}")
    diagnostics = read_json(diagnostics_path)
    if diagnostics.get("provider") != "roma":
        raise RuntimeError("Joint pose optimizer requires RoMa correspondence diagnostics.")
    records = {item.get("object_id"): item for item in diagnostics.get("objects", []) if item.get("object_id")}
    missing = sorted(obj.id for obj in scene.objects if obj.id not in records)
    if missing:
        raise RuntimeError("Joint pose optimizer is missing RoMa records for: " + ", ".join(missing))
    loaded: dict[str, dict[str, Any]] = {}
    for obj in scene.objects:
        record = records[obj.id]
        if record.get("status") != "ok":
            raise RuntimeError(f"Joint pose optimizer cannot use failed RoMa record for {obj.id}: {record.get('status')}")
        path = Path(record.get("correspondence_path") or out_dir / "correspondences" / f"{obj.id}.npz")
        if not path.is_file():
            raise RuntimeError(f"Joint pose optimizer requires correspondence file for {obj.id}: {path}")
        with np.load(path) as data:
            confidence = np.asarray(data["confidence"], dtype=np.float64)
            if "guidance_xy" not in data or "rendered_xy" not in data:
                raise RuntimeError(f"Correspondence file is missing keypoints for {obj.id}: {path}")
            match_count = int(len(confidence))
            mean_confidence = float(np.mean(confidence)) if len(confidence) else 0.0
        if match_count < config.min_correspondences or mean_confidence < config.min_mean_confidence:
            raise RuntimeError(
                f"RoMa correspondence for {obj.id} is below optimizer threshold: "
                f"matches={match_count}, mean_confidence={mean_confidence:.3f}"
            )
        loaded[obj.id] = {
            "path": path,
            "match_count": match_count,
            "mean_confidence": mean_confidence,
            "yaw_delta_deg": float(record.get("yaw_delta_deg") or 0.0),
        }
    return loaded


def depth_xy_targets(
    scene: SceneSpec,
    graph: SceneGraph3D,
    registry: AssetRegistry,
    config: JointPoseOptimizerConfig,
) -> dict[str, tuple[float, float]]:
    poses = {pose.object_id: pose for pose in graph.poses}
    movable = [obj for obj in scene.objects if obj.id in poses and can_translate_xy(obj, scene, registry)]
    if len(movable) < 2:
        return {}
    graph_x = np.asarray([poses[obj.id].x for obj in movable], dtype=np.float64)
    graph_y = np.asarray([poses[obj.id].y for obj in movable], dtype=np.float64)
    if not np.isfinite(graph_x).all() or not np.isfinite(graph_y).all():
        return {}
    width, depth, _height = [float(value) for value in scene.bounds]
    margin_x = width * config.movable_scene_margin_fraction
    margin_y = depth * config.movable_scene_margin_fraction
    x_targets = normalize_to_range(graph_x, margin_x, width - margin_x)
    y_targets = normalize_to_range(graph_y, margin_y, depth - margin_y)
    return {obj.id: (float(x_targets[index]), float(y_targets[index])) for index, obj in enumerate(movable)}


def normalize_to_range(values: np.ndarray, low: float, high: float) -> np.ndarray:
    if float(values.max() - values.min()) < 1e-8:
        return np.full_like(values, (low + high) * 0.5)
    normalized = (values - values.min()) / (values.max() - values.min())
    return low + normalized * (high - low)


def can_translate_xy(obj: ObjectSpec, scene: SceneSpec, registry: AssetRegistry) -> bool:
    if obj.role == "anchor":
        return False
    if obj.parent_id and obj.relation in {"on", "inside"}:
        return False
    return mounted_support_kind(obj, registry) in {None, "ground_and_wall"}


def depth_scale_target(
    current_scale: float,
    bbox_size: np.ndarray,
    asset_dimensions: list[float],
    config: JointPoseOptimizerConfig,
) -> float | None:
    depth_config = DepthPoseRefinementConfig(
        max_scale_delta_fraction=config.max_scale_step_fraction,
        min_yaw_anisotropy=config.min_yaw_anisotropy,
    )
    record = _depth_scale_update(current_scale, bbox_size, asset_dimensions, depth_config)
    if not record.get("applied") and record.get("reason") not in {"bounded_depth_bbox_scale"}:
        return None
    return float(record.get("bounded_target_scale") or record.get("new_scale"))


def depth_yaw_target(depth_yaw_deg: float, bbox_size: np.ndarray, config: JointPoseOptimizerConfig) -> float | None:
    horizontal = sorted([max(float(bbox_size[0]), 0.025), max(float(bbox_size[2]), 0.025)])
    if horizontal[1] / horizontal[0] < config.min_yaw_anisotropy:
        return None
    return float(depth_yaw_deg % 360.0)


def propose_object_update(
    obj: ObjectSpec,
    target: ObjectTargets,
    scene: SceneSpec,
    registry: AssetRegistry,
    config: JointPoseOptimizerConfig,
) -> dict[str, Any]:
    proposal: dict[str, Any] = {"has_update": False}
    if target.depth_xy is not None and can_translate_xy(obj, scene, registry):
        delta_xy = np.asarray(target.depth_xy, dtype=np.float64) - np.asarray([obj.placement.x, obj.placement.y], dtype=np.float64)
        raw_step = delta_xy * config.learning_rate
        norm = float(np.linalg.norm(raw_step))
        if norm > config.max_translation_step_m:
            raw_step *= config.max_translation_step_m / norm
        if float(np.linalg.norm(raw_step)) > 1e-6:
            proposal["translation_delta_m"] = [round(float(raw_step[0]), 6), round(float(raw_step[1]), 6), 0.0]
            proposal["has_update"] = True
    if target.depth_scale is not None:
        max_delta = obj.placement.scale * config.max_scale_step_fraction
        scale_delta = float(np.clip((target.depth_scale - obj.placement.scale) * config.learning_rate, -max_delta, max_delta))
        if abs(scale_delta) > 1e-6:
            proposal["scale_delta"] = round(scale_delta, 6)
            proposal["has_update"] = True
    yaw_deltas: list[tuple[float, float]] = []
    if target.depth_yaw_deg is not None:
        yaw_deltas.append((angle_delta_deg(obj.placement.yaw_deg, target.depth_yaw_deg), config.depth_yaw_weight))
    if target.roma_yaw_deg is not None:
        yaw_deltas.append((angle_delta_deg(obj.placement.yaw_deg, target.roma_yaw_deg), config.roma_yaw_weight))
    if yaw_deltas:
        weighted = sum(delta * weight for delta, weight in yaw_deltas)
        total_weight = sum(weight for _delta, weight in yaw_deltas)
        yaw_delta = float(np.clip((weighted / max(total_weight, 1e-8)) * config.learning_rate, -config.max_yaw_step_deg, config.max_yaw_step_deg))
        if abs(yaw_delta) > 1e-6:
            proposal["yaw_delta_deg"] = round(yaw_delta, 6)
            proposal["has_update"] = True
    return proposal


def apply_proposal(obj: ObjectSpec, proposal: dict[str, Any]) -> None:
    if proposal.get("translation_delta_m"):
        dx, dy, _dz = proposal["translation_delta_m"]
        obj.placement.x += float(dx)
        obj.placement.y += float(dy)
    if proposal.get("scale_delta") is not None:
        obj.placement.scale = max(0.051, float(obj.placement.scale + float(proposal["scale_delta"])))
    if proposal.get("yaw_delta_deg") is not None:
        obj.placement.yaw_deg = float((obj.placement.yaw_deg + float(proposal["yaw_delta_deg"])) % 360.0)


def clamp_to_scene(obj: ObjectSpec, scene: SceneSpec, registry: AssetRegistry) -> None:
    if not obj.asset_id:
        return
    width, depth, height = [float(value) for value in scene.bounds]
    dx, dy, dz = registry.get(obj.asset_id).scaled_dimensions(obj.placement.scale)
    obj.placement.x = min(max(float(obj.placement.x), dx * 0.5), width - dx * 0.5)
    obj.placement.y = min(max(float(obj.placement.y), dy * 0.5), depth - dy * 0.5)
    obj.placement.z = min(max(float(obj.placement.z), dz * 0.5), height - dz * 0.5)


def scene_loss(
    scene: SceneSpec,
    targets: dict[str, ObjectTargets],
    registry: AssetRegistry,
    config: JointPoseOptimizerConfig,
) -> dict[str, Any]:
    object_losses = [object_loss(obj, targets[obj.id], registry, config) for obj in scene.objects]
    return {
        "total_loss": round(float(sum(item["total_loss"] for item in object_losses)), 8),
        "depth_position_loss": round(float(sum(item["depth_position_loss"] for item in object_losses)), 8),
        "depth_scale_loss": round(float(sum(item["depth_scale_loss"] for item in object_losses)), 8),
        "depth_yaw_loss": round(float(sum(item["depth_yaw_loss"] for item in object_losses)), 8),
        "roma_yaw_loss": round(float(sum(item["roma_yaw_loss"] for item in object_losses)), 8),
    }


def object_loss(
    obj: ObjectSpec,
    target: ObjectTargets,
    registry: AssetRegistry,
    config: JointPoseOptimizerConfig,
) -> dict[str, float]:
    depth_position_loss = 0.0
    if target.depth_xy is not None:
        delta = np.asarray([obj.placement.x - target.depth_xy[0], obj.placement.y - target.depth_xy[1]], dtype=np.float64)
        depth_position_loss = float(np.dot(delta, delta)) * config.depth_position_weight
    depth_scale_loss = 0.0
    if target.depth_scale is not None:
        depth_scale_loss = float((np.log(max(obj.placement.scale, 1e-6) / max(target.depth_scale, 1e-6))) ** 2) * config.depth_scale_weight
    depth_yaw_loss = 0.0
    if target.depth_yaw_deg is not None:
        depth_yaw_loss = float((angle_delta_deg(obj.placement.yaw_deg, target.depth_yaw_deg) / 180.0) ** 2) * config.depth_yaw_weight
    roma_yaw_loss = 0.0
    if target.roma_yaw_deg is not None:
        roma_yaw_loss = float((angle_delta_deg(obj.placement.yaw_deg, target.roma_yaw_deg) / 180.0) ** 2) * config.roma_yaw_weight
    total = depth_position_loss + depth_scale_loss + depth_yaw_loss + roma_yaw_loss
    return {
        "total_loss": round(total, 8),
        "depth_position_loss": round(depth_position_loss, 8),
        "depth_scale_loss": round(depth_scale_loss, 8),
        "depth_yaw_loss": round(depth_yaw_loss, 8),
        "roma_yaw_loss": round(roma_yaw_loss, 8),
    }


def serialize_targets(target: ObjectTargets) -> dict[str, Any]:
    return {
        "depth_xy": [round(float(value), 6) for value in target.depth_xy] if target.depth_xy else None,
        "depth_scale": round(float(target.depth_scale), 6) if target.depth_scale is not None else None,
        "depth_yaw_deg": round(float(target.depth_yaw_deg), 6) if target.depth_yaw_deg is not None else None,
        "roma_yaw_deg": round(float(target.roma_yaw_deg), 6) if target.roma_yaw_deg is not None else None,
        "roma_yaw_delta_deg": round(float(target.roma_yaw_delta_deg), 6) if target.roma_yaw_delta_deg is not None else None,
        "correspondence_path": str(target.correspondence_path),
        "match_count": target.match_count,
        "mean_confidence": round(float(target.mean_confidence), 6),
    }
