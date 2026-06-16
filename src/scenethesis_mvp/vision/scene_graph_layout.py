from __future__ import annotations

from copy import deepcopy
from typing import Any

from scenethesis_mvp.assets.procedural_assets import normalize_category
from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.stability import snap_all_to_support
from scenethesis_mvp.schemas.scene_spec import ConstraintSpec, ObjectSpec, SceneSpec
from scenethesis_mvp.schemas.vision import VisionSceneGraph


def apply_vision_relations(
    scene: SceneSpec,
    graph: VisionSceneGraph,
    confidence_threshold: float = 0.35,
) -> tuple[SceneSpec, dict[str, Any]]:
    updated = deepcopy(scene)
    object_map = {obj.id: obj for obj in updated.objects}
    matches = {
        obj.id: obj.matched_object_id
        for obj in graph.objects
        if obj.matched_object_id and obj.matched_object_id in object_map and obj.confidence >= confidence_threshold
    }
    diagnostics: dict[str, Any] = {
        "matched_objects": matches,
        "applied_relations": [],
        "ignored_relations": [],
        "anchor_object_id": None,
    }

    anchor_id = graph.anchor_object_id if graph.anchor_object_id in object_map else None
    if anchor_id:
        _set_anchor(updated, anchor_id)
        diagnostics["anchor_object_id"] = anchor_id

    for vision_obj in graph.objects:
        object_id = matches.get(vision_obj.id)
        if not object_id:
            continue
        obj = object_map[object_id]
        obj.category = normalize_category(vision_obj.category)
        if vision_obj.role == "anchor":
            _set_anchor(updated, obj.id)
            diagnostics["anchor_object_id"] = obj.id

    constraints = {constraint.subject_id: constraint for constraint in updated.constraints}
    for relation in graph.relations:
        if relation.confidence < confidence_threshold:
            diagnostics["ignored_relations"].append(relation.model_dump(mode="json"))
            continue
        subject_id = relation.subject_object_id or matches.get(relation.subject_id)
        target_id = relation.target_object_id or (matches.get(relation.target_id) if relation.target_id else None)
        if not subject_id or subject_id not in object_map or (target_id and target_id not in object_map):
            diagnostics["ignored_relations"].append(relation.model_dump(mode="json"))
            continue
        subject = object_map[subject_id]
        if subject.role == "anchor":
            diagnostics["ignored_relations"].append(relation.model_dump(mode="json"))
            continue
        subject.relation = relation.type
        if relation.type in {"on", "inside"} and target_id:
            subject.parent_id = target_id
            subject.role = "child"
        else:
            subject.parent_id = None
            if subject.role == "child":
                subject.role = "parent"
        constraints[subject_id] = ConstraintSpec(type=relation.type, subject_id=subject_id, target_id=target_id)
        diagnostics["applied_relations"].append(
            {"subject_id": subject_id, "type": relation.type, "target_id": target_id, "confidence": relation.confidence}
        )

    updated.constraints = list(constraints.values())
    return SceneSpec.model_validate(updated.model_dump()), diagnostics


def apply_vision_position_hints(
    scene: SceneSpec,
    graph: VisionSceneGraph,
    registry: AssetRegistry,
    confidence_threshold: float = 0.35,
    scene_fill: float = 0.78,
) -> tuple[SceneSpec, dict[str, Any]]:
    updated = deepcopy(scene)
    object_map = {obj.id: obj for obj in updated.objects}
    graph_by_object = {
        obj.matched_object_id: obj
        for obj in graph.objects
        if obj.matched_object_id and obj.matched_object_id in object_map and obj.confidence >= confidence_threshold
    }
    width, depth, _height = updated.bounds
    diagnostics: dict[str, Any] = {"positioned_objects": [], "ignored_objects": []}

    for object_id, vision_obj in graph_by_object.items():
        obj = object_map[object_id]
        if obj.role == "anchor" or obj.parent_id:
            diagnostics["ignored_objects"].append({"object_id": object_id, "reason": "anchor_or_child"})
            continue
        half_x, half_y = _object_half_extents(obj, registry)
        depth_offset = {"foreground": -0.18, "midground": 0.0, "background": 0.18}[vision_obj.depth] * depth
        hinted_x = (vision_obj.bbox.center_x - 0.5) * width * scene_fill
        hinted_y = (0.5 - vision_obj.bbox.center_y) * depth * scene_fill + depth_offset
        obj.placement.x = _clamp(hinted_x, -width * 0.5 + half_x + 0.15, width * 0.5 - half_x - 0.15)
        obj.placement.y = _clamp(hinted_y, -depth * 0.5 + half_y + 0.15, depth * 0.5 - half_y - 0.15)
        diagnostics["positioned_objects"].append(
            {
                "object_id": object_id,
                "x": round(obj.placement.x, 6),
                "y": round(obj.placement.y, 6),
                "depth": vision_obj.depth,
                "bbox": vision_obj.bbox.model_dump(mode="json"),
            }
        )

    return snap_all_to_support(SceneSpec.model_validate(updated.model_dump()), registry), diagnostics


def _set_anchor(scene: SceneSpec, anchor_id: str) -> None:
    for obj in scene.objects:
        if obj.id == anchor_id:
            obj.role = "anchor"
            obj.parent_id = None
            obj.relation = None
        elif obj.role == "anchor":
            obj.role = "parent"
            obj.parent_id = None
            if not obj.relation:
                obj.relation = "near"
    scene.constraints = [constraint for constraint in scene.constraints if constraint.subject_id != anchor_id]


def _object_half_extents(obj: ObjectSpec, registry: AssetRegistry) -> tuple[float, float]:
    if not obj.asset_id:
        raise ValueError(f"object {obj.id} has no asset_id")
    dx, dy, _dz = registry.get(obj.asset_id).scaled_dimensions(obj.placement.scale)
    return dx * 0.5, dy * 0.5


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)
