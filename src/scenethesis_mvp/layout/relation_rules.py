from __future__ import annotations

import math

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.collision import object_aabb
from scenethesis_mvp.schemas.scene_spec import ConstraintSpec, ObjectSpec, SceneSpec


RELATION_DIRECTIONS: dict[str, tuple[float, float]] = {
    "right_of": (1.0, 0.0),
    "left_of": (-1.0, 0.0),
    "in_front_of": (0.0, -1.0),
    "behind": (0.0, 1.0),
    "next_to": (1.0, 0.0),
    "near": (1.0, -0.5),
    "against_wall": (0.0, 1.0),
}


def local_to_world(parent: ObjectSpec, local_x: float, local_y: float) -> tuple[float, float]:
    yaw = math.radians(parent.placement.yaw_deg)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        parent.placement.x + cos_yaw * local_x - sin_yaw * local_y,
        parent.placement.y + sin_yaw * local_x + cos_yaw * local_y,
    )


def place_relative_to_target(
    obj: ObjectSpec,
    target: ObjectSpec,
    scene: SceneSpec,
    registry: AssetRegistry,
    relation: str | None,
    clearance: float = 0.28,
) -> None:
    relation = relation or "next_to"
    target_box = object_aabb(target, registry)
    obj_box = object_aabb(obj, registry)
    obj_half_x = (obj_box.max_x - obj_box.min_x) * 0.5
    obj_half_y = (obj_box.max_y - obj_box.min_y) * 0.5
    target_half_x = (target_box.max_x - target_box.min_x) * 0.5
    target_half_y = (target_box.max_y - target_box.min_y) * 0.5
    dx, dy = RELATION_DIRECTIONS.get(relation, RELATION_DIRECTIONS["next_to"])
    if relation == "against_wall":
        _, depth, _ = scene.bounds
        obj.placement.x = target.placement.x
        obj.placement.y = depth * 0.5 - obj_half_y - 0.15
        obj.placement.yaw_deg = 180.0
        return
    distance_x = target_half_x + obj_half_x + clearance
    distance_y = target_half_y + obj_half_y + clearance
    obj.placement.x = target.placement.x + dx * distance_x
    obj.placement.y = target.placement.y + dy * distance_y
    if relation == "facing":
        obj.placement.yaw_deg = (target.placement.yaw_deg + 180.0) % 360.0


def relation_penalty(scene: SceneSpec, registry: AssetRegistry) -> float:
    penalty = 0.0
    for constraint in scene.constraints:
        if not constraint.target_id:
            continue
        subject = scene.object_by_id(constraint.subject_id)
        target = scene.object_by_id(constraint.target_id)
        dx = subject.placement.x - target.placement.x
        dy = subject.placement.y - target.placement.y
        if constraint.type == "left_of" and dx >= 0:
            penalty += abs(dx) + 1.0
        elif constraint.type == "right_of" and dx <= 0:
            penalty += abs(dx) + 1.0
        elif constraint.type == "in_front_of" and dy >= 0:
            penalty += abs(dy) + 1.0
        elif constraint.type == "behind" and dy <= 0:
            penalty += abs(dy) + 1.0
        elif constraint.type in {"on", "inside"} and subject.parent_id != target.id:
            penalty += 2.0
    return round(penalty, 6)


def normalize_support_relation_semantics(scene: SceneSpec, registry: AssetRegistry) -> SceneSpec:
    """Use visible support relations for open shelves while preserving true containers."""

    updated = scene.model_copy(deep=True)
    objects = {obj.id: obj for obj in updated.objects}
    constraints = {constraint.subject_id: constraint for constraint in updated.constraints}
    floor_scale_categories = {"pallet", "pallet_load", "forklift", "cart", "barrier", "floor_marking"}
    for obj in updated.objects:
        if obj.parent_id and obj.relation in {"on", "inside"} and obj.category in floor_scale_categories:
            previous_parent_id = obj.parent_id
            obj.parent_id = None
            obj.role = "parent"
            obj.relation = "near"
            constraint = constraints.get(obj.id)
            if constraint:
                constraint.type = "near"
                constraint.target_id = previous_parent_id
            else:
                constraints[obj.id] = ConstraintSpec(type="near", subject_id=obj.id, target_id=previous_parent_id)
            continue
        if not obj.parent_id or obj.relation != "inside":
            continue
        parent = objects.get(obj.parent_id)
        if parent is None:
            continue
        if parent.category != "shelf":
            continue
        obj.relation = "on"
        constraint = constraints.get(obj.id)
        if constraint:
            constraint.type = "on"
            constraint.target_id = parent.id
        else:
            constraints[obj.id] = ConstraintSpec(type="on", subject_id=obj.id, target_id=parent.id)
    updated.constraints = list(constraints.values())
    return SceneSpec.model_validate(updated.model_dump())
