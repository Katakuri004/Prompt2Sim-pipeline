from __future__ import annotations

import math
from copy import deepcopy

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.relation_rules import local_to_world, place_relative_to_target
from scenethesis_mvp.layout.stability import snap_all_to_support
from scenethesis_mvp.schemas.scene_spec import ConstraintSpec, ObjectSpec, SceneSpec


def ordered_objects(scene: SceneSpec) -> list[ObjectSpec]:
    anchor = [obj for obj in scene.objects if obj.role == "anchor"]
    parents = [obj for obj in scene.objects if obj.role != "anchor" and not obj.parent_id]
    children = [obj for obj in scene.objects if obj.parent_id]
    return anchor + parents + children


def _asset_dims(obj: ObjectSpec, registry: AssetRegistry) -> tuple[float, float, float]:
    if not obj.asset_id:
        raise ValueError(f"object {obj.id} has no asset_id")
    return registry.get(obj.asset_id).scaled_dimensions(obj.placement.scale)


def _place_child_on_parent(
    child: ObjectSpec,
    parent: ObjectSpec,
    registry: AssetRegistry,
    sibling_index: int,
    sibling_count: int,
) -> None:
    parent_asset = registry.get(parent.asset_id or "")
    parent_dx, parent_dy, parent_dz = _asset_dims(parent, registry)
    child_dx, child_dy, child_dz = _asset_dims(child, registry)
    usable_x = max(0.05, parent_dx - child_dx - 0.18)
    usable_y = max(0.05, parent_dy - child_dy - 0.18)
    if parent_asset.support_heights and parent_asset.support_kind == "container":
        levels = parent_asset.support_heights
        level_index = sibling_index % len(levels)
        column_index = sibling_index // len(levels)
        columns = max(1, math.ceil(sibling_count / len(levels)))
        if columns == 1:
            local_x = 0.0
        else:
            local_x = -usable_x * 0.5 + usable_x * (column_index / max(1, columns - 1))
        local_y = 0.0
        support_z = parent.placement.z - parent_dz * 0.5 + parent_dz * levels[level_index]
    else:
        cols = max(1, math.ceil(math.sqrt(sibling_count)))
        rows = max(1, math.ceil(sibling_count / cols))
        col = sibling_index % cols
        row = sibling_index // cols
        local_x = 0.0 if cols == 1 else -usable_x * 0.5 + usable_x * (col / (cols - 1))
        local_y = 0.0 if rows == 1 else -usable_y * 0.5 + usable_y * (row / (rows - 1))
        support_z = parent.placement.z + parent_dz * 0.5
    child.placement.x, child.placement.y = local_to_world(parent, local_x, local_y)
    child.placement.z = support_z + child_dz * 0.5
    child.placement.yaw_deg = parent.placement.yaw_deg


def generate_initial_layout(scene: SceneSpec, registry: AssetRegistry) -> SceneSpec:
    updated = deepcopy(scene)
    _normalize_relations(updated)
    objects = {obj.id: obj for obj in updated.objects}
    anchor = next(obj for obj in updated.objects if obj.role == "anchor")
    anchor.placement.x = 0.0
    anchor.placement.y = 0.0
    anchor.placement.yaw_deg = 0.0
    _, _, anchor_dz = _asset_dims(anchor, registry)
    anchor.placement.z = anchor_dz * 0.5

    parent_slots = [
        ("behind", 0.0),
        ("left_of", 0.0),
        ("right_of", 0.0),
        ("in_front_of", 180.0),
        ("near", 45.0),
        ("against_wall", 180.0),
    ]
    parent_index = 0
    near_counts: dict[str, int] = {}
    near_slots = ["right_of", "left_of", "in_front_of", "behind", "near"]
    for obj in ordered_objects(updated):
        if obj.id == anchor.id or obj.parent_id:
            continue
        relation = obj.relation
        target_id = None
        for constraint in updated.constraints:
            if constraint.subject_id == obj.id and constraint.target_id:
                relation = constraint.type
                target_id = constraint.target_id
                break
        if not relation:
            relation, yaw = parent_slots[parent_index % len(parent_slots)]
            obj.placement.yaw_deg = yaw
        target = objects.get(target_id or anchor.id, anchor)
        effective_relation = relation
        if relation in {"near", "next_to"}:
            count_key = target.id
            count = near_counts.get(count_key, 0)
            effective_relation = near_slots[count % len(near_slots)]
            near_counts[count_key] = count + 1
        place_relative_to_target(obj, target, updated, registry, effective_relation)
        _, _, dz = _asset_dims(obj, registry)
        obj.placement.z = dz * 0.5
        parent_index += 1

    child_groups: dict[str, list[ObjectSpec]] = {}
    for obj in ordered_objects(updated):
        if obj.parent_id and obj.relation in {"on", "inside"}:
            child_groups.setdefault(obj.parent_id, []).append(obj)
    for parent_id, children in child_groups.items():
        parent = objects[parent_id]
        inside_children = [child for child in children if child.relation == "inside"]
        other_children = [child for child in children if child.relation != "inside"]
        for index, child in enumerate(inside_children):
            _place_child_on_parent(child, parent, registry, index, len(inside_children))
        for index, child in enumerate(other_children):
            _place_child_on_parent(child, parent, registry, index, len(other_children))

    return snap_all_to_support(SceneSpec.model_validate(updated.model_dump()), registry)


def _normalize_relations(scene: SceneSpec) -> None:
    constraints = {constraint.subject_id: constraint for constraint in scene.constraints}
    for obj in scene.objects:
        if obj.role == "anchor":
            continue
        if obj.parent_id and obj.relation not in {"on", "inside"}:
            target_id = obj.parent_id
            obj.parent_id = None
            if obj.role == "child":
                obj.role = "parent"
            if obj.relation:
                constraints[obj.id] = ConstraintSpec(type=obj.relation, subject_id=obj.id, target_id=target_id)
        elif obj.relation and obj.id not in constraints:
            constraints[obj.id] = ConstraintSpec(type=obj.relation, subject_id=obj.id, target_id=obj.parent_id)
    scene.constraints = list(constraints.values())
