from __future__ import annotations

import math
from copy import deepcopy

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.collision import boundary_violations, detect_collisions, object_aabb
from scenethesis_mvp.layout.initial_layout import _place_child_on_parent
from scenethesis_mvp.layout.relation_rules import RELATION_DIRECTIONS, place_relative_to_target, relation_penalty
from scenethesis_mvp.layout.stability import check_stability, snap_all_to_support
from scenethesis_mvp.schemas.metrics import Metrics
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec


ROLE_PRIORITY = {"anchor": 0, "parent": 1, "child": 2}
DIRECTIONAL_RELATIONS = {"left_of", "right_of", "in_front_of", "behind", "against_wall", "facing"}


def compute_metrics(
    scene: SceneSpec,
    registry: AssetRegistry,
    boundary_margin: float = 0.0,
    contact_tolerance: float = 0.04,
) -> Metrics:
    collisions = detect_collisions(scene, registry)
    unstable = check_stability(scene, registry, contact_tolerance=contact_tolerance)
    floating = [record for record in unstable if record.reason == "floating_or_sinking"]
    unsupported = [record for record in unstable if record.reason != "floating_or_sinking"]
    boundary_count = boundary_violations(scene, registry, margin=boundary_margin)
    rel_penalty = relation_penalty(scene, registry)
    collision_penalty = sum(max(0.01, item.penetration) for item in collisions)
    support_penalty = sum(max(0.01, item.distance) for item in unstable)
    boundary_penalty = float(boundary_count)
    total = collision_penalty * 10.0 + support_penalty * 8.0 + boundary_penalty * 5.0 + rel_penalty
    return Metrics(
        object_count=len(scene.objects),
        collision_count=len(collisions),
        collisions=collisions,
        floating_count=len(floating),
        unsupported_count=len(unsupported),
        unstable=unstable,
        boundary_violations=boundary_count,
        relation_penalty=rel_penalty,
        collision_penalty=round(collision_penalty, 6),
        support_penalty=round(support_penalty, 6),
        boundary_penalty=boundary_penalty,
        total_penalty=round(total, 6),
    )


def _clamp_to_bounds(obj: ObjectSpec, scene: SceneSpec, registry: AssetRegistry, margin: float) -> None:
    width, depth, height = scene.bounds
    box = object_aabb(obj, registry)
    half_x = (box.max_x - box.min_x) * 0.5
    half_y = (box.max_y - box.min_y) * 0.5
    half_z = (box.max_z - box.min_z) * 0.5
    obj.placement.x = min(max(obj.placement.x, -width * 0.5 + half_x + margin), width * 0.5 - half_x - margin)
    obj.placement.y = min(max(obj.placement.y, -depth * 0.5 + half_y + margin), depth * 0.5 - half_y - margin)
    obj.placement.z = min(max(obj.placement.z, half_z), height - half_z)


def _object_to_move(a: ObjectSpec, b: ObjectSpec) -> ObjectSpec:
    if ROLE_PRIORITY[a.role] > ROLE_PRIORITY[b.role]:
        return a
    if ROLE_PRIORITY[b.role] > ROLE_PRIORITY[a.role]:
        return b
    return b if b.id > a.id else a


def _separate_pair(scene: SceneSpec, registry: AssetRegistry, left: ObjectSpec, right: ObjectSpec) -> None:
    move = _object_to_move(left, right)
    other = right if move.id == left.id else left
    move_box = object_aabb(move, registry)
    other_box = object_aabb(other, registry)
    overlap_x = min(move_box.max_x, other_box.max_x) - max(move_box.min_x, other_box.min_x)
    overlap_y = min(move_box.max_y, other_box.max_y) - max(move_box.min_y, other_box.min_y)
    direction_x = 1.0 if move.placement.x >= other.placement.x else -1.0
    direction_y = 1.0 if move.placement.y >= other.placement.y else -1.0
    if abs(move.placement.x - other.placement.x) < 1e-6:
        direction_x = 1.0
    if abs(move.placement.y - other.placement.y) < 1e-6:
        direction_y = 1.0
    if overlap_x < overlap_y:
        move.placement.x += direction_x * (overlap_x + 0.08)
    else:
        move.placement.y += direction_y * (overlap_y + 0.08)


def _directional_relation_satisfied(subject: ObjectSpec, target: ObjectSpec, relation: str) -> bool:
    dx = subject.placement.x - target.placement.x
    dy = subject.placement.y - target.placement.y
    if relation == "left_of":
        return dx < 0
    if relation == "right_of":
        return dx > 0
    if relation == "in_front_of":
        return dy < 0
    if relation == "behind":
        return dy > 0
    return True


def _enforce_directional_relations(scene: SceneSpec, registry: AssetRegistry) -> None:
    object_map = {obj.id: obj for obj in scene.objects}
    for constraint in scene.constraints:
        if constraint.type not in DIRECTIONAL_RELATIONS or not constraint.target_id:
            continue
        subject = object_map.get(constraint.subject_id)
        target = object_map.get(constraint.target_id)
        if not subject or not target or subject.parent_id or subject.role == "anchor":
            continue
        if not _directional_relation_satisfied(subject, target, constraint.type):
            place_relative_to_target(subject, target, scene, registry, constraint.type)


def _object_within_bounds(obj: ObjectSpec, scene: SceneSpec, registry: AssetRegistry, margin: float) -> bool:
    width, depth, height = scene.bounds
    box = object_aabb(obj, registry)
    return (
        box.min_x >= -width * 0.5 + margin
        and box.max_x <= width * 0.5 - margin
        and box.min_y >= -depth * 0.5 + margin
        and box.max_y <= depth * 0.5 - margin
        and box.min_z >= -1e-4
        and box.max_z <= height + 1e-4
    )


def _collisions_involving(scene: SceneSpec, registry: AssetRegistry, object_id: str) -> int:
    return sum(
        1
        for collision in detect_collisions(scene, registry)
        if collision.object_a == object_id or collision.object_b == object_id
    )


def _constraint_for(scene: SceneSpec, obj: ObjectSpec):
    for constraint in scene.constraints:
        if constraint.subject_id == obj.id:
            return constraint
    return None


def _try_place(scene: SceneSpec, registry: AssetRegistry, obj: ObjectSpec, x: float, y: float, margin: float) -> bool:
    old_x = obj.placement.x
    old_y = obj.placement.y
    obj.placement.x = x
    obj.placement.y = y
    _clamp_to_bounds(obj, scene, registry, margin)
    if abs(obj.placement.x - x) > 1e-6 or abs(obj.placement.y - y) > 1e-6:
        obj.placement.x = old_x
        obj.placement.y = old_y
        return False
    if not _object_within_bounds(obj, scene, registry, margin):
        obj.placement.x = old_x
        obj.placement.y = old_y
        return False
    if _collisions_involving(scene, registry, obj.id) == 0:
        return True
    obj.placement.x = old_x
    obj.placement.y = old_y
    return False


def _free_slot_search(scene: SceneSpec, registry: AssetRegistry, obj: ObjectSpec, margin: float) -> bool:
    if obj.role == "anchor" or obj.parent_id:
        return False
    constraint = _constraint_for(scene, obj)
    object_map = {item.id: item for item in scene.objects}
    target = object_map.get(constraint.target_id) if constraint and constraint.target_id else None
    relation = constraint.type if constraint else (obj.relation or "near")

    move_box = object_aabb(obj, registry)
    move_half_x = (move_box.max_x - move_box.min_x) * 0.5
    move_half_y = (move_box.max_y - move_box.min_y) * 0.5
    if target:
        target_box = object_aabb(target, registry)
        base_x = target.placement.x
        base_y = target.placement.y
        target_half_x = (target_box.max_x - target_box.min_x) * 0.5
        target_half_y = (target_box.max_y - target_box.min_y) * 0.5
        radius = max(target_half_x + move_half_x, target_half_y + move_half_y) + 0.28
        dx, dy = RELATION_DIRECTIONS.get(str(relation), RELATION_DIRECTIONS["near"])
        if relation in {"near", "next_to"} and abs(obj.placement.x - base_x) + abs(obj.placement.y - base_y) > 1e-6:
            dx = obj.placement.x - base_x
            dy = obj.placement.y - base_y
        base_angle = math.atan2(dy, dx)
    else:
        base_x = obj.placement.x
        base_y = obj.placement.y
        radius = max(move_half_x, move_half_y) + 0.35
        base_angle = 0.0

    angle_offsets = [0, 45, -45, 90, -90, 135, -135, 180, 225, -225]
    for ring in range(10):
        candidate_radius = radius + ring * 0.35
        for offset in angle_offsets:
            angle = base_angle + math.radians(offset)
            x = base_x + math.cos(angle) * candidate_radius
            y = base_y + math.sin(angle) * candidate_radius
            if _try_place(scene, registry, obj, x, y, margin):
                return True
    return False


def spread_children(scene: SceneSpec, registry: AssetRegistry, parent_id: str | None = None) -> None:
    parent_ids = [parent_id] if parent_id else sorted({obj.parent_id for obj in scene.objects if obj.parent_id})
    object_map = {obj.id: obj for obj in scene.objects}
    for pid in parent_ids:
        if not pid or pid not in object_map:
            continue
        children = [obj for obj in scene.objects if obj.parent_id == pid]
        inside_children = [child for child in children if child.relation == "inside"]
        other_children = [child for child in children if child.relation != "inside"]
        for index, child in enumerate(inside_children):
            _place_child_on_parent(child, object_map[pid], registry, index, len(inside_children))
        for index, child in enumerate(other_children):
            _place_child_on_parent(child, object_map[pid], registry, index, len(other_children))


def optimize_layout(
    scene: SceneSpec,
    registry: AssetRegistry,
    max_iters: int = 80,
    boundary_margin: float = 0.15,
    contact_tolerance: float = 0.04,
) -> tuple[SceneSpec, Metrics]:
    current = deepcopy(scene)
    spread_children(current, registry)
    current = snap_all_to_support(current, registry)

    for _ in range(max_iters):
        _enforce_directional_relations(current, registry)
        for obj in current.objects:
            _clamp_to_bounds(obj, current, registry, boundary_margin)
        current = snap_all_to_support(current, registry)
        collisions = detect_collisions(current, registry)
        if not collisions:
            break
        object_map = {obj.id: obj for obj in current.objects}
        for collision in collisions:
            left = object_map[collision.object_a]
            right = object_map[collision.object_b]
            if left.parent_id and left.parent_id == right.parent_id:
                spread_children(current, registry, left.parent_id)
            else:
                move = _object_to_move(left, right)
                _separate_pair(current, registry, left, right)
                if _collisions_involving(current, registry, move.id):
                    _free_slot_search(current, registry, move, boundary_margin)

    current = snap_all_to_support(current, registry)
    metrics = compute_metrics(
        current,
        registry,
        boundary_margin=boundary_margin,
        contact_tolerance=contact_tolerance,
    )
    return SceneSpec.model_validate(current.model_dump()), metrics
