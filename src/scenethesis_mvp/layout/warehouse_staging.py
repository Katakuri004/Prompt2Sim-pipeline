from __future__ import annotations

from copy import deepcopy

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.relation_rules import local_to_world
from scenethesis_mvp.layout.stability import snap_all_to_support
from scenethesis_mvp.schemas.scene_spec import ConstraintSpec, ObjectSpec, SceneSpec


WAREHOUSE_CATEGORIES = {
    "bag",
    "barrier",
    "bin",
    "box",
    "camera",
    "cart",
    "cabinet",
    "cable",
    "chair",
    "container",
    "door",
    "duct",
    "floor_marking",
    "forklift",
    "hand_truck",
    "ladder",
    "light",
    "pallet",
    "pallet_load",
    "pipe",
    "scanner",
    "shelf",
    "sign",
    "table",
    "tool",
    "utility_box",
}
FLOOR_SCALE_CATEGORIES = {
    "bag",
    "barrier",
    "bin",
    "cabinet",
    "cart",
    "chair",
    "container",
    "cylinder",
    "floor_marking",
    "forklift",
    "hand_truck",
    "ladder",
    "pallet",
    "pallet_load",
    "sign",
}
MOUNTED_CATEGORIES = {"cable", "camera", "duct", "light", "pipe", "utility_box"}


def stage_warehouse_presentation_layout(scene: SceneSpec, registry: AssetRegistry) -> SceneSpec:
    """Seed a paper-style warehouse composition before SDF optimization.

    The planner/vision stages still decide the object inventory. This pass only
    gives common warehouse objects a coherent aisle layout so the physical and
    correspondence optimizers start from a renderable composition instead of a
    visually tangled cluster.
    """

    if not _should_stage(scene):
        return scene

    updated = deepcopy(scene)
    width, depth, height = [float(value) for value in updated.bounds]
    existing_constraints = {constraint.subject_id: constraint for constraint in updated.constraints}
    by_category: dict[str, list[ObjectSpec]] = {}
    for obj in updated.objects:
        by_category.setdefault(obj.category, []).append(obj)

    shelf = _first(by_category, "shelf")
    table = _first(by_category, "table")

    if shelf:
        _set_scale_if_large(shelf, registry, max_height=2.25)
        _place_floor(shelf, registry, width * 0.40, depth - 0.64, 0.0, width, depth)
    if table:
        _place_floor(table, registry, width * 0.70, depth * 0.40, 0.0, width, depth)

    for forklift in by_category.get("forklift", []):
        _set_scale_if_footprint_large(forklift, registry, max_size=1.55)
    _place_sequence(by_category.get("forklift", []), registry, width, depth, [(width * 0.18, depth * 0.52, 8.0)])
    _place_sequence(by_category.get("cart", []), registry, width, depth, [(width * 0.42, depth * 0.23, 0.0)])
    _place_sequence(by_category.get("pallet", []), registry, width, depth, [(width * 0.33, depth * 0.34, 0.0)])
    _place_sequence(by_category.get("pallet_load", []), registry, width, depth, [(width * 0.22, depth * 0.64, 0.0)])
    _place_sequence(by_category.get("barrier", []), registry, width, depth, [(width * 0.56, depth * 0.34, 90.0)])
    _place_sequence(by_category.get("floor_marking", []), registry, width, depth, [(width * 0.52, depth * 0.28, 0.0)])
    _place_sequence(by_category.get("sign", []), registry, width, depth, [(width * 0.78, depth * 0.60, -12.0)])
    _place_sequence(by_category.get("cylinder", []), registry, width, depth, [(width * 0.13, depth * 0.70, 0.0), (width * 0.20, depth * 0.72, 0.0)])
    _place_sequence(by_category.get("container", []), registry, width, depth, [(width * 0.18, depth * 0.43, -8.0), (width * 0.24, depth * 0.42, 10.0)])
    _place_sequence(by_category.get("bin", []), registry, width, depth, [(width * 0.86, depth * 0.46, 0.0), (width * 0.91, depth * 0.52, 0.0)])
    _place_sequence(by_category.get("bag", []), registry, width, depth, [(width * 0.30, depth * 0.46, 6.0), (width * 0.37, depth * 0.45, -8.0)])
    _place_sequence(by_category.get("cabinet", []), registry, width, depth, [(width * 0.72, depth * 0.64, 0.0)])
    _place_sequence(by_category.get("chair", []), registry, width, depth, [(width * 0.82, depth * 0.31, -18.0)])
    _place_sequence(by_category.get("hand_truck", []), registry, width, depth, [(width * 0.76, depth * 0.30, -15.0)])
    _place_sequence(by_category.get("ladder", []), registry, width, depth, [(width * 0.82, depth - 0.42, 180.0)])

    door = _first(by_category, "door")
    if door:
        _place_back_wall(door, registry, width * 0.16, 0.50, width, depth, height)
        door.relation = "against_wall"
        door.parent_id = None
        door.role = "parent" if door.role != "anchor" else "anchor"

    _place_mounted(by_category.get("utility_box", []), registry, width, depth, height, [(width * 0.85, 0.48)])
    _place_mounted(by_category.get("cable", []), registry, width, depth, height, [(width * 0.71, 0.58)])
    _place_mounted(by_category.get("pipe", []), registry, width, depth, height, [(width * 0.78, 0.72)])
    _place_mounted(by_category.get("duct", []), registry, width, depth, height, [(width * 0.55, 0.88)])
    _place_mounted(by_category.get("camera", []), registry, width, depth, height, [(width * 0.48, 0.76)])
    _place_ceiling_lights(by_category.get("light", []), registry, width, depth, height)

    if shelf:
        shelf_children = [
            obj
            for obj in updated.objects
            if obj.category == "box" and (obj.parent_id == shelf.id or obj.relation in {"on", "inside"} or obj.parent_id is None)
        ]
        for child in shelf_children:
            child.parent_id = shelf.id
            child.role = "child"
            child.relation = "on"
        _place_children_on_parent(shelf, shelf_children, registry, front_bias=True)

    if table:
        table_children = [obj for obj in updated.objects if obj.category in {"scanner", "tool", "monitor"}]
        for child in table_children:
            child.parent_id = table.id
            child.role = "child"
            child.relation = "on"
        _place_children_on_parent(table, table_children, registry, front_bias=False)
        _honor_table_near_constraints(updated, table, registry, existing_constraints, width, depth)

    for obj in updated.objects:
        if obj.category in FLOOR_SCALE_CATEGORIES:
            obj.parent_id = None
            obj.role = "parent" if obj.role != "anchor" else "anchor"
            if obj.relation in {"on", "inside"}:
                obj.relation = "near"
        if obj.category in MOUNTED_CATEGORIES:
            obj.parent_id = None
            obj.role = "parent" if obj.role != "anchor" else "anchor"
            obj.relation = "against_wall"

    updated.constraints = _rebuild_constraints(updated, existing_constraints)
    return snap_all_to_support(SceneSpec.model_validate(updated.model_dump()), registry)


def _should_stage(scene: SceneSpec) -> bool:
    prompt = scene.prompt.lower()
    categories = {obj.category for obj in scene.objects}
    return "warehouse" in prompt or bool(categories.intersection(WAREHOUSE_CATEGORIES))


def _first(by_category: dict[str, list[ObjectSpec]], category: str) -> ObjectSpec | None:
    values = by_category.get(category, [])
    return values[0] if values else None


def _dims(obj: ObjectSpec, registry: AssetRegistry) -> tuple[float, float, float]:
    if not obj.asset_id:
        raise RuntimeError(f"presentation staging requires asset_id for {obj.id}")
    return registry.get(obj.asset_id).scaled_dimensions(obj.placement.scale)


def _set_scale_if_large(obj: ObjectSpec, registry: AssetRegistry, max_height: float) -> None:
    _, _, height = _dims(obj, registry)
    if height <= max_height:
        return
    obj.placement.scale = max(0.55, min(obj.placement.scale, max_height / height))


def _set_scale_if_footprint_large(obj: ObjectSpec, registry: AssetRegistry, max_size: float) -> None:
    dx, dy, _ = _dims(obj, registry)
    footprint = max(dx, dy)
    if footprint <= max_size:
        return
    obj.placement.scale = max(0.42, min(obj.placement.scale, max_size / footprint))


def _clamp_xy(x: float, y: float, obj: ObjectSpec, registry: AssetRegistry, width: float, depth: float) -> tuple[float, float]:
    dx, dy, _ = _dims(obj, registry)
    margin = 0.08
    return (
        min(max(x, dx * 0.5 + margin), width - dx * 0.5 - margin),
        min(max(y, dy * 0.5 + margin), depth - dy * 0.5 - margin),
    )


def _place_floor(obj: ObjectSpec, registry: AssetRegistry, x: float, y: float, yaw_deg: float, width: float, depth: float) -> None:
    obj.placement.x, obj.placement.y = _clamp_xy(x, y, obj, registry, width, depth)
    obj.placement.yaw_deg = yaw_deg
    obj.placement.z = _dims(obj, registry)[2] * 0.5


def _place_sequence(
    objects: list[ObjectSpec],
    registry: AssetRegistry,
    width: float,
    depth: float,
    slots: list[tuple[float, float, float]],
) -> None:
    for index, obj in enumerate(objects):
        slot = slots[min(index, len(slots) - 1)]
        x = slot[0] + 0.34 * index
        y = slot[1] + 0.26 * index
        _place_floor(obj, registry, x, y, slot[2], width, depth)


def _place_back_wall(
    obj: ObjectSpec,
    registry: AssetRegistry,
    x: float,
    z_fraction: float,
    width: float,
    depth: float,
    height: float,
) -> None:
    dx, dy, dz = _dims(obj, registry)
    obj.placement.x = min(max(x, dx * 0.5 + 0.08), width - dx * 0.5 - 0.08)
    obj.placement.y = depth - dy * 0.5 - 0.04
    obj.placement.z = min(max(height * z_fraction, dz * 0.5), height - dz * 0.5 - 0.04)
    obj.placement.yaw_deg = 180.0


def _place_mounted(
    objects: list[ObjectSpec],
    registry: AssetRegistry,
    width: float,
    depth: float,
    height: float,
    slots: list[tuple[float, float]],
) -> None:
    for index, obj in enumerate(objects):
        x, z_fraction = slots[min(index, len(slots) - 1)]
        _place_back_wall(obj, registry, x + 0.24 * index, z_fraction, width, depth, height)
        obj.relation = "against_wall"


def _place_ceiling_lights(objects: list[ObjectSpec], registry: AssetRegistry, width: float, depth: float, height: float) -> None:
    slots = [(width * 0.33, depth * 0.82), (width * 0.64, depth * 0.80), (width * 0.50, depth * 0.70)]
    for index, obj in enumerate(objects):
        dx, dy, dz = _dims(obj, registry)
        x, y = slots[min(index, len(slots) - 1)]
        obj.placement.x = min(max(x, dx * 0.5 + 0.10), width - dx * 0.5 - 0.10)
        obj.placement.y = min(max(y, dy * 0.5 + 0.10), depth - dy * 0.5 - 0.10)
        obj.placement.z = height - dz * 0.5 - 0.06
        obj.placement.yaw_deg = 0.0
        obj.parent_id = None
        obj.role = "parent" if obj.role != "anchor" else "anchor"
        obj.relation = "against_wall"


def _place_children_on_parent(parent: ObjectSpec, children: list[ObjectSpec], registry: AssetRegistry, front_bias: bool) -> None:
    if not children:
        return
    parent_dx, parent_dy, parent_dz = _dims(parent, registry)
    parent_asset = registry.get(parent.asset_id or "")
    levels = parent_asset.support_heights or [1.0]
    columns = max(1, (len(children) + len(levels) - 1) // len(levels))
    for index, child in enumerate(children):
        child_dx, child_dy, child_dz = _dims(child, registry)
        level = levels[index % len(levels)]
        column = index // len(levels)
        usable_x = max(0.04, parent_dx - child_dx - 0.18)
        usable_y = max(0.04, parent_dy - child_dy - 0.16)
        local_x = 0.0 if columns == 1 else -usable_x * 0.5 + usable_x * (column / max(1, columns - 1))
        local_y = -usable_y * 0.28 if front_bias else 0.0
        child.placement.x, child.placement.y = local_to_world(parent, local_x, local_y)
        child.placement.z = parent.placement.z - parent_dz * 0.5 + parent_dz * float(level) + child_dz * 0.5
        child.placement.yaw_deg = parent.placement.yaw_deg


def _honor_table_near_constraints(
    scene: SceneSpec,
    table: ObjectSpec,
    registry: AssetRegistry,
    existing_constraints: dict[str, ConstraintSpec],
    width: float,
    depth: float,
) -> None:
    slot_by_category = {
        "cabinet": (1.14, 0.45, 0.0),
        "hand_truck": (1.22, -0.48, -15.0),
        "bin": (1.14, 0.92, 0.0),
        "chair": (0.0, -0.88, 180.0),
    }
    used = 0
    for obj in scene.objects:
        constraint = existing_constraints.get(obj.id)
        if not constraint or constraint.target_id != table.id or constraint.type not in {"near", "next_to", "left_of", "right_of"}:
            continue
        if obj.id == table.id or obj.category in {"scanner", "tool", "monitor"}:
            continue
        base = slot_by_category.get(obj.category, (1.18, 0.18 + 0.36 * used, 0.0))
        local_x, local_y, yaw = base
        if obj.category not in slot_by_category:
            used += 1
        obj.placement.x, obj.placement.y = local_to_world(table, local_x, local_y)
        obj.placement.x, obj.placement.y = _clamp_xy(obj.placement.x, obj.placement.y, obj, registry, width, depth)
        obj.placement.yaw_deg = yaw % 360.0
        obj.parent_id = None
        obj.role = "parent" if obj.role != "anchor" else "anchor"
        obj.relation = "near"


def _rebuild_constraints(scene: SceneSpec, existing_constraints: dict[str, ConstraintSpec] | None = None) -> list[ConstraintSpec]:
    existing_constraints = existing_constraints or {}
    object_ids = {obj.id for obj in scene.objects}
    constraints: dict[str, ConstraintSpec] = {}
    for obj in scene.objects:
        if obj.role == "anchor":
            continue
        existing = existing_constraints.get(obj.id)
        if obj.parent_id and obj.relation in {"on", "inside"}:
            constraints[obj.id] = ConstraintSpec(type=obj.relation, subject_id=obj.id, target_id=obj.parent_id)
        elif obj.relation in {"left_of", "right_of", "in_front_of", "behind", "near", "next_to"}:
            target_id = existing.target_id if existing and existing.target_id in object_ids else None
            constraints[obj.id] = ConstraintSpec(type=obj.relation, subject_id=obj.id, target_id=target_id)
    return list(constraints.values())
