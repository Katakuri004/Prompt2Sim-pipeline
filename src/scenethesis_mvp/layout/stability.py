from __future__ import annotations

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.collision import object_aabb
from scenethesis_mvp.schemas.metrics import StabilityRecord
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec


def mounted_support_kind(obj: ObjectSpec, registry: AssetRegistry) -> str | None:
    if obj.parent_id and obj.relation in {"on", "inside"}:
        return None
    if not obj.asset_id:
        return None
    asset = registry.get(obj.asset_id)
    tags = {tag.lower() for tag in asset.tags}
    if asset.category == "door":
        return "ground_and_wall"
    if "ceiling" in tags and "floor" not in tags:
        return "ceiling_mount"
    if "wall" in tags and "floor" not in tags:
        return "wall_mount"
    return None


def object_height(obj: ObjectSpec, registry: AssetRegistry) -> float:
    if not obj.asset_id:
        raise ValueError(f"object {obj.id} has no asset_id")
    return registry.get(obj.asset_id).scaled_dimensions(obj.placement.scale)[2]


def support_z_candidates(obj: ObjectSpec, scene: SceneSpec, registry: AssetRegistry) -> list[float]:
    if not obj.parent_id or obj.relation not in {"on", "inside"}:
        return [0.0]
    parent = scene.object_by_id(obj.parent_id)
    if not parent.asset_id:
        return [0.0]
    parent_asset = registry.get(parent.asset_id)
    parent_box = object_aabb(parent, registry)
    if parent_asset.support_kind == "container" and parent_asset.support_heights:
        parent_height = parent_box.max_z - parent_box.min_z
        return [parent_box.min_z + parent_height * level for level in parent_asset.support_heights]
    return [parent_box.max_z]


def snap_to_support(obj: ObjectSpec, scene: SceneSpec, registry: AssetRegistry) -> None:
    if mounted_support_kind(obj, registry) not in {None, "ground_and_wall"}:
        return
    height = object_height(obj, registry)
    candidates = support_z_candidates(obj, scene, registry)
    bottom = obj.placement.z - height * 0.5
    target = min(candidates, key=lambda value: abs(value - bottom))
    obj.placement.z = target + height * 0.5


def snap_all_to_support(scene: SceneSpec, registry: AssetRegistry) -> SceneSpec:
    for obj in scene.objects:
        if not obj.parent_id:
            snap_to_support(obj, scene, registry)
    for obj in scene.objects:
        if obj.parent_id:
            snap_to_support(obj, scene, registry)
    return SceneSpec.model_validate(scene.model_dump())


def center_inside_support(child: ObjectSpec, parent: ObjectSpec, registry: AssetRegistry, margin: float = 0.03) -> bool:
    parent_box = object_aabb(parent, registry)
    return (
        parent_box.min_x + margin <= child.placement.x <= parent_box.max_x - margin
        and parent_box.min_y + margin <= child.placement.y <= parent_box.max_y - margin
    )


def check_stability(
    scene: SceneSpec,
    registry: AssetRegistry,
    contact_tolerance: float = 0.04,
) -> list[StabilityRecord]:
    records: list[StabilityRecord] = []
    for obj in scene.objects:
        if mounted_support_kind(obj, registry) not in {None, "ground_and_wall"}:
            continue
        height = object_height(obj, registry)
        bottom = obj.placement.z - height * 0.5
        candidates = support_z_candidates(obj, scene, registry)
        distance = min(abs(bottom - support_z) for support_z in candidates)
        if distance > contact_tolerance:
            records.append(
                StabilityRecord(
                    object_id=obj.id,
                    reason="floating_or_sinking",
                    distance=round(distance, 6),
                )
            )
        if obj.parent_id and obj.relation in {"on", "inside"}:
            parent = scene.object_by_id(obj.parent_id)
            if not center_inside_support(obj, parent, registry):
                records.append(
                    StabilityRecord(
                        object_id=obj.id,
                        reason="center_outside_parent_support_footprint",
                        distance=0.0,
                    )
                )
    return records
