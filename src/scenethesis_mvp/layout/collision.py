from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.schemas.metrics import CollisionRecord
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec


@dataclass(frozen=True)
class AABB:
    object_id: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def center_x(self) -> float:
        return (self.min_x + self.max_x) * 0.5

    @property
    def center_y(self) -> float:
        return (self.min_y + self.max_y) * 0.5

    @property
    def center_z(self) -> float:
        return (self.min_z + self.max_z) * 0.5

    def overlaps(self, other: "AABB", tolerance: float = 1e-4) -> bool:
        return (
            self.min_x < other.max_x - tolerance
            and self.max_x > other.min_x + tolerance
            and self.min_y < other.max_y - tolerance
            and self.max_y > other.min_y + tolerance
            and self.min_z < other.max_z - tolerance
            and self.max_z > other.min_z + tolerance
        )

    def penetration(self, other: "AABB") -> float:
        overlaps = [
            min(self.max_x, other.max_x) - max(self.min_x, other.min_x),
            min(self.max_y, other.max_y) - max(self.min_y, other.min_y),
            min(self.max_z, other.max_z) - max(self.min_z, other.min_z),
        ]
        return max(0.0, min(overlaps))


def object_aabb(obj: ObjectSpec, registry: AssetRegistry) -> AABB:
    if not obj.asset_id:
        raise ValueError(f"object {obj.id} has no asset_id")
    asset = registry.get(obj.asset_id)
    dx, dy, dz = asset.scaled_dimensions(obj.placement.scale)
    yaw = math.radians(obj.placement.yaw_deg)
    cos_yaw = abs(math.cos(yaw))
    sin_yaw = abs(math.sin(yaw))
    half_x = (cos_yaw * dx + sin_yaw * dy) * 0.5
    half_y = (sin_yaw * dx + cos_yaw * dy) * 0.5
    half_z = dz * 0.5
    return AABB(
        object_id=obj.id,
        min_x=obj.placement.x - half_x,
        max_x=obj.placement.x + half_x,
        min_y=obj.placement.y - half_y,
        max_y=obj.placement.y + half_y,
        min_z=obj.placement.z - half_z,
        max_z=obj.placement.z + half_z,
    )


def should_skip_collision(a: ObjectSpec, b: ObjectSpec) -> bool:
    direct_parent_child = (a.parent_id == b.id) or (b.parent_id == a.id)
    if not direct_parent_child:
        return False
    child = a if a.parent_id == b.id else b
    return child.relation in {"on", "inside"}


def detect_collisions(
    scene: SceneSpec,
    registry: AssetRegistry,
    tolerance: float = 1e-4,
) -> list[CollisionRecord]:
    objects = {obj.id: obj for obj in scene.objects}
    boxes = {obj.id: object_aabb(obj, registry) for obj in scene.objects}
    collisions: list[CollisionRecord] = []
    for left_id, right_id in combinations(boxes, 2):
        left_obj = objects[left_id]
        right_obj = objects[right_id]
        if should_skip_collision(left_obj, right_obj):
            continue
        left_box = boxes[left_id]
        right_box = boxes[right_id]
        if left_box.overlaps(right_box, tolerance=tolerance):
            collisions.append(
                CollisionRecord(
                    object_a=left_id,
                    object_b=right_id,
                    penetration=round(left_box.penetration(right_box), 6),
                )
            )
    return collisions


def boundary_violations(scene: SceneSpec, registry: AssetRegistry, margin: float = 0.0) -> int:
    width, depth, height = scene.bounds
    x_min = -width * 0.5 + margin
    x_max = width * 0.5 - margin
    y_min = -depth * 0.5 + margin
    y_max = depth * 0.5 - margin
    count = 0
    for obj in scene.objects:
        box = object_aabb(obj, registry)
        if box.min_x < x_min or box.max_x > x_max:
            count += 1
        if box.min_y < y_min or box.max_y > y_max:
            count += 1
        if box.min_z < -1e-4 or box.max_z > height:
            count += 1
    return count
