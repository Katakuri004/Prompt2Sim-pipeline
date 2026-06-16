from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MeshObjectRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    asset_id: str
    source: str
    mesh_path: str | None = None
    sampled_points: int = 0


class MeshCollisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_a: str
    object_b: str
    min_distance: float
    aabb_penetration: float
    point_a: list[float]
    point_b: list[float]


class MeshSupportRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    support_id: str | None = None
    relation: str | None = None
    bottom_z: float
    support_z: float
    contact_distance: float
    supported: bool


class MeshMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_count: int = 0
    mesh_object_count: int = 0
    proxy_object_count: int = 0
    broad_phase_pair_count: int = 0
    narrow_phase_pair_count: int = 0
    mesh_collision_count: int = 0
    support_failure_count: int = 0
    mesh_collisions: list[MeshCollisionRecord] = Field(default_factory=list)
    supports: list[MeshSupportRecord] = Field(default_factory=list)
    objects: list[MeshObjectRecord] = Field(default_factory=list)
    collision_penalty: float = 0.0
    support_penalty: float = 0.0
    total_penalty: float = 0.0

    @property
    def mesh_clean(self) -> bool:
        return self.mesh_collision_count == 0 and self.support_failure_count == 0
