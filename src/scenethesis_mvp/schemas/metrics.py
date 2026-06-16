from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class CollisionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_a: str
    object_b: str
    penetration: float = 0.0


class StabilityRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    reason: str
    distance: float = 0.0


class Metrics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_count: int = 0
    collision_count: int = 0
    collisions: list[CollisionRecord] = Field(default_factory=list)
    floating_count: int = 0
    unsupported_count: int = 0
    unstable: list[StabilityRecord] = Field(default_factory=list)
    boundary_violations: int = 0
    relation_penalty: float = 0.0
    collision_penalty: float = 0.0
    support_penalty: float = 0.0
    boundary_penalty: float = 0.0
    total_penalty: float = 0.0

    @property
    def stable(self) -> bool:
        return self.floating_count == 0 and self.unsupported_count == 0
