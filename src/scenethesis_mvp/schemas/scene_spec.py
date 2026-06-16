from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RelationType = Literal[
    "on",
    "inside",
    "next_to",
    "left_of",
    "right_of",
    "in_front_of",
    "behind",
    "near",
    "facing",
    "against_wall",
]
ObjectRole = Literal["anchor", "parent", "child"]


class PlacementSpec(BaseModel):
    """5-DoF placement. x/y/z are object-center coordinates in meters."""

    model_config = ConfigDict(extra="forbid")

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw_deg: float = 0.0
    scale: float = Field(default=1.0, gt=0.05, le=5.0)

    @field_validator("yaw_deg")
    @classmethod
    def normalize_yaw(cls, value: float) -> float:
        return float(value % 360.0)


class ObjectSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    name: str | None = None
    asset_id: str | None = None
    role: ObjectRole = "parent"
    parent_id: str | None = None
    relation: RelationType | None = None
    placement: PlacementSpec = Field(default_factory=PlacementSpec)
    description: str | None = None

    @model_validator(mode="after")
    def child_has_parent_for_support_relation(self) -> "ObjectSpec":
        if self.role == "child" and self.relation in {"on", "inside"} and not self.parent_id:
            raise ValueError("child objects with on/inside relation need parent_id")
        return self


class ConstraintSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: RelationType | Literal["boundary", "clearance", "support"]
    subject_id: str = Field(min_length=1)
    target_id: str | None = None


class SceneSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scene_id: str = "scene"
    prompt: str
    units: str = "meters"
    bounds: list[float] = Field(default_factory=lambda: [7.0, 6.0, 3.0])
    objects: list[ObjectSpec] = Field(default_factory=list, min_length=1)
    constraints: list[ConstraintSpec] = Field(default_factory=list)
    notes: str | None = None

    @field_validator("bounds")
    @classmethod
    def bounds_are_positive(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or any(v <= 0 for v in value):
            raise ValueError("bounds must contain three positive values")
        return value

    @model_validator(mode="after")
    def validate_scene_graph(self) -> "SceneSpec":
        ids = [obj.id for obj in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object ids must be unique")
        id_set = set(ids)
        anchors = [obj for obj in self.objects if obj.role == "anchor"]
        if len(anchors) != 1:
            raise ValueError("SceneSpec must contain exactly one anchor object")
        for obj in self.objects:
            if obj.parent_id and obj.parent_id not in id_set:
                raise ValueError(f"unknown parent_id: {obj.parent_id}")
        for constraint in self.constraints:
            if constraint.subject_id not in id_set:
                raise ValueError(f"unknown constraint subject_id: {constraint.subject_id}")
            if constraint.target_id and constraint.target_id not in id_set:
                raise ValueError(f"unknown constraint target_id: {constraint.target_id}")
        return self

    def object_by_id(self, object_id: str) -> ObjectSpec:
        for obj in self.objects:
            if obj.id == object_id:
                return obj
        raise KeyError(object_id)

    def children_of(self, parent_id: str) -> list[ObjectSpec]:
        return [obj for obj in self.objects if obj.parent_id == parent_id]
