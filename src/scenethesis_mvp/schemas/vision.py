from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scenethesis_mvp.schemas.scene_spec import ObjectRole, RelationType


DepthBand = Literal["foreground", "midground", "background"]


class BBox2D(BaseModel):
    """Normalized 2D image-space box: [0, 1], origin at top-left."""

    model_config = ConfigDict(extra="forbid")

    x_min: float = Field(ge=0.0, le=1.0)
    y_min: float = Field(ge=0.0, le=1.0)
    x_max: float = Field(ge=0.0, le=1.0)
    y_max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def max_exceeds_min(self) -> "BBox2D":
        if self.x_max <= self.x_min:
            raise ValueError("x_max must be greater than x_min")
        if self.y_max <= self.y_min:
            raise ValueError("y_max must be greater than y_min")
        return self

    @property
    def center_x(self) -> float:
        return (self.x_min + self.x_max) * 0.5

    @property
    def center_y(self) -> float:
        return (self.y_min + self.y_max) * 0.5


class VisionObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    category: str = Field(min_length=1)
    matched_object_id: str | None
    bbox: BBox2D
    depth: DepthBand
    role: ObjectRole
    confidence: float = Field(ge=0.0, le=1.0)


class VisionRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str = Field(min_length=1)
    target_id: str | None
    subject_object_id: str | None
    target_object_id: str | None
    type: RelationType
    confidence: float = Field(ge=0.0, le=1.0)


class VisionSceneGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt: str
    guidance_description: str
    anchor_object_id: str | None
    objects: list[VisionObject] = Field(default_factory=list)
    relations: list[VisionRelation] = Field(default_factory=list)
    notes: str | None

    @field_validator("objects")
    @classmethod
    def objects_are_unique(cls, value: list[VisionObject]) -> list[VisionObject]:
        ids = [obj.id for obj in value]
        if len(ids) != len(set(ids)):
            raise ValueError("vision object ids must be unique")
        matched = [obj.matched_object_id for obj in value if obj.matched_object_id]
        if len(matched) != len(set(matched)):
            raise ValueError("matched_object_id values must be unique when present")
        return value

    @model_validator(mode="after")
    def relation_subjects_exist(self) -> "VisionSceneGraph":
        ids = {obj.id for obj in self.objects}
        for relation in self.relations:
            if relation.subject_id not in ids:
                raise ValueError(f"unknown vision relation subject_id: {relation.subject_id}")
            if relation.target_id and relation.target_id not in ids:
                raise ValueError(f"unknown vision relation target_id: {relation.target_id}")
        return self
