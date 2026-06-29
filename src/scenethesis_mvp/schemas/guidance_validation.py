from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class GuidanceObjectValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    visible: bool
    fully_in_frame: bool
    identifiable: bool
    matches_description: bool
    bbox_xyxy_norm: list[float] = Field(min_length=4, max_length=4)
    issue: str

    @field_validator("bbox_xyxy_norm")
    @classmethod
    def valid_normalized_bbox(cls, value: list[float]) -> list[float]:
        x0, y0, x1, y1 = value
        if not all(0.0 <= coordinate <= 1.0 for coordinate in value):
            raise ValueError("bbox_xyxy_norm coordinates must be in [0, 1]")
        if value == [0.0, 0.0, 0.0, 0.0]:
            return value
        if x1 <= x0 or y1 <= y0:
            raise ValueError("bbox_xyxy_norm must have positive width and height")
        return value

    @model_validator(mode="after")
    def visible_objects_have_positive_boxes(self) -> "GuidanceObjectValidation":
        if self.visible and self.bbox_xyxy_norm == [0.0, 0.0, 0.0, 0.0]:
            raise ValueError("visible guidance objects must have a positive-area bbox_xyxy_norm")
        return self

    @property
    def ok(self) -> bool:
        return self.visible and self.fully_in_frame and self.identifiable and self.matches_description


class GuidanceRelationValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_id: str
    target_id: str
    relation: str
    satisfied: bool
    issue: str


class GuidanceValidationDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    objects: list[GuidanceObjectValidation] = Field(min_length=1)
    relations: list[GuidanceRelationValidation] = Field(default_factory=list)
    scene_coherent: bool
    notes: str


class GuidanceValidationAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempt: int = Field(ge=1)
    image_path: str
    generation_method: Literal["generate", "edit", "asset_edit", "validate_existing"]
    source_image_path: str | None
    reference_image_paths: list[str] = Field(default_factory=list)
    mask_path: str | None = None
    ok: bool
    decision: GuidanceValidationDecision
    confirmation_decision: GuidanceValidationDecision | None = None
    confirmation_errors: list[str] = Field(default_factory=list)
    errors: list[str]


class GuidanceValidationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    provider: str
    model: str
    attempts: list[GuidanceValidationAttempt]
    final_image_path: str | None
