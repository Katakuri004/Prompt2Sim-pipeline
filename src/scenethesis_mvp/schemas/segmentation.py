from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DetectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str | None = None
    phrase: str
    score: float
    guidance_box_iou: float | None = None
    guidance_box_coverage: float | None = None
    box_source: Literal[
        "groundingdino_box_gpt_verified",
        "gpt_instance_bbox_groundingdino_verified",
        "gpt_support_bbox_groundingdino_verified",
    ] = "gpt_instance_bbox_groundingdino_verified"
    box_xyxy: list[float] = Field(min_length=4, max_length=4)
    dino_box_xyxy: list[float] = Field(min_length=4, max_length=4)
    mask_path: str
    crop_path: str | None = None
    mask_area: int


class SegmentationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "grounded_sam"
    image_path: str
    image_width: int
    image_height: int
    detections: list[DetectionSpec] = Field(default_factory=list)
    missing_object_ids: list[str] = Field(default_factory=list)
    overlay_path: str | None = None

    @property
    def complete(self) -> bool:
        return bool(self.detections) and not self.missing_object_ids
