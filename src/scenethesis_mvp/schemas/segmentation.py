from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DetectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str | None = None
    phrase: str
    score: float
    box_xyxy: list[float] = Field(min_length=4, max_length=4)
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
