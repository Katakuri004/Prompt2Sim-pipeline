from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


AssetViewName = Literal["front", "side", "oblique"]


class GeneratedAssetDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=20)
    visual_features: list[str] = Field(min_length=1)
    materials: list[str] = Field(min_length=1)
    colors: list[str] = Field(min_length=1)
    affordances: list[str] = Field(default_factory=list)
    support_surfaces: list[str] = Field(default_factory=list)
    container_regions: list[str] = Field(default_factory=list)
    articulated_parts: list[str] = Field(default_factory=list)
    manipulable: bool


class AssetVisualProfile(GeneratedAssetDescription):
    model_config = ConfigDict(extra="forbid")

    asset_id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    dimensions_m: list[float] = Field(min_length=3, max_length=3)
    view_paths: dict[AssetViewName, str]
    model: str = Field(min_length=1)

    @field_validator("dimensions_m")
    @classmethod
    def validate_dimensions(cls, value: list[float]) -> list[float]:
        if any(item <= 0 for item in value):
            raise ValueError("dimensions_m must contain positive values")
        return value

    @field_validator("view_paths")
    @classmethod
    def validate_view_names(cls, value: dict[AssetViewName, str]) -> dict[AssetViewName, str]:
        required = {"front", "side", "oblique"}
        if set(value) != required:
            raise ValueError(f"view_paths must contain exactly {sorted(required)}")
        if any(not path for path in value.values()):
            raise ValueError("view_paths cannot contain empty paths")
        return value


class ClipCandidateScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    score: float
    image_score: float
    text_score: float
    metadata_score: float


class ClipObjectShortlist(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    category: str
    candidates: list[ClipCandidateScore] = Field(min_length=1)


class ObservedObjectDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = Field(min_length=20)
    visual_features: list[str] = Field(min_length=1)
    materials: list[str] = Field(default_factory=list)
    colors: list[str] = Field(default_factory=list)
    visible_state: str = Field(min_length=1)
    is_valid_object: bool


class CandidateAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    asset_id: str
    visual_similarity: float = Field(ge=0.0, le=1.0)
    semantic_similarity: float = Field(ge=0.0, le=1.0)
    dimension_compatible: bool
    overall_score: float = Field(ge=0.0, le=1.0)
    notes: str


class AssetMatchDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observed_object: ObservedObjectDescription
    decision: Literal["match", "no_match"]
    selected_asset_id: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    assessments: list[CandidateAssessment] = Field(min_length=1)
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision(self) -> "AssetMatchDecision":
        if self.decision == "match" and not self.selected_asset_id:
            raise ValueError("match decision requires selected_asset_id")
        if self.decision == "no_match" and self.selected_asset_id is not None:
            raise ValueError("no_match decision requires selected_asset_id=null")
        return self


class AssetCorrespondenceObject(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    category: str
    status: Literal["matched", "no_match", "failed"]
    selected_asset_id: str | None
    confidence: float | None
    score_margin: float | None
    observed_object: ObservedObjectDescription | None
    shortlist: list[ClipCandidateScore]
    assessments: list[CandidateAssessment]
    shape_errors: dict[str, float]
    error: str | None


class AssetCorrespondenceReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    provider: Literal["openai_multiview_asset_correspondence"]
    shortlist_provider: Literal["open_clip"]
    model: str
    object_count: int = Field(ge=0)
    matched_object_count: int = Field(ge=0)
    failed_object_count: int = Field(ge=0)
    objects: list[AssetCorrespondenceObject]
    error: str | None = None
