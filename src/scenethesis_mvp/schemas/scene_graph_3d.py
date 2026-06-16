from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Object3DBoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    center: list[float] = Field(min_length=3, max_length=3)
    size: list[float] = Field(min_length=3, max_length=3)
    yaw_deg: float


class ObjectPointCloudSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    phrase: str
    points_path: str
    point_count: int
    bbox: Object3DBoundingBox


class Pose3DSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_id: str
    x: float
    y: float
    z: float
    yaw_deg: float
    scale: float
    source: str = "mask_depth_projection"


class SceneGraph3D(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "grounded_sam_depth_pro"
    pointclouds: list[ObjectPointCloudSpec] = Field(default_factory=list)
    poses: list[Pose3DSpec] = Field(default_factory=list)
    missing_object_ids: list[str] = Field(default_factory=list)
