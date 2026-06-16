from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CameraIntrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float


class DepthResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: str = "depth_pro"
    image_path: str
    depth_path: str
    preview_path: str
    intrinsics: CameraIntrinsics
    min_depth_m: float
    max_depth_m: float
