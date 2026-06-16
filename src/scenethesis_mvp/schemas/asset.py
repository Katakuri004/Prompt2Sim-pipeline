from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SupportKind = Literal["none", "surface", "container"]


class AssetSpec(BaseModel):
    """Registry record for a procedural or external asset."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    name: str = Field(min_length=1)
    dimensions: list[float]
    support_kind: SupportKind = "none"
    support_heights: list[float] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    color: list[float] = Field(default_factory=lambda: [0.6, 0.6, 0.6])
    glb_path: str | None = None
    thumbnail_path: str | None = None
    source: str | None = None
    source_id: str | None = None
    source_url: str | None = None
    license: str | None = None
    attribution: str | None = None

    @field_validator("dimensions")
    @classmethod
    def dimensions_are_positive(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or any(v <= 0 for v in value):
            raise ValueError("dimensions must contain three positive values")
        return value

    @field_validator("support_heights")
    @classmethod
    def support_heights_are_normalized(cls, value: list[float]) -> list[float]:
        for item in value:
            if item <= 0 or item > 1.0:
                raise ValueError("support_heights must be in (0, 1]")
        return value

    @field_validator("color")
    @classmethod
    def color_is_rgb(cls, value: list[float]) -> list[float]:
        if len(value) != 3 or any(v < 0 or v > 1 for v in value):
            raise ValueError("color must contain three values between 0 and 1")
        return value

    def has_local_mesh(self, base_dir: Path) -> bool:
        return bool(self.glb_path and (base_dir / self.glb_path).exists())

    def resolved_mesh_path(self, base_dir: Path) -> Path | None:
        if not self.glb_path:
            return None
        mesh_path = Path(self.glb_path)
        if not mesh_path.is_absolute():
            mesh_path = base_dir / mesh_path
        return mesh_path.resolve()

    def resolved_thumbnail_path(self, base_dir: Path) -> Path | None:
        if not self.thumbnail_path:
            return None
        thumbnail_path = Path(self.thumbnail_path)
        if not thumbnail_path.is_absolute():
            thumbnail_path = base_dir / thumbnail_path
        return thumbnail_path.resolve()

    def scaled_dimensions(self, scale: float) -> tuple[float, float, float]:
        return tuple(round(v * scale, 6) for v in self.dimensions)  # type: ignore[return-value]
