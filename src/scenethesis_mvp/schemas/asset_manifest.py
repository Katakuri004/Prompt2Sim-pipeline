from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ControlledAssetEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    category: str = Field(min_length=1)
    name: str = Field(min_length=1)
    dimensions: list[float]
    support_kind: Literal["none", "surface", "container"] = "none"
    support_heights: list[float] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    color: list[float] = Field(default_factory=lambda: [0.6, 0.6, 0.6])
    repo_id: str = Field(min_length=1)
    repo_type: Literal["dataset", "model", "space"] = "dataset"
    source_usd: str = Field(min_length=1)
    include_prefixes: list[str] = Field(default_factory=list)
    thumbnail_path: str | None = None
    license: str = Field(min_length=1)
    attribution: str = Field(min_length=1)
    source_url: str | None = None

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


class UnresolvedAssetTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str
    reason: str
    desired_terms: list[str] = Field(default_factory=list)


class ControlledAssetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    source_family: str
    bulk_download_allowed: bool = False
    disk_budget_gb: float = Field(default=15.0, gt=0)
    entries: list[ControlledAssetEntry]
    unresolved_targets: list[UnresolvedAssetTarget] = Field(default_factory=list)

    @field_validator("bulk_download_allowed")
    @classmethod
    def bulk_download_must_be_disabled(cls, value: bool) -> bool:
        if value:
            raise ValueError("controlled manifests must not allow bulk downloads")
        return value


def manifest_source_dir(entry: ControlledAssetEntry) -> str:
    return str(Path(entry.source_usd).parent).replace("\\", "/")
