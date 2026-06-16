from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scenethesis_mvp.assets.procedural_assets import REQUIRED_PROCEDURAL_CATEGORIES, normalize_category
from scenethesis_mvp.schemas.asset import AssetSpec
from scenethesis_mvp.utils.io import read_yaml


class AssetRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assets: list[AssetSpec] = Field(default_factory=list)
    base_dir: Path = Field(default_factory=Path.cwd)

    @model_validator(mode="after")
    def validate_registry(self) -> "AssetRegistry":
        ids = [asset.id for asset in self.assets]
        if len(ids) != len(set(ids)):
            raise ValueError("asset ids must be unique")
        categories = {asset.category for asset in self.assets}
        missing = REQUIRED_PROCEDURAL_CATEGORIES - categories
        if missing:
            raise ValueError(f"registry missing procedural categories: {sorted(missing)}")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AssetRegistry":
        source = Path(path)
        data = read_yaml(source)
        assets = [AssetSpec(**item) for item in data.get("assets", [])]
        return cls(assets=assets, base_dir=source.parent)

    @property
    def categories(self) -> list[str]:
        return sorted({asset.category for asset in self.assets})

    def get(self, asset_id: str) -> AssetSpec:
        for asset in self.assets:
            if asset.id == asset_id:
                return asset
        raise KeyError(asset_id)

    def by_category(self, category: str) -> list[AssetSpec]:
        normalized = normalize_category(category)
        return [asset for asset in self.assets if asset.category == normalized]

    def best_for_category(self, category: str, tags: list[str] | None = None) -> AssetSpec:
        normalized = normalize_category(category)
        candidates = self.by_category(normalized)
        if not candidates:
            raise KeyError(f"no asset for category {category}")
        if tags:
            tag_set = {tag.lower() for tag in tags}
            candidates = sorted(
                candidates,
                key=lambda asset: len(tag_set.intersection({tag.lower() for tag in asset.tags})),
                reverse=True,
            )
        return candidates[0]
