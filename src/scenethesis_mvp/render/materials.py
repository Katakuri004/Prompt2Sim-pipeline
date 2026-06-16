from __future__ import annotations

from scenethesis_mvp.assets.procedural_assets import DEFAULT_CATEGORY_COLORS
from scenethesis_mvp.schemas.asset import AssetSpec


def color_for_asset(asset: AssetSpec) -> list[float]:
    return asset.color or list(DEFAULT_CATEGORY_COLORS.get(asset.category, (0.6, 0.6, 0.6)))
