from __future__ import annotations

from copy import deepcopy

from scenethesis_mvp.assets.procedural_assets import normalize_category
from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.schemas.asset import AssetSpec
from scenethesis_mvp.schemas.scene_spec import SceneSpec


class AssetRetriever:
    """Small registry-backed retriever for procedural and local mesh assets."""

    def __init__(self, registry: AssetRegistry):
        self.registry = registry

    def attach_assets(self, scene: SceneSpec) -> SceneSpec:
        updated = deepcopy(scene)
        for obj in updated.objects:
            obj.category = normalize_category(obj.category)
            tags = [obj.role]
            if obj.relation:
                tags.append(obj.relation)
            if obj.asset_id:
                selected = self.registry.get(obj.asset_id)
                replacement = self._best_local_mesh(obj.category, tags)
                if replacement and not selected.glb_path:
                    selected = replacement
                    obj.asset_id = selected.id
                    obj.name = selected.name
                continue
            asset = self.registry.best_for_category(obj.category, tags)
            obj.asset_id = asset.id
            if not obj.name:
                obj.name = asset.name
        return SceneSpec.model_validate(updated.model_dump())

    def _best_local_mesh(self, category: str, tags: list[str]) -> AssetSpec | None:
        candidates = [asset for asset in self.registry.by_category(category) if asset.glb_path]
        if not candidates:
            return None
        tag_set = {tag.lower() for tag in tags}
        return sorted(
            candidates,
            key=lambda asset: len(tag_set.intersection({tag.lower() for tag in asset.tags})),
            reverse=True,
        )[0]
