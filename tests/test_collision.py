from __future__ import annotations

from pathlib import Path

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.retriever import AssetRetriever
from scenethesis_mvp.layout.collision import detect_collisions
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, PlacementSpec, SceneSpec


def _registry() -> AssetRegistry:
    root = Path(__file__).resolve().parents[1]
    return AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")


def test_detects_and_clears_aabb_collision() -> None:
    scene = SceneSpec(
        prompt="two boxes",
        objects=[
            ObjectSpec(id="anchor", category="box", role="anchor", placement=PlacementSpec(z=0.14)),
            ObjectSpec(id="box2", category="box", role="parent", placement=PlacementSpec(x=0.05, z=0.14)),
        ],
    )
    registry = _registry()
    scene = AssetRetriever(registry).attach_assets(scene)
    assert len(detect_collisions(scene, registry)) == 1
    scene.object_by_id("box2").placement.x = 1.0
    assert detect_collisions(scene, registry) == []
