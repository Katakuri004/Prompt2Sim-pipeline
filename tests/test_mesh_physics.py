from __future__ import annotations

from pathlib import Path

import pytest

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.physics.mesh_check import compute_mesh_metrics, refine_mesh_layout
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, PlacementSpec, SceneSpec


def _root() -> Path:
    return Path(__file__).resolve().parents[1]


def _warehouse_registry() -> AssetRegistry:
    return AssetRegistry.from_yaml(_root() / "configs" / "warehouse_asset_registry.yaml")


def test_mesh_metrics_loads_real_warehouse_asset() -> None:
    registry = _warehouse_registry()
    scene = SceneSpec(
        prompt="one warehouse box",
        objects=[
            ObjectSpec(
                id="box",
                category="box",
                asset_id="real_cardboard_box_01",
                role="anchor",
                placement=PlacementSpec(z=0.17),
            )
        ],
    )
    metrics, _samples = compute_mesh_metrics(scene, registry, sample_points=64, require_meshes=True)
    assert metrics.mesh_object_count == 1
    assert metrics.proxy_object_count == 0
    assert metrics.objects[0].source == "mesh"
    assert metrics.support_failure_count == 0


def test_mesh_metrics_accepts_ceiling_mounted_asset() -> None:
    registry = _warehouse_registry()
    scene = SceneSpec(
        prompt="one warehouse ceiling light",
        bounds=(7.0, 6.0, 3.0),
        objects=[
            ObjectSpec(
                id="light",
                category="light",
                asset_id="real_mounted_fluorescent_lights_01",
                role="anchor",
                relation="against_wall",
                placement=PlacementSpec(x=3.0, y=5.0, z=2.85),
            )
        ],
    )
    metrics, _samples = compute_mesh_metrics(scene, registry, sample_points=64, require_meshes=True)
    assert metrics.support_failure_count == 0
    assert metrics.supports[0].support_id == "ceiling_mount"


def test_mesh_metrics_fails_when_required_mesh_is_missing() -> None:
    registry = AssetRegistry.from_yaml(_root() / "configs" / "asset_registry.yaml")
    scene = SceneSpec(
        prompt="one procedural box",
        objects=[
            ObjectSpec(
                id="box",
                category="box",
                asset_id="proc_box_01",
                role="anchor",
                placement=PlacementSpec(z=0.14),
            )
        ],
    )
    with pytest.raises(RuntimeError, match="has no local mesh path"):
        compute_mesh_metrics(scene, registry, sample_points=64, require_meshes=True)


def test_mesh_refinement_separates_overlapping_real_boxes() -> None:
    registry = _warehouse_registry()
    scene = SceneSpec(
        prompt="two overlapping warehouse boxes",
        objects=[
            ObjectSpec(
                id="box_a",
                category="box",
                asset_id="real_cardboard_box_01",
                role="anchor",
                placement=PlacementSpec(z=0.17),
            ),
            ObjectSpec(
                id="box_b",
                category="box",
                asset_id="real_cardboard_box_01",
                role="parent",
                placement=PlacementSpec(x=0.02, z=0.17),
            ),
        ],
    )
    before, _samples = compute_mesh_metrics(scene, registry, sample_points=64, require_meshes=True)
    assert before.mesh_collision_count >= 1

    refined, after, samples = refine_mesh_layout(
        scene,
        registry,
        max_iters=6,
        sample_points=64,
        require_meshes=True,
    )
    assert after.mesh_collision_count == 0
    assert refined.object_by_id("box_b").placement.x > scene.object_by_id("box_b").placement.x
    assert samples["refinement_history"]
