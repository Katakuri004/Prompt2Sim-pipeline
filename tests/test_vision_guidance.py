from __future__ import annotations

import pytest
from pathlib import Path
from pydantic import ValidationError

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.retriever import AssetRetriever
from scenethesis_mvp.layout.initial_layout import generate_initial_layout
from scenethesis_mvp.schemas.scene_spec import ConstraintSpec, ObjectSpec, SceneSpec
from scenethesis_mvp.schemas.vision import BBox2D, VisionObject, VisionRelation, VisionSceneGraph
from scenethesis_mvp.vision.scene_graph_layout import apply_vision_position_hints, apply_vision_relations


def _registry() -> AssetRegistry:
    root = Path(__file__).resolve().parents[1]
    return AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")


def test_bbox_rejects_inverted_coordinates() -> None:
    with pytest.raises(ValidationError):
        BBox2D(x_min=0.8, y_min=0.2, x_max=0.3, y_max=0.6)


def test_vision_graph_updates_relations_and_position_hints() -> None:
    registry = _registry()
    scene = SceneSpec(
        prompt="table with chair and box",
        bounds=[7.0, 6.0, 3.0],
        objects=[
            ObjectSpec(id="table", category="table", role="anchor"),
            ObjectSpec(id="chair", category="chair", role="parent", relation="near"),
            ObjectSpec(id="box", category="box", role="parent", relation="near"),
        ],
        constraints=[
            ConstraintSpec(type="near", subject_id="chair", target_id="table"),
            ConstraintSpec(type="near", subject_id="box", target_id="table"),
        ],
    )
    scene = AssetRetriever(registry).attach_assets(scene)
    graph = VisionSceneGraph(
        prompt=scene.prompt,
        guidance_description="table centered, chair foreground right, box on table",
        anchor_object_id="table",
        objects=[
            VisionObject(
                id="v_table",
                label="table",
                category="table",
                matched_object_id="table",
                bbox=BBox2D(x_min=0.35, y_min=0.35, x_max=0.65, y_max=0.65),
                depth="midground",
                role="anchor",
                confidence=0.95,
            ),
            VisionObject(
                id="v_chair",
                label="chair",
                category="chair",
                matched_object_id="chair",
                bbox=BBox2D(x_min=0.70, y_min=0.62, x_max=0.88, y_max=0.9),
                depth="foreground",
                role="parent",
                confidence=0.9,
            ),
            VisionObject(
                id="v_box",
                label="box",
                category="box",
                matched_object_id="box",
                bbox=BBox2D(x_min=0.45, y_min=0.42, x_max=0.55, y_max=0.52),
                depth="midground",
                role="child",
                confidence=0.9,
            ),
        ],
        relations=[
            VisionRelation(
                subject_id="v_box",
                target_id="v_table",
                subject_object_id="box",
                target_object_id="table",
                type="on",
                confidence=0.88,
            ),
            VisionRelation(
                subject_id="v_chair",
                target_id="v_table",
                subject_object_id="chair",
                target_object_id="table",
                type="next_to",
                confidence=0.8,
            ),
        ],
        notes=None,
    )

    updated, relation_diag = apply_vision_relations(scene, graph)
    assert updated.object_by_id("box").parent_id == "table"
    assert updated.object_by_id("box").relation == "on"
    assert len(relation_diag["applied_relations"]) == 2

    laid_out = generate_initial_layout(updated, registry)
    hinted, position_diag = apply_vision_position_hints(laid_out, graph, registry)
    assert hinted.object_by_id("chair").placement.x > 0.0
    assert hinted.object_by_id("chair").placement.y < 0.0
    assert hinted.object_by_id("box").parent_id == "table"
    assert len(position_diag["positioned_objects"]) == 1
