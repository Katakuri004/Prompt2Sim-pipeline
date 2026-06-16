from __future__ import annotations

from pathlib import Path

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.retriever import AssetRetriever
from scenethesis_mvp.layout.initial_layout import generate_initial_layout
from scenethesis_mvp.layout.optimizer import optimize_layout
from scenethesis_mvp.layout.relation_rules import normalize_support_relation_semantics
from scenethesis_mvp.layout.warehouse_staging import stage_warehouse_presentation_layout
from scenethesis_mvp.schemas.scene_spec import ConstraintSpec, ObjectSpec, SceneSpec


def _constraint(relation: str, subject: str, target: str) -> ConstraintSpec:
    return ConstraintSpec(type=relation, subject_id=subject, target_id=target)


def test_open_shelf_inside_relation_is_normalized_to_on() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")
    scene = SceneSpec(
        prompt="box on shelf",
        objects=[
            ObjectSpec(id="shelf", category="shelf", role="anchor", asset_id="proc_shelf_01"),
            ObjectSpec(id="box", category="box", role="child", parent_id="shelf", relation="inside", asset_id="proc_box_01"),
        ],
        constraints=[_constraint("inside", "box", "shelf")],
    )
    normalized = normalize_support_relation_semantics(scene, registry)
    assert normalized.object_by_id("box").relation == "on"
    assert normalized.constraints[0].type == "on"


def test_floor_scale_assets_are_not_normalized_as_supported_children() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    scene = SceneSpec(
        prompt="warehouse cart with pallet nearby",
        objects=[
            ObjectSpec(id="cart_01", category="cart", role="anchor", asset_id="hf_trolley_01"),
            ObjectSpec(
                id="pallet_01",
                category="pallet",
                role="child",
                parent_id="cart_01",
                relation="on",
                asset_id="hf_wood_pallet_01",
            ),
        ],
        constraints=[_constraint("on", "pallet_01", "cart_01")],
    )
    normalized = normalize_support_relation_semantics(scene, registry)
    pallet = normalized.object_by_id("pallet_01")
    assert pallet.parent_id is None
    assert pallet.role == "parent"
    assert pallet.relation == "near"
    assert normalized.constraints[0].type == "near"
    assert normalized.constraints[0].target_id == "cart_01"


def test_warehouse_presentation_staging_keeps_floor_and_mounted_assets_physical() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    scene = SceneSpec(
        prompt="a spacious warehouse",
        bounds=(8.0, 7.0, 3.2),
        objects=[
            ObjectSpec(id="shelf_01", category="shelf", role="anchor", asset_id="hf_pallet_rack_large_01"),
            ObjectSpec(id="box_01", category="box", role="child", parent_id="shelf_01", relation="on", asset_id="real_cardboard_box_01"),
            ObjectSpec(id="forklift_01", category="forklift", role="parent", asset_id="hf_forklift_orange_01"),
            ObjectSpec(id="light_01", category="light", role="child", parent_id="shelf_01", relation="against_wall", asset_id="real_mounted_fluorescent_lights_01"),
        ],
        constraints=[_constraint("on", "box_01", "shelf_01")],
    )
    staged = stage_warehouse_presentation_layout(scene, registry)
    shelf = staged.object_by_id("shelf_01")
    box = staged.object_by_id("box_01")
    forklift = staged.object_by_id("forklift_01")
    light = staged.object_by_id("light_01")
    assert 0.0 < shelf.placement.y < staged.bounds[1]
    assert box.parent_id == "shelf_01"
    assert box.placement.z > 0.5
    assert forklift.parent_id is None
    assert forklift.placement.z > 0.0
    assert light.parent_id is None
    assert light.placement.z > 2.5


def test_optimizer_produces_stable_robotics_lab_layout() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")
    scene = SceneSpec(
        scene_id="test_robotics_lab",
        prompt="a small robotics lab with a workbench, robot arm, storage shelf, boxes, tools, and a chair",
        bounds=(7.0, 6.0, 3.0),
        objects=[
            ObjectSpec(id="workbench", category="table", role="anchor"),
            ObjectSpec(id="storage_shelf", category="shelf", role="parent", relation="behind"),
            ObjectSpec(id="lab_cabinet", category="cabinet", role="parent", relation="left_of"),
            ObjectSpec(id="chair", category="chair", role="parent", relation="in_front_of"),
            ObjectSpec(id="utility_bin", category="bin", role="parent", relation="right_of"),
            ObjectSpec(id="robot_arm", category="robot_arm", role="child", parent_id="workbench", relation="on"),
            ObjectSpec(id="monitor", category="monitor", role="child", parent_id="workbench", relation="on"),
            ObjectSpec(id="hand_tool", category="tool", role="child", parent_id="workbench", relation="on"),
            ObjectSpec(id="parts_box", category="box", role="child", parent_id="storage_shelf", relation="inside"),
            ObjectSpec(id="spare_box", category="box", role="child", parent_id="storage_shelf", relation="inside"),
            ObjectSpec(id="cylinder_canister", category="cylinder", role="child", parent_id="lab_cabinet", relation="inside"),
        ],
        constraints=[
            _constraint("behind", "storage_shelf", "workbench"),
            _constraint("left_of", "lab_cabinet", "workbench"),
            _constraint("in_front_of", "chair", "workbench"),
            _constraint("right_of", "utility_bin", "workbench"),
            _constraint("on", "robot_arm", "workbench"),
            _constraint("on", "monitor", "workbench"),
            _constraint("on", "hand_tool", "workbench"),
            _constraint("inside", "parts_box", "storage_shelf"),
            _constraint("inside", "spare_box", "storage_shelf"),
            _constraint("inside", "cylinder_canister", "lab_cabinet"),
        ],
    )
    scene = AssetRetriever(registry).attach_assets(scene)
    scene = generate_initial_layout(scene, registry)
    scene, metrics = optimize_layout(scene, registry)
    assert metrics.object_count >= 6
    assert metrics.floating_count == 0
    assert metrics.unsupported_count == 0
    assert metrics.collision_count == 0


def test_optimizer_enforces_directional_relations_after_visual_hints() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")
    scene = SceneSpec(
        scene_id="directional_relation",
        prompt="table with chair in front",
        bounds=(7.0, 6.0, 3.0),
        objects=[
            ObjectSpec(id="table", category="table", role="anchor"),
            ObjectSpec(id="chair", category="chair", role="parent", relation="in_front_of"),
        ],
        constraints=[_constraint("in_front_of", "chair", "table")],
    )
    scene = AssetRetriever(registry).attach_assets(scene)
    scene = generate_initial_layout(scene, registry)
    scene.object_by_id("chair").placement.y = scene.object_by_id("table").placement.y + 1.0
    scene, metrics = optimize_layout(scene, registry)
    assert scene.object_by_id("chair").placement.y < scene.object_by_id("table").placement.y
    assert metrics.relation_penalty == 0.0
