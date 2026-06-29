from __future__ import annotations

from pathlib import Path
import hashlib

from scenethesis_mvp.assets.procedural_assets import REQUIRED_PROCEDURAL_CATEGORIES
from scenethesis_mvp.assets.clip_index import metadata_terms
from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.retriever import AssetRetriever
from scenethesis_mvp.llm.planner import (
    ScenePlanner,
    prompt_asset_trait_requirements,
    prompt_category_requirements,
    required_instance_plan,
    validate_planned_scene,
)
from scenethesis_mvp.schemas.scene_spec import ConstraintSpec, ObjectSpec, SceneSpec
from scenethesis_mvp.utils.io import read_json


def test_asset_registry_has_required_procedural_assets() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")
    assert REQUIRED_PROCEDURAL_CATEGORIES.issubset(set(registry.categories))
    assert registry.best_for_category("workbench").category == "table"


def test_warehouse_registry_prefers_downloaded_mesh_assets() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    asset = registry.best_for_category("box", ["child"])
    assert asset.id.startswith("real_")
    assert asset.resolved_mesh_path(registry.base_dir).is_file()


def test_open_worktable_derivative_has_recorded_source_and_output_hashes() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    asset = registry.get("derived_open_stainless_worktable_01")
    mesh = asset.resolved_mesh_path(registry.base_dir)
    metadata = read_json(root / "assets" / "manifests" / "derived_warehouse_assets.json")["assets"][0]

    assert mesh is not None and mesh.is_file()
    assert metadata["id"] == asset.id
    assert metadata["license"] == "CC-BY-4.0"
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == metadata["output_sha256"]


def test_project_authored_barrier_has_recorded_output_hash() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    asset = registry.get("authored_vertical_slot_barrier_01")
    mesh = asset.resolved_mesh_path(registry.base_dir)
    metadata = read_json(root / "assets" / "manifests" / "project_authored_warehouse_assets.json")["assets"][0]

    assert mesh is not None and mesh.is_file()
    assert metadata["id"] == asset.id
    assert metadata["license"] == "CC0-1.0"
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == metadata["output_sha256"]


def test_project_authored_floor_marking_has_recorded_output_hash() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    asset = registry.get("authored_hazard_floor_marking_01")
    mesh = asset.resolved_mesh_path(registry.base_dir)
    records = read_json(root / "assets" / "manifests" / "project_authored_warehouse_assets.json")["assets"]
    metadata = next(item for item in records if item["id"] == asset.id)

    assert mesh is not None and mesh.is_file()
    assert metadata["license"] == "CC0-1.0"
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == metadata["output_sha256"]


def test_project_authored_clean_box_has_recorded_output_hash() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    asset = registry.get("authored_clean_cardboard_box_01")
    mesh = asset.resolved_mesh_path(registry.base_dir)
    records = read_json(root / "assets" / "manifests" / "project_authored_warehouse_assets.json")["assets"]
    metadata = next(item for item in records if item["id"] == asset.id)

    assert mesh is not None and mesh.is_file()
    assert metadata["license"] == "CC0-1.0"
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == metadata["output_sha256"]


def test_project_authored_packing_table_has_recorded_output_hash() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    asset = registry.get("authored_wood_metal_packing_table_01")
    mesh = asset.resolved_mesh_path(registry.base_dir)
    records = read_json(root / "assets" / "manifests" / "project_authored_warehouse_assets.json")["assets"]
    metadata = next(item for item in records if item["id"] == asset.id)

    assert mesh is not None and mesh.is_file()
    assert metadata["license"] == "CC0-1.0"
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == metadata["output_sha256"]


def test_project_authored_x_braced_crate_has_recorded_output_hash() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    asset = registry.get("authored_x_braced_wooden_crate_01")
    mesh = asset.resolved_mesh_path(registry.base_dir)
    records = read_json(root / "assets" / "manifests" / "project_authored_warehouse_assets.json")["assets"]
    metadata = next(item for item in records if item["id"] == asset.id)

    assert mesh is not None and mesh.is_file()
    assert metadata["license"] == "CC0-1.0"
    assert hashlib.sha256(mesh.read_bytes()).hexdigest() == metadata["output_sha256"]


def test_retriever_upgrades_procedural_choice_when_real_mesh_exists() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    scene = SceneSpec(
        prompt="warehouse with boxes",
        objects=[
            ObjectSpec(id="shelf", category="shelf", role="anchor", asset_id="real_warehouse_shelf_01"),
            ObjectSpec(id="box", category="box", role="child", parent_id="shelf", relation="inside", asset_id="proc_box_01"),
        ],
    )
    updated = AssetRetriever(registry).attach_assets(scene)
    assert updated.object_by_id("box").asset_id == "real_cardboard_box_01"


def test_prompt_requirements_do_not_treat_tool_chest_as_loose_tool() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    requirements = prompt_category_requirements("warehouse with a tool chest and boxes", registry)
    assert requirements["cabinet"] == 1
    assert requirements["box"] == 2
    assert "tool" not in requirements


def test_prompt_requirements_preserve_box_crate_variants() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    prompt = "many cardboard boxes and wooden crates, plastic crates"
    category_requirements = prompt_category_requirements(prompt, registry)
    trait_requirements = prompt_asset_trait_requirements(prompt, registry)
    assert category_requirements["box"] == 5
    assert {"category": "box", "trait": "cardboard", "minimum": 2} in trait_requirements
    assert {"category": "box", "trait": "wooden", "minimum": 1} in trait_requirements
    assert {"category": "box", "trait": "plastic", "minimum": 1} in trait_requirements
    assert required_instance_plan(category_requirements, trait_requirements) == [
        {"id": "cardboard_box_01", "category": "box"},
        {"id": "cardboard_box_02", "category": "box"},
        {"id": "wooden_crate_01", "category": "box"},
        {"id": "plastic_crate_01", "category": "box"},
        {"id": "cardboard_box_03", "category": "box"},
    ]


def test_prompt_requirements_preserve_trash_can_variant() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    trait_requirements = prompt_asset_trait_requirements("warehouse with a trash can", registry)
    assert {"category": "bin", "trait": "trash", "minimum": 1} in trait_requirements


def test_warehouse_prompt_adds_renderable_core_context_before_segmentation() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    requirements = prompt_category_requirements("a spacious warehouse", registry)
    assert requirements["box"] == 2
    assert requirements["forklift"] == 1
    assert requirements["shelf"] == 1
    assert requirements["table"] == 1
    assert "light" not in requirements
    assert "bin" not in requirements
    assert "cart" not in requirements
    assert "ladder" not in requirements
    assert "hand_truck" not in requirements


def test_pallet_stacker_is_planned_as_forklift_family() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")

    requirements = prompt_category_requirements("a walk-behind pallet stacker in a warehouse", registry)

    assert requirements["forklift"] == 1


def test_clip_metadata_terms_keep_subtype_words() -> None:
    terms = metadata_terms("box_03 wooden warehouse crate")
    assert "wooden" in terms
    assert "warehouse" not in terms
    assert "crate" not in terms


def test_clip_metadata_terms_do_not_reward_generic_bin_words() -> None:
    terms = metadata_terms("trash can bin")
    assert terms == {"trash"}


def test_faithful_planner_rejects_inventory_above_guidance_budget() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")
    objects = [ObjectSpec(id="box_00", category="box", role="anchor", asset_id="proc_box_01")]
    constraints = []
    for index in range(1, 19):
        object_id = f"box_{index:02d}"
        objects.append(ObjectSpec(id=object_id, category="box", role="parent", relation="near", asset_id="proc_box_01"))
        constraints.append(ConstraintSpec(type="near", subject_id=object_id, target_id="box_00"))
    scene = SceneSpec(prompt="many boxes", objects=objects, constraints=constraints)

    errors = validate_planned_scene("many boxes", scene, registry, max_objects=18)

    assert any("maximum is 18" in error for error in errors)


def test_faithful_planner_rejects_impossible_prompt_inventory_before_api_call() -> None:
    class UncalledClient:
        configured = True

        def __init__(self) -> None:
            self.call_count = 0

        def chat_json(self, **_: object) -> dict[str, object]:
            self.call_count += 1
            raise AssertionError("OpenAI must not be called for an impossible deterministic inventory")

    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    client = UncalledClient()
    planner = ScenePlanner(client=client, max_objects=16)  # type: ignore[arg-type]
    prompt = (
        "a warehouse with many boxes, a tool chest, hand truck, barrel, ladder, bin, "
        "industrial light, rack, table, forklift, pallet, barrier, and floor markings"
    )

    try:
        planner.plan(prompt, registry, (10.0, 8.0, 3.5))
    except RuntimeError as exc:
        assert "17 deterministic object instances" in str(exc)
        assert "maximum_object_count=16" in str(exc)
    else:
        raise AssertionError("planner accepted an impossible deterministic inventory")
    assert client.call_count == 0


def test_faithful_planner_rejects_unrequested_facing_relation() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")
    scene = SceneSpec(
        prompt="a box near a shelf",
        objects=[
            ObjectSpec(id="shelf_01", category="shelf", role="anchor"),
            ObjectSpec(id="box_01", category="box", role="parent", relation="facing"),
        ],
        constraints=[ConstraintSpec(type="facing", subject_id="box_01", target_id="shelf_01")],
    )

    errors = validate_planned_scene(scene.prompt, scene, registry)

    assert any("does not request an orientation relation" in error for error in errors)
