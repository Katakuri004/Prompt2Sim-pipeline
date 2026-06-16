from __future__ import annotations

from pathlib import Path

from scenethesis_mvp.assets.procedural_assets import REQUIRED_PROCEDURAL_CATEGORIES
from scenethesis_mvp.assets.clip_index import metadata_terms
from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.retriever import AssetRetriever
from scenethesis_mvp.llm.planner import prompt_asset_trait_requirements, prompt_category_requirements
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec


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
    assert requirements["box"] >= 3
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


def test_prompt_requirements_preserve_trash_can_variant() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    trait_requirements = prompt_asset_trait_requirements("warehouse with a trash can", registry)
    assert {"category": "bin", "trait": "trash", "minimum": 1} in trait_requirements


def test_warehouse_prompt_adds_dense_context_props_before_segmentation() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    requirements = prompt_category_requirements("a spacious warehouse", registry)
    assert requirements["box"] >= 6
    assert requirements["ladder"] == 1
    assert requirements["cylinder"] >= 2
    assert requirements["hand_truck"] == 1
    assert requirements["forklift"] == 1
    assert requirements["bin"] == 1


def test_clip_metadata_terms_keep_subtype_words() -> None:
    terms = metadata_terms("box_03 wooden warehouse crate")
    assert "wooden" in terms
    assert "warehouse" not in terms
    assert "crate" not in terms


def test_clip_metadata_terms_do_not_reward_generic_bin_words() -> None:
    terms = metadata_terms("trash can bin")
    assert terms == {"trash"}
