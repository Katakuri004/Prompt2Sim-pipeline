from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.schemas.guidance_validation import GuidanceValidationDecision
from scenethesis_mvp.schemas.scene_spec import ConstraintSpec, ObjectSpec, SceneSpec
from scenethesis_mvp.utils.io import read_json, write_json
from scenethesis_mvp.vision.image_guidance import (
    ImageGuidanceGenerator,
    build_guidance_edit_mask,
    build_guidance_repair_prompt,
    guidance_repair_reference_images,
    validate_guidance_decision,
)
from scenethesis_mvp.vision.guidance import build_guidance_prompt
from scenethesis_mvp.vision.grounded_sam import (
    clip_mask_to_box,
    expanded_pixel_crop,
    select_sam_prompt_box,
    select_semantic_detection,
)


class FakeGuidanceClient:
    configured = True

    def __init__(self, decisions: list[dict[str, Any]]):
        self.decisions = list(decisions)
        self.prompts: list[str] = []
        self.generated: list[Path] = []
        self.edited: list[tuple[Path, Path]] = []
        self.reference_images: list[list[Path]] = []

    def generate_image(self, prompt: str, output_path: Path, **kwargs: Any) -> dict[str, Any]:
        self.prompts.append(prompt)
        self.generated.append(Path(output_path))
        Image.new("RGB", (32, 32), (125, 125, 125)).save(output_path)
        return {"model": kwargs["model"], "path": str(output_path)}

    def edit_image(self, image_path: Path, prompt: str, output_path: Path, **kwargs: Any) -> dict[str, Any]:
        self.prompts.append(prompt)
        self.edited.append((Path(image_path), Path(output_path)))
        self.reference_images.append([Path(path) for path in kwargs.get("reference_image_paths", [])])
        Image.open(image_path).save(output_path)
        return {"model": kwargs["model"], "path": str(output_path), "operation": "edit"}

    def vision_json(self, **kwargs: Any) -> dict[str, Any]:
        return self.decisions.pop(0)


def _decision(ok: bool) -> dict[str, Any]:
    return {
        "objects": [
            {
                "object_id": "box_01",
                "visible": ok,
                "fully_in_frame": ok,
                "identifiable": ok,
                "matches_description": ok,
                "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.9],
                "issue": "none" if ok else "The requested box is not visible.",
            }
        ],
        "relations": [],
        "scene_coherent": ok,
        "notes": "coherent" if ok else "required object is absent",
    }


def _scene() -> SceneSpec:
    return SceneSpec(
        prompt="a box",
        objects=[
            ObjectSpec(
                id="box_01",
                category="box",
                role="anchor",
                asset_id="authored_clean_cardboard_box_01",
            )
        ],
    )


def _registry() -> AssetRegistry:
    root = Path(__file__).resolve().parents[1]
    return AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")


def test_guidance_generator_edits_rejected_image_until_inventory_is_valid(tmp_path: Path) -> None:
    prompt_path = tmp_path / "guidance_system.txt"
    prompt_path.write_text("strict inventory validation", encoding="utf-8")
    client = FakeGuidanceClient([_decision(False), _decision(True), _decision(True)])
    generator = ImageGuidanceGenerator(
        client=client,  # type: ignore[arg-type]
        image_model="test-image",
        vision_model="test-vision",
        validation_prompt_path=prompt_path,
        max_validation_attempts=2,
    )

    result = generator.run("a box", _scene(), _registry(), tmp_path, image_size="32x32")

    assert result.guidance_path.is_file()
    assert len(client.prompts) == 2
    assert len(client.generated) == 1
    assert client.edited == [(tmp_path / "guidance_attempt_01.png", tmp_path / "guidance_attempt_02.png")]
    assert "Edit the provided warehouse scene image in place" in client.prompts[1]
    report = read_json(tmp_path / "guidance_validation.json")
    assert report["ok"] is True
    assert len(report["attempts"]) == 2
    assert report["attempts"][0]["ok"] is False
    assert report["attempts"][0]["generation_method"] == "generate"
    assert report["attempts"][1]["ok"] is True
    assert report["attempts"][1]["generation_method"] == "edit"
    assert report["attempts"][1]["confirmation_decision"] is not None


def test_guidance_generator_fails_after_attempt_limit(tmp_path: Path) -> None:
    prompt_path = tmp_path / "guidance_system.txt"
    prompt_path.write_text("strict inventory validation", encoding="utf-8")
    generator = ImageGuidanceGenerator(
        client=FakeGuidanceClient([_decision(False)]),  # type: ignore[arg-type]
        validation_prompt_path=prompt_path,
        max_validation_attempts=1,
    )

    with pytest.raises(RuntimeError, match="failed strict inventory validation"):
        generator.run("a box", _scene(), _registry(), tmp_path, image_size="32x32")

    report = read_json(tmp_path / "guidance_validation.json")
    assert report["ok"] is False
    assert not (tmp_path / "guidance.png").exists()


def test_guidance_generator_repairs_when_confirmation_disagrees(tmp_path: Path) -> None:
    prompt_path = tmp_path / "guidance_system.txt"
    prompt_path.write_text("strict inventory validation", encoding="utf-8")
    client = FakeGuidanceClient(
        [_decision(True), _decision(False), _decision(True), _decision(True)]
    )
    generator = ImageGuidanceGenerator(
        client=client,  # type: ignore[arg-type]
        image_model="test-image",
        vision_model="test-vision",
        validation_prompt_path=prompt_path,
        max_validation_attempts=2,
    )

    generator.run("a box", _scene(), _registry(), tmp_path, image_size="32x32")

    report = read_json(tmp_path / "guidance_validation.json")
    assert report["ok"] is True
    assert report["attempts"][0]["ok"] is False
    assert report["attempts"][0]["confirmation_errors"]
    assert report["attempts"][1]["ok"] is True
    assert (tmp_path / "guidance_mask_02.png").is_file()


def test_guidance_correction_uses_registered_thumbnail_for_shape_mismatch(tmp_path: Path) -> None:
    prompt_path = tmp_path / "guidance_system.txt"
    prompt_path.write_text("strict inventory validation", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    scene = SceneSpec(
        prompt="a hand truck",
        objects=[
            ObjectSpec(
                id="hand_truck_01",
                category="hand_truck",
                role="anchor",
                asset_id="real_hand_truck_01",
            )
        ],
    )
    rejected = {
        "objects": [
            {
                "object_id": "hand_truck_01",
                "visible": True,
                "fully_in_frame": True,
                "identifiable": True,
                "matches_description": False,
                "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.9],
                "issue": "This is a pallet jack, not a two-wheel hand truck.",
            }
        ],
        "relations": [],
        "scene_coherent": True,
        "notes": "wrong object geometry",
    }
    accepted = {
        **rejected,
        "objects": [{**rejected["objects"][0], "matches_description": True, "issue": "none"}],
        "notes": "coherent",
    }
    client = FakeGuidanceClient([rejected, accepted, accepted])
    generator = ImageGuidanceGenerator(
        client=client,  # type: ignore[arg-type]
        image_model="test-image",
        vision_model="test-vision",
        validation_prompt_path=prompt_path,
        max_validation_attempts=2,
    )

    generator.run(scene.prompt, scene, registry, tmp_path, image_size="32x32")

    expected = registry.get("real_hand_truck_01").resolved_thumbnail_path(registry.base_dir)
    assert client.reference_images == [[expected]]
    assert "Image 2: exact required appearance for hand_truck_01" in client.prompts[1]
    report = read_json(tmp_path / "guidance_validation.json")
    assert report["attempts"][1]["reference_image_paths"] == [str(expected)]


def test_guidance_repair_unlocks_objects_named_by_relation_errors() -> None:
    prompt = build_guidance_repair_prompt(
        "complete contract",
        [
            "barrier_01: does not match description; wrong shape",
            "relation barrel_01 near table_01 is not satisfied; too far away",
        ],
        2,
    )

    assert "The affected objects are: barrier_01, barrel_01" in prompt
    assert "do not preserve their incorrect location or geometry" in prompt


def test_guidance_prompt_includes_discriminative_registry_traits() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    scene = SceneSpec(
        prompt="an orange safety barrier",
        objects=[
            ObjectSpec(
                id="barrier_01",
                category="barrier",
                role="anchor",
                asset_id="authored_vertical_slot_barrier_01",
            )
        ],
    )

    prompt = build_guidance_prompt(scene.prompt, scene, registry)

    assert "four_vertical_openings" in prompt
    assert "no_bulky_base" in prompt
    assert "dimensions_m=[1.25, 0.1, 1.05]" in prompt


def test_guidance_correction_prefers_complete_multiview_profile(tmp_path: Path) -> None:
    class FakeProfileStore:
        def __init__(self) -> None:
            self.requested: list[str] = []
            self.paths = {}
            for view in ("front", "side", "oblique"):
                path = tmp_path / f"{view}.png"
                Image.new("RGB", (32, 32), (100, 100, 100)).save(path)
                self.paths[view] = path

        def ensure_profiles(self, asset_ids: list[str], registry: AssetRegistry) -> dict[str, object]:
            self.requested.extend(asset_ids)
            return {}

        def view_paths(self, asset_id: str) -> dict[str, Path]:
            return self.paths

    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    scene = SceneSpec(
        prompt="a hand truck",
        objects=[
            ObjectSpec(
                id="hand_truck_01",
                category="hand_truck",
                role="anchor",
                asset_id="real_hand_truck_01",
            )
        ],
    )
    store = FakeProfileStore()

    references = guidance_repair_reference_images(
        ["hand_truck_01: does not match description; pallet jack"],
        scene,
        registry,
        store,  # type: ignore[arg-type]
    )

    assert store.requested == ["real_hand_truck_01"]
    assert references == [("hand_truck_01", store.paths[name]) for name in ("front", "side", "oblique")]


def test_guidance_edit_mask_targets_missing_object_relation_region(tmp_path: Path) -> None:
    source = tmp_path / "scene.png"
    Image.new("RGB", (100, 100), (120, 120, 120)).save(source)
    scene = SceneSpec(
        prompt="a hand truck left of a rack",
        objects=[
            ObjectSpec(id="rack_01", category="shelf", role="anchor"),
            ObjectSpec(id="hand_truck_01", category="hand_truck", role="parent", relation="left_of"),
        ],
        constraints=[ConstraintSpec(type="left_of", subject_id="hand_truck_01", target_id="rack_01")],
    )
    decision = GuidanceValidationDecision.model_validate(
        {
            "objects": [
                {
                    "object_id": "rack_01",
                    "visible": True,
                    "fully_in_frame": True,
                    "identifiable": True,
                    "matches_description": True,
                    "bbox_xyxy_norm": [0.4, 0.1, 0.8, 0.9],
                    "issue": "none",
                },
                {
                    "object_id": "hand_truck_01",
                    "visible": False,
                    "fully_in_frame": False,
                    "identifiable": False,
                    "matches_description": False,
                    "bbox_xyxy_norm": [0.0, 0.0, 0.0, 0.0],
                    "issue": "missing",
                },
            ],
            "relations": [
                {
                    "subject_id": "hand_truck_01",
                    "target_id": "rack_01",
                    "relation": "left_of",
                    "satisfied": False,
                    "issue": "missing",
                }
            ],
            "scene_coherent": False,
            "notes": "missing hand truck",
        }
    )

    output = build_guidance_edit_mask(
        source,
        scene,
        decision,
        [
            "hand_truck_01: not visible, does not match description; missing",
            "relation hand_truck_01 left_of rack_01 is not satisfied; missing",
        ],
        tmp_path / "mask.png",
    )

    with Image.open(output) as mask:
        assert mask.mode == "RGBA"
        assert mask.size == (100, 100)
        assert mask.getpixel((30, 50))[3] == 0
        assert mask.getpixel((95, 50))[3] == 255


def test_existing_guidance_is_revalidated_without_image_generation(tmp_path: Path) -> None:
    prompt_path = tmp_path / "guidance_system.txt"
    prompt_path.write_text("strict inventory validation", encoding="utf-8")
    Image.new("RGB", (32, 32), (125, 125, 125)).save(tmp_path / "guidance.png")
    write_json(tmp_path / "guidance_image.json", {"model": "test-image", "operation": "edit"})
    client = FakeGuidanceClient([_decision(True), _decision(True)])
    generator = ImageGuidanceGenerator(
        client=client,  # type: ignore[arg-type]
        image_model="test-image",
        vision_model="test-vision",
        validation_prompt_path=prompt_path,
    )

    result = generator.validate_existing("a box", _scene(), _registry(), tmp_path)

    assert result.object_boxes == {"box_01": [0.1, 0.1, 0.9, 0.9]}
    assert client.generated == []
    assert client.edited == []
    report = read_json(tmp_path / "guidance_validation.json")
    assert report["ok"] is True
    assert report["attempts"][0]["generation_method"] == "validate_existing"
    assert report["attempts"][0]["confirmation_decision"] is not None


def test_asset_aware_guidance_repair_uses_reference_views_and_revalidates_scene(tmp_path: Path) -> None:
    prompt_path = tmp_path / "guidance_system.txt"
    prompt_path.write_text("strict inventory validation", encoding="utf-8")
    Image.new("RGB", (32, 32), (125, 125, 125)).save(tmp_path / "guidance.png")
    references = []
    for view in ("front", "side", "oblique"):
        path = tmp_path / f"{view}.png"
        Image.new("RGB", (32, 32), (90, 90, 90)).save(path)
        references.append(path)
    client = FakeGuidanceClient([_decision(True)])
    generator = ImageGuidanceGenerator(
        client=client,  # type: ignore[arg-type]
        image_model="test-image",
        vision_model="test-vision",
        validation_prompt_path=prompt_path,
    )

    result = generator.repair_object_to_asset(
        prompt="a box",
        scene=_scene(),
        registry=_registry(),
        out_dir=tmp_path,
        object_id="box_01",
        target_asset_id="authored_clean_cardboard_box_01",
        reference_view_paths=references,
        failure_reason="candidate geometry differs",
        repair_index=1,
        image_size="32x32",
        image_quality="low",
    )

    assert result.guidance_path.is_file()
    assert client.edited == [(tmp_path / "guidance.png", tmp_path / "guidance_asset_repair_01.png")]
    report = read_json(tmp_path / "guidance_validation.json")
    assert report["ok"] is True
    assert report["attempts"][-1]["generation_method"] == "asset_edit"
    repair_log = read_json(tmp_path / "guidance_asset_repairs.json")
    assert repair_log["repairs"][0]["target_asset_id"] == "authored_clean_cardboard_box_01"


def test_guidance_validation_requires_exact_object_coverage() -> None:
    decision = GuidanceValidationDecision.model_validate(
        {
            "objects": [
                {
                    "object_id": "wrong_id",
                    "visible": True,
                    "fully_in_frame": True,
                    "identifiable": True,
                    "matches_description": True,
                    "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.9],
                    "issue": "none",
                }
            ],
            "relations": [],
            "scene_coherent": True,
            "notes": "coherent",
        }
    )

    errors = validate_guidance_decision(_scene(), decision)

    assert any("object coverage mismatch" in error for error in errors)


def test_repeated_instance_dino_crop_expands_and_clamps_guidance_box() -> None:
    assert expanded_pixel_crop(np.asarray([100.0, 100.0, 200.0, 200.0]), 300, 300) == (55, 55, 245, 245)
    assert expanded_pixel_crop(np.asarray([0.0, 5.0, 40.0, 45.0]), 300, 300) == (0, 0, 64, 69)


def test_guidance_validation_allows_zero_box_only_for_invisible_object() -> None:
    invisible = GuidanceValidationDecision.model_validate(
        {
            "objects": [
                {
                    "object_id": "box_01",
                    "visible": False,
                    "fully_in_frame": False,
                    "identifiable": False,
                    "matches_description": False,
                    "bbox_xyxy_norm": [0.0, 0.0, 0.0, 0.0],
                    "issue": "The box is absent.",
                }
            ],
            "relations": [],
            "scene_coherent": False,
            "notes": "missing object",
        }
    )
    assert invisible.objects[0].bbox_xyxy_norm == [0.0, 0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="visible guidance objects must have a positive-area"):
        GuidanceValidationDecision.model_validate(
            {
                "objects": [
                    {
                        "object_id": "box_01",
                        "visible": True,
                        "fully_in_frame": True,
                        "identifiable": True,
                        "matches_description": True,
                        "bbox_xyxy_norm": [0.0, 0.0, 0.0, 0.0],
                        "issue": "invalid visible box",
                    }
                ],
                "relations": [],
                "scene_coherent": True,
                "notes": "invalid",
            }
        )


def test_guidance_validation_allows_identifiable_ceiling_fixture_to_reach_frame_boundary() -> None:
    scene = SceneSpec(
        prompt="warehouse light",
        objects=[ObjectSpec(id="light_01", category="light", role="anchor", asset_id="proc_monitor_01")],
    )
    decision = GuidanceValidationDecision.model_validate(
        {
            "objects": [
                {
                    "object_id": "light_01",
                    "visible": True,
                    "fully_in_frame": False,
                    "identifiable": True,
                    "matches_description": True,
                    "bbox_xyxy_norm": [0.1, 0.0, 0.9, 0.4],
                    "issue": "Suspension continues to the ceiling boundary.",
                }
            ],
            "relations": [],
            "scene_coherent": True,
            "notes": "coherent",
        }
    )

    assert validate_guidance_decision(scene, decision) == []


def test_guidance_validation_requires_exact_satisfied_relations() -> None:
    scene = SceneSpec(
        prompt="a box on a shelf",
        objects=[
            ObjectSpec(id="shelf_01", category="shelf", role="anchor"),
            ObjectSpec(
                id="box_01",
                category="box",
                role="child",
                parent_id="shelf_01",
                relation="on",
            ),
        ],
        constraints=[ConstraintSpec(type="on", subject_id="box_01", target_id="shelf_01")],
    )
    decision = GuidanceValidationDecision.model_validate(
        {
            "objects": [
                {
                    "object_id": object_id,
                    "visible": True,
                    "fully_in_frame": True,
                    "identifiable": True,
                    "matches_description": True,
                    "bbox_xyxy_norm": [0.1, 0.1, 0.9, 0.9],
                    "issue": "none",
                }
                for object_id in ("shelf_01", "box_01")
            ],
            "relations": [
                {
                    "subject_id": "box_01",
                    "target_id": "shelf_01",
                    "relation": "on",
                    "satisfied": False,
                    "issue": "The box is floating above the shelf.",
                }
            ],
            "scene_coherent": True,
            "notes": "coherent apart from the failed support relation",
        }
    )

    errors = validate_guidance_decision(scene, decision)

    assert any("box_01 on shelf_01 is not satisfied" in error for error in errors)


def test_guidance_validation_rejects_duplicate_boxes_for_repeated_category_instances() -> None:
    scene = SceneSpec(
        prompt="two boxes",
        objects=[
            ObjectSpec(id="box_01", category="box", role="anchor"),
            ObjectSpec(id="box_02", category="box", role="parent"),
        ],
    )
    decision = GuidanceValidationDecision.model_validate(
        {
            "objects": [
                {
                    "object_id": object_id,
                    "visible": True,
                    "fully_in_frame": True,
                    "identifiable": True,
                    "matches_description": True,
                    "bbox_xyxy_norm": box,
                    "issue": "",
                }
                for object_id, box in (
                    ("box_01", [0.1, 0.1, 0.5, 0.5]),
                    ("box_02", [0.11, 0.11, 0.51, 0.51]),
                )
            ],
            "relations": [],
            "scene_coherent": True,
            "notes": "",
        }
    )

    errors = validate_guidance_decision(scene, decision)

    assert any("duplicate instance boxes" in error for error in errors)


class ArrayTensor:
    def __init__(self, values: list[Any]):
        self.values = np.asarray(values, dtype=float)

    def detach(self) -> "ArrayTensor":
        return self

    def cpu(self) -> "ArrayTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


def test_grounded_sam_prefers_guidance_location_over_higher_wrong_object_score() -> None:
    boxes = ArrayTensor([[10, 10, 90, 90], [150, 20, 230, 100]])
    logits = ArrayTensor([0.45, 0.95])

    selection = select_semantic_detection(
        boxes,
        logits,
        expected_xyxy=np.asarray([0, 0, 100, 100], dtype=float),
        min_expected_box_iou=0.10,
    )

    assert selection is not None
    assert selection[0] == 0
    assert selection[1] > 0.6


def test_grounded_sam_allows_one_group_proposal_to_verify_distinct_instance_boxes() -> None:
    boxes = ArrayTensor([[10, 10, 190, 100]])
    logits = ArrayTensor([0.91])

    left = select_semantic_detection(
        boxes,
        logits,
        expected_xyxy=np.asarray([10, 10, 95, 100], dtype=float),
        min_expected_box_iou=0.10,
    )
    right = select_semantic_detection(
        boxes,
        logits,
        expected_xyxy=np.asarray([105, 10, 190, 100], dtype=float),
        min_expected_box_iou=0.10,
    )

    assert left is not None and right is not None
    assert left[0] == right[0] == 0
    assert left[1] > 0.4 and right[1] > 0.4


def test_grounded_sam_accepts_strict_partial_box_coverage_but_rejects_unrelated_proposal() -> None:
    partial = select_semantic_detection(
        ArrayTensor([[0, 0, 100, 100]]),
        ArrayTensor([0.8]),
        expected_xyxy=np.asarray([20, 90, 80, 140], dtype=float),
        min_expected_box_iou=0.10,
        min_expected_box_coverage=0.20,
    )
    unrelated = select_semantic_detection(
        ArrayTensor([[0, 0, 100, 100]]),
        ArrayTensor([0.8]),
        expected_xyxy=np.asarray([120, 120, 180, 180], dtype=float),
        min_expected_box_iou=0.10,
        min_expected_box_coverage=0.20,
    )

    assert partial is not None
    assert partial[1] < 0.10
    assert partial[2] == pytest.approx(0.20)
    assert unrelated is None


def test_grounded_sam_clips_instance_mask_to_validated_guidance_box() -> None:
    mask = np.ones((6, 8), dtype=bool)

    clipped = clip_mask_to_box(mask, np.asarray([1, 2, 5, 5], dtype=float))

    assert int(clipped.sum()) == 12
    assert clipped[2:5, 1:5].all()
    assert not clipped[:2].any()


def test_sam_prompt_fusion_uses_gpt_for_repeated_instances_and_support_parents() -> None:
    expected = np.asarray([10, 10, 50, 50], dtype=float)
    dino = np.asarray([8, 8, 60, 60], dtype=float)

    unique_box, unique_source = select_sam_prompt_box(expected, dino, category_instance_count=1)
    repeated_box, repeated_source = select_sam_prompt_box(expected, dino, category_instance_count=2)
    support_box, support_source = select_sam_prompt_box(
        expected,
        dino,
        category_instance_count=1,
        supports_children=True,
    )

    assert unique_box.tolist() == dino.tolist()
    assert unique_source == "groundingdino_box_gpt_verified"
    assert repeated_box.tolist() == expected.tolist()
    assert repeated_source == "gpt_instance_bbox_groundingdino_verified"
    assert support_box.tolist() == expected.tolist()
    assert support_source == "gpt_support_bbox_groundingdino_verified"
