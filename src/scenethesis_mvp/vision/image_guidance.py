from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.visual_profiles import VIEW_NAMES, AssetVisualProfileStore
from scenethesis_mvp.llm.openai_client import OpenAIClient
from scenethesis_mvp.schemas.guidance_validation import (
    GuidanceValidationAttempt,
    GuidanceValidationDecision,
    GuidanceValidationReport,
)
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import read_json, read_text, write_json, write_text
from scenethesis_mvp.vision.guidance import build_guidance_prompt, build_retrieval_candidates


@dataclass(frozen=True)
class ImageGuidanceResult:
    guidance_path: Path
    image_metadata: dict[str, Any]
    upsampled_prompt: str
    candidates: list[dict[str, Any]]
    object_boxes: dict[str, list[float]] = field(default_factory=dict)


class ImageGuidanceGenerator:
    def __init__(
        self,
        client: OpenAIClient | None = None,
        image_model: str = "gpt-image-1",
        vision_model: str = "gpt-5.5",
        validation_prompt_path: str | Path | None = None,
        profile_store: AssetVisualProfileStore | None = None,
        max_validation_attempts: int = 3,
        correction_mode: str = "edit_high_fidelity",
        max_retries: int = 3,
    ):
        self.client = client or OpenAIClient()
        self.image_model = os.getenv("OPENAI_IMAGE_MODEL", image_model)
        self.vision_model = os.getenv("OPENAI_VISION_MODEL", vision_model)
        self.validation_prompt_path = Path(validation_prompt_path) if validation_prompt_path else None
        self.profile_store = profile_store
        self.max_validation_attempts = max_validation_attempts
        self.correction_mode = correction_mode
        self.max_retries = max_retries

    def run(
        self,
        prompt: str,
        scene: SceneSpec,
        registry: AssetRegistry,
        out_dir: str | Path,
        image_size: str = "1024x1024",
        image_quality: str = "low",
    ) -> ImageGuidanceResult:
        if not self.client.configured:
            raise RuntimeError("OPENAI_API_KEY is required for image guidance generation.")
        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        candidates = build_retrieval_candidates(scene, registry)
        base_prompt = build_guidance_prompt(prompt, scene, registry)
        write_json(target_dir / "retrieval_candidates.json", candidates)
        if not self.validation_prompt_path or not self.validation_prompt_path.is_file():
            raise RuntimeError("Guidance validation system prompt is required and must exist.")
        if self.max_validation_attempts < 1:
            raise RuntimeError("max_validation_attempts must be at least 1")
        if self.correction_mode != "edit_high_fidelity":
            raise RuntimeError("Faithful image guidance requires correction_mode=edit_high_fidelity.")
        attempts: list[GuidanceValidationAttempt] = []
        current_prompt = base_prompt
        image_metadata: dict[str, Any] | None = None
        final_path = target_dir / "guidance.png"
        for attempt_index in range(1, self.max_validation_attempts + 1):
            attempt_path = target_dir / f"guidance_attempt_{attempt_index:02d}.png"
            source_path: Path | None = None
            reference_images: list[tuple[str, Path]] = []
            mask_path: Path | None = None
            if attempt_index == 1:
                metadata = self.client.generate_image(
                    prompt=current_prompt,
                    output_path=attempt_path,
                    model=self.image_model,
                    size=image_size,
                    quality=image_quality,
                    max_retries=self.max_retries,
                )
                generation_method = "generate"
            else:
                source_path = target_dir / f"guidance_attempt_{attempt_index - 1:02d}.png"
                reference_images = guidance_repair_reference_images(
                    attempts[-1].errors,
                    scene,
                    registry,
                    self.profile_store,
                )
                mask_path = build_guidance_edit_mask(
                    source_path=source_path,
                    scene=scene,
                    decision=attempts[-1].decision,
                    errors=attempts[-1].errors,
                    output_path=target_dir / f"guidance_mask_{attempt_index:02d}.png",
                )
                metadata = self.client.edit_image(
                    image_path=source_path,
                    prompt=current_prompt,
                    output_path=attempt_path,
                    model=self.image_model,
                    size=image_size,
                    quality=image_quality,
                    reference_image_paths=[path for _, path in reference_images],
                    mask_path=mask_path,
                    max_retries=self.max_retries,
                )
                generation_method = "edit"
            decision = self._validate_guidance(prompt, scene, registry, attempt_path)
            errors = validate_guidance_decision(scene, decision)
            confirmation_decision: GuidanceValidationDecision | None = None
            confirmation_errors: list[str] = []
            if not errors:
                confirmation_decision = self._validate_guidance(prompt, scene, registry, attempt_path)
                confirmation_errors = validate_guidance_decision(scene, confirmation_decision)
                errors = list(confirmation_errors)
            attempt = GuidanceValidationAttempt(
                attempt=attempt_index,
                image_path=str(attempt_path.resolve()),
                generation_method=generation_method,
                source_image_path=str(source_path.resolve()) if source_path else None,
                reference_image_paths=[str(path.resolve()) for _, path in reference_images],
                mask_path=str(mask_path.resolve()) if mask_path else None,
                ok=not errors,
                decision=decision,
                confirmation_decision=confirmation_decision,
                confirmation_errors=confirmation_errors,
                errors=errors,
            )
            attempts.append(attempt)
            report = GuidanceValidationReport(
                ok=not errors,
                provider="openai_vision_guidance_inventory",
                model=self.vision_model,
                attempts=attempts,
                final_image_path=str(final_path.resolve()) if not errors else None,
            )
            write_json(target_dir / "guidance_validation.json", report.model_dump(mode="json"))
            if not errors:
                shutil.copy2(attempt_path, final_path)
                image_metadata = dict(metadata)
                image_metadata["path"] = str(final_path)
                write_text(target_dir / "upsampled_prompt.txt", current_prompt)
                break
            if attempt_index < self.max_validation_attempts:
                next_reference_images = guidance_repair_reference_images(
                    errors,
                    scene,
                    registry,
                    self.profile_store,
                )
                current_prompt = build_guidance_repair_prompt(
                    base_prompt,
                    errors,
                    attempt_index + 1,
                    guidance_reference_prompt(next_reference_images),
                )
        if image_metadata is None:
            raise RuntimeError(
                f"Guidance image failed strict inventory validation after {self.max_validation_attempts} attempts; "
                f"see {target_dir / 'guidance_validation.json'}"
            )
        write_json(target_dir / "guidance_image.json", image_metadata)
        return ImageGuidanceResult(
            guidance_path=final_path,
            image_metadata=image_metadata,
            upsampled_prompt=current_prompt,
            candidates=candidates,
            object_boxes={item.object_id: item.bbox_xyxy_norm for item in attempts[-1].decision.objects},
        )

    def _validate_guidance(
        self,
        prompt: str,
        scene: SceneSpec,
        registry: AssetRegistry,
        image_path: Path,
    ) -> GuidanceValidationDecision:
        payload = {
            "original_prompt": prompt,
            "planned_objects": [
                {
                    "object_id": obj.id,
                    "category": obj.category,
                    "name": registry.get(obj.asset_id).name if obj.asset_id else obj.name,
                    "description": obj.description,
                    "dimensions_m": registry.get(obj.asset_id).dimensions if obj.asset_id else None,
                    "tags": registry.get(obj.asset_id).tags if obj.asset_id else [],
                    "requires_full_visibility": obj.category not in {"cable", "door", "duct", "floor_marking", "light", "pipe"},
                }
                for obj in scene.objects
            ],
            "required_object_ids": [obj.id for obj in scene.objects],
            "required_relations": planned_guidance_relations(scene),
        }
        response = self.client.vision_json(
            system_prompt=read_text(self.validation_prompt_path),
            user_prompt=json.dumps(payload, indent=2),
            image_path=image_path,
            model=self.vision_model,
            json_schema=GuidanceValidationDecision.model_json_schema(),
            schema_name="GuidanceValidationDecision",
            max_retries=self.max_retries,
            image_detail="high",
        )
        return GuidanceValidationDecision.model_validate(response)

    def validate_existing(
        self,
        prompt: str,
        scene: SceneSpec,
        registry: AssetRegistry,
        out_dir: str | Path,
    ) -> ImageGuidanceResult:
        if not self.client.configured:
            raise RuntimeError("OPENAI_API_KEY is required to revalidate existing image guidance.")
        target_dir = Path(out_dir)
        guidance_path = target_dir / "guidance.png"
        metadata_path = target_dir / "guidance_image.json"
        if not guidance_path.is_file() or not metadata_path.is_file():
            raise RuntimeError("Existing guidance revalidation requires guidance.png and guidance_image.json.")
        metadata = read_json(metadata_path)
        if metadata.get("model") != self.image_model:
            raise RuntimeError(
                f"Existing guidance model mismatch: expected {self.image_model}, found {metadata.get('model')}"
            )
        if not self.validation_prompt_path or not self.validation_prompt_path.is_file():
            raise RuntimeError("Guidance validation system prompt is required and must exist.")
        candidates = build_retrieval_candidates(scene, registry)
        base_prompt = build_guidance_prompt(prompt, scene, registry)
        decision = self._validate_guidance(prompt, scene, registry, guidance_path)
        errors = validate_guidance_decision(scene, decision)
        confirmation_decision: GuidanceValidationDecision | None = None
        confirmation_errors: list[str] = []
        if not errors:
            confirmation_decision = self._validate_guidance(prompt, scene, registry, guidance_path)
            confirmation_errors = validate_guidance_decision(scene, confirmation_decision)
            errors = list(confirmation_errors)
        attempt = GuidanceValidationAttempt(
            attempt=1,
            image_path=str(guidance_path.resolve()),
            generation_method="validate_existing",
            source_image_path=str(guidance_path.resolve()),
            reference_image_paths=[],
            mask_path=None,
            ok=not errors,
            decision=decision,
            confirmation_decision=confirmation_decision,
            confirmation_errors=confirmation_errors,
            errors=errors,
        )
        report = GuidanceValidationReport(
            ok=not errors,
            provider="openai_vision_guidance_inventory",
            model=self.vision_model,
            attempts=[attempt],
            final_image_path=str(guidance_path.resolve()) if not errors else None,
        )
        write_json(target_dir / "guidance_validation.json", report.model_dump(mode="json"))
        if errors:
            raise RuntimeError(
                "Existing guidance image failed strict inventory/location validation; "
                f"see {target_dir / 'guidance_validation.json'}"
            )
        write_json(target_dir / "retrieval_candidates.json", candidates)
        write_text(target_dir / "upsampled_prompt.txt", base_prompt)
        return ImageGuidanceResult(
            guidance_path=guidance_path,
            image_metadata=metadata,
            upsampled_prompt=base_prompt,
            candidates=candidates,
            object_boxes={item.object_id: item.bbox_xyxy_norm for item in decision.objects},
        )

    def repair_object_to_asset(
        self,
        prompt: str,
        scene: SceneSpec,
        registry: AssetRegistry,
        out_dir: str | Path,
        object_id: str,
        target_asset_id: str,
        reference_view_paths: list[str | Path],
        failure_reason: str,
        repair_index: int,
        image_size: str,
        image_quality: str,
    ) -> ImageGuidanceResult:
        if repair_index < 1:
            raise ValueError("repair_index must be at least 1")
        target_dir = Path(out_dir)
        guidance_path = target_dir / "guidance.png"
        if not guidance_path.is_file():
            raise RuntimeError(f"Asset-aware guidance repair requires guidance.png: {guidance_path}")
        target_asset = registry.get(target_asset_id)
        target_scene = scene.model_copy(deep=True)
        target_object = target_scene.object_by_id(object_id)
        if target_object.category != target_asset.category:
            raise RuntimeError(
                f"Asset-aware repair category mismatch for {object_id}: "
                f"{target_object.category} != {target_asset.category}"
            )
        target_object.asset_id = target_asset.id
        target_object.name = target_asset.name
        repair_path = target_dir / f"guidance_asset_repair_{repair_index:02d}.png"
        repair_prompt = (
            "Edit image 1 in place with high input fidelity. Images 2, 3, and 4 are front, side, and oblique "
            f"reference renders of the exact replacement asset for {object_id}. Replace only {object_id} with that exact "
            "asset geometry, construction, proportions, and visible material. Preserve its existing location, support, scale, "
            "and required relation. Preserve every other object, the camera, room, lighting, and composition exactly. "
            "Do not add or remove objects. The previous strict correspondence failure was: "
            f"{failure_reason}\n\nComplete corrected scene contract:\n"
            + build_guidance_prompt(prompt, target_scene, registry)
        )
        metadata = self.client.edit_image(
            image_path=guidance_path,
            reference_image_paths=reference_view_paths,
            prompt=repair_prompt,
            output_path=repair_path,
            model=self.image_model,
            size=image_size,
            quality=image_quality,
            max_retries=self.max_retries,
        )
        decision = self._validate_guidance(prompt, target_scene, registry, repair_path)
        errors = validate_guidance_decision(target_scene, decision)
        report_path = target_dir / "guidance_validation.json"
        previous_attempts: list[GuidanceValidationAttempt] = []
        if report_path.is_file():
            previous = GuidanceValidationReport.model_validate(read_json(report_path))
            previous_attempts = list(previous.attempts)
        attempt = GuidanceValidationAttempt(
            attempt=len(previous_attempts) + 1,
            image_path=str(repair_path.resolve()),
            generation_method="asset_edit",
            source_image_path=str(guidance_path.resolve()),
            reference_image_paths=[str(Path(path).resolve()) for path in reference_view_paths],
            mask_path=None,
            ok=not errors,
            decision=decision,
            errors=errors,
        )
        attempts = [*previous_attempts, attempt]
        report = GuidanceValidationReport(
            ok=not errors,
            provider="openai_vision_guidance_inventory",
            model=self.vision_model,
            attempts=attempts,
            final_image_path=str(guidance_path.resolve()) if not errors else None,
        )
        write_json(report_path, report.model_dump(mode="json"))
        repair_log_path = target_dir / "guidance_asset_repairs.json"
        repair_log = read_json(repair_log_path) if repair_log_path.is_file() else {"repairs": []}
        repair_record = {
            "repair_index": repair_index,
            "object_id": object_id,
            "target_asset_id": target_asset_id,
            "reference_view_paths": [str(Path(path).resolve()) for path in reference_view_paths],
            "failure_reason": failure_reason,
            "output_image": str(repair_path.resolve()),
            "ok": not errors,
            "validation_errors": errors,
        }
        repair_log.setdefault("repairs", []).append(repair_record)
        write_json(repair_log_path, repair_log)
        if errors:
            raise RuntimeError(
                f"Asset-aware guidance repair failed strict validation for {object_id}; "
                f"see {report_path}"
            )
        shutil.copy2(repair_path, guidance_path)
        metadata = dict(metadata)
        metadata["path"] = str(guidance_path)
        metadata["asset_repair_object_id"] = object_id
        metadata["asset_repair_target_asset_id"] = target_asset_id
        write_json(target_dir / "guidance_image.json", metadata)
        candidates = build_retrieval_candidates(target_scene, registry)
        write_json(target_dir / "retrieval_candidates.json", candidates)
        write_text(target_dir / "upsampled_prompt.txt", repair_prompt)
        return ImageGuidanceResult(
            guidance_path=guidance_path,
            image_metadata=metadata,
            upsampled_prompt=repair_prompt,
            candidates=candidates,
            object_boxes={item.object_id: item.bbox_xyxy_norm for item in decision.objects},
        )


def validate_guidance_decision(scene: SceneSpec, decision: GuidanceValidationDecision) -> list[str]:
    expected = {obj.id for obj in scene.objects}
    categories = {obj.id: obj.category for obj in scene.objects}
    partial_frame_categories = {"cable", "door", "duct", "floor_marking", "light", "pipe"}
    received = [item.object_id for item in decision.objects]
    errors: list[str] = []
    if len(received) != len(set(received)) or set(received) != expected:
        errors.append(
            f"object coverage mismatch: missing={sorted(expected - set(received))}, "
            f"extra={sorted(set(received) - expected)}"
        )
    if not decision.scene_coherent:
        errors.append(f"scene is not coherent: {decision.notes}")
    for item in decision.objects:
        if item.object_id not in expected:
            continue
        failed = []
        if not item.visible:
            failed.append("not visible")
        if not item.fully_in_frame and categories[item.object_id] not in partial_frame_categories:
            failed.append("not fully in frame")
        if not item.identifiable:
            failed.append("not identifiable")
        if not item.matches_description:
            failed.append("does not match description")
        if failed:
            errors.append(f"{item.object_id}: {', '.join(failed)}; {item.issue}")
    valid_records = [item for item in decision.objects if item.object_id in expected]
    for index, left in enumerate(valid_records):
        for right in valid_records[index + 1 :]:
            if categories[left.object_id] != categories[right.object_id]:
                continue
            overlap = normalized_box_iou(left.bbox_xyxy_norm, right.bbox_xyxy_norm)
            if overlap >= 0.85:
                errors.append(
                    f"duplicate instance boxes for {left.object_id} and {right.object_id}: IoU={overlap:.3f}"
                )
    expected_relations = {
        (item["subject_id"], item["target_id"], item["relation"])
        for item in planned_guidance_relations(scene)
    }
    received_relations = [
        (item.subject_id, item.target_id, item.relation)
        for item in decision.relations
    ]
    received_relation_set = set(received_relations)
    if len(received_relations) != len(received_relation_set) or received_relation_set != expected_relations:
        errors.append(
            "relation coverage mismatch: "
            f"missing={sorted(expected_relations - received_relation_set)}, "
            f"extra={sorted(received_relation_set - expected_relations)}"
        )
    for item in decision.relations:
        key = (item.subject_id, item.target_id, item.relation)
        if key in expected_relations and not item.satisfied:
            errors.append(
                f"relation {item.subject_id} {item.relation} {item.target_id} is not satisfied; {item.issue}"
            )
    return errors


def normalized_box_iou(a: list[float], b: list[float]) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return 0.0 if union <= 0.0 else intersection / union


def build_guidance_edit_mask(
    source_path: str | Path,
    scene: SceneSpec,
    decision: GuidanceValidationDecision,
    errors: list[str],
    output_path: str | Path,
) -> Path:
    affected_ids = guidance_repair_affected_objects(errors)
    if not affected_ids:
        raise RuntimeError(f"Cannot derive a local guidance edit mask from errors: {errors}")
    records = {item.object_id: item for item in decision.objects}
    relation_subjects = {
        match.group(1)
        for error in errors
        if (match := re.match(r"^relation\s+([a-zA-Z0-9_\-]+)\s+", error))
    }
    constraints = {constraint.subject_id: constraint for constraint in scene.constraints}
    regions: list[list[float]] = []
    for object_id in affected_ids:
        record = records.get(object_id)
        old_box = record.bbox_xyxy_norm if record and normalized_box_has_area(record.bbox_xyxy_norm) else None
        if old_box:
            regions.append(expand_normalized_box(old_box, 0.25))
        if old_box and object_id not in relation_subjects:
            continue
        constraint = constraints.get(object_id)
        if not constraint:
            if old_box:
                continue
            raise RuntimeError(f"Cannot place missing guidance object {object_id}; relation constraint is missing")
        target_record = records.get(constraint.target_id)
        if not target_record or not normalized_box_has_area(target_record.bbox_xyxy_norm):
            raise RuntimeError(
                f"Cannot place {object_id} relative to {constraint.target_id}; target guidance box is unavailable"
            )
        asset = scene.object_by_id(object_id)
        regions.append(
            expand_normalized_box(
                proposed_relation_box(
                    relation=constraint.type,
                    target_box=target_record.bbox_xyxy_norm,
                    source_box=old_box,
                    source_dimensions=asset.placement.scale,
                ),
                0.20,
            )
        )
    if not regions:
        raise RuntimeError(f"Cannot derive a non-empty local guidance edit mask from errors: {errors}")
    source = Path(source_path)
    output = Path(output_path)
    with Image.open(source) as image:
        width, height = image.size
    mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    for x0, y0, x1, y1 in regions:
        draw.rectangle(
            (
                int(round(x0 * width)),
                int(round(y0 * height)),
                int(round(x1 * width)),
                int(round(y1 * height)),
            ),
            fill=(0, 0, 0, 0),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    mask.save(output, format="PNG")
    return output


def normalized_box_has_area(box: list[float]) -> bool:
    return len(box) == 4 and box[2] > box[0] and box[3] > box[1]


def expand_normalized_box(box: list[float], fraction: float) -> list[float]:
    width = box[2] - box[0]
    height = box[3] - box[1]
    return [
        max(0.0, box[0] - width * fraction),
        max(0.0, box[1] - height * fraction),
        min(1.0, box[2] + width * fraction),
        min(1.0, box[3] + height * fraction),
    ]


def proposed_relation_box(
    relation: str,
    target_box: list[float],
    source_box: list[float] | None,
    source_dimensions: float,
) -> list[float]:
    target_width = target_box[2] - target_box[0]
    target_height = target_box[3] - target_box[1]
    if source_box:
        width = source_box[2] - source_box[0]
        height = source_box[3] - source_box[1]
    else:
        width = min(0.24, max(0.08, target_width * 0.38 * source_dimensions))
        height = min(0.36, max(0.10, target_height * 0.46 * source_dimensions))
    gap = 0.02
    target_cx = (target_box[0] + target_box[2]) * 0.5
    target_cy = (target_box[1] + target_box[3]) * 0.5
    if relation == "left_of":
        cx = target_box[0] - gap - width * 0.5
        cy = min(target_box[3] - height * 0.5, target_cy)
    elif relation == "right_of":
        cx = target_box[2] + gap + width * 0.5
        cy = min(target_box[3] - height * 0.5, target_cy)
    elif relation == "in_front_of":
        cx = target_cx
        cy = target_box[3] + gap + height * 0.35
    elif relation == "on":
        cx = target_cx
        cy = target_box[1] - gap - height * 0.5
    elif relation == "inside":
        cx = target_cx
        cy = target_cy
        width = min(width, target_width * 0.70)
        height = min(height, target_height * 0.70)
    else:
        prefer_right = target_box[2] + gap + width <= 0.97
        cx = target_box[2] + gap + width * 0.5 if prefer_right else target_box[0] - gap - width * 0.5
        cy = target_cy
    cx = min(0.97 - width * 0.5, max(0.03 + width * 0.5, cx))
    cy = min(0.97 - height * 0.5, max(0.03 + height * 0.5, cy))
    return [cx - width * 0.5, cy - height * 0.5, cx + width * 0.5, cy + height * 0.5]


def planned_guidance_relations(scene: SceneSpec) -> list[dict[str, str]]:
    constraints = {constraint.subject_id: constraint for constraint in scene.constraints}
    relations: list[dict[str, str]] = []
    for obj in scene.objects:
        if obj.role == "anchor" or not obj.relation:
            continue
        constraint = constraints.get(obj.id)
        target_id = obj.parent_id or (constraint.target_id if constraint else None)
        if not target_id:
            raise RuntimeError(f"Guidance relation for {obj.id} has no target object.")
        relations.append(
            {
                "subject_id": obj.id,
                "target_id": target_id,
                "relation": obj.relation,
            }
        )
    return relations


def guidance_repair_reference_images(
    errors: list[str],
    scene: SceneSpec,
    registry: AssetRegistry,
    profile_store: AssetVisualProfileStore | None = None,
) -> list[tuple[str, Path]]:
    mismatch_markers = ("does not match description", "not identifiable")
    object_ids: list[str] = []
    scene_ids = {obj.id for obj in scene.objects}
    for error in errors:
        if not any(marker in error for marker in mismatch_markers):
            continue
        object_id = error.split(":", 1)[0].strip()
        if object_id in scene_ids and object_id not in object_ids:
            object_ids.append(object_id)
    if len(object_ids) > 5:
        raise RuntimeError(
            "Guidance correction requires exact references for more than five mismatched assets; "
            f"cannot fit one source plus all references in a single image edit: {object_ids}"
        )
    references: list[tuple[str, Path]] = []
    for object_id in object_ids:
        obj = scene.object_by_id(object_id)
        if not obj.asset_id:
            raise RuntimeError(f"Guidance correction cannot reference {object_id}; planned asset_id is missing")
        asset = registry.get(obj.asset_id)
        if profile_store is not None:
            profile_store.ensure_profiles([asset.id], registry)
            view_paths = profile_store.view_paths(asset.id)
            missing_views = [name for name in VIEW_NAMES if not view_paths[name].is_file()]
            if missing_views:
                raise RuntimeError(
                    f"Guidance correction requires complete rendered profile views for {object_id}/{asset.id}; "
                    f"missing={missing_views}"
                )
            references.extend((object_id, view_paths[name]) for name in VIEW_NAMES)
            continue
        thumbnail = asset.resolved_thumbnail_path(registry.base_dir)
        if not thumbnail or not thumbnail.is_file():
            raise RuntimeError(
                f"Guidance correction requires a registered local thumbnail for {object_id}/{asset.id}: {thumbnail}"
            )
        references.append((object_id, thumbnail))
    return references


def guidance_reference_prompt(references: list[tuple[str, Path]]) -> str:
    if not references:
        return ""
    lines = [
        "Image 1 is the scene to edit. The following images are exact registered asset references; "
        "use their object geometry and construction only, without copying their background, camera, or framing:"
    ]
    for image_index, (object_id, path) in enumerate(references, start=2):
        lines.append(f"- Image {image_index}: exact required appearance for {object_id} ({path.name})")
    return "\n".join(lines)


def build_guidance_repair_prompt(
    base_prompt: str,
    errors: list[str],
    next_attempt: int,
    reference_prompt: str = "",
) -> str:
    affected_objects = guidance_repair_affected_objects(errors)
    affected_text = ", ".join(affected_objects) if affected_objects else "the objects named in the errors"
    return (
        "Edit the provided warehouse scene image in place using high input fidelity. "
        "Preserve the camera, room, lighting, and every unaffected valid object, including its identity, location, scale, "
        "floor contact, and relation. Do not remove, merge, duplicate, or redesign unaffected valid objects. "
        f"The affected objects are: {affected_text}. Move, resize, or replace only those affected objects as required "
        "to satisfy every listed object and relation error; do not preserve their incorrect location or geometry. "
        f"This is strict correction attempt {next_attempt}. Correct every issue below:\n- "
        + "\n- ".join(errors)
        + ("\n\n" + reference_prompt if reference_prompt else "")
        + "\n\nThe complete target contract remains:\n"
        + base_prompt
    )


def guidance_repair_affected_objects(errors: list[str]) -> list[str]:
    object_ids: list[str] = []
    for error in errors:
        direct_match = re.match(r"^([a-zA-Z0-9_\-]+):", error)
        relation_match = re.match(r"^relation\s+([a-zA-Z0-9_\-]+)\s+", error)
        match = direct_match or relation_match
        if match and match.group(1) not in object_ids:
            object_ids.append(match.group(1))
    return object_ids
