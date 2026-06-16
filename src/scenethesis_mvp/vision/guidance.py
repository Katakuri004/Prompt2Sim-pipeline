from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.llm.openai_client import OpenAIClient
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.schemas.vision import VisionSceneGraph
from scenethesis_mvp.utils.io import read_text, write_json, write_text


@dataclass(frozen=True)
class VisionGuidanceResult:
    guidance_path: Path
    graph: VisionSceneGraph
    image_metadata: dict[str, Any]
    upsampled_prompt: str
    candidates: list[dict[str, Any]]


class VisionGuidance:
    def __init__(
        self,
        client: OpenAIClient | None = None,
        image_model: str = "gpt-image-1",
        vision_model: str = "gpt-4o-mini",
        system_prompt_path: str | Path | None = None,
        max_retries: int = 3,
    ):
        self.client = client or OpenAIClient()
        self.image_model = os.getenv("OPENAI_IMAGE_MODEL", image_model)
        self.vision_model = os.getenv("OPENAI_VISION_MODEL", vision_model)
        self.system_prompt_path = Path(system_prompt_path) if system_prompt_path else None
        self.max_retries = max_retries

    def run(
        self,
        prompt: str,
        scene: SceneSpec,
        registry: AssetRegistry,
        out_dir: str | Path,
        image_size: str = "1024x1024",
        image_quality: str = "low",
    ) -> VisionGuidanceResult:
        if not self.client.configured:
            raise RuntimeError("OPENAI_API_KEY is required for vision-guided layout.")
        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        candidates = build_retrieval_candidates(scene, registry)
        upsampled_prompt = build_guidance_prompt(prompt, scene, registry)
        write_text(target_dir / "upsampled_prompt.txt", upsampled_prompt)
        write_json(target_dir / "retrieval_candidates.json", candidates)

        image_metadata = self.client.generate_image(
            prompt=upsampled_prompt,
            output_path=target_dir / "guidance.png",
            model=self.image_model,
            size=image_size,
            quality=image_quality,
            max_retries=self.max_retries,
        )
        write_json(target_dir / "guidance_image.json", image_metadata)

        graph = self.extract_scene_graph(prompt, scene, registry, target_dir / "guidance.png")
        write_json(target_dir / "scene_graph.json", graph)
        return VisionGuidanceResult(
            guidance_path=target_dir / "guidance.png",
            graph=graph,
            image_metadata=image_metadata,
            upsampled_prompt=upsampled_prompt,
            candidates=candidates,
        )

    def extract_scene_graph(
        self,
        prompt: str,
        scene: SceneSpec,
        registry: AssetRegistry,
        guidance_path: str | Path,
    ) -> VisionSceneGraph:
        system_prompt = read_text(self.system_prompt_path) if self.system_prompt_path else ""
        base_payload = {
            "original_prompt": prompt,
            "candidate_object_ids": [obj.id for obj in scene.objects],
            "planned_objects": [
                {
                    "id": obj.id,
                    "category": obj.category,
                    "asset_id": obj.asset_id,
                    "name": obj.name,
                    "role": obj.role,
                    "parent_id": obj.parent_id,
                    "relation": obj.relation,
                    "asset_dimensions": registry.get(obj.asset_id).dimensions if obj.asset_id else None,
                }
                for obj in scene.objects
            ],
            "planned_relations": [constraint.model_dump(mode="json") for constraint in scene.constraints],
            "output_contract": [
                "Use normalized 2D bounding boxes in [0, 1] with top-left origin.",
                "Set matched_object_id only to one of candidate_object_ids, otherwise null.",
                "Set subject_object_id and target_object_id when the relation maps to known planned objects.",
                "Use only relation types allowed by the schema.",
                "Prefer visible layout evidence from guidance.png over the original coarse layout.",
            ],
        }
        last_error: Exception | None = None
        for _ in range(self.max_retries):
            payload = dict(base_payload)
            if last_error:
                payload["previous_error"] = str(last_error)
                payload["correction_instruction"] = "Return corrected strict JSON only."
            try:
                graph_data = self.client.vision_json(
                    system_prompt=system_prompt,
                    user_prompt=json.dumps(payload, indent=2),
                    image_path=guidance_path,
                    model=self.vision_model,
                    json_schema=VisionSceneGraph.model_json_schema(),
                    schema_name="VisionSceneGraph",
                    max_retries=1,
                )
                graph = VisionSceneGraph.model_validate(graph_data)
                errors = validate_scene_graph_matches(graph, scene)
                if errors:
                    raise ValueError("; ".join(errors))
                return graph
            except (ValidationError, RuntimeError, ValueError) as exc:
                last_error = exc
        raise RuntimeError(f"vision scene graph extraction failed: {last_error}")


def build_guidance_prompt(prompt: str, scene: SceneSpec, registry: AssetRegistry) -> str:
    object_lines = []
    for obj in scene.objects:
        asset_name = registry.get(obj.asset_id).name if obj.asset_id else (obj.name or obj.category)
        relation = "anchor" if obj.role == "anchor" else (f"{obj.relation} {obj.parent_id}" if obj.parent_id else obj.relation)
        object_lines.append(f"- {obj.id}: {asset_name}, category {obj.category}, {relation}")
    return (
        "Create a realistic single wide-angle reference image for a 3D scene layout. "
        "Use a three-quarter camera view with visible floor contact, clear object separation, and no text labels. "
        "The image is guidance for extracting spatial relations, so every planned object below must be fully visible, "
        "inside the frame, unhidden, and separated enough for object segmentation. Do not crop required objects at the image edge. "
        "For repeated categories, show separate visible instances. Show hand trucks/dollies and barrels/drums as recognizable warehouse objects.\n\n"
        f"User prompt: {prompt}\n\n"
        "Planned objects and coarse relations:\n"
        + "\n".join(object_lines)
    )


def build_retrieval_candidates(scene: SceneSpec, registry: AssetRegistry) -> list[dict[str, Any]]:
    candidates = []
    for obj in scene.objects:
        asset = registry.get(obj.asset_id) if obj.asset_id else None
        candidates.append(
            {
                "object_id": obj.id,
                "category": obj.category,
                "asset_id": obj.asset_id,
                "asset_name": asset.name if asset else None,
                "dimensions": asset.dimensions if asset else None,
                "source": asset.source if asset else None,
                "source_id": asset.source_id if asset else None,
                "license": asset.license if asset else None,
            }
        )
    return candidates


def validate_scene_graph_matches(graph: VisionSceneGraph, scene: SceneSpec) -> list[str]:
    object_ids = {obj.id for obj in scene.objects}
    errors: list[str] = []
    if not graph.objects:
        errors.append("vision scene graph must contain at least one object")
    if graph.anchor_object_id and graph.anchor_object_id not in object_ids:
        errors.append(f"anchor_object_id is not in planned scene: {graph.anchor_object_id}")
    matched = [obj.matched_object_id for obj in graph.objects if obj.matched_object_id]
    unknown = sorted(set(matched) - object_ids)
    if unknown:
        errors.append(f"unknown matched_object_id values: {unknown}")
    for relation in graph.relations:
        for field_name, object_id in {
            "subject_object_id": relation.subject_object_id,
            "target_object_id": relation.target_object_id,
        }.items():
            if object_id and object_id not in object_ids:
                errors.append(f"{field_name} is not in planned scene: {object_id}")
    return errors
