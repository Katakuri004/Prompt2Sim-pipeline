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
    relation_lines = []
    constraints = {constraint.subject_id: constraint for constraint in scene.constraints}
    role_order = {"anchor": 0, "parent": 1, "child": 2}
    ordered_objects = sorted(scene.objects, key=lambda obj: (role_order[obj.role], obj.id))
    for obj in ordered_objects:
        asset = registry.get(obj.asset_id) if obj.asset_id else None
        asset_name = asset.name if asset else (obj.name or obj.category)
        description = f", description: {obj.description}" if obj.description else ""
        dimensions = f"; dimensions_m={asset.dimensions}" if asset else ""
        visual_traits = guidance_visual_traits(asset.tags) if asset else []
        traits = f"; exact_visual_traits={visual_traits}" if visual_traits else ""
        object_lines.append(
            f"- {obj.id}: {asset_name}; category={obj.category}; role={obj.role}"
            f"{dimensions}{traits}{description}"
        )
        if obj.role != "anchor" and obj.relation:
            constraint = constraints.get(obj.id)
            target_id = obj.parent_id or (constraint.target_id if constraint else None)
            if not target_id:
                raise RuntimeError(f"Guidance prompt relation for {obj.id} has no target object.")
            relation_lines.append(f"- {obj.id} must be clearly {obj.relation} {target_id}")
    return (
        "Create one realistic landscape warehouse reference image for 3D scene reconstruction. "
        "Use a wide three-quarter camera with generous margin on every edge, visible floor contact, realistic indoor scale, "
        "and clear separation between objects. The exact inventory count is mandatory. Do not add other distinct movable props. "
        "Every listed instance must be visible, identifiable, unmerged, and large enough for segmentation and asset matching. "
        "Keep floor objects fully in frame. Keep small supported objects in the foreground or midground and unobstructed. "
        "Repeated categories must appear as separate instances, not as a pile or fused group. Do not add text labels.\n\n"
        f"User prompt: {prompt}\n\n"
        f"Exact planned inventory ({len(scene.objects)} objects):\n"
        + "\n".join(object_lines)
        + "\n\nRequired visible relations:\n"
        + ("\n".join(relation_lines) if relation_lines else "- The single anchor has no binary relation.")
    )


def guidance_visual_traits(tags: list[str]) -> list[str]:
    generic = {
        "warehouse",
        "industrial",
        "floor",
        "child",
        "parent",
        "anchor",
        "real",
        "asset",
    }
    return [tag for tag in tags if tag.lower() not in generic]


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
