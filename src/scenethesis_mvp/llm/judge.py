from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from scenethesis_mvp.llm.openai_client import OpenAIClient
from scenethesis_mvp.schemas.mesh_metrics import MeshMetrics
from scenethesis_mvp.schemas.metrics import Metrics
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import read_text


class JudgeScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object_category_accuracy: float = Field(ge=0.0, le=1.0)
    orientation_alignment: float = Field(ge=0.0, le=1.0)
    spatial_coherence: float = Field(ge=0.0, le=1.0)
    physical_plausibility: float = Field(ge=0.0, le=1.0)
    prompt_alignment: float = Field(ge=0.0, le=1.0)


class JudgeRepairAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["change_relation", "move_near", "spread_children", "reoptimize"]
    object_id: str | None = None
    target_id: str | None = None
    parent_id: str | None = None
    relation: Literal["on", "inside", "next_to", "near", "in_front_of", "behind", "left_of", "right_of"] | None = None


class JudgeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: JudgeScores
    needs_repair: bool
    repair_actions: list[JudgeRepairAction]
    notes: str


class SceneJudge:
    def __init__(
        self,
        client: OpenAIClient | None = None,
        model: str = "gpt-4o-mini",
        system_prompt_path: str | Path | None = None,
        max_retries: int = 3,
    ):
        self.client = client or OpenAIClient()
        self.model = os.getenv("OPENAI_VISION_MODEL", model)
        self.system_prompt_path = Path(system_prompt_path) if system_prompt_path else None
        self.max_retries = max_retries

    def judge(
        self,
        prompt: str,
        scene: SceneSpec,
        render_path: str | Path,
        metrics: Metrics,
        mesh_metrics: MeshMetrics | None = None,
        extra_image_paths: list[str | Path] | None = None,
    ) -> dict[str, Any]:
        if not self.client.configured:
            raise RuntimeError("OPENAI_API_KEY is required for vision judging. Set it in the environment or .env.")
        if not Path(render_path).exists():
            raise RuntimeError(f"render image not found for vision judging: {render_path}")
        try:
            judge = self._judge_with_openai(prompt, scene, render_path, metrics, mesh_metrics, extra_image_paths=extra_image_paths)
            if (
                _all_repairs_are_noops(scene, judge.get("repair_actions", []))
                and metrics.stable
                and metrics.collision_count == 0
                and (mesh_metrics is None or mesh_metrics.mesh_clean)
            ):
                judge = self._judge_with_openai(
                    prompt,
                    scene,
                    render_path,
                    metrics,
                    mesh_metrics,
                    extra_image_paths=extra_image_paths,
                    extra_context=(
                        "Your previous repair_actions were already satisfied by the SceneSpec. "
                        "Re-evaluate carefully using the object inventory and metrics. "
                        "Only set needs_repair=true if a new, actionable repair remains."
                    ),
                )
            return judge
        except Exception as exc:
            raise RuntimeError(f"OpenAI vision judge failed: {exc}") from exc

    def _judge_with_openai(
        self,
        prompt: str,
        scene: SceneSpec,
        render_path: str | Path,
        metrics: Metrics,
        mesh_metrics: MeshMetrics | None = None,
        extra_image_paths: list[str | Path] | None = None,
        extra_context: str | None = None,
    ) -> dict[str, Any]:
        system_prompt = read_text(self.system_prompt_path) if self.system_prompt_path else ""
        last_error: Exception | None = None
        last_validation_errors: list[str] = []
        for _ in range(self.max_retries):
            previous_errors_were_only_noops = bool(last_validation_errors) and _only_noop_errors(last_validation_errors)
            payload = {
                "prompt": prompt,
                "scene_spec": scene.model_dump(mode="json"),
                "valid_object_ids": [obj.id for obj in scene.objects],
                "repair_id_rule": "Use only valid_object_ids in repair_actions. Do not use asset_id values.",
                "needs_repair_semantics": (
                    "needs_repair=true means at least one repair_action would change the current SceneSpec or request "
                    "a real reoptimization. If the scene is imperfect but every proposed parent/relation action is "
                    "already true, set needs_repair=false, leave repair_actions empty, lower the relevant scores, "
                    "and explain the residual issue in notes."
                ),
                "repair_action_schemas": [
                    {
                        "type": "change_relation",
                        "required": ["object_id", "target_id", "relation"],
                        "use": "Change an object's explicit relation/target. Use relation on or inside for support parent changes.",
                    },
                    {
                        "type": "move_near",
                        "required": ["object_id", "target_id"],
                        "use": "Move an object near another object only if it is not already near that target.",
                    },
                    {
                        "type": "spread_children",
                        "required": ["parent_id"],
                        "use": "Request support-surface redistribution for children of a shelf/table.",
                    },
                    {
                        "type": "reoptimize",
                        "required": [],
                        "use": "Request a real optimization rerun when metrics or visible layout require it.",
                    },
                ],
                "object_inventory": [
                    {
                        "id": obj.id,
                        "category": obj.category,
                        "asset_id": obj.asset_id,
                        "name": obj.name,
                        "parent_id": obj.parent_id,
                        "relation": obj.relation,
                    }
                    for obj in scene.objects
                ],
                "current_constraints": [
                    {
                        "subject_id": constraint.subject_id,
                        "type": constraint.type,
                        "target_id": constraint.target_id,
                    }
                    for constraint in scene.constraints
                ],
                "metrics": metrics.model_dump(mode="json"),
                "mesh_metrics": mesh_metrics.model_dump(mode="json") if mesh_metrics else None,
                "image_order": ["primary_render"] + [f"extra_image_{index + 1}" for index, _ in enumerate(extra_image_paths or [])],
                "multi_view_instruction": (
                    "When extra images are present, judge the same scene across all views. "
                    "Use the guidance image for prompt/layout intent and rendered alternate views for occlusion and support checks."
                ),
                "extra_context": extra_context,
            }
            if last_error:
                payload["previous_validation_error"] = str(last_error)
                payload["correction_instruction"] = (
                    "Return corrected strict JSON only. Repair actions must reference scene object ids, "
                    "not asset ids. If previous_validation_error says a repair action is a no-op, do not repeat it. "
                    "Set needs_repair=false with repair_actions=[] unless you can propose an action that changes "
                    "the current SceneSpec."
                )
            if previous_errors_were_only_noops and metrics.stable and metrics.collision_count == 0:
                payload["nonactionable_repair_correction"] = {
                    "previous_errors": last_validation_errors,
                    "mandatory_rule": (
                        "The previous repair actions were already true in current_constraints. "
                        "Do not emit change_relation or move_near actions for those facts again. "
                        "If the remaining issue is only visual quality or imperfect composition, set "
                        "needs_repair=false, repair_actions=[], lower the relevant score, and explain it in notes. "
                        "Only use reoptimize if the render shows a concrete issue that a real optimization rerun can change."
                    ),
                }
            image_paths = [render_path] + list(extra_image_paths or [])
            judge = self.client.vision_json_multi(
                system_prompt=system_prompt,
                user_prompt=json.dumps(payload, indent=2),
                image_paths=image_paths,
                model=self.model,
                json_schema=JudgeResponse.model_json_schema(),
                schema_name="SceneJudgeResponse",
                max_retries=self.max_retries,
            )
            judge = JudgeResponse.model_validate(judge).model_dump(mode="json")
            judge = drop_noop_repair_actions_when_actionable_remains(scene, judge)
            errors = validate_judge_response(scene, judge)
            if not errors:
                return judge
            last_validation_errors = errors
            last_error = ValueError("; ".join(errors))
        raise RuntimeError(f"vision judge produced invalid repair actions: {last_error}")


def _all_repairs_are_noops(scene: SceneSpec, repair_actions: list[dict[str, Any]]) -> bool:
    if not repair_actions:
        return False
    return all(_repair_action_is_noop(scene, action) for action in repair_actions)


def drop_noop_repair_actions_when_actionable_remains(scene: SceneSpec, judge: dict[str, Any]) -> dict[str, Any]:
    actions = judge.get("repair_actions", [])
    if not isinstance(actions, list) or not actions:
        return judge
    actionable = [action for action in actions if isinstance(action, dict) and not _repair_action_is_noop(scene, action)]
    if not actionable or len(actionable) == len(actions):
        return judge
    updated = dict(judge)
    removed = len(actions) - len(actionable)
    updated["repair_actions"] = actionable
    note = str(updated.get("notes") or "")
    suffix = f" Removed {removed} no-op repair action(s) that were already satisfied by the SceneSpec."
    updated["notes"] = (note + suffix).strip()
    return updated


def _only_noop_errors(errors: list[str]) -> bool:
    return bool(errors) and all("no-op" in error for error in errors)


def _repair_action_is_noop(scene: SceneSpec, action: dict[str, Any]) -> bool:
    objects = {obj.id: obj for obj in scene.objects}
    constraints = {(constraint.subject_id, constraint.type, constraint.target_id) for constraint in scene.constraints}
    action_type = action.get("type")
    object_id = action.get("object_id")
    if action_type == "reoptimize":
        return False
    if action_type == "spread_children":
        parent_id = action.get("parent_id")
        return bool(parent_id and any(obj.parent_id == parent_id for obj in scene.objects))
    if object_id not in objects:
        return False
    obj = objects[object_id]
    target_id = action.get("target_id") or action.get("parent_id")
    relation = action.get("relation")
    if action_type == "set_parent":
        expected_relation = relation or "on"
        if expected_relation in {"on", "inside"}:
            return bool(target_id and obj.parent_id == target_id and obj.relation == expected_relation)
        return bool(target_id and obj.relation == expected_relation and (object_id, expected_relation, target_id) in constraints)
    if action_type == "change_relation":
        if not target_id or not relation or obj.relation != relation:
            return False
        if relation in {"on", "inside"} and obj.parent_id != target_id:
            return False
        return (object_id, relation, target_id) in constraints
    if action_type == "move_near":
        return bool(target_id and obj.relation == "near" and (object_id, "near", target_id) in constraints)
    return False


def validate_judge_response(scene: SceneSpec, judge: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    object_ids = {obj.id for obj in scene.objects}
    asset_ids = {obj.asset_id for obj in scene.objects if obj.asset_id}
    actions = judge.get("repair_actions", [])
    if judge.get("needs_repair") and not actions:
        errors.append("needs_repair is true but repair_actions is empty")
    if not isinstance(actions, list):
        return ["repair_actions must be a list"]
    allowed_action_types = {"change_relation", "move_near", "set_parent", "spread_children", "reoptimize"}
    allowed_relations = {"on", "inside", "next_to", "near", "in_front_of", "behind", "left_of", "right_of"}
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            errors.append(f"repair_actions[{index}] must be an object")
            continue
        action_type = action.get("type")
        if action_type not in allowed_action_types:
            errors.append(f"repair_actions[{index}].type is invalid: {action_type!r}")
            continue
        if action_type in {"change_relation", "move_near", "set_parent"}:
            object_id = action.get("object_id")
            if object_id not in object_ids:
                source = "asset_id" if object_id in asset_ids else "unknown id"
                errors.append(f"repair_actions[{index}].object_id must be a scene object id, got {object_id!r} ({source})")
        relation = action.get("relation")
        if action_type == "set_parent":
            if relation not in {"on", "inside"}:
                errors.append(f"repair_actions[{index}].relation for set_parent must be 'on' or 'inside', got {relation!r}")
            if not action.get("parent_id"):
                errors.append(f"repair_actions[{index}].parent_id is required for set_parent")
        if action_type == "change_relation" and relation not in allowed_relations:
            errors.append(f"repair_actions[{index}].relation is invalid: {relation!r}")
        if action_type == "move_near" and not (action.get("target_id") or action.get("parent_id")):
            errors.append(f"repair_actions[{index}] move_near requires target_id")
        if action_type == "spread_children":
            parent_id = action.get("parent_id")
            if parent_id not in object_ids:
                source = "asset_id" if parent_id in asset_ids else "unknown id"
                errors.append(f"repair_actions[{index}].parent_id must be a scene object id, got {parent_id!r} ({source})")
        for field in ("target_id", "parent_id", "new_parent_id"):
            value = action.get(field)
            if value and value not in object_ids:
                source = "asset_id" if value in asset_ids else "unknown id"
                errors.append(f"repair_actions[{index}].{field} must be a scene object id, got {value!r} ({source})")
        if judge.get("needs_repair") and _repair_action_is_noop(scene, action):
            errors.append(f"repair_actions[{index}] is already satisfied by the SceneSpec and is a no-op")
    return errors
