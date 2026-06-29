from __future__ import annotations

import os
import json
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.procedural_assets import normalize_category
from scenethesis_mvp.llm.openai_client import OpenAIClient
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import read_text


class ScenePlanner:
    def __init__(
        self,
        client: OpenAIClient | None = None,
        model: str = "gpt-4o-mini",
        system_prompt_path: str | Path | None = None,
        max_retries: int = 3,
        max_objects: int = 18,
    ):
        self.client = client or OpenAIClient()
        self.model = os.getenv("OPENAI_MODEL", model)
        self.system_prompt_path = Path(system_prompt_path) if system_prompt_path else None
        self.max_retries = max_retries
        self.max_objects = max_objects

    def plan(self, prompt: str, registry: AssetRegistry, bounds: tuple[float, float, float]) -> SceneSpec:
        if not self.client.configured:
            raise RuntimeError("OPENAI_API_KEY is required for scene planning. Set it in the environment or .env.")
        try:
            return self._plan_with_openai(prompt, registry, bounds)
        except Exception as exc:
            raise RuntimeError(f"OpenAI planner failed to produce a valid SceneSpec: {exc}") from exc

    def _plan_with_openai(
        self,
        prompt: str,
        registry: AssetRegistry,
        bounds: tuple[float, float, float],
    ) -> SceneSpec:
        system_prompt = read_text(self.system_prompt_path) if self.system_prompt_path else ""
        registry_summary = [
            {
                "id": asset.id,
                "category": asset.category,
                "name": asset.name,
                "dimensions": list(asset.dimensions),
                "support_kind": asset.support_kind,
                "support_heights": asset.support_heights,
            }
            for asset in registry.assets
        ]
        category_requirements = prompt_category_requirements(prompt, registry)
        trait_requirements = prompt_asset_trait_requirements(prompt, registry)
        instance_plan = required_instance_plan(category_requirements, trait_requirements)
        if len(instance_plan) > self.max_objects:
            raise ValueError(
                "prompt requires "
                f"{len(instance_plan)} deterministic object instances, exceeding maximum_object_count="
                f"{self.max_objects}: {json.dumps(instance_plan, sort_keys=True)}"
            )
        user_payload = {
            "prompt": prompt,
            "scene_bounds": bounds,
            "available_assets": registry_summary,
            "required_category_counts": category_requirements,
            "required_instance_plan": instance_plan,
            "maximum_object_count": self.max_objects,
            "required_asset_traits": trait_requirements,
            "required_category_contract": {
                "rule": "For each required_category_counts entry, create at least that many objects whose category exactly equals the entry key.",
                "examples": required_category_examples(category_requirements),
            },
            "required_asset_trait_contract": {
                "rule": (
                    "When required_asset_traits is non-empty, satisfy each item using object name/description text. "
                    "For example, a trait 'wooden' in category 'box' requires a box object named or described as a wooden crate."
                )
            },
            "minimum_requirements": [
                "At least 8 objects for a lab/workbench scene.",
                "Include every prompt-critical category available in the registry.",
                "If the prompt uses a plural object word such as boxes or tools, include at least two instances when possible.",
                "If the prompt names material/type variants such as cardboard boxes, wooden crates, or plastic crates, preserve those variants in object names/descriptions.",
                "Every non-anchor object must have a relation.",
                "Every child object must have a parent_id.",
                "Create a constraint for every non-anchor object relation.",
                "Object ids must be semantic instance ids such as shelf_01 or box_01, never registry asset ids.",
            ],
            "object_count_contract": (
                "Do not exceed maximum_object_count. Treat required_instance_plan as the default complete inventory. "
                "Add an object outside that plan only when the user prompt explicitly names its category, and remain "
                "within maximum_object_count. Every planned object must be independently visible in one guidance image, "
                "segmented, depth-projected, and matched to a real asset. Do not add decorative inventory."
            ),
            "object_id_contract": {
                "rule": "object.id identifies a scene instance; object.asset_id identifies the selected asset.",
                "valid_examples": ["shelf_01", "table_01", "box_01", "chair_01"],
                "invalid_examples": [asset.id for asset in registry.assets[:8]],
            },
            "constraint_contract": {
                "required": True,
                "rule": "Every object whose role is not anchor must have exactly one matching entry in constraints.",
                "matching_fields": {
                    "constraints[].subject_id": "must exactly equal the non-anchor object id",
                    "constraints[].type": "must exactly equal the non-anchor object relation",
                    "constraints[].target_id": "must equal parent_id when parent_id is set; otherwise use the anchor or nearby object used for the relation",
                },
                "example": {
                    "object": {"id": "box_01", "role": "child", "parent_id": "shelf_01", "relation": "on"},
                    "constraint": {"type": "on", "subject_id": "box_01", "target_id": "shelf_01"},
                },
            },
        }
        last_error: Exception | None = None
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ]
        for _ in range(self.max_retries):
            try:
                data = self.client.chat_json(
                    messages=messages,
                    model=self.model,
                    json_schema=SceneSpec.model_json_schema(),
                    schema_name="SceneSpec",
                    max_retries=1,
                )
                scene = SceneSpec.model_validate(data)
                normalize_planned_categories(scene, registry)
                errors = validate_planned_scene(prompt, scene, registry, max_objects=self.max_objects)
                if errors:
                    raise ValueError(
                        planner_validation_feedback(scene, errors, category_requirements, trait_requirements)
                    )
                return scene
            except (ValidationError, RuntimeError, ValueError) as exc:
                last_error = exc
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous SceneSpec failed validation: "
                            f"{exc}. Return a corrected strict JSON SceneSpec only. "
                            "Do not omit constraints for any non-anchor object."
                        ),
                    }
                )
        raise RuntimeError(f"planner failed to produce valid SceneSpec: {last_error}")


def validate_planned_scene(
    prompt: str,
    scene: SceneSpec,
    registry: AssetRegistry,
    max_objects: int = 18,
) -> list[str]:
    prompt_lower = prompt.lower()
    categories = [normalize_category(obj.category) for obj in scene.objects]
    category_requirements = prompt_category_requirements(prompt, registry)
    errors: list[str] = []
    if len(scene.objects) < 6:
        errors.append("scene must contain at least 6 objects")
    if len(scene.objects) > max_objects:
        errors.append(f"scene contains {len(scene.objects)} objects; maximum is {max_objects}")
    if any(term in prompt_lower for term in ["lab", "workbench", "robotics"]) and len(scene.objects) < 8:
        errors.append("robotics/workbench lab scenes must contain at least 8 objects")
    for category, minimum in category_requirements.items():
        planned_count = categories.count(category)
        if planned_count < minimum:
            planned_ids = [obj.id for obj in scene.objects if normalize_category(obj.category) == category]
            missing_ids = [item["id"] for item in required_instance_plan({category: minimum})][planned_count:minimum]
            errors.append(
                f"prompt requires at least {minimum} {category} object(s), but only {planned_count} were planned; "
                f"planned ids={planned_ids}; add missing instance ids={missing_ids}"
            )
    for requirement in prompt_asset_trait_requirements(prompt, registry):
        count = count_trait_objects(scene, requirement["category"], requirement["trait"])
        if count < requirement["minimum"]:
            errors.append(
                f"prompt requires at least {requirement['minimum']} {requirement['trait']} "
                f"{requirement['category']} object(s), but only {count} were planned with that trait"
            )
    anchors = [obj for obj in scene.objects if obj.role == "anchor"]
    if len(anchors) != 1:
        errors.append("scene must contain exactly one anchor")
    asset_ids = {asset.id for asset in registry.assets}
    registry_categories = set(registry.categories)
    for obj in scene.objects:
        if obj.id in asset_ids:
            errors.append(f"{obj.id} uses a registry asset id as object id; use a semantic instance id instead")
        if obj.category not in registry_categories:
            errors.append(f"{obj.id} has unknown category {obj.category!r}; use one of {sorted(registry_categories)}")
        if obj.role != "anchor" and not obj.relation:
            errors.append(f"{obj.id} is non-anchor but has no relation")
        if obj.relation in {"on", "inside"} and not obj.parent_id:
            errors.append(f"{obj.id} is child but has no parent_id")
        if obj.relation == "facing" and not prompt_requests_facing_relation(prompt_lower):
            errors.append(f"{obj.id} uses facing, but the prompt does not request an orientation relation")
    constraint_subjects = {constraint.subject_id for constraint in scene.constraints}
    for obj in scene.objects:
        if obj.role != "anchor" and obj.id not in constraint_subjects:
            errors.append(f"{obj.id} is missing an explicit relation constraint")
    return errors


def normalize_planned_categories(scene: SceneSpec, registry: AssetRegistry) -> None:
    registry_categories = set(registry.categories)
    for obj in scene.objects:
        normalized = normalize_category(obj.category)
        if normalized in registry_categories:
            obj.category = normalized


def planner_validation_feedback(
    scene: SceneSpec,
    errors: list[str],
    category_requirements: dict[str, int] | None = None,
    trait_requirements: list[dict[str, Any]] | None = None,
) -> str:
    required_constraints = []
    anchors = [obj.id for obj in scene.objects if obj.role == "anchor"]
    default_target = anchors[0] if anchors else None
    constraint_subjects = {constraint.subject_id for constraint in scene.constraints}
    missing_subjects = []
    for obj in scene.objects:
        if obj.role == "anchor":
            continue
        if obj.id not in constraint_subjects:
            missing_subjects.append(obj.id)
        required_constraints.append(
            {
                "type": obj.relation,
                "subject_id": obj.id,
                "target_id": obj.parent_id or default_target,
            }
        )
    return (
        "; ".join(errors)
        + ". Required instance plan: "
        + json.dumps(required_instance_plan(category_requirements or {}, trait_requirements), sort_keys=True)
        + ". Missing constraint subjects: "
        + json.dumps(missing_subjects, sort_keys=True)
        + ". Required non-anchor constraints: "
        + json.dumps(required_constraints, sort_keys=True)
    )


def required_category_examples(requirements: dict[str, int]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for category, minimum in sorted(requirements.items()):
        examples.append(
            {
                "category": category,
                "minimum": minimum,
                "example_object_ids": [f"{category}_{index:02d}" for index in range(1, minimum + 1)],
            }
        )
    return examples


def required_instance_plan(
    requirements: dict[str, int],
    trait_requirements: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    named_ids = {
        "bag": ["cement_bag_01", "cement_bag_02"],
        "barrier": ["safety_barrier_01", "safety_barrier_02"],
        "bin": ["bin_01", "trash_can_01"],
        "box": [
            "cardboard_box_01",
            "cardboard_box_02",
            "cardboard_box_03",
            "wooden_crate_01",
            "wooden_crate_02",
            "plastic_crate_01",
            "plastic_crate_02",
            "supply_crate_01",
        ],
        "cabinet": ["tool_chest_01", "cabinet_01"],
        "cable": ["cable_01", "cable_02"],
        "camera": ["security_camera_01"],
        "cart": ["platform_cart_01"],
        "chair": ["chair_01"],
        "container": ["jerrycan_01", "jerrycan_02", "oil_can_01"],
        "cylinder": ["barrel_01", "barrel_02", "propane_tank_01"],
        "door": ["roller_shutter_door_01"],
        "duct": ["air_duct_01"],
        "floor_marking": ["safety_tape_01"],
        "forklift": ["forklift_01"],
        "hand_truck": ["hand_truck_01"],
        "ladder": ["ladder_01"],
        "light": ["light_01", "light_02", "light_03"],
        "pallet": ["pallet_01", "pallet_02"],
        "pallet_load": ["wrapped_pallet_load_01"],
        "pipe": ["pipe_01"],
        "scanner": ["barcode_scanner_01"],
        "shelf": ["pallet_rack_01"],
        "sign": ["safety_sign_01"],
        "table": ["packing_table_01"],
        "tool": ["tool_01", "tool_02", "tool_03"],
        "utility_box": ["utility_box_01"],
    }
    plan: list[dict[str, Any]] = []
    for category, minimum in sorted(requirements.items()):
        preferred = trait_instance_ids(category, trait_requirements or [])
        preferred.extend(object_id for object_id in named_ids.get(category, []) if object_id not in preferred)
        for index in range(minimum):
            object_id = preferred[index] if index < len(preferred) else f"{category}_{index + 1:02d}"
            plan.append({"id": object_id, "category": category})
    return plan


def trait_instance_ids(category: str, requirements: list[dict[str, Any]]) -> list[str]:
    prefixes = {
        ("box", "cardboard"): "cardboard_box",
        ("box", "wooden"): "wooden_crate",
        ("box", "plastic"): "plastic_crate",
        ("bin", "trash"): "trash_can",
    }
    object_ids: list[str] = []
    for requirement in requirements:
        if requirement.get("category") != category:
            continue
        prefix = prefixes.get((category, str(requirement.get("trait"))))
        if not prefix:
            continue
        for index in range(1, int(requirement.get("minimum", 0)) + 1):
            object_ids.append(f"{prefix}_{index:02d}")
    return object_ids


def count_trait_objects(scene: SceneSpec, category: str, trait: str) -> int:
    trait_terms = {
        "cardboard": {"cardboard", "shipping"},
        "wooden": {"wooden", "wood", "crate"},
        "plastic": {"plastic", "crate"},
        "trash": {"trash", "garbage"},
    }.get(trait, {trait})
    count = 0
    for obj in scene.objects:
        if normalize_category(obj.category) != category:
            continue
        text = " ".join([obj.id, obj.name or "", obj.description or ""]).lower()
        if any(term in text for term in trait_terms):
            count += 1
    return count


def prompt_asset_trait_requirements(prompt: str, registry: AssetRegistry) -> list[dict[str, Any]]:
    requirements: list[dict[str, Any]] = []
    prompt_lower = prompt.lower()
    if "box" in registry.categories:
        if prompt_has_term(prompt_lower, "cardboard box") or prompt_has_term(prompt_lower, "cardboard boxes"):
            requirements.append({"category": "box", "trait": "cardboard", "minimum": 2 if prompt_has_term(prompt_lower, "cardboard boxes") else 1})
        if prompt_has_term(prompt_lower, "wooden crate") or prompt_has_term(prompt_lower, "wooden crates"):
            requirements.append({"category": "box", "trait": "wooden", "minimum": 1})
        if prompt_has_term(prompt_lower, "plastic crate") or prompt_has_term(prompt_lower, "plastic crates"):
            requirements.append({"category": "box", "trait": "plastic", "minimum": 1})
    if "bin" in registry.categories and (prompt_has_term(prompt_lower, "trash can") or prompt_has_term(prompt_lower, "garbage can")):
        requirements.append({"category": "bin", "trait": "trash", "minimum": 1})
    return requirements


def prompt_category_requirements(prompt: str, registry: AssetRegistry) -> dict[str, int]:
    prompt_lower = prompt.lower()
    term_categories = {
        "workbench": "table",
        "packing table": "table",
        "table": "table",
        "shelf": "shelf",
        "shelves": "shelf",
        "rack": "shelf",
        "racks": "shelf",
        "chair": "chair",
        "robot arm": "robot_arm",
        "robot": "robot_arm",
        "monitor": "monitor",
        "bin": "bin",
        "trash can": "bin",
        "container": "container",
        "jerry can": "container",
        "jerrycan": "container",
        "fuel can": "container",
        "cement bag": "bag",
        "bag": "bag",
        "box": "box",
        "crate": "box",
        "wooden crate": "box",
        "plastic crate": "box",
        "barrel": "cylinder",
        "drum": "cylinder",
        "tank": "cylinder",
        "propane": "cylinder",
        "hand truck": "hand_truck",
        "dolly": "hand_truck",
        "cart": "cart",
        "carts": "cart",
        "trolley": "cart",
        "trolleys": "cart",
        "forklift": "forklift",
        "forklifts": "forklift",
        "pallet stacker": "forklift",
        "walk-behind stacker": "forklift",
        "pallet": "pallet",
        "pallets": "pallet",
        "wrapped pallet": "pallet_load",
        "wrapped pallets": "pallet_load",
        "pallet load": "pallet_load",
        "pallet loads": "pallet_load",
        "barcode scanner": "scanner",
        "scanner": "scanner",
        "scanners": "scanner",
        "tool chest": "cabinet",
        "toolbox": "tool",
        "drill": "tool",
        "wrench": "tool",
        "bench vice": "tool",
        "ladder": "ladder",
        "safety sign": "sign",
        "wet floor sign": "sign",
        "light": "light",
        "fluorescent": "light",
        "lamp": "light",
        "roll-up door": "door",
        "rollup door": "door",
        "roller shutter": "door",
        "warehouse door": "door",
        "power box": "utility_box",
        "utility box": "utility_box",
        "electrical box": "utility_box",
        "air duct": "duct",
        "duct": "duct",
        "pipe": "pipe",
        "pipes": "pipe",
        "cable": "cable",
        "cables": "cable",
        "electric cable": "cable",
        "electric cables": "cable",
        "barrier": "barrier",
        "barriers": "barrier",
        "safety barrier": "barrier",
        "safety tape": "floor_marking",
        "floor marking": "floor_marking",
        "floor markings": "floor_marking",
        "aisle marking": "floor_marking",
        "aisle markings": "floor_marking",
        "fence": "barrier",
        "chainlink": "barrier",
        "chain link": "barrier",
        "security camera": "camera",
        "camera": "camera",
        "surveillance camera": "camera",
        "crowbar": "tool",
        "sledgehammer": "tool",
        "hammer": "tool",
        "bolt cutters": "tool",
        "measuring tape": "tool",
        "screwdriver": "tool",
    }
    requirements: dict[str, int] = {}
    for term, category in term_categories.items():
        if prompt_has_term(prompt_lower, term) and category in registry.categories:
            requirements[category] = max(requirements.get(category, 0), 1)
    plural_counts = {
        "boxes": ("box", 2),
        "crates": ("box", 2),
        "barrels": ("cylinder", 2),
        "drums": ("cylinder", 2),
        "tanks": ("cylinder", 2),
        "tools": ("tool", 2),
        "jerry cans": ("container", 2),
        "containers": ("container", 2),
        "cement bags": ("bag", 2),
        "forklifts": ("forklift", 1),
        "pallets": ("pallet", 2),
        "wrapped pallets": ("pallet_load", 1),
        "pallet loads": ("pallet_load", 1),
        "carts": ("cart", 1),
        "trolleys": ("cart", 1),
        "scanners": ("scanner", 1),
        "lights": ("light", 2),
        "pipes": ("pipe", 1),
        "cables": ("cable", 1),
        "barriers": ("barrier", 1),
        "floor markings": ("floor_marking", 1),
        "aisle markings": ("floor_marking", 1),
    }
    for term, (category, minimum) in plural_counts.items():
        if prompt_has_term(prompt_lower, term) and category in registry.categories:
            requirements[category] = max(requirements.get(category, 0), minimum)
    if prompt_has_term(prompt_lower, "many") and (
        prompt_has_term(prompt_lower, "boxes") or prompt_has_term(prompt_lower, "crates")
    ) and "box" in registry.categories:
        requirements["box"] = max(requirements.get("box", 0), 5)
    if prompt_has_term(prompt_lower, "warehouse"):
        contextual = {
            "shelf": 1,
            "table": 1,
            "forklift": 1,
            "barrier": 1,
            "pallet": 1,
            "floor_marking": 1,
            "box": 2,
        }
        for category, minimum in contextual.items():
            if category in registry.categories:
                requirements[category] = max(requirements.get(category, 0), minimum)
    return requirements


def prompt_has_term(prompt_lower: str, term: str) -> bool:
    pattern = r"(?<![a-z0-9_])" + re.escape(term.lower()) + r"(?![a-z0-9_])"
    return re.search(pattern, prompt_lower) is not None


def prompt_requests_facing_relation(prompt_lower: str) -> bool:
    return any(
        phrase in prompt_lower
        for phrase in (
            "facing",
            "faces the",
            "face the",
            "oriented toward",
            "oriented towards",
            "pointing toward",
            "pointing towards",
        )
    )
