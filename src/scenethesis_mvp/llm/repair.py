from __future__ import annotations

from copy import deepcopy
from typing import Any

from scenethesis_mvp.schemas.scene_spec import ConstraintSpec, SceneSpec


class RepairEngine:
    """Applies simple relation edits requested by the judge."""

    def apply(self, scene: SceneSpec, repair_actions: list[dict[str, Any]]) -> SceneSpec:
        updated = deepcopy(scene)
        objects = {obj.id: obj for obj in updated.objects}

        def upsert_constraint(subject_id: str, relation: str, target_id: str | None) -> None:
            for constraint in updated.constraints:
                if constraint.subject_id == subject_id:
                    constraint.type = relation
                    constraint.target_id = target_id
                    return
            updated.constraints.append(ConstraintSpec(type=relation, subject_id=subject_id, target_id=target_id))

        for action in repair_actions:
            action_type = action.get("type")
            object_id = action.get("object_id")
            if action_type == "change_relation" and object_id in objects:
                obj = objects[object_id]
                target_id = action.get("target_id") or action.get("parent_id") or action.get("new_parent_id")
                relation = action.get("relation")
                if target_id in objects and relation:
                    obj.parent_id = target_id if relation in {"on", "inside"} else obj.parent_id
                    obj.relation = relation
                    upsert_constraint(obj.id, relation, target_id)
            elif action_type == "set_parent" and object_id in objects:
                parent_id = action.get("parent_id") or action.get("new_parent_id")
                relation = action.get("relation", "on")
                if parent_id in objects:
                    objects[object_id].relation = relation
                    if relation in {"on", "inside"}:
                        objects[object_id].parent_id = parent_id
                        objects[object_id].role = "child"
                    else:
                        objects[object_id].parent_id = None
                        if objects[object_id].role == "child":
                            objects[object_id].role = "parent"
                    upsert_constraint(object_id, relation, parent_id)
            elif action_type == "move_near" and object_id in objects:
                target_id = action.get("target_id") or action.get("parent_id") or action.get("new_parent_id")
                if target_id in objects:
                    objects[object_id].parent_id = None
                    objects[object_id].relation = "near"
                    if objects[object_id].role == "child":
                        objects[object_id].role = "parent"
                    upsert_constraint(object_id, "near", target_id)
            elif action_type == "spread_children":
                continue
            elif action_type == "reoptimize":
                continue
        return SceneSpec.model_validate(updated.model_dump())
