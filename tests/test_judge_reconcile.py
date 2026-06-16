from __future__ import annotations

from scenethesis_mvp.llm.judge import drop_noop_repair_actions_when_actionable_remains, validate_judge_response
from scenethesis_mvp.schemas.scene_spec import ConstraintSpec, ObjectSpec, SceneSpec


def test_judge_repair_actions_must_use_scene_object_ids() -> None:
    scene = SceneSpec(
        prompt="warehouse",
        objects=[
            ObjectSpec(id="shelf_01", category="shelf", asset_id="real_warehouse_shelf_01", role="anchor"),
            ObjectSpec(id="chair_01", category="chair", asset_id="real_warehouse_chair_01", role="parent"),
        ],
    )
    judge = {
        "needs_repair": True,
        "repair_actions": [{"type": "move_near", "object_id": "real_warehouse_chair_01", "target_id": "shelf_01"}],
    }
    errors = validate_judge_response(scene, judge)
    assert errors
    assert "asset_id" in errors[0]


def test_judge_rejects_invalid_set_parent_relation() -> None:
    scene = SceneSpec(
        prompt="warehouse",
        objects=[
            ObjectSpec(id="shelf_01", category="shelf", role="anchor"),
            ObjectSpec(id="barrel_01", category="cylinder", role="parent", relation="near"),
        ],
    )
    judge = {
        "needs_repair": True,
        "repair_actions": [{"type": "set_parent", "object_id": "barrel_01", "parent_id": "shelf_01", "relation": "near"}],
    }
    errors = validate_judge_response(scene, judge)
    assert any("set_parent" in error and "on" in error for error in errors)


def test_judge_rejects_noop_repair_action() -> None:
    scene = SceneSpec(
        prompt="warehouse",
        objects=[
            ObjectSpec(id="shelf_01", category="shelf", role="anchor"),
            ObjectSpec(id="box_01", category="box", role="child", parent_id="shelf_01", relation="on"),
        ],
    )
    judge = {
        "needs_repair": True,
        "repair_actions": [{"type": "set_parent", "object_id": "box_01", "parent_id": "shelf_01", "relation": "on"}],
    }
    errors = validate_judge_response(scene, judge)
    assert any("no-op" in error for error in errors)


def test_judge_drops_noop_actions_only_when_actionable_repairs_remain() -> None:
    scene = SceneSpec(
        prompt="warehouse",
        objects=[
            ObjectSpec(id="shelf_01", category="shelf", role="anchor"),
            ObjectSpec(id="box_01", category="box", role="child", parent_id="shelf_01", relation="on"),
            ObjectSpec(id="cart_01", category="cart", role="parent", relation="near"),
        ],
        constraints=[ConstraintSpec(type="on", subject_id="box_01", target_id="shelf_01")],
    )
    judge = {
        "needs_repair": True,
        "repair_actions": [
            {"type": "change_relation", "object_id": "box_01", "target_id": "shelf_01", "relation": "on"},
            {"type": "move_near", "object_id": "cart_01", "target_id": "shelf_01"},
        ],
        "notes": "mixed",
    }
    cleaned = drop_noop_repair_actions_when_actionable_remains(scene, judge)
    assert cleaned["repair_actions"] == [{"type": "move_near", "object_id": "cart_01", "target_id": "shelf_01"}]
    assert validate_judge_response(scene, cleaned) == []


def test_judge_keeps_all_noop_actions_invalid() -> None:
    scene = SceneSpec(
        prompt="warehouse",
        objects=[
            ObjectSpec(id="shelf_01", category="shelf", role="anchor"),
            ObjectSpec(id="box_01", category="box", role="child", parent_id="shelf_01", relation="on"),
        ],
        constraints=[ConstraintSpec(type="on", subject_id="box_01", target_id="shelf_01")],
    )
    judge = {
        "needs_repair": True,
        "repair_actions": [{"type": "change_relation", "object_id": "box_01", "target_id": "shelf_01", "relation": "on"}],
        "notes": "noop only",
    }
    cleaned = drop_noop_repair_actions_when_actionable_remains(scene, judge)
    assert cleaned == judge
    assert validate_judge_response(scene, cleaned)
