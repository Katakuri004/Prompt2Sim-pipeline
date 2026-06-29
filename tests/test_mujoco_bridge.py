from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import trimesh
import pytest

from scenethesis_mvp.mujoco_bridge.evaluator import evaluate_scene
from scenethesis_mvp.mujoco_bridge.mesh_assets import prepare_mesh_assets
from scenethesis_mvp.mujoco_bridge.mjcf_emitter import compile_scene_to_mjcf
from scenethesis_mvp.mujoco_bridge.mujoco_env import _grasp_frame_candidates, _score_grasp_probe
from scenethesis_mvp.mujoco_bridge.policies import ScriptedPickPlacePolicy, TeacherPickPlacePolicy, TeacherPlanUnavailable, make_policy
from scenethesis_mvp.mujoco_bridge.scene_ir import build_scene_ir, load_mujoco_config, _swept_path_clearance
from scenethesis_mvp.mujoco_bridge.schemas import PolicyContract
from scenethesis_mvp.mujoco_bridge.task_validation import validate_task_feasibility
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, PlacementSpec, SceneSpec
from scenethesis_mvp.utils.io import read_json, write_json


def _write_accepted_run(tmp_path: Path, scene: SceneSpec) -> Path:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    write_json(run_dir / "scene_spec.json", scene)
    write_json(run_dir / "qualification.json", {"accepted": True, "status": "accepted"})
    (run_dir / "scene.glb").write_bytes(b"placeholder")
    return run_dir


def _write_proxy_config(tmp_path: Path) -> Path:
    config = load_mujoco_config("configs/mujoco_eval.yaml")
    config["visual_scene"] = {"mode": "proxy"}
    config_path = tmp_path / "mujoco_eval_proxy.yaml"
    write_json(config_path, config)
    return config_path


def _write_visual_config(tmp_path: Path) -> Path:
    config = load_mujoco_config("configs/mujoco_eval.yaml")
    config["visual_scene"] = {"mode": "full_glb_visual", "bounds_tolerance_m": 0.9}
    config["task"]["destination_reach_margin_m"] = 0.0
    config_path = tmp_path / "mujoco_eval_visual.yaml"
    write_json(config_path, config)
    return config_path


def _write_tiny_semantic_glb(path: Path) -> None:
    scene = trimesh.Scene()
    floor = _mujoco_world_box_as_gltf_y_up(extents=(3.0, 3.0, 0.04), center=(1.5, 1.5, 0.0))
    target = _mujoco_world_box_as_gltf_y_up(extents=(0.38, 0.32, 0.28), center=(1.5, 1.5, 0.2))
    scene.add_geometry(floor, node_name="ground_visual", geom_name="ground_visual")
    scene.add_geometry(target, node_name="box_0_cardboard_box", geom_name="box_0_cardboard_box")
    path.write_bytes(scene.export(file_type="glb"))


def _mujoco_world_box_as_gltf_y_up(extents: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    vertices = mesh.vertices.copy()
    mesh.vertices = [
        [float(vertex[0]), float(vertex[2]), float(-vertex[1])]
        for vertex in vertices
    ]
    return mesh


def test_scene_ir_builds_target_task_and_physics_confidence(tmp_path: Path) -> None:
    scene = SceneSpec(
        scene_id="mujoco_fixture",
        prompt="box on table",
        objects=[
            ObjectSpec(id="table", category="table", asset_id="proc_table_01", role="anchor", placement=PlacementSpec(x=1.0, y=1.0, z=0.41)),
            ObjectSpec(id="box", category="box", asset_id="proc_box_01", role="child", parent_id="table", relation="on", placement=PlacementSpec(x=1.0, y=1.0, z=0.96)),
        ],
    )
    run_dir = _write_accepted_run(tmp_path, scene)

    scene_ir = build_scene_ir(run_dir, target_object="box")

    assert scene_ir.task.target_object == "box"
    assert scene_ir.task.support_id == "table"
    assert scene_ir.task.destination_position[:2] != [scene.object_by_id("box").placement.x, scene.object_by_id("box").placement.y]
    assert scene_ir.object_by_id("box").mobility == "dynamic"
    assert scene_ir.object_by_id("box").physics.confidence == "estimated"
    assert scene_ir.object_by_id("table").mobility == "static"


def test_destination_swept_path_clearance_rejects_obstacle_edge_route() -> None:
    occupied = [(0.0, 0.0, 0.25, 0.10)]

    near_edge = _swept_path_clearance((-0.50, 0.11), (0.22, 0.12), occupied, required=0.04)
    clear_route = _swept_path_clearance((-0.50, 0.28), (0.22, 0.28), occupied, required=0.04)

    assert near_edge < 0.0
    assert clear_route > 0.0


def test_scene_ir_rejects_unaccepted_run(tmp_path: Path) -> None:
    scene = SceneSpec(
        prompt="box",
        objects=[ObjectSpec(id="box", category="box", asset_id="proc_box_01", role="anchor")],
    )
    run_dir = _write_accepted_run(tmp_path, scene)
    write_json(run_dir / "qualification.json", {"accepted": False, "status": "unqualified"})

    with pytest.raises(RuntimeError, match="accepted=true"):
        build_scene_ir(run_dir, target_object="box")


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco is not installed")
def test_compile_scene_to_mjcf_and_step_smoke(tmp_path: Path) -> None:
    scene = SceneSpec(
        scene_id="mujoco_compile_fixture",
        prompt="box on table",
        objects=[
            ObjectSpec(id="table", category="table", asset_id="proc_table_01", role="anchor", placement=PlacementSpec(x=1.0, y=1.0, z=0.41)),
            ObjectSpec(id="box", category="box", asset_id="proc_box_01", role="child", parent_id="table", relation="on", placement=PlacementSpec(x=1.0, y=1.0, z=0.96)),
        ],
    )
    run_dir = _write_accepted_run(tmp_path, scene)
    scene_ir = build_scene_ir(run_dir, target_object="box")

    result = compile_scene_to_mjcf(scene_ir, tmp_path / "mujoco_eval", config={"rollout": {"timestep": 0.002}, "collision": {"use_coacd": False}})

    assert Path(result.xml_path).is_file()
    assert result.mjb_path is not None and Path(result.mjb_path).is_file()
    assert result.dynamic_object_count == 1

    import mujoco

    model = mujoco.MjModel.from_xml_path(result.xml_path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=2)
    assert data.time > 0


def test_shelf_uses_primitive_collision_not_raw_mesh(tmp_path: Path) -> None:
    scene = SceneSpec(
        scene_id="mujoco_shelf_fixture",
        prompt="shelf and box",
        objects=[
            ObjectSpec(id="shelf", category="shelf", asset_id="proc_shelf_01", role="anchor", placement=PlacementSpec(x=1.0, y=1.0, z=0.875)),
            ObjectSpec(id="box", category="box", asset_id="proc_box_01", role="child", parent_id="shelf", relation="on", placement=PlacementSpec(x=1.0, y=1.0, z=1.3)),
        ],
    )
    run_dir = _write_accepted_run(tmp_path, scene)
    scene_ir = build_scene_ir(run_dir, target_object="box")
    result = compile_scene_to_mjcf(scene_ir, tmp_path / "mujoco_eval", config={"rollout": {"timestep": 0.002}, "collision": {"use_coacd": False}})
    compiled_ir = Path(result.scene_ir_path).read_text(encoding="utf-8")

    assert '"id": "shelf"' in compiled_ir
    assert '"primitive_type": "box"' in compiled_ir
    assert '"mesh_name": null' in compiled_ir


def test_dynamic_scanner_uses_compound_primitive_proxy(tmp_path: Path) -> None:
    scene = SceneSpec(
        scene_id="mujoco_scanner_fixture",
        prompt="scanner on table",
        objects=[
            ObjectSpec(id="table", category="table", asset_id="proc_table_01", role="anchor", placement=PlacementSpec(x=1.0, y=1.0, z=0.41)),
            ObjectSpec(id="scanner", category="scanner", asset_id="hf_barcode_scanner_01", role="child", parent_id="table", relation="on", placement=PlacementSpec(x=1.0, y=1.0, z=0.9)),
        ],
    )
    run_dir = _write_accepted_run(tmp_path, scene)
    scene_ir = build_scene_ir(run_dir, config_path=_write_proxy_config(tmp_path), target_object="scanner")

    compiled_ir, _report = prepare_mesh_assets(scene_ir, tmp_path / "scanner_eval", config={"visual_scene": {"mode": "proxy"}, "collision": {"use_coacd": False}})
    scanner = compiled_ir.object_by_id("scanner")

    assert scanner.mobility == "dynamic"
    assert len(scanner.collision) >= 3
    assert {spec.kind for spec in scanner.collision} == {"primitive"}
    assert {spec.primitive_type for spec in scanner.collision} == {"box"}
    assert scanner.collision_meshes == []


def test_scanner_grasp_candidates_include_opening_axis_offsets() -> None:
    candidates = _grasp_frame_candidates(
        np.eye(3),
        np.asarray([0.0, -0.8, 0.0]),
        np.asarray([0.0, 0.0, 0.0]),
        category="scanner",
        dimensions=[0.12, 0.07, 0.04],
        gripper_max_width_m=0.08,
    )

    head_front = [
        item
        for item in candidates
        if item["base_grasp_name"] == "front_side_grasp"
        and item["grasp_target_label"] == "scanner_head_narrow_axis"
        and item["close_standoff_m"] == 0.0
    ]

    assert {round(float(item["opening_axis_offset_m"]), 3) for item in head_front} == {-0.006, -0.003, 0.0, 0.003, 0.006}
    assert "front_side_grasp_scanner_head_narrow_axis_center" in {item["name"] for item in head_front}
    assert "front_side_grasp_scanner_head_narrow_axis_center_open_neg006" in {item["name"] for item in head_front}
    zero_offset = next(item for item in head_front if float(item["opening_axis_offset_m"]) == 0.0)
    positive_offset = next(item for item in head_front if round(float(item["opening_axis_offset_m"]), 3) == 0.006)
    np.testing.assert_allclose(
        positive_offset["local_grasp_offset"] - zero_offset["local_grasp_offset"],
        positive_offset["frame"][:, 1] * 0.006,
    )
    body_center = next(
        item
        for item in candidates
        if item["base_grasp_name"] == "front_side_grasp"
        and item["grasp_target_label"] == "scanner_body_narrow_axis"
        and item["close_standoff_m"] == 0.0
        and float(item["opening_axis_offset_m"]) == 0.0
    )
    assert float(body_center["local_grasp_offset"][2]) > -0.005


def test_grasp_probe_score_prefers_balanced_low_support_lift() -> None:
    base_probe = {
        "feasible": False,
        "stable_grasp": True,
        "two_finger_contact": True,
        "left_contact": True,
        "right_contact": True,
        "lift_delta_z_m": 0.014,
        "scanner_table_contact_after_lift": True,
        "grasp_loss": False,
        "failure_reason": "micro_lift_failed",
    }
    unbalanced = {
        **base_probe,
        "phase_reports": [
            {"phase": "GRASP_SETTLE", "target_contact_force_summary": {"left_finger_n": 3.0, "right_finger_n": 0.2, "support_n": 3.5}},
            {"phase": "MICRO_LIFT", "target_contact_force_summary": {"left_finger_n": 0.12, "right_finger_n": 0.02, "support_n": 1.4}},
        ],
    }
    balanced = {
        **base_probe,
        "phase_reports": [
            {"phase": "GRASP_SETTLE", "target_contact_force_summary": {"left_finger_n": 1.7, "right_finger_n": 1.5, "support_n": 3.2}},
            {"phase": "MICRO_LIFT", "target_contact_force_summary": {"left_finger_n": 0.18, "right_finger_n": 0.16, "support_n": 0.5}},
        ],
    }

    assert _score_grasp_probe(balanced) > _score_grasp_probe(unbalanced)


def test_scripted_policy_requires_verified_contact_before_lift() -> None:
    policy = ScriptedPickPlacePolicy(PolicyContract())
    policy.reset(1)
    target = np.asarray([1.0, 1.0, 0.8])
    dest = np.asarray([1.25, 1.0, 0.8])

    def obs(ee: np.ndarray, two_finger: bool = False, verified: bool = False) -> dict:
        return {
            "state": {
                "ee_position": ee,
                "target_position": target,
                "destination_position": dest,
                "two_finger_contact": two_finger,
                "verified_grasp": verified,
            }
        }

    for _ in range(3):
        policy.act(obs(target + np.asarray([0.0, 0.0, 0.18])))
    for _ in range(3):
        policy.act(obs(target + np.asarray([0.0, 0.0, 0.12])))
    for _ in range(2):
        policy.act(obs(target + np.asarray([0.0, 0.0, 0.035])))
    assert policy.phase == "CLOSE_GRIPPER"

    for _ in range(10):
        close_action = policy.act(obs(target + np.asarray([0.0, 0.0, 0.035])))
    assert close_action[6] == -1.0
    assert policy.phase == "GRASP_VERIFY"

    for _ in range(3):
        policy.act(obs(target + np.asarray([0.0, 0.0, 0.04])))
    assert policy.phase == "GRASP_VERIFY"

    policy.act(obs(target + np.asarray([0.0, 0.0, 0.04]), two_finger=True, verified=True))
    assert policy.phase == "LIFT"


def test_teacher_policy_is_registered() -> None:
    policy = make_policy("teacher_pick_place", PolicyContract())
    assert isinstance(policy, TeacherPickPlacePolicy)
    assert policy.require_joint_plan is True

    debug_policy = make_policy("teacher_delta_debug", PolicyContract())
    assert isinstance(debug_policy, TeacherPickPlacePolicy)
    assert debug_policy.require_joint_plan is False


def test_teacher_policy_does_not_close_before_grasp_envelope() -> None:
    policy = TeacherPickPlacePolicy(PolicyContract(), require_joint_plan=False)
    policy.reset(1)
    target = np.asarray([1.0, 1.0, 0.8])
    dest = np.asarray([1.25, 1.0, 0.8])
    obs = {
        "state": {
            "ee_position": target + np.asarray([0.45, 0.45, 0.35]),
            "ee_xmat": np.eye(3),
            "target_position": target,
            "target_xmat": np.eye(3),
            "destination_position": dest,
            "gripper_width": 0.08,
            "two_finger_contact": False,
            "verified_grasp": False,
        }
    }

    action = policy.act(obs)

    assert action[6] == 1.0
    assert policy.phase == "PRE_GRASP_HIGH"


def test_teacher_policy_waits_for_verified_grasp_before_lift() -> None:
    policy = TeacherPickPlacePolicy(PolicyContract(), require_joint_plan=False)
    policy.reset(1)
    target = np.asarray([1.0, 1.0, 0.8])
    dest = np.asarray([1.25, 1.0, 0.8])
    aligned_xmat = np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )

    def obs(ee: np.ndarray, two_finger: bool = False, verified: bool = False, stable: bool = False, width: float = 0.08) -> dict:
        return {
            "state": {
                "ee_position": ee,
                "ee_xmat": aligned_xmat,
                "target_position": target,
                "target_xmat": np.eye(3),
                "destination_position": dest,
                "gripper_width": width,
                "two_finger_contact": two_finger,
                "verified_grasp": verified,
                "stable_grasp": stable,
            }
        }

    for _ in range(3):
        policy.act(obs(target + np.asarray([0.0, 0.0, 0.32])))
    for _ in range(3):
        policy.act(obs(target + np.asarray([0.0, 0.0, 0.24])))
    for _ in range(2):
        policy.act(obs(target + np.asarray([0.0, 0.0, 0.20])))
    policy.act(obs(target + np.asarray([0.0, 0.0, 0.16])))
    close_action = policy.act(obs(target + np.asarray([0.0, 0.0, 0.045])))
    assert close_action[6] == -1.0
    assert policy.phase == "CLOSE_GRIPPER"

    for _ in range(20):
        policy.act(obs(target + np.asarray([0.0, 0.0, 0.045]), width=0.06))
    assert policy.phase == "CLOSE_GRIPPER"

    for _ in range(3):
        policy.act(obs(target + np.asarray([0.0, 0.0, 0.045]), two_finger=True, verified=True, stable=False, width=0.06))
    assert policy.phase == "CLOSE_GRIPPER"

    policy.act(obs(target + np.asarray([0.0, 0.0, 0.045]), two_finger=True, verified=True, stable=True, width=0.06))
    assert policy.phase == "GRASP_SETTLE"


def test_strict_teacher_policy_requires_joint_waypoint_plan() -> None:
    policy = TeacherPickPlacePolicy(PolicyContract())
    policy.reset(1)
    with pytest.raises(TeacherPlanUnavailable):
        policy.act(
            {
                "state": {
                    "ee_position": np.asarray([0.0, 0.0, 0.0]),
                    "target_position": np.asarray([1.0, 1.0, 0.8]),
                    "destination_position": np.asarray([1.2, 1.0, 0.8]),
                    "teacher_plan": {"ok": False, "failed_phase": "PRE_CLOSE", "reason": "ik_failed_pre_close"},
                }
            }
        )


def test_joint_position_teacher_executes_waypoints_before_representation_guard() -> None:
    policy = TeacherPickPlacePolicy(PolicyContract(action_representation="joint_position"))
    policy.reset(1)
    waypoint = {"phase": "HOME", "qpos": [0.14, -0.14, 0.10, -0.10, 0.08, 0.12, -0.08, 0.02]}

    action = policy.act(
        {
            "proprio": np.zeros(15, dtype=float),
            "state": {
                "target_position": np.asarray([1.0, 1.0, 0.8]),
                "teacher_joint_waypoints": [waypoint],
            },
        }
    )

    assert np.linalg.norm(action[:7]) > 0.0
    assert action[-1] == pytest.approx(0.02)


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco is not installed")
def test_full_glb_visual_replaces_dynamic_target(tmp_path: Path) -> None:
    scene = SceneSpec(
        scene_id="mujoco_visual_fixture",
        prompt="box in workspace",
        bounds=[3.0, 3.0, 1.0],
        objects=[
            ObjectSpec(id="anchor", category="table", asset_id="proc_table_01", role="anchor", placement=PlacementSpec(x=0.35, y=0.35, z=0.41)),
            ObjectSpec(id="box", category="box", asset_id="proc_box_01", role="parent", placement=PlacementSpec(x=1.5, y=1.5, z=0.2)),
        ],
    )
    run_dir = _write_accepted_run(tmp_path, scene)
    _write_tiny_semantic_glb(run_dir / "scene.glb")
    scene_ir = build_scene_ir(run_dir, config_path=_write_visual_config(tmp_path), target_object="box")

    result = compile_scene_to_mjcf(scene_ir, tmp_path / "visual_eval", config=load_mujoco_config(_write_visual_config(tmp_path)))
    manifest = (tmp_path / "visual_eval" / "entity_manifest.json").read_text(encoding="utf-8")

    assert Path(result.xml_path).is_file()
    assert '"visible_target_instances": 1' in manifest
    assert '"dynamic_target_bodies": 1' in manifest
    assert '"static_target_mesh_instances": 0' in manifest
    assert '"box_0_cardboard_box"' in manifest


@pytest.mark.skipif(importlib.util.find_spec("mujoco") is None, reason="mujoco is not installed")
def test_evaluator_compile_only_writes_report(tmp_path: Path) -> None:
    scene = SceneSpec(
        scene_id="mujoco_eval_fixture",
        prompt="box on table",
        objects=[
            ObjectSpec(id="table", category="table", asset_id="proc_table_01", role="anchor", placement=PlacementSpec(x=1.0, y=1.0, z=0.41)),
            ObjectSpec(id="box", category="box", asset_id="proc_box_01", role="child", parent_id="table", relation="on", placement=PlacementSpec(x=1.0, y=1.0, z=0.96)),
        ],
    )
    run_dir = _write_accepted_run(tmp_path, scene)

    report = evaluate_scene(
        run_dir,
        tmp_path / "eval",
        config_path=_write_proxy_config(tmp_path),
        target_object="box",
        episodes=1,
        policy_id="noop",
        compile_only=True,
    )

    assert report.episodes == []
    assert Path(report.compile_artifacts["xml"]).is_file()
    assert Path(report.compile_artifacts["coordinate_manifest"]).is_file()
    assert Path(report.compile_artifacts["physics_settling_report"]).is_file()
    assert "physics_settling_success" in report.outcome
    assert (tmp_path / "eval" / "evaluation_report.json").is_file()
    assert read_json(tmp_path / "eval" / "coordinate_manifest.json")["canonical_world"] == "Z_UP_METERS"


def test_task_feasibility_rejects_non_graspable_target(tmp_path: Path) -> None:
    scene = SceneSpec(
        scene_id="mujoco_feasibility_fixture",
        prompt="large box on table",
        objects=[
            ObjectSpec(id="table", category="table", asset_id="proc_table_01", role="anchor", placement=PlacementSpec(x=1.0, y=1.0, z=0.41)),
            ObjectSpec(id="box", category="box", asset_id="proc_box_01", role="child", parent_id="table", relation="on", placement=PlacementSpec(x=1.0, y=1.0, z=0.96)),
        ],
    )
    run_dir = _write_accepted_run(tmp_path, scene)
    scene_ir = build_scene_ir(run_dir, target_object="box")

    with pytest.raises(RuntimeError, match="not feasible"):
        validate_task_feasibility(scene_ir, {"robot": {"gripper_max_width_m": 0.08, "reach_m": 0.95}, "task": {"require_graspable_target": True}})
