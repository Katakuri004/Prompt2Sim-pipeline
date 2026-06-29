from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.mujoco_bridge.coordinate_manifest import write_coordinate_manifest
from scenethesis_mvp.mujoco_bridge.mjcf_emitter import compile_scene_to_mjcf
from scenethesis_mvp.mujoco_bridge.mujoco_env import MujocoSceneEnv
from scenethesis_mvp.mujoco_bridge.physics_settling import validate_physics_settling
from scenethesis_mvp.mujoco_bridge.policies import TeacherPlanUnavailable, make_policy
from scenethesis_mvp.mujoco_bridge.replay import EpisodeRecorder
from scenethesis_mvp.mujoco_bridge.runtime import raise_if_unavailable, validate_mujoco_runtime
from scenethesis_mvp.mujoco_bridge.scene_ir import build_scene_ir, load_mujoco_config
from scenethesis_mvp.mujoco_bridge.schemas import EvaluationReport, EpisodeResult, SceneIR
from scenethesis_mvp.mujoco_bridge.task_validation import validate_compiled_task_feasibility, validate_task_feasibility
from scenethesis_mvp.mujoco_bridge.visual_identity import write_visual_identity_report
from scenethesis_mvp.mujoco_bridge.visual_twin import render_blender_visual_twin
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import read_json, write_json
from scenethesis_mvp.utils.paths import project_root, resolve_path


def evaluate_scene(
    run_dir: str | Path,
    out_dir: str | Path,
    config_path: str | Path = "configs/mujoco_eval.yaml",
    target_object: str | None = None,
    episodes: int = 5,
    policy_id: str = "scripted_pick_place",
    policy_path: str | Path | None = None,
    compile_only: bool = False,
    render_rgb: bool = False,
    save_video: bool = False,
    video_camera: str | None = None,
    strict_task: bool = True,
    visual_renderer: str = "both",
    blender_path: str | None = None,
    debug_teacher_plan: bool = False,
    run_grasp_probe: bool = False,
) -> EvaluationReport:
    root = project_root()
    config = load_mujoco_config(config_path)
    target_run = Path(run_dir)
    if not target_run.is_absolute():
        target_run = root / target_run
    target_out = Path(out_dir)
    if not target_out.is_absolute():
        target_out = root / target_out
    _clear_previous_outputs(target_out)
    registry = AssetRegistry.from_yaml(resolve_path(config.get("paths", {}).get("asset_registry", "configs/warehouse_asset_registry.yaml"), root))
    scene = SceneSpec.model_validate(read_json(target_run / "scene_spec.json")) if (target_run / "scene_spec.json").is_file() else None
    runtime = validate_mujoco_runtime(config, root, run_dir=target_run, out_dir=target_out, registry=registry, scene=scene)
    raise_if_unavailable(runtime)
    write_json(target_out / "mujoco_runtime_report.json", runtime)

    scene_ir = build_scene_ir(target_run, config_path=config_path, target_object=target_object)
    feasibility = (
        validate_task_feasibility(scene_ir, config, raise_on_failure=False)
        if strict_task
        else {"ok": None, "skipped": True}
    )
    write_json(target_out / "task_feasibility_report.json", feasibility)
    if strict_task and not compile_only and feasibility.get("ok") is False and bool(config.get("task", {}).get("require_graspable_target", True)):
        failed = [str(check.get("detail")) for check in feasibility.get("checks", []) if isinstance(check, dict) and not check.get("ok")]
        raise RuntimeError("Task is not feasible for the configured Panda pick-place policy: " + "; ".join(failed))
    compile_result = compile_scene_to_mjcf(scene_ir, target_out, config)
    compiled_scene_ir = SceneIR.model_validate(read_json(compile_result.scene_ir_path))
    coordinate_manifest = write_coordinate_manifest(compiled_scene_ir, target_out)
    artifacts = {
        "scene_ir": compile_result.scene_ir_path,
        "xml": compile_result.xml_path,
        "mesh_dir": compile_result.mesh_dir,
        "compile_report": compile_result.compile_report_path,
    }
    for optional_name in ("visual_scene_report", "entity_manifest", "camera_manifest"):
        optional_path = target_out / f"{optional_name}.json"
        if optional_path.is_file():
            artifacts[optional_name] = str(optional_path)
    artifacts["coordinate_manifest"] = str(target_out / "coordinate_manifest.json")
    if compile_result.mjb_path:
        artifacts["mjb"] = compile_result.mjb_path

    report = EvaluationReport(scene_id=compiled_scene_ir.scene_id, policy_id=policy_id, compile_artifacts=artifacts)
    report.import_success = True
    model_path = compile_result.mjb_path or compile_result.xml_path
    compiled_feasibility = validate_compiled_task_feasibility(
        compiled_scene_ir,
        config,
        model_path=model_path,
        raise_on_failure=False,
    )
    report.task_feasibility = compiled_feasibility
    report.task_feasibility_success = bool(compiled_feasibility.get("ok", False))
    settling_report = validate_physics_settling(compiled_scene_ir, model_path, target_out, config)
    report.physics_settling = settling_report
    report.compile_artifacts["physics_settling_report"] = str(target_out / "physics_settling_report.json")
    write_json(target_out / "task_feasibility_report.json", compiled_feasibility)
    if compile_only:
        report.outcome = {
            "import_success": report.import_success,
            "task_feasibility_success": report.task_feasibility_success,
            "policy_success": False,
            "policy_ran": False,
            "physics_settling_success": bool(settling_report.get("ok", False)),
        }
        write_json(target_out / "evaluation_report.json", report)
        return report
    if strict_task and not report.task_feasibility_success and bool(config.get("task", {}).get("require_graspable_target", True)):
        reasons = [str(item) for item in compiled_feasibility.get("failure_reasons", [])]
        raise RuntimeError("Task is not feasible for the configured Panda pick-place policy: " + "; ".join(reasons))

    env = MujocoSceneEnv(
        model_path=model_path,
        scene_ir=compiled_scene_ir,
        physics_steps_per_action=int(config.get("rollout", {}).get("physics_steps_per_action", 50)),
        render_rgb=render_rgb,
    )
    max_steps = int(config.get("rollout", {}).get("max_steps", 250))
    seed_base = int(config.get("rollout", {}).get("seed", 7))
    viz_cfg = config.get("visualization", {})
    camera_id = video_camera or str(viz_cfg.get("camera", "report_task_closeup"))
    resolution = tuple(int(item) for item in viz_cfg.get("resolution", [960, 720]))
    fps = int(viz_cfg.get("fps", 20))
    frame_stride = int(viz_cfg.get("frame_stride", 2))
    save_debug_video = save_video or visual_renderer in {"both", "mujoco_debug"}
    try:
        for episode_index in range(max(0, episodes)):
            seed = seed_base + episode_index
            policy = make_policy(policy_id, compiled_scene_ir.policy, policy_path=policy_path)
            policy.reset(seed)
            observation = env.reset(seed)
            if episode_index == 0 and (debug_teacher_plan or policy_id == "teacher_pick_place"):
                frame_audit_path = target_out / "coordinate_frame_audit.json"
                teacher_search_path = target_out / "teacher_plan_search.json"
                teacher_plan_path = target_out / "teacher_plan.json"
                teacher_diagnostics_path = target_out / "teacher_waypoint_diagnostics.json"
                grasp_probe_search_path = target_out / "grasp_probe_search.json"
                write_json(frame_audit_path, env.coordinate_frame_audit_report())
                write_json(teacher_search_path, env.teacher_plan_search_report())
                write_json(teacher_plan_path, env.teacher_plan_report())
                write_json(teacher_diagnostics_path, env.teacher_waypoint_diagnostics())
                write_json(grasp_probe_search_path, env.grasp_probe_search_report())
                report.compile_artifacts["coordinate_frame_audit"] = str(frame_audit_path)
                report.compile_artifacts["teacher_plan_search"] = str(teacher_search_path)
                report.compile_artifacts["teacher_plan"] = str(teacher_plan_path)
                report.compile_artifacts["teacher_waypoint_diagnostics"] = str(teacher_diagnostics_path)
                report.compile_artifacts["grasp_probe_search"] = str(grasp_probe_search_path)
                if env.teacher_candidate_report():
                    candidates_path = target_out / "grasp_candidates.json"
                    write_json(candidates_path, env.teacher_candidate_report())
                    report.compile_artifacts["grasp_candidates"] = str(candidates_path)
            if episode_index == 0 and run_grasp_probe:
                probe_report = env.run_grasp_probe(target_out, camera_id=camera_id)
                probe_path = target_out / "grasp_probe.json"
                write_json(probe_path, probe_report)
                report.compile_artifacts["grasp_probe"] = str(probe_path)
                observation = env.get_observation()
            recorder = EpisodeRecorder(
                target_out,
                episode_index=episode_index,
                enabled=save_debug_video,
                camera_id=camera_id,
                resolution=(resolution[0], resolution[1]),
                fps=fps,
                frame_stride=frame_stride,
            )
            initial_action_size = len(compiled_scene_ir.robot.actuator_names) if compiled_scene_ir.policy.action_representation == "joint_position" else 7
            initial_metrics = env.compute_metrics()
            initial_metrics["success"] = False
            initial_metrics["terminated_reason"] = "initial"
            recorder.record_step(0, [0.0] * initial_action_size, initial_metrics, observation, env=env)
            recorder.maybe_capture(env, step=0)
            final_metrics: dict[str, Any] = {}
            terminated = False
            steps = 0
            if policy_id == "teacher_pick_place" and not env.teacher_plan_ok:
                final_metrics = env.compute_metrics()
                final_metrics["success"] = False
                final_metrics["terminated_reason"] = "teacher_plan_unavailable"
                final_metrics["teacher_plan"] = env.teacher_plan_report()
                terminated = True
            elif run_grasp_probe and policy_id == "teacher_pick_place":
                probe = read_json(target_out / "grasp_probe.json") if (target_out / "grasp_probe.json").is_file() else {}
                if probe.get("feasible") is False:
                    final_metrics = env.compute_metrics()
                    final_metrics["success"] = False
                    final_metrics["terminated_reason"] = str(probe.get("failure_reason") or "grasp_probe_failed")
                    terminated = True
            if not terminated:
                for step_index in range(max_steps):
                    try:
                        action = policy.act(observation)
                    except TeacherPlanUnavailable:
                        final_metrics = env.compute_metrics()
                        final_metrics["success"] = False
                        final_metrics["terminated_reason"] = "teacher_plan_unavailable"
                        final_metrics["teacher_plan"] = env.teacher_plan_report()
                        terminated = True
                        steps = step_index
                        break
                    observation, final_metrics, terminated = env.step(action)
                    steps = step_index + 1
                    recorder.record_step(steps, action, final_metrics, observation, env=env)
                    recorder.maybe_capture(env, step=steps)
                    if terminated:
                        break
            if not final_metrics:
                final_metrics = env.compute_metrics()
                final_metrics["success"] = env.success_oracle()
                final_metrics["terminated_reason"] = "max_steps"
            success = bool(final_metrics.get("success", False))
            reason = str(final_metrics.get("terminated_reason", "max_steps" if not terminated else "terminated"))
            if not terminated and not success:
                reason = _policy_timeout_reason(final_metrics)
                final_metrics["terminated_reason"] = reason
            episode_artifacts = recorder.write()
            if episode_index == 0:
                identity = write_visual_identity_report(
                    target_out,
                    compiled_scene_ir,
                    env.model,
                    env.mujoco,
                    snapshot_path=episode_artifacts.get("snapshot_path"),
                )
                if (target_out / "visual_identity_report.json").is_file():
                    report.compile_artifacts["visual_identity_report"] = str(target_out / "visual_identity_report.json")
                if visual_renderer in {"both", "blender"}:
                    visual_twin_report = render_blender_visual_twin(
                        compiled_scene_ir,
                        target_out,
                        state_trace_path=episode_artifacts["state_trace_path"],
                        config=config,
                        blender_path=blender_path,
                    )
                    report.visual_twin = visual_twin_report
                    report.compile_artifacts["visual_twin_report"] = str(target_out / "visual_twin_report.json")
                    if visual_twin_report.get("benchmark_visual_artifact"):
                        report.visualization_artifacts.append(str(visual_twin_report["benchmark_visual_artifact"]))
            for artifact in episode_artifacts.values():
                report.visualization_artifacts.append(artifact)
            report.episodes.append(
                EpisodeResult(
                    episode=episode_index,
                    seed=seed,
                    success=success,
                    steps=steps,
                    terminated_reason=reason,
                    max_contact_force=float(final_metrics.get("max_contact_force", 0.0)),
                    collision_count=int(final_metrics.get("collision_count", 0)),
                    workspace_violation=bool(final_metrics.get("workspace_violation", False)),
                    object_drop=bool(final_metrics.get("object_drop", False)),
                    recovery_success=False,
                    grasp_attempted=bool(final_metrics.get("grasp_attempted", False)),
                    released_after_grasp=bool(final_metrics.get("released_after_grasp", False)),
                    target_lifted=bool(final_metrics.get("target_lifted", False)),
                    target_placed=bool(final_metrics.get("target_placed", False)),
                    final_target_distance_m=float(final_metrics.get("target_distance_m")) if final_metrics.get("target_distance_m") is not None else None,
                    trace_path=episode_artifacts.get("state_trace_path") or episode_artifacts.get("trace_path"),
                    video_path=episode_artifacts.get("video_path"),
                    snapshot_path=episode_artifacts.get("snapshot_path"),
                    time_to_completion_s=(steps * float(config.get("rollout", {}).get("physics_steps_per_action", 50)) * float(config.get("rollout", {}).get("timestep", 0.002))) if success else None,
                )
            )
    finally:
        env.close()
    _finalize_report(report)
    report.policy_success = report.success_rate > 0.0
    report.outcome = {
        "import_success": report.import_success,
        "task_feasibility_success": report.task_feasibility_success,
        "policy_success": report.policy_success,
        "policy_ran": True,
        "success_rate": report.success_rate,
    }
    write_json(target_out / "evaluation_report.json", report)
    return report


def _finalize_report(report: EvaluationReport) -> None:
    if not report.episodes:
        return
    total = len(report.episodes)
    report.success_rate = sum(1 for item in report.episodes if item.success) / total
    report.collision_rate = sum(1 for item in report.episodes if item.collision_count > 0) / total
    report.object_drop_rate = sum(1 for item in report.episodes if item.object_drop) / total
    report.workspace_violation_rate = sum(1 for item in report.episodes if item.workspace_violation) / total
    report.max_contact_force = max((item.max_contact_force for item in report.episodes), default=0.0)


def _policy_timeout_reason(metrics: dict[str, Any]) -> str:
    if bool(metrics.get("workspace_violation", False)):
        return "workspace_violation"
    if bool(metrics.get("object_drop", False)):
        return "object_drop"
    if bool(metrics.get("grasp_lost", False)):
        return "grasp_lost"
    if not bool(metrics.get("grasp_attempted", False)):
        return "grasp_not_attempted"
    if not bool(metrics.get("stable_grasp", metrics.get("verified_grasp", False))):
        return "grasp_failure"
    if not bool(metrics.get("target_lifted", False)):
        return "lift_failure"
    if not bool(metrics.get("target_placed", False)):
        return "placement_failure"
    if not bool(metrics.get("released_after_grasp", False)):
        return "release_failure"
    return "stability_timeout"


def _clear_previous_outputs(target_out: Path) -> None:
    target_out.mkdir(parents=True, exist_ok=True)
    file_names = {
        "mujoco_runtime_report.json",
        "task_feasibility_report.json",
        "evaluation_report.json",
        "compile_report.json",
        "mesh_compile_report.json",
        "mujoco_scene_ir.json",
        "visual_scene_report.json",
        "entity_manifest.json",
        "camera_manifest.json",
        "visual_identity_report.json",
        "visual_twin_report.json",
        "visual_twin_payload.json",
        "physics_settling_report.json",
        "coordinate_manifest.json",
        "rollout_state_trace.json",
        "coordinate_frame_audit.json",
        "teacher_plan_search.json",
        "teacher_plan.json",
        "teacher_waypoint_diagnostics.json",
        "grasp_candidates.json",
        "grasp_probe_search.json",
        "grasp_probe.json",
        "scene.xml",
        "scene.mjb",
    }
    dir_names = {"meshes", "robots", "episodes", "visual_twin_frames", "phase_snapshots"}
    for name in file_names:
        path = target_out / name
        if path.is_file():
            path.unlink()
    for name in dir_names:
        path = target_out / name
        if path.is_dir():
            shutil.rmtree(path)
