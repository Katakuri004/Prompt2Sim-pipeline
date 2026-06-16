from __future__ import annotations

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenethesis_mvp.assets.clip_index import ClipAssetRetriever, ClipIndexConfig
from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.optimizer import spread_children
from scenethesis_mvp.layout.relation_rules import normalize_support_relation_semantics, place_relative_to_target
from scenethesis_mvp.layout.warehouse_staging import stage_warehouse_presentation_layout
from scenethesis_mvp.llm.judge import SceneJudge
from scenethesis_mvp.llm.planner import ScenePlanner
from scenethesis_mvp.llm.repair import RepairEngine
from scenethesis_mvp.optimization.roma_correspondence import run_roma_correspondence_refinement
from scenethesis_mvp.optimization.sdf_optimizer import SDFOptimizerConfig, SDFPhysicsOptimizer
from scenethesis_mvp.pipeline.qualification import build_failure_qualification, build_success_qualification, write_qualification
from scenethesis_mvp.pipeline.diagnostics import write_pipeline_diagnostics
from scenethesis_mvp.render.blender_runner import RenderResult, render_scene
from scenethesis_mvp.runtime.faithful import raise_if_unavailable, validate_faithful_runtime
from scenethesis_mvp.schemas.depth import DepthResult
from scenethesis_mvp.schemas.metrics import Metrics, StabilityRecord
from scenethesis_mvp.schemas.scene_graph_3d import SceneGraph3D
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.schemas.segmentation import SegmentationResult
from scenethesis_mvp.utils.io import read_json, read_yaml, write_json, write_text
from scenethesis_mvp.utils.paths import project_root, resolve_path
from scenethesis_mvp.utils.seeds import seed_everything
from scenethesis_mvp.vision.depth_pro_runner import DepthProConfig, DepthProRunner
from scenethesis_mvp.vision.depth_pose_refinement import apply_depth_pose_refinement
from scenethesis_mvp.vision.grounded_sam import GroundedSAMConfig, GroundedSAMSegmenter
from scenethesis_mvp.vision.image_guidance import ImageGuidanceGenerator, ImageGuidanceResult
from scenethesis_mvp.vision.pointcloud import build_pointcloud_scene_graph


@dataclass(frozen=True)
class FaithfulPipelineResult:
    out_dir: Path
    scene: SceneSpec
    graph: SceneGraph3D
    render: RenderResult
    metrics: Metrics
    judge: dict[str, Any]
    repair_history: list[dict[str, Any]]


def run_faithful_pipeline(
    prompt: str,
    out_dir: str | Path,
    config_path: str | Path = "configs/scenethesis_faithful.yaml",
    repair_rounds: int | None = None,
    resume_from_existing: bool = False,
) -> FaithfulPipelineResult:
    root = project_root()
    config_file = resolve_path(config_path, root)
    config = read_yaml(config_file)
    seed_everything(int(config.get("seed", 7)))
    target_dir = Path(out_dir)
    if not target_dir.is_absolute():
        target_dir = root / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    stale_failure = target_dir / "failure.json"
    if stale_failure.exists():
        stale_failure.unlink()

    stage = "startup"
    try:
        stage = "runtime_validation"
        paths = config.get("paths", {})
        registry = AssetRegistry.from_yaml(resolve_path(paths.get("asset_registry", "configs/warehouse_asset_registry.yaml"), root))
        runtime_report = validate_faithful_runtime(config, root, registry=registry)
        write_json(target_dir / "faithful_runtime_report.json", runtime_report)
        raise_if_unavailable(runtime_report)

        openai_cfg = config.get("openai", {})
        scene_cfg = config.get("scene", {})
        image_cfg = config.get("image_guidance", {})
        segmentation_cfg = config.get("segmentation", {})
        depth_cfg = config.get("depth", {})
        pose_cfg = config.get("pose_extraction", {})
        depth_pose_cfg = config.get("depth_pose_refinement", {})
        retrieval_cfg = config.get("asset_retrieval", {})
        physics_cfg = config.get("physics", {})
        correspondence_cfg = config.get("correspondence", {})
        render_cfg = config.get("render", {})
        repair_limit = int(config.get("repair", {}).get("rounds", 2)) if repair_rounds is None else repair_rounds

        if resume_from_existing:
            stage = "resume_validation"
            scene, guidance, segmentation, depth = load_existing_faithful_artifacts(target_dir)
            validate_resume_artifacts(scene, guidance, segmentation, depth)
            scene = normalize_support_relation_semantics(scene, registry)
        else:
            stage = "llm_planning"
            planner = ScenePlanner(
                model=openai_cfg.get("model", "gpt-4o-mini"),
                system_prompt_path=resolve_path(paths.get("planner_prompt", "configs/prompts/planner_system.txt"), root),
                max_retries=int(openai_cfg.get("max_retries", 3)),
            )
            scene = planner.plan(prompt, registry, tuple(scene_cfg.get("bounds", [8.0, 7.0, 3.2])))  # type: ignore[arg-type]
            scene = normalize_support_relation_semantics(scene, registry)
            write_json(target_dir / "coarse_scene_spec.json", scene)

            stage = "image_guidance"
            guidance = ImageGuidanceGenerator(
                image_model=openai_cfg.get("image_model", "gpt-image-1"),
                max_retries=int(openai_cfg.get("max_retries", 3)),
            ).run(
                prompt=prompt,
                scene=scene,
                registry=registry,
                out_dir=target_dir,
                image_size=str(image_cfg.get("image_size", "1024x1024")),
                image_quality=str(image_cfg.get("image_quality", "low")),
            )

            stage = "segmentation"
            segmentation = GroundedSAMSegmenter(
                GroundedSAMConfig(
                    grounding_dino_config=resolve_path(segmentation_cfg["grounding_dino_config"], root),
                    grounding_dino_checkpoint=resolve_path(segmentation_cfg["grounding_dino_checkpoint"], root),
                    sam_checkpoint=resolve_path(segmentation_cfg["sam_checkpoint"], root),
                    sam_model_type=str(segmentation_cfg.get("sam_model_type", "vit_h")),
                    box_threshold=float(segmentation_cfg.get("box_threshold", 0.30)),
                    text_threshold=float(segmentation_cfg.get("text_threshold", 0.25)),
                    device=str(segmentation_cfg.get("device", "cuda")),
                    min_mask_pixels=int(pose_cfg.get("min_mask_pixels", 128)),
                )
            ).segment(guidance.guidance_path, scene, target_dir)

            stage = "depth_estimation"
            depth = DepthProRunner(
                DepthProConfig(
                    repo_dir=resolve_path(depth_cfg["repo_dir"], root),
                    checkpoint_dir=resolve_path(depth_cfg["checkpoint_dir"], root),
                    device=str(depth_cfg.get("device", "cuda")),
                )
            ).estimate(guidance.guidance_path, target_dir)

        stage = "scene_graph_3d"
        graph = build_pointcloud_scene_graph(
            segmentation=segmentation,
            depth=depth,
            out_dir=target_dir,
            max_points_per_object=int(pose_cfg.get("max_points_per_object", 5000)),
            min_mask_pixels=int(pose_cfg.get("min_mask_pixels", 128)),
        )

        stage = "asset_retrieval"
        scene = ClipAssetRetriever(
            ClipIndexConfig(
                index_path=resolve_path(retrieval_cfg["index_path"], root),
                device=str(retrieval_cfg.get("device", "cuda")),
                min_score=float(retrieval_cfg.get("min_score", 0.18)),
                text_weight=float(retrieval_cfg.get("text_weight", 0.20)),
                metadata_weight=float(retrieval_cfg.get("metadata_weight", 0.35)),
            )
        ).retrieve(scene, segmentation, registry, target_dir)
        scene = stage_warehouse_presentation_layout(scene, registry)
        stage = "depth_pose_refinement"
        scene, _depth_pose_report = apply_depth_pose_refinement(
            scene,
            graph,
            registry,
            target_dir,
            depth_pose_cfg,
        )
        write_json(target_dir / "presentation_layout.json", scene)
        scene_runtime_report = validate_faithful_runtime(config, root, registry=registry, scene=scene, check_disk=False)
        write_json(target_dir / "faithful_scene_runtime_report.json", scene_runtime_report)
        raise_if_unavailable(scene_runtime_report)

        stage = "sdf_optimization"
        scene = run_sdf_optimizer(scene, graph, registry, target_dir, physics_cfg)

        stage = "render"
        resolution = tuple(render_cfg.get("resolution", [1200, 900]))
        render_environment = resolve_render_environment(render_cfg.get("environment", {}), root)
        render_result = render_scene(
            scene,
            registry,
            target_dir,
            resolution=resolution,  # type: ignore[arg-type]
            blender_path=render_cfg.get("blender_path"),
            environment=render_environment,
        )
        if correspondence_cfg.get("enabled", False):
            stage = "roma_correspondence"
            scene, correspondence_report = run_roma_correspondence_refinement(
                scene,
                segmentation,
                registry,
                target_dir,
                correspondence_cfg,
                root,
            )
            if int(correspondence_report.get("applied_updates", 0)) > 0:
                stage = "roma_refined_sdf_optimization"
                scene = run_sdf_optimizer(scene, graph, registry, target_dir, physics_cfg)
                stage = "roma_refined_render"
                render_result = render_scene(
                    scene,
                    registry,
                    target_dir,
                    resolution=resolution,  # type: ignore[arg-type]
                    blender_path=render_cfg.get("blender_path"),
                    environment=render_environment,
                )
        stage = "metrics"
        metrics = metrics_from_sdf_report(scene, target_dir / "sdf_optimizer.json", float(physics_cfg.get("support_tolerance_m", 0.08)))
        stage = "judge"
        judge = SceneJudge(
            model=openai_cfg.get("vision_model", "gpt-4o-mini"),
            system_prompt_path=resolve_path(paths.get("judge_prompt", "configs/prompts/judge_system.txt"), root),
            max_retries=int(openai_cfg.get("max_retries", 3)),
        ).judge(prompt, scene, render_result.render_path, metrics, extra_image_paths=collect_judge_image_paths(target_dir, guidance.guidance_path))
        repair_history: list[dict[str, Any]] = []
        repair_engine = RepairEngine()
        for round_index in range(max(0, repair_limit)):
            if not judge.get("needs_repair"):
                break
            actions = list(judge.get("repair_actions", []))
            repair_record: dict[str, Any] = {
                "round": round_index + 1,
                "actions": actions,
                "pre_metrics": metrics.model_dump(mode="json"),
                "pre_judge": judge,
            }
            scene = repair_engine.apply(scene, actions)
            scene = normalize_support_relation_semantics(scene, registry)
            apply_faithful_repair_placements(scene, registry, actions)
            scene = stage_warehouse_presentation_layout(scene, registry)
            stage = f"repair_{round_index + 1}_depth_pose_refinement"
            scene, depth_pose_report = apply_depth_pose_refinement(
                scene,
                graph,
                registry,
                target_dir,
                depth_pose_cfg,
                artifact_name=f"repair_{round_index + 1}_depth_pose_refinement.json",
            )
            repair_record["depth_pose_refinement"] = depth_pose_report
            stage = f"repair_{round_index + 1}_sdf_optimization"
            scene = run_sdf_optimizer(scene, graph, registry, target_dir, physics_cfg)
            stage = f"repair_{round_index + 1}_render"
            render_result = render_scene(
                scene,
                registry,
                target_dir,
                resolution=resolution,  # type: ignore[arg-type]
                blender_path=render_cfg.get("blender_path"),
                environment=render_environment,
            )
            stage = f"repair_{round_index + 1}_judge"
            metrics = metrics_from_sdf_report(scene, target_dir / "sdf_optimizer.json", float(physics_cfg.get("support_tolerance_m", 0.08)))
            judge = SceneJudge(
                model=openai_cfg.get("vision_model", "gpt-4o-mini"),
                system_prompt_path=resolve_path(paths.get("judge_prompt", "configs/prompts/judge_system.txt"), root),
                max_retries=int(openai_cfg.get("max_retries", 3)),
            ).judge(prompt, scene, render_result.render_path, metrics, extra_image_paths=collect_judge_image_paths(target_dir, guidance.guidance_path))
            repair_record["post_metrics"] = metrics.model_dump(mode="json")
            repair_record["post_judge"] = judge
            repair_history.append(repair_record)
        stage = "final_artifacts"
        write_json(target_dir / "scene_spec.json", scene)
        write_json(target_dir / "metrics.json", metrics)
        write_json(target_dir / "judge.json", judge)
        write_json(target_dir / "repair_history.json", repair_history)
        write_pipeline_diagnostics(scene, metrics, judge, target_dir)
        qualification = build_success_qualification(scene, metrics, judge, target_dir)
        write_qualification(target_dir, qualification)
        write_text(target_dir / "report.md", build_faithful_report(prompt, scene, metrics, judge, render_result, target_dir, repair_history))
        return FaithfulPipelineResult(
            out_dir=target_dir,
            scene=scene,
            graph=graph,
            render=render_result,
            metrics=metrics,
            judge=judge,
            repair_history=repair_history,
        )
    except Exception as exc:
        write_qualification(target_dir, build_failure_qualification(stage, exc))
        write_json(
            target_dir / "failure.json",
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


def load_existing_faithful_artifacts(target_dir: str | Path) -> tuple[SceneSpec, ImageGuidanceResult, SegmentationResult, DepthResult]:
    run_dir = Path(target_dir)
    scene = SceneSpec.model_validate(read_json(require_resume_file(run_dir, "coarse_scene_spec.json")))
    guidance_path = require_resume_file(run_dir, "guidance.png")
    guidance = ImageGuidanceResult(
        guidance_path=guidance_path,
        image_metadata=read_json(require_resume_file(run_dir, "guidance_image.json")),
        upsampled_prompt=require_resume_file(run_dir, "upsampled_prompt.txt").read_text(encoding="utf-8"),
        candidates=read_json(require_resume_file(run_dir, "retrieval_candidates.json")),
    )
    segmentation = SegmentationResult.model_validate(read_json(require_resume_file(run_dir, "segmentation.json")))
    depth = DepthResult.model_validate(read_json(require_resume_file(run_dir, "depth.json")))
    return scene, guidance, segmentation, depth


def require_resume_file(run_dir: Path, relative_path: str) -> Path:
    path = run_dir / relative_path
    if not path.is_file():
        raise RuntimeError(f"Cannot resume faithful pipeline; required artifact is missing: {path}")
    return path


def validate_resume_artifacts(
    scene: SceneSpec,
    guidance: ImageGuidanceResult,
    segmentation: SegmentationResult,
    depth: DepthResult,
) -> None:
    guidance_path = guidance.guidance_path.resolve()
    if not guidance_path.is_file():
        raise RuntimeError(f"Cannot resume faithful pipeline; guidance image is missing: {guidance_path}")

    scene_ids = {obj.id for obj in scene.objects}
    detected_ids = {item.object_id for item in segmentation.detections if item.object_id}
    missing_ids = sorted(scene_ids - detected_ids)
    extra_ids = sorted(detected_ids - scene_ids)
    if segmentation.missing_object_ids:
        missing_ids = sorted(set(missing_ids) | set(segmentation.missing_object_ids))
    if missing_ids:
        raise RuntimeError("Cannot resume faithful pipeline; segmentation is missing objects: " + ", ".join(missing_ids))
    if extra_ids:
        raise RuntimeError("Cannot resume faithful pipeline; segmentation has objects not present in scene: " + ", ".join(extra_ids))
    if not segmentation.detections:
        raise RuntimeError("Cannot resume faithful pipeline; segmentation has no detections.")
    for detection in segmentation.detections:
        if not detection.object_id:
            raise RuntimeError("Cannot resume faithful pipeline; segmentation contains an unassigned detection.")
        if not Path(detection.mask_path).is_file():
            raise RuntimeError(f"Cannot resume faithful pipeline; mask is missing for {detection.object_id}: {detection.mask_path}")
        if not detection.crop_path or not Path(detection.crop_path).is_file():
            raise RuntimeError(f"Cannot resume faithful pipeline; crop is missing for {detection.object_id}: {detection.crop_path}")

    if Path(segmentation.image_path).resolve() != guidance_path:
        raise RuntimeError("Cannot resume faithful pipeline; segmentation image_path does not match guidance.png.")
    if Path(depth.image_path).resolve() != guidance_path:
        raise RuntimeError("Cannot resume faithful pipeline; depth image_path does not match guidance.png.")
    if not Path(depth.depth_path).is_file():
        raise RuntimeError(f"Cannot resume faithful pipeline; depth array is missing: {depth.depth_path}")
    if not Path(depth.preview_path).is_file():
        raise RuntimeError(f"Cannot resume faithful pipeline; depth preview is missing: {depth.preview_path}")


def metrics_from_sdf_report(scene: SceneSpec, report_path: str | Path, support_tolerance_m: float) -> Metrics:
    report = read_json(report_path)
    unstable: list[StabilityRecord] = []
    support_penalty = 0.0
    for item in report.get("objects", []):
        support_error = float(item.get("support_error_m", 0.0) or 0.0)
        support_penalty += support_error
        if support_error > support_tolerance_m:
            unstable.append(
                StabilityRecord(
                    object_id=str(item.get("object_id")),
                    reason="sdf_support_error",
                    distance=support_error,
                )
            )
    status = report.get("status")
    collision_count = 0 if status == "ok" else 1
    return Metrics(
        object_count=len(scene.objects),
        collision_count=collision_count,
        floating_count=0,
        unsupported_count=len(unstable),
        unstable=unstable,
        support_penalty=round(support_penalty, 6),
        total_penalty=round(support_penalty + collision_count, 6),
    )


def run_sdf_optimizer(
    scene: SceneSpec,
    graph: SceneGraph3D,
    registry: AssetRegistry,
    target_dir: Path,
    physics_cfg: dict[str, Any],
) -> SceneSpec:
    return SDFPhysicsOptimizer(
        SDFOptimizerConfig(
            device=str(physics_cfg.get("device", "cuda")),
            surface_samples=int(physics_cfg.get("surface_samples", 400)),
            optimizer=str(physics_cfg.get("optimizer", "sgd")),
            max_iters=int(physics_cfg.get("max_iters", 120)),
            learning_rate=float(physics_cfg.get("learning_rate", 0.03)),
        )
    ).optimize(scene, graph, registry, target_dir)


def resolve_render_environment(environment_cfg: dict[str, Any], root: Path) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for material_key in ("floor_material", "wall_material"):
        material = environment_cfg.get(material_key)
        if not material:
            continue
        resolved_material: dict[str, str] = {}
        for texture_key, raw_path in material.items():
            path = resolve_path(raw_path, root)
            if not path.is_file():
                raise RuntimeError(f"Render environment texture is missing: {path}")
            resolved_material[texture_key] = str(path)
        resolved[material_key] = resolved_material
    return resolved


def collect_judge_image_paths(out_dir: str | Path, guidance_path: str | Path) -> list[Path]:
    target = Path(out_dir)
    paths = [Path(guidance_path)]
    views_path = target / "render_views.json"
    if views_path.is_file():
        views = read_json(views_path)
        for path in list(views.get("scene_views", {}).values())[:4]:
            paths.append(Path(path))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Judge image inputs are missing: " + ", ".join(missing))
    return paths


def apply_faithful_repair_placements(scene: SceneSpec, registry: AssetRegistry, actions: list[dict[str, Any]]) -> None:
    objects = {obj.id: obj for obj in scene.objects}
    for action in actions:
        action_type = action.get("type")
        if action_type == "spread_children":
            spread_children(scene, registry, action.get("parent_id"))
            continue
        object_id = action.get("object_id")
        target_id = action.get("target_id") or action.get("parent_id")
        if object_id not in objects or target_id not in objects:
            continue
        obj = objects[object_id]
        target = objects[target_id]
        relation = action.get("relation") or ("near" if action_type == "move_near" else obj.relation)
        if relation in {"on", "inside"}:
            continue
        place_relative_to_target(obj, target, scene, registry, relation, clearance=0.45)
        snap_to_ground(obj, registry)


def snap_to_ground(obj: Any, registry: AssetRegistry) -> None:
    if obj.asset_id is None:
        return
    asset = registry.get(obj.asset_id)
    obj.placement.z = asset.dimensions[2] * obj.placement.scale * 0.5


def build_faithful_report(
    prompt: str,
    scene: SceneSpec,
    metrics: Metrics,
    judge: dict[str, Any],
    render_result: RenderResult,
    out_dir: Path,
    repair_history: list[dict[str, Any]] | None = None,
) -> str:
    sdf_report = read_json(out_dir / "sdf_optimizer.json") if (out_dir / "sdf_optimizer.json").exists() else {}
    render_validation = read_json(out_dir / "render_validation.json") if (out_dir / "render_validation.json").exists() else {}
    qualification = read_json(out_dir / "qualification.json") if (out_dir / "qualification.json").exists() else {}
    correspondence = read_json(out_dir / "correspondence_diagnostics.json") if (out_dir / "correspondence_diagnostics.json").exists() else {}
    depth_pose = read_json(out_dir / "depth_pose_refinement.json") if (out_dir / "depth_pose_refinement.json").exists() else {}
    lines = [
        "# Scenethesis Faithful Pipeline Report",
        "",
        f"Prompt: {prompt}",
        "",
        "## Artifacts",
        "",
        f"- Render: {render_result.render_path}",
        f"- GLB: {render_result.glb_path}",
        f"- Renderer: {render_result.renderer}",
        f"- Qualification: {qualification.get('status', 'pending')}",
        "",
        "## Object List",
        "",
    ]
    constraint_targets = {constraint.subject_id: constraint for constraint in scene.constraints}
    for obj in scene.objects:
        p = obj.placement
        constraint = constraint_targets.get(obj.id)
        if constraint and constraint.target_id:
            relation = f"{constraint.type} {constraint.target_id}"
        elif obj.relation and obj.parent_id:
            relation = f"{obj.relation} {obj.parent_id}"
        elif obj.relation:
            relation = obj.relation
        else:
            relation = "anchor"
        lines.append(
            f"- {obj.id}: category={obj.category}, asset={obj.asset_id}, role={obj.role}, "
            f"relation={relation}, pose=({p.x:.2f}, {p.y:.2f}, {p.z:.2f}, yaw={p.yaw_deg:.1f}, scale={p.scale:.2f})"
        )
    lines.extend(["", "## Relations", ""])
    for constraint in scene.constraints:
        target = f" -> {constraint.target_id}" if constraint.target_id else ""
        lines.append(f"- {constraint.subject_id}: {constraint.type}{target}")
    lines.extend(
        [
            "",
            "## SDF Metrics",
            "",
            f"- Objects: {metrics.object_count}",
            f"- Collision count: {metrics.collision_count}",
            f"- Floating count: {metrics.floating_count}",
            f"- Unsupported count: {metrics.unsupported_count}",
            f"- Support penalty: {metrics.support_penalty}",
            f"- Total penalty: {metrics.total_penalty}",
            "",
            "## Render Visual Support",
            "",
            f"- Status: {'ok' if render_validation.get('ok', False) else 'unknown/failed'}",
            f"- Visual support failures: {render_validation.get('visual_support_failure_count', 'unknown')}",
            "",
            "## Qualification",
            "",
            f"- Accepted: {qualification.get('accepted', 'pending')}",
            f"- Stage: {qualification.get('stage', 'pending')}",
            "",
            "## Depth Pose Refinement",
            "",
            f"- Status: {'ok' if depth_pose.get('ok', False) else 'missing/failed'}",
            f"- Scale updates: {depth_pose.get('applied_scale_updates', 'unknown')}",
            f"- Yaw updates: {depth_pose.get('applied_yaw_updates', 'unknown')}",
            "",
            "## RoMa Correspondence",
            "",
            f"- Status: {'ok' if correspondence.get('ok', False) else 'missing/failed'}",
            f"- Failed objects: {correspondence.get('failed_object_count', 'unknown')}",
            f"- Applied yaw updates: {correspondence.get('applied_updates', 'unknown')}",
        ]
    )
    for reason in qualification.get("reasons", []):
        lines.append(f"- Reason: {reason}")
    lines.extend(
        [
            "",
            "## SDF Optimization",
            "",
            f"- Status: {sdf_report.get('status', 'unknown')}",
            f"- Method: {sdf_report.get('method', 'mesh surface samples + signed-distance queries')}",
            f"- Surface samples per object: {sdf_report.get('surface_samples_per_object', 'unknown')}",
        ]
    )
    for item in sdf_report.get("objects", []):
        iterations = item.get("iterations", [])
        last = iterations[-1] if iterations else {}
        lines.append(
            f"- {item.get('object_id')}: status={item.get('status')}, support_error={item.get('support_error_m', 0)}, "
            f"last_penetrating_points={last.get('penetrating_points', 0)}"
        )
    lines.extend(["", "## Judge Scores", ""])
    for key, value in judge.get("scores", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Repair Actions", ""])
    actions = judge.get("repair_actions", [])
    if actions:
        for action in actions:
            lines.append(f"- {action}")
    else:
        lines.append("- No repair actions requested by judge.")
    lines.extend(["", "## Repair History", ""])
    if repair_history:
        for item in repair_history:
            post_judge = item.get("post_judge", {})
            lines.append(
                f"- Round {item.get('round')}: actions={item.get('actions', [])}, "
                f"post_needs_repair={post_judge.get('needs_repair')}"
            )
    else:
        lines.append("- No repair rounds executed.")
    if judge.get("notes"):
        lines.extend(["", "## Judge Notes", "", str(judge["notes"])])
    lines.append("")
    return "\n".join(lines)
