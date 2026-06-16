from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.retriever import AssetRetriever
from scenethesis_mvp.layout.initial_layout import generate_initial_layout
from scenethesis_mvp.layout.optimizer import compute_metrics, optimize_layout, spread_children
from scenethesis_mvp.llm.judge import SceneJudge
from scenethesis_mvp.llm.openai_client import OpenAIClient
from scenethesis_mvp.llm.planner import ScenePlanner
from scenethesis_mvp.llm.repair import RepairEngine
from scenethesis_mvp.physics.mesh_check import refine_mesh_layout
from scenethesis_mvp.render.blender_runner import RenderResult, render_scene, resolve_blender_path
from scenethesis_mvp.schemas.mesh_metrics import MeshMetrics
from scenethesis_mvp.schemas.metrics import Metrics
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.layout.relation_rules import normalize_support_relation_semantics
from scenethesis_mvp.utils.io import read_yaml, write_json, write_text
from scenethesis_mvp.utils.paths import project_root, resolve_path
from scenethesis_mvp.utils.seeds import seed_everything
from scenethesis_mvp.vision.guidance import VisionGuidance
from scenethesis_mvp.vision.scene_graph_layout import apply_vision_position_hints, apply_vision_relations


@dataclass(frozen=True)
class PipelineResult:
    out_dir: Path
    scene: SceneSpec
    metrics: Metrics
    judge: dict[str, Any]
    render: RenderResult
    repair_history: list[dict[str, Any]]
    vision_artifacts: dict[str, Any] | None = None
    mesh_metrics: MeshMetrics | None = None


def run_pipeline(
    prompt: str,
    out_dir: str | Path,
    repair_rounds: int = 2,
    config_path: str | Path = "configs/default.yaml",
) -> PipelineResult:
    root = project_root()
    config_file = resolve_path(config_path, root)
    config = read_yaml(config_file)
    seed_everything(int(config.get("seed", 7)))
    target_dir = Path(out_dir)
    if not target_dir.is_absolute():
        target_dir = root / target_dir

    paths = config.get("paths", {})
    registry_path = resolve_path(paths.get("asset_registry", "configs/asset_registry.yaml"), root)
    registry = AssetRegistry.from_yaml(registry_path)
    bounds = tuple(config.get("scene", {}).get("bounds", [7.0, 6.0, 3.0]))
    openai_cfg = config.get("openai", {})
    layout_cfg = config.get("layout", {})
    render_cfg = config.get("render", {})
    vision_cfg = config.get("vision_guidance", {})
    mesh_cfg = config.get("mesh_physics", {})
    validate_runtime(render_cfg)

    target_dir.mkdir(parents=True, exist_ok=True)
    clear_optional_mesh_outputs(target_dir)

    planner = ScenePlanner(
        model=openai_cfg.get("model", "gpt-4o-mini"),
        system_prompt_path=resolve_path(paths.get("planner_prompt", "configs/prompts/planner_system.txt"), root),
        max_retries=int(openai_cfg.get("max_retries", 3)),
    )
    scene = planner.plan(prompt, registry, bounds)  # type: ignore[arg-type]
    scene = AssetRetriever(registry).attach_assets(scene)
    scene = normalize_support_relation_semantics(scene, registry)
    vision_artifacts: dict[str, Any] | None = None
    vision_graph = None
    if bool(vision_cfg.get("enabled", False)):
        vision = VisionGuidance(
            image_model=openai_cfg.get("image_model", "gpt-image-1"),
            vision_model=openai_cfg.get("vision_model", "gpt-4o-mini"),
            system_prompt_path=resolve_path(
                paths.get("vision_scene_graph_prompt", "configs/prompts/vision_scene_graph_system.txt"),
                root,
            ),
            max_retries=int(openai_cfg.get("max_retries", 3)),
        )
        vision_result = vision.run(
            prompt=prompt,
            scene=scene,
            registry=registry,
            out_dir=target_dir,
            image_size=str(vision_cfg.get("image_size", "1024x1024")),
            image_quality=str(vision_cfg.get("image_quality", "low")),
        )
        vision_graph = vision_result.graph
        scene, relation_diagnostics = apply_vision_relations(
            scene,
            vision_graph,
            confidence_threshold=float(vision_cfg.get("confidence_threshold", 0.35)),
        )
        vision_artifacts = {
            "guidance_path": str(vision_result.guidance_path),
            "image_metadata": vision_result.image_metadata,
            "relation_diagnostics": relation_diagnostics,
        }

    scene = generate_initial_layout(scene, registry)
    if vision_graph is not None:
        scene, position_diagnostics = apply_vision_position_hints(
            scene,
            vision_graph,
            registry,
            confidence_threshold=float(vision_cfg.get("confidence_threshold", 0.35)),
            scene_fill=float(vision_cfg.get("scene_fill", 0.78)),
        )
        if vision_artifacts is not None:
            vision_artifacts["position_diagnostics"] = position_diagnostics
        write_json(target_dir / "pose_init_from_vision.json", scene)
    scene, metrics = optimize_layout(
        scene,
        registry,
        max_iters=int(layout_cfg.get("max_optimizer_iters", 80)),
        boundary_margin=float(layout_cfg.get("boundary_margin", 0.15)),
        contact_tolerance=float(layout_cfg.get("contact_tolerance", 0.04)),
    )
    mesh_metrics: MeshMetrics | None = None
    mesh_samples: dict[str, Any] | None = None
    scene, metrics, mesh_metrics, mesh_samples = maybe_refine_with_mesh_physics(
        scene,
        metrics,
        registry,
        layout_cfg,
        mesh_cfg,
        target_dir,
    )
    if vision_graph is not None:
        write_json(target_dir / "pose_optimized.json", scene)

    resolution = tuple(render_cfg.get("resolution", [1200, 900]))
    render_result = render_scene(
        scene,
        registry,
        target_dir,
        resolution=resolution,  # type: ignore[arg-type]
        blender_path=render_cfg.get("blender_path"),
    )

    judge = SceneJudge(
        model=openai_cfg.get("vision_model", "gpt-4o-mini"),
        system_prompt_path=resolve_path(paths.get("judge_prompt", "configs/prompts/judge_system.txt"), root),
        max_retries=int(openai_cfg.get("max_retries", 3)),
    ).judge(prompt, scene, render_result.render_path, metrics, mesh_metrics=mesh_metrics)

    repair_history: list[dict[str, Any]] = []
    repair_engine = RepairEngine()
    for round_index in range(max(0, repair_rounds)):
        if not judge.get("needs_repair"):
            break
        actions = list(judge.get("repair_actions", []))
        repair_history.append({"round": round_index + 1, "actions": actions, "pre_metrics": metrics.model_dump(mode="json")})
        if mesh_metrics is not None:
            repair_history[-1]["pre_mesh_metrics"] = mesh_metrics.model_dump(mode="json")
        scene = repair_engine.apply(scene, actions)
        scene = normalize_support_relation_semantics(scene, registry)
        for action in actions:
            if action.get("type") == "spread_children":
                spread_children(scene, registry, action.get("parent_id"))
        scene = generate_initial_layout(scene, registry)
        if vision_graph is not None:
            scene, position_diagnostics = apply_vision_position_hints(
                scene,
                vision_graph,
                registry,
                confidence_threshold=float(vision_cfg.get("confidence_threshold", 0.35)),
                scene_fill=float(vision_cfg.get("scene_fill", 0.78)),
            )
            repair_history[-1]["vision_position_diagnostics"] = position_diagnostics
        scene, metrics = optimize_layout(
            scene,
            registry,
            max_iters=int(layout_cfg.get("max_optimizer_iters", 80)),
            boundary_margin=float(layout_cfg.get("boundary_margin", 0.15)),
            contact_tolerance=float(layout_cfg.get("contact_tolerance", 0.04)),
        )
        scene, metrics, mesh_metrics, mesh_samples = maybe_refine_with_mesh_physics(
            scene,
            metrics,
            registry,
            layout_cfg,
            mesh_cfg,
            target_dir,
        )
        if vision_graph is not None:
            write_json(target_dir / "pose_optimized.json", scene)
        render_result = render_scene(
            scene,
            registry,
            target_dir,
            resolution=resolution,  # type: ignore[arg-type]
            blender_path=render_cfg.get("blender_path"),
        )
        judge = SceneJudge(
            model=openai_cfg.get("vision_model", "gpt-4o-mini"),
            system_prompt_path=resolve_path(paths.get("judge_prompt", "configs/prompts/judge_system.txt"), root),
            max_retries=int(openai_cfg.get("max_retries", 3)),
        ).judge(prompt, scene, render_result.render_path, metrics, mesh_metrics=mesh_metrics)
        repair_history[-1]["post_metrics"] = metrics.model_dump(mode="json")
        if mesh_metrics is not None:
            repair_history[-1]["post_mesh_metrics"] = mesh_metrics.model_dump(mode="json")
        repair_history[-1]["post_judge"] = judge

    write_json(target_dir / "scene_spec.json", scene)
    write_json(target_dir / "metrics.json", metrics)
    write_json(target_dir / "judge.json", judge)
    write_json(target_dir / "repair_history.json", repair_history)
    if vision_artifacts is not None:
        write_json(target_dir / "vision_artifacts.json", vision_artifacts)
    write_text(
        target_dir / "report.md",
        build_report(
            prompt,
            scene,
            metrics,
            judge,
            repair_history,
            render_result,
            vision_artifacts=vision_artifacts,
            mesh_metrics=mesh_metrics,
        ),
    )
    return PipelineResult(
        out_dir=target_dir,
        scene=scene,
        metrics=metrics,
        judge=judge,
        render=render_result,
        repair_history=repair_history,
        vision_artifacts=vision_artifacts,
        mesh_metrics=mesh_metrics,
    )


def clear_optional_mesh_outputs(target_dir: Path) -> None:
    for name in ("mesh_metrics.json", "sampled_collision_points.json"):
        path = target_dir / name
        if path.exists():
            path.unlink()


def maybe_refine_with_mesh_physics(
    scene: SceneSpec,
    metrics: Metrics,
    registry: AssetRegistry,
    layout_cfg: dict[str, Any],
    mesh_cfg: dict[str, Any],
    target_dir: Path,
) -> tuple[SceneSpec, Metrics, MeshMetrics | None, dict[str, Any] | None]:
    if not bool(mesh_cfg.get("enabled", False)):
        return scene, metrics, None, None
    refined_scene, mesh_metrics, mesh_samples = refine_mesh_layout(
        scene,
        registry,
        max_iters=int(mesh_cfg.get("max_refine_iters", 6)),
        sample_points=int(mesh_cfg.get("sample_points", 256)),
        collision_distance=float(mesh_cfg.get("collision_distance", 0.035)),
        support_tolerance=float(mesh_cfg.get("support_tolerance", layout_cfg.get("contact_tolerance", 0.04))),
        require_meshes=bool(mesh_cfg.get("require_meshes", False)),
    )
    refined_metrics = compute_metrics(
        refined_scene,
        registry,
        boundary_margin=float(layout_cfg.get("boundary_margin", 0.15)),
        contact_tolerance=float(layout_cfg.get("contact_tolerance", 0.04)),
    )
    write_json(target_dir / "mesh_metrics.json", mesh_metrics)
    write_json(target_dir / "sampled_collision_points.json", mesh_samples)
    return refined_scene, refined_metrics, mesh_metrics, mesh_samples


def validate_runtime(render_cfg: dict[str, Any]) -> None:
    missing: list[str] = []
    if not OpenAIClient().configured:
        missing.append("OPENAI_API_KEY is missing or the OpenAI client could not be initialized")
    if not resolve_blender_path(render_cfg.get("blender_path")):
        missing.append("Blender executable was not found; install Blender or set BLENDER_PATH")
    if missing:
        details = "\n- ".join(missing)
        raise RuntimeError(f"Runtime requirements missing:\n- {details}")


def build_report(
    prompt: str,
    scene: SceneSpec,
    metrics: Metrics,
    judge: dict[str, Any],
    repair_history: list[dict[str, Any]],
    render_result: RenderResult,
    vision_artifacts: dict[str, Any] | None = None,
    mesh_metrics: MeshMetrics | None = None,
) -> str:
    lines = [
        "# Scenethesis MVP Report",
        "",
        f"Prompt: {prompt}",
        "",
        "## Renderer",
        "",
        f"- Renderer: {render_result.renderer}",
        f"- Notes: {render_result.notes}",
        "",
    ]
    if vision_artifacts:
        relation_diag = vision_artifacts.get("relation_diagnostics", {})
        position_diag = vision_artifacts.get("position_diagnostics", {})
        lines.extend(
            [
                "## Vision Guidance",
                "",
                f"- Guidance image: {vision_artifacts.get('guidance_path')}",
                f"- Applied visual relations: {len(relation_diag.get('applied_relations', []))}",
                f"- Positioned floor objects from image hints: {len(position_diag.get('positioned_objects', []))}",
                "",
            ]
        )
    lines.extend(["## Object List", ""])
    for obj in scene.objects:
        placement = obj.placement
        relation = f"{obj.relation} {obj.parent_id}" if obj.parent_id else (obj.relation or "floor")
        lines.append(
            f"- {obj.id}: category={obj.category}, asset={obj.asset_id}, role={obj.role}, relation={relation}, "
            f"pose=({placement.x:.2f}, {placement.y:.2f}, {placement.z:.2f}, yaw={placement.yaw_deg:.1f}, scale={placement.scale:.2f})"
        )
    lines.extend(["", "## Relations", ""])
    for constraint in scene.constraints:
        target = f" -> {constraint.target_id}" if constraint.target_id else ""
        lines.append(f"- {constraint.subject_id}: {constraint.type}{target}")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            f"- Objects: {metrics.object_count}",
            f"- Collisions: {metrics.collision_count}",
            f"- Floating objects: {metrics.floating_count}",
            f"- Unsupported objects: {metrics.unsupported_count}",
            f"- Boundary violations: {metrics.boundary_violations}",
            f"- Total penalty: {metrics.total_penalty}",
            "",
        ]
    )
    if mesh_metrics is not None:
        lines.extend(
            [
                "## Mesh Physics",
                "",
                f"- Mesh-backed objects: {mesh_metrics.mesh_object_count}",
                f"- Proxy objects: {mesh_metrics.proxy_object_count}",
                f"- Broad-phase overlapping pairs: {mesh_metrics.broad_phase_pair_count}",
                f"- Narrow-phase sampled pairs: {mesh_metrics.narrow_phase_pair_count}",
                f"- Mesh collisions: {mesh_metrics.mesh_collision_count}",
                f"- Mesh support failures: {mesh_metrics.support_failure_count}",
                f"- Total mesh penalty: {mesh_metrics.total_penalty}",
                "",
            ]
        )
        if mesh_metrics.proxy_object_count:
            proxy_ids = [record.object_id for record in mesh_metrics.objects if record.source != "mesh"]
            lines.append(f"- Warning: non-mesh proxy checks used for {', '.join(proxy_ids)}.")
            lines.append("")
        if mesh_metrics.mesh_collisions:
            lines.extend(["### Mesh Collision Details", ""])
            for collision in mesh_metrics.mesh_collisions:
                lines.append(
                    f"- {collision.object_a} intersects {collision.object_b}; "
                    f"distance={collision.min_distance:.4f}, aabb_penetration={collision.aabb_penetration:.4f}"
                )
            lines.append("")
        failed_supports = [record for record in mesh_metrics.supports if not record.supported]
        if failed_supports:
            lines.extend(["### Mesh Support Failures", ""])
            for support in failed_supports:
                lines.append(
                    f"- {support.object_id}: bottom_z={support.bottom_z:.4f}, support_z={support.support_z:.4f}, "
                    f"contact_distance={support.contact_distance:.4f}"
                )
            lines.append("")
    lines.extend(["## Judge Scores", ""])
    for key, value in judge.get("scores", {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Repair Actions", ""])
    if repair_history:
        for item in repair_history:
            lines.append(f"- Round {item['round']}: {json.dumps(item.get('actions', []), sort_keys=True)}")
    else:
        lines.append("- No repair actions were needed.")
    if metrics.collisions:
        lines.extend(["", "## Collision Details", ""])
        for collision in metrics.collisions:
            lines.append(f"- {collision.object_a} intersects {collision.object_b}; penetration={collision.penetration:.4f}")
    lines.append("")
    return "\n".join(lines)
