from __future__ import annotations

from pathlib import Path
from typing import Any

from scenethesis_mvp.schemas.mesh_metrics import MeshMetrics
from scenethesis_mvp.schemas.metrics import Metrics
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import read_json


REQUIRED_OUTPUTS = [
    "scene_spec.json",
    "scene.glb",
    "render.png",
    "metrics.json",
    "judge.json",
    "pipeline_diagnostics.json",
    "qualification.json",
    "report.md",
]


def validate_run(out_dir: str | Path, min_objects: int = 6) -> dict[str, Any]:
    target = Path(out_dir)
    missing = [name for name in REQUIRED_OUTPUTS if not (target / name).exists()]
    result: dict[str, Any] = {"ok": not missing, "missing": missing}
    if missing:
        return result
    scene = SceneSpec.model_validate(read_json(target / "scene_spec.json"))
    metrics = Metrics.model_validate(read_json(target / "metrics.json"))
    judge = read_json(target / "judge.json")
    qualification = read_json(target / "qualification.json")
    mesh_metrics_path = target / "mesh_metrics.json"
    mesh_metrics = MeshMetrics.model_validate(read_json(mesh_metrics_path)) if mesh_metrics_path.exists() else None
    render_validation_path = target / "render_validation.json"
    render_validation = read_json(render_validation_path) if render_validation_path.exists() else {"ok": True}
    correspondence_path = target / "correspondence_diagnostics.json"
    correspondence = read_json(correspondence_path) if correspondence_path.exists() else {}
    joint_pose_path = target / "joint_pose_optimizer.json"
    joint_pose = read_json(joint_pose_path) if joint_pose_path.exists() else {}
    result.update(
        {
            "object_count": len(scene.objects),
            "collision_count": metrics.collision_count,
            "floating_count": metrics.floating_count,
            "unsupported_count": metrics.unsupported_count,
            "judge_needs_repair": bool(judge.get("needs_repair")),
            "min_object_count_ok": len(scene.objects) >= min_objects,
            "no_floating_ok": metrics.floating_count == 0,
            "collision_reported_ok": metrics.collision_count >= 0,
            "judge_ok": not bool(judge.get("needs_repair")),
            "qualification_accepted": bool(qualification.get("accepted", False)),
            "qualification_status": qualification.get("status"),
            "render_visual_support_ok": bool(render_validation.get("ok", False)),
            "render_visual_support_failure_count": int(render_validation.get("visual_support_failure_count", 0)),
            "roma_correspondence_ok": bool(correspondence.get("ok", False)),
            "roma_failed_object_count": correspondence.get("failed_object_count"),
            "joint_pose_optimizer_ok": bool(joint_pose.get("ok", False)),
            "joint_pose_initial_loss": joint_pose.get("initial_loss", {}).get("total_loss"),
            "joint_pose_final_loss": joint_pose.get("final_loss", {}).get("total_loss"),
            "joint_pose_applied_updates": joint_pose.get("applied_updates"),
        }
    )
    if mesh_metrics is not None:
        result.update(
            {
                "mesh_object_count": mesh_metrics.mesh_object_count,
                "mesh_proxy_object_count": mesh_metrics.proxy_object_count,
                "mesh_collision_count": mesh_metrics.mesh_collision_count,
                "mesh_support_failure_count": mesh_metrics.support_failure_count,
                "mesh_physics_ok": mesh_metrics.mesh_clean,
            }
        )
    else:
        result["mesh_physics_ok"] = True
    result["ok"] = (
        result["ok"]
        and result["min_object_count_ok"]
        and result["no_floating_ok"]
        and result["collision_reported_ok"]
        and result["judge_ok"]
        and result["qualification_accepted"]
        and result["mesh_physics_ok"]
        and result["render_visual_support_ok"]
    )
    return result
