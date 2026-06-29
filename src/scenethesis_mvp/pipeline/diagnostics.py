from __future__ import annotations

from pathlib import Path
from typing import Any

from scenethesis_mvp.schemas.metrics import Metrics
from scenethesis_mvp.schemas.scene_graph_3d import SceneGraph3D
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.schemas.segmentation import SegmentationResult
from scenethesis_mvp.utils.io import read_json, write_json


def build_pipeline_diagnostics(
    scene: SceneSpec,
    metrics: Metrics,
    judge: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    target = Path(out_dir)
    object_ids = {obj.id for obj in scene.objects}
    anchors = [obj.id for obj in scene.objects if obj.role == "anchor"]
    missing_asset_ids = [obj.id for obj in scene.objects if not obj.asset_id]

    segmentation = _read_model(target / "segmentation.json", SegmentationResult)
    graph = _read_model(target / "scene_graph_3d.json", SceneGraph3D)
    sdf = read_json(target / "sdf_optimizer.json") if (target / "sdf_optimizer.json").is_file() else {}
    render_validation = read_json(target / "render_validation.json") if (target / "render_validation.json").is_file() else {}
    correspondence = read_json(target / "correspondence_diagnostics.json") if (target / "correspondence_diagnostics.json").is_file() else {}
    depth_pose = read_json(target / "depth_pose_refinement.json") if (target / "depth_pose_refinement.json").is_file() else {}
    joint_pose = read_json(target / "joint_pose_optimizer.json") if (target / "joint_pose_optimizer.json").is_file() else {}
    asset_correspondence = read_json(target / "asset_correspondence.json") if (target / "asset_correspondence.json").is_file() else {}
    guidance_validation = read_json(target / "guidance_validation.json") if (target / "guidance_validation.json").is_file() else {}

    segmentation_ids = {
        detection.object_id
        for detection in segmentation.detections
        if detection.object_id
    } if segmentation else set()
    graph_ids = {pointcloud.object_id for pointcloud in graph.pointclouds} if graph else set()
    sdf_objects = sdf.get("objects", [])
    sdf_failed = [item.get("object_id") for item in sdf_objects if item.get("status") != "ok"]

    checks = [
        _check("anchor_count", len(anchors) == 1, f"anchors={anchors}"),
        _check(
            "guidance_inventory",
            bool(guidance_validation.get("ok", False)),
            f"ok={guidance_validation.get('ok', False)}, attempts={len(guidance_validation.get('attempts', []))}",
        ),
        _check("asset_assignment", not missing_asset_ids, f"missing_asset_ids={missing_asset_ids}"),
        _check(
            "asset_correspondence",
            (
                bool(asset_correspondence.get("ok", False))
                and int(asset_correspondence.get("matched_object_count", -1)) == len(scene.objects)
                and int(asset_correspondence.get("failed_object_count", -1)) == 0
            ),
            (
                f"matched={asset_correspondence.get('matched_object_count', 'missing')}/{len(scene.objects)}, "
                f"failed={asset_correspondence.get('failed_object_count', 'missing')}"
            ),
        ),
        _check(
            "segmentation_coverage",
            segmentation is not None and not segmentation.missing_object_ids and object_ids.issubset(segmentation_ids),
            f"detections={len(segmentation_ids)}, missing={sorted(object_ids - segmentation_ids)}",
        ),
        _check(
            "scene_graph_coverage",
            graph is not None and object_ids.issubset(graph_ids) and not graph.missing_object_ids,
            f"pointclouds={len(graph_ids)}, missing={sorted(object_ids - graph_ids)}",
        ),
        _check(
            "depth_pose_refinement",
            bool(depth_pose.get("ok", False)),
            (
                f"scale_updates={depth_pose.get('applied_scale_updates', 'missing')}, "
                f"yaw_updates={depth_pose.get('applied_yaw_updates', 'missing')}"
            ),
        ),
        _check("sdf_status", sdf.get("status") == "ok" and not sdf_failed, f"status={sdf.get('status')}, failed={sdf_failed}"),
        _check(
            "collision_loss",
            metrics.collision_count == 0,
            f"collision_count={metrics.collision_count}, collision_penalty={metrics.collision_penalty}",
        ),
        _check(
            "support_loss",
            metrics.floating_count == 0 and metrics.unsupported_count == 0,
            f"floating={metrics.floating_count}, unsupported={metrics.unsupported_count}, support_penalty={metrics.support_penalty}",
        ),
        _check(
            "render_visual_support",
            bool(render_validation.get("ok", False)),
            (
                f"visual_support_failure_count={render_validation.get('visual_support_failure_count', 'missing')}, "
                f"visual_collision_failure_count={render_validation.get('visual_collision_failure_count', 'missing')}"
            ),
        ),
        _check(
            "roma_correspondence",
            not correspondence or bool(correspondence.get("ok", False)),
            f"failed_object_count={correspondence.get('failed_object_count', 'not_run')}",
        ),
        _check(
            "joint_pose_optimizer",
            bool(joint_pose.get("ok", False)),
            (
                f"initial_loss={joint_pose.get('initial_loss', {}).get('total_loss', 'missing')}, "
                f"final_loss={joint_pose.get('final_loss', {}).get('total_loss', 'missing')}, "
                f"applied_updates={joint_pose.get('applied_updates', 'missing')}"
            ),
        ),
        _check("judge", not bool(judge.get("needs_repair")), f"needs_repair={bool(judge.get('needs_repair'))}"),
    ]
    ok = all(item["ok"] for item in checks)
    return {
        "ok": ok,
        "checks": checks,
        "summary": {
            "object_count": len(scene.objects),
            "anchor_ids": anchors,
            "segmentation_detection_count": len(segmentation_ids),
            "scene_graph_pointcloud_count": len(graph_ids),
            "asset_correspondence_matched_count": asset_correspondence.get("matched_object_count"),
            "asset_correspondence_failed_count": asset_correspondence.get("failed_object_count"),
            "guidance_validation_attempt_count": len(guidance_validation.get("attempts", [])),
            "depth_pose_scale_updates": depth_pose.get("applied_scale_updates"),
            "depth_pose_yaw_updates": depth_pose.get("applied_yaw_updates"),
            "joint_pose_initial_loss": joint_pose.get("initial_loss", {}).get("total_loss"),
            "joint_pose_final_loss": joint_pose.get("final_loss", {}).get("total_loss"),
            "joint_pose_applied_updates": joint_pose.get("applied_updates"),
            "sdf_object_count": len(sdf_objects),
            "collision_count": metrics.collision_count,
            "floating_count": metrics.floating_count,
            "unsupported_count": metrics.unsupported_count,
            "judge_needs_repair": bool(judge.get("needs_repair")),
        },
    }


def write_pipeline_diagnostics(scene: SceneSpec, metrics: Metrics, judge: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    report = build_pipeline_diagnostics(scene, metrics, judge, out_dir)
    write_json(Path(out_dir) / "pipeline_diagnostics.json", report)
    return report


def _read_model(path: Path, model: Any) -> Any | None:
    if not path.is_file():
        return None
    return model.model_validate(read_json(path))


def _check(name: str, ok: bool, detail: str) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "detail": detail}
