from __future__ import annotations

from pathlib import Path

from scenethesis_mvp.pipeline.qualification import build_success_qualification
from scenethesis_mvp.schemas.metrics import Metrics
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec
from scenethesis_mvp.utils.io import write_json


def _write_non_asset_success_artifacts(tmp_path: Path) -> None:
    for name in [
        "scene_spec.json",
        "scene.glb",
        "render.png",
        "metrics.json",
        "judge.json",
        "pipeline_diagnostics.json",
        "depth_pose_refinement.json",
        "joint_pose_optimizer.json",
        "guidance_validation.json",
    ]:
        (tmp_path / name).touch()
    write_json(
        tmp_path / "render_validation.json",
        {"ok": True, "visual_support_failure_count": 0, "visual_collision_failure_count": 0},
    )
    write_json(tmp_path / "correspondence_diagnostics.json", {"ok": True, "failed_object_count": 0, "applied_updates": 0})
    write_json(tmp_path / "depth_pose_refinement.json", {"ok": True, "applied_scale_updates": 0, "applied_yaw_updates": 0})
    write_json(
        tmp_path / "joint_pose_optimizer.json",
        {"ok": True, "initial_loss": {"total_loss": 1.0}, "final_loss": {"total_loss": 1.0}, "applied_updates": 0},
    )
    write_json(
        tmp_path / "guidance_validation.json",
        {"ok": True, "provider": "openai_vision_guidance_inventory", "attempts": [{}]},
    )


def test_qualification_requires_complete_multimodal_asset_correspondence(tmp_path: Path) -> None:
    scene = SceneSpec(
        prompt="warehouse box",
        objects=[ObjectSpec(id="box_01", category="box", role="anchor", asset_id="real_cardboard_box_01")],
    )
    _write_non_asset_success_artifacts(tmp_path)

    missing_report = build_success_qualification(scene, Metrics(object_count=1), {"needs_repair": False}, tmp_path, min_objects=1)
    asset_check = next(check for check in missing_report.checks if check.name == "asset_correspondence")
    assert missing_report.accepted is False
    assert asset_check.ok is False

    write_json(
        tmp_path / "asset_correspondence.json",
        {
            "ok": True,
            "provider": "openai_multiview_asset_correspondence",
            "matched_object_count": 1,
            "failed_object_count": 0,
        },
    )
    accepted_report = build_success_qualification(scene, Metrics(object_count=1), {"needs_repair": False}, tmp_path, min_objects=1)
    assert accepted_report.accepted is True
