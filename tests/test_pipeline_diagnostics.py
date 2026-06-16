from __future__ import annotations

from pathlib import Path

from scenethesis_mvp.pipeline.diagnostics import build_pipeline_diagnostics
from scenethesis_mvp.schemas.metrics import Metrics
from scenethesis_mvp.schemas.scene_graph_3d import Object3DBoundingBox, ObjectPointCloudSpec, Pose3DSpec, SceneGraph3D
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec
from scenethesis_mvp.schemas.segmentation import DetectionSpec, SegmentationResult
from scenethesis_mvp.utils.io import write_json


def test_pipeline_diagnostics_reports_stage_coverage(tmp_path: Path) -> None:
    scene = SceneSpec(
        prompt="warehouse",
        objects=[
            ObjectSpec(id="shelf_01", category="shelf", role="anchor", asset_id="real_warehouse_shelf_01"),
            ObjectSpec(id="box_01", category="box", role="child", parent_id="shelf_01", relation="on", asset_id="real_cardboard_box_01"),
        ],
    )
    write_json(
        tmp_path / "segmentation.json",
        SegmentationResult(
            image_path=str(tmp_path / "guidance.png"),
            image_width=8,
            image_height=8,
            detections=[
                DetectionSpec(object_id="shelf_01", phrase="shelf", score=0.9, box_xyxy=[0, 0, 4, 4], mask_path="shelf.png", mask_area=16),
                DetectionSpec(object_id="box_01", phrase="box", score=0.9, box_xyxy=[4, 4, 8, 8], mask_path="box.png", mask_area=16),
            ],
        ),
    )
    bbox = Object3DBoundingBox(center=[0.0, 0.0, 1.0], size=[1.0, 1.0, 1.0], yaw_deg=0.0)
    write_json(
        tmp_path / "scene_graph_3d.json",
        SceneGraph3D(
            pointclouds=[
                ObjectPointCloudSpec(object_id="shelf_01", phrase="shelf", points_path="shelf.ply", point_count=128, bbox=bbox),
                ObjectPointCloudSpec(object_id="box_01", phrase="box", points_path="box.ply", point_count=128, bbox=bbox),
            ],
            poses=[
                Pose3DSpec(object_id="shelf_01", x=0.0, y=0.0, z=0.0, yaw_deg=0.0, scale=1.0),
                Pose3DSpec(object_id="box_01", x=0.0, y=0.0, z=0.0, yaw_deg=0.0, scale=1.0),
            ],
        ),
    )
    write_json(tmp_path / "sdf_optimizer.json", {"status": "ok", "objects": [{"object_id": "shelf_01", "status": "ok"}, {"object_id": "box_01", "status": "ok"}]})
    write_json(tmp_path / "render_validation.json", {"ok": True, "visual_support_failure_count": 0})
    write_json(tmp_path / "correspondence_diagnostics.json", {"ok": True, "failed_object_count": 0})

    diagnostics = build_pipeline_diagnostics(scene, Metrics(object_count=2), {"needs_repair": False}, tmp_path)

    assert diagnostics["ok"] is True
    assert diagnostics["summary"]["segmentation_detection_count"] == 2
    assert diagnostics["summary"]["scene_graph_pointcloud_count"] == 2
