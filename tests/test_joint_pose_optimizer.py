from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.optimization.joint_pose_optimizer import run_joint_pose_optimizer
from scenethesis_mvp.schemas.scene_graph_3d import Object3DBoundingBox, ObjectPointCloudSpec, Pose3DSpec, SceneGraph3D
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, PlacementSpec, SceneSpec
from scenethesis_mvp.utils.io import write_json


def _registry() -> AssetRegistry:
    root = Path(__file__).resolve().parents[1]
    return AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")


def _write_correspondences(tmp_path: Path, object_ids: list[str]) -> None:
    corr_dir = tmp_path / "correspondences"
    corr_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for object_id in object_ids:
        path = corr_dir / f"{object_id}.npz"
        np.savez_compressed(
            path,
            guidance_xy=np.ones((20, 2), dtype="float32"),
            rendered_xy=np.zeros((20, 2), dtype="float32"),
            confidence=np.ones(20, dtype="float32") * 0.95,
        )
        records.append(
            {
                "object_id": object_id,
                "status": "ok",
                "match_count": 20,
                "inlier_count": 20,
                "mean_confidence": 0.95,
                "yaw_delta_deg": 10.0,
                "correspondence_path": str(path),
            }
        )
    write_json(
        tmp_path / "correspondence_diagnostics.json",
        {
            "ok": True,
            "provider": "roma",
            "objects": records,
        },
    )


def _graph(tmp_path: Path) -> SceneGraph3D:
    table_bbox = Object3DBoundingBox(center=[0.0, 0.0, 1.0], size=[1.8, 0.82, 0.85], yaw_deg=0.0)
    box_bbox = Object3DBoundingBox(center=[0.2, 0.0, 2.0], size=[0.456, 0.336, 0.20], yaw_deg=45.0)
    cylinder_bbox = Object3DBoundingBox(center=[-0.2, 0.0, 1.2], size=[0.32, 0.55, 0.32], yaw_deg=0.0)
    return SceneGraph3D(
        pointclouds=[
            ObjectPointCloudSpec(object_id="table", phrase="table", points_path=str(tmp_path / "table.ply"), point_count=256, bbox=table_bbox),
            ObjectPointCloudSpec(object_id="box", phrase="box", points_path=str(tmp_path / "box.ply"), point_count=256, bbox=box_bbox),
            ObjectPointCloudSpec(
                object_id="cylinder",
                phrase="cylinder",
                points_path=str(tmp_path / "cylinder.ply"),
                point_count=256,
                bbox=cylinder_bbox,
            ),
        ],
        poses=[
            Pose3DSpec(object_id="table", x=0.0, y=1.0, z=0.0, yaw_deg=0.0, scale=1.0),
            Pose3DSpec(object_id="box", x=0.6, y=2.0, z=0.0, yaw_deg=45.0, scale=1.2),
            Pose3DSpec(object_id="cylinder", x=-0.6, y=1.2, z=0.0, yaw_deg=0.0, scale=1.0),
        ],
    )


def test_joint_pose_optimizer_lowers_depth_roma_loss(tmp_path: Path) -> None:
    registry = _registry()
    scene = SceneSpec(
        prompt="warehouse table box cylinder",
        bounds=[6.0, 5.0, 3.0],
        objects=[
            ObjectSpec(id="table", category="table", asset_id="proc_table_01", role="anchor", placement=PlacementSpec(x=3.0, y=4.0, z=0.41)),
            ObjectSpec(id="box", category="box", asset_id="proc_box_01", role="parent", placement=PlacementSpec(x=1.0, y=1.0, z=0.14)),
            ObjectSpec(
                id="cylinder",
                category="cylinder",
                asset_id="proc_cylinder_01",
                role="parent",
                placement=PlacementSpec(x=4.5, y=2.5, z=0.275),
            ),
        ],
    )
    _write_correspondences(tmp_path, ["table", "box", "cylinder"])

    optimized, report = run_joint_pose_optimizer(
        scene,
        _graph(tmp_path),
        registry,
        tmp_path,
        {"max_iters": 4, "max_yaw_step_deg": 6.0, "max_translation_step_m": 0.08},
    )

    assert report["ok"] is True
    assert report["final_loss"]["total_loss"] < report["initial_loss"]["total_loss"]
    assert report["applied_updates"] > 0
    assert optimized.object_by_id("box").placement.scale > scene.object_by_id("box").placement.scale
    assert optimized.object_by_id("box").placement.yaw_deg != scene.object_by_id("box").placement.yaw_deg
    assert (tmp_path / "joint_pose_optimizer.json").is_file()
    assert (tmp_path / "pose_loss_history.json").is_file()


def test_joint_pose_optimizer_requires_correspondence_files(tmp_path: Path) -> None:
    registry = _registry()
    scene = SceneSpec(
        prompt="warehouse",
        objects=[ObjectSpec(id="table", category="table", asset_id="proc_table_01", role="anchor")],
    )
    write_json(
        tmp_path / "correspondence_diagnostics.json",
        {"ok": True, "provider": "roma", "objects": [{"object_id": "table", "status": "ok", "yaw_delta_deg": 0.0}]},
    )

    with pytest.raises(RuntimeError, match="requires correspondence file"):
        run_joint_pose_optimizer(scene, _graph(tmp_path), registry, tmp_path)
