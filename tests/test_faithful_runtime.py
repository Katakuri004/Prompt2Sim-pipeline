from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scenethesis_mvp.assets.clip_index import ClipAssetRetriever, ClipIndexConfig
from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.optimization.sdf_optimizer import MeshTemplate, PlacedMesh, SDFOptimizerConfig, SDFPhysicsOptimizer
from scenethesis_mvp.pipeline.run_faithful_pipeline import (
    apply_existing_asset_correspondence,
    load_asset_repair_state,
    load_existing_faithful_artifacts,
    validate_resume_artifacts,
)
from scenethesis_mvp.runtime.faithful import validate_faithful_runtime
from scenethesis_mvp.schemas.depth import CameraIntrinsics, DepthResult
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, PlacementSpec, SceneSpec
from scenethesis_mvp.schemas.segmentation import DetectionSpec, SegmentationResult
from scenethesis_mvp.schemas.scene_graph_3d import Object3DBoundingBox, ObjectPointCloudSpec, SceneGraph3D
from scenethesis_mvp.vision.depth_pose_refinement import apply_depth_pose_refinement
from scenethesis_mvp.vision.grounded_sam import GroundedSAMConfig, GroundedSAMSegmenter
from scenethesis_mvp.vision.image_guidance import ImageGuidanceResult
from scenethesis_mvp.vision.pointcloud import build_pointcloud_scene_graph


def test_asset_guidance_repair_state_preserves_attempts_and_sequence(tmp_path: Path) -> None:
    (tmp_path / "guidance_asset_repairs.json").write_text(
        json.dumps(
            {
                "repairs": [
                    {
                        "repair_index": 2,
                        "object_id": "table_01",
                        "target_asset_id": "table_asset_01",
                    },
                    {
                        "repair_index": 5,
                        "object_id": "barrier_01",
                        "target_asset_id": "barrier_asset_01",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    attempted, highest_index = load_asset_repair_state(tmp_path)

    assert attempted == {
        ("table_01", "table_asset_01"),
        ("barrier_01", "barrier_asset_01"),
    }
    assert highest_index == 5


def test_asset_correspondence_resume_requires_exact_fresh_success_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    scene = SceneSpec(
        prompt="box",
        objects=[ObjectSpec(id="box_01", category="box", role="anchor")],
    )
    inputs = [tmp_path / name for name in ("coarse.json", "guidance.png", "segmentation.json", "index.npz")]
    for path in inputs:
        path.write_bytes(b"input")
    report = tmp_path / "asset_correspondence.json"
    report.write_text(
        json.dumps(
            {
                "ok": True,
                "provider": "openai_multiview_asset_correspondence",
                "objects": [
                    {
                        "object_id": "box_01",
                        "status": "matched",
                        "selected_asset_id": "authored_clean_cardboard_box_01",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    class ProfileStore:
        def ensure_profiles(self, asset_ids: list[str], _registry: AssetRegistry) -> dict[str, object]:
            assert asset_ids == ["authored_clean_cardboard_box_01"]
            return {}

    resumed = apply_existing_asset_correspondence(
        scene,
        report,
        inputs,
        registry,
        ProfileStore(),  # type: ignore[arg-type]
    )

    assert resumed.object_by_id("box_01").asset_id == "authored_clean_cardboard_box_01"


def test_asset_correspondence_resume_rejects_stale_report(tmp_path: Path) -> None:
    report = tmp_path / "asset_correspondence.json"
    report.write_text(
        json.dumps({"ok": True, "provider": "openai_multiview_asset_correspondence", "objects": []}),
        encoding="utf-8",
    )
    newer_input = tmp_path / "segmentation.json"
    newer_input.write_bytes(b"newer")
    report_time = report.stat().st_mtime_ns
    newer_time = report_time + 2_000_000_000
    newer_input.touch()
    import os

    os.utime(newer_input, ns=(newer_time, newer_time))

    with pytest.raises(RuntimeError, match="older than its inputs"):
        apply_existing_asset_correspondence(
            SceneSpec(prompt="box", objects=[ObjectSpec(id="box_01", category="box", role="anchor")]),
            report,
            [newer_input],
            AssetRegistry.from_yaml(Path(__file__).resolve().parents[1] / "configs" / "warehouse_asset_registry.yaml"),
            object(),  # type: ignore[arg-type]
        )


def test_faithful_runtime_rejects_substitutes(tmp_path: Path) -> None:
    config = {
        "paper_faithful": {"enabled": True, "allow_substitutes": True, "min_free_disk_gb": 0},
        "render": {},
        "segmentation": {},
        "depth": {},
        "asset_retrieval": {},
    }
    report = validate_faithful_runtime(config, tmp_path)
    assert report.ok is False
    assert any("allow_substitutes" in error for error in report.errors)


def test_faithful_runtime_rejects_unbounded_or_regenerative_guidance(tmp_path: Path) -> None:
    config = {
        "paper_faithful": {"enabled": True, "allow_substitutes": False, "min_free_disk_gb": 0},
        "scene": {"max_objects": 18},
        "image_guidance": {"max_validation_attempts": 3, "correction_mode": "regenerate"},
        "render": {},
        "segmentation": {},
        "depth": {},
        "asset_retrieval": {},
    }

    report = validate_faithful_runtime(config, tmp_path)

    assert any("image_guidance.correction_mode" in error for error in report.errors)
    assert any("scene.max_objects" in error for error in report.errors)


def test_faithful_runtime_accepts_bounded_sixteen_object_guidance(tmp_path: Path) -> None:
    config = {
        "paper_faithful": {"enabled": True, "allow_substitutes": False, "min_free_disk_gb": 0},
        "scene": {"max_objects": 16},
        "image_guidance": {"max_validation_attempts": 4, "correction_mode": "edit_high_fidelity"},
        "render": {},
        "segmentation": {},
        "depth": {},
        "asset_retrieval": {"guidance_repair_rounds": 2},
    }

    report = validate_faithful_runtime(config, tmp_path)

    scene_check = next(check for check in report.checks if check.name == "scene.max_objects")
    assert scene_check.ok is True


def test_faithful_resume_requires_saved_artifacts(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="coarse_scene_spec.json"):
        load_existing_faithful_artifacts(tmp_path)


def test_faithful_resume_rejects_missing_crop(tmp_path: Path) -> None:
    guidance_path = tmp_path / "guidance.png"
    Image.new("RGB", (8, 8), "white").save(guidance_path)
    mask_path = tmp_path / "mask.png"
    Image.new("L", (8, 8), 255).save(mask_path)
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.ones((8, 8), dtype="float32"))
    preview_path = tmp_path / "depth_preview.png"
    Image.new("L", (8, 8), 128).save(preview_path)

    scene = SceneSpec(prompt="warehouse", objects=[ObjectSpec(id="box", category="box", role="anchor")])
    guidance = ImageGuidanceResult(
        guidance_path=guidance_path,
        image_metadata={},
        upsampled_prompt="",
        candidates=[],
        object_boxes={"box": [0.0, 0.0, 1.0, 1.0]},
    )
    segmentation = SegmentationResult(
        image_path=str(guidance_path),
        image_width=8,
        image_height=8,
        detections=[
            DetectionSpec(
                object_id="box",
                phrase="box",
                score=0.9,
                box_xyxy=[0, 0, 8, 8],
                dino_box_xyxy=[0, 0, 8, 8],
                mask_path=str(mask_path),
                crop_path=str(tmp_path / "missing_crop.png"),
                mask_area=64,
            )
        ],
    )
    depth = DepthResult(
        image_path=str(guidance_path),
        depth_path=str(depth_path),
        preview_path=str(preview_path),
        intrinsics=CameraIntrinsics(width=8, height=8, fx=8.0, fy=8.0, cx=3.5, cy=3.5),
        min_depth_m=1.0,
        max_depth_m=1.0,
    )

    with pytest.raises(RuntimeError, match="crop is missing"):
        validate_resume_artifacts(scene, guidance, segmentation, depth)


def test_grounded_sam_fails_on_missing_model_files(tmp_path: Path) -> None:
    scene = SceneSpec(prompt="warehouse", objects=[ObjectSpec(id="shelf", category="shelf", role="anchor")])
    image_path = tmp_path / "guidance.png"
    Image.new("RGB", (32, 32), "white").save(image_path)
    segmenter = GroundedSAMSegmenter(
        GroundedSAMConfig(
            grounding_dino_config=tmp_path / "missing_config.py",
            grounding_dino_checkpoint=tmp_path / "missing_dino.pth",
            sam_checkpoint=tmp_path / "missing_sam.pth",
        )
    )
    with pytest.raises(RuntimeError, match="Missing Grounded-SAM files"):
        segmenter.segment(image_path, scene, tmp_path)


def test_mask_depth_projection_writes_pointcloud_graph(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.png"
    mask = np.zeros((8, 8), dtype="uint8")
    mask[2:6, 2:6] = 255
    Image.fromarray(mask).save(mask_path)
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.ones((8, 8), dtype="float32") * 2.0)
    segmentation = SegmentationResult(
        image_path=str(tmp_path / "guidance.png"),
        image_width=8,
        image_height=8,
        detections=[
            DetectionSpec(
                object_id="box",
                phrase="box",
                score=0.9,
                box_xyxy=[2, 2, 6, 6],
                dino_box_xyxy=[2, 2, 6, 6],
                mask_path=str(mask_path),
                crop_path=None,
                mask_area=16,
            )
        ],
    )
    depth = DepthResult(
        image_path=str(tmp_path / "guidance.png"),
        depth_path=str(depth_path),
        preview_path=str(tmp_path / "depth_preview.png"),
        intrinsics=CameraIntrinsics(width=8, height=8, fx=8.0, fy=8.0, cx=3.5, cy=3.5),
        min_depth_m=2.0,
        max_depth_m=2.0,
    )
    graph = build_pointcloud_scene_graph(segmentation, depth, tmp_path, min_mask_pixels=4)
    assert graph.poses[0].object_id == "box"
    assert Path(graph.pointclouds[0].points_path).is_file()


def test_depth_pose_refinement_applies_bounded_scale_and_yaw(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")
    scene = SceneSpec(
        prompt="box",
        objects=[
            ObjectSpec(
                id="box",
                category="box",
                asset_id="proc_box_01",
                role="anchor",
                placement=PlacementSpec(scale=1.0, yaw_deg=0.0, z=0.14),
            )
        ],
    )
    graph = SceneGraph3D(
        pointclouds=[
            ObjectPointCloudSpec(
                object_id="box",
                phrase="box",
                points_path=str(tmp_path / "box.ply"),
                point_count=256,
                bbox=Object3DBoundingBox(center=[0, 0, 1], size=[0.76, 0.56, 0.20], yaw_deg=45.0),
            )
        ],
    )

    refined, report = apply_depth_pose_refinement(
        scene,
        graph,
        registry,
        tmp_path,
        {"max_scale": 2.5, "max_scale_delta_fraction": 0.20, "max_yaw_delta_deg": 12.0},
    )

    box = refined.object_by_id("box")
    assert box.placement.scale == pytest.approx(1.2)
    assert box.placement.yaw_deg == pytest.approx(12.0)
    assert box.placement.z == pytest.approx(registry.get("proc_box_01").dimensions[2] * box.placement.scale * 0.5)
    assert report["applied_scale_updates"] == 1
    assert report["applied_yaw_updates"] == 1
    assert (tmp_path / "depth_pose_refinement.json").is_file()


def test_depth_pose_refinement_requires_graph_coverage(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")
    scene = SceneSpec(
        prompt="box",
        objects=[ObjectSpec(id="box", category="box", asset_id="proc_box_01", role="anchor")],
    )

    with pytest.raises(RuntimeError, match="missing graph point clouds"):
        apply_depth_pose_refinement(scene, SceneGraph3D(pointclouds=[]), registry, tmp_path)


def test_clip_retriever_fails_when_index_missing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    scene = SceneSpec(
        prompt="warehouse",
        objects=[
            ObjectSpec(
                id="box",
                category="box",
                asset_id="real_cardboard_box_01",
                role="anchor",
                placement=PlacementSpec(z=0.17),
            )
        ],
    )
    segmentation = SegmentationResult(
        image_path=str(tmp_path / "guidance.png"),
        image_width=8,
        image_height=8,
        detections=[],
    )
    retriever = ClipAssetRetriever(ClipIndexConfig(index_path=tmp_path / "missing_index.npz"))
    with pytest.raises(RuntimeError, match="CLIP asset index is missing"):
        retriever.shortlist(scene, segmentation, registry, tmp_path)


def test_sdf_optimizer_does_not_downgrade_to_mesh_proxy(tmp_path: Path) -> None:
    pytest.importorskip("pytorch3d")
    pytest.importorskip("pytorch3d._C")
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    scene = SceneSpec(
        prompt="warehouse",
        objects=[ObjectSpec(id="shelf", category="shelf", asset_id="real_warehouse_shelf_01", role="anchor")],
    )
    optimizer = SDFPhysicsOptimizer(SDFOptimizerConfig(surface_samples=400))
    optimized = optimizer.optimize(scene, graph=object(), registry=registry, out_dir=tmp_path)  # type: ignore[arg-type]
    report = json.loads((tmp_path / "sdf_optimizer.json").read_text(encoding="utf-8"))
    assert optimized.object_by_id("shelf").asset_id == "real_warehouse_shelf_01"
    assert report["method"] == "mesh surface samples + signed-distance queries"
    assert report["status"] == "ok"


def test_sdf_optimizer_derives_imported_rack_support_planes_from_mesh() -> None:
    import trimesh

    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    asset = registry.get("hf_pallet_rack_large_01")
    optimizer = SDFPhysicsOptimizer(SDFOptimizerConfig(surface_samples=400))
    mesh, components = optimizer._load_mesh_with_components(asset.resolved_mesh_path(registry.base_dir), trimesh)  # type: ignore[arg-type]
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    source_size = bounds[1] - bounds[0]
    source_center = (bounds[0] + bounds[1]) * 0.5
    scale = np.asarray(asset.dimensions, dtype=np.float64) / source_size
    vertices = (np.asarray(mesh.vertices, dtype=np.float64) - source_center) * (
        scale
    )
    normalized_mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)
    normalized_components = [
        trimesh.Trimesh(
            vertices=(np.asarray(component.vertices, dtype=np.float64) - source_center) * scale,
            faces=np.asarray(component.faces, dtype=np.int64),
            process=False,
        )
        for component in components
    ]
    support_planes = optimizer._derive_template_support_planes(asset, normalized_mesh, normalized_components)
    assert len(support_planes) == 2
    assert len(support_planes) != len(asset.support_heights)
    assert support_planes[0] == pytest.approx(-0.89, abs=0.04)
    assert support_planes[1] == pytest.approx(0.83, abs=0.04)


def test_sdf_optimizer_derives_connected_shelf_support_planes_from_faces() -> None:
    import trimesh

    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    asset = registry.get("real_warehouse_shelf_01")
    optimizer = SDFPhysicsOptimizer(SDFOptimizerConfig(surface_samples=400))
    mesh, components = optimizer._load_mesh_with_components(asset.resolved_mesh_path(registry.base_dir), trimesh)  # type: ignore[arg-type]
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    source_size = bounds[1] - bounds[0]
    source_center = (bounds[0] + bounds[1]) * 0.5
    scale = np.asarray(asset.dimensions, dtype=np.float64) / source_size
    vertices = (np.asarray(mesh.vertices, dtype=np.float64) - source_center) * scale
    normalized_mesh = trimesh.Trimesh(vertices=vertices, faces=np.asarray(mesh.faces, dtype=np.int64), process=False)
    normalized_components = [
        trimesh.Trimesh(
            vertices=(np.asarray(component.vertices, dtype=np.float64) - source_center) * scale,
            faces=np.asarray(component.faces, dtype=np.int64),
            process=False,
        )
        for component in components
    ]
    support_planes = optimizer._derive_template_support_planes(asset, normalized_mesh, normalized_components)
    assert len(support_planes) >= 4
    assert support_planes[0] == pytest.approx(-0.84, abs=0.08)
    assert support_planes[-1] == pytest.approx(0.97, abs=0.05)


def test_sdf_support_target_prefers_mesh_planes_over_registry_heights() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    optimizer = SDFPhysicsOptimizer(SDFOptimizerConfig(surface_samples=400))
    template = MeshTemplate(
        object_id="box",
        asset_id="real_cardboard_box_01",
        vertices=np.asarray([[-0.2, -0.2, -0.1], [0.2, 0.2, 0.2]], dtype=np.float64),
        faces=np.asarray([[0, 1, 1]], dtype=np.int64),
        surface_points=np.zeros((1, 3), dtype=np.float64),
        bottom_points=np.zeros((1, 3), dtype=np.float64),
        support_planes=[],
    )
    child = ObjectSpec(
        id="box",
        category="box",
        asset_id="real_cardboard_box_01",
        parent_id="rack",
        relation="on",
        placement=PlacementSpec(z=0.85),
    )
    parent = PlacedMesh(
        object_id="rack",
        asset_id="hf_pallet_rack_large_01",
        mesh=None,
        query=None,
        centroid=np.zeros(3, dtype=np.float64),
        bounds=np.asarray([[0.0, 0.0, 0.0], [2.8, 0.95, 2.75]], dtype=np.float64),
        support_planes=[0.4, 1.8],
    )
    target, support_model = optimizer._support_target(child, template, parent, registry)
    assert support_model == "mesh_derived_support_plane"
    assert target == pytest.approx(0.4)


def test_sdf_optimizer_accepts_ceiling_mounted_assets(tmp_path: Path) -> None:
    pytest.importorskip("pytorch3d")
    pytest.importorskip("pytorch3d._C")
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")
    scene = SceneSpec(
        prompt="warehouse ceiling light",
        bounds=(7.0, 6.0, 3.0),
        objects=[
            ObjectSpec(
                id="light",
                category="light",
                asset_id="real_mounted_fluorescent_lights_01",
                role="anchor",
                relation="against_wall",
            )
        ],
    )
    optimizer = SDFPhysicsOptimizer(SDFOptimizerConfig(surface_samples=400))
    optimized = optimizer.optimize(scene, graph=object(), registry=registry, out_dir=tmp_path)  # type: ignore[arg-type]
    report = json.loads((tmp_path / "sdf_optimizer.json").read_text(encoding="utf-8"))
    assert optimized.object_by_id("light").placement.z > 2.0
    assert report["objects"][0]["support"] == "ceiling_mount"
    assert report["objects"][0]["support_error_m"] == 0.0
    assert report["status"] == "ok"


def test_sdf_optimizer_fails_when_mesh_is_missing(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    registry = AssetRegistry.from_yaml(root / "configs" / "asset_registry.yaml")
    scene = SceneSpec(
        prompt="warehouse",
        objects=[ObjectSpec(id="box", category="box", asset_id="proc_box_01", role="anchor")],
    )
    optimizer = SDFPhysicsOptimizer(SDFOptimizerConfig(surface_samples=400))
    with pytest.raises(RuntimeError):
        optimizer.optimize(scene, graph=object(), registry=registry, out_dir=tmp_path)  # type: ignore[arg-type]
