from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from scenethesis_mvp.assets.grs_retriever import AssetCorrespondenceNoMatch, GRSAssetRetrievalConfig, GRSAssetRetriever
from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.visual_profiles import (
    allows_thin_view,
    validate_asset_view_image,
    validate_profile_view_freshness,
)
from scenethesis_mvp.schemas.asset_correspondence import AssetVisualProfile, ClipCandidateScore, ClipObjectShortlist
from scenethesis_mvp.schemas.scene_graph_3d import Object3DBoundingBox, ObjectPointCloudSpec, SceneGraph3D
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec
from scenethesis_mvp.schemas.segmentation import DetectionSpec, SegmentationResult
from scenethesis_mvp.utils.io import read_json


CANDIDATES = ("real_cardboard_box_01", "real_plastic_crate_01")


class FakeShortlistProvider:
    def shortlist(self, scene: SceneSpec, segmentation: SegmentationResult, registry: AssetRegistry, out_dir: Path, top_k: int):
        assert top_k == 2
        return [
            ClipObjectShortlist(
                object_id="box_01",
                category="box",
                candidates=[
                    ClipCandidateScore(asset_id=CANDIDATES[0], score=0.82, image_score=0.80, text_score=0.84, metadata_score=0.85),
                    ClipCandidateScore(asset_id=CANDIDATES[1], score=0.71, image_score=0.73, text_score=0.68, metadata_score=0.70),
                ],
            )
        ]


class FakeProfileStore:
    def __init__(self, profiles: dict[str, AssetVisualProfile]):
        self.profiles = profiles

    def ensure_profiles(self, asset_ids: list[str], registry: AssetRegistry) -> dict[str, AssetVisualProfile]:
        assert set(asset_ids) == set(CANDIDATES)
        return self.profiles


class FakeVisionClient:
    configured = True

    def __init__(self, response: dict[str, Any]):
        self.response = response
        self.image_count = 0

    def vision_json_multi(self, **kwargs: Any) -> dict[str, Any]:
        self.image_count = len(kwargs["image_paths"])
        return self.response


def _registry() -> AssetRegistry:
    root = Path(__file__).resolve().parents[1]
    return AssetRegistry.from_yaml(root / "configs" / "warehouse_asset_registry.yaml")


def _inputs(tmp_path: Path) -> tuple[SceneSpec, SegmentationResult, SceneGraph3D]:
    crop = tmp_path / "crop.png"
    Image.new("RGB", (32, 32), (155, 112, 70)).save(crop)
    Image.new("RGB", (32, 32), (120, 120, 120)).save(tmp_path / "guidance.png")
    scene = SceneSpec(
        prompt="warehouse box",
        objects=[ObjectSpec(id="box_01", category="box", role="anchor", asset_id=CANDIDATES[1])],
    )
    segmentation = SegmentationResult(
        image_path=str(tmp_path / "guidance.png"),
        image_width=32,
        image_height=32,
        detections=[
            DetectionSpec(
                object_id="box_01",
                phrase="cardboard box",
                score=0.95,
                box_xyxy=[0, 0, 32, 32],
                dino_box_xyxy=[0, 0, 32, 32],
                mask_path=str(tmp_path / "mask.png"),
                crop_path=str(crop),
                mask_area=1024,
            )
        ],
    )
    graph = SceneGraph3D(
        pointclouds=[
            ObjectPointCloudSpec(
                object_id="box_01",
                phrase="cardboard box",
                points_path=str(tmp_path / "box.ply"),
                point_count=128,
                bbox=Object3DBoundingBox(center=[0.0, 0.0, 1.0], size=[0.46, 0.36, 0.34], yaw_deg=0.0),
            )
        ]
    )
    return scene, segmentation, graph


def _profiles(tmp_path: Path, registry: AssetRegistry) -> dict[str, AssetVisualProfile]:
    profiles = {}
    for asset_id in CANDIDATES:
        paths = {}
        for view in ("front", "side", "oblique"):
            path = tmp_path / asset_id / f"{view}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (100, 100, 100)).save(path)
            paths[view] = str(path)
        asset = registry.get(asset_id)
        profiles[asset_id] = AssetVisualProfile(
            asset_id=asset.id,
            category=asset.category,
            dimensions_m=asset.dimensions,
            view_paths=paths,
            model="test-vision",
            description=f"A detailed visible warehouse asset profile for {asset.name}.",
            visual_features=[asset.name],
            materials=["visible material"],
            colors=["visible color"],
            affordances=[],
            support_surfaces=[],
            container_regions=[],
            articulated_parts=[],
            manipulable=True,
        )
    return profiles


def _decision(decision: str = "match", assessments: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "observed_object": {
            "description": "A rectangular brown corrugated cardboard shipping box.",
            "visual_features": ["rectangular", "closed top"],
            "materials": ["cardboard"],
            "colors": ["brown"],
            "visible_state": "closed",
            "is_valid_object": True,
        },
        "decision": decision,
        "selected_asset_id": CANDIDATES[0] if decision == "match" else None,
        "confidence": 0.91 if decision == "match" else 0.40,
        "assessments": assessments
        or [
            {
                "asset_id": CANDIDATES[0],
                "visual_similarity": 0.94,
                "semantic_similarity": 0.95,
                "dimension_compatible": True,
                "overall_score": 0.94,
                "notes": "Matches cardboard construction and proportions.",
            },
            {
                "asset_id": CANDIDATES[1],
                "visual_similarity": 0.45,
                "semantic_similarity": 0.50,
                "dimension_compatible": True,
                "overall_score": 0.47,
                "notes": "Plastic open crate is a different subtype.",
            },
        ],
        "reasoning": "The first candidate matches the observed material and closed-box geometry.",
    }


def _retriever(tmp_path: Path, response: dict[str, Any]) -> tuple[GRSAssetRetriever, FakeVisionClient]:
    registry = _registry()
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("strict match", encoding="utf-8")
    client = FakeVisionClient(response)
    retriever = GRSAssetRetriever(
        GRSAssetRetrievalConfig(
            model="test-vision",
            system_prompt_path=prompt,
            top_k=2,
            min_match_confidence=0.72,
            min_score_margin=0.05,
            max_shape_error=0.70,
        ),
        shortlist_provider=FakeShortlistProvider(),  # type: ignore[arg-type]
        profile_store=FakeProfileStore(_profiles(tmp_path, registry)),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
    )
    return retriever, client


def test_grs_retriever_requires_multiview_decision_for_final_asset(tmp_path: Path) -> None:
    registry = _registry()
    scene, segmentation, graph = _inputs(tmp_path)
    retriever, client = _retriever(tmp_path, _decision())

    updated = retriever.retrieve(scene, segmentation, graph, registry, tmp_path)

    assert updated.object_by_id("box_01").asset_id == CANDIDATES[0]
    assert client.image_count == 8  # crop, full scene, and three views for each of two candidates
    report = read_json(tmp_path / "asset_correspondence.json")
    assert report["ok"] is True
    assert report["matched_object_count"] == 1
    assert report["failed_object_count"] == 0


def test_grs_retriever_stops_on_explicit_no_match(tmp_path: Path) -> None:
    registry = _registry()
    scene, segmentation, graph = _inputs(tmp_path)
    retriever, _client = _retriever(tmp_path, _decision(decision="no_match"))

    with pytest.raises(AssetCorrespondenceNoMatch, match="No acceptable asset match") as exc_info:
        retriever.retrieve(scene, segmentation, graph, registry, tmp_path)

    assert exc_info.value.object_id == "box_01"
    assert exc_info.value.target_asset_id == CANDIDATES[0]
    report = read_json(tmp_path / "asset_correspondence.json")
    assert report["ok"] is False
    assert report["failed_object_count"] == 1
    assert report["objects"][0]["status"] == "no_match"


def test_grs_retriever_rejects_incomplete_candidate_assessments(tmp_path: Path) -> None:
    registry = _registry()
    scene, segmentation, graph = _inputs(tmp_path)
    incomplete = _decision()["assessments"][:1]
    retriever, _client = _retriever(tmp_path, _decision(assessments=incomplete))

    with pytest.raises(RuntimeError, match="cover each shortlisted candidate exactly once"):
        retriever.retrieve(scene, segmentation, graph, registry, tmp_path)

    assert read_json(tmp_path / "asset_correspondence.json")["ok"] is False


def test_grs_retriever_rejects_ambiguous_score_margin(tmp_path: Path) -> None:
    registry = _registry()
    scene, segmentation, graph = _inputs(tmp_path)
    assessments = _decision()["assessments"]
    assessments[1]["overall_score"] = 0.92
    retriever, _client = _retriever(tmp_path, _decision(assessments=assessments))

    with pytest.raises(RuntimeError, match="score margin"):
        retriever.retrieve(scene, segmentation, graph, registry, tmp_path)

    assert read_json(tmp_path / "asset_correspondence.json")["ok"] is False


def test_asset_profile_view_rejects_fully_transparent_render(tmp_path: Path) -> None:
    path = tmp_path / "black.png"
    Image.new("RGBA", (64, 64), (0, 0, 0, 0)).save(path)

    with pytest.raises(RuntimeError, match="empty/transparent"):
        validate_asset_view_image(path, "rack", "front")


def test_asset_profile_view_accepts_visible_framed_render(tmp_path: Path) -> None:
    path = tmp_path / "visible.png"
    image = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    image.paste((120, 120, 120, 255), (12, 8, 52, 58))
    image.save(path)

    validate_asset_view_image(path, "rack", "front")


def test_asset_profile_view_accepts_visible_thin_floor_marking(tmp_path: Path) -> None:
    path = tmp_path / "tape.png"
    image = Image.new("RGBA", (512, 512), (0, 0, 0, 0))
    image.paste((220, 180, 20, 255), (80, 254, 432, 258))
    image.save(path)

    validate_asset_view_image(path, "safety_tape", "front", allow_thin=True)


def test_asset_profile_allows_geometry_justified_thin_side_view_only() -> None:
    dimensions = [1.25, 0.10, 1.05]

    assert allows_thin_view("barrier", dimensions, "side") is True
    assert allows_thin_view("barrier", dimensions, "front") is False


def test_asset_profile_rejects_views_newer_than_profile(tmp_path: Path) -> None:
    profile = tmp_path / "profile.json"
    profile.write_text("{}", encoding="utf-8")
    view = tmp_path / "front.png"
    view.write_bytes(b"new-view")
    profile_stat = profile.stat()
    newer_time = profile_stat.st_mtime_ns + 2_000_000_000
    os.utime(view, ns=(newer_time, newer_time))

    with pytest.raises(RuntimeError, match="newer rendered views"):
        validate_profile_view_freshness(profile, "rack", {"front": view})
