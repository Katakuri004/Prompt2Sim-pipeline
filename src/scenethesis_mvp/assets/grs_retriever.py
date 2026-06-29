from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.assets.clip_index import ClipAssetRetriever
from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.visual_profiles import AssetVisualProfileStore, VIEW_NAMES
from scenethesis_mvp.llm.openai_client import OpenAIClient
from scenethesis_mvp.schemas.asset_correspondence import (
    AssetCorrespondenceObject,
    AssetCorrespondenceReport,
    AssetMatchDecision,
    AssetVisualProfile,
    ClipObjectShortlist,
)
from scenethesis_mvp.schemas.scene_graph_3d import SceneGraph3D
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.schemas.segmentation import SegmentationResult
from scenethesis_mvp.utils.io import read_text, write_json


@dataclass(frozen=True)
class GRSAssetRetrievalConfig:
    model: str
    system_prompt_path: Path
    top_k: int = 3
    min_match_confidence: float = 0.72
    min_score_margin: float = 0.05
    max_shape_error: float = 0.70
    max_retries: int = 3


class AssetCorrespondenceNoMatch(RuntimeError):
    def __init__(self, object_id: str, target_asset_id: str, candidate_ids: list[str], reason: str):
        self.object_id = object_id
        self.target_asset_id = target_asset_id
        self.candidate_ids = candidate_ids
        self.reason = reason
        super().__init__(f"No acceptable asset match for {object_id}: {reason}")


class GRSAssetRetriever:
    def __init__(
        self,
        config: GRSAssetRetrievalConfig,
        shortlist_provider: ClipAssetRetriever,
        profile_store: AssetVisualProfileStore,
        client: OpenAIClient | None = None,
    ):
        self.config = config
        self.shortlist_provider = shortlist_provider
        self.profile_store = profile_store
        self.client = client or OpenAIClient()

    def retrieve(
        self,
        scene: SceneSpec,
        segmentation: SegmentationResult,
        graph: SceneGraph3D,
        registry: AssetRegistry,
        out_dir: str | Path,
    ) -> SceneSpec:
        target = Path(out_dir)
        target.mkdir(parents=True, exist_ok=True)
        records: list[AssetCorrespondenceObject] = []
        if not self.client.configured:
            error = "OPENAI_API_KEY is required for final multimodal asset correspondence."
            self._write_report(target, scene, records, error)
            raise RuntimeError(error)
        try:
            shortlists = self.shortlist_provider.shortlist(
                scene,
                segmentation,
                registry,
                target,
                top_k=self.config.top_k,
            )
            self._validate_shortlist_coverage(scene, shortlists)
            candidate_ids = [candidate.asset_id for item in shortlists for candidate in item.candidates]
            profiles = self.profile_store.ensure_profiles(candidate_ids, registry)
            updated = scene.model_copy(deep=True)
            detections = {item.object_id: item for item in segmentation.detections if item.object_id}
            pointclouds = {item.object_id: item for item in graph.pointclouds}
            scene_image_path = Path(segmentation.image_path)
            if not scene_image_path.is_file():
                raise RuntimeError(f"Full scene image is missing for asset correspondence: {scene_image_path}")
            for shortlist in shortlists:
                try:
                    record = self._match_object(
                        shortlist,
                        updated,
                        detections,
                        pointclouds,
                        profiles,
                        registry,
                        scene_image_path,
                    )
                except Exception as exc:
                    records.append(
                        AssetCorrespondenceObject(
                            object_id=shortlist.object_id,
                            category=shortlist.category,
                            status="failed",
                            selected_asset_id=None,
                            confidence=None,
                            score_margin=None,
                            observed_object=None,
                            shortlist=shortlist.candidates,
                            assessments=[],
                            shape_errors={},
                            error=str(exc),
                        )
                    )
                    raise
                records.append(record)
                if record.status != "matched" or not record.selected_asset_id:
                    if not record.assessments:
                        raise RuntimeError(f"No acceptable asset match for {record.object_id}: no candidate assessments")
                    target_asset_id = max(record.assessments, key=lambda assessment: assessment.overall_score).asset_id
                    raise AssetCorrespondenceNoMatch(
                        object_id=record.object_id,
                        target_asset_id=target_asset_id,
                        candidate_ids=[candidate.asset_id for candidate in record.shortlist],
                        reason=record.error or "no_match",
                    )
                selected = registry.get(record.selected_asset_id)
                obj = updated.object_by_id(record.object_id)
                obj.asset_id = selected.id
                obj.name = selected.name
            self._write_report(target, scene, records, None)
            return SceneSpec.model_validate(updated.model_dump())
        except AssetCorrespondenceNoMatch as exc:
            self._write_report(target, scene, records, str(exc))
            raise
        except Exception as exc:
            self._write_report(target, scene, records, str(exc))
            raise RuntimeError(f"GRS multimodal asset correspondence failed: {exc}") from exc

    def _match_object(
        self,
        shortlist: ClipObjectShortlist,
        scene: SceneSpec,
        detections: dict[str, Any],
        pointclouds: dict[str, Any],
        profiles: dict[str, AssetVisualProfile],
        registry: AssetRegistry,
        scene_image_path: Path,
    ) -> AssetCorrespondenceObject:
        detection = detections.get(shortlist.object_id)
        if detection is None or not detection.crop_path or not Path(detection.crop_path).is_file():
            raise RuntimeError(f"Segmented crop is missing for {shortlist.object_id}")
        pointcloud = pointclouds.get(shortlist.object_id)
        if pointcloud is None:
            raise RuntimeError(f"3D bounding box is missing for {shortlist.object_id}")
        observed_dims = [float(value) for value in pointcloud.bbox.size]
        if len(observed_dims) != 3 or any(not np.isfinite(value) or value <= 0 for value in observed_dims):
            raise RuntimeError(f"Invalid observed 3D dimensions for {shortlist.object_id}: {observed_dims}")
        shape_errors = {
            candidate.asset_id: round(scale_invariant_shape_error(observed_dims, registry.get(candidate.asset_id).dimensions), 6)
            for candidate in shortlist.candidates
        }
        image_paths: list[Path] = [Path(detection.crop_path), scene_image_path]
        candidate_image_order: list[dict[str, Any]] = []
        candidate_payloads: list[dict[str, Any]] = []
        for candidate in shortlist.candidates:
            profile = profiles.get(candidate.asset_id)
            if profile is None:
                raise RuntimeError(f"Visual profile is missing for shortlisted asset {candidate.asset_id}")
            indices: list[int] = []
            for view_name in VIEW_NAMES:
                path = Path(profile.view_paths[view_name])
                if not path.is_file():
                    raise RuntimeError(f"Required candidate view is missing: {candidate.asset_id}/{view_name}: {path}")
                image_paths.append(path)
                indices.append(len(image_paths))
            candidate_image_order.append(
                {
                    "asset_id": candidate.asset_id,
                    "image_numbers": indices,
                    "view_order": list(VIEW_NAMES),
                }
            )
            candidate_payloads.append(
                {
                    "clip_scores": candidate.model_dump(mode="json"),
                    "profile": profile.model_dump(mode="json", exclude={"view_paths"}),
                    "scale_invariant_shape_error": shape_errors[candidate.asset_id],
                }
            )
        obj = scene.object_by_id(shortlist.object_id)
        payload = {
            "object_id": obj.id,
            "planned_category": obj.category,
            "planned_name": obj.name,
            "planned_description": obj.description,
            "observed_bbox_dimensions_m": observed_dims,
            "image_number_1": "segmented observed-object crop",
            "image_number_2": "full scene context containing the observed object",
            "candidate_image_order": candidate_image_order,
            "candidates": candidate_payloads,
            "thresholds": {
                "minimum_match_confidence": self.config.min_match_confidence,
                "minimum_selected_score_margin": self.config.min_score_margin,
                "maximum_scale_invariant_shape_error": self.config.max_shape_error,
            },
        }
        response = self.client.vision_json_multi(
            system_prompt=read_text(self.config.system_prompt_path),
            user_prompt=json.dumps(payload, indent=2),
            image_paths=image_paths,
            model=self.config.model,
            json_schema=AssetMatchDecision.model_json_schema(),
            schema_name="AssetMatchDecision",
            max_retries=self.config.max_retries,
            image_detail="high",
        )
        decision = AssetMatchDecision.model_validate(response)
        return self._validate_decision(shortlist, decision, shape_errors)

    def _validate_decision(
        self,
        shortlist: ClipObjectShortlist,
        decision: AssetMatchDecision,
        shape_errors: dict[str, float],
    ) -> AssetCorrespondenceObject:
        candidate_ids = [candidate.asset_id for candidate in shortlist.candidates]
        assessment_ids = [assessment.asset_id for assessment in decision.assessments]
        if len(assessment_ids) != len(set(assessment_ids)) or set(assessment_ids) != set(candidate_ids):
            raise RuntimeError(
                f"Asset assessments for {shortlist.object_id} must cover each shortlisted candidate exactly once: "
                f"expected={candidate_ids}, received={assessment_ids}"
            )
        if decision.decision == "no_match" or not decision.observed_object.is_valid_object:
            return AssetCorrespondenceObject(
                object_id=shortlist.object_id,
                category=shortlist.category,
                status="no_match",
                selected_asset_id=None,
                confidence=decision.confidence,
                score_margin=None,
                observed_object=decision.observed_object,
                shortlist=shortlist.candidates,
                assessments=decision.assessments,
                shape_errors=shape_errors,
                error=decision.reasoning,
            )
        selected_id = decision.selected_asset_id
        if selected_id not in candidate_ids:
            raise RuntimeError(f"Selected asset for {shortlist.object_id} is not in the shortlist: {selected_id}")
        ordered = sorted(decision.assessments, key=lambda item: item.overall_score, reverse=True)
        if ordered[0].asset_id != selected_id:
            raise RuntimeError(
                f"Selected asset for {shortlist.object_id} is not the highest-scored assessment: "
                f"selected={selected_id}, highest={ordered[0].asset_id}"
            )
        score_margin = ordered[0].overall_score - ordered[1].overall_score if len(ordered) > 1 else 1.0
        selected = ordered[0]
        errors: list[str] = []
        if decision.confidence < self.config.min_match_confidence:
            errors.append(f"confidence {decision.confidence:.3f} < {self.config.min_match_confidence:.3f}")
        if score_margin < self.config.min_score_margin:
            errors.append(f"score margin {score_margin:.3f} < {self.config.min_score_margin:.3f}")
        if not selected.dimension_compatible:
            errors.append("selected candidate is not dimension-compatible")
        if shape_errors[selected_id] > self.config.max_shape_error:
            errors.append(
                f"shape error {shape_errors[selected_id]:.3f} > {self.config.max_shape_error:.3f}"
            )
        if errors:
            raise RuntimeError(f"Asset match rejected for {shortlist.object_id}/{selected_id}: " + "; ".join(errors))
        return AssetCorrespondenceObject(
            object_id=shortlist.object_id,
            category=shortlist.category,
            status="matched",
            selected_asset_id=selected_id,
            confidence=decision.confidence,
            score_margin=round(score_margin, 6),
            observed_object=decision.observed_object,
            shortlist=shortlist.candidates,
            assessments=decision.assessments,
            shape_errors=shape_errors,
            error=None,
        )

    def _validate_shortlist_coverage(self, scene: SceneSpec, shortlists: list[ClipObjectShortlist]) -> None:
        expected = {obj.id for obj in scene.objects}
        received = [item.object_id for item in shortlists]
        if len(received) != len(set(received)) or set(received) != expected:
            raise RuntimeError(
                f"CLIP shortlist coverage mismatch: missing={sorted(expected - set(received))}, "
                f"extra={sorted(set(received) - expected)}"
            )

    def _write_report(
        self,
        target: Path,
        scene: SceneSpec,
        records: list[AssetCorrespondenceObject],
        error: str | None,
    ) -> None:
        matched = sum(item.status == "matched" for item in records)
        ok = error is None and matched == len(scene.objects) and len(records) == len(scene.objects)
        report = AssetCorrespondenceReport(
            ok=ok,
            provider="openai_multiview_asset_correspondence",
            shortlist_provider="open_clip",
            model=self.config.model,
            object_count=len(scene.objects),
            matched_object_count=matched,
            failed_object_count=0 if ok else len(scene.objects) - matched,
            objects=records,
            error=error,
        )
        write_json(target / "asset_correspondence.json", report.model_dump(mode="json"))


def scale_invariant_shape_error(observed_dimensions: list[float], asset_dimensions: list[float]) -> float:
    observed = np.sort(np.asarray(observed_dimensions, dtype=np.float64))
    asset = np.sort(np.asarray(asset_dimensions, dtype=np.float64))
    if observed.shape != (3,) or asset.shape != (3,) or np.any(observed <= 0) or np.any(asset <= 0):
        raise ValueError("shape error requires three positive observed and asset dimensions")
    observed /= observed.max()
    asset /= asset.max()
    return float(np.max(np.abs(observed - asset)))
