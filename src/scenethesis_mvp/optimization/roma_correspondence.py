from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.schemas.segmentation import SegmentationResult
from scenethesis_mvp.utils.io import read_json, write_json
from scenethesis_mvp.utils.paths import resolve_path


@dataclass(frozen=True)
class RoMaCorrespondenceConfig:
    enabled: bool = True
    provider: str = "roma"
    device: str = "cuda"
    model: str = "roma_outdoor"
    weights_path: Path | None = None
    dinov2_weights_path: Path | None = None
    confidence_threshold: float = 0.6
    max_correspondences: int = 100
    min_correspondences: int = 12
    max_yaw_delta_deg: float = 12.0
    apply_updates: bool = True


class RoMaCorrespondenceRefiner:
    def __init__(self, config: RoMaCorrespondenceConfig):
        self.config = config

    def refine(
        self,
        scene: SceneSpec,
        segmentation: SegmentationResult,
        registry: AssetRegistry,
        out_dir: str | Path,
    ) -> tuple[SceneSpec, dict[str, Any]]:
        if self.config.provider != "roma":
            raise RuntimeError(f"Unsupported correspondence provider: {self.config.provider}")
        out_path = Path(out_dir)
        views_path = out_path / "render_views.json"
        if not views_path.is_file():
            raise RuntimeError(f"RoMa correspondence requires object alignment views: {views_path}")
        views = read_json(views_path)
        object_views = views.get("object_alignment_views", {})
        if not object_views:
            raise RuntimeError("RoMa correspondence requires non-empty object_alignment_views.")

        model = self._load_model()
        updated = scene.model_copy(deep=True)
        objects = {obj.id: obj for obj in updated.objects}
        detections = {item.object_id: item for item in segmentation.detections if item.object_id}
        records: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []
        applied_updates = 0
        applied_yaw_updates = 0
        failed = 0
        correspondence_dir = out_path / "correspondences"
        correspondence_dir.mkdir(parents=True, exist_ok=True)

        for obj in updated.objects:
            detection = detections.get(obj.id)
            view_path = Path(object_views.get(obj.id, ""))
            before = objects[obj.id].placement.model_dump(mode="json")
            if detection is None or not detection.crop_path:
                records.append({"object_id": obj.id, "status": "missing_guidance_crop"})
                history.append({"object_id": obj.id, "status": "missing_guidance_crop", "before": before, "after": before})
                failed += 1
                continue
            crop_path = Path(detection.crop_path)
            if not crop_path.is_file():
                records.append({"object_id": obj.id, "status": "missing_guidance_crop", "path": str(crop_path)})
                history.append({"object_id": obj.id, "status": "missing_guidance_crop", "before": before, "after": before})
                failed += 1
                continue
            if not view_path.is_file():
                records.append({"object_id": obj.id, "status": "missing_rendered_object_view", "path": str(view_path)})
                history.append({"object_id": obj.id, "status": "missing_rendered_object_view", "before": before, "after": before})
                failed += 1
                continue
            if not obj.asset_id:
                records.append({"object_id": obj.id, "status": "missing_asset_id"})
                history.append({"object_id": obj.id, "status": "missing_asset_id", "before": before, "after": before})
                failed += 1
                continue
            _ = registry.get(obj.asset_id)
            record = self._match_object(model, obj.id, crop_path, view_path, correspondence_dir)
            if record["status"] == "ok" and record.get("yaw_delta_deg") is not None:
                delta = float(record["yaw_delta_deg"])
                if self.config.apply_updates and abs(delta) > 0:
                    objects[obj.id].placement.yaw_deg = (objects[obj.id].placement.yaw_deg + delta) % 360.0
                    applied_updates += 1
                    applied_yaw_updates += 1
                    record["applied_yaw_delta_deg"] = round(delta, 4)
                elif not self.config.apply_updates:
                    record["proposed_yaw_delta_deg"] = round(delta, 4)
            if record["status"] != "ok":
                failed += 1
            history.append(
                {
                    "object_id": obj.id,
                    "status": record["status"],
                    "before": before,
                    "update": {
                        "yaw_delta_deg": record.get("applied_yaw_delta_deg", 0.0),
                        "scale_delta": 0.0,
                        "scale_update_status": "not_applied_object_alignment_views_are_scale_normalized",
                    },
                    "after": objects[obj.id].placement.model_dump(mode="json"),
                    "correspondence_path": record.get("correspondence_path"),
                }
            )
            records.append(record)

        history_path = out_path / "pose_alignment_history.json"
        write_json(
            history_path,
            {
                "provider": "roma",
                "model": self.config.model,
                "source": "guidance object crops matched against rendered object alignment views",
                "note": (
                    "Object alignment views are orthographically re-centered per object, so RoMa updates yaw only here. "
                    "Metric scale is refined separately from Depth Pro scene_graph_3d bounding boxes."
                ),
                "objects": history,
            },
        )
        report = {
            "ok": failed == 0,
            "provider": "roma",
            "model": self.config.model,
            "confidence_threshold": self.config.confidence_threshold,
            "min_correspondences": self.config.min_correspondences,
            "max_correspondences": self.config.max_correspondences,
            "apply_updates": self.config.apply_updates,
            "applied_updates": applied_updates,
            "applied_yaw_updates": applied_yaw_updates,
            "applied_scale_updates": 0,
            "failed_object_count": failed,
            "correspondence_dir": str(correspondence_dir),
            "pose_alignment_history_path": str(history_path),
            "objects": records,
        }
        write_json(out_path / "correspondence_diagnostics.json", report)
        return SceneSpec.model_validate(updated.model_dump()), report

    def _load_model(self) -> Any:
        try:
            import torch
            import romatch
        except Exception as exc:
            raise RuntimeError(f"RoMa correspondence dependencies are not importable: {exc}") from exc
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RoMa requested CUDA, but torch.cuda.is_available() is false.")
        if self.config.weights_path is None or not self.config.weights_path.is_file():
            raise RuntimeError(f"RoMa weights_path is missing: {self.config.weights_path}")
        weights = load_torch_weights(torch, self.config.weights_path, self.config.device)
        model_factory = getattr(romatch, self.config.model, None)
        if model_factory is None:
            raise RuntimeError(f"romatch does not expose model factory {self.config.model!r}")
        kwargs: dict[str, Any] = {"device": self.config.device, "weights": weights}
        if self.config.model in {"roma_outdoor", "roma_indoor"}:
            if self.config.dinov2_weights_path is None or not self.config.dinov2_weights_path.is_file():
                raise RuntimeError(f"RoMa dinov2_weights_path is missing: {self.config.dinov2_weights_path}")
            kwargs["dinov2_weights"] = load_torch_weights(torch, self.config.dinov2_weights_path, self.config.device)
        model = model_factory(**kwargs)
        model.eval()
        return model

    def _match_object(self, model: Any, object_id: str, crop_path: Path, rendered_path: Path, correspondence_dir: Path) -> dict[str, Any]:
        try:
            import torch
            import cv2
        except Exception as exc:
            raise RuntimeError(f"RoMa correspondence post-processing dependencies are missing: {exc}") from exc
        with Image.open(crop_path) as crop_image:
            crop_size = crop_image.size
        with Image.open(rendered_path) as rendered_image:
            rendered_size = rendered_image.size
        try:
            matches, certainty = model.match(str(crop_path), str(rendered_path), device=self.config.device)
            sampled_matches, sampled_certainty = model.sample(matches, certainty, num=self.config.max_correspondences)
            keep = sampled_certainty >= self.config.confidence_threshold
            sampled_matches = sampled_matches[keep]
            sampled_certainty = sampled_certainty[keep]
            if len(sampled_matches) < self.config.min_correspondences:
                return {
                    "object_id": object_id,
                    "status": "insufficient_correspondence",
                    "match_count": int(len(sampled_matches)),
                    "mean_confidence": float(sampled_certainty.mean().item()) if len(sampled_certainty) else 0.0,
                    "guidance_crop": str(crop_path),
                    "rendered_object_view": str(rendered_path),
                }
            kpts_a, kpts_b = model.to_pixel_coordinates(
                sampled_matches,
                crop_size[1],
                crop_size[0],
                rendered_size[1],
                rendered_size[0],
            )
            points_a = kpts_a.detach().cpu().numpy().astype("float32")
            points_b = kpts_b.detach().cpu().numpy().astype("float32")
            confidence = sampled_certainty.detach().cpu().numpy().astype("float32")
            correspondence_path = correspondence_dir / f"{object_id}.npz"
            np.savez_compressed(
                correspondence_path,
                guidance_xy=points_a,
                rendered_xy=points_b,
                confidence=confidence,
            )
            affine, inliers = cv2.estimateAffinePartial2D(points_b, points_a, method=cv2.RANSAC, ransacReprojThreshold=5.0)
            yaw_delta = 0.0
            inlier_count = int(inliers.sum()) if inliers is not None else 0
            affine_scale = None
            if affine is not None and inlier_count >= self.config.min_correspondences:
                raw_angle = float(np.degrees(np.arctan2(affine[1, 0], affine[0, 0])))
                yaw_delta = float(np.clip(raw_angle, -self.config.max_yaw_delta_deg, self.config.max_yaw_delta_deg))
                affine_scale = float(np.sqrt(max(0.0, np.linalg.det(affine[:, :2]))))
            record = {
                "object_id": object_id,
                "status": "ok",
                "match_count": int(len(sampled_matches)),
                "inlier_count": inlier_count,
                "mean_confidence": float(sampled_certainty.mean().item()),
                "yaw_delta_deg": round(yaw_delta, 4),
                "affine_scale_render_to_guidance": round(float(affine_scale), 6) if affine_scale is not None else None,
                "guidance_crop": str(crop_path),
                "rendered_object_view": str(rendered_path),
                "correspondence_path": str(correspondence_path),
            }
            write_json(correspondence_dir / f"{object_id}.json", record)
            return record
        finally:
            if self.config.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_roma_correspondence_refinement(
    scene: SceneSpec,
    segmentation: SegmentationResult,
    registry: AssetRegistry,
    out_dir: str | Path,
    cfg: dict[str, Any],
    root: Path,
) -> tuple[SceneSpec, dict[str, Any]]:
    config = RoMaCorrespondenceConfig(
        enabled=bool(cfg.get("enabled", True)),
        provider=str(cfg.get("provider", "roma")),
        device=str(cfg.get("device", "cuda")),
        model=str(cfg.get("model", "roma_outdoor")),
        weights_path=resolve_path(cfg["weights_path"], root) if cfg.get("weights_path") else None,
        dinov2_weights_path=resolve_path(cfg["dinov2_weights_path"], root) if cfg.get("dinov2_weights_path") else None,
        confidence_threshold=float(cfg.get("confidence_threshold", 0.6)),
        max_correspondences=int(cfg.get("max_correspondences", 100)),
        min_correspondences=int(cfg.get("min_correspondences", 12)),
        max_yaw_delta_deg=float(cfg.get("max_yaw_delta_deg", 12)),
        apply_updates=bool(cfg.get("apply_updates", True)),
    )
    if not config.enabled:
        raise RuntimeError("RoMa correspondence refinement is disabled in faithful config.")
    return RoMaCorrespondenceRefiner(config).refine(scene, segmentation, registry, out_dir)


def load_torch_weights(torch: Any, path: Path, device: str) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
