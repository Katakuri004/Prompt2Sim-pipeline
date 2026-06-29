from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.schemas.asset_correspondence import ClipCandidateScore, ClipObjectShortlist
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.schemas.segmentation import SegmentationResult
from scenethesis_mvp.utils.io import write_json


@dataclass(frozen=True)
class ClipIndexConfig:
    index_path: Path
    device: str = "cuda"
    model_name: str = "ViT-L-14"
    pretrained: str = "openai"
    min_score: float = 0.18
    text_weight: float = 0.20
    metadata_weight: float = 0.35


class ClipAssetRetriever:
    def __init__(self, config: ClipIndexConfig):
        self.config = config

    def shortlist(
        self,
        scene: SceneSpec,
        segmentation: SegmentationResult,
        registry: AssetRegistry,
        out_dir: str | Path,
        top_k: int = 3,
        artifact_name: str | None = "clip_shortlist.json",
    ) -> list[ClipObjectShortlist]:
        if top_k < 1:
            raise ValueError("CLIP shortlist top_k must be at least 1")
        if not self.config.index_path.is_file():
            raise RuntimeError(f"CLIP asset index is missing: {self.config.index_path}")
        try:
            import torch
            import open_clip
        except Exception as exc:
            raise RuntimeError(f"OpenCLIP dependencies are not installed correctly: {exc}") from exc
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CLIP retrieval requested CUDA, but torch.cuda.is_available() is false.")

        index = np.load(self.config.index_path, allow_pickle=False)
        asset_ids = [str(value) for value in index["asset_ids"].tolist()]
        embeddings = index["embeddings"].astype("float32")
        if embeddings.ndim != 2 or not asset_ids:
            raise RuntimeError(f"CLIP index is invalid: {self.config.index_path}")

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.config.model_name,
            pretrained=self.config.pretrained,
            device=self.config.device,
        )
        model.eval()
        detections = {item.object_id: item for item in segmentation.detections if item.object_id}
        shortlists: list[ClipObjectShortlist] = []
        with torch.no_grad():
            for obj in scene.objects:
                detection = detections.get(obj.id)
                if detection is None or detection.crop_path is None:
                    raise RuntimeError(f"CLIP retrieval missing crop for object {obj.id}")
                crop = Image.open(detection.crop_path).convert("RGB")
                image_tensor = preprocess(crop).unsqueeze(0).to(self.config.device)
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                image_query = image_features.detach().cpu().numpy().astype("float32")[0]
                image_scores = embeddings @ image_query
                text_scores = self._text_scores(model, open_clip, embeddings, obj)
                metadata_scores = self._metadata_scores(asset_ids, registry, obj)
                text_weight = min(max(float(self.config.text_weight), 0.0), 1.0)
                metadata_weight = min(max(float(self.config.metadata_weight), 0.0), 1.0)
                if text_weight + metadata_weight > 0.9:
                    scale = 0.9 / (text_weight + metadata_weight)
                    text_weight *= scale
                    metadata_weight *= scale
                image_weight = 1.0 - text_weight - metadata_weight
                scores = image_scores * image_weight + text_scores * text_weight + metadata_scores * metadata_weight
                category_asset_ids = [
                    asset.id
                    for asset in registry.by_category(obj.category)
                    if asset.resolved_mesh_path(registry.base_dir) and asset.resolved_mesh_path(registry.base_dir).is_file()
                ]
                candidate_indices = [idx for idx, asset_id in enumerate(asset_ids) if asset_id in category_asset_ids]
                if not candidate_indices:
                    raise RuntimeError(f"No mesh-backed CLIP candidates for object {obj.id} category {obj.category}")
                ranked_indices = sorted(candidate_indices, key=lambda idx: float(scores[idx]), reverse=True)
                best_score = float(scores[ranked_indices[0]])
                if best_score < self.config.min_score:
                    raise RuntimeError(
                        f"CLIP retrieval score too low for {obj.id}: {best_score:.4f} < {self.config.min_score:.4f}"
                    )
                shortlists.append(
                    ClipObjectShortlist(
                        object_id=obj.id,
                        category=obj.category,
                        candidates=[
                            ClipCandidateScore(
                                asset_id=asset_ids[index],
                                score=round(float(scores[index]), 6),
                                image_score=round(float(image_scores[index]), 6),
                                text_score=round(float(text_scores[index]), 6),
                                metadata_score=round(float(metadata_scores[index]), 6),
                            )
                            for index in ranked_indices[:top_k]
                        ],
                    )
                )
        if artifact_name:
            write_json(Path(out_dir) / artifact_name, [item.model_dump(mode="json") for item in shortlists])
        return shortlists

    def _text_scores(self, model: Any, open_clip: Any, embeddings: np.ndarray, obj: Any) -> np.ndarray:
        import torch

        text = " ".join(
            part
            for part in [
                obj.id.replace("_", " "),
                obj.name or "",
                obj.description or "",
                obj.category.replace("_", " "),
            ]
            if part
        )
        tokens = open_clip.tokenize([text]).to(self.config.device)
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_query = text_features.detach().cpu().numpy().astype("float32")[0]
        return embeddings @ text_query

    def _metadata_scores(self, asset_ids: list[str], registry: AssetRegistry, obj: Any) -> np.ndarray:
        query_terms = metadata_terms(" ".join([obj.id, obj.name or "", obj.description or "", obj.category]))
        if not query_terms:
            return np.zeros(len(asset_ids), dtype="float32")
        scores = []
        for asset_id in asset_ids:
            asset = registry.get(asset_id)
            asset_terms = metadata_terms(" ".join([asset.id, asset.name, asset.category, " ".join(asset.tags)]))
            overlap = len(query_terms & asset_terms)
            scores.append(overlap / max(1, len(query_terms)))
        return np.asarray(scores, dtype="float32")


def metadata_terms(text: str) -> set[str]:
    stop = {
        "real",
        "warehouse",
        "storage",
        "child",
        "parent",
        "floor",
        "support",
        "box",
        "bin",
        "can",
        "crate",
        "tool",
        "container",
        "shelf",
        "01",
        "02",
        "03",
        "04",
        "05",
    }
    terms = {term for term in text.lower().replace("_", " ").replace("-", " ").split() if len(term) > 2}
    return {term for term in terms if term not in stop}


def build_clip_index(
    registry: AssetRegistry,
    output_path: str | Path,
    device: str = "cuda",
    model_name: str = "ViT-L-14",
    pretrained: str = "openai",
) -> None:
    try:
        import torch
        import open_clip
    except Exception as exc:
        raise RuntimeError(f"OpenCLIP dependencies are not installed correctly: {exc}") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CLIP index build requested CUDA, but torch.cuda.is_available() is false.")
    records = []
    for asset in registry.assets:
        mesh = asset.resolved_mesh_path(registry.base_dir)
        thumbnail = asset.resolved_thumbnail_path(registry.base_dir)
        if not mesh or not mesh.is_file():
            continue
        if not thumbnail or not thumbnail.is_file():
            raise RuntimeError(
                f"Asset {asset.id} has a mesh but no thumbnail_path. Faithful CLIP retrieval needs real asset images."
            )
        records.append((asset.id, thumbnail))
    if not records:
        raise RuntimeError("No mesh-backed assets with thumbnails were found for CLIP indexing.")

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device=device)
    model.eval()
    embeddings = []
    asset_ids = []
    with torch.no_grad():
        for asset_id, thumbnail_path in records:
            image = Image.open(thumbnail_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            features = model.encode_image(image_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
            embeddings.append(features.detach().cpu().numpy().astype("float32")[0])
            asset_ids.append(asset_id)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, asset_ids=np.asarray(asset_ids), embeddings=np.vstack(embeddings))
