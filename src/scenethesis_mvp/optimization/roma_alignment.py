from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from scenethesis_mvp.utils.io import write_json


@dataclass(frozen=True)
class RoMaConfig:
    device: str = "cuda"
    model: str = "roma_outdoor"
    confidence_threshold: float = 0.6
    max_correspondences: int = 100


class RoMaCorrespondenceMatcher:
    def __init__(self, config: RoMaConfig):
        self.config = config

    def match_pair(
        self,
        guidance_image: str | Path,
        rendered_image: str | Path,
        out_dir: str | Path,
        object_id: str,
    ) -> Path:
        try:
            import torch
            from romatch import roma_indoor, roma_outdoor
        except Exception as exc:
            raise RuntimeError(f"RoMa dependencies are not installed correctly: {exc}") from exc
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("RoMa requested CUDA, but torch.cuda.is_available() is false.")

        model_factory = {"roma_outdoor": roma_outdoor, "roma_indoor": roma_indoor}.get(self.config.model)
        if model_factory is None:
            raise RuntimeError(f"Unsupported RoMa model: {self.config.model}")
        matcher = model_factory(device=self.config.device)
        warp, certainty = matcher.match(str(guidance_image), str(rendered_image), device=self.config.device)
        matches, match_certainty = matcher.sample(warp, certainty)
        if len(matches) == 0:
            raise RuntimeError(f"RoMa produced no correspondences for {object_id}")

        height_a, width_a = image_size(guidance_image)
        height_b, width_b = image_size(rendered_image)
        keypoints_a, keypoints_b = matcher.to_pixel_coordinates(matches, height_a, width_a, height_b, width_b)
        keypoints_a_np = keypoints_a.detach().cpu().numpy().astype("float32")
        keypoints_b_np = keypoints_b.detach().cpu().numpy().astype("float32")
        certainty_np = match_certainty.detach().cpu().numpy().astype("float32")
        keep = certainty_np >= self.config.confidence_threshold
        if int(keep.sum()) < self.config.max_correspondences:
            raise RuntimeError(
                f"RoMa produced only {int(keep.sum())} correspondences above "
                f"{self.config.confidence_threshold}; required {self.config.max_correspondences}"
            )
        indices = np.flatnonzero(keep)[: self.config.max_correspondences]
        target_dir = Path(out_dir) / "correspondences"
        target_dir.mkdir(parents=True, exist_ok=True)
        output_path = target_dir / f"{object_id}.npz"
        np.savez_compressed(
            output_path,
            guidance_xy=keypoints_a_np[indices],
            rendered_xy=keypoints_b_np[indices],
            confidence=certainty_np[indices],
        )
        write_json(
            target_dir / f"{object_id}.json",
            {
                "object_id": object_id,
                "guidance_image": str(guidance_image),
                "rendered_image": str(rendered_image),
                "correspondence_path": str(output_path),
                "count": len(indices),
                "confidence_threshold": self.config.confidence_threshold,
            },
        )
        return output_path


def image_size(path: str | Path) -> tuple[int, int]:
    image = Image.open(path)
    width, height = image.size
    return height, width
