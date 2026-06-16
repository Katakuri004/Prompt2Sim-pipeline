from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.llm.openai_client import OpenAIClient
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import write_json, write_text
from scenethesis_mvp.vision.guidance import build_guidance_prompt, build_retrieval_candidates


@dataclass(frozen=True)
class ImageGuidanceResult:
    guidance_path: Path
    image_metadata: dict[str, Any]
    upsampled_prompt: str
    candidates: list[dict[str, Any]]


class ImageGuidanceGenerator:
    def __init__(
        self,
        client: OpenAIClient | None = None,
        image_model: str = "gpt-image-1",
        max_retries: int = 3,
    ):
        self.client = client or OpenAIClient()
        self.image_model = os.getenv("OPENAI_IMAGE_MODEL", image_model)
        self.max_retries = max_retries

    def run(
        self,
        prompt: str,
        scene: SceneSpec,
        registry: AssetRegistry,
        out_dir: str | Path,
        image_size: str = "1024x1024",
        image_quality: str = "low",
    ) -> ImageGuidanceResult:
        if not self.client.configured:
            raise RuntimeError("OPENAI_API_KEY is required for image guidance generation.")
        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        candidates = build_retrieval_candidates(scene, registry)
        upsampled_prompt = build_guidance_prompt(prompt, scene, registry)
        write_text(target_dir / "upsampled_prompt.txt", upsampled_prompt)
        write_json(target_dir / "retrieval_candidates.json", candidates)
        image_metadata = self.client.generate_image(
            prompt=upsampled_prompt,
            output_path=target_dir / "guidance.png",
            model=self.image_model,
            size=image_size,
            quality=image_quality,
            max_retries=self.max_retries,
        )
        write_json(target_dir / "guidance_image.json", image_metadata)
        return ImageGuidanceResult(
            guidance_path=target_dir / "guidance.png",
            image_metadata=image_metadata,
            upsampled_prompt=upsampled_prompt,
            candidates=candidates,
        )
