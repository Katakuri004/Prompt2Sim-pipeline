from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from scenethesis_mvp.schemas.depth import CameraIntrinsics, DepthResult
from scenethesis_mvp.utils.io import write_json


@dataclass(frozen=True)
class DepthProConfig:
    repo_dir: Path
    checkpoint_dir: Path
    device: str = "cuda"


class DepthProRunner:
    def __init__(self, config: DepthProConfig):
        self.config = config

    def estimate(
        self,
        image_path: str | Path,
        out_dir: str | Path,
    ) -> DepthResult:
        self._validate_paths()
        target_dir = Path(out_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            import depth_pro
            from depth_pro.depth_pro import DEFAULT_MONODEPTH_CONFIG_DICT
        except Exception as exc:
            raise RuntimeError(f"Depth Pro dependencies are not installed correctly: {exc}") from exc

        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Depth Pro requested CUDA, but torch.cuda.is_available() is false.")

        checkpoint_path = self._checkpoint_path()
        depth_config = replace(DEFAULT_MONODEPTH_CONFIG_DICT, checkpoint_uri=str(checkpoint_path))
        model, transform = depth_pro.create_model_and_transforms(
            config=depth_config,
            device=torch.device(self.config.device),
        )
        model = model.to(self.config.device)
        model.eval()
        image, _, f_px = depth_pro.load_rgb(str(image_path))
        transformed = transform(image)
        if hasattr(transformed, "to"):
            transformed = transformed.to(self.config.device)
        with torch.no_grad():
            prediction = model.infer(transformed, f_px=f_px)
        depth = prediction["depth"].detach().cpu().numpy().astype("float32")
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise RuntimeError("Depth Pro returned an invalid depth map.")

        depth_path = target_dir / "depth.npy"
        preview_path = target_dir / "depth_preview.png"
        np.save(depth_path, depth)
        save_depth_preview(depth, preview_path)

        height, width = depth.shape
        focal = float(prediction["focallength_px"].detach().cpu().item())
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=focal,
            fy=focal,
            cx=(width - 1) * 0.5,
            cy=(height - 1) * 0.5,
        )
        result = DepthResult(
            image_path=str(image_path),
            depth_path=str(depth_path),
            preview_path=str(preview_path),
            intrinsics=intrinsics,
            min_depth_m=float(np.nanmin(depth)),
            max_depth_m=float(np.nanmax(depth)),
        )
        write_json(target_dir / "depth.json", result)
        write_json(target_dir / "camera_intrinsics.json", intrinsics)
        return result

    def _validate_paths(self) -> None:
        missing = []
        if not self.config.repo_dir.exists():
            missing.append(f"Depth Pro repo: {self.config.repo_dir}")
        if not self.config.checkpoint_dir.exists():
            missing.append(f"Depth Pro checkpoints: {self.config.checkpoint_dir}")
        if missing:
            raise RuntimeError("Missing Depth Pro files:\n- " + "\n- ".join(missing))
        self._checkpoint_path()

    def _checkpoint_path(self) -> Path:
        preferred = self.config.checkpoint_dir / "depth_pro.pt"
        if preferred.is_file():
            return preferred.resolve()
        candidates = sorted(self.config.checkpoint_dir.glob("*.pt"))
        if len(candidates) == 1:
            return candidates[0].resolve()
        raise RuntimeError(f"Depth Pro checkpoint file was not found in {self.config.checkpoint_dir}")


def save_depth_preview(depth: np.ndarray, output_path: str | Path) -> None:
    lo = float(np.percentile(depth, 2))
    hi = float(np.percentile(depth, 98))
    if hi <= lo:
        raise RuntimeError("Depth Pro preview cannot be created because depth range is degenerate.")
    normalized = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    preview = (255.0 * (1.0 - normalized)).astype("uint8")
    Image.fromarray(preview, mode="L").save(output_path)
