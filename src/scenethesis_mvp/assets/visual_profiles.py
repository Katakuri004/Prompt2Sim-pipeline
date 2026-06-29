from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.llm.openai_client import OpenAIClient
from scenethesis_mvp.render.blender_runner import resolve_blender_path
from scenethesis_mvp.schemas.asset import AssetSpec
from scenethesis_mvp.schemas.asset_correspondence import AssetVisualProfile, GeneratedAssetDescription
from scenethesis_mvp.utils.io import read_json, read_text, write_json
from scenethesis_mvp.utils.paths import project_root


VIEW_NAMES = ("front", "side", "oblique")
THIN_VIEW_CATEGORIES = {"cable", "floor_marking", "pipe"}


@dataclass(frozen=True)
class AssetProfileConfig:
    profile_dir: Path
    view_dir: Path
    system_prompt_path: Path
    model: str
    max_retries: int = 3
    resolution: int = 512
    blender_path: str | None = None


class AssetVisualProfileStore:
    def __init__(self, config: AssetProfileConfig, client: OpenAIClient | None = None):
        self.config = config
        self.client = client or OpenAIClient()

    def ensure_profiles(self, asset_ids: list[str], registry: AssetRegistry) -> dict[str, AssetVisualProfile]:
        unique_ids = sorted(set(asset_ids))
        assets = [registry.get(asset_id) for asset_id in unique_ids]
        profiles: dict[str, AssetVisualProfile] = {}
        missing: list[AssetSpec] = []
        for asset in assets:
            profile_path = self.profile_path(asset.id)
            if profile_path.is_file():
                profiles[asset.id] = self._load_valid_profile(profile_path, asset)
            else:
                missing.append(asset)
        if not missing:
            return profiles
        if not self.client.configured:
            raise RuntimeError("OPENAI_API_KEY is required to build missing multi-view asset profiles.")
        self._render_missing_views(missing, registry)
        for asset in missing:
            profile = self._generate_profile(asset)
            write_json(self.profile_path(asset.id), profile.model_dump(mode="json"))
            profiles[asset.id] = profile
        return profiles

    def profile_path(self, asset_id: str) -> Path:
        return self.config.profile_dir / f"{asset_id}.json"

    def view_paths(self, asset_id: str) -> dict[str, Path]:
        return {name: self.config.view_dir / asset_id / f"{name}.png" for name in VIEW_NAMES}

    def _load_valid_profile(self, path: Path, asset: AssetSpec) -> AssetVisualProfile:
        try:
            profile = AssetVisualProfile.model_validate(read_json(path))
        except Exception as exc:
            raise RuntimeError(f"Asset visual profile is invalid: {path}: {exc}") from exc
        if profile.asset_id != asset.id:
            raise RuntimeError(f"Asset profile id mismatch in {path}: {profile.asset_id} != {asset.id}")
        if profile.category != asset.category:
            raise RuntimeError(f"Asset profile category mismatch for {asset.id}: {profile.category} != {asset.category}")
        if profile.dimensions_m != asset.dimensions:
            raise RuntimeError(f"Asset profile dimensions are stale for {asset.id}")
        if profile.model != self.config.model:
            raise RuntimeError(
                f"Asset profile model is stale for {asset.id}: {profile.model} != {self.config.model}. "
                "Delete or explicitly rebuild the profile with the configured model."
            )
        expected_views = self.view_paths(asset.id)
        for name, expected in expected_views.items():
            stored = Path(profile.view_paths[name]).resolve()
            if stored != expected.resolve():
                raise RuntimeError(f"Asset profile view path mismatch for {asset.id}/{name}: {stored} != {expected}")
            if not stored.is_file():
                raise RuntimeError(f"Asset profile view is missing for {asset.id}/{name}: {stored}")
            validate_asset_view_image(
                stored,
                asset.id,
                name,
                allow_thin=allows_thin_view(asset.category, asset.dimensions, name),
            )
        validate_profile_view_freshness(path, asset.id, expected_views)
        return profile

    def _render_missing_views(self, assets: list[AssetSpec], registry: AssetRegistry) -> None:
        render_assets: list[dict[str, Any]] = []
        for asset in assets:
            mesh_path = asset.resolved_mesh_path(registry.base_dir)
            if not mesh_path or not mesh_path.is_file():
                raise RuntimeError(f"Cannot build visual profile for {asset.id}; local mesh is missing")
            views = self.view_paths(asset.id)
            if all(path.is_file() for path in views.values()):
                continue
            render_assets.append(
                {
                    "asset_id": asset.id,
                    "category": asset.category,
                    "mesh_path": str(mesh_path),
                    "thumbnail_path": str(self.config.view_dir / asset.id / "thumbnail.png"),
                    "view_paths": {name: str(path) for name, path in views.items()},
                    "dimensions": asset.dimensions,
                }
            )
        if not render_assets:
            return
        blender = resolve_blender_path(self.config.blender_path)
        if not blender:
            raise RuntimeError("Blender executable was not found for multi-view asset profile rendering.")
        self.config.view_dir.mkdir(parents=True, exist_ok=True)
        payload_path = self.config.view_dir / "asset_profile_render_input.json"
        write_json(
            payload_path,
            {
                "resolution": [self.config.resolution, self.config.resolution],
                "assets": render_assets,
            },
        )
        script = project_root() / "src" / "scenethesis_mvp" / "render" / "thumbnail_blender_script.py"
        result = subprocess.run(
            [blender, "--background", "--python", str(script), "--", "--input", str(payload_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "Blender returned no diagnostic output")[-6000:]
            raise RuntimeError(f"Multi-view asset rendering failed with exit code {result.returncode}: {details}")
        for item in render_assets:
            for name, raw_path in item["view_paths"].items():
                if not Path(raw_path).is_file():
                    raise RuntimeError(f"Blender did not produce required asset view {item['asset_id']}/{name}: {raw_path}")
                validate_asset_view_image(
                    Path(raw_path),
                    item["asset_id"],
                    name,
                    allow_thin=allows_thin_view(item["category"], item["dimensions"], name),
                )

    def _generate_profile(self, asset: AssetSpec) -> AssetVisualProfile:
        view_paths = self.view_paths(asset.id)
        payload = {
            "asset_id": asset.id,
            "registry_category": asset.category,
            "registry_name": asset.name,
            "dimensions_m": asset.dimensions,
            "registry_tags": asset.tags,
            "image_order": list(VIEW_NAMES),
            "instruction": (
                "Describe only properties supported by the three renders and registry metadata. "
                "Do not infer hidden mechanisms, mass, or articulation that is not visible."
            ),
        }
        response = self.client.vision_json_multi(
            system_prompt=read_text(self.config.system_prompt_path),
            user_prompt=json.dumps(payload, indent=2),
            image_paths=[view_paths[name] for name in VIEW_NAMES],
            model=self.config.model,
            json_schema=GeneratedAssetDescription.model_json_schema(),
            schema_name="GeneratedAssetDescription",
            max_retries=self.config.max_retries,
            image_detail="high",
        )
        description = GeneratedAssetDescription.model_validate(response)
        return AssetVisualProfile(
            **description.model_dump(),
            asset_id=asset.id,
            category=asset.category,
            dimensions_m=asset.dimensions,
            view_paths={name: str(path.resolve()) for name, path in view_paths.items()},
            model=self.config.model,
        )


def validate_asset_view_image(path: Path, asset_id: str, view_name: str, allow_thin: bool = False) -> None:
    with Image.open(path) as source:
        image = source.convert("RGBA")
    alpha = image.getchannel("A")
    alpha_bbox = alpha.getbbox()
    if alpha_bbox is None:
        raise RuntimeError(f"Asset view is empty/transparent: {asset_id}/{view_name}: {path}")
    foreground_pixels = int(ImageStat.Stat(alpha).sum[0] / 255.0)
    minimum_pixels = 128 if allow_thin else max(256, int(image.width * image.height * 0.005))
    if foreground_pixels < minimum_pixels:
        raise RuntimeError(
            f"Asset view foreground is too small: {asset_id}/{view_name}: "
            f"pixels={foreground_pixels}, minimum={minimum_pixels}: {path}"
        )
    x0, y0, x1, y1 = alpha_bbox
    width = x1 - x0
    height = y1 - y0
    poorly_framed = (
        max(width / image.width, height / image.height) < 0.08
        if allow_thin
        else width < image.width * 0.08 or height < image.height * 0.08
    )
    if poorly_framed:
        raise RuntimeError(f"Asset view is poorly framed: {asset_id}/{view_name}: bbox={alpha_bbox}: {path}")


def allows_thin_view(category: str, dimensions: list[float], view_name: str) -> bool:
    if category in THIN_VIEW_CATEGORIES:
        return True
    if len(dimensions) != 3 or view_name != "side":
        return False
    width, depth, height = dimensions
    return depth / max(width, height, 1e-6) <= 0.12


def validate_profile_view_freshness(
    profile_path: Path,
    asset_id: str,
    view_paths: dict[str, Path],
) -> None:
    profile_mtime = profile_path.stat().st_mtime_ns
    newer_views = [name for name, path in view_paths.items() if path.stat().st_mtime_ns > profile_mtime]
    if newer_views:
        raise RuntimeError(
            f"Asset visual profile is stale for {asset_id}; newer rendered views: {sorted(newer_views)}. "
            "Explicitly rebuild the profile."
        )
