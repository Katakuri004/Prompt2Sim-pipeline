from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.assets.visual_profiles import AssetProfileConfig, AssetVisualProfileStore
from scenethesis_mvp.llm.openai_client import OpenAIClient
from scenethesis_mvp.utils.io import read_yaml
from scenethesis_mvp.utils.paths import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and describe strict multi-view asset profiles.")
    parser.add_argument("--config", default="configs/scenethesis_faithful.yaml")
    parser.add_argument("--asset-id", action="append", default=[])
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--all", action="store_true", help="Build profiles for every mesh-backed registry asset.")
    parser.add_argument("--force", action="store_true", help="Explicitly rebuild selected profiles.")
    args = parser.parse_args()
    if not args.all and not args.asset_id and not args.category:
        raise RuntimeError("Select --asset-id, --category, or --all; implicit bulk API calls are not allowed.")

    config = read_yaml(resolve_path(args.config, ROOT))
    paths = config.get("paths", {})
    retrieval = config.get("asset_retrieval", {})
    openai = config.get("openai", {})
    registry = AssetRegistry.from_yaml(resolve_path(paths["asset_registry"], ROOT))
    selected = []
    requested_ids = set(args.asset_id)
    requested_categories = set(args.category)
    for asset in registry.assets:
        mesh = asset.resolved_mesh_path(registry.base_dir)
        if not mesh or not mesh.is_file():
            continue
        if args.all or asset.id in requested_ids or asset.category in requested_categories:
            selected.append(asset.id)
    missing_ids = requested_ids - {asset.id for asset in registry.assets}
    if missing_ids:
        raise RuntimeError("Unknown asset ids: " + ", ".join(sorted(missing_ids)))
    if not selected:
        raise RuntimeError("No mesh-backed assets matched the requested selection.")

    model = os.getenv("OPENAI_VISION_MODEL", str(openai.get("vision_model", "gpt-5.5")))
    store = AssetVisualProfileStore(
        AssetProfileConfig(
            profile_dir=resolve_path(retrieval["profile_dir"], ROOT),
            view_dir=resolve_path(retrieval["view_dir"], ROOT),
            system_prompt_path=resolve_path(paths["asset_profile_prompt"], ROOT),
            model=model,
            max_retries=int(openai.get("max_retries", 3)),
            resolution=int(retrieval.get("profile_resolution", 512)),
            blender_path=config.get("render", {}).get("blender_path"),
        ),
        client=OpenAIClient(),
    )
    if args.force:
        for asset_id in selected:
            profile_path = store.profile_path(asset_id)
            if profile_path.is_file():
                profile_path.unlink()
    profiles = store.ensure_profiles(selected, registry)
    print(f"built or validated profiles: {len(profiles)}")
    print(f"model: {model}")
    print(f"profile_dir: {store.config.profile_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        sys.exit(1)
