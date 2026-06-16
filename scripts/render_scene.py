from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.render.blender_runner import render_scene
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import read_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an existing scene_spec.json.")
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--registry", default="configs/asset_registry.yaml")
    args = parser.parse_args()

    scene = SceneSpec.model_validate(read_json(args.scene))
    registry = AssetRegistry.from_yaml(ROOT / args.registry)
    result = render_scene(scene, registry, args.out)
    print(f"render.png: {result.render_path}")
    print(f"scene.glb: {result.glb_path}")
    print(f"renderer: {result.renderer}")


if __name__ == "__main__":
    main()
