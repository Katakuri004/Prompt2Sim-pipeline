from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.render.blender_runner import resolve_blender_path
from scenethesis_mvp.schemas.asset import AssetSpec
from scenethesis_mvp.utils.io import write_json
from scenethesis_mvp.utils.paths import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render real mesh asset thumbnails for CLIP indexing.")
    parser.add_argument("--registry", default="configs/warehouse_asset_registry.yaml")
    parser.add_argument("--out-registry", default=None)
    parser.add_argument("--thumbnail-dir", default="assets/thumbnails/warehouse")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--blender-path", default=None)
    args = parser.parse_args()

    registry_path = resolve_path(args.registry, ROOT)
    registry_dir = registry_path.parent
    output_registry = resolve_path(args.out_registry, ROOT) if args.out_registry else registry_path
    thumbnail_dir = resolve_path(args.thumbnail_dir, ROOT)
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    assets = data.get("assets", [])
    render_items = []
    for item in assets:
        asset = AssetSpec(**item)
        mesh_path = asset.resolved_mesh_path(registry_dir)
        if not mesh_path or not mesh_path.is_file():
            continue
        thumbnail_path = thumbnail_dir / f"{asset.id}.png"
        item["thumbnail_path"] = os.path.relpath(thumbnail_path, registry_dir)
        render_items.append(
            {
                "asset_id": asset.id,
                "mesh_path": str(mesh_path),
                "thumbnail_path": str(thumbnail_path),
                "dimensions": asset.dimensions,
            }
        )
    if not render_items:
        raise RuntimeError("No mesh-backed assets were found for thumbnail rendering.")
    payload_path = thumbnail_dir / "thumbnail_input.json"
    write_json(payload_path, {"resolution": [args.resolution, args.resolution], "assets": render_items})

    blender = resolve_blender_path(args.blender_path)
    if not blender:
        raise RuntimeError("Blender executable was not found.")
    script = ROOT / "src" / "scenethesis_mvp" / "render" / "thumbnail_blender_script.py"
    subprocess.run([blender, "--background", "--python", str(script), "--", "--input", str(payload_path)], check=True)
    output_registry.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    print(f"rendered thumbnails: {len(render_items)}")
    print(f"updated registry: {output_registry}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        sys.exit(1)
