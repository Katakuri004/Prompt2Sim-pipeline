from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import trimesh

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.utils.io import write_json


ASSET_ID = "derived_open_stainless_worktable_01"
SOURCE_ID = "hf_metal_stainless_table_01"


def main() -> None:
    parser = argparse.ArgumentParser(description="Derive an open-underframe worktable from the licensed stainless table.")
    parser.add_argument(
        "--source",
        default="assets/library/hf_simready/hf_metal_stainless_table_01/hf_metal_stainless_table_01.glb",
    )
    parser.add_argument(
        "--out",
        default=f"assets/library/derived/{ASSET_ID}.glb",
    )
    parser.add_argument(
        "--metadata",
        default="assets/manifests/derived_warehouse_assets.json",
    )
    args = parser.parse_args()

    source = resolve(args.source)
    output = resolve(args.out)
    metadata_path = resolve(args.metadata)
    if not source.is_file():
        raise RuntimeError(f"Source worktable GLB is missing: {source}")

    scene = trimesh.load(source, force="scene")
    body_names = [name for name in scene.geometry if "MetalStainlessTable" in name and "Body" in name]
    if len(body_names) != 1:
        raise RuntimeError(f"Expected exactly one stainless table body geometry, found: {body_names}")
    body_name = body_names[0]
    body = scene.geometry[body_name]
    components = list(body.split(only_watertight=False))
    kept = []
    removed = []
    for component in components:
        y_min, y_max = (float(component.bounds[0, 1]), float(component.bounds[1, 1]))
        remove_lower_shelf_component = y_min > 10.0 and y_max < 25.0
        if remove_lower_shelf_component:
            removed.append(component)
        else:
            kept.append(component)
    if len(removed) != 10:
        raise RuntimeError(
            f"Source topology changed; expected 10 lower-shelf components, found {len(removed)}. Refusing derivation."
        )
    scene.geometry[body_name] = trimesh.util.concatenate(kept)
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output)

    derived = trimesh.load(output, force="scene")
    dimensions = [round(float(value), 4) for value in derived.extents]
    expected = [1.7999, 0.85, 0.6989]
    if any(abs(actual - target) > 0.002 for actual, target in zip(dimensions, expected)):
        raise RuntimeError(f"Derived worktable dimensions changed unexpectedly: {dimensions} != {expected}")

    write_json(
        metadata_path,
        {
            "assets": [
                {
                    "id": ASSET_ID,
                    "source_asset_id": SOURCE_ID,
                    "source_glb": str(source.relative_to(ROOT)),
                    "output_glb": str(output.relative_to(ROOT)),
                    "source_sha256": sha256(source),
                    "output_sha256": sha256(output),
                    "dimensions_m": dimensions,
                    "derivation": "Removed exactly ten connected lower-shelf and shelf-hardware components; retained source tabletop, frame, and four legs.",
                    "license": "CC-BY-4.0",
                    "attribution": "Derived from NVIDIA PhysicalAI-SimReady-Warehouse-01 SM_MetalStainlessTable_A03_01",
                }
            ]
        },
    )
    print(f"derived asset: {output}")
    print(f"metadata: {metadata_path}")
    print(f"dimensions: {dimensions}")


def resolve(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
