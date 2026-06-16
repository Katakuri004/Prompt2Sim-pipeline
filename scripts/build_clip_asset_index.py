from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.assets.clip_index import build_clip_index
from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.utils.paths import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a strict OpenCLIP local asset index.")
    parser.add_argument("--registry", default="configs/warehouse_asset_registry.yaml")
    parser.add_argument("--out", default="assets/indexes/warehouse_clip_index.npz")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="ViT-L-14")
    parser.add_argument("--pretrained", default="openai")
    args = parser.parse_args()

    registry = AssetRegistry.from_yaml(resolve_path(args.registry, ROOT))
    build_clip_index(
        registry=registry,
        output_path=resolve_path(args.out, ROOT),
        device=args.device,
        model_name=args.model,
        pretrained=args.pretrained,
    )
    print(f"wrote: {resolve_path(args.out, ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"failed: {exc}", file=sys.stderr)
        sys.exit(1)
