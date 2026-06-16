from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.pipeline.run_faithful_pipeline import run_faithful_pipeline
from scenethesis_mvp.runtime.faithful import validate_faithful_runtime
from scenethesis_mvp.utils.io import read_yaml, write_json
from scenethesis_mvp.utils.paths import resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict paper-faithful Scenethesis pipeline.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="configs/scenethesis_faithful.yaml")
    parser.add_argument("--repair-rounds", type=int, default=None)
    parser.add_argument(
        "--resume-from-existing",
        action="store_true",
        help="Strictly resume from existing coarse_scene_spec/guidance/segmentation/depth artifacts in --out.",
    )
    parser.add_argument("--check-runtime-only", action="store_true")
    args = parser.parse_args()

    if args.check_runtime_only:
        config = read_yaml(resolve_path(args.config, ROOT))
        registry = AssetRegistry.from_yaml(resolve_path(config.get("paths", {}).get("asset_registry"), ROOT))
        report = validate_faithful_runtime(config, ROOT, registry=registry)
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "faithful_runtime_report.json", report)
        print(f"Runtime ok: {report.ok}")
        for check in report.checks:
            status = "ok" if check.ok else "missing"
            print(f"{status}: {check.name}: {check.detail}")
        if not report.ok:
            sys.exit(1)
        return

    result = run_faithful_pipeline(
        prompt=args.prompt,
        out_dir=args.out,
        config_path=args.config,
        repair_rounds=args.repair_rounds,
        resume_from_existing=args.resume_from_existing,
    )
    print(f"Output directory: {result.out_dir}")
    print(f"Objects: {len(result.scene.objects)}")
    print(f"Render: {result.render.render_path}")


if __name__ == "__main__":
    main()
