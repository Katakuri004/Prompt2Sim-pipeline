from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.pipeline.run_pipeline import run_pipeline
from scenethesis_mvp.pipeline.validate_outputs import validate_run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Scenethesis MVP demo pipeline.")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--repair-rounds", type=int, default=2)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    result = run_pipeline(
        prompt=args.prompt,
        out_dir=args.out,
        repair_rounds=args.repair_rounds,
        config_path=args.config,
    )
    validation = validate_run(result.out_dir)
    print(f"Output directory: {result.out_dir}")
    print(f"Objects: {result.metrics.object_count}")
    print(f"Collisions: {result.metrics.collision_count}")
    print(f"Floating: {result.metrics.floating_count}")
    if result.mesh_metrics is not None:
        print(f"Mesh collisions: {result.mesh_metrics.mesh_collision_count}")
        print(f"Mesh support failures: {result.mesh_metrics.support_failure_count}")
    print(f"Renderer: {result.render.renderer}")
    print(f"Validation: {'ok' if validation['ok'] else 'failed'}")
    if not validation["ok"]:
        print(f"Validation details: {validation}")
        sys.exit(1)


if __name__ == "__main__":
    main()
