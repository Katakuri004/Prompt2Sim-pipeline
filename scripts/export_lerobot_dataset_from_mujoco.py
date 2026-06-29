from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.lerobot_bridge.dataset_export import export_lerobot_dataset_from_raw_demos


def main() -> None:
    parser = argparse.ArgumentParser(description="Export accepted MuJoCo demos to a LeRobot dataset.")
    parser.add_argument("--raw-demo-root", default="data/lerobot_cache/raw_demos/warehouse_scanner_v001")
    parser.add_argument("--out", default="data/lerobot_cache/datasets/warehouse_scanner_v001")
    parser.add_argument("--repo-id", default="local/warehouse_scanner_v001")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--canonical-only", action="store_true", help="Validate and write canonical export without requiring LeRobot.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report = export_lerobot_dataset_from_raw_demos(
        raw_demo_root=args.raw_demo_root,
        output_dir=args.out,
        repo_id=args.repo_id,
        fps=args.fps,
        canonical_only=args.canonical_only,
        overwrite=args.overwrite,
    )
    print(f"status: {report['status']}")
    print(f"episode_count: {report['episode_count']}")
    print(f"state_dim: {report['state_dim']}")
    print(f"action_dim: {report['action_dim']}")
    print(f"manifest: {Path(args.out) / 'meta' / 'scenethesis_lerobot_export_manifest.json'}")


if __name__ == "__main__":
    main()
