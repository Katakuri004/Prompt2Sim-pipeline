from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.artifacts.drive_sync import load_sync_config, pull_artifact, push_artifact, verify_artifact


COMMAND_TYPES = {
    "push-dataset": "dataset",
    "pull-dataset": "dataset",
    "push-checkpoint": "checkpoint",
    "pull-checkpoint": "checkpoint",
    "push-eval": "eval",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync LeRobot Phase 1 datasets, checkpoints, and eval artifacts.")
    parser.add_argument(
        "command",
        choices=["push-dataset", "pull-dataset", "push-checkpoint", "pull-checkpoint", "push-eval", "verify"],
    )
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--artifact-type", choices=["dataset", "checkpoint", "eval"], default=None)
    parser.add_argument("--local-path", default=None)
    parser.add_argument("--remote-path", default=None)
    parser.add_argument("--remote-root", default=None)
    parser.add_argument("--config", default="configs/artifact_sync.yaml")
    parser.add_argument("--backend", choices=["rclone", "local"], default=None)
    args = parser.parse_args()

    cfg = load_sync_config(args.config)
    artifact_type = args.artifact_type or COMMAND_TYPES.get(args.command)
    if artifact_type is None:
        raise SystemExit("--artifact-type is required for verify")

    if args.command.startswith("push-"):
        manifest = push_artifact(
            artifact_id=args.artifact_id,
            artifact_type=artifact_type,
            local_path=args.local_path,
            remote_path=args.remote_path,
            config=cfg,
            backend=args.backend,
            remote_root=args.remote_root,
        )
    elif args.command.startswith("pull-"):
        manifest = pull_artifact(
            artifact_id=args.artifact_id,
            artifact_type=artifact_type,
            local_path=args.local_path,
            remote_path=args.remote_path,
            config=cfg,
            backend=args.backend,
            remote_root=args.remote_root,
        )
    else:
        if args.local_path is None:
            raise SystemExit("--local-path is required for verify")
        manifest = verify_artifact(local_path=args.local_path, manifest_name=cfg.manifest_name)

    print(f"artifact_id: {manifest.artifact_id}")
    print(f"artifact_type: {manifest.artifact_type}")
    print(f"status: {manifest.status}")
    print(f"backend: {manifest.backend}")
    print(f"local_path: {manifest.local_path}")
    print(f"remote_path: {manifest.remote_path}")
    print(f"file_count: {manifest.file_count}")
    print(f"byte_size: {manifest.byte_size}")


if __name__ == "__main__":
    main()
