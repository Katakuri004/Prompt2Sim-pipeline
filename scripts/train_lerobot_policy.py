from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.artifacts.drive_sync import load_sync_config, push_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a LeRobot ACT policy from the local Phase 1 dataset cache.")
    parser.add_argument("--dataset-repo-id", default="local/warehouse_scanner_v001")
    parser.add_argument("--local-dataset-path", default="data/lerobot_cache/datasets/warehouse_scanner_v001")
    parser.add_argument("--output-dir", default="outputs/lerobot_phase1/checkpoints/act_warehouse_scanner_v001")
    parser.add_argument("--job-name", default="act_warehouse_scanner_v001")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[], help="Additional raw lerobot-train Hydra argument.")
    parser.add_argument("--push-checkpoint", action="store_true")
    parser.add_argument("--checkpoint-artifact-id", default="act_warehouse_scanner_v001")
    parser.add_argument("--sync-config", default="configs/artifact_sync.yaml")
    parser.add_argument("--sync-backend", choices=["rclone", "local"], default=None)
    parser.add_argument("--remote-root", default=None)
    args = parser.parse_args()

    dataset_path = Path(args.local_dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = ROOT / dataset_path
    if not dataset_path.is_dir():
        raise SystemExit(f"Local dataset path does not exist: {dataset_path}")
    if "gdrive:" in str(dataset_path).lower():
        raise SystemExit("Training must run from local SSD/cache, not directly from Google Drive.")
    executable = shutil.which("lerobot-train")
    if executable is None and not args.dry_run:
        raise SystemExit("lerobot-train is not available on PATH. Install LeRobot before running ACT training.")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    command = [
        executable or "lerobot-train",
        f"--dataset.repo_id={args.dataset_repo_id}",
        "--policy.type=act",
        f"--output_dir={output_dir}",
        f"--job_name={args.job_name}",
        f"--policy.device={args.device}",
        *args.extra_arg,
    ]
    print("command:")
    print(" ".join(str(item) for item in command))
    if args.dry_run:
        return
    subprocess.run(command, check=True)
    if not output_dir.is_dir():
        raise SystemExit(f"Training completed but checkpoint output directory was not created: {output_dir}")
    if args.push_checkpoint:
        cfg = load_sync_config(args.sync_config)
        manifest = push_artifact(
            artifact_id=args.checkpoint_artifact_id,
            artifact_type="checkpoint",
            local_path=output_dir,
            config=cfg,
            backend=args.sync_backend,
            remote_root=args.remote_root,
        )
        print(f"checkpoint_sync_status: {manifest.status}")
        print(f"checkpoint_remote_path: {manifest.remote_path}")


if __name__ == "__main__":
    main()
