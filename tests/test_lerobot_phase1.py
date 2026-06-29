from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from scenethesis_mvp.lerobot_bridge.dataset_export import export_lerobot_dataset_from_raw_demos
from scenethesis_mvp.lerobot_bridge.demo_acceptance import collect_successful_demos
from scenethesis_mvp.utils.io import read_json, write_json


def test_collect_successful_demos_rejects_failed_policy(tmp_path: Path) -> None:
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    trace_path = eval_dir / "episodes" / "episode_000_rollout_state_trace.json"
    trace_path.parent.mkdir()
    write_json(trace_path, {"trace": []})
    write_json(
        eval_dir / "evaluation_report.json",
        {
            "episodes": [
                {
                    "episode": 0,
                    "success": False,
                    "terminated_reason": "pregrasp_target_motion",
                    "trace_path": str(trace_path),
                    "collision_count": 0,
                    "object_drop": False,
                    "workspace_violation": False,
                }
            ]
        },
    )
    manifest = collect_successful_demos(
        evaluation_dir=eval_dir,
        demo_root=tmp_path / "raw_demos",
        dataset_id="warehouse_scanner_v001",
        min_accepted=0,
    )
    assert manifest["accepted_count"] == 0
    assert manifest["rejected"][0]["reasons"]


def test_export_canonical_lerobot_dataset_from_successful_demo(tmp_path: Path) -> None:
    raw_root = _write_raw_demo(tmp_path)
    out = tmp_path / "dataset"
    report = export_lerobot_dataset_from_raw_demos(
        raw_demo_root=raw_root,
        output_dir=out,
        repo_id="local/test",
        canonical_only=True,
    )
    assert report["status"] == "canonical_exported"
    assert report["episode_count"] == 1
    assert report["action_dim"] == 9
    manifest = read_json(out / "meta" / "scenethesis_lerobot_export_manifest.json")
    assert manifest["camera_keys"] == ["overhead_rgb", "wrist_rgb"]
    assert (out / "scenethesis_canonical" / "episode_000000.json").is_file()


def test_export_requires_rgb_images(tmp_path: Path) -> None:
    raw_root = _write_raw_demo(tmp_path, with_rgb=False)
    with pytest.raises(RuntimeError, match="--render-rgb"):
        export_lerobot_dataset_from_raw_demos(
            raw_demo_root=raw_root,
            output_dir=tmp_path / "dataset",
            repo_id="local/test",
            canonical_only=True,
        )


def _write_raw_demo(tmp_path: Path, with_rgb: bool = True) -> Path:
    raw_root = tmp_path / "raw_demos"
    episodes = raw_root / "episodes"
    episodes.mkdir(parents=True)
    rgb_paths = {}
    if with_rgb:
        for camera in ("overhead_rgb", "wrist_rgb"):
            image_path = raw_root / "rgb" / "episode_000000" / camera / "frame_000000.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(image_path, np.zeros((224, 224, 3), dtype=np.uint8))
            rgb_paths[camera] = image_path.relative_to(raw_root).as_posix()
    write_json(
        episodes / "episode_000000_rollout_state_trace.json",
        {
            "trace": [
                {
                    "step": 0,
                    "time_s": 0.0,
                    "proprio": [0.0] * 15,
                    "ee_position": [0.4, 0.1, 0.5],
                    "target_pose": {"position": [0.5, 0.1, 0.2]},
                    "destination_position": [0.6, 0.1, 0.2],
                    "actuator_ctrl": [0.0] * 9,
                    "rgb_paths": rgb_paths,
                }
            ]
        },
    )
    return raw_root
