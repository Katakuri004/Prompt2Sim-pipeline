from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from scenethesis_mvp.utils.io import read_json, write_json


STRICT_SUCCESS_GATES = (
    "success",
    "grasp_attempted",
    "released_after_grasp",
    "target_lifted",
    "target_placed",
)


def collect_successful_demos(
    *,
    evaluation_dir: str | Path,
    demo_root: str | Path,
    dataset_id: str,
    min_accepted: int = 0,
) -> dict[str, Any]:
    source_root = Path(evaluation_dir)
    target_root = Path(demo_root)
    report_path = source_root / "evaluation_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing evaluation report: {report_path}")
    report = read_json(report_path)
    target_root.mkdir(parents=True, exist_ok=True)
    episodes_dir = target_root / "episodes"
    rgb_dir = target_root / "rgb"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    rgb_dir.mkdir(parents=True, exist_ok=True)

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for episode in report.get("episodes", []):
        ok, reasons = _episode_success_reasons(episode)
        if not ok:
            rejected.append(
                {
                    "source_episode": episode.get("episode"),
                    "trace_path": episode.get("trace_path"),
                    "terminated_reason": episode.get("terminated_reason"),
                    "reasons": reasons,
                }
            )
            continue
        demo_index = len(accepted)
        trace_path = Path(str(episode.get("trace_path", "")))
        if not trace_path.is_absolute():
            trace_path = source_root / trace_path
        if not trace_path.is_file():
            rejected.append(
                {
                    "source_episode": episode.get("episode"),
                    "trace_path": str(trace_path),
                    "terminated_reason": episode.get("terminated_reason"),
                    "reasons": ["missing_state_trace"],
                }
            )
            continue
        demo_path = episodes_dir / f"episode_{demo_index:06d}_rollout_state_trace.json"
        normalized = _copy_trace_with_images(
            source_trace_path=trace_path,
            source_root=source_root,
            target_trace_path=demo_path,
            target_root=target_root,
            demo_index=demo_index,
        )
        accepted.append(
            {
                "demo_episode": demo_index,
                "source_episode": episode.get("episode"),
                "trace_path": str(demo_path),
                "frame_count": len(normalized.get("trace", [])),
                "source_terminated_reason": episode.get("terminated_reason"),
            }
        )

    manifest = {
        "dataset_id": dataset_id,
        "source_evaluation_dir": str(source_root),
        "demo_root": str(target_root),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "accepted": accepted,
        "rejected": rejected,
        "strict_success_gates": list(STRICT_SUCCESS_GATES),
    }
    write_json(target_root / "demo_manifest.json", manifest)
    if len(accepted) < int(min_accepted):
        raise RuntimeError(
            f"Only {len(accepted)} accepted demos were recorded; required {min_accepted}. "
            f"See {target_root / 'demo_manifest.json'} for rejection reasons."
        )
    return manifest


def _episode_success_reasons(episode: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for gate in STRICT_SUCCESS_GATES:
        if not bool(episode.get(gate, False)):
            reasons.append(f"missing_{gate}")
    if int(episode.get("collision_count", 0)) > 0:
        reasons.append("forbidden_or_bad_collision")
    if bool(episode.get("object_drop", False)):
        reasons.append("object_drop")
    if bool(episode.get("workspace_violation", False)):
        reasons.append("workspace_violation")
    terminated = str(episode.get("terminated_reason", ""))
    if terminated and terminated != "success":
        reasons.append(terminated)
    return not reasons, reasons


def _copy_trace_with_images(
    *,
    source_trace_path: Path,
    source_root: Path,
    target_trace_path: Path,
    target_root: Path,
    demo_index: int,
) -> dict[str, Any]:
    payload = read_json(source_trace_path)
    for frame in payload.get("trace", []):
        rgb_paths = frame.get("rgb_paths", {})
        if not isinstance(rgb_paths, dict):
            frame["rgb_paths"] = {}
            continue
        updated: dict[str, str] = {}
        for camera_id, rel_path in sorted(rgb_paths.items()):
            source_image = source_root / str(rel_path)
            if not source_image.is_file():
                raise FileNotFoundError(f"Missing RGB demo image: {source_image}")
            destination = target_root / "rgb" / f"episode_{demo_index:06d}" / str(camera_id) / source_image.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_image, destination)
            updated[str(camera_id)] = destination.relative_to(target_root).as_posix()
        frame["rgb_paths"] = updated
    target_trace_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(target_trace_path, payload)
    return payload
