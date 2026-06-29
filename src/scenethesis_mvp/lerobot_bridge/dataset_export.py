from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.utils.io import read_json, write_json

CAMERA_KEYS = ("overhead_rgb", "wrist_rgb")
TASK_INSTRUCTION = "Pick barcode_scanner_01 from packing_table_01 and place it at the sampled destination."


def export_lerobot_dataset_from_raw_demos(
    *,
    raw_demo_root: str | Path,
    output_dir: str | Path,
    repo_id: str,
    fps: int = 20,
    canonical_only: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    source = Path(raw_demo_root)
    target = Path(output_dir)
    if not source.is_dir():
        raise FileNotFoundError(f"Raw demo root does not exist: {source}")
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"Dataset output already exists: {target}")
        shutil.rmtree(target)
    episodes = _load_demo_episodes(source)
    if not episodes:
        raise RuntimeError(f"No accepted demo traces found under {source / 'episodes'}")
    state_dim, action_dim = _validate_episode_shapes(episodes)
    if canonical_only:
        return _write_canonical_export(
            raw_demo_root=source,
            output_dir=target,
            repo_id=repo_id,
            fps=fps,
            episodes=episodes,
            state_dim=state_dim,
            action_dim=action_dim,
        )
    _write_with_lerobot_api(
        raw_demo_root=source,
        output_dir=target,
        repo_id=repo_id,
        fps=fps,
        episodes=episodes,
        state_dim=state_dim,
        action_dim=action_dim,
    )
    report = _export_report(
        output_dir=target,
        repo_id=repo_id,
        fps=fps,
        episodes=episodes,
        state_dim=state_dim,
        action_dim=action_dim,
        status="lerobot_exported",
    )
    write_json(target / "meta" / "scenethesis_lerobot_export_manifest.json", report)
    return report


def _load_demo_episodes(raw_demo_root: Path) -> list[dict[str, Any]]:
    episodes = []
    for trace_path in sorted((raw_demo_root / "episodes").glob("*_rollout_state_trace.json")):
        payload = read_json(trace_path)
        frames = payload.get("trace", [])
        if not frames:
            continue
        episodes.append({"trace_path": trace_path, "frames": frames})
    return episodes


def _validate_episode_shapes(episodes: list[dict[str, Any]]) -> tuple[int, int]:
    state_dim: int | None = None
    action_dim: int | None = None
    for episode in episodes:
        for frame in episode["frames"]:
            state = _state_vector(frame)
            action = _action_vector(frame)
            if state_dim is None:
                state_dim = len(state)
            if action_dim is None:
                action_dim = len(action)
            if len(state) != state_dim:
                raise RuntimeError(f"Inconsistent observation.state shape in {episode['trace_path']}")
            if len(action) != action_dim:
                raise RuntimeError(f"Inconsistent action shape in {episode['trace_path']}")
            if len(action) not in {8, 9}:
                raise RuntimeError(
                    f"LeRobot Phase 1 expects an 8D real-Panda or 9D legacy joint-position command, got {len(action)} in {episode['trace_path']}. "
                    "Record traces with actuator_ctrl available from the MuJoCo adapter."
                )
            rgb_paths = frame.get("rgb_paths", {})
            missing = [camera for camera in CAMERA_KEYS if not rgb_paths.get(camera)]
            if missing:
                raise RuntimeError(
                    f"Frame {frame.get('step')} in {episode['trace_path']} is missing RGB images for {missing}. "
                    "Record demos with --render-rgb."
                )
    if state_dim is None or action_dim is None:
        raise RuntimeError("No valid frames found in accepted demos.")
    return state_dim, action_dim


def _write_canonical_export(
    *,
    raw_demo_root: Path,
    output_dir: Path,
    repo_id: str,
    fps: int,
    episodes: list[dict[str, Any]],
    state_dim: int,
    action_dim: int,
) -> dict[str, Any]:
    meta_dir = output_dir / "meta"
    canonical_dir = output_dir / "scenethesis_canonical"
    image_dir = canonical_dir / "images"
    meta_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    episode_records: list[dict[str, Any]] = []
    for episode_index, episode in enumerate(episodes):
        frame_records = []
        for frame_index, frame in enumerate(episode["frames"]):
            copied_images = _copy_frame_images(raw_demo_root, image_dir, episode_index, frame_index, frame)
            frame_records.append(
                {
                    "episode_index": episode_index,
                    "frame_index": frame_index,
                    "timestamp": float(frame.get("time_s", frame_index / fps)),
                    "observation.state": _state_vector(frame).tolist(),
                    "action": _action_vector(frame).tolist(),
                    "task": _task(frame),
                    "images": copied_images,
                    "source_step": frame.get("step"),
                }
            )
        episode_path = canonical_dir / f"episode_{episode_index:06d}.json"
        write_json(episode_path, {"frames": frame_records})
        episode_records.append(
            {
                "episode_index": episode_index,
                "frame_count": len(frame_records),
                "source_trace_path": str(episode["trace_path"]),
                "canonical_path": str(episode_path),
            }
        )
    report = {
        **_export_report(
            output_dir=output_dir,
            repo_id=repo_id,
            fps=fps,
            episodes=episodes,
            state_dim=state_dim,
            action_dim=action_dim,
            status="canonical_exported",
        ),
        "episodes": episode_records,
    }
    write_json(meta_dir / "scenethesis_lerobot_export_manifest.json", report)
    return report


def _export_report(
    *,
    output_dir: Path,
    repo_id: str,
    fps: int,
    episodes: list[dict[str, Any]],
    state_dim: int,
    action_dim: int,
    status: str,
) -> dict[str, Any]:
    return {
        "status": status,
        "repo_id": repo_id,
        "output_dir": str(output_dir),
        "fps": int(fps),
        "episode_count": len(episodes),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "camera_keys": list(CAMERA_KEYS),
        "task": TASK_INSTRUCTION,
    }


def _write_with_lerobot_api(
    *,
    raw_demo_root: Path,
    output_dir: Path,
    repo_id: str,
    fps: int,
    episodes: list[dict[str, Any]],
    state_dim: int,
    action_dim: int,
) -> None:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # pragma: no cover - depends on optional external stack.
        raise RuntimeError(
            "LeRobot is not installed or its dataset API is unavailable. Install a LeRobot build that exposes "
            "lerobot.datasets.lerobot_dataset.LeRobotDataset, or rerun with --canonical-only for schema validation."
        ) from exc

    features = {
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (action_dim,), "names": ["action"]},
        "observation.images.overhead_rgb": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist_rgb": {
            "dtype": "image",
            "shape": (224, 224, 3),
            "names": ["height", "width", "channel"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        fps=int(fps),
        robot_type="scenethesis_panda_mujoco",
        features=features,
        use_videos=True,
    )
    for episode in episodes:
        for frame in episode["frames"]:
            rgb_paths = frame.get("rgb_paths", {})
            dataset.add_frame(
                {
                    "observation.state": _state_vector(frame).astype(np.float32),
                    "observation.images.overhead_rgb": str(raw_demo_root / rgb_paths["overhead_rgb"]),
                    "observation.images.wrist_rgb": str(raw_demo_root / rgb_paths["wrist_rgb"]),
                    "action": _action_vector(frame).astype(np.float32),
                    "task": _task(frame),
                }
            )
        dataset.save_episode()
    if hasattr(dataset, "finalize"):
        dataset.finalize()


def _copy_frame_images(
    raw_demo_root: Path,
    image_dir: Path,
    episode_index: int,
    frame_index: int,
    frame: dict[str, Any],
) -> dict[str, str]:
    copied: dict[str, str] = {}
    for camera in CAMERA_KEYS:
        source = raw_demo_root / str(frame.get("rgb_paths", {}).get(camera, ""))
        if not source.is_file():
            raise FileNotFoundError(f"Missing source image for {camera}: {source}")
        destination = image_dir / f"episode_{episode_index:06d}" / camera / f"frame_{frame_index:06d}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied[f"observation.images.{camera}"] = destination.relative_to(image_dir.parents[1]).as_posix()
    return copied


def _state_vector(frame: dict[str, Any]) -> np.ndarray:
    target = _pose_position(frame.get("target_pose"))
    destination = np.asarray(frame.get("destination_position", [0.0, 0.0, 0.0]), dtype=float)
    return np.concatenate(
        [
            np.asarray(frame.get("proprio", []), dtype=float).reshape(-1),
            np.asarray(frame.get("ee_position", [0.0, 0.0, 0.0]), dtype=float).reshape(-1),
            target,
            destination.reshape(-1),
        ]
    ).astype(np.float32)


def _action_vector(frame: dict[str, Any]) -> np.ndarray:
    if "actuator_ctrl" in frame:
        return np.asarray(frame["actuator_ctrl"], dtype=float).reshape(-1).astype(np.float32)
    return np.asarray(frame.get("action", []), dtype=float).reshape(-1).astype(np.float32)


def _pose_position(pose: Any) -> np.ndarray:
    if isinstance(pose, dict) and "position" in pose:
        return np.asarray(pose["position"], dtype=float).reshape(-1)
    return np.zeros(3, dtype=float)


def _task(frame: dict[str, Any]) -> str:
    value = str(frame.get("language_instruction") or "").strip()
    return value or TASK_INSTRUCTION
