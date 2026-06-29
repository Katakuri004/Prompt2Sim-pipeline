from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.utils.io import write_json


class EpisodeRecorder:
    def __init__(
        self,
        out_dir: str | Path,
        episode_index: int,
        enabled: bool,
        camera_id: str,
        resolution: tuple[int, int],
        fps: int,
        frame_stride: int,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.episode_index = int(episode_index)
        self.enabled = bool(enabled)
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = int(fps)
        self.frame_stride = max(1, int(frame_stride))
        self.frames: list[np.ndarray] = []
        self.trace: list[dict[str, Any]] = []
        self.state_trace: list[dict[str, Any]] = []
        self.episode_dir = self.out_dir / "episodes"
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir = self.episode_dir / f"episode_{self.episode_index:03d}_rgb"

    def maybe_capture(self, env, step: int) -> None:
        if not self.enabled or step % self.frame_stride != 0:
            return
        width, height = self.resolution
        self.frames.append(env.render_camera(self.camera_id, width, height))

    def record_step(self, step: int, action: Any, metrics: dict[str, Any], observation: dict[str, Any], env: Any | None = None) -> None:
        state = observation.get("state", {})
        rgb_paths = self._write_rgb_observation(step, observation)
        self.trace.append(
            {
                "step": step,
                "action": _jsonable(action),
                "metrics": _jsonable(metrics),
                "rgb_paths": rgb_paths,
                "state": {
                    "ee_position": _jsonable(state.get("ee_position")),
                    "target_position": _jsonable(state.get("target_position")),
                    "destination_position": _jsonable(state.get("destination_position")),
                },
            }
        )
        if env is not None and hasattr(env, "dense_state_snapshot"):
            snapshot = _jsonable(env.dense_state_snapshot(step, action, metrics))
            snapshot["rgb_paths"] = rgb_paths
            self.state_trace.append(snapshot)

    def write(self) -> dict[str, str]:
        episode_name = f"episode_{self.episode_index:03d}"
        artifacts: dict[str, str] = {}
        trace_path = self.episode_dir / f"{episode_name}_trace.json"
        write_json(trace_path, {"camera": self.camera_id, "trace": self.trace})
        artifacts["trace_path"] = str(trace_path)
        state_trace_path = self.episode_dir / f"{episode_name}_rollout_state_trace.json"
        state_payload = {
            "camera": self.camera_id,
            "episode": self.episode_index,
            "trace": self.state_trace,
        }
        write_json(state_trace_path, state_payload)
        write_json(self.out_dir / "rollout_state_trace.json", state_payload)
        artifacts["state_trace_path"] = str(state_trace_path)
        artifacts["rollout_state_trace_path"] = str(self.out_dir / "rollout_state_trace.json")
        if self.enabled:
            if not self.frames:
                raise RuntimeError("Video recording was enabled, but no MuJoCo frames were captured.")
            import imageio.v2 as imageio

            snapshot_path = self.episode_dir / f"{episode_name}_first_frame.png"
            video_path = self.episode_dir / f"{episode_name}.mp4"
            imageio.imwrite(snapshot_path, self.frames[0])
            imageio.mimsave(video_path, self.frames, fps=self.fps)
            artifacts["snapshot_path"] = str(snapshot_path)
            artifacts["video_path"] = str(video_path)
        return artifacts

    def _write_rgb_observation(self, step: int, observation: dict[str, Any]) -> dict[str, str]:
        rgb = observation.get("rgb", {})
        if not isinstance(rgb, dict) or not rgb:
            return {}
        import imageio.v2 as imageio

        paths: dict[str, str] = {}
        for camera_id, frame in sorted(rgb.items()):
            array = np.asarray(frame)
            if array.ndim != 3 or array.shape[2] < 3:
                raise RuntimeError(f"RGB observation for camera {camera_id} is not an HWC image.")
            target = self.rgb_dir / str(camera_id) / f"frame_{int(step):06d}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(target, array[:, :, :3].astype(np.uint8))
            paths[str(camera_id)] = target.relative_to(self.out_dir).as_posix()
        return paths


def _jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
