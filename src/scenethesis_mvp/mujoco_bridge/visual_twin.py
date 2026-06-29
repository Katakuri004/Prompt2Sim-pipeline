from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from scenethesis_mvp.mujoco_bridge.schemas import SceneIR
from scenethesis_mvp.render.blender_runner import resolve_blender_path
from scenethesis_mvp.utils.io import write_json


def render_blender_visual_twin(
    scene_ir: SceneIR,
    out_dir: str | Path,
    state_trace_path: str | Path,
    config: dict[str, Any],
    blender_path: str | None = None,
) -> dict[str, Any]:
    target = Path(out_dir).resolve()
    report_path = target / "visual_twin_report.json"
    blender = resolve_blender_path(blender_path)
    if not blender:
        report = {
            "ok": False,
            "status": "blocked",
            "renderer": "blender",
            "reason": "Blender executable not found. Install Blender or set BLENDER_PATH.",
            "source_scene_glb": scene_ir.source_scene_glb,
            "target_object": scene_ir.task.target_object,
            "benchmark_visual_artifact": None,
            "mujoco_debug_video_is_benchmark": False,
        }
        write_json(report_path, report)
        return report

    visual_cfg = config.get("visual_twin", {})
    viz_cfg = config.get("visualization", {})
    frames_dir = target / "visual_twin_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    payload_path = target / "visual_twin_payload.json"
    payload = {
        "source_scene_glb": scene_ir.source_scene_glb,
        "source_run_dir": scene_ir.source_run_dir,
        "state_trace_path": str(state_trace_path),
        "coordinate_manifest_path": str(target / "coordinate_manifest.json"),
        "camera_manifest_path": str(target / "camera_manifest.json"),
        "entity_manifest_path": str(target / "entity_manifest.json"),
        "frames_dir": str(frames_dir),
        "target_object": scene_ir.task.target_object,
        "support_id": scene_ir.task.support_id,
        "destination_position": scene_ir.task.destination_position,
        "camera_name": str(visual_cfg.get("camera", "report_task_closeup")),
        "resolution": [int(item) for item in visual_cfg.get("resolution", viz_cfg.get("resolution", [960, 720]))],
        "frame_stride": int(visual_cfg.get("frame_stride", 1)),
        "max_frames": int(visual_cfg.get("max_frames", 240)),
    }
    write_json(payload_path, payload)
    script = Path(__file__).resolve().with_name("blender_visual_twin_script.py")
    command = [
        blender,
        "--background",
        "--python",
        str(script),
        "--",
        "--input",
        str(payload_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        report = {
            "ok": False,
            "status": "error",
            "renderer": "blender",
            "reason": "Blender visual twin render failed.",
            "source_scene_glb": scene_ir.source_scene_glb,
            "target_object": scene_ir.task.target_object,
            "returncode": exc.returncode,
            "stdout_tail": (exc.stdout or "")[-4000:],
            "stderr_tail": (exc.stderr or "")[-4000:],
            "benchmark_visual_artifact": None,
            "mujoco_debug_video_is_benchmark": False,
        }
        write_json(report_path, report)
        return report

    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        report = {
            "ok": False,
            "status": "error",
            "renderer": "blender",
            "reason": "Blender completed without producing visual twin frames.",
            "source_scene_glb": scene_ir.source_scene_glb,
            "target_object": scene_ir.task.target_object,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
            "benchmark_visual_artifact": None,
            "mujoco_debug_video_is_benchmark": False,
        }
        write_json(report_path, report)
        return report

    import imageio.v2 as imageio

    video_path = target / "visual_twin_blender.mp4"
    frames = [imageio.imread(path) for path in frame_paths]
    imageio.mimsave(video_path, frames, fps=int(visual_cfg.get("fps", viz_cfg.get("fps", 20))))
    report = {
        "ok": True,
        "status": "rendered",
        "renderer": "blender",
        "source_scene_glb": scene_ir.source_scene_glb,
        "target_object": scene_ir.task.target_object,
        "semantic_replacement": {
            "static_target_mesh_hidden": True,
            "dynamic_target_visual_bound_to_trace": True,
            "visible_target_instances": 1,
        },
        "frame_count": len(frame_paths),
        "frames_dir": str(frames_dir),
        "benchmark_visual_artifact": str(video_path),
        "mujoco_debug_video_is_benchmark": False,
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }
    write_json(report_path, report)
    return report
