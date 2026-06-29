from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.mujoco_bridge.schemas import SceneIR
from scenethesis_mvp.utils.io import read_json


def derive_mujoco_cameras(scene_ir: SceneIR, config: dict[str, Any]) -> list[dict[str, Any]]:
    width, depth, height = scene_ir.bounds
    target = scene_ir.object_by_id(scene_ir.task.target_object)
    destination = np.asarray(scene_ir.task.destination_position, dtype=float)
    target_pos = np.asarray(target.pose.position, dtype=float)
    robot_pos = np.asarray(scene_ir.robot.base_position, dtype=float)
    task_center = ((target_pos + destination + robot_pos) / 3.0).tolist()
    task_center[2] = max(0.45, float(target_pos[2]) + 0.08)
    scene_center = [width * 0.5, depth * 0.5, height * 0.42]
    span = max(width, depth, height, 3.0)

    cameras = _authored_reference_cameras(scene_ir)
    cameras.extend(
        [
            _camera_record(
                "policy_overview",
                _clamp_position([task_center[0] + 1.35, task_center[1] - 1.35, task_center[2] + 1.25], scene_ir.bounds, margin=0.15),
                task_center,
                "derived_from_task_anchors",
                "estimated",
                "packing_table_01" if scene_ir.task.support_id else scene_ir.task.target_object,
                camera_class="policy",
            ),
            _camera_record(
                "reference_front",
                [scene_center[0] + span * 0.10, -span * 0.72, height * 0.72],
                scene_center,
                "reconstructed_from_prompt2scene_render_algorithm",
                "reconstructed",
                "scene_bounds",
                camera_class="reference",
            ),
            _camera_record(
                "reference_left",
                [-span * 0.72, scene_center[1] - span * 0.12, height * 0.75],
                scene_center,
                "reconstructed_from_prompt2scene_render_algorithm",
                "reconstructed",
                "scene_bounds",
                camera_class="reference",
            ),
            _camera_record(
                "reference_right",
                [width + span * 0.72, scene_center[1] - span * 0.12, height * 0.75],
                scene_center,
                "reconstructed_from_prompt2scene_render_algorithm",
                "reconstructed",
                "scene_bounds",
                camera_class="reference",
            ),
            _camera_record(
                "reference_top_oblique",
                [scene_center[0] + span * 0.18, scene_center[1] - span * 0.25, height + span * 0.95],
                scene_center,
                "reconstructed_from_prompt2scene_render_algorithm",
                "reconstructed",
                "scene_bounds",
                camera_class="reference",
            ),
            _camera_record(
                "report_task_closeup",
                _clamp_position([task_center[0] + 1.05, task_center[1] - 1.25, task_center[2] + 0.75], scene_ir.bounds, margin=0.15),
                task_center,
                "derived_from_task_anchors",
                "estimated",
                scene_ir.task.target_object,
                camera_class="report",
            ),
        ]
    )
    for camera in scene_ir.cameras:
        if camera.id == "wrist_rgb" or any(item["camera_name"] == camera.id for item in cameras):
            continue
        cameras.append(
            _camera_record(
                camera.id,
                [width * 0.5, depth * 0.5, height + 1.6],
                scene_center,
                "derived_from_policy_contract",
                "estimated",
                "scene_bounds",
                fovy=float(camera.fovy_deg),
                camera_class="policy",
            )
        )
    return cameras


def _camera_record(
    name: str,
    pos: list[float],
    target: list[float],
    source: str,
    confidence: str,
    target_anchor: str,
    fovy: float = 58.0,
    camera_class: str = "report",
) -> dict[str, Any]:
    return {
        "camera_name": name,
        "camera_class": camera_class,
        "source": source,
        "pose_confidence": confidence,
        "fov_deg": float(fovy),
        "target_anchor": target_anchor,
        "pos": [round(float(item), 6) for item in pos],
        "target": [round(float(item), 6) for item in target],
        "xyaxes": [round(float(item), 8) for item in _camera_xyaxes(pos, target)],
    }


def _authored_reference_cameras(scene_ir: SceneIR) -> list[dict[str, Any]]:
    candidate = Path(scene_ir.source_run_dir) / "render_camera_manifest.json"
    if not candidate.is_file():
        return []
    try:
        data = read_json(candidate)
    except Exception:
        return []
    cameras: list[dict[str, Any]] = []
    for item in data.get("cameras", []):
        name = str(item.get("camera_name", "reference_camera")).replace("render_", "reference_")
        cameras.append(
            {
                "camera_name": name,
                "camera_class": "reference",
                "source": item.get("source", "prompt2scene_blender_camera"),
                "pose_confidence": item.get("pose_confidence", "authored"),
                "fov_deg": item.get("fov_deg") or 58.0,
                "target_anchor": "scene_bounds",
                "pos": item.get("position"),
                "target": None,
                "xyaxes": None,
                "source_manifest": str(candidate),
                "blender_matrix_world": item.get("matrix_world"),
                "blender_quaternion_wxyz": item.get("quaternion_wxyz"),
                "resolution": item.get("resolution"),
            }
        )
    return cameras


def _camera_xyaxes(pos: list[float], target: list[float]) -> list[float]:
    eye = np.asarray(pos, dtype=float)
    look = np.asarray(target, dtype=float)
    forward = look - eye
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    z_axis = -forward
    up = np.asarray([0.0, 0.0, 1.0], dtype=float)
    x_axis = np.cross(up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(np.linalg.norm(y_axis), 1e-9)
    return [float(item) for item in [*x_axis.tolist(), *y_axis.tolist()]]


def _clamp_position(pos: list[float], bounds: list[float], margin: float) -> list[float]:
    return [
        min(max(float(pos[0]), -bounds[0] * 0.2), bounds[0] * 1.2),
        min(max(float(pos[1]), -bounds[1] * 0.2), bounds[1] * 1.2),
        min(max(float(pos[2]), margin), bounds[2] + 2.5),
    ]
