from __future__ import annotations

from pathlib import Path
from typing import Any

from scenethesis_mvp.mujoco_bridge.schemas import SceneIR
from scenethesis_mvp.utils.io import write_json


def write_coordinate_manifest(scene_ir: SceneIR, out_dir: str | Path) -> dict[str, Any]:
    transform = {}
    visual_transform = scene_ir.visual_scene.get("transform") if isinstance(scene_ir.visual_scene, dict) else None
    if isinstance(visual_transform, dict):
        transform = dict(visual_transform)
    manifest = {
        "canonical_world": "Z_UP_METERS",
        "units_scale": float(transform.get("unit_scale", 1.0)),
        "glb_to_canonical": {
            "source_coordinate_system": transform.get("source_coordinate_system", "unknown"),
            "target_coordinate_system": "MuJoCo Z-up",
            "applied_transform": transform.get("applied_transform", "unknown"),
            "matrix": transform.get("matrix"),
            "determinant": transform.get("determinant"),
            "normal_orientation_valid": transform.get("normal_orientation_valid"),
        },
        "usd_to_canonical": {
            "source": scene_ir.source_scene_usd,
            "status": "verified" if transform.get("usd_axis_consistent") else "unverified_or_missing",
            "usd_axis_consistent": bool(transform.get("usd_axis_consistent", False)),
        },
        "mujoco_to_canonical": {
            "applied_transform": "identity",
            "units": "meters",
            "up_axis": "Z",
        },
        "scene_origin_transform": {
            "translation": [0.0, 0.0, 0.0],
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "robot_base_transform": {
            "body": "panda_base",
            "position": scene_ir.robot.base_position,
            "yaw_rad": scene_ir.robot.base_yaw_rad,
        },
        "validation_anchors": _validation_anchors(scene_ir),
    }
    write_json(Path(out_dir) / "coordinate_manifest.json", manifest)
    return manifest


def _validation_anchors(scene_ir: SceneIR) -> dict[str, Any]:
    anchors: dict[str, Any] = {
        "target": _object_anchor(scene_ir, scene_ir.task.target_object),
        "destination": {
            "position": scene_ir.task.destination_position,
            "support_id": scene_ir.task.support_id,
        },
    }
    if scene_ir.task.support_id:
        anchors["support"] = _object_anchor(scene_ir, scene_ir.task.support_id)
    for category in ("shelf", "table", "barrier"):
        for obj in scene_ir.objects:
            if obj.category == category:
                anchors.setdefault(category, _object_anchor(scene_ir, obj.id))
                break
    return anchors


def _object_anchor(scene_ir: SceneIR, object_id: str) -> dict[str, Any]:
    obj = scene_ir.object_by_id(object_id)
    return {
        "entity_id": obj.id,
        "category": obj.category,
        "position": obj.pose.position,
        "quaternion_wxyz": obj.pose.quaternion,
        "dimensions": obj.dimensions,
    }
