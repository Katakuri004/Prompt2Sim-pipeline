from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.mujoco_bridge.schemas import SceneIR
from scenethesis_mvp.utils.io import write_json


def write_visual_identity_report(
    out_dir: str | Path,
    scene_ir: SceneIR,
    model: Any,
    mujoco: Any,
    snapshot_path: str | None = None,
) -> dict[str, Any]:
    target = Path(out_dir)
    frame_report = _frame_report(snapshot_path)
    checks = [
        {
            "name": "warehouse_shell",
            "ok": any(part.entity_id is None for part in scene_ir.static_visual_meshes),
            "detail": "unmapped shell/floor/wall GLB meshes imported as static visual world",
        },
        {
            "name": "packing_table",
            "ok": any(part.entity_id == scene_ir.task.support_id for part in scene_ir.static_visual_meshes),
            "detail": f"support_id={scene_ir.task.support_id}",
        },
        {
            "name": "panda",
            "ok": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_base") >= 0,
            "detail": "panda_base body present",
        },
        {
            "name": "barcode_scanner",
            "ok": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, scene_ir.task.target_object) >= 0
            and bool(scene_ir.object_by_id(scene_ir.task.target_object).visual_parts),
            "detail": f"dynamic target body and visual parts for {scene_ir.task.target_object}",
        },
        {
            "name": "destination_marker",
            "ok": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, scene_ir.task.destination_region) >= 0,
            "detail": f"site={scene_ir.task.destination_region}",
        },
    ]
    report = {
        "ok": all(bool(item["ok"]) for item in checks) and bool(frame_report.get("nonblank", True)),
        "checks": checks,
        "frame": frame_report,
    }
    write_json(target / "visual_identity_report.json", report)
    return report


def _frame_report(snapshot_path: str | None) -> dict[str, Any]:
    if not snapshot_path:
        return {"available": False}
    path = Path(snapshot_path)
    if not path.is_file():
        return {"available": False, "path": str(path)}
    try:
        import imageio.v2 as imageio

        frame = np.asarray(imageio.imread(path))
        if frame.size == 0:
            return {"available": True, "path": str(path), "nonblank": False}
        rgb = frame[..., :3].astype(float)
        green = (rgb[..., 1] > 120) & (rgb[..., 0] < 80) & (rgb[..., 2] < 120)
        return {
            "available": True,
            "path": str(path),
            "nonblank": bool(float(rgb.max()) > float(rgb.min())),
            "mean_rgb": [round(float(item), 3) for item in rgb.reshape(-1, 3).mean(axis=0)],
            "green_destination_like_pixels": int(green.sum()),
        }
    except Exception as exc:
        return {"available": True, "path": str(path), "nonblank": False, "error": str(exc)}
