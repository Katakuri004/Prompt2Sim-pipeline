from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.layout.collision import object_aabb
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.paths import resolve_path
from scenethesis_mvp.mujoco_bridge.schemas import RobotSpec


def place_panda_robot(scene: SceneSpec, registry: AssetRegistry, target_object_id: str, config: dict[str, Any], root: Path) -> RobotSpec:
    robot_cfg = config.get("robot", {})
    target = scene.object_by_id(target_object_id)
    clearance = float(robot_cfg.get("base_clearance_m", 0.72))
    base_size = tuple(float(item) for item in robot_cfg.get("base_size_m", [0.46, 0.46, 0.18]))
    target_x = target.placement.x
    target_y = target.placement.y
    bounds = scene.bounds
    candidate_angles = [
        math.pi,
        -math.pi * 0.75,
        math.pi * 0.75,
        0.0,
        -math.pi / 2,
        math.pi / 2,
        -math.pi * 0.5,
        math.pi * 0.5,
        -math.pi * 0.25,
        math.pi * 0.25,
        -math.pi * 0.125,
        math.pi * 0.125,
        -math.pi * 0.875,
        math.pi * 0.875,
    ]
    candidate_distances = [
        clearance,
        clearance + 0.12,
        clearance + 0.24,
    ]
    visual_only_categories = {str(item) for item in config.get("collision", {}).get("visual_only_categories", [])}
    static_boxes = [
        object_aabb(obj, registry)
        for obj in scene.objects
        if obj.id != target_object_id and obj.category not in visual_only_categories
    ]
    for distance in candidate_distances:
        for angle in candidate_angles:
            x = target_x + math.cos(angle) * distance
            y = target_y + math.sin(angle) * distance
            x = min(max(x, base_size[0] * 0.5), bounds[0] - base_size[0] * 0.5)
            y = min(max(y, base_size[1] * 0.5), bounds[1] - base_size[1] * 0.5)
            if _base_collides((x, y, float(robot_cfg.get("base_height_m", 0.0))), base_size, static_boxes):
                continue
            yaw = math.atan2(target_y - y, target_x - x)
            return RobotSpec(
                id=str(robot_cfg.get("id", "panda")),
                name=str(robot_cfg.get("name", "Franka Panda")),
                mjcf_path=str(resolve_path(robot_cfg.get("mjcf_path", "models/robots/panda/panda.xml"), root)),
                base_position=[round(x, 6), round(y, 6), float(robot_cfg.get("base_height_m", 0.0))],
                base_yaw_rad=round(yaw, 6),
                arm_joint_names=[str(item) for item in robot_cfg.get("arm_joint_names", [])],
                gripper_joint_names=[str(item) for item in robot_cfg.get("gripper_joint_names", [])],
                actuator_names=[str(item) for item in robot_cfg.get("actuator_names", [])],
                home_qpos=[float(item) for item in robot_cfg.get("home_qpos", [])],
                ee_site=str(robot_cfg.get("ee_site", "panda_gripper_site")),
                gripper_max_width_m=float(robot_cfg.get("gripper_max_width_m", 0.08)),
            )
    raise RuntimeError(f"Could not place Panda base near target {target_object_id} without colliding with static scene geometry.")


def _base_collides(position: tuple[float, float, float], size: tuple[float, float, float], boxes: list[Any]) -> bool:
    x, y, z = position
    dx, dy, dz = size
    base_min = (x - dx * 0.5, y - dy * 0.5, z)
    base_max = (x + dx * 0.5, y + dy * 0.5, z + dz)
    for box in boxes:
        if (
            base_min[0] <= box.max_x
            and base_max[0] >= box.min_x
            and base_min[1] <= box.max_y
            and base_max[1] >= box.min_y
            and base_min[2] <= box.max_z
            and base_max[2] >= box.min_z
        ):
            return True
    return False
