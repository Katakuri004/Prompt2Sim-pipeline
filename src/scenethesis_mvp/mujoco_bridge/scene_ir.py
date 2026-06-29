from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.schemas.asset import AssetSpec
from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec
from scenethesis_mvp.utils.io import read_json, read_yaml
from scenethesis_mvp.utils.paths import project_root, resolve_path
from scenethesis_mvp.mujoco_bridge.robot_registry import place_panda_robot
from scenethesis_mvp.mujoco_bridge.schemas import (
    CameraSpec,
    CollisionSpec,
    PhysicsSpec,
    PolicyContract,
    ResetSpec,
    SceneIR,
    SceneIRObject,
    TaskSpec,
)


def load_mujoco_config(config_path: str | Path = "configs/mujoco_eval.yaml") -> dict[str, Any]:
    root = project_root()
    return read_yaml(resolve_path(config_path, root))


def build_scene_ir(
    run_dir: str | Path,
    config_path: str | Path = "configs/mujoco_eval.yaml",
    target_object: str | None = None,
) -> SceneIR:
    root = project_root()
    config = load_mujoco_config(config_path)
    target_dir = Path(run_dir)
    if not target_dir.is_absolute():
        target_dir = root / target_dir
    _require_accepted_run(target_dir)
    scene = SceneSpec.model_validate(read_json(target_dir / "scene_spec.json"))
    registry_path = resolve_path(config.get("paths", {}).get("asset_registry", "configs/warehouse_asset_registry.yaml"), root)
    registry = AssetRegistry.from_yaml(registry_path)
    selected_target = target_object or _default_target_object(scene, registry, config)
    if selected_target not in {obj.id for obj in scene.objects}:
        raise RuntimeError(f"target object is not present in scene: {selected_target}")
    robot = place_panda_robot(scene, registry, selected_target, config, root)
    task = _build_task(scene, selected_target, registry, robot.base_position, config)
    policy = _build_policy_contract(config)
    reset = _build_reset_spec(config)
    objects = [_build_object_ir(obj, registry, selected_target, config) for obj in scene.objects]
    return SceneIR(
        scene_id=scene.scene_id,
        source_run_dir=str(target_dir),
        source_scene_glb=str(target_dir / "scene.glb"),
        source_scene_usd=str(target_dir / "scene.usd") if (target_dir / "scene.usd").is_file() else None,
        bounds=[float(item) for item in scene.bounds],
        objects=objects,
        cameras=policy.observation_cameras,
        robot=robot,
        task=task,
        reset=reset,
        policy=policy,
    )


def _require_accepted_run(run_dir: Path) -> None:
    required = ["scene_spec.json", "scene.glb", "qualification.json"]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise RuntimeError("MuJoCo evaluation requires accepted run artifacts; missing: " + ", ".join(missing))
    qualification = read_json(run_dir / "qualification.json")
    if qualification.get("accepted") is not True:
        raise RuntimeError(f"MuJoCo evaluation requires qualification accepted=true: {run_dir / 'qualification.json'}")


def _default_target_object(scene: SceneSpec, registry: AssetRegistry, config: dict[str, Any]) -> str:
    allowed = {str(item) for item in config.get("task", {}).get("target_categories", ["box", "container", "cylinder", "tool", "scanner"])}
    require_graspable = bool(config.get("task", {}).get("require_graspable_target", True))
    gripper_width = float(config.get("robot", {}).get("gripper_max_width_m", 0.08))
    for obj in scene.objects:
        if obj.role == "anchor" or obj.category not in allowed:
            continue
        if obj.asset_id:
            asset = registry.get(obj.asset_id)
            scaled_xy = [float(asset.dimensions[0]) * obj.placement.scale, float(asset.dimensions[1]) * obj.placement.scale]
            if require_graspable and min(scaled_xy) > gripper_width:
                continue
            if asset.resolved_mesh_path(registry.base_dir) is not None or asset.category in allowed:
                return obj.id
    raise RuntimeError(
        "Could not choose a Panda-graspable default pick target; pass --target-object for explicit validation "
        "or generate a scene containing a small tool/scanner/container within the gripper width."
    )


def _build_object_ir(obj: ObjectSpec, registry: AssetRegistry, target_object: str, config: dict[str, Any]) -> SceneIRObject:
    if not obj.asset_id:
        raise RuntimeError(f"Scene object has no asset_id: {obj.id}")
    asset = registry.get(obj.asset_id)
    scaled_dims = [round(float(item) * obj.placement.scale, 6) for item in asset.dimensions]
    mesh_path = asset.resolved_mesh_path(registry.base_dir)
    mobility = _mobility_for(obj, asset, target_object, config)
    collision = _initial_collision_specs(asset, mobility, scaled_dims, config)
    return SceneIRObject(
        id=obj.id,
        category=obj.category,
        name=obj.name,
        asset_id=obj.asset_id,
        usd_prim=f"/World/{obj.id}",
        source_visual_path=str(mesh_path) if mesh_path else None,
        pose={
            "position": [float(obj.placement.x), float(obj.placement.y), float(obj.placement.z)],
            "quaternion": _yaw_quaternion(obj.placement.yaw_deg),
        },
        dimensions=scaled_dims,
        mobility=mobility,
        physics=_resolve_physics(asset, config),
        collision=collision,
        support_id=obj.parent_id,
    )


def _mobility_for(obj: ObjectSpec, asset: AssetSpec, target_object: str, config: dict[str, Any]) -> str:
    visual_only = {str(item) for item in config.get("collision", {}).get("visual_only_categories", [])}
    if obj.category in visual_only:
        return "visual_only"
    if obj.id == target_object:
        return "dynamic"
    if obj.role == "anchor" or obj.category in {"shelf", "table", "pallet", "cabinet", "barrier"}:
        return "static"
    profile = config.get("physics_profiles", {}).get("categories", {}).get(asset.category, {})
    dynamic_categories = {str(item) for item in config.get("collision", {}).get("dynamic_categories", [])}
    max_volume = float(config.get("collision", {}).get("max_dynamic_volume_m3", 0.35))
    volume = asset.dimensions[0] * asset.dimensions[1] * asset.dimensions[2] * (obj.placement.scale**3)
    if bool(profile.get("dynamic_by_default", False)) and obj.category in dynamic_categories and volume <= max_volume:
        return "dynamic"
    return "static"


def _initial_collision_specs(asset: AssetSpec, mobility: str, dimensions: list[float], config: dict[str, Any]) -> list[CollisionSpec]:
    if mobility == "visual_only":
        return [CollisionSpec(kind="visual_only")]
    primitive_categories = {str(item) for item in config.get("collision", {}).get("primitive_static_categories", [])}
    if asset.category in primitive_categories or mobility == "static":
        return [CollisionSpec(kind="primitive", primitive_type="box", size=[round(item * 0.5, 6) for item in dimensions])]
    return []


def _resolve_physics(asset: AssetSpec, config: dict[str, Any]) -> PhysicsSpec:
    profiles = config.get("physics_profiles", {})
    profile = dict(profiles.get("default", {}))
    profile.update(dict(profiles.get("categories", {}).get(asset.category, {})))
    asset_override = dict(profiles.get("assets", {}).get(asset.id, {}))
    profile.update(asset_override)
    return PhysicsSpec(
        mass_kg=profile.get("mass_kg"),
        density=profile.get("density"),
        friction=[float(item) for item in profile.get("friction", [0.8, 0.02, 0.001])],
        restitution=float(profile.get("restitution", 0.02)),
        confidence=str(profile.get("confidence", "defaulted")),  # type: ignore[arg-type]
    )


def _build_task(
    scene: SceneSpec,
    target_object: str,
    registry: AssetRegistry,
    robot_base: list[float],
    config: dict[str, Any],
) -> TaskSpec:
    target = scene.object_by_id(target_object)
    task_cfg = config.get("task", {})
    dest = _sample_reachable_destination(scene, target, registry, robot_base, config)
    return TaskSpec(
        target_object=target_object,
        destination_position=[round(float(item), 6) for item in dest],
        destination_size=[float(item) for item in task_cfg.get("destination_size_m", [0.12, 0.12, 0.08])],
        support_id=target.parent_id,
        max_position_error_m=float(task_cfg.get("max_position_error_m", 0.05)),
        max_rotation_error_deg=float(task_cfg.get("max_rotation_error_deg", 15.0)),
        stable_steps=int(task_cfg.get("stable_steps", 15)),
        max_pregrasp_target_drift_m=float(task_cfg.get("max_pregrasp_target_drift_m", 0.03)),
        forbidden_contacts=[str(item) for item in task_cfg.get("forbidden_contact_prefixes", [])],
    )


def _sample_reachable_destination(
    scene: SceneSpec,
    target: ObjectSpec,
    registry: AssetRegistry,
    robot_base: list[float],
    config: dict[str, Any],
) -> list[float]:
    task_cfg = config.get("task", {})
    robot_cfg = config.get("robot", {})
    reach = float(robot_cfg.get("reach_m", 0.95))
    reach_margin = float(task_cfg.get("destination_reach_margin_m", 0.05))
    edge_clearance = float(task_cfg.get("support_edge_clearance_m", 0.08))
    object_clearance = float(task_cfg.get("occupied_clearance_m", 0.08))
    target_asset = registry.get(target.asset_id) if target.asset_id else None
    target_dims = [float(item) * target.placement.scale for item in (target_asset.dimensions if target_asset else [0.12, 0.08, 0.08])]
    swept_clearance = object_clearance + max(target_dims[0], target_dims[1]) * 0.40
    support = scene.object_by_id(target.parent_id) if target.parent_id else None
    if support is None or not support.asset_id:
        return _fallback_floor_destination(scene, target, robot_base, reach, config)
    support_asset = registry.get(support.asset_id)
    support_dims = [float(item) * support.placement.scale for item in support_asset.dimensions]
    support_yaw = math.radians(float(support.placement.yaw_deg))
    half_x = max(0.02, support_dims[0] * 0.5 - edge_clearance - target_dims[0] * 0.5)
    half_y = max(0.02, support_dims[1] * 0.5 - edge_clearance - target_dims[1] * 0.5)
    grid_x = int(task_cfg.get("destination_grid_x", 9))
    grid_y = int(task_cfg.get("destination_grid_y", 7))
    occupied = _support_occupied_regions(scene, target, support, registry, support_yaw)
    candidates: list[tuple[float, list[float]]] = []
    target_local = _to_local_xy([target.placement.x, target.placement.y], [support.placement.x, support.placement.y], support_yaw)
    for lx in _linspace(-half_x, half_x, grid_x):
        for ly in _linspace(-half_y, half_y, grid_y):
            distance_from_target = math.hypot(lx - target_local[0], ly - target_local[1])
            if distance_from_target < max(0.14, object_clearance * 1.25):
                continue
            clearance = _candidate_clearance((lx, ly), occupied, object_clearance)
            if clearance < 0:
                continue
            path_clearance = _swept_path_clearance(target_local, (lx, ly), occupied, swept_clearance)
            world_xy = _to_world_xy([lx, ly], [support.placement.x, support.placement.y], support_yaw)
            if _xy_distance(world_xy, robot_base) > max(0.1, reach - reach_margin):
                continue
            edge_score = min(half_x - abs(lx), half_y - abs(ly))
            base_distance = _xy_distance(world_xy, robot_base)
            reach_score = max(0.0, reach - base_distance)
            path_bonus = max(0.0, path_clearance) * 1.5
            path_penalty = max(0.0, -path_clearance) * 3.0
            distance_penalty = max(0.0, distance_from_target - 0.35) * 1.5
            score = clearance + path_bonus + edge_score + min(distance_from_target, 0.35) * 0.10 + reach_score * 3.0 - path_penalty - distance_penalty
            z = support.placement.z + support_dims[2] * 0.5 + target_dims[2] * 0.5 + 0.003
            candidates.append((score, [world_xy[0], world_xy[1], z]))
    if not candidates:
        raise RuntimeError(
            f"Could not sample a reachable destination on {support.id} for {target.id}; "
            "no table-top candidate satisfied edge clearance, occupied-region clearance, and Panda reach."
        )
    candidates.sort(key=lambda item: (-item[0], item[1][0], item[1][1], item[1][2]))
    return candidates[0][1]


def _fallback_floor_destination(
    scene: SceneSpec,
    target: ObjectSpec,
    robot_base: list[float],
    reach: float,
    config: dict[str, Any],
) -> list[float]:
    task_cfg = config.get("task", {})
    offsets = task_cfg.get("destination_offsets_m", [[0.28, 0.0, 0.0], [0.0, 0.28, 0.0], [-0.28, 0.0, 0.0]])
    margin = 0.15
    for offset in offsets:
        dest = [target.placement.x + float(offset[0]), target.placement.y + float(offset[1]), max(0.02, target.placement.z + float(offset[2]))]
        dest[0] = min(max(dest[0], margin), scene.bounds[0] - margin)
        dest[1] = min(max(dest[1], margin), scene.bounds[1] - margin)
        if _xy_distance(dest, robot_base) <= reach:
            return dest
    raise RuntimeError(f"Could not sample a reachable floor destination for {target.id}.")


def _support_occupied_regions(
    scene: SceneSpec,
    target: ObjectSpec,
    support: ObjectSpec,
    registry: AssetRegistry,
    support_yaw: float,
) -> list[tuple[float, float, float, float]]:
    regions: list[tuple[float, float, float, float]] = []
    for obj in scene.children_of(support.id):
        if obj.id == target.id or not obj.asset_id:
            continue
        asset = registry.get(obj.asset_id)
        dims = [float(item) * obj.placement.scale for item in asset.dimensions]
        local = _to_local_xy([obj.placement.x, obj.placement.y], [support.placement.x, support.placement.y], support_yaw)
        regions.append((local[0], local[1], dims[0] * 0.5, dims[1] * 0.5))
    return regions


def _candidate_clearance(candidate: tuple[float, float], occupied: list[tuple[float, float, float, float]], required: float) -> float:
    if not occupied:
        return 1.0
    best = 10.0
    x, y = candidate
    for ox, oy, hx, hy in occupied:
        dx = max(abs(x - ox) - hx, 0.0)
        dy = max(abs(y - oy) - hy, 0.0)
        clearance = math.hypot(dx, dy)
        if abs(x - ox) <= hx and abs(y - oy) <= hy:
            clearance = -1.0
        best = min(best, clearance - required)
    return best


def _swept_path_clearance(
    start: list[float] | tuple[float, float],
    end: tuple[float, float],
    occupied: list[tuple[float, float, float, float]],
    required: float,
) -> float:
    if not occupied:
        return 1.0
    best = 10.0
    for ox, oy, hx, hy in occupied:
        clearance = _segment_rect_distance(start, end, (ox, oy, hx, hy)) - required
        best = min(best, clearance)
    return best


def _segment_rect_distance(
    start: list[float] | tuple[float, float],
    end: list[float] | tuple[float, float],
    rect: tuple[float, float, float, float],
) -> float:
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    ox, oy, hx, hy = rect
    xmin, xmax = ox - hx, ox + hx
    ymin, ymax = oy - hy, oy + hy
    if _point_in_rect((sx, sy), xmin, xmax, ymin, ymax) or _point_in_rect((ex, ey), xmin, xmax, ymin, ymax):
        return 0.0
    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    edges = [
        ((xmin, ymin), (xmin, ymax)),
        ((xmin, ymax), (xmax, ymax)),
        ((xmax, ymax), (xmax, ymin)),
        ((xmax, ymin), (xmin, ymin)),
    ]
    segment = ((sx, sy), (ex, ey))
    if any(_segments_intersect(segment[0], segment[1], edge[0], edge[1]) for edge in edges):
        return 0.0
    distances = [_point_rect_distance((sx, sy), xmin, xmax, ymin, ymax), _point_rect_distance((ex, ey), xmin, xmax, ymin, ymax)]
    distances.extend(_point_segment_distance(corner, segment[0], segment[1]) for corner in corners)
    return min(distances)


def _point_in_rect(point: tuple[float, float], xmin: float, xmax: float, ymin: float, ymax: float) -> bool:
    return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax


def _point_rect_distance(point: tuple[float, float], xmin: float, xmax: float, ymin: float, ymax: float) -> float:
    dx = max(xmin - point[0], 0.0, point[0] - xmax)
    dy = max(ymin - point[1], 0.0, point[1] - ymax)
    return math.hypot(dx, dy)


def _point_segment_distance(point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]) -> float:
    px, py = point
    sx, sy = start
    ex, ey = end
    vx, vy = ex - sx, ey - sy
    denom = vx * vx + vy * vy
    if denom <= 1e-12:
        return math.hypot(px - sx, py - sy)
    t = max(0.0, min(1.0, ((px - sx) * vx + (py - sy) * vy) / denom))
    closest = (sx + t * vx, sy + t * vy)
    return math.hypot(px - closest[0], py - closest[1])


def _segments_intersect(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> bool:
    def orientation(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> float:
        return (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    def on_segment(p: tuple[float, float], q: tuple[float, float], r: tuple[float, float]) -> bool:
        return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

    o1 = orientation(a0, a1, b0)
    o2 = orientation(a0, a1, b1)
    o3 = orientation(b0, b1, a0)
    o4 = orientation(b0, b1, a1)
    if o1 * o2 < 0.0 and o3 * o4 < 0.0:
        return True
    eps = 1e-12
    return (
        abs(o1) <= eps
        and on_segment(a0, b0, a1)
        or abs(o2) <= eps
        and on_segment(a0, b1, a1)
        or abs(o3) <= eps
        and on_segment(b0, a0, b1)
        or abs(o4) <= eps
        and on_segment(b0, a1, b1)
    )


def _to_local_xy(point: list[float], origin: list[float], yaw: float) -> list[float]:
    dx = float(point[0]) - float(origin[0])
    dy = float(point[1]) - float(origin[1])
    c = math.cos(-yaw)
    s = math.sin(-yaw)
    return [dx * c - dy * s, dx * s + dy * c]


def _to_world_xy(point: list[float], origin: list[float], yaw: float) -> list[float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return [float(origin[0]) + point[0] * c - point[1] * s, float(origin[1]) + point[0] * s + point[1] * c]


def _linspace(low: float, high: float, count: int) -> list[float]:
    if count <= 1:
        return [(low + high) * 0.5]
    step = (high - low) / float(count - 1)
    return [low + step * index for index in range(count)]


def _xy_distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _build_policy_contract(config: dict[str, Any]) -> PolicyContract:
    policy_cfg = config.get("policy", {})
    obs_cfg = policy_cfg.get("observation", {})
    action_cfg = policy_cfg.get("action", {})
    bounds = action_cfg.get("bounds", {})
    cameras = [CameraSpec(**item) for item in obs_cfg.get("cameras", [])]
    return PolicyContract(
        observation_cameras=cameras,
        proprio=[str(item) for item in obs_cfg.get("proprio", ["joint_position", "joint_velocity", "gripper_width"])],
        language_instruction=bool(obs_cfg.get("language_instruction", True)),
        action_representation=str(action_cfg.get("representation", "delta_ee_pose_gripper")),  # type: ignore[arg-type]
        action_rate_hz=int(action_cfg.get("rate_hz", 10)),
        translation_bound_m=float(bounds.get("translation_m", 0.03)),
        rotation_bound_rad=float(bounds.get("rotation_rad", 0.15)),
        gripper_bounds=[float(item) for item in bounds.get("gripper", [-1.0, 1.0])],
    )


def _build_reset_spec(config: dict[str, Any]) -> ResetSpec:
    randomization = config.get("randomization", {})
    pose = randomization.get("object_pose", {})
    physics = randomization.get("physics", {})
    return ResetSpec(
        object_xy_noise_m=[float(item) for item in pose.get("xy_noise_m", [0.0, 0.0])],
        object_yaw_noise_deg=[float(item) for item in pose.get("yaw_deg", [0.0, 0.0])],
        mass_scale=[float(item) for item in physics.get("mass_scale", [1.0, 1.0])],
        friction_scale=[float(item) for item in physics.get("friction_scale", [1.0, 1.0])],
    )


def _yaw_quaternion(yaw_deg: float) -> list[float]:
    half = math.radians(yaw_deg) * 0.5
    return [round(math.cos(half), 8), 0.0, 0.0, round(math.sin(half), 8)]
