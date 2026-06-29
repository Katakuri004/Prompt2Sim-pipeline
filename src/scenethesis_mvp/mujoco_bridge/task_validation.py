from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.mujoco_bridge.schemas import SceneIR


DEFAULT_IK_OPTIONS: dict[str, float | int | None] = {
    "position_weight": 1.0,
    "z_weight": 3.0,
    "rotation_weight": 0.0,
    "posture_weight": 0.02,
    "position_tolerance_m": 0.03,
    "orientation_tolerance_rad": None,
    "max_iterations": 220,
    "max_step_rad": 0.05,
    "damping": 2e-3,
}


def validate_task_feasibility(scene_ir: SceneIR, config: dict[str, Any], raise_on_failure: bool = True) -> dict[str, object]:
    target = scene_ir.object_by_id(scene_ir.task.target_object)
    robot_cfg = config.get("robot", {})
    task_cfg = config.get("task", {})
    gripper_width = float(robot_cfg.get("gripper_max_width_m", 0.08))
    reach = float(robot_cfg.get("reach_m", 0.95))
    target_dims = [float(item) for item in target.dimensions]
    min_grasp_width = min(target_dims[0], target_dims[1])
    base = scene_ir.robot.base_position
    target_distance = _xy_distance(base, target.pose.position)
    destination_distance = _xy_distance(base, scene_ir.task.destination_position)
    checks = [
        {
            "name": "target_dynamic",
            "ok": target.mobility == "dynamic",
            "detail": f"mobility={target.mobility}",
        },
        {
            "name": "target_grasp_width",
            "ok": min_grasp_width <= gripper_width,
            "detail": f"min_horizontal_width={min_grasp_width:.3f}m, gripper_max_width={gripper_width:.3f}m",
        },
        {
            "name": "target_reachable",
            "ok": target_distance <= reach,
            "detail": f"base_to_target_xy={target_distance:.3f}m, reach={reach:.3f}m",
        },
        {
            "name": "destination_reachable",
            "ok": destination_distance <= reach,
            "detail": f"base_to_destination_xy={destination_distance:.3f}m, reach={reach:.3f}m",
        },
    ]
    ok = all(bool(check["ok"]) for check in checks)
    failure_reasons = [_operational_reason(check) for check in checks if not check["ok"]]
    report: dict[str, object] = {
        "ok": ok,
        "target_object": target.id,
        "target_dimensions_m": target_dims,
        "robot": scene_ir.robot.id,
        "checks": checks,
        "failure_reasons": failure_reasons,
    }
    if raise_on_failure and bool(task_cfg.get("require_graspable_target", True)) and not ok:
        details = "; ".join(str(check["detail"]) for check in checks if not check["ok"])
        raise RuntimeError(f"Task is not feasible for the configured Panda pick-place policy: {details}")
    return report


def validate_compiled_task_feasibility(
    scene_ir: SceneIR,
    config: dict[str, Any],
    model_path: str | Path,
    raise_on_failure: bool = True,
) -> dict[str, object]:
    report = validate_task_feasibility(scene_ir, config, raise_on_failure=False)
    try:
        import mujoco

        path = Path(model_path)
        if path.suffix.lower() == ".mjb" and hasattr(mujoco.MjModel, "from_binary_path"):
            model = mujoco.MjModel.from_binary_path(str(path))
        else:
            model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        _seed_robot_home(mujoco, model, data, scene_ir)
        mujoco.mj_forward(model, data)
        waypoints = _task_waypoints(scene_ir)
        waypoint_reports = []
        current_qpos = data.qpos.copy()
        for index, waypoint in enumerate(waypoints):
            data.qpos[:] = current_qpos
            solved, err = _solve_ik_to_site(mujoco, model, data, scene_ir.robot.ee_site, waypoint)
            current_qpos = data.qpos.copy()
            contact_free = _robot_path_contact_free(mujoco, model, data, scene_ir)
            waypoint_reports.append(
                {
                    "index": index,
                    "target": [round(float(item), 6) for item in waypoint],
                    "ik_solved": bool(solved),
                    "position_error_m": round(float(err), 6),
                    "collision_free": bool(contact_free),
                }
            )
        ik_ok = all(bool(item["ik_solved"]) for item in waypoint_reports)
        path_ok = all(bool(item["collision_free"]) for item in waypoint_reports)
        compiled_checks = [
            {"name": "trajectory_ik", "ok": ik_ok, "detail": "all pick/place waypoints solved" if ik_ok else "one or more pick/place waypoints failed IK"},
            {"name": "trajectory_collision_free", "ok": path_ok, "detail": "no robot contacts with static proxies at solved waypoints" if path_ok else "robot contacts static proxies at one or more solved waypoints"},
        ]
        report["compiled_checks"] = compiled_checks
        report["trajectory_waypoints"] = waypoint_reports
        report["ok"] = bool(report["ok"]) and ik_ok and path_ok
        report["failure_reasons"] = list(report.get("failure_reasons", [])) + [
            _operational_reason(check) for check in compiled_checks if not check["ok"]
        ]
    except Exception as exc:
        report["compiled_checks"] = [
            {"name": "compiled_feasibility_runtime", "ok": False, "detail": str(exc)}
        ]
        report["ok"] = False
        report["failure_reasons"] = list(report.get("failure_reasons", [])) + [f"compiled feasibility validation failed: {exc}"]
    if raise_on_failure and bool(config.get("task", {}).get("require_graspable_target", True)) and not report.get("ok"):
        raise RuntimeError("Task is not feasible for the configured Panda pick-place policy: " + "; ".join(str(item) for item in report.get("failure_reasons", [])))
    return report


def _xy_distance(left: list[float], right: list[float]) -> float:
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def _operational_reason(check: dict[str, object]) -> str:
    name = str(check.get("name"))
    detail = str(check.get("detail"))
    if name == "target_grasp_width":
        return f"target exceeds gripper grasp envelope: {detail}"
    if name == "target_reachable":
        return f"target is outside Panda reach: {detail}"
    if name == "destination_reachable":
        return f"sampled destination is outside Panda reach: {detail}"
    if name == "target_dynamic":
        return f"target is not bound to a dynamic body: {detail}"
    if name == "trajectory_ik":
        return "no valid IK solution for the complete pre-grasp/grasp/lift/place/retreat path"
    if name == "trajectory_collision_free":
        return "candidate pick/place path collides with static scene proxies"
    return detail


def _seed_robot_home(mujoco: Any, model: Any, data: Any, scene_ir: SceneIR) -> None:
    joint_names = list(scene_ir.robot.arm_joint_names) + list(scene_ir.robot.gripper_joint_names)
    for index, value in enumerate(scene_ir.robot.home_qpos[: len(joint_names)]):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_names[index])
        if joint_id < 0:
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        data.qpos[qadr] = float(value)


def _task_waypoints(scene_ir: SceneIR) -> list[np.ndarray]:
    target = np.asarray(scene_ir.object_by_id(scene_ir.task.target_object).pose.position, dtype=float)
    dest = np.asarray(scene_ir.task.destination_position, dtype=float)
    base = np.asarray(scene_ir.robot.base_position, dtype=float)
    approach = np.zeros(3, dtype=float)
    offset_xy = base[:2] - target[:2]
    norm = float(np.linalg.norm(offset_xy))
    if norm > 1e-6:
        approach[:2] = offset_xy / norm * 0.16
    return [
        target + approach * 0.35 + np.asarray([0.0, 0.0, 0.14]),
        target + np.asarray([0.0, 0.0, 0.035]),
        target + approach * 0.5 + np.asarray([0.0, 0.0, 0.16]),
        dest + np.asarray([0.0, 0.0, 0.16]),
        dest + np.asarray([0.0, 0.0, 0.08]),
        dest + np.asarray([0.0, 0.0, 0.16]),
    ]


def _solve_ik_to_site(
    mujoco: Any,
    model: Any,
    data: Any,
    site_name: str,
    target: np.ndarray,
    target_xmat: np.ndarray | None = None,
    options: dict[str, float | int | None] | None = None,
) -> tuple[bool, float]:
    result = _solve_ik_to_site_diagnostic(mujoco, model, data, site_name, target, target_xmat, options)
    return bool(result["ok"]), float(result["position_error_m"])


def _solve_ik_to_site_diagnostic(
    mujoco: Any,
    model: Any,
    data: Any,
    site_name: str,
    target: np.ndarray,
    target_xmat: np.ndarray | None = None,
    options: dict[str, float | int | None] | None = None,
) -> dict[str, Any]:
    cfg = dict(DEFAULT_IK_OPTIONS)
    if options:
        cfg.update(options)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id < 0:
        raise RuntimeError(f"MuJoCo model is missing site {site_name}")
    arm_joint_names = [f"joint{index}" for index in range(1, 8)]
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, arm_joint_names[0]) < 0:
        arm_joint_names = [f"panda_joint{index}" for index in range(1, 8)]
    joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in arm_joint_names]
    joint_ids = [item for item in joint_ids if item >= 0]
    dof_ids = [int(model.jnt_dofadr[joint_id]) for joint_id in joint_ids]
    qpos_ids = [int(model.jnt_qposadr[joint_id]) for joint_id in joint_ids]
    seed_q = np.asarray([data.qpos[index] for index in qpos_ids], dtype=float)
    best_position_error = float("inf")
    best_orientation_error = 0.0
    best_iteration = 0
    max_iterations = int(cfg["max_iterations"] or 220)
    rotation_weight = float(cfg["rotation_weight"] or 0.0)
    position_tolerance = float(cfg["position_tolerance_m"] or 0.03)
    orientation_tolerance = cfg["orientation_tolerance_rad"]
    if orientation_tolerance is None:
        orientation_tolerance = 10.0 if rotation_weight <= 0.0 else 0.18
    orientation_tolerance = float(orientation_tolerance)
    for iteration in range(max_iterations):
        mujoco.mj_forward(model, data)
        current = data.site_xpos[site_id].copy()
        err = target - current
        position_error = float(np.linalg.norm(err))
        rotation_error = np.zeros(3, dtype=float)
        if target_xmat is not None:
            current_xmat = data.site_xmat[site_id].copy().reshape(3, 3)
            rotation_error = _rotation_error_vector(current_xmat, target_xmat)
        orientation_error = float(np.linalg.norm(rotation_error))
        if position_error < best_position_error:
            best_position_error = position_error
            best_orientation_error = orientation_error
            best_iteration = iteration
        if position_error <= position_tolerance and (target_xmat is None or rotation_weight <= 0.0 or orientation_error <= orientation_tolerance):
            return {
                "ok": True,
                "position_error_m": position_error,
                "orientation_error_rad": orientation_error,
                "iterations": iteration + 1,
                "reason": "converged",
                "manipulability": _manipulability(mujoco, model, data, site_id, dof_ids),
            }
        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        rows = []
        errors = []
        position_weights = np.asarray([float(cfg["position_weight"] or 1.0), float(cfg["position_weight"] or 1.0), float(cfg["z_weight"] or 3.0)], dtype=float)
        rows.append(np.diag(position_weights) @ jacp[:, dof_ids])
        errors.append(position_weights * err)
        if target_xmat is not None and rotation_weight > 0.0:
            rot_weights = np.full(3, rotation_weight, dtype=float)
            rows.append(np.diag(rot_weights) @ jacr[:, dof_ids])
            errors.append(rot_weights * rotation_error)
        posture_weight = float(cfg["posture_weight"] or 0.0)
        if posture_weight > 0.0:
            current_q = np.asarray([data.qpos[index] for index in qpos_ids], dtype=float)
            rows.append(np.eye(len(qpos_ids), dtype=float) * posture_weight)
            errors.append((seed_q - current_q) * posture_weight)
        j = np.vstack(rows)
        err_eff = np.concatenate(errors)
        damping = float(cfg["damping"] or 2e-3)
        dq = j.T @ np.linalg.solve(j @ j.T + damping * np.eye(j.shape[0]), err_eff)
        dq = np.clip(dq, -float(cfg["max_step_rad"] or 0.05), float(cfg["max_step_rad"] or 0.05))
        for index, qpos_id in enumerate(qpos_ids):
            joint_id = joint_ids[index]
            low, high = model.jnt_range[joint_id]
            data.qpos[qpos_id] = float(np.clip(data.qpos[qpos_id] + dq[index], low, high))
    mujoco.mj_forward(model, data)
    final_position_error = float(np.linalg.norm(target - data.site_xpos[site_id]))
    final_orientation_error = 0.0
    if target_xmat is not None:
        final_orientation_error = float(np.linalg.norm(_rotation_error_vector(data.site_xmat[site_id].copy().reshape(3, 3), target_xmat)))
    return {
        "ok": False,
        "position_error_m": final_position_error,
        "orientation_error_rad": final_orientation_error,
        "best_position_error_m": best_position_error,
        "best_orientation_error_rad": best_orientation_error,
        "best_iteration": best_iteration,
        "iterations": max_iterations,
        "reason": "max_iterations",
        "manipulability": _manipulability(mujoco, model, data, site_id, dof_ids),
    }


def _rotation_error_vector(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=float).reshape(3, 3)
    desired = np.asarray(desired, dtype=float).reshape(3, 3)
    return 0.5 * (
        np.cross(current[:, 0], desired[:, 0])
        + np.cross(current[:, 1], desired[:, 1])
        + np.cross(current[:, 2], desired[:, 2])
    )


def _manipulability(mujoco: Any, model: Any, data: Any, site_id: int, dof_ids: list[int]) -> float:
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    j = np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
    try:
        return float(np.sqrt(max(float(np.linalg.det(j @ j.T)), 0.0)))
    except np.linalg.LinAlgError:
        return 0.0


def _robot_path_contact_free(mujoco: Any, model: Any, data: Any, scene_ir: SceneIR) -> bool:
    mujoco.mj_forward(model, data)
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = _body_name_for_geom(mujoco, model, int(contact.geom1))
        body2 = _body_name_for_geom(mujoco, model, int(contact.geom2))
        body1_is_robot = _is_robot_body(body1)
        body2_is_robot = _is_robot_body(body2)
        if body1_is_robot != body2_is_robot:
            scene_body = body2 if body1_is_robot else body1
            if scene_body == scene_ir.task.target_object:
                continue
            return False
    return True


def _is_robot_body(body_name: str) -> bool:
    return body_name.startswith("panda") or body_name in {
        "link0",
        "link1",
        "link2",
        "link3",
        "link4",
        "link5",
        "link6",
        "link7",
        "hand",
        "left_finger",
        "right_finger",
    }


def _body_name_for_geom(mujoco: Any, model: Any, geom_id: int) -> str:
    body_id = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"
