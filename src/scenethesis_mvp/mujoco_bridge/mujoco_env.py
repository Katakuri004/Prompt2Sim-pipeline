from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.mujoco_bridge.policy_adapter import PolicyAdapter
from scenethesis_mvp.mujoco_bridge.schemas import SceneIR
from scenethesis_mvp.mujoco_bridge.task_validation import _solve_ik_to_site, _solve_ik_to_site_diagnostic
from scenethesis_mvp.utils.io import write_json


class MujocoSceneEnv:
    def __init__(self, model_path: str | Path, scene_ir: SceneIR, physics_steps_per_action: int = 50, render_rgb: bool = True) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model_path = Path(model_path)
        if self.model_path.suffix.lower() == ".mjb" and hasattr(mujoco.MjModel, "from_binary_path"):
            self.model = mujoco.MjModel.from_binary_path(str(self.model_path))
        else:
            self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.scene_ir = scene_ir
        self.physics_steps_per_action = int(physics_steps_per_action)
        self.adapter = PolicyAdapter(self.model, self.data, scene_ir.robot, scene_ir.policy)
        self.render_rgb = render_rgb
        self._success_streak = 0
        self._renderers: dict[tuple[str, int, int], Any] = {}
        self._render_errors: dict[str, str] = {}
        self._grasp_attempted = False
        self._released_after_grasp = False
        self._target_lifted = False
        self._verified_grasp = False
        self._two_finger_contact = False
        self._two_finger_contact_streak = 0
        self._contact_loss_steps = 0
        self._stable_grasp = False
        self._grasp_lost = False
        self._target_following_ee = False
        self._target_ee_vector_at_grasp: np.ndarray | None = None
        self._task_phase = "RESET"
        self._initial_target_z = 0.0
        self._initial_target_position = np.zeros(3, dtype=float)
        self._teacher_joint_waypoints: list[dict[str, Any]] = []
        self._teacher_plan: dict[str, Any] = {"ok": False, "joint_waypoints": [], "reason": "not_planned"}
        self._teacher_candidates: list[dict[str, Any]] = []
        self._teacher_search: dict[str, Any] = {"candidates": []}
        self._grasp_probe_search: dict[str, Any] = {"candidates": []}
        self._coordinate_frame_audit: dict[str, Any] = {}
        self.target_body_id = _body_id(self.model, scene_ir.task.target_object)
        self.ee_site_id = self.adapter.ee_site_id
        self.robot_base_body_id = self._find_robot_base_body_id()

    def reset(self, scenario_seed: int | None = None) -> dict[str, Any]:
        rng = np.random.default_rng(scenario_seed)
        self.mujoco.mj_resetData(self.model, self.data)
        self.model.body_pos[self.robot_base_body_id] = np.asarray(self.scene_ir.robot.base_position, dtype=float)
        self.model.body_quat[self.robot_base_body_id] = _yaw_quat_wxyz(float(self.scene_ir.robot.base_yaw_rad))
        self._apply_robot_home_qpos()
        for obj in self.scene_ir.objects:
            if obj.mobility != "dynamic":
                continue
            joint_name = f"{obj.id}_freejoint"
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            qadr = int(self.model.jnt_qposadr[joint_id])
            pos = np.asarray(obj.pose.position, dtype=float)
            if obj.id == self.scene_ir.task.target_object:
                pos = self._support_aligned_dynamic_position(obj, pos)
                xy_noise = self.scene_ir.reset.object_xy_noise_m
                pos[0] += rng.uniform(xy_noise[0], xy_noise[1])
                pos[1] += rng.uniform(xy_noise[0], xy_noise[1])
            self.data.qpos[qadr : qadr + 3] = pos
            self.data.qpos[qadr + 3 : qadr + 7] = np.asarray(obj.pose.quaternion, dtype=float)
        self.mujoco.mj_forward(self.model, self.data)
        self._seed_ctrl_from_qpos()
        self._success_streak = 0
        self._grasp_attempted = False
        self._released_after_grasp = False
        self._target_lifted = False
        self._verified_grasp = False
        self._two_finger_contact = False
        self._two_finger_contact_streak = 0
        self._contact_loss_steps = 0
        self._stable_grasp = False
        self._grasp_lost = False
        self._target_following_ee = False
        self._target_ee_vector_at_grasp = None
        self._task_phase = "RESET"
        self.mujoco.mj_forward(self.model, self.data)
        self._initial_target_z = float(self.data.xpos[self.target_body_id][2])
        self._initial_target_position = self.data.xpos[self.target_body_id].copy()
        self._teacher_plan = self._build_teacher_joint_plan()
        self._teacher_joint_waypoints = list(self._teacher_plan.get("joint_waypoints", [])) if bool(self._teacher_plan.get("ok", False)) else []
        return self.get_observation()

    def step(self, policy_action: Any) -> tuple[dict[str, Any], dict[str, Any], bool]:
        self.adapter.apply(policy_action)
        self.mujoco.mj_step(self.model, self.data, nstep=self.physics_steps_per_action)
        self._update_manipulation_state(policy_action)
        observation = self.get_observation()
        metrics = self.compute_metrics()
        failure_reason = self.failure_oracle()
        success = False if failure_reason else self.success_oracle()
        terminated = success or failure_reason is not None
        metrics["success"] = success
        metrics["terminated_reason"] = "success" if success else (failure_reason or "running")
        return observation, metrics, terminated

    def get_observation(self) -> dict[str, Any]:
        ee_position = self.data.site_xpos[self.ee_site_id].copy()
        target_position = self.data.xpos[self.target_body_id].copy()
        state = {
            "ee_position": ee_position,
            "ee_xmat": self.data.site_xmat[self.ee_site_id].copy().reshape(3, 3),
            "target_position": target_position,
            "target_xmat": self.data.xmat[self.target_body_id].copy().reshape(3, 3),
            "target_relative_ee_vector": target_position - ee_position,
            "destination_position": np.asarray(self.scene_ir.task.destination_position, dtype=float),
            "gripper_width": self._gripper_width(),
            "verified_grasp": self._verified_grasp,
            "stable_grasp": self._stable_grasp,
            "grasp_lost": self._grasp_lost,
            "target_following_ee": self._target_following_ee,
            "two_finger_contact": self._two_finger_contact,
            "two_finger_contact_streak": self._two_finger_contact_streak,
            "left_finger_contact": self._target_contacted_by_finger({"panda_leftfinger", "left_finger"}),
            "right_finger_contact": self._target_contacted_by_finger({"panda_rightfinger", "right_finger"}),
            "task_phase": self._task_phase,
            "teacher_joint_waypoints": self._teacher_joint_waypoints,
            "teacher_plan": self.teacher_plan_report(),
        }
        obs: dict[str, Any] = {
            "proprio": self.adapter.proprio(),
            "state": state,
            "language_instruction": self.scene_ir.scene_id if self.scene_ir.policy.language_instruction else None,
        }
        if self.scene_ir.policy.observation_cameras:
            obs["rgb"] = self._render_cameras() if self.render_rgb else {}
        return obs

    def compute_metrics(self) -> dict[str, Any]:
        forces = []
        bad_contacts = 0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if not self._is_bad_contact(int(contact.geom1), int(contact.geom2)):
                continue
            bad_contacts += 1
            force = np.zeros(6, dtype=float)
            self.mujoco.mj_contactForce(self.model, self.data, index, force)
            norm = float(np.linalg.norm(force[:3]))
            if np.isfinite(norm):
                forces.append(norm)
        target_pos = self.data.xpos[self.target_body_id]
        destination = np.asarray(self.scene_ir.task.destination_position, dtype=float)
        return {
            "collision_count": int(bad_contacts),
            "max_contact_force": max(forces) if forces else 0.0,
            "target_distance_m": float(np.linalg.norm(target_pos - destination)),
            "object_drop": bool(target_pos[2] < 0.015),
            "workspace_violation": self._workspace_violation(),
            "grasp_attempted": self._grasp_attempted,
            "verified_grasp": self._verified_grasp,
            "stable_grasp": self._stable_grasp,
            "grasp_lost": self._grasp_lost,
            "target_following_ee": self._target_following_ee,
            "two_finger_contact": self._two_finger_contact,
            "two_finger_contact_streak": self._two_finger_contact_streak,
            "released_after_grasp": self._released_after_grasp,
            "target_lifted": self._target_lifted,
            "target_placed": float(np.linalg.norm(target_pos - destination)) <= self.scene_ir.task.max_position_error_m,
            "pregrasp_target_drift_m": self._pregrasp_target_drift(),
            "task_phase": self._task_phase,
        }

    def success_oracle(self) -> bool:
        if not self._verified_grasp or not self._released_after_grasp:
            self._success_streak = 0
            return False
        if self._has_bad_contact():
            self._success_streak = 0
            return False
        target_pos = self.data.xpos[self.target_body_id]
        destination = np.asarray(self.scene_ir.task.destination_position, dtype=float)
        if float(np.linalg.norm(target_pos - destination)) <= self.scene_ir.task.max_position_error_m:
            self._success_streak += 1
        else:
            self._success_streak = 0
        return self._success_streak >= self.scene_ir.task.stable_steps

    def failure_oracle(self) -> str | None:
        target_pos = self.data.xpos[self.target_body_id]
        if target_pos[2] < 0.015:
            return "object_drop"
        if self._workspace_violation():
            return "workspace_violation"
        if not self._grasp_attempted and self._pregrasp_target_drift() > self.scene_ir.task.max_pregrasp_target_drift_m:
            return "pregrasp_target_motion"
        if self._grasp_lost:
            return "grasp_lost"
        if self._has_bad_contact():
            return "bad_contact"
        forbidden = tuple(self.scene_ir.task.forbidden_contacts)
        if forbidden:
            for index in range(self.data.ncon):
                contact = self.data.contact[index]
                name1 = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or ""
                name2 = self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or ""
                if any(name1.startswith(prefix) or name2.startswith(prefix) for prefix in forbidden):
                    return "forbidden_contact"
        return None

    def close(self) -> None:
        self._renderers.clear()

    @property
    def teacher_plan_ok(self) -> bool:
        return bool(self._teacher_plan.get("ok", False))

    def teacher_plan_report(self) -> dict[str, Any]:
        report = {key: value for key, value in self._teacher_plan.items() if key != "joint_waypoints"}
        report["joint_waypoint_count"] = len(self._teacher_plan.get("joint_waypoints", []))
        return _jsonable(report)

    def teacher_waypoint_diagnostics(self) -> dict[str, Any]:
        return _jsonable(
            {
                "ok": self.teacher_plan_ok,
                "selected_candidate": self._teacher_plan.get("selected_candidate"),
                "failed_phase": self._teacher_plan.get("failed_phase"),
                "reason": self._teacher_plan.get("reason"),
                "waypoints": self._teacher_plan.get("waypoint_diagnostics", []),
            }
        )

    def teacher_candidate_report(self) -> dict[str, Any]:
        return _jsonable({"candidates": self._teacher_candidates})

    def teacher_plan_search_report(self) -> dict[str, Any]:
        return _jsonable(self._teacher_search)

    def grasp_probe_search_report(self) -> dict[str, Any]:
        return _jsonable(self._grasp_probe_search)

    def coordinate_frame_audit_report(self) -> dict[str, Any]:
        return _jsonable(self._coordinate_frame_audit)

    def run_grasp_probe(self, out_dir: str | Path, camera_id: str = "report_task_closeup") -> dict[str, Any]:
        target = Path(out_dir)
        snapshots_dir = target / "phase_snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        return self._probe_plan_grasp(
            self._teacher_plan,
            snapshots_dir=snapshots_dir,
            write_snapshots=True,
            camera_id=camera_id,
            include_approach_phases=True,
        )

    def _probe_plan_grasp(
        self,
        plan: dict[str, Any],
        *,
        snapshots_dir: Path | None = None,
        write_snapshots: bool = False,
        camera_id: str = "report_task_closeup",
        start_qpos: np.ndarray | None = None,
        start_qvel: np.ndarray | None = None,
        start_ctrl: np.ndarray | None = None,
        start_state: dict[str, Any] | None = None,
        include_approach_phases: bool = False,
    ) -> dict[str, Any]:
        if not bool(plan.get("ok", False)):
            return {
                "feasible": False,
                "failure_reason": "teacher_plan_unavailable",
                "teacher_plan": {key: value for key, value in plan.items() if key != "joint_waypoints"},
                "snapshots_dir": str(snapshots_dir) if snapshots_dir is not None else None,
                "probe_score": -1e9,
            }
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        saved_ctrl = self.data.ctrl.copy()
        saved_state = self._manipulation_state_snapshot()
        saved_base_pos = self.model.body_pos[self.robot_base_body_id].copy()
        saved_base_quat = self.model.body_quat[self.robot_base_body_id].copy()
        lift_delta = 0.0
        failure_reason: str | None = None
        phase_reports: list[dict[str, Any]] = []
        try:
            selected_base = plan.get("base_candidate")
            if isinstance(selected_base, dict):
                self._apply_robot_base_candidate(selected_base)
            if start_qpos is not None:
                self.data.qpos[:] = np.asarray(start_qpos, dtype=float)
            else:
                self._apply_robot_home_qpos()
            if start_qvel is not None:
                self.data.qvel[:] = np.asarray(start_qvel, dtype=float)
            else:
                self.data.qvel[:] = 0.0
            if start_ctrl is not None:
                self.data.ctrl[:] = np.asarray(start_ctrl, dtype=float)
            else:
                self._seed_ctrl_from_qpos()
            if start_state is not None:
                self._restore_manipulation_state(start_state)
            self.mujoco.mj_forward(self.model, self.data)
            initial_target_z = float(self.data.xpos[self.target_body_id][2])
            probe_phases = {"PRE_CLOSE", "CLOSE_GRIPPER", "GRASP_SETTLE", "MICRO_LIFT"}
            if include_approach_phases:
                probe_phases = {"HOME", "SAFE_APPROACH", "PREGRASP", *probe_phases}
            for waypoint in plan.get("joint_waypoints", []):
                phase = str(waypoint.get("phase", "UNKNOWN"))
                if phase not in probe_phases:
                    continue
                command = np.asarray(waypoint.get("qpos", []), dtype=float)
                if command.size == 0:
                    failure_reason = "teacher_plan_unavailable"
                    break
                if phase in {"HOME", "SAFE_APPROACH", "PREGRASP"}:
                    min_repetitions = 8
                    max_repetitions = 70
                elif phase == "PRE_CLOSE":
                    min_repetitions = 8
                    max_repetitions = 45
                else:
                    min_repetitions = 10
                    max_repetitions = 70 if phase == "MICRO_LIFT" else 30
                for repetition in range(max_repetitions):
                    streamed = command.copy()
                    current_arm = np.asarray([self.data.qpos[index] for index in self.adapter.arm_qpos_ids], dtype=float)
                    streamed[:7] = current_arm + np.clip(command[:7] - current_arm, -0.025, 0.025)
                    self.adapter.apply(streamed)
                    self.mujoco.mj_step(self.model, self.data, nstep=self.physics_steps_per_action)
                    self._update_manipulation_state(streamed)
                    arm_error = float(np.linalg.norm(command[:7] - np.asarray([self.data.qpos[index] for index in self.adapter.arm_qpos_ids], dtype=float)))
                    if phase in {"HOME", "SAFE_APPROACH", "PREGRASP"} and repetition >= min_repetitions and arm_error < 0.08:
                        break
                    if phase == "PRE_CLOSE" and repetition >= min_repetitions and arm_error < 0.06:
                        break
                    if phase in {"CLOSE_GRIPPER", "GRASP_SETTLE"} and repetition >= min_repetitions and self._stable_grasp:
                        break
                    if phase == "MICRO_LIFT":
                        target_z_now = float(self.data.xpos[self.target_body_id][2])
                        if repetition >= min_repetitions and (target_z_now - initial_target_z >= 0.018 or self._grasp_lost):
                            break
                metrics = self.compute_metrics()
                target_z = float(self.data.xpos[self.target_body_id][2])
                lift_delta = max(lift_delta, target_z - initial_target_z)
                left_contact = self._target_contacted_by_finger({"panda_leftfinger", "left_finger"})
                right_contact = self._target_contacted_by_finger({"panda_rightfinger", "right_finger"})
                contact_forces = self._target_contact_force_summary()
                phase_report = {
                    "phase": phase,
                    "left_contact": bool(left_contact),
                    "right_contact": bool(right_contact),
                    "two_finger_contact": bool(left_contact and right_contact),
                    "target_contact_force_summary": contact_forces,
                    "stable_grasp": bool(metrics.get("stable_grasp", False)),
                    "grasp_lost": bool(metrics.get("grasp_lost", False)),
                    "scanner_table_contact": self._target_support_contact(),
                    "lift_delta_z_m": float(lift_delta),
                    "target_z_m": target_z,
                    "ee_target_distance_m": float(np.linalg.norm(self.data.site_xpos[self.ee_site_id] - self.data.xpos[self.target_body_id])),
                    "gripper_width_m": float(self._gripper_width()),
                }
                phase_reports.append(phase_report)
                if write_snapshots and snapshots_dir is not None:
                    write_json(snapshots_dir / f"{len(phase_reports):02d}_{phase.lower()}.json", _jsonable(self.dense_state_snapshot(len(phase_reports), command, metrics)))
                if phase == "GRASP_SETTLE" and not bool(metrics.get("stable_grasp", False)):
                    failure_reason = "bilateral_contact_not_stable"
                    break
                if phase == "MICRO_LIFT":
                    if bool(metrics.get("grasp_lost", False)):
                        failure_reason = "grasp_lost_during_lift"
                    elif lift_delta < 0.018:
                        failure_reason = "micro_lift_failed"
                    elif self._target_support_contact():
                        failure_reason = "scanner_table_jammed"
                    break
        finally:
            self.model.body_pos[self.robot_base_body_id] = saved_base_pos
            self.model.body_quat[self.robot_base_body_id] = saved_base_quat
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            self.data.ctrl[:] = saved_ctrl
            self._restore_manipulation_state(saved_state)
            self.mujoco.mj_forward(self.model, self.data)
        feasible = failure_reason is None and lift_delta >= 0.018
        result = {
            "feasible": bool(feasible),
            "left_contact": any(bool(item.get("left_contact")) for item in phase_reports),
            "right_contact": any(bool(item.get("right_contact")) for item in phase_reports),
            "two_finger_contact": any(bool(item.get("two_finger_contact")) for item in phase_reports),
            "stable_grasp": any(bool(item.get("stable_grasp")) for item in phase_reports),
            "scanner_table_contact_after_lift": bool(phase_reports[-1].get("scanner_table_contact", False)) if phase_reports else None,
            "lift_delta_z_m": float(lift_delta),
            "grasp_loss": any(bool(item.get("grasp_lost")) for item in phase_reports),
            "failure_reason": failure_reason,
            "phase_reports": phase_reports,
            "snapshots_dir": str(snapshots_dir) if snapshots_dir is not None else None,
            "base_candidate_id": plan.get("base_candidate_id"),
            "grasp_candidate_id": plan.get("selected_candidate"),
            "camera_id": camera_id,
        }
        result["probe_score"] = _score_grasp_probe(result)
        return result

    def _probe_plan_completion(
        self,
        plan: dict[str, Any],
        *,
        start_qpos: np.ndarray,
        start_qvel: np.ndarray,
        start_ctrl: np.ndarray,
        start_state: dict[str, Any],
        max_steps: int = 250,
    ) -> dict[str, Any]:
        if not bool(plan.get("ok", False)):
            return {"success": False, "terminated_reason": "teacher_plan_unavailable", "completion_score": -1e9}
        from scenethesis_mvp.mujoco_bridge.policies import TeacherPlanUnavailable, make_policy

        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        saved_ctrl = self.data.ctrl.copy()
        saved_state = self._manipulation_state_snapshot()
        saved_base_pos = self.model.body_pos[self.robot_base_body_id].copy()
        saved_base_quat = self.model.body_quat[self.robot_base_body_id].copy()
        saved_plan = self._teacher_plan
        saved_waypoints = list(self._teacher_joint_waypoints)
        saved_initial_target_z = float(self._initial_target_z)
        saved_initial_target_position = self._initial_target_position.copy()
        final_metrics: dict[str, Any] = {}
        steps = 0
        try:
            selected_base = plan.get("base_candidate")
            if isinstance(selected_base, dict):
                self._apply_robot_base_candidate(selected_base)
            self.data.qpos[:] = np.asarray(start_qpos, dtype=float)
            self.data.qvel[:] = np.asarray(start_qvel, dtype=float)
            self.data.ctrl[:] = np.asarray(start_ctrl, dtype=float)
            self._restore_manipulation_state(start_state)
            self.mujoco.mj_forward(self.model, self.data)
            self._initial_target_z = float(self.data.xpos[self.target_body_id][2])
            self._initial_target_position = self.data.xpos[self.target_body_id].copy()
            self._teacher_plan = plan
            self._teacher_joint_waypoints = list(plan.get("joint_waypoints", []))
            policy = make_policy("teacher_pick_place", self.scene_ir.policy)
            policy.reset(0)
            observation = self.get_observation()
            for step_index in range(max_steps):
                try:
                    action = policy.act(observation)
                except TeacherPlanUnavailable:
                    final_metrics = self.compute_metrics()
                    final_metrics["success"] = False
                    final_metrics["terminated_reason"] = "teacher_plan_unavailable"
                    steps = step_index
                    break
                observation, final_metrics, terminated = self.step(action)
                steps = step_index + 1
                if terminated:
                    break
            if not final_metrics:
                final_metrics = self.compute_metrics()
                final_metrics["success"] = self.success_oracle()
                final_metrics["terminated_reason"] = "max_steps"
            if not bool(final_metrics.get("success", False)) and str(final_metrics.get("terminated_reason")) in {"running", "max_steps"}:
                final_metrics["terminated_reason"] = _completion_timeout_reason(final_metrics)
            result = {
                "success": bool(final_metrics.get("success", False)),
                "terminated_reason": str(final_metrics.get("terminated_reason", "unknown")),
                "steps": int(steps),
                "target_distance_m": final_metrics.get("target_distance_m"),
                "collision_count": int(final_metrics.get("collision_count", 0)),
                "max_contact_force": float(final_metrics.get("max_contact_force", 0.0)),
                "grasp_attempted": bool(final_metrics.get("grasp_attempted", False)),
                "stable_grasp": bool(final_metrics.get("stable_grasp", False)),
                "grasp_lost": bool(final_metrics.get("grasp_lost", False)),
                "released_after_grasp": bool(final_metrics.get("released_after_grasp", False)),
                "target_lifted": bool(final_metrics.get("target_lifted", False)),
                "target_placed": bool(final_metrics.get("target_placed", False)),
            }
            result["completion_score"] = _score_completion_probe(result)
            return result
        finally:
            self.model.body_pos[self.robot_base_body_id] = saved_base_pos
            self.model.body_quat[self.robot_base_body_id] = saved_base_quat
            self.data.qpos[:] = saved_qpos
            self.data.qvel[:] = saved_qvel
            self.data.ctrl[:] = saved_ctrl
            self._restore_manipulation_state(saved_state)
            self._teacher_plan = saved_plan
            self._teacher_joint_waypoints = saved_waypoints
            self._initial_target_z = saved_initial_target_z
            self._initial_target_position = saved_initial_target_position
            self.mujoco.mj_forward(self.model, self.data)

    def _seed_ctrl_from_qpos(self) -> None:
        for joint_name in self.scene_ir.robot.arm_joint_names:
            actuator_id = self.adapter.actuator_by_joint_name.get(joint_name)
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if actuator_id is None or joint_id < 0:
                continue
            qadr = int(self.model.jnt_qposadr[joint_id])
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = np.clip(self.data.qpos[qadr], low, high)
        for actuator_id in self.adapter.gripper_actuator_ids:
            self.data.ctrl[actuator_id] = self._gripper_actuator_value(0.04)

    def _manipulation_state_snapshot(self) -> dict[str, Any]:
        return {
            "success_streak": self._success_streak,
            "grasp_attempted": self._grasp_attempted,
            "released_after_grasp": self._released_after_grasp,
            "target_lifted": self._target_lifted,
            "verified_grasp": self._verified_grasp,
            "two_finger_contact": self._two_finger_contact,
            "two_finger_contact_streak": self._two_finger_contact_streak,
            "contact_loss_steps": self._contact_loss_steps,
            "stable_grasp": self._stable_grasp,
            "grasp_lost": self._grasp_lost,
            "target_following_ee": self._target_following_ee,
            "target_ee_vector_at_grasp": None if self._target_ee_vector_at_grasp is None else self._target_ee_vector_at_grasp.copy(),
            "task_phase": self._task_phase,
        }

    def _restore_manipulation_state(self, snapshot: dict[str, Any]) -> None:
        self._success_streak = int(snapshot["success_streak"])
        self._grasp_attempted = bool(snapshot["grasp_attempted"])
        self._released_after_grasp = bool(snapshot["released_after_grasp"])
        self._target_lifted = bool(snapshot["target_lifted"])
        self._verified_grasp = bool(snapshot["verified_grasp"])
        self._two_finger_contact = bool(snapshot["two_finger_contact"])
        self._two_finger_contact_streak = int(snapshot["two_finger_contact_streak"])
        self._contact_loss_steps = int(snapshot["contact_loss_steps"])
        self._stable_grasp = bool(snapshot["stable_grasp"])
        self._grasp_lost = bool(snapshot["grasp_lost"])
        self._target_following_ee = bool(snapshot["target_following_ee"])
        vector = snapshot["target_ee_vector_at_grasp"]
        self._target_ee_vector_at_grasp = None if vector is None else np.asarray(vector, dtype=float)
        self._task_phase = str(snapshot["task_phase"])

    def _find_robot_base_body_id(self) -> int:
        for name in ("panda_base", "link0"):
            body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                return int(body_id)
        raise RuntimeError("MuJoCo model is missing a Panda base body named panda_base or link0")

    def _build_coordinate_frame_audit(self, grasp_frame: np.ndarray) -> dict[str, Any]:
        self.mujoco.mj_forward(self.model, self.data)
        base_pos = self.model.body_pos[self.robot_base_body_id].copy()
        ee = self.data.site_xpos[self.ee_site_id].copy()
        ee_body_id = int(self.model.site_bodyid[self.ee_site_id])
        flange_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "link7")
        hand_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "hand")
        flange_world = self.data.xpos[flange_id].copy() if flange_id >= 0 else self.data.xpos[ee_body_id].copy()
        hand_world = self.data.xpos[hand_id].copy() if hand_id >= 0 else self.data.xpos[ee_body_id].copy()
        target = self.data.xpos[self.target_body_id].copy()
        support = self.scene_ir.object_by_id(self.scene_ir.task.support_id) if self.scene_ir.task.support_id else None
        support_top_z = None
        support_pose = None
        if support is not None:
            support_top_z = float(support.pose.position[2]) + float(support.dimensions[2]) * 0.5
            support_pose = support.pose.model_dump(mode="json")
        tool_offset = ee - flange_world
        ok = bool(np.isfinite(ee).all() and np.isfinite(target).all() and float(np.linalg.norm(tool_offset)) < 0.35)
        return {
            "ok": ok,
            "reason": None if ok else "invalid_or_implausible_tool_site_offset",
            "robot_base_world": base_pos,
            "robot_base_scene_ir": self.scene_ir.robot.base_position,
            "ee_site_name": self.scene_ir.robot.ee_site,
            "ee_site_world": ee,
            "ee_site_body": self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, ee_body_id) or "",
            "flange_world": flange_world,
            "hand_world": hand_world,
            "tool_offset_flange_to_site_m": tool_offset,
            "tool_offset_norm_m": float(np.linalg.norm(tool_offset)),
            "scanner_center_world": target,
            "scanner_scene_ir_position": self.scene_ir.object_by_id(self.scene_ir.task.target_object).pose.position,
            "scanner_grasp_frame_world": {
                "position": target,
                "orientation_quat_xyzw": _matrix_to_quat_xyzw(grasp_frame),
                "x_axis": grasp_frame[:, 0],
                "opening_axis": grasp_frame[:, 1],
                "approach_axis": grasp_frame[:, 2],
            },
            "table_top_world_z": support_top_z,
            "support_pose": support_pose,
            "ik_target_is_controlled_site": True,
        }

    def _base_pose_candidates(self, target: np.ndarray) -> list[dict[str, Any]]:
        base0 = self.model.body_pos[self.robot_base_body_id].copy()
        base_yaw = _yaw_from_quat_wxyz(self.model.body_quat[self.robot_base_body_id])
        target_yaw = float(np.arctan2(float(target[1] - base0[1]), float(target[0] - base0[0])))
        yaw_offsets = [0.0, np.deg2rad(10.0), -np.deg2rad(10.0), np.deg2rad(20.0), -np.deg2rad(20.0)]
        xy_offsets = [(0.0, 0.0), (0.10, 0.0), (-0.10, 0.0), (0.0, 0.10), (0.0, -0.10), (0.20, 0.0), (-0.20, 0.0), (0.0, 0.20), (0.0, -0.20)]
        candidates: list[dict[str, Any]] = []
        seen: set[tuple[float, float, float]] = set()
        index = 0
        for dx, dy in xy_offsets:
            for offset in yaw_offsets:
                yaw = target_yaw + float(offset)
                pos = base0.copy()
                pos[0] += dx
                pos[1] += dy
                key = (round(float(pos[0]), 3), round(float(pos[1]), 3), round(float(yaw), 3))
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(
                    {
                        "id": f"base_{index:02d}",
                        "position": pos.tolist(),
                        "yaw_rad": yaw,
                        "yaw_source": "toward_target_with_offset",
                        "offset_xy_m": [float(dx), float(dy)],
                        "is_current_xy": bool(abs(dx) < 1e-9 and abs(dy) < 1e-9),
                    }
                )
                index += 1
        return candidates

    def _validate_base_candidate(self, candidate: dict[str, Any], saved_pos: np.ndarray, saved_quat: np.ndarray) -> dict[str, Any]:
        width, depth, _height = [float(item) for item in self.scene_ir.bounds]
        pos = np.asarray(candidate["position"], dtype=float)
        if pos[0] < 0.0 or pos[0] > width or pos[1] < 0.0 or pos[1] > depth:
            return {"ok": False, "reason": "base_outside_scene_bounds"}
        target = self.data.xpos[self.target_body_id].copy()
        forward = np.asarray([np.cos(float(candidate["yaw_rad"])), np.sin(float(candidate["yaw_rad"]))], dtype=float)
        target_vector = target[:2] - pos[:2]
        if float(np.dot(forward, target_vector)) < -0.05:
            return {"ok": False, "reason": "target_behind_preferred_workspace"}
        self._apply_robot_base_candidate(candidate)
        self._apply_robot_home_qpos()
        self._seed_ctrl_from_qpos()
        self.mujoco.mj_forward(self.model, self.data)
        contact_free = self._robot_scene_contact_free()
        self.model.body_pos[self.robot_base_body_id] = saved_pos
        self.model.body_quat[self.robot_base_body_id] = saved_quat
        self.mujoco.mj_forward(self.model, self.data)
        if not contact_free:
            return {"ok": False, "reason": "base_collides_static_scene"}
        return {"ok": True, "reason": None}

    def _apply_robot_base_candidate(self, candidate: dict[str, Any]) -> None:
        self.model.body_pos[self.robot_base_body_id] = np.asarray(candidate["position"], dtype=float)
        self.model.body_quat[self.robot_base_body_id] = _yaw_quat_wxyz(float(candidate["yaw_rad"]))
        self.mujoco.mj_forward(self.model, self.data)

    def _apply_robot_home_qpos(self) -> None:
        home = list(self.scene_ir.robot.home_qpos)
        joint_names = list(self.scene_ir.robot.arm_joint_names) + list(self.scene_ir.robot.gripper_joint_names)
        for index, joint_name in enumerate(joint_names[: len(home)]):
            joint_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            qadr = int(self.model.jnt_qposadr[joint_id])
            self.data.qpos[qadr] = float(home[index])

    def _render_cameras(self) -> dict[str, np.ndarray]:
        frames: dict[str, np.ndarray] = {}
        for camera in self.scene_ir.policy.observation_cameras:
            width, height = int(camera.resolution[0]), int(camera.resolution[1])
            frames[camera.id] = self.render_camera(camera.id, width, height)
        return frames

    def _workspace_violation(self) -> bool:
        pos = self.data.site_xpos[self.ee_site_id]
        width, depth, height = self.scene_ir.bounds
        return bool(pos[0] < -0.1 or pos[0] > width + 0.1 or pos[1] < -0.1 or pos[1] > depth + 0.1 or pos[2] < -0.05 or pos[2] > height + 1.0)

    def _update_manipulation_state(self, policy_action: Any) -> None:
        values = np.asarray(policy_action, dtype=float).reshape(-1)
        joint_position_action = len(values) == len(self.scene_ir.robot.actuator_names)
        if not joint_position_action and (self.scene_ir.policy.action_representation != "delta_ee_pose_gripper" or len(values) < 7):
            return
        grip = float(values[6])
        if joint_position_action and len(values) >= 9:
            grip = -1.0 if float(np.mean(values[-2:])) < 0.02 else 1.0
        elif joint_position_action and len(values) >= 8:
            actuator_id = self.adapter.actuator_ids[-1]
            low, high = self.model.actuator_ctrlrange[actuator_id]
            opening = (float(values[-1]) - float(low)) / max(float(high - low), 1e-9)
            grip = -1.0 if opening < 0.5 else 1.0
        ee = self.data.site_xpos[self.ee_site_id]
        target = self.data.xpos[self.target_body_id]
        dest = np.asarray(self.scene_ir.task.destination_position, dtype=float)
        ee_target_distance = float(np.linalg.norm(ee - target))
        target_dest_distance = float(np.linalg.norm(target - dest))
        self._two_finger_contact = self._target_contacted_by_both_fingers()
        gripper_closed = grip < -0.5
        if float(target[2]) > self._initial_target_z + 0.04 and self._stable_grasp:
            self._target_lifted = True
        if gripper_closed and ee_target_distance < 0.12:
            self._grasp_attempted = True
            self._task_phase = "CLOSE_GRIPPER"

        contact_in_envelope = bool(self._two_finger_contact and ee_target_distance < 0.15)
        if self._grasp_attempted and gripper_closed and contact_in_envelope:
            self._two_finger_contact_streak += 1
        else:
            self._two_finger_contact_streak = 0

        required_contact_steps = 3
        if self._two_finger_contact_streak >= required_contact_steps and not self._grasp_lost:
            if not self._stable_grasp:
                self._target_ee_vector_at_grasp = target - ee
            self._stable_grasp = True
            self._verified_grasp = True
            self._task_phase = "GRASP_VERIFY"

        if self._stable_grasp and self._target_ee_vector_at_grasp is not None:
            relative_error = float(np.linalg.norm((target - ee) - self._target_ee_vector_at_grasp))
            self._target_following_ee = bool(self._two_finger_contact or relative_error <= 0.08)
        else:
            self._target_following_ee = False

        if self._stable_grasp and not self._released_after_grasp:
            if self._two_finger_contact:
                self._contact_loss_steps = 0
            else:
                self._contact_loss_steps += 1
            if self._contact_loss_steps > 5 and (not self._target_lifted or not self._target_following_ee):
                self._grasp_lost = True
                self._stable_grasp = False
                self._verified_grasp = False
                self._task_phase = "GRASP_LOST"

        if self._stable_grasp and self._target_lifted:
            self._task_phase = "LIFT"
        if self._stable_grasp and self._target_lifted and grip > 0.5 and target_dest_distance <= self.scene_ir.task.max_position_error_m * 2.0:
            self._released_after_grasp = True
            self._task_phase = "RELEASE"

    def _build_teacher_joint_plan(self) -> dict[str, Any]:
        saved_qpos = self.data.qpos.copy()
        saved_qvel = self.data.qvel.copy()
        saved_ctrl = self.data.ctrl.copy()
        saved_state = self._manipulation_state_snapshot()
        saved_base_pos = self.model.body_pos[self.robot_base_body_id].copy()
        saved_base_quat = self.model.body_quat[self.robot_base_body_id].copy()
        target = self.data.xpos[self.target_body_id].copy()
        dest = np.asarray(self.scene_ir.task.destination_position, dtype=float)
        target_xmat = self.data.xmat[self.target_body_id].copy().reshape(3, 3)
        target_obj = self.scene_ir.object_by_id(self.scene_ir.task.target_object)
        initial_candidates = _grasp_frame_candidates(
            target_xmat,
            np.asarray(self.scene_ir.robot.base_position, dtype=float),
            target,
            target_obj.category,
            target_obj.dimensions,
            self.scene_ir.robot.gripper_max_width_m,
        )
        if not initial_candidates:
            self._coordinate_frame_audit = self._build_coordinate_frame_audit(_side_grasp_frame(target_xmat, np.asarray(self.scene_ir.robot.base_position, dtype=float), target))
            self._teacher_search = {
                "ok": False,
                "tested_pair_count": 0,
                "valid_pair_count": 0,
                "candidates": [],
                "reason": "no_grasp_candidate_within_gripper_width",
                "target_dimensions_m": target_obj.dimensions,
                "gripper_max_width_m": self.scene_ir.robot.gripper_max_width_m,
            }
            return {
                "ok": False,
                "joint_waypoints": [],
                "failed_phase": "GRASP_CANDIDATE_SELECTION",
                "reason": "no_grasp_candidate_within_gripper_width",
                "scanner_dimensions_m": target_obj.dimensions,
            }
        initial_frame = initial_candidates[0]["frame"]
        self._coordinate_frame_audit = self._build_coordinate_frame_audit(initial_frame)
        if not bool(self._coordinate_frame_audit.get("ok", False)):
            return {
                "ok": False,
                "joint_waypoints": [],
                "failed_phase": "FRAME_AUDIT",
                "reason": "frame_audit_failed",
                "frame_audit": self._coordinate_frame_audit,
            }
        best_failure: dict[str, Any] | None = None
        valid_plans: list[dict[str, Any]] = []
        search_records: list[dict[str, Any]] = []
        self._teacher_candidates = []
        self._grasp_probe_search = {"candidates": []}
        try:
            for base_candidate in self._base_pose_candidates(target):
                base_validation = self._validate_base_candidate(base_candidate, saved_base_pos, saved_base_quat)
                if not base_validation["ok"]:
                    search_records.append(
                        {
                            "base_candidate_id": base_candidate["id"],
                            "base_position": base_candidate["position"],
                            "base_yaw_rad": base_candidate["yaw_rad"],
                            "grasp_candidate_id": None,
                            "plan_ok": False,
                            "score": -1e9,
                            "reason": base_validation["reason"],
                            "base_validation": base_validation,
                        }
                    )
                    continue
                self._apply_robot_base_candidate(base_candidate)
                for candidate in _grasp_frame_candidates(
                    target_xmat,
                    np.asarray(base_candidate["position"], dtype=float),
                    target,
                    target_obj.category,
                    target_obj.dimensions,
                    self.scene_ir.robot.gripper_max_width_m,
                ):
                    self.data.qpos[:] = saved_qpos
                    self.data.qvel[:] = saved_qvel
                    self.data.ctrl[:] = saved_ctrl
                    self._apply_robot_home_qpos()
                    self._seed_ctrl_from_qpos()
                    self.mujoco.mj_forward(self.model, self.data)
                    plan = self._solve_teacher_candidate(base_candidate, candidate, target, dest)
                    record = {key: value for key, value in plan.items() if key not in {"joint_waypoints", "base_candidate"}}
                    record["base_candidate_id"] = base_candidate["id"]
                    record["grasp_candidate_id"] = candidate["name"]
                    record["base_grasp_name"] = candidate.get("base_grasp_name", candidate["name"])
                    record["opening_axis_offset_m"] = candidate.get("opening_axis_offset_m", 0.0)
                    record["close_standoff_m"] = candidate.get("close_standoff_m")
                    search_records.append(record)
                    self._teacher_candidates.append(record)
                    if plan.get("ok"):
                        valid_plans.append(plan)
                    elif best_failure is None or _failure_rank(plan) > _failure_rank(best_failure):
                        best_failure = plan
            self._teacher_search = {
                "ok": bool(valid_plans),
                "tested_pair_count": len(search_records),
                "valid_pair_count": len(valid_plans),
                "candidates": search_records,
            }
            if valid_plans:
                valid_plans.sort(key=lambda item: (-float(item.get("score", -1e9)), str(item.get("base_candidate_id")), str(item.get("selected_candidate"))))
                probe_plans = _select_grasp_probe_plans(valid_plans, limit=24)
                probe_records: list[dict[str, Any]] = []
                for plan in probe_plans:
                    probe = self._probe_plan_grasp(
                        plan,
                        write_snapshots=False,
                        start_qpos=saved_qpos,
                        start_qvel=saved_qvel,
                        start_ctrl=saved_ctrl,
                        start_state=saved_state,
                        include_approach_phases=True,
                    )
                    probe_summary = _grasp_probe_summary(probe)
                    plan["grasp_probe_preview"] = probe_summary
                    plan["probe_score"] = float(probe_summary["probe_score"])
                    plan["combined_score"] = float(plan.get("score", -1e9)) + float(probe_summary["probe_score"])
                    probe_records.append(
                        {
                            "base_candidate_id": plan.get("base_candidate_id"),
                            "grasp_candidate_id": plan.get("selected_candidate"),
                            "opening_axis_offset_m": plan.get("opening_axis_offset_m"),
                            "close_standoff_m": plan.get("close_standoff_m"),
                            "kinematic_score": plan.get("score"),
                            "combined_score": plan.get("combined_score"),
                            **probe_summary,
                            "phase_reports": probe.get("phase_reports", []),
                        }
                    )
                completion_probe_plans = [
                    item
                    for item in probe_plans
                    if bool(item.get("grasp_probe_preview", {}).get("feasible", False))
                ]
                completion_probe_plans.sort(
                    key=lambda item: (
                        -float(item.get("probe_score", -1e9)),
                        -float(item.get("score", -1e9)),
                        str(item.get("base_candidate_id")),
                        str(item.get("selected_candidate")),
                    )
                )
                completion_probe_plans = completion_probe_plans[:8]
                completion_records: list[dict[str, Any]] = []
                for plan in completion_probe_plans:
                    completion = self._probe_plan_completion(
                        plan,
                        start_qpos=saved_qpos,
                        start_qvel=saved_qvel,
                        start_ctrl=saved_ctrl,
                        start_state=saved_state,
                        max_steps=250,
                    )
                    plan["completion_probe_preview"] = completion
                    plan["completion_score"] = float(completion.get("completion_score", -1e9))
                    plan["combined_score"] = float(plan.get("combined_score", -1e9)) + float(plan["completion_score"])
                    completion_record = {
                        "base_candidate_id": plan.get("base_candidate_id"),
                        "grasp_candidate_id": plan.get("selected_candidate"),
                        "opening_axis_offset_m": plan.get("opening_axis_offset_m"),
                        "close_standoff_m": plan.get("close_standoff_m"),
                        **completion,
                    }
                    completion_records.append(completion_record)
                    for record in probe_records:
                        if (
                            str(record.get("base_candidate_id")) == str(plan.get("base_candidate_id"))
                            and str(record.get("grasp_candidate_id")) == str(plan.get("selected_candidate"))
                            and float(record.get("opening_axis_offset_m", 0.0)) == float(plan.get("opening_axis_offset_m", 0.0))
                            and float(record.get("close_standoff_m", 0.0)) == float(plan.get("close_standoff_m", 0.0))
                        ):
                            record["completion_probe_preview"] = completion
                            record["completion_score"] = plan["completion_score"]
                            record["combined_score"] = plan["combined_score"]
                            break
                feasible_probe_count = sum(1 for item in probe_records if bool(item.get("feasible", False)))
                completion_success_count = sum(1 for item in completion_records if bool(item.get("success", False)))
                self._grasp_probe_search = {
                    "ok": completion_success_count > 0 or feasible_probe_count > 0,
                    "valid_plan_count": len(valid_plans),
                    "tested_plan_count": len(probe_records),
                    "omitted_valid_plan_count": max(0, len(valid_plans) - len(probe_records)),
                    "feasible_count": feasible_probe_count,
                    "completion_probe_count": len(completion_records),
                    "completion_success_count": completion_success_count,
                    "selection_rule": "prefer_completion_success_then_feasible_probe_then_combined_score",
                    "candidates": probe_records,
                    "completion_candidates": completion_records,
                }
                probed_plan_ids = {
                    (
                        str(item.get("base_candidate_id")),
                        str(item.get("selected_candidate")),
                        float(item.get("opening_axis_offset_m", 0.0)),
                        float(item.get("close_standoff_m", 0.0)),
                    )
                    for item in probe_plans
                }
                selectable_plans = [
                    item
                    for item in valid_plans
                    if (
                        str(item.get("base_candidate_id")),
                        str(item.get("selected_candidate")),
                        float(item.get("opening_axis_offset_m", 0.0)),
                        float(item.get("close_standoff_m", 0.0)),
                    )
                    in probed_plan_ids
                ]
                selectable_plans.sort(
                    key=lambda item: (
                        -int(bool(item.get("completion_probe_preview", {}).get("success", False))),
                        -int(bool(item.get("completion_probe_preview", {}).get("target_placed", False))),
                        -int(bool(item.get("completion_probe_preview", {}).get("released_after_grasp", False))),
                        -int(bool(item.get("grasp_probe_preview", {}).get("feasible", False))),
                        -float(item.get("combined_score", -1e9)),
                        -float(item.get("completion_score", -1e9)),
                        -float(item.get("score", -1e9)),
                        str(item.get("base_candidate_id")),
                        str(item.get("selected_candidate")),
                        float(item.get("opening_axis_offset_m", 0.0)),
                    )
                )
                selected = selectable_plans[0]
                selected_base = selected.get("base_candidate")
                if isinstance(selected_base, dict):
                    self._apply_robot_base_candidate(selected_base)
                self.data.qpos[:] = saved_qpos
                self.data.qvel[:] = saved_qvel
                self.data.ctrl[:] = saved_ctrl
                self._apply_robot_home_qpos()
                self._seed_ctrl_from_qpos()
                self.mujoco.mj_forward(self.model, self.data)
                self._teacher_search["selected"] = {
                    "base_candidate_id": selected.get("base_candidate_id"),
                    "grasp_candidate_id": selected.get("selected_candidate"),
                    "opening_axis_offset_m": selected.get("opening_axis_offset_m"),
                    "close_standoff_m": selected.get("close_standoff_m"),
                    "score": selected.get("score"),
                    "probe_score": selected.get("probe_score"),
                    "completion_score": selected.get("completion_score"),
                    "combined_score": selected.get("combined_score"),
                    "probe_feasible": selected.get("grasp_probe_preview", {}).get("feasible"),
                    "probe_failure_reason": selected.get("grasp_probe_preview", {}).get("failure_reason"),
                    "completion_success": selected.get("completion_probe_preview", {}).get("success"),
                    "completion_terminated_reason": selected.get("completion_probe_preview", {}).get("terminated_reason"),
                }
                self._grasp_probe_search["selected"] = self._teacher_search["selected"]
                return selected
        finally:
            if not valid_plans:
                self.model.body_pos[self.robot_base_body_id] = saved_base_pos
                self.model.body_quat[self.robot_base_body_id] = saved_base_quat
                self.data.qpos[:] = saved_qpos
                self.data.qvel[:] = saved_qvel
                self.data.ctrl[:] = saved_ctrl
                self.mujoco.mj_forward(self.model, self.data)
        if best_failure is None:
            return {
                "ok": False,
                "joint_waypoints": [],
                "failed_phase": None,
                "reason": "no_valid_base_or_grasp_candidates",
                "scanner_dimensions_m": self.scene_ir.object_by_id(self.scene_ir.task.target_object).dimensions,
                "teacher_plan_search": self._teacher_search,
            }
        return {
            **best_failure,
            "ok": False,
            "joint_waypoints": [],
            "reason": str(best_failure.get("reason") or "all_grasp_candidates_failed"),
            "teacher_plan_search": self._teacher_search,
        }

    def _solve_teacher_candidate(self, base_candidate: dict[str, Any], candidate: dict[str, Any], target: np.ndarray, dest: np.ndarray) -> dict[str, Any]:
        frame = np.asarray(candidate["frame"], dtype=float).reshape(3, 3)
        target_xmat = self.data.xmat[self.target_body_id].copy().reshape(3, 3)
        local_grasp_offset = np.asarray(candidate.get("local_grasp_offset", [0.0, 0.0, 0.0]), dtype=float)
        grasp_center = target + target_xmat @ local_grasp_offset
        approach_axis = -frame[:, 2]
        close_standoff = float(candidate.get("close_standoff_m", 0.03))
        preclose_standoff = close_standoff + 0.027
        pregrasp_standoff = close_standoff + 0.115
        safe_standoff = close_standoff + 0.165
        retreat_standoff = close_standoff + 0.045
        grasp_finger_target = float(
            candidate.get(
                "grasp_finger_target_m",
                _grasp_finger_target(candidate.get("grasp_width_m"), self.scene_ir.robot.gripper_max_width_m),
            )
        )
        vertical_approach = abs(float(approach_axis[2])) > 0.70
        safe_extra_z = 0.060 if vertical_approach else 0.180
        pregrasp_extra_z = 0.020 if vertical_approach else 0.120
        preclose_extra_z = 0.0 if vertical_approach else 0.020
        waypoint_specs = [
            ("HOME", self.data.site_xpos[self.ee_site_id].copy(), 0.04),
            ("SAFE_APPROACH", grasp_center + approach_axis * safe_standoff + np.asarray([0.0, 0.0, safe_extra_z]), 0.04),
            ("PREGRASP", grasp_center + approach_axis * pregrasp_standoff + np.asarray([0.0, 0.0, pregrasp_extra_z]), 0.04),
            ("PRE_CLOSE", grasp_center + approach_axis * preclose_standoff + np.asarray([0.0, 0.0, preclose_extra_z]), 0.04),
            ("CLOSE_GRIPPER", grasp_center + approach_axis * close_standoff, grasp_finger_target),
            ("GRASP_SETTLE", grasp_center + approach_axis * close_standoff, grasp_finger_target),
            ("MICRO_LIFT", grasp_center + approach_axis * close_standoff + np.asarray([0.0, 0.0, 0.060]), grasp_finger_target),
            ("RETREAT_LIFT", grasp_center + approach_axis * retreat_standoff + np.asarray([0.0, 0.0, 0.095]), grasp_finger_target),
            ("PLACE_APPROACH", dest + np.asarray([0.0, 0.0, 0.16]), grasp_finger_target),
            ("PLACE_DESCENT", dest + np.asarray([0.0, 0.0, 0.08]), grasp_finger_target),
            ("RELEASE", dest + np.asarray([0.0, 0.0, 0.08]), 0.04),
            ("RETREAT", dest + np.asarray([0.0, 0.0, 0.16]), 0.04),
        ]
        waypoints: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        previous_qpos: np.ndarray | None = None
        max_joint_delta = 0.0
        min_clearance = 0.01
        max_position_error = 0.0
        max_orientation_error = 0.0
        min_manipulability = float("inf")
        for phase, site_target, finger_target in waypoint_specs:
            solved = True
            position_error = 0.0
            orientation_error = 0.0
            ik_result: dict[str, Any] = {
                "ok": True,
                "position_error_m": 0.0,
                "orientation_error_rad": 0.0,
                "reason": "home",
                "manipulability": 1.0,
            }
            if phase != "HOME":
                ik_result = _solve_ik_to_site_diagnostic(
                    self.mujoco,
                    self.model,
                    self.data,
                    self.scene_ir.robot.ee_site,
                    site_target,
                    frame,
                    _ik_options_for_phase(phase),
                )
                solved = bool(ik_result["ok"])
                position_error = float(ik_result["position_error_m"])
                orientation_error = float(ik_result["orientation_error_rad"])
            collision_free = self._robot_scene_contact_free()
            command = [float(self.data.qpos[index]) for index in self.adapter.arm_qpos_ids]
            if len(self.adapter.actuator_ids) == len(self.adapter.arm_qpos_ids) + 1:
                command.append(self._gripper_actuator_value(finger_target))
            else:
                command.extend([float(finger_target)] * len(self.adapter.gripper_qpos_ids))
            arm_qpos = np.asarray(command[:7], dtype=float)
            joint_delta = 0.0 if previous_qpos is None else float(np.max(np.abs(arm_qpos - previous_qpos)))
            previous_qpos = arm_qpos.copy()
            max_joint_delta = max(max_joint_delta, joint_delta)
            max_position_error = max(max_position_error, position_error)
            max_orientation_error = max(max_orientation_error, orientation_error)
            min_manipulability = min(min_manipulability, float(ik_result.get("manipulability", 0.0)))
            min_clearance = min(min_clearance, 0.01 if collision_free else 0.0)
            diagnostic = {
                "phase": phase,
                "desired_ee_position": site_target.tolist(),
                "desired_ee_quaternion_xyzw": _matrix_to_quat_xyzw(frame),
                "solved_joint_qpos": [float(item) for item in command[:7]],
                "ik_converged": bool(solved),
                "position_error_m": float(position_error),
                "orientation_error_rad": float(orientation_error),
                "orientation_error_deg": float(np.degrees(orientation_error)),
                "collision_free": bool(collision_free),
                "joint_limit_ok": True,
                "joint_delta_from_previous": joint_delta,
                "min_static_clearance_m": 0.01 if collision_free else 0.0,
                "manipulability": float(ik_result.get("manipulability", 0.0)),
                "ik_reason": str(ik_result.get("reason", "")),
                "gripper_opening_axis_world": np.asarray(frame[:, 1], dtype=float).tolist(),
            }
            diagnostics.append(diagnostic)
            if not solved:
                score = _score_plan(max_position_error, max_orientation_error, max_joint_delta, min_clearance, min_manipulability, False)
                return {
                    "ok": False,
                    "joint_waypoints": [],
                    "base_candidate": base_candidate,
                    "base_candidate_id": base_candidate["id"],
                    "selected_candidate": candidate["name"],
                    "base_grasp_name": candidate.get("base_grasp_name", candidate["name"]),
                    "failed_phase": phase,
                    "reason": f"ik_failed_{phase.lower()}",
                    "position_error_m": float(position_error),
                    "orientation_error_rad": float(orientation_error),
                    "grasp_axis_world": np.asarray(frame[:, 1], dtype=float).tolist(),
                    "grasp_center_world": grasp_center.tolist(),
                    "grasp_target_label": candidate.get("grasp_target_label"),
                    "opening_axis_offset_m": candidate.get("opening_axis_offset_m", 0.0),
                    "local_grasp_offset": local_grasp_offset.tolist(),
                    "grasp_width_m": candidate.get("grasp_width_m"),
                    "grasp_finger_target_m": grasp_finger_target,
                    "close_standoff_m": close_standoff,
                    "scanner_dimensions_m": self.scene_ir.object_by_id(self.scene_ir.task.target_object).dimensions,
                    "waypoint_diagnostics": diagnostics,
                    "plan_ok": False,
                    "score": score,
                }
            if not collision_free:
                score = _score_plan(max_position_error, max_orientation_error, max_joint_delta, min_clearance, min_manipulability, False)
                return {
                    "ok": False,
                    "joint_waypoints": [],
                    "base_candidate": base_candidate,
                    "base_candidate_id": base_candidate["id"],
                    "selected_candidate": candidate["name"],
                    "base_grasp_name": candidate.get("base_grasp_name", candidate["name"]),
                    "failed_phase": phase,
                    "reason": "approach_collision",
                    "position_error_m": float(position_error),
                    "orientation_error_rad": float(orientation_error),
                    "grasp_axis_world": np.asarray(frame[:, 1], dtype=float).tolist(),
                    "grasp_center_world": grasp_center.tolist(),
                    "grasp_target_label": candidate.get("grasp_target_label"),
                    "opening_axis_offset_m": candidate.get("opening_axis_offset_m", 0.0),
                    "local_grasp_offset": local_grasp_offset.tolist(),
                    "grasp_width_m": candidate.get("grasp_width_m"),
                    "grasp_finger_target_m": grasp_finger_target,
                    "close_standoff_m": close_standoff,
                    "scanner_dimensions_m": self.scene_ir.object_by_id(self.scene_ir.task.target_object).dimensions,
                    "waypoint_diagnostics": diagnostics,
                    "plan_ok": False,
                    "score": score,
                }
            waypoints.append(
                {
                    "phase": phase,
                    "qpos": command,
                    "site_target": site_target.tolist(),
                    "ik_error_m": float(position_error),
                    "orientation_error_rad": float(orientation_error),
                }
            )
        gate = _gate0_report(diagnostics, waypoints)
        score = _score_plan(max_position_error, max_orientation_error, max_joint_delta, min_clearance, min_manipulability, bool(gate["ok"]))
        if not bool(gate["ok"]):
            return {
                "ok": False,
                "joint_waypoints": [],
                "base_candidate": base_candidate,
                "base_candidate_id": base_candidate["id"],
                "selected_candidate": candidate["name"],
                "base_grasp_name": candidate.get("base_grasp_name", candidate["name"]),
                "failed_phase": gate.get("failed_phase"),
                "reason": gate.get("reason"),
                "position_error_m": float(max_position_error),
                "orientation_error_rad": float(max_orientation_error),
                "grasp_axis_world": np.asarray(frame[:, 1], dtype=float).tolist(),
                "grasp_center_world": grasp_center.tolist(),
                "grasp_target_label": candidate.get("grasp_target_label"),
                "opening_axis_offset_m": candidate.get("opening_axis_offset_m", 0.0),
                "local_grasp_offset": local_grasp_offset.tolist(),
                "grasp_width_m": candidate.get("grasp_width_m"),
                "grasp_finger_target_m": grasp_finger_target,
                "close_standoff_m": close_standoff,
                "scanner_dimensions_m": self.scene_ir.object_by_id(self.scene_ir.task.target_object).dimensions,
                "waypoint_diagnostics": diagnostics,
                "gate0": gate,
                "plan_ok": False,
                "score": score,
            }
        return {
            "ok": True,
            "joint_waypoints": waypoints,
            "base_candidate": base_candidate,
            "base_candidate_id": base_candidate["id"],
            "selected_candidate": candidate["name"],
            "base_grasp_name": candidate.get("base_grasp_name", candidate["name"]),
            "failed_phase": None,
            "reason": None,
            "position_error_m": float(max_position_error),
            "orientation_error_rad": float(max_orientation_error),
            "grasp_axis_world": np.asarray(frame[:, 1], dtype=float).tolist(),
            "grasp_center_world": grasp_center.tolist(),
            "grasp_target_label": candidate.get("grasp_target_label"),
            "opening_axis_offset_m": candidate.get("opening_axis_offset_m", 0.0),
            "local_grasp_offset": local_grasp_offset.tolist(),
            "grasp_width_m": candidate.get("grasp_width_m"),
            "grasp_finger_target_m": grasp_finger_target,
            "close_standoff_m": close_standoff,
            "scanner_dimensions_m": self.scene_ir.object_by_id(self.scene_ir.task.target_object).dimensions,
            "waypoint_diagnostics": diagnostics,
            "gate0": gate,
            "plan_ok": True,
            "score": score,
        }

    def dense_state_snapshot(self, step: int, action: Any, metrics: dict[str, Any]) -> dict[str, Any]:
        body_poses: dict[str, Any] = {}
        names = [
            "panda_base",
            "panda_link1",
            "panda_link2",
            "panda_link3",
            "panda_link4",
            "panda_link5",
            "panda_link6",
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
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
            self.scene_ir.task.target_object,
        ]
        if self.scene_ir.task.support_id:
            names.append(self.scene_ir.task.support_id)
        for name in sorted(set(names)):
            body_id = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                continue
            body_poses[name] = {
                "position": self.data.xpos[body_id].copy(),
                "xmat": self.data.xmat[body_id].copy().reshape(3, 3),
            }
        contacts = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            force = np.zeros(6, dtype=float)
            self.mujoco.mj_contactForce(self.model, self.data, index, force)
            contacts.append(
                {
                    "geom1": self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or "",
                    "geom2": self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or "",
                    "body1": self._body_name_for_geom(int(contact.geom1)),
                    "body2": self._body_name_for_geom(int(contact.geom2)),
                    "position": contact.pos.copy(),
                    "force_norm": float(np.linalg.norm(force[:3])),
                }
            )
        return {
            "step": int(step),
            "time_s": float(self.data.time),
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "actuator_ctrl": np.asarray([self.data.ctrl[index] for index in self.adapter.actuator_ids], dtype=float),
            "action": np.asarray(action, dtype=float).copy(),
            "proprio": self.adapter.proprio(),
            "metrics": metrics,
            "body_poses": body_poses,
            "target_pose": body_poses.get(self.scene_ir.task.target_object),
            "destination_position": np.asarray(self.scene_ir.task.destination_position, dtype=float),
            "ee_position": self.data.site_xpos[self.ee_site_id].copy(),
            "gripper_width": self._gripper_width(),
            "contacts": contacts,
            "task_phase": self._task_phase,
            "language_instruction": self.scene_ir.policy.language_instruction
            or f"Pick {self.scene_ir.task.target_object} and place it at the destination.",
        }

    def render_camera(self, camera_id: str, width: int, height: int) -> np.ndarray:
        key = (camera_id, int(width), int(height))
        renderer = self._renderers.get(key)
        if renderer is None:
            renderer = self.mujoco.Renderer(self.model, height=int(height), width=int(width))
            self._renderers[key] = renderer
        renderer.update_scene(self.data, camera=camera_id)
        frame = renderer.render()
        if frame.size == 0 or int(frame.max()) == int(frame.min()):
            raise RuntimeError(f"MuJoCo render produced a blank frame for camera {camera_id}")
        return frame

    def _has_bad_contact(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self._is_bad_contact(int(contact.geom1), int(contact.geom2)):
                return True
        return False

    def _target_contacted_by_both_fingers(self) -> bool:
        return self._target_contacted_by_finger({"panda_leftfinger", "left_finger"}) and self._target_contacted_by_finger({"panda_rightfinger", "right_finger"})

    def _target_contacted_by_finger(self, finger_names: set[str]) -> bool:
        target = self.scene_ir.task.target_object
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = self._body_name_for_geom(int(contact.geom1))
            body2 = self._body_name_for_geom(int(contact.geom2))
            names = {body1, body2}
            if target in names and bool(names & finger_names):
                return True
        return False

    def _target_support_contact(self) -> bool:
        support = self.scene_ir.task.support_id
        if not support:
            return False
        target = self.scene_ir.task.target_object
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = self._body_name_for_geom(int(contact.geom1))
            body2 = self._body_name_for_geom(int(contact.geom2))
            if {body1, body2} == {target, support}:
                return True
        return False

    def _target_contact_force_summary(self) -> dict[str, float]:
        target = self.scene_ir.task.target_object
        support = self.scene_ir.task.support_id
        summary = {
            "left_finger_n": 0.0,
            "right_finger_n": 0.0,
            "support_n": 0.0,
            "other_n": 0.0,
            "total_target_n": 0.0,
        }
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = self._body_name_for_geom(geom1)
            body2 = self._body_name_for_geom(geom2)
            if target not in {body1, body2}:
                continue
            other = body2 if body1 == target else body1
            force = np.zeros(6, dtype=float)
            self.mujoco.mj_contactForce(self.model, self.data, index, force)
            norm = float(np.linalg.norm(force[:3]))
            summary["total_target_n"] += norm
            if other in {"panda_leftfinger", "left_finger"}:
                summary["left_finger_n"] += norm
            elif other in {"panda_rightfinger", "right_finger"}:
                summary["right_finger_n"] += norm
            elif support and other == support:
                summary["support_n"] += norm
            else:
                summary["other_n"] += norm
        return summary

    def _robot_scene_contact_free(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = self._body_name_for_geom(int(contact.geom1))
            body2 = self._body_name_for_geom(int(contact.geom2))
            body1_is_robot = self._is_robot_body(body1)
            body2_is_robot = self._is_robot_body(body2)
            if body1_is_robot == body2_is_robot:
                continue
            scene_body = body2 if body1_is_robot else body1
            if scene_body == self.scene_ir.task.target_object:
                continue
            return False
        return True

    def _support_aligned_dynamic_position(self, obj: Any, position: np.ndarray) -> np.ndarray:
        if not obj.support_id:
            return position
        try:
            support = self.scene_ir.object_by_id(obj.support_id)
        except KeyError:
            return position
        support_top_z = float(support.pose.position[2]) + float(support.dimensions[2]) * 0.5
        bottom_offset = self._collision_bottom_offset(obj)
        aligned = position.copy()
        aligned[2] = support_top_z - bottom_offset + 0.0005
        return aligned

    def _collision_bottom_offset(self, obj: Any) -> float:
        bottoms: list[float] = []
        for spec in obj.collision:
            pos_z = float(spec.pos[2])
            if spec.kind != "primitive" or not spec.size:
                continue
            if spec.primitive_type == "box":
                bottoms.append(pos_z - float(spec.size[2]))
            elif spec.primitive_type == "cylinder":
                bottoms.append(pos_z - float(spec.size[-1]))
            elif spec.primitive_type == "sphere":
                bottoms.append(pos_z - float(spec.size[0]))
        if bottoms:
            return min(bottoms)
        return -float(obj.dimensions[2]) * 0.5

    def _pregrasp_target_drift(self) -> float:
        if self._grasp_attempted:
            return 0.0
        return float(np.linalg.norm(self.data.xpos[self.target_body_id] - self._initial_target_position))

    def _gripper_width(self) -> float:
        if not self.adapter.gripper_qpos_ids:
            return 0.0
        return float(sum(float(self.data.qpos[index]) for index in self.adapter.gripper_qpos_ids))

    def _gripper_actuator_value(self, finger_target: float) -> float:
        if not self.adapter.gripper_actuator_ids:
            return float(finger_target)
        actuator_id = self.adapter.gripper_actuator_ids[-1]
        low, high = self.model.actuator_ctrlrange[actuator_id]
        opening = float(np.clip(finger_target / 0.04, 0.0, 1.0))
        return float(low + (high - low) * opening)

    def _is_bad_contact(self, geom1: int, geom2: int) -> bool:
        body1 = self._body_name_for_geom(geom1)
        body2 = self._body_name_for_geom(geom2)
        body1_is_robot = self._is_robot_body(body1)
        body2_is_robot = self._is_robot_body(body2)
        if body1_is_robot != body2_is_robot:
            robot_body = body1 if body1_is_robot else body2
            scene_body = body2 if body1_is_robot else body1
            if scene_body == self.scene_ir.task.target_object and robot_body in {"panda_leftfinger", "panda_rightfinger", "panda_hand", "left_finger", "right_finger", "hand"}:
                return False
            return True
        target = self.scene_ir.task.target_object
        support = self.scene_ir.task.support_id
        names = {body1, body2}
        if target in names:
            other = body2 if body1 == target else body1
            if support and other == support:
                return False
            if not support and other == "world":
                return False
            return other not in {"floor", "world"}
        return False

    def _is_robot_body(self, body_name: str) -> bool:
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

    def _body_name_for_geom(self, geom_id: int) -> str:
        body_id = int(self.model.geom_bodyid[geom_id])
        return self.mujoco.mj_id2name(self.model, self.mujoco.mjtObj.mjOBJ_BODY, body_id) or "world"


def _body_id(model, name: str) -> int:
    import mujoco

    item_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if item_id < 0:
        raise RuntimeError(f"MuJoCo model is missing body named {name}")
    return int(item_id)


def _side_grasp_frame(target_xmat: np.ndarray, robot_base: np.ndarray, target_position: np.ndarray) -> np.ndarray:
    target_xmat = np.asarray(target_xmat, dtype=float).reshape(3, 3)
    target_position = np.asarray(target_position, dtype=float)
    robot_base = np.asarray(robot_base, dtype=float)
    opening_axis = target_xmat[:, 1].copy()
    opening_axis[2] = 0.0
    if float(np.linalg.norm(opening_axis)) < 1e-6:
        opening_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
    opening_axis = opening_axis / float(np.linalg.norm(opening_axis))

    approach_axis = target_xmat[:, 0].copy()
    approach_axis[2] = 0.0
    if float(np.linalg.norm(approach_axis)) < 1e-6:
        approach_axis = np.asarray([1.0, 0.0, 0.0], dtype=float)
    approach_axis = approach_axis / float(np.linalg.norm(approach_axis))
    target_to_robot = robot_base[:2] - target_position[:2]
    if float(np.dot(approach_axis[:2], target_to_robot)) > 0.0:
        approach_axis = -approach_axis

    x_axis = np.cross(opening_axis, approach_axis)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
    x_axis = x_axis / float(np.linalg.norm(x_axis))
    if x_axis[2] < 0.0:
        approach_axis = -approach_axis
        x_axis = np.cross(opening_axis, approach_axis)
        x_axis = x_axis / float(np.linalg.norm(x_axis))
    opening_axis = np.cross(approach_axis, x_axis)
    opening_axis = opening_axis / float(np.linalg.norm(opening_axis))
    return np.column_stack([x_axis, opening_axis, approach_axis])


def _grasp_frame_candidates(
    target_xmat: np.ndarray,
    robot_base: np.ndarray,
    target_position: np.ndarray,
    category: str = "",
    dimensions: list[float] | tuple[float, float, float] | None = None,
    gripper_max_width_m: float = 0.08,
) -> list[dict[str, Any]]:
    target_xmat = np.asarray(target_xmat, dtype=float).reshape(3, 3)
    robot_base = np.asarray(robot_base, dtype=float)
    target_position = np.asarray(target_position, dtype=float)
    dims = np.asarray(dimensions if dimensions is not None else [0.10, 0.06, 0.06], dtype=float)
    x_axis = _flat_axis(target_xmat[:, 0], np.asarray([1.0, 0.0, 0.0]))
    y_axis = _flat_axis(target_xmat[:, 1], np.asarray([0.0, 1.0, 0.0]))
    z_axis = np.asarray([0.0, 0.0, -1.0], dtype=float)
    to_robot = robot_base[:2] - target_position[:2]
    if category == "scanner":
        grasp_targets = [
            (
                "scanner_body_narrow_axis",
                np.asarray([0.0, -float(dims[1]) * 0.10, float(dims[2]) * 0.02], dtype=float),
                float(dims[1]) * 0.50,
            ),
            (
                "scanner_head_narrow_axis",
                np.asarray([0.0, float(dims[1]) * 0.18, float(dims[2]) * 0.16], dtype=float),
                float(dims[1]) * 0.60,
            ),
        ]
        opening_axis_offsets = [-0.006, -0.003, 0.0, 0.003, 0.006]
        x_width = float(dims[0]) * 0.84
        y_width = float(dims[1]) * 0.64
    else:
        grasp_targets = [("object_center", np.zeros(3, dtype=float), float(min(dims[0], dims[1])))]
        opening_axis_offsets = [0.0]
        x_width = float(dims[0])
        y_width = float(dims[1])

    candidates = [
        ("front_side_grasp", y_axis, y_width, _face_robot(x_axis, to_robot)),
        ("rear_side_grasp", y_axis, y_width, -_face_robot(x_axis, to_robot)),
        ("left_side_grasp", x_axis, x_width, _face_robot(y_axis, to_robot)),
        ("right_side_grasp", x_axis, x_width, -_face_robot(y_axis, to_robot)),
        ("top_down_grasp", y_axis, y_width, z_axis),
    ]
    result: list[dict[str, Any]] = []
    for name, opening_axis, grasp_width, approach_axis in candidates:
        if grasp_width > float(gripper_max_width_m) + 0.012:
            continue
        frame = _frame_from_opening_and_approach(opening_axis, approach_axis)
        depth_variants = [("center", 0.0)] if name == "top_down_grasp" else [("mid", 0.015), ("center", 0.0), ("deep", -0.015)]
        for target_label, local_grasp_offset, target_width in grasp_targets:
            effective_width = min(float(grasp_width), float(target_width))
            if effective_width > float(gripper_max_width_m) + 0.012:
                continue
            for depth_name, close_standoff in depth_variants:
                for opening_axis_offset in opening_axis_offsets:
                    world_offset = frame[:, 1] * float(opening_axis_offset)
                    shifted_local_offset = local_grasp_offset + target_xmat.T @ world_offset
                    offset_suffix = _opening_offset_suffix(float(opening_axis_offset))
                    result.append(
                        {
                            "name": f"{name}_{target_label}_{depth_name}{offset_suffix}",
                            "base_grasp_name": name,
                            "frame": frame,
                            "gripper_opening_axis_world": frame[:, 1],
                            "approach_axis_world": frame[:, 2],
                            "local_grasp_offset": shifted_local_offset,
                            "base_local_grasp_offset": local_grasp_offset,
                            "grasp_target_label": target_label,
                            "opening_axis_offset_m": float(opening_axis_offset),
                            "grasp_width_m": effective_width,
                            "grasp_finger_target_m": _grasp_finger_target(
                                effective_width,
                                gripper_max_width_m,
                            ),
                            "close_standoff_m": close_standoff,
                        }
                    )
    return result


def _opening_offset_suffix(offset_m: float) -> str:
    if abs(float(offset_m)) < 1e-9:
        return ""
    sign = "pos" if offset_m > 0.0 else "neg"
    return f"_open_{sign}{int(round(abs(float(offset_m)) * 1000)):03d}"


def _flat_axis(axis: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    value = np.asarray(axis, dtype=float).copy()
    value[2] = 0.0
    if float(np.linalg.norm(value)) < 1e-6:
        value = np.asarray(fallback, dtype=float)
    return value / float(np.linalg.norm(value))


def _face_robot(axis: np.ndarray, target_to_robot_xy: np.ndarray) -> np.ndarray:
    value = np.asarray(axis, dtype=float)
    if float(np.dot(value[:2], target_to_robot_xy)) < 0.0:
        value = -value
    return value


def _frame_from_opening_and_approach(opening_axis: np.ndarray, approach_axis: np.ndarray) -> np.ndarray:
    opening = np.asarray(opening_axis, dtype=float)
    approach = np.asarray(approach_axis, dtype=float)
    opening = opening / max(float(np.linalg.norm(opening)), 1e-9)
    approach = approach / max(float(np.linalg.norm(approach)), 1e-9)
    x_axis = np.cross(opening, approach)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.asarray([0.0, 0.0, 1.0], dtype=float)
    x_axis = x_axis / float(np.linalg.norm(x_axis))
    if x_axis[2] < 0.0:
        x_axis = -x_axis
    opening = np.cross(approach, x_axis)
    opening = opening / float(np.linalg.norm(opening))
    return np.column_stack([x_axis, opening, approach])


def _rotation_error_rad(current: np.ndarray, desired: np.ndarray) -> float:
    current = np.asarray(current, dtype=float).reshape(3, 3)
    desired = np.asarray(desired, dtype=float).reshape(3, 3)
    vector = 0.5 * (
        np.cross(current[:, 0], desired[:, 0])
        + np.cross(current[:, 1], desired[:, 1])
        + np.cross(current[:, 2], desired[:, 2])
    )
    return float(np.linalg.norm(vector))


def _ik_options_for_phase(phase: str) -> dict[str, float | int | None]:
    if phase == "SAFE_APPROACH":
        return {
            "position_weight": 1.0,
            "z_weight": 2.5,
            "rotation_weight": 0.0,
            "posture_weight": 0.08,
            "position_tolerance_m": 0.06,
            "orientation_tolerance_rad": None,
            "max_iterations": 260,
            "max_step_rad": 0.05,
            "damping": 3e-3,
        }
    if phase == "PREGRASP":
        return {
            "position_weight": 1.0,
            "z_weight": 2.8,
            "rotation_weight": 0.08,
            "posture_weight": 0.05,
            "position_tolerance_m": 0.045,
            "orientation_tolerance_rad": 1.2,
            "max_iterations": 280,
            "max_step_rad": 0.045,
            "damping": 2e-3,
        }
    if phase == "MICRO_LIFT":
        return {
            "position_weight": 1.1,
            "z_weight": 4.0,
            "rotation_weight": 0.55,
            "posture_weight": 0.14,
            "position_tolerance_m": 0.026,
            "orientation_tolerance_rad": 0.28,
            "max_iterations": 260,
            "max_step_rad": 0.025,
            "damping": 3e-3,
        }
    if phase in {"PRE_CLOSE", "CLOSE_GRIPPER", "GRASP_SETTLE"}:
        return {
            "position_weight": 1.2,
            "z_weight": 3.2,
            "rotation_weight": 0.9,
            "posture_weight": 0.02,
            "position_tolerance_m": 0.026,
            "orientation_tolerance_rad": 0.22,
            "max_iterations": 320,
            "max_step_rad": 0.04,
            "damping": 2e-3,
        }
    return {
        "position_weight": 1.0,
        "z_weight": 3.0,
        "rotation_weight": 0.45,
        "posture_weight": 0.03,
        "position_tolerance_m": 0.035,
        "orientation_tolerance_rad": 0.45,
        "max_iterations": 280,
        "max_step_rad": 0.045,
        "damping": 2e-3,
    }


def _gate0_report(diagnostics: list[dict[str, Any]], waypoints: list[dict[str, Any]]) -> dict[str, Any]:
    required = [
        "HOME",
        "SAFE_APPROACH",
        "PREGRASP",
        "PRE_CLOSE",
        "CLOSE_GRIPPER",
        "GRASP_SETTLE",
        "MICRO_LIFT",
        "RETREAT_LIFT",
        "PLACE_APPROACH",
        "PLACE_DESCENT",
        "RELEASE",
        "RETREAT",
    ]
    phases = [str(item.get("phase")) for item in diagnostics]
    missing = [phase for phase in required if phase not in phases]
    if missing:
        return {"ok": False, "reason": "missing_required_phases", "failed_phase": missing[0], "missing_phases": missing}
    if len(waypoints) != len(required):
        return {"ok": False, "reason": "incomplete_waypoint_count", "failed_phase": None}
    for item in diagnostics:
        phase = str(item.get("phase"))
        if not bool(item.get("ik_converged", False)):
            return {"ok": False, "reason": f"ik_failed_{phase.lower()}", "failed_phase": phase}
        if not bool(item.get("collision_free", False)):
            return {"ok": False, "reason": "approach_collision", "failed_phase": phase}
        if not bool(item.get("joint_limit_ok", False)):
            return {"ok": False, "reason": "joint_limit_violation", "failed_phase": phase}
        if float(item.get("joint_delta_from_previous", 0.0)) > 1.4:
            return {"ok": False, "reason": "joint_discontinuity", "failed_phase": phase, "joint_delta": item.get("joint_delta_from_previous")}
        if float(item.get("min_static_clearance_m", 0.0)) < 0.005:
            return {"ok": False, "reason": "low_static_clearance", "failed_phase": phase}
        if phase != "HOME" and float(item.get("manipulability", 0.0)) <= 0.0:
            return {"ok": False, "reason": "singular_or_zero_manipulability", "failed_phase": phase}
    return {
        "ok": True,
        "reason": None,
        "failed_phase": None,
        "max_joint_delta_between_waypoints": max(float(item.get("joint_delta_from_previous", 0.0)) for item in diagnostics),
        "minimum_static_clearance_m": min(float(item.get("min_static_clearance_m", 0.0)) for item in diagnostics),
        "all_joint_limits_respected": True,
        "all_required_phases_present": True,
    }


def _score_plan(max_position_error: float, max_orientation_error: float, max_joint_delta: float, min_clearance: float, min_manipulability: float, gate_ok: bool) -> float:
    score = 0.0
    score -= max_position_error * 8.0
    score -= max_orientation_error * 1.5
    score -= max_joint_delta * 0.2
    score += min_clearance * 10.0
    score += min(min_manipulability, 1.0) * 0.5
    if gate_ok:
        score += 10.0
    return float(score)


def _failure_rank(plan: dict[str, Any]) -> tuple[int, float]:
    diagnostics = plan.get("waypoint_diagnostics", [])
    solved_count = sum(1 for item in diagnostics if bool(item.get("ik_converged", False)) and bool(item.get("collision_free", False)))
    return int(solved_count), float(plan.get("score", -1e9))


def _grasp_finger_target(grasp_width_m: Any, gripper_max_width_m: float) -> float:
    try:
        grasp_width = float(grasp_width_m)
    except (TypeError, ValueError):
        grasp_width = min(float(gripper_max_width_m) * 0.6, 0.048)
    target = grasp_width * 0.04
    return float(np.clip(target, 0.0, min(0.034, float(gripper_max_width_m) * 0.5)))


def _grasp_probe_group_key(plan: dict[str, Any]) -> str:
    return "|".join(
        [
            str(plan.get("base_grasp_name") or plan.get("selected_candidate")),
            str(plan.get("grasp_target_label") or ""),
            f"{float(plan.get('opening_axis_offset_m', 0.0)):.4f}",
            f"{float(plan.get('close_standoff_m', 0.0)):.4f}",
        ]
    )


def _select_grasp_probe_plans(valid_plans: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(
        valid_plans,
        key=lambda item: (
            -float(item.get("score", -1e9)),
            str(item.get("base_candidate_id")),
            str(item.get("selected_candidate")),
            float(item.get("opening_axis_offset_m", 0.0)),
        ),
    )
    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[str, str, float, float]] = set()

    def add(plan: dict[str, Any]) -> None:
        key = (
            str(plan.get("base_candidate_id")),
            str(plan.get("selected_candidate")),
            float(plan.get("opening_axis_offset_m", 0.0)),
            float(plan.get("close_standoff_m", 0.0)),
        )
        if key in selected_keys:
            return
        selected_keys.add(key)
        selected.append(plan)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for plan in ranked:
        grouped.setdefault(_grasp_probe_group_key(plan), []).append(plan)
    for group_name in sorted(grouped):
        add(grouped[group_name][0])
    for plan in ranked:
        if len(selected) >= limit:
            break
        add(plan)
    return selected[:limit]


def _score_grasp_probe(probe: dict[str, Any]) -> float:
    score = 0.0
    if bool(probe.get("feasible", False)):
        score += 100.0
    if bool(probe.get("stable_grasp", False)):
        score += 30.0
    if bool(probe.get("two_finger_contact", False)):
        score += 20.0
    if bool(probe.get("left_contact", False)):
        score += 5.0
    if bool(probe.get("right_contact", False)):
        score += 5.0
    score += min(max(float(probe.get("lift_delta_z_m", 0.0)), 0.0), 0.05) * 400.0
    if probe.get("scanner_table_contact_after_lift") is False:
        score += 4.0
    force_metrics = _grasp_probe_force_metrics(probe)
    score += min(force_metrics["settle_min_finger_force_n"], 2.0) * 3.0
    score += force_metrics["settle_force_balance_ratio"] * 4.0
    score += min(force_metrics["micro_lift_min_finger_force_n"], 1.5) * 6.0
    score += force_metrics["micro_lift_force_balance_ratio"] * 2.0
    score -= min(force_metrics["micro_lift_support_force_n"], 4.0) * 2.5
    if bool(probe.get("grasp_loss", False)):
        score -= 20.0
    failure_reason = str(probe.get("failure_reason") or "")
    if failure_reason == "bilateral_contact_not_stable":
        score -= 6.0
    elif failure_reason in {"micro_lift_failed", "scanner_table_jammed"}:
        score -= 10.0
    elif failure_reason == "grasp_lost_during_lift":
        score -= 25.0
    elif failure_reason == "teacher_plan_unavailable":
        score -= 100.0
    return float(score)


def _score_completion_probe(probe: dict[str, Any]) -> float:
    score = 0.0
    if bool(probe.get("success", False)):
        score += 500.0
    if bool(probe.get("target_placed", False)):
        score += 150.0
    if bool(probe.get("released_after_grasp", False)):
        score += 100.0
    if bool(probe.get("target_lifted", False)):
        score += 60.0
    if bool(probe.get("stable_grasp", False)):
        score += 25.0
    if int(probe.get("collision_count", 0) or 0) == 0:
        score += 20.0
    try:
        score -= max(float(probe.get("target_distance_m", 1.0)), 0.0) * 250.0
    except (TypeError, ValueError):
        score -= 250.0
    reason = str(probe.get("terminated_reason") or "")
    if reason == "success":
        score += 50.0
    elif reason in {"bad_contact", "grasp_lost"}:
        score -= 150.0
    elif reason in {"lift_failure", "placement_failure", "release_failure"}:
        score -= 80.0
    elif reason == "teacher_plan_unavailable":
        score -= 300.0
    return float(score)


def _completion_timeout_reason(metrics: dict[str, Any]) -> str:
    if bool(metrics.get("workspace_violation", False)):
        return "workspace_violation"
    if bool(metrics.get("object_drop", False)):
        return "object_drop"
    if bool(metrics.get("grasp_lost", False)):
        return "grasp_lost"
    if not bool(metrics.get("grasp_attempted", False)):
        return "grasp_not_attempted"
    if not bool(metrics.get("stable_grasp", metrics.get("verified_grasp", False))):
        return "grasp_failure"
    if not bool(metrics.get("target_lifted", False)):
        return "lift_failure"
    if not bool(metrics.get("target_placed", False)):
        return "placement_failure"
    if not bool(metrics.get("released_after_grasp", False)):
        return "release_failure"
    return "stability_timeout"


def _grasp_probe_force_metrics(probe: dict[str, Any]) -> dict[str, float]:
    settle = _phase_contact_force_summary(probe, "GRASP_SETTLE")
    micro_lift = _phase_contact_force_summary(probe, "MICRO_LIFT")
    if not micro_lift:
        phase_reports = list(probe.get("phase_reports", []))
        if phase_reports:
            micro_lift = dict(phase_reports[-1].get("target_contact_force_summary") or {})
    settle_left = _force_value(settle, "left_finger_n")
    settle_right = _force_value(settle, "right_finger_n")
    micro_left = _force_value(micro_lift, "left_finger_n")
    micro_right = _force_value(micro_lift, "right_finger_n")
    return {
        "settle_min_finger_force_n": min(settle_left, settle_right),
        "settle_force_balance_ratio": _force_balance_ratio(settle_left, settle_right),
        "micro_lift_min_finger_force_n": min(micro_left, micro_right),
        "micro_lift_force_balance_ratio": _force_balance_ratio(micro_left, micro_right),
        "micro_lift_support_force_n": _force_value(micro_lift, "support_n"),
    }


def _phase_contact_force_summary(probe: dict[str, Any], phase: str) -> dict[str, Any]:
    for item in reversed(list(probe.get("phase_reports", []))):
        if str(item.get("phase")) == phase:
            return dict(item.get("target_contact_force_summary") or {})
    return {}


def _force_value(summary: dict[str, Any], key: str) -> float:
    try:
        return max(float(summary.get(key, 0.0)), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _force_balance_ratio(left_n: float, right_n: float) -> float:
    larger = max(float(left_n), float(right_n), 0.0)
    if larger <= 1e-9:
        return 0.0
    return float(min(float(left_n), float(right_n)) / larger)


def _grasp_probe_summary(probe: dict[str, Any]) -> dict[str, Any]:
    phase_reports = list(probe.get("phase_reports", []))
    min_ee_target_distance = None
    min_gripper_width = None
    if phase_reports:
        min_ee_target_distance = min(float(item.get("ee_target_distance_m", 1e9)) for item in phase_reports)
        min_gripper_width = min(float(item.get("gripper_width_m", 1e9)) for item in phase_reports)
    force_metrics = _grasp_probe_force_metrics(probe)
    return {
        "feasible": bool(probe.get("feasible", False)),
        "left_contact": bool(probe.get("left_contact", False)),
        "right_contact": bool(probe.get("right_contact", False)),
        "two_finger_contact": bool(probe.get("two_finger_contact", False)),
        "stable_grasp": bool(probe.get("stable_grasp", False)),
        "grasp_loss": bool(probe.get("grasp_loss", False)),
        "scanner_table_contact_after_lift": probe.get("scanner_table_contact_after_lift"),
        "lift_delta_z_m": float(probe.get("lift_delta_z_m", 0.0)),
        "failure_reason": probe.get("failure_reason"),
        "probe_score": float(probe.get("probe_score", _score_grasp_probe(probe))),
        "phase_count": len(phase_reports),
        "min_ee_target_distance_m": min_ee_target_distance,
        "min_gripper_width_m": min_gripper_width,
        **force_metrics,
    }


def _yaw_quat_wxyz(yaw: float) -> np.ndarray:
    half = yaw * 0.5
    return np.asarray([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=float)


def _yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    q = np.asarray(quat, dtype=float).reshape(4)
    w, x, y, z = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _matrix_to_quat_xyzw(matrix: np.ndarray) -> list[float]:
    m = np.asarray(matrix, dtype=float).reshape(3, 3)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = float(np.sqrt(trace + 1.0) * 2.0)
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0)
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s
        elif axis == 1:
            s = float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0)
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s
        else:
            s = float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0)
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=float)
    quat = quat / max(float(np.linalg.norm(quat)), 1e-9)
    return [float(item) for item in quat]


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
