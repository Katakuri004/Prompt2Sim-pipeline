from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from scenethesis_mvp.mujoco_bridge.schemas import PolicyContract


class TeacherPlanUnavailable(RuntimeError):
    """Raised when strict teacher evaluation has no executable joint plan."""


class Policy(Protocol):
    id: str

    def reset(self, seed: int | None = None) -> None:
        ...

    def act(self, observation: dict) -> np.ndarray:
        ...


@dataclass
class NoOpPolicy:
    contract: PolicyContract
    id: str = "noop"

    def reset(self, seed: int | None = None) -> None:
        return None

    def act(self, observation: dict) -> np.ndarray:
        return np.zeros(_action_size(self.contract), dtype=float)


@dataclass
class RandomPolicy:
    contract: PolicyContract
    id: str = "random"

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: dict) -> np.ndarray:
        rng = getattr(self, "_rng", np.random.default_rng())
        if self.contract.action_representation == "joint_position":
            return rng.uniform(-0.2, 0.2, size=_action_size(self.contract))
        action = np.zeros(7, dtype=float)
        action[:3] = rng.uniform(-self.contract.translation_bound_m, self.contract.translation_bound_m, size=3)
        action[3:6] = rng.uniform(-self.contract.rotation_bound_rad, self.contract.rotation_bound_rad, size=3)
        action[6] = rng.uniform(self.contract.gripper_bounds[0], self.contract.gripper_bounds[1])
        return action


@dataclass
class ScriptedPickPlacePolicy:
    contract: PolicyContract
    id: str = "scripted_pick_place"

    def reset(self, seed: int | None = None) -> None:
        self.phase = "PRE_GRASP"
        self.hold_count = 0
        self.phase_steps = 0

    def act(self, observation: dict) -> np.ndarray:
        if self.contract.action_representation == "joint_position":
            return np.zeros(_action_size(self.contract), dtype=float)
        state = observation.get("state", {})
        ee = np.asarray(state.get("ee_position", np.zeros(3)), dtype=float)
        target = np.asarray(state.get("target_position", np.zeros(3)), dtype=float)
        dest = np.asarray(state.get("destination_position", np.zeros(3)), dtype=float)
        if not hasattr(self, "_initial_target_z"):
            self._initial_target_z = float(target[2])
        phase = str(getattr(self, "phase", "PRE_GRASP"))
        verified_grasp = bool(state.get("verified_grasp", False))
        two_finger_contact = bool(state.get("two_finger_contact", False))
        target_lifted = float(target[2]) > float(self._initial_target_z) + 0.04
        self.phase_steps = int(getattr(self, "phase_steps", 0)) + 1
        waypoint, gripper, tolerance, hold_required = self._command_for_phase(phase, target, dest)
        delta = waypoint - ee
        reached = float(np.linalg.norm(delta)) < tolerance
        self.hold_count = int(getattr(self, "hold_count", 0)) + 1 if reached else 0
        next_phase = phase
        if phase == "PRE_GRASP" and reached and self.hold_count >= hold_required:
            next_phase = "ALIGN"
        elif phase == "ALIGN" and reached and self.hold_count >= hold_required:
            next_phase = "DESCEND"
        elif phase == "DESCEND" and reached and self.hold_count >= hold_required:
            next_phase = "CLOSE_GRIPPER"
        elif phase == "CLOSE_GRIPPER" and self.phase_steps >= 10:
            next_phase = "GRASP_VERIFY"
        elif phase == "GRASP_VERIFY" and (two_finger_contact or verified_grasp):
            next_phase = "LIFT"
        elif phase == "LIFT" and (target_lifted or (reached and self.hold_count >= hold_required)):
            next_phase = "MOVE_TO_PLACE"
        elif phase == "MOVE_TO_PLACE" and reached and self.hold_count >= hold_required:
            next_phase = "DESCEND_TO_PLACE"
        elif phase == "DESCEND_TO_PLACE" and reached and self.hold_count >= hold_required:
            next_phase = "RELEASE"
        elif phase == "RELEASE" and self.phase_steps >= 10:
            next_phase = "RETREAT"
        elif phase == "RETREAT" and reached and self.hold_count >= hold_required:
            next_phase = "STABILITY_VERIFY"
        if next_phase != phase:
            self.phase = next_phase
            self.phase_steps = 0
            self.hold_count = 0
            waypoint, gripper, _tolerance, _hold_required = self._command_for_phase(next_phase, target, dest)
            delta = waypoint - ee
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(delta, -self.contract.translation_bound_m, self.contract.translation_bound_m)
        action[6] = gripper
        return action

    def _command_for_phase(self, phase: str, target: np.ndarray, dest: np.ndarray) -> tuple[np.ndarray, float, float, int]:
        if phase == "PRE_GRASP":
            return target + np.asarray([0.0, 0.0, 0.18]), 1.0, 0.035, 3
        if phase == "ALIGN":
            return target + np.asarray([0.0, 0.0, 0.12]), 1.0, 0.030, 3
        if phase == "DESCEND":
            return target + np.asarray([0.0, 0.0, 0.035]), 1.0, 0.025, 2
        if phase == "CLOSE_GRIPPER":
            return target + np.asarray([0.0, 0.0, 0.035]), -1.0, 0.030, 10
        if phase == "GRASP_VERIFY":
            return target + np.asarray([0.0, 0.0, 0.04]), -1.0, 0.030, 1
        if phase == "LIFT":
            return target + np.asarray([0.0, 0.0, 0.18]), -1.0, 0.040, 2
        if phase == "MOVE_TO_PLACE":
            return dest + np.asarray([0.0, 0.0, 0.18]), -1.0, 0.045, 3
        if phase == "DESCEND_TO_PLACE":
            return dest + np.asarray([0.0, 0.0, 0.055]), -1.0, 0.035, 2
        if phase == "RELEASE":
            return dest + np.asarray([0.0, 0.0, 0.055]), 1.0, 0.040, 10
        if phase == "RETREAT":
            return dest + np.asarray([0.0, 0.0, 0.18]), 1.0, 0.045, 3
        return dest + np.asarray([0.0, 0.0, 0.20]), 1.0, 0.060, 1


@dataclass
class TeacherPickPlacePolicy:
    contract: PolicyContract
    id: str = "teacher_pick_place"
    require_joint_plan: bool = True

    def reset(self, seed: int | None = None) -> None:
        self.phase = "PRE_GRASP_HIGH"
        self.hold_count = 0
        self.phase_steps = 0
        self._initial_target_z: float | None = None
        self._desired_frame: np.ndarray | None = None
        self._grasp_anchor: np.ndarray | None = None
        self._approach_offset: np.ndarray | None = None
        self._joint_waypoint_index = 0

    def act(self, observation: dict) -> np.ndarray:
        state = observation.get("state", {})
        target = np.asarray(state.get("target_position", np.zeros(3)), dtype=float)
        if self._initial_target_z is None:
            self._initial_target_z = float(target[2])
        teacher_waypoints = state.get("teacher_joint_waypoints") or []
        if teacher_waypoints:
            return self._act_joint_waypoints(observation, state, teacher_waypoints)
        if self.contract.action_representation == "joint_position":
            raise TeacherPlanUnavailable("Joint-position teacher has no valid waypoint plan")
        ee = np.asarray(state.get("ee_position", np.zeros(3)), dtype=float)
        dest = np.asarray(state.get("destination_position", np.zeros(3)), dtype=float)
        ee_xmat = np.asarray(state.get("ee_xmat", np.eye(3)), dtype=float).reshape(3, 3)
        verified_grasp = bool(state.get("verified_grasp", False))
        stable_grasp = bool(state.get("stable_grasp", False))
        two_finger_contact = bool(state.get("two_finger_contact", False))
        gripper_width = float(state.get("gripper_width", 0.08))
        if self.require_joint_plan:
            plan = state.get("teacher_plan") or {}
            reason = str(plan.get("reason") or "No valid real-Panda joint waypoint plan")
            failed_phase = plan.get("failed_phase")
            if failed_phase:
                reason = f"{reason}; failed_phase={failed_phase}"
            raise TeacherPlanUnavailable(reason)
        target_xmat = np.asarray(state.get("target_xmat", np.eye(3)), dtype=float).reshape(3, 3)
        target_grasp_frame = _top_down_grasp_frame(target_xmat)
        if self._desired_frame is None:
            self._desired_frame = target_grasp_frame.copy()
        if self._approach_offset is None:
            offset_xy = ee[:2] - target[:2]
            norm = float(np.linalg.norm(offset_xy))
            if norm < 1e-6:
                self._approach_offset = np.zeros(3, dtype=float)
            else:
                offset_xy = offset_xy / norm
                self._approach_offset = np.asarray([offset_xy[0], offset_xy[1], 0.0], dtype=float) * 0.18
        if self._grasp_anchor is None and stable_grasp:
            self._grasp_anchor = target.copy()
            self._grasp_anchor[2] = max(float(self._grasp_anchor[2]), float(self._initial_target_z))
        self.phase_steps += 1
        phase = str(self.phase)
        ee_target_distance = float(np.linalg.norm(ee - target))
        desired_frame = self._desired_frame
        orientation_ready = _orientation_close(ee_xmat, desired_frame, threshold=0.16)
        if phase in {"DESCEND_CLEAR", "PRE_CLOSE"} and orientation_ready and ee_target_distance < 0.10:
            self.phase = "CLOSE_GRIPPER"
            self.phase_steps = 0
            self.hold_count = 0
            phase = "CLOSE_GRIPPER"
        command_target = self._command_target(phase, target)
        waypoint, gripper, tolerance, hold_required = self._command_for_phase(phase, command_target, dest)
        delta = waypoint - ee
        reached = float(np.linalg.norm(delta)) < tolerance
        self.hold_count = self.hold_count + 1 if reached else 0
        next_phase = phase
        if phase == "PRE_GRASP_HIGH" and reached and self.hold_count >= hold_required:
            next_phase = "PRE_GRASP_OFFSET"
        elif phase == "PRE_GRASP_OFFSET" and reached and self.hold_count >= hold_required:
            next_phase = "ALIGN_ORIENTATION"
        elif phase == "ALIGN_ORIENTATION" and reached and orientation_ready:
            next_phase = "DESCEND_CLEAR"
        elif phase == "DESCEND_CLEAR" and (reached or ee_target_distance < 0.13):
            next_phase = "PRE_CLOSE"
        elif phase == "PRE_CLOSE" and orientation_ready and (reached or ee_target_distance < 0.10):
            next_phase = "CLOSE_GRIPPER"
        elif phase == "CLOSE_GRIPPER" and stable_grasp:
            next_phase = "GRASP_SETTLE"
        elif phase == "GRASP_SETTLE" and stable_grasp and self.phase_steps >= 8:
            next_phase = "MICRO_LIFT"
        elif phase == "MICRO_LIFT" and stable_grasp and float(target[2]) > float(self._initial_target_z) + 0.018:
            next_phase = "RETREAT_LIFT"
        elif phase == "RETREAT_LIFT" and stable_grasp and float(target[2]) > float(self._initial_target_z) + 0.055:
            next_phase = "MOVE_TO_PLACE"
        elif phase == "MOVE_TO_PLACE" and stable_grasp and reached and self.hold_count >= hold_required:
            next_phase = "DESCEND_TO_PLACE"
        elif phase == "DESCEND_TO_PLACE" and stable_grasp and reached and self.hold_count >= hold_required:
            next_phase = "RELEASE"
        elif phase == "RELEASE" and self.phase_steps >= 12:
            next_phase = "RETREAT"
        elif phase == "RETREAT" and reached and self.hold_count >= hold_required:
            next_phase = "STABILITY_WAIT"
        if next_phase != phase:
            self.phase = next_phase
            self.phase_steps = 0
            self.hold_count = 0
            command_target = self._command_target(next_phase, target)
            waypoint, gripper, _tolerance, _hold_required = self._command_for_phase(next_phase, command_target, dest)
            delta = waypoint - ee
        action = np.zeros(7, dtype=float)
        action[:3] = np.clip(delta, -self.contract.translation_bound_m, self.contract.translation_bound_m)
        if self.phase == "PRE_GRASP_HIGH":
            action[3:6] = 0.0
        else:
            rotation_error = _rotation_error_vector(ee_xmat, desired_frame)
            action[3:6] = np.clip(rotation_error, -self.contract.rotation_bound_rad, self.contract.rotation_bound_rad)
        action[6] = gripper
        return action

    def _act_joint_waypoints(self, observation: dict, state: dict, waypoints: list[dict]) -> np.ndarray:
        proprio = np.asarray(observation.get("proprio", np.zeros(15)), dtype=float)
        qpos = proprio[:7] if len(proprio) >= 7 else np.zeros(7, dtype=float)
        target = np.asarray(state.get("target_position", np.zeros(3)), dtype=float)
        gripper_width = float(state.get("gripper_width", 0.08))
        two_finger_contact = bool(state.get("two_finger_contact", False))
        verified_grasp = bool(state.get("verified_grasp", False))
        stable_grasp = bool(state.get("stable_grasp", False))
        self.phase_steps += 1
        index = min(int(self._joint_waypoint_index), len(waypoints) - 1)
        waypoint = waypoints[index]
        command = np.asarray(waypoint.get("qpos", np.zeros(8)), dtype=float)
        phase = str(waypoint.get("phase", "JOINT_WAYPOINT"))
        self.phase = phase
        arm_error = float(np.linalg.norm(command[:7] - qpos))
        advance = False
        if phase == "CLOSE_GRIPPER":
            advance = stable_grasp
        elif phase == "GRASP_SETTLE":
            advance = bool(stable_grasp and self.phase_steps >= 8)
        elif phase == "MICRO_LIFT":
            advance = bool(stable_grasp and self._initial_target_z is not None and float(target[2]) > float(self._initial_target_z) + 0.018)
        elif phase == "RETREAT_LIFT":
            advance = bool(stable_grasp and self._initial_target_z is not None and float(target[2]) > float(self._initial_target_z) + 0.045)
        elif phase in {"MOVE_TO_PLACE", "DESCEND_TO_PLACE", "PLACE_APPROACH", "PLACE_DESCENT"}:
            advance = bool(stable_grasp and arm_error < 0.08 and self.phase_steps >= 5)
        elif phase == "RELEASE":
            advance = bool(stable_grasp and self.phase_steps >= 12)
        else:
            advance = (arm_error < 0.08 and self.phase_steps >= 5) or self.phase_steps >= 60
        if advance and index < len(waypoints) - 1:
            self._joint_waypoint_index = index + 1
            self.phase_steps = 0
            waypoint = waypoints[self._joint_waypoint_index]
            self.phase = str(waypoint.get("phase", "JOINT_WAYPOINT"))
            command = np.asarray(waypoint.get("qpos", command), dtype=float)
        streamed = command.copy()
        streamed[:7] = qpos + np.clip(command[:7] - qpos, -0.035, 0.035)
        return streamed

    def _command_target(self, phase: str, target: np.ndarray) -> np.ndarray:
        if self._grasp_anchor is not None and phase in {
            "CLOSE_GRIPPER",
            "GRASP_SETTLE",
            "MICRO_LIFT",
            "RETREAT_LIFT",
            "MOVE_TO_PLACE",
            "DESCEND_TO_PLACE",
            "RELEASE",
            "RETREAT",
            "STABILITY_WAIT",
        }:
            return self._grasp_anchor
        if self._approach_offset is not None:
            if phase in {"PRE_GRASP_HIGH", "PRE_GRASP_OFFSET", "ALIGN_ORIENTATION"}:
                return target + self._approach_offset
            if phase == "DESCEND_CLEAR":
                return target + self._approach_offset * 0.65
            if phase == "PRE_CLOSE":
                return target + self._approach_offset * 0.25
        return target

    def _command_for_phase(self, phase: str, target: np.ndarray, dest: np.ndarray) -> tuple[np.ndarray, float, float, int]:
        if phase == "PRE_GRASP_HIGH":
            return target + np.asarray([0.0, 0.0, 0.32]), 1.0, 0.070, 2
        if phase == "PRE_GRASP_OFFSET":
            return target + np.asarray([0.0, 0.0, 0.24]), 1.0, 0.060, 2
        if phase == "ALIGN_ORIENTATION":
            return target + np.asarray([0.0, 0.0, 0.20]), 1.0, 0.060, 1
        if phase == "DESCEND_CLEAR":
            return target + np.asarray([0.0, 0.0, 0.16]), 1.0, 0.050, 1
        if phase == "PRE_CLOSE":
            return target + np.asarray([0.0, 0.0, 0.045]), 1.0, 0.035, 1
        if phase == "CLOSE_GRIPPER":
            return target + np.asarray([0.0, 0.0, 0.035]), -1.0, 0.045, 18
        if phase == "GRASP_SETTLE":
            return target + np.asarray([0.0, 0.0, 0.035]), -1.0, 0.045, 8
        if phase == "MICRO_LIFT":
            return target + np.asarray([0.0, 0.0, 0.075]), -1.0, 0.050, 2
        if phase == "RETREAT_LIFT":
            return target + np.asarray([0.0, 0.0, 0.16]), -1.0, 0.055, 2
        if phase == "MOVE_TO_PLACE":
            return dest + np.asarray([0.0, 0.0, 0.16]), -1.0, 0.060, 2
        if phase == "DESCEND_TO_PLACE":
            return dest + np.asarray([0.0, 0.0, 0.075]), -1.0, 0.045, 2
        if phase == "RELEASE":
            return dest + np.asarray([0.0, 0.0, 0.075]), 1.0, 0.050, 12
        if phase == "RETREAT":
            return dest + np.asarray([0.0, 0.0, 0.16]), 1.0, 0.055, 2
        return dest + np.asarray([0.0, 0.0, 0.16]), 1.0, 0.060, 1


class CallablePolicy:
    def __init__(self, policy_id: str, fn) -> None:
        self.id = policy_id
        self.fn = fn

    def reset(self, seed: int | None = None) -> None:
        if hasattr(self.fn, "reset"):
            self.fn.reset(seed)

    def act(self, observation: dict) -> np.ndarray:
        return np.asarray(self.fn(observation), dtype=float)


def make_policy(policy_id: str, contract: PolicyContract, policy_path: str | Path | None = None) -> Policy:
    if policy_id == "noop":
        return NoOpPolicy(contract)
    if policy_id == "random":
        policy = RandomPolicy(contract)
        policy.reset(None)
        return policy
    if policy_id == "scripted_pick_place":
        policy = ScriptedPickPlacePolicy(contract)
        policy.reset(None)
        return policy
    if policy_id == "teacher_pick_place":
        policy = TeacherPickPlacePolicy(contract)
        policy.reset(None)
        return policy
    if policy_id == "teacher_delta_debug":
        policy = TeacherPickPlacePolicy(contract, id="teacher_delta_debug", require_joint_plan=False)
        policy.reset(None)
        return policy
    if policy_id == "lerobot":
        if policy_path is None:
            raise ValueError("--policy-path is required when --policy lerobot")
        from scenethesis_mvp.mujoco_bridge.lerobot_policy import LeRobotPolicy

        return LeRobotPolicy(contract, policy_path=policy_path)
    raise ValueError(f"unknown policy: {policy_id}")


def _action_size(contract: PolicyContract) -> int:
    if contract.action_representation == "joint_position":
        return 9
    return 7


def _top_down_grasp_frame(target_xmat: np.ndarray) -> np.ndarray:
    x_axis = np.asarray([0.0, 0.0, -1.0], dtype=float)
    y_axis = np.asarray(target_xmat[:, 1], dtype=float)
    y_axis[2] = 0.0
    if float(np.linalg.norm(y_axis)) < 1e-6:
        y_axis = np.asarray([0.0, 1.0, 0.0], dtype=float)
    y_axis = y_axis / float(np.linalg.norm(y_axis))
    z_axis = np.cross(x_axis, y_axis)
    z_axis = z_axis / float(np.linalg.norm(z_axis))
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack([x_axis, y_axis, z_axis])


def _rotation_error_vector(current: np.ndarray, desired: np.ndarray) -> np.ndarray:
    current = np.asarray(current, dtype=float).reshape(3, 3)
    desired = np.asarray(desired, dtype=float).reshape(3, 3)
    return 0.5 * (
        np.cross(current[:, 0], desired[:, 0])
        + np.cross(current[:, 1], desired[:, 1])
        + np.cross(current[:, 2], desired[:, 2])
    )


def _orientation_close(current: np.ndarray, desired: np.ndarray, threshold: float) -> bool:
    return bool(float(np.linalg.norm(_rotation_error_vector(current, desired))) <= float(threshold))
