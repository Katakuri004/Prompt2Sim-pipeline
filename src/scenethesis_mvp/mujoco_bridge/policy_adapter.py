from __future__ import annotations

from typing import Sequence

import numpy as np

from scenethesis_mvp.mujoco_bridge.schemas import PolicyContract, RobotSpec


class PolicyAdapter:
    def __init__(self, model, data, robot: RobotSpec, contract: PolicyContract) -> None:
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.robot = robot
        self.contract = contract
        self.arm_joint_ids = [_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in robot.arm_joint_names]
        self.arm_dof_ids = [int(model.jnt_dofadr[joint_id]) for joint_id in self.arm_joint_ids]
        self.arm_qpos_ids = [int(model.jnt_qposadr[joint_id]) for joint_id in self.arm_joint_ids]
        self.gripper_joint_ids = [_name_id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in robot.gripper_joint_names]
        self.gripper_qpos_ids = [int(model.jnt_qposadr[joint_id]) for joint_id in self.gripper_joint_ids]
        self.actuator_ids = [_name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in robot.actuator_names]
        self.actuator_by_name = {name: _name_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in robot.actuator_names}
        self.actuator_by_joint_name: dict[str, int] = {}
        self.gripper_actuator_ids: list[int] = []
        joint_trn = int(mujoco.mjtTrn.mjTRN_JOINT)
        tendon_trn = int(mujoco.mjtTrn.mjTRN_TENDON)
        gripper_joint_set = set(self.gripper_joint_ids)
        for actuator_id in self.actuator_ids:
            trn_type = int(model.actuator_trntype[actuator_id])
            trn_id = int(model.actuator_trnid[actuator_id][0])
            if trn_type == joint_trn and trn_id >= 0:
                joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, trn_id)
                if joint_name:
                    self.actuator_by_joint_name[joint_name] = actuator_id
                if trn_id in gripper_joint_set:
                    self.gripper_actuator_ids.append(actuator_id)
            elif trn_type == tendon_trn:
                self.gripper_actuator_ids.append(actuator_id)
        self.ee_site_id = _name_id(model, mujoco.mjtObj.mjOBJ_SITE, robot.ee_site)
        self.arm_home_qpos = np.asarray(robot.home_qpos[: len(self.arm_qpos_ids)], dtype=float)

    @property
    def action_size(self) -> int:
        if self.contract.action_representation == "joint_position":
            return len(self.actuator_ids)
        return 7

    def apply(self, action: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(action, dtype=float).reshape(-1)
        if len(values) == len(self.actuator_ids):
            return self._apply_joint_position(values)
        if self.contract.action_representation == "joint_position":
            return self._apply_joint_position(values)
        return self._apply_delta_ee(values)

    def _apply_joint_position(self, values: np.ndarray) -> np.ndarray:
        ctrl = self.data.ctrl.copy()
        limit = min(len(values), len(self.actuator_ids))
        for index in range(limit):
            actuator_id = self.actuator_ids[index]
            low, high = self.model.actuator_ctrlrange[actuator_id]
            ctrl[actuator_id] = np.clip(values[index], low, high)
        self.data.ctrl[:] = ctrl
        return ctrl

    def _apply_delta_ee(self, values: np.ndarray) -> np.ndarray:
        if len(values) < 7:
            padded = np.zeros(7, dtype=float)
            padded[: len(values)] = values
            values = padded
        delta = np.clip(values[:3], -self.contract.translation_bound_m, self.contract.translation_bound_m)
        rot_delta = np.clip(values[3:6], -self.contract.rotation_bound_rad, self.contract.rotation_bound_rad)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        self.mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_site_id)
        if float(np.linalg.norm(rot_delta)) <= 1e-9:
            j = jacp[:, self.arm_dof_ids]
            weights = np.diag([1.0, 1.0, 3.0])
            j_eff = weights @ j
            delta_eff = weights @ delta
            damping = 1e-4
            pinv = j_eff.T @ np.linalg.solve(j_eff @ j_eff.T + damping * np.eye(3), np.eye(3))
            dq = pinv @ delta_eff
        else:
            j = np.vstack([jacp[:, self.arm_dof_ids], jacr[:, self.arm_dof_ids]])
            weights = np.diag([1.0, 1.0, 3.0, 0.75, 0.75, 0.75])
            j_eff = weights @ j
            err = np.concatenate([delta, rot_delta])
            err_eff = weights @ err
            damping = 1e-3
            pinv = j_eff.T @ np.linalg.solve(j_eff @ j_eff.T + damping * np.eye(6), np.eye(6))
            dq = pinv @ err_eff
        if len(self.arm_home_qpos) == len(self.arm_qpos_ids):
            q = np.asarray([self.data.qpos[index] for index in self.arm_qpos_ids], dtype=float)
            posture_error = self.arm_home_qpos - q
            posture_step = np.asarray([0.04, 0.18, 0.04, 0.12, 0.04, 0.10, 0.03], dtype=float) * posture_error
            nullspace = np.eye(len(self.arm_qpos_ids)) - pinv @ j_eff
            dq = dq + nullspace @ posture_step
        dq = np.clip(dq, -0.09, 0.09)
        ctrl = self.data.ctrl.copy()
        for index, qpos_id in enumerate(self.arm_qpos_ids):
            actuator_id = self.actuator_by_joint_name.get(self.robot.arm_joint_names[index])
            if actuator_id is None:
                continue
            low, high = self.model.actuator_ctrlrange[actuator_id]
            ctrl[actuator_id] = np.clip(self.data.qpos[qpos_id] + dq[index], low, high)
        grip = float(np.clip(values[6], self.contract.gripper_bounds[0], self.contract.gripper_bounds[1]))
        grip_norm = (grip - self.contract.gripper_bounds[0]) / (self.contract.gripper_bounds[1] - self.contract.gripper_bounds[0])
        finger_target = 0.04 * grip_norm
        joint_gripper_actuators = [self.actuator_by_joint_name[joint_name] for joint_name in self.robot.gripper_joint_names if joint_name in self.actuator_by_joint_name]
        if joint_gripper_actuators:
            for actuator_id in joint_gripper_actuators:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                ctrl[actuator_id] = np.clip(finger_target, low, high)
        else:
            for actuator_id in self.gripper_actuator_ids:
                low, high = self.model.actuator_ctrlrange[actuator_id]
                ctrl[actuator_id] = low + (high - low) * grip_norm
        self.data.ctrl[:] = ctrl
        return ctrl

    def proprio(self) -> np.ndarray:
        qpos = np.asarray([self.data.qpos[index] for index in self.arm_qpos_ids], dtype=float)
        qvel = np.asarray([self.data.qvel[self.model.jnt_dofadr[joint_id]] for joint_id in self.arm_joint_ids], dtype=float)
        fingers = np.asarray([self.data.qpos[index] for index in self.gripper_qpos_ids], dtype=float)
        gripper_width = np.asarray([float(fingers.sum())], dtype=float)
        return np.concatenate([qpos, qvel, gripper_width])


def _name_id(model, obj_type, name: str) -> int:
    import mujoco

    item_id = mujoco.mj_name2id(model, obj_type, name)
    if item_id < 0:
        raise RuntimeError(f"MuJoCo model is missing {obj_type} named {name}")
    return int(item_id)
