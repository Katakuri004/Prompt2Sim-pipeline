from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.mujoco_bridge.schemas import PolicyContract


class LeRobotPolicy:
    id = "lerobot"

    def __init__(self, contract: PolicyContract, policy_path: str | Path, device: str | None = None) -> None:
        self.contract = contract
        self.policy_path = Path(policy_path)
        if not self.policy_path.exists():
            raise FileNotFoundError(f"LeRobot checkpoint path does not exist: {self.policy_path}")
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional dependency.
            raise RuntimeError("LeRobot policy evaluation requires PyTorch and LeRobot to be installed.") from exc
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = self._load_policy()
        if hasattr(self.policy, "to"):
            self.policy = self.policy.to(self.device)
        if hasattr(self.policy, "eval"):
            self.policy.eval()
        self._queued_actions: list[np.ndarray] = []

    def reset(self, seed: int | None = None) -> None:
        self._queued_actions.clear()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        if seed is not None:
            self.torch.manual_seed(int(seed))

    def act(self, observation: dict[str, Any]) -> np.ndarray:
        if self._queued_actions:
            return self._queued_actions.pop(0)
        batch = self._build_batch(observation)
        with self.torch.no_grad():
            if hasattr(self.policy, "select_action"):
                output = self.policy.select_action(batch)
            else:
                output = self.policy(batch)
        actions = self._normalize_policy_output(output)
        if len(actions) > 1:
            self._queued_actions.extend(actions[1:])
        return actions[0]

    def _load_policy(self) -> Any:
        errors: list[str] = []
        for module_name in (
            "lerobot.policies.act.modeling_act",
            "lerobot.common.policies.act.modeling_act",
        ):
            try:
                module = __import__(module_name, fromlist=["ACTPolicy"])
                policy_cls = getattr(module, "ACTPolicy")
                if hasattr(policy_cls, "from_pretrained"):
                    return policy_cls.from_pretrained(str(self.policy_path))
            except Exception as exc:  # pragma: no cover - depends on external LeRobot version.
                errors.append(f"{module_name}: {exc}")
        raise RuntimeError(
            "Could not load a LeRobot ACT checkpoint. Expected ACTPolicy.from_pretrained(path) in the "
            "installed LeRobot package. Loader errors: " + " | ".join(errors)
        )

    def _build_batch(self, observation: dict[str, Any]) -> dict[str, Any]:
        state = np.concatenate(
            [
                np.asarray(observation.get("proprio", []), dtype=np.float32).reshape(-1),
                np.asarray(observation.get("state", {}).get("ee_position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1),
                np.asarray(observation.get("state", {}).get("target_position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1),
                np.asarray(observation.get("state", {}).get("destination_position", [0.0, 0.0, 0.0]), dtype=np.float32).reshape(-1),
            ]
        )
        batch: dict[str, Any] = {
            "observation.state": self.torch.as_tensor(state, dtype=self.torch.float32, device=self.device).unsqueeze(0),
            "task": [str(observation.get("language_instruction") or "")],
        }
        rgb = observation.get("rgb", {})
        required = [camera.id for camera in self.contract.observation_cameras]
        missing = [camera_id for camera_id in required if camera_id not in rgb]
        if missing:
            raise RuntimeError(
                f"LeRobot policy requires RGB observations for {missing}. Run evaluation with --render-rgb."
            )
        for camera_id, frame in sorted(rgb.items()):
            array = np.asarray(frame, dtype=np.float32)
            if array.ndim != 3 or array.shape[2] < 3:
                raise RuntimeError(f"RGB observation for {camera_id} is not an HWC image.")
            tensor = self.torch.as_tensor(array[:, :, :3] / 255.0, dtype=self.torch.float32, device=self.device)
            batch[f"observation.images.{camera_id}"] = tensor.permute(2, 0, 1).unsqueeze(0)
        return batch

    def _normalize_policy_output(self, output: Any) -> list[np.ndarray]:
        if isinstance(output, dict):
            output = output.get("action", output.get("actions"))
        if output is None:
            raise RuntimeError("LeRobot policy returned no action.")
        if hasattr(output, "detach"):
            output = output.detach().cpu().numpy()
        array = np.asarray(output, dtype=np.float32)
        if array.ndim == 1:
            actions = [array]
        elif array.ndim == 2:
            actions = [row for row in array]
        elif array.ndim == 3:
            actions = [row for row in array[0]]
        else:
            raise RuntimeError(f"Unsupported LeRobot action output shape: {array.shape}")
        expected = 9 if self.contract.action_representation == "joint_position" else 7
        normalized: list[np.ndarray] = []
        for action in actions:
            action = np.asarray(action, dtype=float).reshape(-1)
            if len(action) != expected:
                raise RuntimeError(
                    f"LeRobot policy returned action dimension {len(action)}, but MuJoCo contract expects {expected}."
                )
            normalized.append(action)
        return normalized
