from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.mujoco_bridge.mujoco_env import MujocoSceneEnv
from scenethesis_mvp.mujoco_bridge.schemas import SceneIR
from scenethesis_mvp.utils.io import write_json


def validate_physics_settling(
    scene_ir: SceneIR,
    model_path: str | Path,
    out_dir: str | Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    cfg = config.get("settling", {})
    duration_s = float(cfg.get("duration_s", 2.0))
    max_drift = float(cfg.get("max_target_drift_m", 0.002))
    max_force = float(cfg.get("max_contact_force_n", 80.0))
    timestep = float(config.get("rollout", {}).get("timestep", 0.002))
    steps = max(1, int(duration_s / max(timestep, 1e-6)))
    env = MujocoSceneEnv(model_path, scene_ir, physics_steps_per_action=1, render_rgb=False)
    try:
        env.reset(int(config.get("rollout", {}).get("seed", 7)))
        start = env.data.xpos[env.target_body_id].copy()
        max_contact_force = 0.0
        finite = True
        for _ in range(steps):
            env.mujoco.mj_step(env.model, env.data)
            if not np.isfinite(env.data.qpos).all() or not np.isfinite(env.data.qvel).all():
                finite = False
                break
            for index in range(env.data.ncon):
                force = np.zeros(6, dtype=float)
                env.mujoco.mj_contactForce(env.model, env.data, index, force)
                norm = float(np.linalg.norm(force[:3]))
                if np.isfinite(norm):
                    max_contact_force = max(max_contact_force, norm)
        end = env.data.xpos[env.target_body_id].copy()
        drift = float(np.linalg.norm(end - start))
        report = {
            "ok": bool(finite and drift <= max_drift and max_contact_force <= max_force),
            "duration_s": duration_s,
            "steps": steps,
            "target_object": scene_ir.task.target_object,
            "target_start": start.tolist(),
            "target_end": end.tolist(),
            "target_drift_m": drift,
            "max_allowed_target_drift_m": max_drift,
            "max_contact_force_n": max_contact_force,
            "max_allowed_contact_force_n": max_force,
            "finite_state": finite,
            "failure_reasons": [],
        }
        if drift > max_drift:
            report["failure_reasons"].append(f"target drift {drift:.6f}m exceeds {max_drift:.6f}m during settling")
        if max_contact_force > max_force:
            report["failure_reasons"].append(f"contact force {max_contact_force:.3f}N exceeds {max_force:.3f}N during settling")
        if not finite:
            report["failure_reasons"].append("MuJoCo state became non-finite during settling")
    finally:
        env.close()
    write_json(Path(out_dir) / "physics_settling_report.json", report)
    return report
