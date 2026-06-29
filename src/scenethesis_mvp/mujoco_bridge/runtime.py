from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import read_json
from scenethesis_mvp.utils.paths import resolve_path


class MujocoRuntimeCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    ok: bool
    detail: str


class MujocoRuntimeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    checks: list[MujocoRuntimeCheck] = Field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [f"{check.name}: {check.detail}" for check in self.checks if not check.ok]


def validate_mujoco_runtime(
    config: dict[str, Any],
    root: Path,
    run_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    registry: AssetRegistry | None = None,
    scene: SceneSpec | None = None,
    require_coacd: bool = False,
) -> MujocoRuntimeReport:
    checks: list[MujocoRuntimeCheck] = []
    checks.append(_check_import("mujoco", "MuJoCo Python package"))
    if str(config.get("visual_scene", {}).get("mode", "proxy")) == "full_glb_visual":
        checks.append(_check_import("trimesh", "trimesh GLB inspection package"))
        if importlib.util.find_spec("pxr") is not None:
            checks.append(MujocoRuntimeCheck(name="import.pxr", ok=True, detail="USD validation package importable"))
        else:
            checks.append(MujocoRuntimeCheck(name="import.pxr", ok=True, detail="USD validation package not installed; USD cross-check will be skipped"))
    if require_coacd or bool(config.get("collision", {}).get("use_coacd", False)):
        checks.append(_check_import("coacd", "COACD convex decomposition package"))
    checks.append(_check_import("imageio", "imageio replay package"))

    robot_path = resolve_path(config.get("robot", {}).get("mjcf_path", ""), root)
    checks.append(
        MujocoRuntimeCheck(
            name="robot.mjcf_path",
            ok=robot_path.is_file(),
            detail=str(robot_path) if robot_path.is_file() else f"missing robot MJCF: {robot_path}",
        )
    )

    if run_dir is not None:
        target = Path(run_dir)
        if not target.is_absolute():
            target = root / target
        checks.extend(_check_run_dir(target))

    if out_dir is not None:
        target_out = Path(out_dir)
        if not target_out.is_absolute():
            target_out = root / target_out
        try:
            target_out.mkdir(parents=True, exist_ok=True)
            probe = target_out / ".mujoco_write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            checks.append(MujocoRuntimeCheck(name="out_dir.writable", ok=True, detail=str(target_out)))
        except Exception as exc:
            checks.append(MujocoRuntimeCheck(name="out_dir.writable", ok=False, detail=str(exc)))

    if registry is not None and scene is not None:
        checks.extend(_check_scene_meshes(scene, registry))

    return MujocoRuntimeReport(ok=all(check.ok for check in checks), checks=checks)


def raise_if_unavailable(report: MujocoRuntimeReport) -> None:
    if report.ok:
        return
    details = "\n- ".join(report.errors)
    raise RuntimeError(f"MuJoCo evaluation runtime requirements are not satisfied:\n- {details}")


def _check_import(module: str, purpose: str) -> MujocoRuntimeCheck:
    ok = importlib.util.find_spec(module) is not None
    return MujocoRuntimeCheck(
        name=f"import.{module}",
        ok=ok,
        detail=f"{purpose} importable" if ok else f"{purpose} is not installed",
    )


def _check_run_dir(run_dir: Path) -> list[MujocoRuntimeCheck]:
    checks: list[MujocoRuntimeCheck] = []
    required = ["scene_spec.json", "scene.glb", "qualification.json"]
    for name in required:
        path = run_dir / name
        ok = path.is_file()
        checks.append(
            MujocoRuntimeCheck(
                name=f"run_dir.{name}",
                ok=ok,
                detail=str(path) if ok else f"missing or invalid required artifact: {path}",
            )
        )
    qualification_path = run_dir / "qualification.json"
    if qualification_path.is_file():
        try:
            qualification = read_json(qualification_path)
            accepted = qualification.get("accepted") is True
            checks.append(
                MujocoRuntimeCheck(
                    name="run_dir.qualification.accepted",
                    ok=accepted,
                    detail="accepted=true" if accepted else "qualification accepted flag is not true",
                )
            )
        except Exception as exc:
            checks.append(MujocoRuntimeCheck(name="run_dir.qualification.parse", ok=False, detail=str(exc)))
    return checks


def _check_scene_meshes(scene: SceneSpec, registry: AssetRegistry) -> list[MujocoRuntimeCheck]:
    checks: list[MujocoRuntimeCheck] = []
    for obj in scene.objects:
        if not obj.asset_id:
            checks.append(MujocoRuntimeCheck(name=f"asset.{obj.id}", ok=False, detail="object has no asset_id"))
            continue
        asset = registry.get(obj.asset_id)
        mesh_path = asset.resolved_mesh_path(registry.base_dir)
        if mesh_path is None:
            checks.append(
                MujocoRuntimeCheck(
                    name=f"asset.{obj.id}",
                    ok=True,
                    detail=f"asset {asset.id} has no mesh; procedural primitive fallback will be used",
                )
            )
            continue
        checks.append(
            MujocoRuntimeCheck(
                name=f"asset.{obj.id}",
                ok=mesh_path.is_file(),
                detail=str(mesh_path) if mesh_path.is_file() else f"missing mesh for asset {asset.id}: {mesh_path}",
            )
        )
    return checks
