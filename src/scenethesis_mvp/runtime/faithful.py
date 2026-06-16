from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.llm.openai_client import OpenAIClient
from scenethesis_mvp.render.blender_runner import resolve_blender_path
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.paths import resolve_path


class RuntimeCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    ok: bool
    detail: str


class FaithfulRuntimeReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool
    checks: list[RuntimeCheck] = Field(default_factory=list)

    @property
    def errors(self) -> list[str]:
        return [f"{check.name}: {check.detail}" for check in self.checks if not check.ok]


def validate_faithful_runtime(
    config: dict[str, Any],
    root: Path,
    registry: AssetRegistry | None = None,
    scene: SceneSpec | None = None,
    check_disk: bool = True,
) -> FaithfulRuntimeReport:
    checks: list[RuntimeCheck] = []
    faithful_cfg = config.get("paper_faithful", {})
    checks.append(
        _check_bool(
            "paper_faithful.enabled",
            bool(faithful_cfg.get("enabled", False)),
            "paper_faithful.enabled must be true for the faithful runner",
        )
    )
    checks.append(
        _check_bool(
            "paper_faithful.allow_substitutes",
            faithful_cfg.get("allow_substitutes", True) is False,
            "allow_substitutes must be false; replacement components are not allowed",
        )
    )
    checks.append(_check_python_version())
    checks.append(_check_openai())
    checks.append(_check_blender(config.get("render", {})))
    checks.append(_check_nvidia_smi())
    if check_disk:
        checks.append(_check_free_disk(root, float(faithful_cfg.get("min_free_disk_gb", 15))))

    required_imports = {
        "torch": "PyTorch with CUDA",
        "torchvision": "torchvision for Grounded-SAM box conversion",
        "groundingdino": "GroundingDINO package",
        "segment_anything": "Segment Anything package",
        "depth_pro": "Apple Depth Pro package",
        "romatch": "RoMa dense correspondence package",
        "open_clip": "OpenCLIP package",
        "pytorch3d": "PyTorch3D package",
        "cv2": "OpenCV package",
        "PIL": "Pillow package",
        "numpy": "NumPy package",
        "rtree": "Rtree spatial index for trimesh signed-distance queries",
    }
    checks.extend(_check_import(module, purpose) for module, purpose in required_imports.items())
    checks.append(_check_torch_cuda())
    checks.append(_check_torch_pytorch3d_supported())
    checks.append(_check_pytorch3d_extension())
    checks.append(_check_groundingdino_extension())
    checks.extend(_check_configured_files(config, root))
    if registry is not None and scene is not None:
        checks.extend(_check_scene_has_real_meshes(scene, registry))

    report = FaithfulRuntimeReport(ok=all(check.ok for check in checks), checks=checks)
    return report


def raise_if_unavailable(report: FaithfulRuntimeReport) -> None:
    if report.ok:
        return
    details = "\n- ".join(report.errors)
    raise RuntimeError(f"Paper-faithful runtime requirements are not satisfied:\n- {details}")


def _check_bool(name: str, ok: bool, detail: str) -> RuntimeCheck:
    return RuntimeCheck(name=name, ok=ok, detail="ok" if ok else detail)


def _check_python_version() -> RuntimeCheck:
    version = sys.version_info
    ok = (3, 10) <= (version.major, version.minor) <= (3, 12)
    detail = f"Python {version.major}.{version.minor}.{version.micro}"
    if not ok:
        detail += "; use a dedicated Python 3.10-3.12 CUDA environment for the faithful stack"
    return RuntimeCheck(name="python.version", ok=ok, detail=detail)


def _check_openai() -> RuntimeCheck:
    ok = OpenAIClient().configured
    return RuntimeCheck(
        name="openai",
        ok=ok,
        detail="OpenAI client configured" if ok else "OPENAI_API_KEY is missing or OpenAI package/client failed to initialize",
    )


def _check_blender(render_cfg: dict[str, Any]) -> RuntimeCheck:
    blender = resolve_blender_path(render_cfg.get("blender_path"))
    return RuntimeCheck(
        name="blender",
        ok=bool(blender),
        detail=str(blender) if blender else "Blender executable was not found",
    )


def _check_nvidia_smi() -> RuntimeCheck:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return RuntimeCheck(name="nvidia-smi", ok=False, detail="nvidia-smi was not found on PATH")
    try:
        result = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, check=True)
        return RuntimeCheck(name="nvidia-smi", ok=True, detail=result.stdout.strip())
    except Exception as exc:
        return RuntimeCheck(name="nvidia-smi", ok=False, detail=str(exc))


def _check_free_disk(root: Path, minimum_gb: float) -> RuntimeCheck:
    usage = shutil.disk_usage(root)
    free_gb = usage.free / (1024**3)
    return RuntimeCheck(
        name="disk.free",
        ok=free_gb >= minimum_gb,
        detail=f"{free_gb:.1f} GB free; requires at least {minimum_gb:.1f} GB",
    )


def _check_import(module: str, purpose: str) -> RuntimeCheck:
    ok = importlib.util.find_spec(module) is not None
    return RuntimeCheck(
        name=f"import.{module}",
        ok=ok,
        detail=f"{purpose} importable" if ok else f"{purpose} is not installed",
    )


def _check_torch_cuda() -> RuntimeCheck:
    if importlib.util.find_spec("torch") is None:
        return RuntimeCheck(name="torch.cuda", ok=False, detail="torch is not installed")
    try:
        import torch

        ok = bool(torch.cuda.is_available())
        detail = torch.cuda.get_device_name(0) if ok else "torch.cuda.is_available() returned false"
        return RuntimeCheck(name="torch.cuda", ok=ok, detail=detail)
    except Exception as exc:
        return RuntimeCheck(name="torch.cuda", ok=False, detail=str(exc))


def _check_torch_pytorch3d_supported() -> RuntimeCheck:
    if importlib.util.find_spec("torch") is None:
        return RuntimeCheck(name="torch.pytorch3d_supported", ok=False, detail="torch is not installed")
    try:
        import torch

        version = _parse_version_triplet(torch.__version__)
        ok = (2, 1, 0) <= version <= (2, 5, 1)
        detail = f"torch {torch.__version__}; this project has been validated with source-built PyTorch3D through torch 2.5.1"
        return RuntimeCheck(name="torch.pytorch3d_supported", ok=ok, detail=detail)
    except Exception as exc:
        return RuntimeCheck(name="torch.pytorch3d_supported", ok=False, detail=str(exc))


def _parse_version_triplet(raw: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", raw)
    if not match:
        return (0, 0, 0)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3)))


def _check_pytorch3d_extension() -> RuntimeCheck:
    if importlib.util.find_spec("pytorch3d") is None:
        return RuntimeCheck(name="pytorch3d.extension", ok=False, detail="pytorch3d is not installed")
    try:
        import torch  # noqa: F401
        import pytorch3d._C  # noqa: F401

        return RuntimeCheck(name="pytorch3d.extension", ok=True, detail="native PyTorch3D extension imports after torch")
    except Exception as exc:
        return RuntimeCheck(name="pytorch3d.extension", ok=False, detail=str(exc))


def _check_groundingdino_extension() -> RuntimeCheck:
    if importlib.util.find_spec("groundingdino") is None:
        return RuntimeCheck(name="groundingdino.extension", ok=False, detail="groundingdino is not installed")
    try:
        import torch  # noqa: F401
        import groundingdino._C  # noqa: F401

        return RuntimeCheck(name="groundingdino.extension", ok=True, detail="native GroundingDINO extension imports after torch")
    except Exception as exc:
        return RuntimeCheck(name="groundingdino.extension", ok=False, detail=str(exc))


def _check_configured_files(config: dict[str, Any], root: Path) -> list[RuntimeCheck]:
    file_specs = [
        ("segmentation.grounding_dino_config", config.get("segmentation", {}).get("grounding_dino_config")),
        ("segmentation.grounding_dino_checkpoint", config.get("segmentation", {}).get("grounding_dino_checkpoint")),
        ("segmentation.sam_checkpoint", config.get("segmentation", {}).get("sam_checkpoint")),
        ("depth.repo_dir", config.get("depth", {}).get("repo_dir")),
        ("depth.checkpoint_dir", config.get("depth", {}).get("checkpoint_dir")),
        ("asset_retrieval.index_path", config.get("asset_retrieval", {}).get("index_path")),
    ]
    correspondence_cfg = config.get("correspondence", {})
    if correspondence_cfg.get("enabled", False):
        file_specs.extend(
            [
                ("correspondence.weights_path", correspondence_cfg.get("weights_path")),
                ("correspondence.dinov2_weights_path", correspondence_cfg.get("dinov2_weights_path")),
            ]
        )
    checks: list[RuntimeCheck] = []
    for name, raw_path in file_specs:
        if not raw_path:
            checks.append(RuntimeCheck(name=name, ok=False, detail="path is not configured"))
            continue
        path = resolve_path(raw_path, root)
        if name == "depth.checkpoint_dir":
            ok = path.is_dir() and any(item.is_file() for item in path.rglob("*"))
            checks.append(
                RuntimeCheck(
                    name=name,
                    ok=ok,
                    detail=str(path) if ok else f"missing or empty required checkpoint directory: {path}",
                )
            )
            continue
        checks.append(
            RuntimeCheck(
                name=name,
                ok=path.exists(),
                detail=str(path) if path.exists() else f"missing required path: {path}",
            )
        )
    return checks


def _check_scene_has_real_meshes(scene: SceneSpec, registry: AssetRegistry) -> list[RuntimeCheck]:
    checks: list[RuntimeCheck] = []
    for obj in scene.objects:
        if not obj.asset_id:
            checks.append(RuntimeCheck(name=f"asset.{obj.id}", ok=False, detail="object has no asset_id"))
            continue
        asset = registry.get(obj.asset_id)
        mesh_path = asset.resolved_mesh_path(registry.base_dir)
        ok = bool(mesh_path and mesh_path.is_file())
        checks.append(
            RuntimeCheck(
                name=f"asset.{obj.id}",
                ok=ok,
                detail=str(mesh_path) if ok else f"asset {asset.id} has no local mesh; procedural substitute is not allowed",
            )
        )
    return checks
