from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import read_json, write_json


@dataclass(frozen=True)
class RenderResult:
    render_path: Path
    glb_path: Path
    usd_path: Path | None
    used_blender: bool
    renderer: str
    notes: str


def render_scene(
    scene: SceneSpec,
    registry: AssetRegistry,
    out_dir: str | Path,
    resolution: tuple[int, int] = (1200, 900),
    blender_path: str | None = None,
    environment: dict | None = None,
) -> RenderResult:
    target_dir = Path(out_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    render_input = target_dir / "render_input.json"
    assets_payload = {}
    for asset in registry.assets:
        payload = asset.model_dump(mode="json")
        mesh_path = asset.resolved_mesh_path(registry.base_dir)
        if mesh_path is not None:
            if mesh_path.suffix.lower() not in {".glb", ".gltf"}:
                raise RuntimeError(f"Unsupported mesh format for asset {asset.id}: {mesh_path}")
            if not mesh_path.is_file():
                raise RuntimeError(f"Asset {asset.id} references missing local mesh: {mesh_path}")
            payload["resolved_glb_path"] = str(mesh_path)
        assets_payload[asset.id] = payload
    render_payload = {
        "scene": scene.model_dump(mode="json"),
        "assets": assets_payload,
        "resolution": list(resolution),
        "environment": environment or {},
    }
    write_json(render_input, render_payload)

    blender = resolve_blender_path(blender_path)
    if blender:
        script = Path(__file__).resolve().with_name("blender_script.py")
        command = [
            blender,
            "--background",
            "--python",
            str(script),
            "--",
            "--input",
            str(render_input),
            "--out",
            str(target_dir),
        ]
        subprocess.run(command, check=True)
        validation_path = target_dir / "render_validation.json"
        if not validation_path.is_file():
            raise RuntimeError(f"Blender did not produce render_validation.json: {validation_path}")
        validation = read_json(validation_path)
        if not validation.get("ok", False):
            failures = validation.get("failures", [])
            details = "; ".join(
                f"{item.get('object_id') or item.get('object_a')} error={item.get('error_m', item.get('reason', 'n/a'))}"
                for item in failures[:8]
            )
            raise RuntimeError(f"Blender visual validation failed: {details}")
        if not (target_dir / "render.png").is_file() or not (target_dir / "scene.glb").is_file():
            raise RuntimeError("Blender finished without required render.png and scene.glb outputs.")
        views_path = target_dir / "render_views.json"
        if not views_path.is_file():
            raise RuntimeError(f"Blender did not produce render_views.json: {views_path}")
        views = read_json(views_path)
        if not views.get("scene_views") or not views.get("object_alignment_views"):
            raise RuntimeError("Blender render_views.json is missing scene views or object alignment views.")
        return RenderResult(
            render_path=target_dir / "render.png",
            glb_path=target_dir / "scene.glb",
            usd_path=(target_dir / "scene.usd") if (target_dir / "scene.usd").exists() else None,
            used_blender=True,
            renderer="blender",
            notes="Rendered with headless Blender subprocess.",
        )

    raise RuntimeError("Blender executable not found. Install Blender or set BLENDER_PATH.")


def resolve_blender_path(blender_path: str | None = None) -> str | None:
    requested = blender_path or os.getenv("BLENDER_PATH") or "blender"
    candidate = Path(requested)
    if candidate.is_file():
        return str(candidate)
    from_path = shutil.which(requested)
    if from_path:
        return from_path
    program_files = os.getenv("ProgramFiles", r"C:\Program Files")
    common_roots = [
        Path(program_files) / "Blender Foundation",
        Path(os.getenv("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Blender Foundation",
    ]
    for root in common_roots:
        if not root.exists():
            continue
        candidates = sorted(root.glob("Blender*\\blender.exe"), reverse=True)
        if candidates:
            return str(candidates[0])
    return None
