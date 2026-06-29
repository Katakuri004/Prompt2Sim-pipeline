from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from scenethesis_mvp.mujoco_bridge.mesh_names import sanitize_name
from scenethesis_mvp.mujoco_bridge.schemas import CollisionSpec, SceneIR, SceneIRObject
from scenethesis_mvp.mujoco_bridge.visual_scene import (
    compile_visual_scene_assets,
    finalize_entity_manifest,
    visual_scene_enabled,
)
from scenethesis_mvp.utils.io import write_json


def prepare_mesh_assets(scene_ir: SceneIR, out_dir: str | Path, config: dict[str, Any]) -> tuple[SceneIR, dict[str, Any]]:
    target = Path(out_dir).resolve()
    mesh_dir = target / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    updated = scene_ir.model_copy(deep=True)
    report: dict[str, Any] = {
        "mesh_dir": str(mesh_dir),
        "objects": [],
        "coacd_requested": bool(config.get("collision", {}).get("use_coacd", False)),
    }
    entity_manifest: dict[str, Any] = {"entities": {}}
    if visual_scene_enabled(config):
        updated, visual_report, entity_manifest = compile_visual_scene_assets(updated, target, config)
        report["visual_scene"] = visual_report
    for obj in updated.objects:
        visual_mesh = _collision_source_mesh(obj, mesh_dir) if visual_scene_enabled(config) else _load_or_create_local_mesh(obj)
        if not visual_scene_enabled(config):
            visual_name = sanitize_name(f"{obj.id}_visual")
            visual_path = mesh_dir / f"{visual_name}.obj"
            visual_mesh.export(visual_path)
            obj.visual_mesh = visual_name
        object_record: dict[str, Any] = {
            "object_id": obj.id,
            "visual_mesh": obj.visual_mesh,
            "visual_parts": [part.model_dump(mode="json") for part in obj.visual_parts],
            "mobility": obj.mobility,
            "collision": [],
        }
        if obj.mobility == "visual_only":
            obj.collision = [CollisionSpec(kind="visual_only")]
            report["objects"].append(object_record)
            continue
        if _should_use_compound_primitives(obj):
            obj.collision = _compound_primitives_for(obj)
            object_record["collision"] = [item.model_dump(mode="json") for item in obj.collision]
            report["objects"].append(object_record)
            continue
        if obj.mobility == "dynamic":
            collision_specs = _dynamic_collision_meshes(obj, visual_mesh, mesh_dir, config)
            obj.collision = collision_specs
            obj.collision_meshes = [spec.mesh_path or "" for spec in collision_specs if spec.mesh_path]
            object_record["collision"] = [item.model_dump(mode="json") for item in obj.collision]
            report["objects"].append(object_record)
            continue
        obj.collision = [CollisionSpec(kind="primitive", primitive_type="box", size=_half_extents(obj.dimensions))]
        object_record["collision"] = [item.model_dump(mode="json") for item in obj.collision]
        report["objects"].append(object_record)
    if visual_scene_enabled(config):
        entity_manifest = finalize_entity_manifest(updated, entity_manifest, target)
        report["entity_manifest"] = entity_manifest
    write_json(target / "mesh_compile_report.json", report)
    return updated, report


def _load_or_create_local_mesh(obj: SceneIRObject) -> trimesh.Trimesh:
    if obj.source_visual_path and Path(obj.source_visual_path).is_file():
        loaded = trimesh.load(obj.source_visual_path, force="scene")
        if isinstance(loaded, trimesh.Scene):
            mesh = loaded.dump(concatenate=True)
        elif isinstance(loaded, trimesh.Trimesh):
            mesh = loaded
        else:
            raise RuntimeError(f"unsupported mesh load result for {obj.source_visual_path}: {type(loaded)}")
        if mesh.vertices.size == 0:
            raise RuntimeError(f"mesh has no vertices: {obj.source_visual_path}")
        return _scale_mesh_to_dimensions(mesh, obj.dimensions)
    return _primitive_visual_mesh(obj)


def _collision_source_mesh(obj: SceneIRObject, mesh_dir: Path) -> trimesh.Trimesh:
    if obj.visual_parts:
        chunks: list[trimesh.Trimesh] = []
        for part in obj.visual_parts:
            path = mesh_dir / part.file
            if path.is_file():
                loaded = trimesh.load(path, force="mesh")
                if isinstance(loaded, trimesh.Trimesh) and len(loaded.vertices) and len(loaded.faces):
                    chunks.append(loaded)
        if chunks:
            return trimesh.util.concatenate(chunks)
    return _load_or_create_local_mesh(obj)


def _scale_mesh_to_dimensions(mesh: trimesh.Trimesh, dimensions: list[float]) -> trimesh.Trimesh:
    scaled = mesh.copy()
    bounds = np.asarray(scaled.bounds, dtype=float)
    source_min = bounds[0]
    source_max = bounds[1]
    source_size = np.maximum(source_max - source_min, 1e-9)
    source_center = (source_min + source_max) * 0.5
    target_dims = np.asarray(dimensions, dtype=float)
    scaled.vertices = (np.asarray(scaled.vertices, dtype=float) - source_center) * (target_dims / source_size)
    return scaled


def _primitive_visual_mesh(obj: SceneIRObject) -> trimesh.Trimesh:
    dx, dy, dz = obj.dimensions
    if obj.category == "cylinder":
        return trimesh.creation.cylinder(radius=max(dx, dy) * 0.5, height=dz, sections=32)
    return trimesh.creation.box(extents=(dx, dy, dz))


def _should_use_compound_primitives(obj: SceneIRObject) -> bool:
    return obj.category in {
        "shelf",
        "table",
        "pallet",
        "cabinet",
        "barrier",
        "scanner",
        "cart",
        "hand_truck",
        "ladder",
        "forklift",
        "pallet_load",
        "cylinder",
        "bin",
    } or obj.mobility == "static"


def _compound_primitives_for(obj: SceneIRObject) -> list[CollisionSpec]:
    dx, dy, dz = obj.dimensions
    if obj.category == "scanner":
        support_pad = CollisionSpec(
            kind="primitive",
            primitive_type="box",
            size=[max(dx * 0.28, 0.024), max(dy * 0.24, 0.016), max(dz * 0.035, 0.006)],
            pos=[0.0, round(dy * 0.02, 6), round(-dz * 0.455, 6)],
        )
        body = CollisionSpec(
            kind="primitive",
            primitive_type="box",
            size=[max(dx * 0.36, 0.030), max(dy * 0.32, 0.024), max(dz * 0.26, 0.022)],
            pos=[0.0, 0.0, round(-dz * 0.18, 6)],
        )
        head = CollisionSpec(
            kind="primitive",
            primitive_type="box",
            size=[max(dx * 0.42, 0.034), max(dy * 0.30, 0.022), max(dz * 0.16, 0.016)],
            pos=[0.0, round(dy * 0.18, 6), round(dz * 0.16, 6)],
        )
        handle = CollisionSpec(
            kind="primitive",
            primitive_type="box",
            size=[max(dx * 0.18, 0.014), max(dy * 0.20, 0.014), max(dz * 0.30, 0.020)],
            pos=[0.0, round(-dy * 0.18, 6), round(-dz * 0.10, 6)],
        )
        trigger_guard = CollisionSpec(
            kind="primitive",
            primitive_type="box",
            size=[max(dx * 0.12, 0.008), max(dy * 0.08, 0.008), max(dz * 0.10, 0.008)],
            pos=[0.0, round(-dy * 0.06, 6), round(-dz * 0.34, 6)],
        )
        return [support_pad, body, head, handle, trigger_guard]
    if obj.category in {"cylinder", "bin"}:
        return [
            CollisionSpec(
                kind="primitive",
                primitive_type="cylinder",
                size=[round(max(dx, dy) * 0.5, 6), round(dz * 0.5, 6)],
            )
        ]
    if obj.category == "shelf":
        specs: list[CollisionSpec] = []
        shelf_thickness = max(0.025, dz * 0.025)
        for z in (-0.33 * dz, 0.0, 0.33 * dz):
            specs.append(
                CollisionSpec(
                    kind="primitive",
                    primitive_type="box",
                    size=[dx * 0.5, dy * 0.5, shelf_thickness * 0.5],
                    pos=[0.0, 0.0, round(z, 6)],
                )
            )
        post = max(0.025, min(dx, dy) * 0.035)
        for sx in (-0.45, 0.45):
            for sy in (-0.42, 0.42):
                specs.append(
                    CollisionSpec(
                        kind="primitive",
                        primitive_type="box",
                        size=[post * 0.5, post * 0.5, dz * 0.5],
                        pos=[round(sx * dx, 6), round(sy * dy, 6), 0.0],
                    )
                )
        return specs
    if obj.category == "table":
        specs = [
            CollisionSpec(
                kind="primitive",
                primitive_type="box",
                size=[dx * 0.5, dy * 0.5, max(0.025, dz * 0.06)],
                pos=[0.0, 0.0, round(dz * 0.44, 6)],
            )
        ]
        leg = max(0.025, min(dx, dy) * 0.04)
        for sx in (-0.42, 0.42):
            for sy in (-0.38, 0.38):
                specs.append(
                    CollisionSpec(
                        kind="primitive",
                        primitive_type="box",
                        size=[leg * 0.5, leg * 0.5, dz * 0.38],
                        pos=[round(sx * dx, 6), round(sy * dy, 6), round(-dz * 0.08, 6)],
                    )
                )
        return specs
    if obj.category == "pallet":
        board_h = max(0.018, dz * 0.18)
        specs = [
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.5, dy * 0.12, board_h], pos=[0.0, -dy * 0.32, dz * 0.15]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.5, dy * 0.12, board_h], pos=[0.0, 0.0, dz * 0.15]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.5, dy * 0.12, board_h], pos=[0.0, dy * 0.32, dz * 0.15]),
        ]
        return specs
    if obj.category == "cart":
        deck_h = max(0.025, dz * 0.08)
        rail = max(0.018, min(dx, dy) * 0.035)
        return [
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.45, dy * 0.36, deck_h], pos=[0.0, 0.0, round(-dz * 0.28, 6)]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[rail, rail, dz * 0.42], pos=[round(-dx * 0.42, 6), round(-dy * 0.32, 6), round(dz * 0.04, 6)]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[rail, rail, dz * 0.42], pos=[round(-dx * 0.42, 6), round(dy * 0.32, 6), round(dz * 0.04, 6)]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[rail, dy * 0.36, rail], pos=[round(-dx * 0.42, 6), 0.0, round(dz * 0.46, 6)]),
        ]
    if obj.category == "hand_truck":
        rail = max(0.018, min(dx, dy) * 0.04)
        return [
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.35, dy * 0.18, rail], pos=[0.0, round(-dy * 0.28, 6), round(-dz * 0.42, 6)]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[rail, rail, dz * 0.45], pos=[round(-dx * 0.28, 6), 0.0, 0.0]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[rail, rail, dz * 0.45], pos=[round(dx * 0.28, 6), 0.0, 0.0]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.28, rail, rail], pos=[0.0, 0.0, round(dz * 0.42, 6)]),
        ]
    if obj.category == "ladder":
        rail = max(0.015, min(dx, dy) * 0.025)
        rung_count = 5
        specs = [
            CollisionSpec(kind="primitive", primitive_type="box", size=[rail, rail, dz * 0.48], pos=[round(-dx * 0.32, 6), 0.0, 0.0]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[rail, rail, dz * 0.48], pos=[round(dx * 0.32, 6), 0.0, 0.0]),
        ]
        for index in range(rung_count):
            z = -dz * 0.36 + index * (dz * 0.72 / max(1, rung_count - 1))
            specs.append(CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.30, rail, rail], pos=[0.0, 0.0, round(z, 6)]))
        return specs
    if obj.category == "barrier":
        post = max(0.025, min(dx, dy) * 0.06)
        return [
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.48, dy * 0.10, max(0.025, dz * 0.08)], pos=[0.0, 0.0, round(-dz * 0.42, 6)]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[post, post, dz * 0.42], pos=[round(-dx * 0.42, 6), 0.0, 0.0]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[post, post, dz * 0.42], pos=[round(dx * 0.42, 6), 0.0, 0.0]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.38, post, post], pos=[0.0, 0.0, round(dz * 0.25, 6)]),
        ]
    if obj.category == "forklift":
        return [
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.32, dy * 0.34, dz * 0.18], pos=[0.0, 0.0, round(-dz * 0.25, 6)]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.08, dy * 0.45, dz * 0.36], pos=[round(dx * 0.36, 6), 0.0, round(dz * 0.04, 6)]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.32, max(0.018, dy * 0.035), max(0.018, dz * 0.025)], pos=[round(dx * 0.42, 6), round(-dy * 0.26, 6), round(-dz * 0.45, 6)]),
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.32, max(0.018, dy * 0.035), max(0.018, dz * 0.025)], pos=[round(dx * 0.42, 6), round(dy * 0.26, 6), round(-dz * 0.45, 6)]),
        ]
    if obj.category == "pallet_load":
        return [
            CollisionSpec(kind="primitive", primitive_type="box", size=[dx * 0.48, dy * 0.48, dz * 0.44], pos=[0.0, 0.0, round(dz * 0.04, 6)])
        ]
    return [CollisionSpec(kind="primitive", primitive_type="box", size=_half_extents(obj.dimensions))]


def _dynamic_collision_meshes(obj: SceneIRObject, visual_mesh: trimesh.Trimesh, mesh_dir: Path, config: dict[str, Any]) -> list[CollisionSpec]:
    use_coacd = bool(config.get("collision", {}).get("use_coacd", False))
    if use_coacd and importlib.util.find_spec("coacd") is not None:
        try:
            chunks = _run_coacd(visual_mesh, int(config.get("collision", {}).get("max_coacd_parts", 8)))
            if chunks:
                specs = []
                for index, chunk in enumerate(chunks):
                    name = sanitize_name(f"{obj.id}_collision_{index}")
                    path = mesh_dir / f"{name}.obj"
                    chunk.export(path)
                    specs.append(CollisionSpec(kind="mesh", mesh_name=name, mesh_path=str(path)))
                return specs
        except Exception:
            pass
    hull = visual_mesh.convex_hull
    if len(hull.faces) == 0:
        hull = trimesh.creation.box(extents=obj.dimensions)
    name = sanitize_name(f"{obj.id}_collision_0")
    path = mesh_dir / f"{name}.obj"
    hull.export(path)
    return [CollisionSpec(kind="mesh", mesh_name=name, mesh_path=str(path))]


def _run_coacd(mesh: trimesh.Trimesh, max_parts: int) -> list[trimesh.Trimesh]:
    import coacd

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    if len(vertices) == 0 or len(faces) == 0:
        return []
    coacd_mesh = coacd.Mesh(vertices, faces)
    result = coacd.run_coacd(coacd_mesh, max_convex_hull=max_parts)
    chunks: list[trimesh.Trimesh] = []
    for vertices_part, faces_part in result[:max_parts]:
        chunk = trimesh.Trimesh(vertices=np.asarray(vertices_part), faces=np.asarray(faces_part), process=False)
        if len(chunk.vertices) and len(chunk.faces):
            chunks.append(chunk)
    return chunks


def _half_extents(dimensions: list[float]) -> list[float]:
    return [round(float(item) * 0.5, 6) for item in dimensions]


def yaw_from_quaternion(quat: list[float]) -> float:
    w, _x, _y, z = quat
    return math.atan2(2.0 * w * z, 1.0 - 2.0 * z * z)
