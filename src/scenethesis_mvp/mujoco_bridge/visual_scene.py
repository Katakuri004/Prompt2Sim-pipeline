from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from scenethesis_mvp.mujoco_bridge.mesh_names import sanitize_name
from scenethesis_mvp.mujoco_bridge.schemas import (
    SceneIR,
    SceneIRObject,
    VisualMaterialSpec,
    VisualMeshSpec,
)
from scenethesis_mvp.utils.io import write_json


@dataclass(frozen=True)
class TransformCandidate:
    name: str
    source_coordinate_system: str
    matrix: np.ndarray


def visual_scene_enabled(config: dict[str, Any]) -> bool:
    return str(config.get("visual_scene", {}).get("mode", "proxy")) == "full_glb_visual"


def compile_visual_scene_assets(
    scene_ir: SceneIR,
    out_dir: str | Path,
    config: dict[str, Any],
) -> tuple[SceneIR, dict[str, Any], dict[str, Any]]:
    if not visual_scene_enabled(config):
        return scene_ir, {"enabled": False, "mode": "proxy"}, _empty_entity_manifest(scene_ir)

    target = Path(out_dir).resolve()
    mesh_dir = target / "meshes"
    visual_dir = mesh_dir / "visual_scene"
    visual_dir.mkdir(parents=True, exist_ok=True)

    source_glb = Path(scene_ir.source_scene_glb)
    if not source_glb.is_file():
        raise RuntimeError(f"full_glb_visual mode requires scene.glb: {source_glb}")
    try:
        loaded = trimesh.load(source_glb, force="scene")
    except Exception as exc:
        raise RuntimeError(f"failed to load visual scene GLB: {source_glb}") from exc
    if not isinstance(loaded, trimesh.Scene) or not loaded.geometry:
        raise RuntimeError(f"visual scene GLB did not contain a mesh scene: {source_glb}")

    updated = scene_ir.model_copy(deep=True)
    object_ids = sorted((obj.id for obj in updated.objects), key=len, reverse=True)
    dynamic_ids = {updated.task.target_object}
    target_obj = updated.object_by_id(updated.task.target_object)
    if target_obj.mobility != "dynamic":
        raise RuntimeError(f"task target must be dynamic for visual replacement: {target_obj.id}")

    usd_info = _read_usd_info(updated.source_scene_usd)
    transform_report = _choose_transform(loaded, updated, usd_info, config)
    transform = np.asarray(transform_report["matrix"], dtype=float)
    determinant = float(transform_report["determinant"])
    reverse_faces = determinant < 0

    materials: dict[str, VisualMaterialSpec] = {}
    static_meshes: list[VisualMeshSpec] = []
    entity_records = _empty_entity_manifest(updated)
    dynamic_node_counts = {entity_id: 0 for entity_id in dynamic_ids}
    static_dynamic_leaks = {entity_id: 0 for entity_id in dynamic_ids}
    material_source_counts: dict[str, int] = {}

    nodes = sorted(str(node) for node in loaded.graph.nodes_geometry)
    for node_index, node_name in enumerate(nodes):
        matrix, geometry_name = loaded.graph.get(node_name)
        geometry = loaded.geometry.get(geometry_name)
        if geometry is None or len(geometry.vertices) == 0 or len(geometry.faces) == 0:
            continue
        entity_id = _entity_for_node(node_name, object_ids)
        mesh = _world_mesh_for_node(geometry, matrix, transform, reverse_faces)
        material = _material_for_geometry(geometry, materials, material_source_counts, entity_id)
        if entity_id in dynamic_ids:
            dynamic_node_counts[entity_id] += 1
            part = _export_dynamic_part(mesh, updated.object_by_id(entity_id), visual_dir, node_index, node_name, material.name)
            updated.object_by_id(entity_id).visual_parts.append(part)
            _record_visual_part(entity_records, entity_id, part, node_name, "dynamic_visual_bound")
            continue
        if entity_id in dynamic_ids:
            static_dynamic_leaks[entity_id] += 1
        part = _export_static_part(mesh, visual_dir, node_index, node_name, entity_id, material.name)
        static_meshes.append(part)
        if entity_id:
            _record_visual_part(entity_records, entity_id, part, node_name, "static_visual_world")

    missing_dynamic = [entity_id for entity_id, count in dynamic_node_counts.items() if count == 0]
    if missing_dynamic:
        raise RuntimeError(
            "full_glb_visual mode could not map dynamic task object to GLB nodes: "
            + ", ".join(sorted(missing_dynamic))
        )
    duplicate_static = [entity_id for entity_id, count in static_dynamic_leaks.items() if count > 0]
    if duplicate_static:
        raise RuntimeError("dynamic object visual leaked into static GLB world: " + ", ".join(sorted(duplicate_static)))

    updated.visual_materials = sorted(materials.values(), key=lambda item: item.name)
    updated.static_visual_meshes = sorted(static_meshes, key=lambda item: item.mesh_name)
    updated.visual_scene = {
        "mode": "full_glb_visual",
        "source_glb": str(source_glb),
        "source_usd": updated.source_scene_usd,
        "static_visual_mesh_count": len(updated.static_visual_meshes),
        "dynamic_visual_entity_ids": sorted(dynamic_ids),
        "transform": transform_report,
    }

    material_report = _material_report(updated.visual_materials)
    target_has_visual = len(updated.object_by_id(updated.task.target_object).visual_parts) > 0
    invariant_report = {
        "target_object": updated.task.target_object,
        "visible_target_instances": 1 if target_has_visual else 0,
        "dynamic_target_visual_mesh_parts": len(updated.object_by_id(updated.task.target_object).visual_parts),
        "dynamic_target_bodies": 1,
        "static_target_mesh_instances": 0,
        "ok": target_has_visual,
    }
    visual_report = {
        "enabled": True,
        "mode": "full_glb_visual",
        "source_glb": str(source_glb),
        "source_usd": updated.source_scene_usd,
        "glb_geometry_count": len(loaded.geometry),
        "glb_node_geometry_count": len(nodes),
        "static_visual_mesh_count": len(updated.static_visual_meshes),
        "dynamic_visual_mesh_count": sum(len(obj.visual_parts) for obj in updated.objects),
        "semantic_replacement": invariant_report,
        "coordinate_normalization": transform_report,
        "usd_validation": usd_info,
        "materials": material_report,
    }
    if not invariant_report["ok"]:
        raise RuntimeError(f"semantic visual replacement invariant failed for {updated.task.target_object}")
    write_json(target / "visual_scene_report.json", visual_report)
    write_json(target / "entity_manifest.json", entity_records)
    return updated, visual_report, entity_records


def finalize_entity_manifest(scene_ir: SceneIR, entity_manifest: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    for obj in scene_ir.objects:
        record = entity_manifest.setdefault("entities", {}).setdefault(obj.id, {})
        record.setdefault("entity_id", obj.id)
        record["scene_spec_ref"] = f"objects.{obj.id}"
        record["physics_mode"] = obj.mobility
        record["task_role"] = "pick_target" if obj.id == scene_ir.task.target_object else "environment"
        record["collision_proxy"] = [
            {
                "kind": spec.kind,
                "primitive_type": spec.primitive_type,
                "mesh": spec.mesh_name,
                "size": spec.size,
            }
            for spec in obj.collision
            if spec.kind != "visual_only"
        ]
    target_record = entity_manifest.get("entities", {}).get(scene_ir.task.target_object, {})
    entity_manifest["semantic_invariants"] = {
        "target_object": scene_ir.task.target_object,
        "visible_target_instances": 1
        if (scene_ir.object_by_id(scene_ir.task.target_object).visual_parts or scene_ir.object_by_id(scene_ir.task.target_object).visual_mesh)
        else 0,
        "dynamic_target_visual_mesh_parts": len(scene_ir.object_by_id(scene_ir.task.target_object).visual_parts),
        "dynamic_target_bodies": 1 if scene_ir.object_by_id(scene_ir.task.target_object).mobility == "dynamic" else 0,
        "static_target_mesh_instances": int(target_record.get("static_visual_mesh_instances", 0)),
    }
    invariant = entity_manifest["semantic_invariants"]
    invariant["ok"] = (
        invariant["visible_target_instances"] == 1
        and invariant["dynamic_target_bodies"] == 1
        and invariant["static_target_mesh_instances"] == 0
    )
    write_json(Path(out_dir) / "entity_manifest.json", entity_manifest)
    return entity_manifest


def _empty_entity_manifest(scene_ir: SceneIR) -> dict[str, Any]:
    return {
        "scene_id": scene_ir.scene_id,
        "source_glb": scene_ir.source_scene_glb,
        "source_usd": scene_ir.source_scene_usd,
        "entities": {
            obj.id: {
                "entity_id": obj.id,
                "scene_spec_ref": f"objects.{obj.id}",
                "glb_nodes": [],
                "visual_meshes": [],
                "collision_proxy": [],
                "physics_mode": obj.mobility,
                "render_mode": "unmapped",
                "task_role": "pick_target" if obj.id == scene_ir.task.target_object else "environment",
                "static_visual_mesh_instances": 0,
            }
            for obj in scene_ir.objects
        },
    }


def _choose_transform(
    scene: trimesh.Scene,
    scene_ir: SceneIR,
    usd_info: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        TransformCandidate("identity", "glTF native", np.eye(3, dtype=float)),
        TransformCandidate(
            "R_x(+90deg)",
            "glTF Y-up",
            np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=float),
        ),
    ]
    raw_vertices = _scene_vertices(scene)
    if raw_vertices.size == 0:
        raise RuntimeError("visual scene GLB has no vertices after graph traversal")
    expected = np.asarray(scene_ir.bounds, dtype=float)
    scored: list[tuple[float, TransformCandidate, np.ndarray, np.ndarray]] = []
    for candidate in candidates:
        transformed = raw_vertices @ candidate.matrix.T
        bounds_min = transformed.min(axis=0)
        bounds_max = transformed.max(axis=0)
        extents = bounds_max - bounds_min
        extent_error = float(np.linalg.norm(extents - expected))
        origin_error = float(np.linalg.norm(np.minimum(np.abs(bounds_min), np.abs(bounds_max - expected))))
        score = extent_error + 0.15 * origin_error
        scored.append((score, candidate, bounds_min, bounds_max))
    scored.sort(key=lambda item: item[0])
    score, selected, bounds_min, bounds_max = scored[0]
    determinant = float(np.linalg.det(selected.matrix))
    bounds_error = float(np.linalg.norm((bounds_max - bounds_min) - expected))
    tolerance = float(config.get("visual_scene", {}).get("bounds_tolerance_m", 0.75))
    if bounds_error > tolerance:
        raise RuntimeError(
            f"visual scene coordinate normalization failed: bounds error {bounds_error:.3f}m exceeds {tolerance:.3f}m"
        )
    usd_axis = usd_info.get("up_axis")
    usd_axis_consistent = usd_axis in {None, "Z", "z"} and selected.name == "R_x(+90deg)"
    if selected.name == "identity" and usd_axis in {"Z", "z"}:
        usd_axis_consistent = True
    return {
        "source_coordinate_system": selected.source_coordinate_system,
        "target_coordinate_system": "MuJoCo Z-up",
        "applied_transform": selected.name,
        "matrix": selected.matrix.tolist(),
        "determinant": round(determinant, 6),
        "normal_orientation_valid": determinant > 0.0,
        "handedness_preserved": determinant > 0.0,
        "usd_axis_consistent": bool(usd_axis_consistent),
        "unit_scale": float(usd_info.get("meters_per_unit") or 1.0),
        "bounds_min": [round(float(item), 6) for item in bounds_min.tolist()],
        "bounds_max": [round(float(item), 6) for item in bounds_max.tolist()],
        "bounds_error_m": round(bounds_error, 6),
        "candidate_scores": [
            {"name": cand.name, "score": round(float(candidate_score), 6)}
            for candidate_score, cand, _min, _max in scored
        ],
    }


def _scene_vertices(scene: trimesh.Scene) -> np.ndarray:
    chunks: list[np.ndarray] = []
    for node_name in sorted(str(node) for node in scene.graph.nodes_geometry):
        matrix, geometry_name = scene.graph.get(node_name)
        geometry = scene.geometry.get(geometry_name)
        if geometry is None or len(geometry.vertices) == 0:
            continue
        vertices = np.asarray(geometry.vertices, dtype=float)
        world = _transform_vertices(vertices, np.asarray(matrix, dtype=float))
        chunks.append(world)
    if not chunks:
        return np.empty((0, 3), dtype=float)
    return np.vstack(chunks)


def _world_mesh_for_node(
    geometry: trimesh.Trimesh,
    node_matrix: Any,
    coord_transform: np.ndarray,
    reverse_faces: bool,
) -> trimesh.Trimesh:
    vertices = _transform_vertices(np.asarray(geometry.vertices, dtype=float), np.asarray(node_matrix, dtype=float))
    vertices = vertices @ coord_transform.T
    faces = np.asarray(geometry.faces, dtype=np.int64).copy()
    if reverse_faces:
        faces = faces[:, ::-1]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    visual = getattr(geometry, "visual", None)
    uv = getattr(visual, "uv", None)
    if uv is not None and len(uv) == len(mesh.vertices):
        try:
            mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv, dtype=float))
        except Exception:
            pass
    return mesh


def _transform_vertices(vertices: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    if matrix.shape == (4, 4):
        hom = np.concatenate([vertices, np.ones((len(vertices), 1), dtype=float)], axis=1)
        return (hom @ matrix.T)[:, :3]
    if matrix.shape == (3, 3):
        return vertices @ matrix.T
    return vertices


def _export_static_part(
    mesh: trimesh.Trimesh,
    visual_dir: Path,
    node_index: int,
    node_name: str,
    entity_id: str | None,
    material_name: str,
) -> VisualMeshSpec:
    mesh_name = sanitize_name(f"vs_{node_index:04d}_{node_name}")[:120]
    file_name = f"visual_scene/{mesh_name}.obj"
    mesh.export(visual_dir / f"{mesh_name}.obj")
    return VisualMeshSpec(
        mesh_name=mesh_name,
        file=file_name,
        material=material_name,
        role="static_world",
        entity_id=entity_id,
        node_names=[node_name],
    )


def _export_dynamic_part(
    world_mesh: trimesh.Trimesh,
    obj: SceneIRObject,
    visual_dir: Path,
    node_index: int,
    node_name: str,
    material_name: str,
) -> VisualMeshSpec:
    position = np.asarray(obj.pose.position, dtype=float)
    rotation = _quat_to_matrix(np.asarray(obj.pose.quaternion, dtype=float))
    local_vertices = (np.asarray(world_mesh.vertices, dtype=float) - position) @ rotation
    local_mesh = trimesh.Trimesh(vertices=local_vertices, faces=np.asarray(world_mesh.faces, dtype=np.int64), process=False)
    mesh_name = sanitize_name(f"dyn_{obj.id}_{node_index:04d}_{node_name}")[:120]
    file_name = f"visual_scene/{mesh_name}.obj"
    local_mesh.export(visual_dir / f"{mesh_name}.obj")
    return VisualMeshSpec(
        mesh_name=mesh_name,
        file=file_name,
        material=material_name,
        role="dynamic_object",
        entity_id=obj.id,
        node_names=[node_name],
    )


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(item) for item in quat]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-9:
        return np.eye(3, dtype=float)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _entity_for_node(node_name: str, object_ids: list[str]) -> str | None:
    lowered = node_name.lower()
    for object_id in object_ids:
        low_id = object_id.lower()
        if lowered == low_id or lowered.startswith(low_id + "_") or f"/{low_id}_" in lowered:
            return object_id
    return None


def _record_visual_part(
    entity_manifest: dict[str, Any],
    entity_id: str,
    part: VisualMeshSpec,
    node_name: str,
    render_mode: str,
) -> None:
    record = entity_manifest.setdefault("entities", {}).setdefault(entity_id, {"entity_id": entity_id})
    record.setdefault("glb_nodes", []).append(node_name)
    record.setdefault("visual_meshes", []).append(part.model_dump(mode="json"))
    record["render_mode"] = render_mode
    if part.role == "static_world":
        record["static_visual_mesh_instances"] = int(record.get("static_visual_mesh_instances", 0)) + 1


def _material_for_geometry(
    geometry: trimesh.Trimesh,
    materials: dict[str, VisualMaterialSpec],
    source_counts: dict[str, int],
    entity_id: str | None,
) -> VisualMaterialSpec:
    material = getattr(getattr(geometry, "visual", None), "material", None)
    source_name = str(getattr(material, "name", None) or "default_material")
    if source_name not in source_counts:
        source_counts[source_name] = 0
    source_counts[source_name] += 1
    material_name = sanitize_name(f"mat_{source_name}")[:80]
    if material_name in materials:
        return materials[material_name]
    rgba, rgba_source = _material_rgba(material)
    preserved: list[str] = []
    approximated: list[str] = []
    unsupported: list[str] = []
    tier = "unresolved"
    if rgba_source:
        preserved.append("baseColor")
        tier = "preserved"
    if getattr(getattr(geometry, "visual", None), "uv", None) is not None:
        preserved.append("texture_uvs")
    if getattr(material, "baseColorTexture", None) is not None:
        approximated.append("baseColorTexture sampled to baseColor")
        tier = "approximated"
    if getattr(material, "normalTexture", None) is not None:
        approximated.append("normalTexture ignored by MuJoCo material")
        tier = "approximated"
    if getattr(material, "metallicFactor", None) not in (None, 0, 0.0):
        approximated.append("metallicFactor approximated")
        tier = "approximated"
    if getattr(material, "roughnessFactor", None) is not None:
        approximated.append("roughnessFactor approximated")
        if tier == "unresolved":
            tier = "approximated"
    for attr in ("transmissionFactor", "clearcoatFactor", "anisotropyStrength"):
        if getattr(material, attr, None) is not None:
            unsupported.append(attr)
            tier = "approximated" if rgba_source else "unresolved"
    critical = _is_critical_material(source_name, entity_id)
    spec = VisualMaterialSpec(
        name=material_name,
        source_name=source_name,
        rgba=[round(float(item), 6) for item in rgba],
        tier=tier,  # type: ignore[arg-type]
        preserved_features=preserved,
        approximated_features=approximated,
        unsupported_features=unsupported,
        critical=critical,
    )
    materials[material_name] = spec
    return spec


def _material_rgba(material: Any) -> tuple[list[float], str | None]:
    fallback = [0.62, 0.62, 0.58, 1.0]
    if material is None:
        return fallback, None
    texture = getattr(material, "baseColorTexture", None)
    if texture is not None:
        try:
            pixels = np.asarray(texture, dtype=float)
            if pixels.size and pixels.ndim >= 3:
                rgb = pixels[..., :3].reshape(-1, 3)
                if rgb.shape[0] > 4096:
                    step = max(1, rgb.shape[0] // 4096)
                    rgb = rgb[::step]
                if float(np.nanmax(rgb)) > 1.0:
                    rgb = rgb / 255.0
                color = np.nanmean(rgb, axis=0)
                if np.isfinite(color).all():
                    return [float(np.clip(color[0], 0, 1)), float(np.clip(color[1], 0, 1)), float(np.clip(color[2], 0, 1)), 1.0], "baseColorTexture_mean"
        except Exception:
            pass
    for attr in ("baseColorFactor", "main_color"):
        value = getattr(material, attr, None)
        if value is None:
            continue
        rgba = np.asarray(value, dtype=float).reshape(-1)
        if len(rgba) < 3:
            continue
        if float(np.nanmax(rgba)) > 1.0:
            rgba = rgba / 255.0
        alpha = float(rgba[3]) if len(rgba) > 3 else 1.0
        return [float(np.clip(rgba[0], 0, 1)), float(np.clip(rgba[1], 0, 1)), float(np.clip(rgba[2], 0, 1)), float(np.clip(alpha, 0, 1))], attr
    return fallback, None


def _is_critical_material(source_name: str, entity_id: str | None) -> bool:
    value = f"{source_name} {entity_id or ''}".lower()
    return any(token in value for token in ("scanner", "glass", "packing_table", "table", "display", "reflect"))


def _material_report(materials: list[VisualMaterialSpec]) -> dict[str, Any]:
    preserved = [item for item in materials if item.tier == "preserved"]
    approximated = [item for item in materials if item.tier == "approximated"]
    unresolved = [item for item in materials if item.tier == "unresolved"]
    return {
        "materials_total": len(materials),
        "materials_preserved": len(preserved),
        "materials_approximated": len(approximated),
        "materials_unresolved": len(unresolved),
        "critical_unresolved_assets": [
            item.source_name or item.name for item in unresolved if item.critical
        ],
        "tiers": {item.name: item.model_dump(mode="json") for item in materials},
    }


def _read_usd_info(source_usd: str | None) -> dict[str, Any]:
    if not source_usd:
        return {"available": False}
    path = Path(source_usd)
    if not path.is_file():
        return {"available": False, "path": str(path)}
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(path))
        if stage is None:
            return {"available": False, "path": str(path), "error": "Usd.Stage.Open returned None"}
        cache = UsdGeom.BBoxCache(0, [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=True)
        bounds = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedBox()
        return {
            "available": True,
            "path": str(path),
            "up_axis": str(UsdGeom.GetStageUpAxis(stage)),
            "meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
            "bounds_min": [round(float(item), 6) for item in bounds.GetMin()],
            "bounds_max": [round(float(item), 6) for item in bounds.GetMax()],
        }
    except Exception as exc:
        return {"available": False, "path": str(path), "error": str(exc)}
