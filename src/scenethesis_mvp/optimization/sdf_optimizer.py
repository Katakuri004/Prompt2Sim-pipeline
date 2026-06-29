from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.schemas.scene_graph_3d import SceneGraph3D
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import write_json


@dataclass(frozen=True)
class SDFOptimizerConfig:
    device: str = "cuda"
    surface_samples: int = 400
    optimizer: str = "sgd"
    max_iters: int = 120
    learning_rate: float = 0.03
    stall_patience: int = 12
    min_iters_before_stall: int = 12
    collision_tolerance_m: float = 0.01
    support_tolerance_m: float = 0.08
    scale_step_limit: float = 0.015
    translation_step_limit_m: float = 0.12
    min_scale: float = 0.20
    seed: int = 7


@dataclass
class MeshTemplate:
    object_id: str
    asset_id: str
    vertices: np.ndarray
    faces: np.ndarray
    surface_points: np.ndarray
    bottom_points: np.ndarray
    support_planes: list[float]


@dataclass
class PlacedMesh:
    object_id: str
    asset_id: str
    mesh: Any
    query: Any
    centroid: np.ndarray
    bounds: np.ndarray
    support_planes: list[float]


class SDFPhysicsOptimizer:
    def __init__(self, config: SDFOptimizerConfig):
        self.config = config

    def optimize(
        self,
        scene: SceneSpec,
        graph: SceneGraph3D,
        registry: AssetRegistry,
        out_dir: str | Path,
    ) -> SceneSpec:
        try:
            import torch
            import pytorch3d  # noqa: F401
            import pytorch3d._C  # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"SDF/PyTorch3D dependencies are not installed correctly: {exc}") from exc
        try:
            import rtree  # noqa: F401
            import trimesh
            from trimesh.proximity import ProximityQuery
        except Exception as exc:
            raise RuntimeError(f"SDF mesh dependencies are not installed correctly: {exc}") from exc
        if self.config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("SDF optimizer requested CUDA, but torch.cuda.is_available() is false.")
        if self.config.optimizer.lower() != "sgd":
            raise RuntimeError("Faithful Scenethesis SDF optimization uses SGD; other optimizers are not allowed here.")
        if self.config.surface_samples != 400:
            raise RuntimeError("Faithful Scenethesis SDF optimization requires 400 surface samples per object.")

        output_dir = Path(out_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "sdf_optimizer.json",
            {
                "status": "running",
                "method": "mesh surface samples + signed-distance queries",
                "surface_samples_per_object": self.config.surface_samples,
                "objects": [],
            },
        )
        optimized = scene.model_copy(deep=True)
        rng = np.random.default_rng(self.config.seed)
        templates = self._load_templates(optimized, registry, trimesh, rng)
        self._spread_support_siblings(optimized, registry, templates)
        order = self._hierarchy_order(optimized)
        fixed: list[PlacedMesh] = []
        fixed_by_id: dict[str, PlacedMesh] = {}
        events: list[dict[str, Any]] = []

        for obj in order:
            template = templates[obj.id]
            asset = registry.get(obj.asset_id or "")
            event: dict[str, Any] = {"object_id": obj.id, "asset_id": asset.id, "iterations": []}
            self._apply_support_constraint(obj, template, fixed_by_id, registry, event, optimized.bounds)
            self._enforce_scene_bounds(obj, template, optimized.bounds)

            unresolved = 0
            best_unresolved = 10**9
            stalled_iterations = 0
            for iteration in range(self.config.max_iters):
                world_points = self._transform_points(template.surface_points, obj.placement)
                world_centroid = world_points.mean(axis=0)
                world_bounds = self._world_bounds(template, obj.placement)
                update, penetration_count, mean_penetration, scale_update = self._collision_update(
                    world_points,
                    world_centroid,
                    world_bounds,
                    fixed,
                    obj.parent_id if obj.relation in {"on", "inside"} else None,
                )
                unresolved = penetration_count
                if penetration_count < best_unresolved:
                    best_unresolved = penetration_count
                    stalled_iterations = 0
                else:
                    stalled_iterations += 1
                event["iterations"].append(
                    {
                        "iteration": iteration,
                        "penetrating_points": penetration_count,
                        "mean_penetration_m": round(mean_penetration, 6),
                    }
                )
                if penetration_count == 0:
                    break
                if iteration >= self.config.min_iters_before_stall and stalled_iterations >= self.config.stall_patience:
                    event["iterations"][-1]["stopped_reason"] = "sdf_descent_stalled"
                    break
                step_norm = float(np.linalg.norm(update))
                if step_norm > self.config.translation_step_limit_m:
                    update = update * (self.config.translation_step_limit_m / step_norm)
                obj.placement.x += float(update[0])
                obj.placement.y += float(update[1])
                obj.placement.z += float(update[2])
                if scale_update < 0:
                    obj.placement.scale = max(self.config.min_scale, obj.placement.scale + scale_update)
                self._apply_support_constraint(obj, template, fixed_by_id, registry, event, optimized.bounds)
                self._enforce_scene_bounds(obj, template, optimized.bounds)

            if unresolved > 0:
                search_result = self._free_slot_search(
                    obj=obj,
                    template=template,
                    fixed=fixed,
                    fixed_by_id=fixed_by_id,
                    registry=registry,
                    scene_bounds=optimized.bounds,
                    event=event,
                )
                unresolved = search_result["penetrating_points"]
                if unresolved > 0:
                    event["status"] = "failed"
                    events.append(event)
                    write_json(output_dir / "sdf_optimizer.json", {"status": "failed", "objects": events})
                    raise RuntimeError(
                        f"SDF optimizer could not resolve collisions for {obj.id}: "
                        f"{unresolved} sampled surface points remain inside existing scene SDFs."
                    )

            placed = self._make_placed_mesh(obj, template, trimesh, ProximityQuery)
            support_error = self._support_error(obj, template, fixed_by_id, registry)
            event["final_placement"] = obj.placement.model_dump()
            event["support_error_m"] = round(support_error, 6)
            event["status"] = "ok"
            if support_error > self.config.support_tolerance_m:
                event["status"] = "failed"
                events.append(event)
                write_json(output_dir / "sdf_optimizer.json", {"status": "failed", "objects": events})
                raise RuntimeError(
                    f"SDF stability check failed for {obj.id}: support error {support_error:.3f} m "
                    f"exceeds tolerance {self.config.support_tolerance_m:.3f} m."
                )
            fixed.append(placed)
            fixed_by_id[obj.id] = placed
            events.append(event)
            write_json(
                output_dir / "sdf_optimizer.json",
                {
                    "status": "running",
                    "method": "mesh surface samples + signed-distance queries",
                    "surface_samples_per_object": self.config.surface_samples,
                    "objects": events,
                },
            )

        write_json(
            output_dir / "sdf_optimizer.json",
            {
                "status": "ok",
                "method": "mesh surface samples + signed-distance queries",
                "surface_samples_per_object": self.config.surface_samples,
                "objects": events,
            },
        )
        return optimized

    def _load_templates(self, scene: SceneSpec, registry: AssetRegistry, trimesh: Any, rng: np.random.Generator) -> dict[str, MeshTemplate]:
        templates: dict[str, MeshTemplate] = {}
        for obj in scene.objects:
            if not obj.asset_id:
                raise RuntimeError(f"SDF optimizer requires an asset_id for object {obj.id}.")
            asset = registry.get(obj.asset_id)
            mesh_path = asset.resolved_mesh_path(registry.base_dir)
            if not mesh_path or not mesh_path.is_file():
                raise RuntimeError(f"SDF optimizer requires a local mesh for {obj.id}; asset {asset.id} has none.")
            mesh, source_components = self._load_mesh_with_components(mesh_path, trimesh)
            bounds = np.asarray(mesh.bounds, dtype=np.float64)
            source_size = bounds[1] - bounds[0]
            if np.any(source_size <= 0):
                raise RuntimeError(f"asset mesh has invalid bounds for {asset.id}: {mesh_path}")
            source_center = (bounds[0] + bounds[1]) * 0.5
            target_dims = np.asarray(asset.dimensions, dtype=np.float64)
            scale = target_dims / source_size
            vertices = (np.asarray(mesh.vertices, dtype=np.float64) - source_center) * (target_dims / source_size)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            surface = self._sample_surface(vertices, faces, self.config.surface_samples, rng)
            bottom = self._bottom_contact_points(vertices)
            normalized_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
            normalized_components = []
            for component in source_components:
                component_vertices = (np.asarray(component.vertices, dtype=np.float64) - source_center) * scale
                normalized_components.append(
                    trimesh.Trimesh(
                        vertices=component_vertices,
                        faces=np.asarray(component.faces, dtype=np.int64),
                        process=False,
                    )
                )
            templates[obj.id] = MeshTemplate(
                object_id=obj.id,
                asset_id=asset.id,
                vertices=vertices,
                faces=faces,
                surface_points=surface,
                bottom_points=bottom,
                support_planes=self._derive_template_support_planes(asset, normalized_mesh, normalized_components),
            )
        return templates

    def _load_mesh(self, mesh_path: Path, trimesh: Any) -> Any:
        mesh, _components = self._load_mesh_with_components(mesh_path, trimesh)
        return mesh

    def _load_mesh_with_components(self, mesh_path: Path, trimesh: Any) -> tuple[Any, list[Any]]:
        loaded = trimesh.load(mesh_path, force="scene", skip_materials=True)
        if hasattr(loaded, "geometry"):
            components = []
            for node_name in loaded.graph.nodes_geometry:
                transform, geometry_name = loaded.graph[node_name]
                geometry = loaded.geometry[geometry_name]
                if not hasattr(geometry, "faces") or len(geometry.faces) == 0:
                    continue
                component = geometry.copy()
                component.apply_transform(transform)
                components.append(self._to_blender_glTF_axes(component, mesh_path))
            if not components:
                raise RuntimeError(f"asset file contains no triangle mesh geometry: {mesh_path}")
            mesh = trimesh.util.concatenate(components)
            return mesh, components
        if not hasattr(loaded, "faces") or len(loaded.faces) == 0:
            raise RuntimeError(f"asset file contains no triangle mesh geometry: {mesh_path}")
        mesh = self._to_blender_glTF_axes(loaded, mesh_path)
        return mesh, [mesh]

    def _to_blender_glTF_axes(self, mesh: Any, mesh_path: Path) -> Any:
        if mesh_path.suffix.lower() not in {".glb", ".gltf"}:
            return mesh
        converted = mesh.copy()
        vertices = np.asarray(converted.vertices, dtype=np.float64)
        converted.vertices = np.column_stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]])
        return converted

    def _derive_template_support_planes(self, asset: Any, mesh: Any, components: list[Any] | None = None) -> list[float]:
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        size = bounds[1] - bounds[0]
        if np.any(size <= 0):
            return []
        if asset.support_kind == "surface":
            return [float(bounds[1, 2])]
        if asset.support_kind != "container":
            return []

        min_x_span = float(size[0] * 0.45)
        min_y_span = float(size[1] * 0.45)
        max_slab_height = max(0.08, float(size[2] * 0.18))
        raw_planes: list[float] = []
        source_components = components
        if source_components is None:
            try:
                source_components = list(mesh.split(only_watertight=False))
            except Exception:
                source_components = []
        for component in source_components:
            if not hasattr(component, "bounds") or len(component.vertices) < 8:
                continue
            component_bounds = np.asarray(component.bounds, dtype=np.float64)
            component_size = component_bounds[1] - component_bounds[0]
            if (
                component_size[0] >= min_x_span
                and component_size[1] >= min_y_span
                and component_size[2] <= max_slab_height
            ):
                raw_planes.append(float(component_bounds[1, 2]))
        if not raw_planes:
            raw_planes = self._derive_face_support_planes(mesh, bounds)
        return self._merge_support_planes(raw_planes, tolerance=max(0.025, float(size[2] * 0.025)))

    def _derive_face_support_planes(self, mesh: Any, bounds: np.ndarray) -> list[float]:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if len(vertices) == 0 or len(faces) == 0:
            return []
        size = bounds[1] - bounds[0]
        triangles = vertices[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        normal_lengths = np.linalg.norm(normals, axis=1)
        areas = normal_lengths * 0.5
        valid = normal_lengths > 1e-8
        normals[valid] = normals[valid] / normal_lengths[valid, None]
        upward = valid & (normals[:, 2] > 0.85) & (areas > 1e-5)
        if not bool(np.any(upward)):
            return []
        upward_triangles = triangles[upward]
        upward_areas = areas[upward]
        centroids = upward_triangles.mean(axis=1)
        bin_width = max(0.015, float(size[2] * 0.018))
        bins = np.round(centroids[:, 2] / bin_width) * bin_width
        min_area = max(0.01, float(size[0] * size[1] * 0.08))
        min_x_span = float(size[0] * 0.35)
        min_y_span = float(size[1] * 0.35)
        planes: list[float] = []
        for level in sorted(set(float(value) for value in bins)):
            mask = np.abs(centroids[:, 2] - level) <= bin_width
            if float(upward_areas[mask].sum()) < min_area:
                continue
            points = upward_triangles[mask].reshape(-1, 3)
            span = points.max(axis=0) - points.min(axis=0)
            if span[0] >= min_x_span and span[1] >= min_y_span:
                planes.append(float(points[:, 2].max()))
        return planes

    def _merge_support_planes(self, planes: list[float], tolerance: float) -> list[float]:
        merged: list[float] = []
        for plane in sorted(float(value) for value in planes):
            if not merged or abs(plane - merged[-1]) > tolerance:
                merged.append(plane)
            else:
                merged[-1] = float((merged[-1] + plane) * 0.5)
        return merged

    def _sample_surface(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        triangles = vertices[faces]
        cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        areas = np.linalg.norm(cross, axis=1) * 0.5
        total_area = float(areas.sum())
        if total_area <= 0:
            raise RuntimeError("cannot sample an asset mesh with zero surface area")
        face_indices = rng.choice(len(faces), size=count, replace=True, p=areas / total_area)
        chosen = triangles[face_indices]
        r1 = np.sqrt(rng.random(count))
        r2 = rng.random(count)
        return (
            (1.0 - r1)[:, None] * chosen[:, 0]
            + (r1 * (1.0 - r2))[:, None] * chosen[:, 1]
            + (r1 * r2)[:, None] * chosen[:, 2]
        )

    def _bottom_contact_points(self, vertices: np.ndarray) -> np.ndarray:
        lower = vertices.min(axis=0)
        upper = vertices.max(axis=0)
        xs = np.linspace(lower[0], upper[0], 4)
        ys = np.linspace(lower[1], upper[1], 4)
        return np.asarray([[x, y, lower[2]] for x in xs for y in ys], dtype=np.float64)

    def _hierarchy_order(self, scene: SceneSpec) -> list[Any]:
        anchors = [obj for obj in scene.objects if obj.role == "anchor"]
        parents = [obj for obj in scene.objects if obj.role == "parent"]
        children = [obj for obj in scene.objects if obj.role == "child"]
        return anchors + parents + children

    def _spread_support_siblings(
        self,
        scene: SceneSpec,
        registry: AssetRegistry,
        templates: dict[str, MeshTemplate],
    ) -> None:
        objects = {obj.id: obj for obj in scene.objects}
        groups: dict[str, list[Any]] = {}
        for obj in scene.objects:
            if obj.parent_id and obj.relation in {"on", "inside"}:
                groups.setdefault(obj.parent_id, []).append(obj)
        for parent_id, children in groups.items():
            parent = objects.get(parent_id)
            if parent is None or not parent.asset_id:
                continue
            parent_asset = registry.get(parent.asset_id)
            parent_template = templates.get(parent_id)
            mesh_planes = parent_template.support_planes if parent_template else []
            if mesh_planes:
                support_planes = [float(parent.placement.z + plane * parent.placement.scale) for plane in mesh_planes]
            elif parent_asset.support_heights:
                parent_dims = np.asarray(parent_asset.dimensions, dtype=np.float64) * float(parent.placement.scale)
                parent_bottom_z = float(parent.placement.z - parent_dims[2] * 0.5)
                support_planes = [float(parent_bottom_z + parent_dims[2] * h) for h in parent_asset.support_heights]
            else:
                continue
            parent_dims = np.asarray(parent_asset.dimensions, dtype=np.float64) * float(parent.placement.scale)
            count = len(children)
            parent_tags = {tag.lower() for tag in parent_asset.tags}
            shelf_like = parent_asset.category == "shelf" or bool(parent_tags.intersection({"shelf", "pallet_rack", "rack"}))
            if shelf_like:
                level_limit = 2 if "pallet_rack" in parent_tags else min(3, len(support_planes))
                support_planes = support_planes[: max(1, level_limit)]
            columns = max(1, int(np.ceil(np.sqrt(count))))
            rows = max(1, int(np.ceil(count / columns)))
            for index, child in enumerate(children):
                if child.id not in templates or not child.asset_id:
                    continue
                child_asset = registry.get(child.asset_id)
                child_dims = np.asarray(child_asset.dimensions, dtype=np.float64) * float(child.placement.scale)
                if shelf_like:
                    level_count = len(support_planes)
                    level_index = index % level_count
                    slot_index = index // level_count
                    slots_on_level = max(1, (count + level_count - 1 - level_index) // level_count)
                    usable_x = max(0.0, parent_dims[0] - child_dims[0] - 0.28)
                    local_x = 0.0 if slots_on_level == 1 else -usable_x * 0.5 + usable_x * (slot_index / max(1, slots_on_level - 1))
                    shelf_clearance_y = max(0.0, parent_dims[1] - child_dims[1])
                    local_y = -min(parent_dims[1] * 0.18, shelf_clearance_y * 0.45)
                else:
                    level_index = index % len(support_planes)
                    col = index % columns
                    row = index // columns
                    usable_x = max(0.05, parent_dims[0] - child_dims[0] - 0.18)
                    usable_y = max(0.05, parent_dims[1] - child_dims[1] - 0.14)
                    local_x = 0.0 if columns == 1 else -usable_x * 0.5 + usable_x * (col / max(1, columns - 1))
                    local_y = 0.0 if rows == 1 else -usable_y * 0.5 + usable_y * (row / max(1, rows - 1))
                child.placement.x, child.placement.y = self._local_to_world_xy(parent, local_x, local_y)
                target_surface_z = support_planes[level_index]
                local_min_z = float(templates[child.id].vertices[:, 2].min())
                child.placement.z = float(target_surface_z - local_min_z * child.placement.scale)
                child.placement.yaw_deg = parent.placement.yaw_deg

    def _local_to_world_xy(self, parent: Any, local_x: float, local_y: float) -> tuple[float, float]:
        theta = np.deg2rad(float(parent.placement.yaw_deg))
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        return (
            float(parent.placement.x + c * local_x - s * local_y),
            float(parent.placement.y + s * local_x + c * local_y),
        )

    def _rotation(self, yaw_deg: float) -> np.ndarray:
        theta = np.deg2rad(yaw_deg)
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    def _transform_points(self, points: np.ndarray, placement: Any) -> np.ndarray:
        scaled = points * float(placement.scale)
        rotated = scaled @ self._rotation(float(placement.yaw_deg)).T
        translation = np.asarray([placement.x, placement.y, placement.z], dtype=np.float64)
        return rotated + translation

    def _world_bounds(self, template: MeshTemplate, placement: Any) -> np.ndarray:
        points = self._transform_points(template.vertices, placement)
        return np.vstack([points.min(axis=0), points.max(axis=0)])

    def _apply_support_constraint(
        self,
        obj: Any,
        template: MeshTemplate,
        fixed_by_id: dict[str, PlacedMesh],
        registry: AssetRegistry,
        event: dict[str, Any],
        scene_bounds: list[float] | None = None,
    ) -> None:
        local_min_z = float(template.vertices[:, 2].min())
        local_max_z = float(template.vertices[:, 2].max())
        asset = registry.get(template.asset_id)
        tags = {tag.lower() for tag in asset.tags}
        if scene_bounds is not None and (not obj.parent_id or obj.relation not in {"on", "inside"}):
            width, depth, height = [float(value) for value in scene_bounds]
            if "ceiling" in tags and "floor" not in tags:
                obj.placement.z = float(height - 0.06 - local_max_z * obj.placement.scale)
                obj.placement.x = min(max(float(obj.placement.x), 0.35), width - 0.35)
                obj.placement.y = min(max(float(obj.placement.y), 0.35), depth - 0.35)
                event["support"] = "ceiling_mount"
                return
            if "wall" in tags and asset.category != "door" and "floor" not in tags:
                world = self._world_bounds(template, obj.placement)
                half_y = float((world[1, 1] - world[0, 1]) * 0.5)
                obj.placement.y = float(depth - half_y - 0.035)
                mount_fraction = 0.74 if asset.category in {"light", "camera", "duct", "pipe", "cable"} else 0.50
                min_center_z = -local_min_z * obj.placement.scale
                max_center_z = height - local_max_z * obj.placement.scale
                obj.placement.z = float(min(max(height * mount_fraction, min_center_z), max_center_z))
                obj.placement.yaw_deg = 180.0
                event["support"] = "wall_mount"
                return
            if asset.category == "door":
                world = self._world_bounds(template, obj.placement)
                half_y = float((world[1, 1] - world[0, 1]) * 0.5)
                obj.placement.y = float(depth - half_y - 0.025)
                obj.placement.z = -local_min_z * obj.placement.scale
                obj.placement.yaw_deg = 180.0
                event["support"] = "ground_and_wall"
                return
        if not obj.parent_id or obj.relation not in {"on", "inside"}:
            obj.placement.z = -local_min_z * obj.placement.scale
            event["support"] = "ground" if not obj.parent_id else f"ground; relation {obj.relation} is not a support relation"
            return

        if obj.parent_id not in fixed_by_id:
            raise RuntimeError(f"SDF support requires parent {obj.parent_id} to be optimized before child {obj.id}.")
        parent = fixed_by_id[obj.parent_id]
        target_surface_z, support_model = self._support_target(obj, template, parent, registry)
        obj.placement.z = float(target_surface_z - local_min_z * obj.placement.scale)
        self._clamp_to_parent_xy(obj, template, parent)
        event["support"] = {
            "parent_id": parent.object_id,
            "target_surface_z": round(float(target_surface_z), 6),
            "support_model": support_model,
        }

    def _clamp_to_parent_xy(self, obj: Any, template: MeshTemplate, parent: PlacedMesh) -> None:
        child_bounds = self._world_bounds(template, obj.placement)
        child_size = child_bounds[1] - child_bounds[0]
        for axis, attr in [(0, "x"), (1, "y")]:
            low = parent.bounds[0, axis] + child_size[axis] * 0.5
            high = parent.bounds[1, axis] - child_size[axis] * 0.5
            if low > high:
                raise RuntimeError(f"child object {obj.id} does not fit within parent support footprint {parent.object_id}.")
            value = float(getattr(obj.placement, attr))
            setattr(obj.placement, attr, min(max(value, low), high))

    def _enforce_scene_bounds(self, obj: Any, template: MeshTemplate, bounds: list[float]) -> None:
        room = np.asarray(bounds, dtype=np.float64)
        world = self._world_bounds(template, obj.placement)
        limits = [(0.0, room[0]), (0.0, room[1]), (0.0, room[2])]
        for axis, attr in [(0, "x"), (1, "y"), (2, "z")]:
            size = world[1, axis] - world[0, axis]
            if size > limits[axis][1] - limits[axis][0]:
                raise RuntimeError(f"object {obj.id} is too large for scene bounds on axis {axis}.")
            shift = 0.0
            if world[0, axis] < limits[axis][0]:
                shift = limits[axis][0] - world[0, axis]
            elif world[1, axis] > limits[axis][1]:
                shift = limits[axis][1] - world[1, axis]
            if shift:
                setattr(obj.placement, attr, float(getattr(obj.placement, attr) + shift))
                world[:, axis] += shift

    def _free_slot_search(
        self,
        obj: Any,
        template: MeshTemplate,
        fixed: list[PlacedMesh],
        fixed_by_id: dict[str, PlacedMesh],
        registry: AssetRegistry,
        scene_bounds: list[float],
        event: dict[str, Any],
    ) -> dict[str, Any]:
        start = (float(obj.placement.x), float(obj.placement.y), float(obj.placement.z), float(obj.placement.scale))
        support_parent_id = obj.parent_id if obj.relation in {"on", "inside"} else None
        best: dict[str, Any] = {
            "penetrating_points": 10**9,
            "mean_penetration_m": float("inf"),
            "candidate": None,
        }
        checked = 0
        for x, y in self._candidate_xy_positions(obj, template, fixed_by_id, scene_bounds):
            obj.placement.x = float(x)
            obj.placement.y = float(y)
            self._apply_support_constraint(obj, template, fixed_by_id, registry, event, scene_bounds)
            self._enforce_scene_bounds(obj, template, scene_bounds)
            points = self._transform_points(template.surface_points, obj.placement)
            bounds = self._world_bounds(template, obj.placement)
            centroid = points.mean(axis=0)
            _, penetration_count, mean_penetration, _ = self._collision_update(
                points,
                centroid,
                bounds,
                fixed,
                support_parent_id,
            )
            checked += 1
            if (
                penetration_count < best["penetrating_points"]
                or (
                    penetration_count == best["penetrating_points"]
                    and mean_penetration < best["mean_penetration_m"]
                )
            ):
                best = {
                    "penetrating_points": int(penetration_count),
                    "mean_penetration_m": float(mean_penetration),
                    "candidate": {
                        "x": float(obj.placement.x),
                        "y": float(obj.placement.y),
                        "z": float(obj.placement.z),
                        "scale": float(obj.placement.scale),
                    },
                }
            if penetration_count == 0:
                event["free_slot_search"] = {
                    "status": "resolved",
                    "checked_candidates": checked,
                    "placement": {
                        "x": float(obj.placement.x),
                        "y": float(obj.placement.y),
                        "z": float(obj.placement.z),
                        "scale": float(obj.placement.scale),
                    },
                }
                event["iterations"].append(
                    {
                        "iteration": "free_slot_search",
                        "penetrating_points": 0,
                        "mean_penetration_m": 0.0,
                    }
                )
                return {"penetrating_points": 0, "mean_penetration_m": 0.0}

        obj.placement.x, obj.placement.y, obj.placement.z, obj.placement.scale = start
        event["free_slot_search"] = {
            "status": "failed",
            "checked_candidates": checked,
            "best": best,
        }
        return {
            "penetrating_points": int(best["penetrating_points"]),
            "mean_penetration_m": float(best["mean_penetration_m"]),
        }

    def _candidate_xy_positions(
        self,
        obj: Any,
        template: MeshTemplate,
        fixed_by_id: dict[str, PlacedMesh],
        scene_bounds: list[float],
    ) -> list[tuple[float, float]]:
        current_bounds = self._world_bounds(template, obj.placement)
        half_size = (current_bounds[1] - current_bounds[0])[:2] * 0.5
        margin = max(self.config.collision_tolerance_m * 3.0, 0.03)
        if obj.parent_id and obj.relation in {"on", "inside"}:
            if obj.parent_id not in fixed_by_id:
                raise RuntimeError(f"cannot search support slots for {obj.id}; parent {obj.parent_id} is not fixed")
            parent = fixed_by_id[obj.parent_id]
            min_x = float(parent.bounds[0, 0] + half_size[0] + margin)
            max_x = float(parent.bounds[1, 0] - half_size[0] - margin)
            min_y = float(parent.bounds[0, 1] + half_size[1] + margin)
            max_y = float(parent.bounds[1, 1] - half_size[1] - margin)
        else:
            room = np.asarray(scene_bounds, dtype=np.float64)
            min_x = float(half_size[0] + margin)
            max_x = float(room[0] - half_size[0] - margin)
            min_y = float(half_size[1] + margin)
            max_y = float(room[1] - half_size[1] - margin)
        if min_x > max_x or min_y > max_y:
            raise RuntimeError(f"object {obj.id} has no feasible XY slot within required support or scene bounds")

        step = max(0.22, float(max(half_size[0], half_size[1]) * 0.9))
        xs = self._axis_candidates(min_x, max_x, step)
        ys = self._axis_candidates(min_y, max_y, step)
        start_xy = (float(obj.placement.x), float(obj.placement.y))
        candidates = [(start_xy[0], start_xy[1])]
        candidates.extend((x, y) for x in xs for y in ys)
        unique = []
        seen: set[tuple[int, int]] = set()
        for x, y in candidates:
            key = (int(round(x * 1000)), int(round(y * 1000)))
            if key in seen:
                continue
            seen.add(key)
            if min_x <= x <= max_x and min_y <= y <= max_y:
                unique.append((float(x), float(y)))
        unique.sort(key=lambda xy: (xy[0] - start_xy[0]) ** 2 + (xy[1] - start_xy[1]) ** 2)
        return unique

    def _axis_candidates(self, minimum: float, maximum: float, step: float) -> list[float]:
        if maximum - minimum <= step:
            return [float((minimum + maximum) * 0.5)]
        values = list(np.arange(minimum, maximum + step * 0.5, step, dtype=np.float64))
        values = [float(min(max(value, minimum), maximum)) for value in values]
        values.extend([minimum, maximum, (minimum + maximum) * 0.5])
        deduped = sorted({round(value, 6) for value in values})
        return [float(value) for value in deduped]

    def _collision_update(
        self,
        points: np.ndarray,
        centroid: np.ndarray,
        moving_bounds: np.ndarray,
        fixed: list[PlacedMesh],
        support_parent_id: str | None = None,
    ) -> tuple[np.ndarray, int, float, float]:
        if not fixed:
            return np.zeros(3, dtype=np.float64), 0, 0.0, 0.0
        updates: list[np.ndarray] = []
        penetrations: list[np.ndarray] = []
        direction_sets = 0
        for placed in fixed:
            if support_parent_id and placed.object_id == support_parent_id:
                continue
            overlap = np.minimum(moving_bounds[1], placed.bounds[1]) - np.maximum(moving_bounds[0], placed.bounds[0])
            if bool(np.any(overlap <= -self.config.collision_tolerance_m)):
                continue
            if bool(np.all(overlap > self.config.collision_tolerance_m)):
                horizontal_axes = [axis for axis in (0, 1) if overlap[axis] > self.config.collision_tolerance_m]
                axis = min(horizontal_axes, key=lambda value: float(overlap[value])) if horizontal_axes else int(np.argmin(overlap))
                sign = 1.0 if centroid[axis] >= placed.centroid[axis] else -1.0
                if abs(float(centroid[axis] - placed.centroid[axis])) < 1e-8:
                    sign = 1.0
                update = np.zeros(3, dtype=np.float64)
                update[axis] = sign * float(overlap[axis] + self.config.collision_tolerance_m * 2.0)
                updates.append(update)
                penetrations.append(np.asarray([float(overlap[axis])], dtype=np.float64))
                direction_sets += 1
            signed = placed.query.signed_distance(points)
            inside = signed > self.config.collision_tolerance_m
            if not bool(np.any(inside)):
                continue
            penetration = signed[inside]
            direction = centroid - placed.centroid
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm < 1e-8:
                hit_centroid = points[inside].mean(axis=0)
                direction = centroid - hit_centroid
                direction_norm = float(np.linalg.norm(direction))
            if direction_norm < 1e-8:
                direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
                direction_norm = 1.0
            direction = direction / direction_norm
            updates.append(direction * float(np.mean(penetration)))
            penetrations.append(penetration)
            direction_sets += 1
        if not updates:
            return np.zeros(3, dtype=np.float64), 0, 0.0, 0.0
        all_penetrations = np.concatenate(penetrations)
        mean_penetration = float(np.mean(all_penetrations))
        update = np.mean(np.vstack(updates), axis=0)
        scale_update = 0.0
        if direction_sets > 1:
            scale_update = -min(self.config.scale_step_limit, mean_penetration * 0.15)
        return update, int(len(all_penetrations)), mean_penetration, scale_update

    def _make_placed_mesh(self, obj: Any, template: MeshTemplate, trimesh: Any, proximity_query: Any) -> PlacedMesh:
        vertices = self._transform_points(template.vertices, obj.placement)
        mesh = trimesh.Trimesh(vertices=vertices, faces=template.faces, process=False)
        bounds = np.asarray(mesh.bounds, dtype=np.float64)
        support_planes = [float(obj.placement.z + plane * obj.placement.scale) for plane in template.support_planes]
        return PlacedMesh(
            object_id=obj.id,
            asset_id=template.asset_id,
            mesh=mesh,
            query=proximity_query(mesh),
            centroid=np.asarray(mesh.centroid, dtype=np.float64),
            bounds=bounds,
            support_planes=support_planes,
        )

    def _support_target(
        self,
        obj: Any,
        template: MeshTemplate,
        parent: PlacedMesh,
        registry: AssetRegistry,
    ) -> tuple[float, str]:
        parent_spec = registry.get(parent.asset_id)
        current_bottom = self._world_bounds(template, obj.placement)[0, 2]
        if parent.support_planes:
            return (
                float(min(parent.support_planes, key=lambda z: abs(z - current_bottom))),
                "mesh_derived_support_plane",
            )
        if parent_spec.support_kind == "container":
            raise RuntimeError(
                f"parent mesh {parent.asset_id} for {obj.id} has no derived support planes; "
                "cannot validate child support from rendered geometry."
            )
        if parent_spec.support_heights:
            parent_height = float(parent.bounds[1, 2] - parent.bounds[0, 2])
            target_candidates = [parent.bounds[0, 2] + parent_height * h for h in parent_spec.support_heights]
            return (
                float(min(target_candidates, key=lambda z: abs(z - current_bottom))),
                "registry_support_plane",
            )
        return float(parent.bounds[1, 2]), "parent_mesh_top"

    def _support_error(
        self,
        obj: Any,
        template: MeshTemplate,
        fixed_by_id: dict[str, PlacedMesh],
        registry: AssetRegistry,
    ) -> float:
        asset = registry.get(template.asset_id)
        tags = {tag.lower() for tag in asset.tags}
        if (not obj.parent_id or obj.relation not in {"on", "inside"}) and (
            asset.category == "door"
            or ("ceiling" in tags and "floor" not in tags)
            or ("wall" in tags and "floor" not in tags)
        ):
            return 0.0
        if not obj.parent_id or obj.relation not in {"on", "inside"}:
            return float(abs(self._world_bounds(template, obj.placement)[0, 2]))
        parent = fixed_by_id[obj.parent_id]
        target_surface_z, _support_model = self._support_target(obj, template, parent, registry)
        bottom_z = float(self._world_bounds(template, obj.placement)[0, 2])
        return abs(bottom_z - target_surface_z)
