from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Vector


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    for item in payload["assets"]:
        render_thumbnail(item, tuple(payload.get("resolution", [512, 512])))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    else:
        argv = []
    parser.add_argument("--input", required=True)
    return parser.parse_args(argv)


def render_thumbnail(item: dict, resolution: tuple[int, int]) -> None:
    clear_scene()
    bpy.ops.import_scene.gltf(filepath=item["mesh_path"])
    imported = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not imported:
        raise RuntimeError(f"Imported mesh has no mesh objects: {item['mesh_path']}")
    normalize_objects(imported, item["dimensions"])
    camera, center, size = setup_camera(imported)
    setup_lights()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    render_output(scene, Path(item["thumbnail_path"]))
    view_paths = item.get("view_paths") or {}
    expected_views = {"front", "side", "oblique"}
    if view_paths and set(view_paths) != expected_views:
        raise RuntimeError(f"view_paths must contain exactly {sorted(expected_views)} for {item['asset_id']}")
    view_offsets = {
        "front": Vector((0.0, -2.6, 0.75)),
        "side": Vector((2.6, 0.0, 0.75)),
        "oblique": Vector((1.7, -2.4, 1.25)),
    }
    for view_name, output in view_paths.items():
        set_camera_view(camera, center, size, view_offsets[view_name])
        render_output(scene, Path(output))


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def normalize_objects(objects: list, dimensions: list[float]) -> None:
    bpy.context.view_layer.update()
    min_corner, max_corner = object_bounds(objects)
    source_size = max_corner - min_corner
    target = Vector(dimensions)
    scale = min(
        target.x / max(source_size.x, 1e-6),
        target.y / max(source_size.y, 1e-6),
        target.z / max(source_size.z, 1e-6),
    )
    center = (min_corner + max_corner) * 0.5
    transform = Matrix.Translation((-center.x * scale, -center.y * scale, -min_corner.z * scale)) @ Matrix.Scale(scale, 4)
    world_matrices = {obj: obj.matrix_world.copy() for obj in objects}
    for obj in objects:
        obj.parent = None
    for obj in objects:
        obj.matrix_world = transform @ world_matrices[obj]
    bpy.context.view_layer.update()


def object_bounds(objects: list) -> tuple[Vector, Vector]:
    points = []
    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    min_corner = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_corner = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_corner, max_corner


def setup_camera(objects: list) -> tuple[object, Vector, float]:
    min_corner, max_corner = object_bounds(objects)
    center = (min_corner + max_corner) * 0.5
    size = max((max_corner - min_corner).x, (max_corner - min_corner).y, (max_corner - min_corner).z, 0.5)
    camera_data = bpy.data.cameras.new("thumbnail_camera")
    camera = bpy.data.objects.new("thumbnail_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = size * 1.45
    bpy.context.scene.camera = camera
    set_camera_view(camera, center, size, Vector((1.7, -2.4, 1.25)))
    return camera, center, size


def set_camera_view(camera: object, center: Vector, size: float, direction_scale: Vector) -> None:
    camera.location = center + Vector(
        (
            size * direction_scale.x,
            size * direction_scale.y,
            size * direction_scale.z,
        )
    )
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def render_output(scene: object, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def setup_lights() -> None:
    world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.color = (1.0, 1.0, 1.0)
    light_data = bpy.data.lights.new("key_light", "AREA")
    light_data.energy = 450
    light_data.size = 4.0
    light = bpy.data.objects.new("key_light", light_data)
    bpy.context.collection.objects.link(light)
    light.location = (3.0, -4.0, 5.0)


if __name__ == "__main__":
    main()
