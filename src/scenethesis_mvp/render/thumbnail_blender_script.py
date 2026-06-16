from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


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
    setup_camera(imported)
    setup_lights()
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.film_transparent = True
    scene.render.image_settings.file_format = "PNG"
    output_path = Path(item["thumbnail_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_path)
    bpy.ops.render.render(write_still=True)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def normalize_objects(objects: list, dimensions: list[float]) -> None:
    min_corner, max_corner = object_bounds(objects)
    source_size = max_corner - min_corner
    target = Vector(dimensions)
    scale = min(
        target.x / max(source_size.x, 1e-6),
        target.y / max(source_size.y, 1e-6),
        target.z / max(source_size.z, 1e-6),
    )
    center = (min_corner + max_corner) * 0.5
    for obj in objects:
        obj.location -= center
        obj.scale *= scale
    min_corner, max_corner = object_bounds(objects)
    center = (min_corner + max_corner) * 0.5
    for obj in objects:
        obj.location -= center
        obj.location.z -= min_corner.z


def object_bounds(objects: list) -> tuple[Vector, Vector]:
    points = []
    for obj in objects:
        for corner in obj.bound_box:
            points.append(obj.matrix_world @ Vector(corner))
    min_corner = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    max_corner = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return min_corner, max_corner


def setup_camera(objects: list) -> None:
    min_corner, max_corner = object_bounds(objects)
    center = (min_corner + max_corner) * 0.5
    size = max((max_corner - min_corner).x, (max_corner - min_corner).y, (max_corner - min_corner).z, 0.5)
    camera_data = bpy.data.cameras.new("thumbnail_camera")
    camera = bpy.data.objects.new("thumbnail_camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = center + Vector((size * 1.4, -size * 2.2, size * 1.2))
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = size * 1.45
    bpy.context.scene.camera = camera


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
