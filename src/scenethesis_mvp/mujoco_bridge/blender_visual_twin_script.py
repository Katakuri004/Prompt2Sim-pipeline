from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Matrix, Quaternion, Vector


def main() -> None:
    args = _parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    frames_dir = Path(payload["frames_dir"])
    frames_dir.mkdir(parents=True, exist_ok=True)
    trace = json.loads(Path(payload["state_trace_path"]).read_text(encoding="utf-8")).get("trace", [])
    if not trace:
        raise RuntimeError("rollout state trace has no frames")

    _clear_scene()
    bpy.ops.import_scene.gltf(filepath=payload["source_scene_glb"])
    target_objects = _objects_for_entity(payload["target_object"])
    if not target_objects:
        raise RuntimeError(f"could not find target visual objects in GLB: {payload['target_object']}")
    dynamic_root = _duplicate_dynamic_entity(payload["target_object"], target_objects, trace[0].get("target_pose"))
    _add_destination_marker(payload["destination_position"])
    panda_parts = _create_panda_markers()
    camera = _setup_camera(payload, trace[0])
    _configure_render(payload["resolution"])
    _configure_lighting()

    stride = max(1, int(payload.get("frame_stride", 1)))
    max_frames = int(payload.get("max_frames", 240))
    frame_index = 0
    for sample in trace[::stride][:max_frames]:
        _apply_target_pose(dynamic_root, sample.get("target_pose"))
        _apply_panda_pose(panda_parts, sample)
        _update_camera(camera, payload, sample)
        bpy.context.scene.frame_set(frame_index)
        bpy.context.scene.render.filepath = str(frames_dir / f"frame_{frame_index:05d}.png")
        bpy.ops.render.render(write_still=True)
        frame_index += 1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    argv = []
    if "--" in __import__("sys").argv:
        argv = __import__("sys").argv[__import__("sys").argv.index("--") + 1 :]
    return parser.parse_args(argv)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def _objects_for_entity(entity_id: str) -> list:
    prefix = entity_id.lower()
    return [
        obj
        for obj in bpy.context.scene.objects
        if obj.type == "MESH" and obj.name.lower().startswith(prefix)
    ]


def _duplicate_dynamic_entity(entity_id: str, source_objects: list, initial_pose: dict | None):
    root = bpy.data.objects.new(f"{entity_id}_dynamic_root", None)
    bpy.context.collection.objects.link(root)
    _apply_target_pose(root, initial_pose)
    for source in source_objects:
        duplicate = source.copy()
        duplicate.data = source.data.copy()
        duplicate.animation_data_clear()
        bpy.context.collection.objects.link(duplicate)
        duplicate.matrix_world = source.matrix_world.copy()
        duplicate.parent = root
        duplicate.matrix_parent_inverse = root.matrix_world.inverted()
        source.hide_render = True
        source.hide_viewport = True
    return root


def _add_destination_marker(position: list[float]) -> None:
    loc = _canonical_to_blender_vec(position)
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    marker = bpy.context.object
    marker.name = "generated_destination_marker"
    marker.dimensions = (0.16, 0.16, 0.035)
    mat = bpy.data.materials.new("task_destination_green")
    mat.diffuse_color = (0.05, 0.85, 0.08, 0.45)
    marker.data.materials.append(mat)


def _create_panda_markers() -> dict[str, object]:
    mat = bpy.data.materials.new("mujoco_panda_marker")
    mat.diffuse_color = (0.86, 0.86, 0.82, 1.0)
    dark_mat = bpy.data.materials.new("mujoco_panda_dark")
    dark_mat.diffuse_color = (0.05, 0.05, 0.05, 1.0)
    parts = {"segments": {}, "spheres": {}}
    for name in ("panda_base", "panda_ee"):
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=0.055)
        obj = bpy.context.object
        obj.name = name
        obj.data.materials.append(mat)
        parts["spheres"][name] = obj
    pairs = [
        ("panda_base", "panda_link1"),
        ("panda_link1", "panda_link2"),
        ("panda_link2", "panda_link3"),
        ("panda_link3", "panda_link4"),
        ("panda_link4", "panda_link5"),
        ("panda_link5", "panda_link6"),
        ("panda_link6", "panda_hand"),
        ("panda_hand", "panda_ee"),
    ]
    for start_name, end_name in pairs:
        bpy.ops.mesh.primitive_cylinder_add(vertices=24, radius=1.0, depth=1.0)
        obj = bpy.context.object
        obj.name = f"visual_{start_name}_to_{end_name}"
        obj.data.materials.append(dark_mat if "finger" in start_name or "finger" in end_name else mat)
        parts["segments"][(start_name, end_name)] = obj
    return parts


def _setup_camera(payload: dict, first_sample: dict):
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.name = str(payload.get("camera_name", "report_task_closeup"))
    camera.data.lens = 28
    bpy.context.scene.camera = camera
    _update_camera(camera, payload, first_sample)
    return camera


def _update_camera(camera, payload: dict, sample: dict) -> None:
    target_pose = sample.get("target_pose") or {}
    target = target_pose.get("position") or payload.get("destination_position") or [0, 0, 0]
    target_blender = _canonical_to_blender_vec(target)
    offset = Vector((1.25, -1.55, 0.95))
    camera.location = target_blender + offset
    _look_at(camera, target_blender + Vector((0.0, 0.0, 0.08)))


def _apply_target_pose(root, target_pose: dict | None) -> None:
    if not target_pose:
        return
    position = target_pose.get("position")
    xmat = target_pose.get("xmat")
    if not position:
        return
    root.location = _canonical_to_blender_vec(position)
    if xmat:
        root.rotation_euler = _canonical_xmat_to_blender_quat(xmat).to_euler()


def _apply_panda_pose(parts: dict[str, object], sample: dict) -> None:
    body_poses = sample.get("body_poses", {})
    positions = {}
    for name, pose in body_poses.items():
        if isinstance(pose, dict) and pose.get("position"):
            positions[name] = _canonical_to_blender_vec(pose.get("position", [0, 0, 0]))
    ee = sample.get("ee_position")
    if ee:
        positions["panda_ee"] = _canonical_to_blender_vec(ee)
    for name, obj in parts.get("spheres", {}).items():
        if name in positions:
            obj.location = positions[name]
    for pair, obj in parts.get("segments", {}).items():
        start_name, end_name = pair
        if start_name not in positions or end_name not in positions:
            obj.hide_render = True
            obj.hide_viewport = True
            continue
        _set_cylinder_between(obj, positions[start_name], positions[end_name], 0.035 if end_name != "panda_ee" else 0.026)


def _set_cylinder_between(obj, start: Vector, end: Vector, radius: float) -> None:
    direction = end - start
    length = direction.length
    if length <= 1e-5:
        obj.hide_render = True
        obj.hide_viewport = True
        return
    obj.hide_render = False
    obj.hide_viewport = False
    obj.location = start + direction * 0.5
    obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
    obj.scale = (radius, radius, length)


def _canonical_to_blender_vec(value: list[float]) -> Vector:
    x, y, z = [float(item) for item in value]
    return Vector((x, y, z))


def _canonical_xmat_to_blender_quat(xmat: list) -> Quaternion:
    mat = Matrix(((float(xmat[0][0]), float(xmat[0][1]), float(xmat[0][2])),
                  (float(xmat[1][0]), float(xmat[1][1]), float(xmat[1][2])),
                  (float(xmat[2][0]), float(xmat[2][1]), float(xmat[2][2]))))
    return mat.to_quaternion()


def _look_at(obj, target: Vector) -> None:
    direction = target - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def _configure_render(resolution: list[int]) -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except Exception:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.5, 0.5, 0.52)


def _configure_lighting() -> None:
    bpy.ops.object.light_add(type="AREA", location=(3.5, 4.0, 3.5))
    key = bpy.context.object
    key.name = "visual_twin_key_light"
    key.data.energy = 600
    key.data.size = 4.0
    bpy.ops.object.light_add(type="POINT", location=(6.0, 2.5, 2.2))
    fill = bpy.context.object
    fill.name = "visual_twin_fill_light"
    fill.data.energy = 120


if __name__ == "__main__":
    main()
