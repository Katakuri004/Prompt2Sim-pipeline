from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import combinations
from pathlib import Path

import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def make_material(name: str, color: list[float], texture: dict | None = None):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Roughness"].default_value = 0.72
        if texture:
            diffuse = texture.get("diffuse")
            if diffuse and Path(diffuse).is_file():
                tex_node = nodes.new("ShaderNodeTexImage")
                tex_node.image = bpy.data.images.load(str(diffuse))
                links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])
            normal = texture.get("normal")
            if normal and Path(normal).is_file():
                normal_tex = nodes.new("ShaderNodeTexImage")
                normal_tex.image = bpy.data.images.load(str(normal))
                normal_tex.image.colorspace_settings.name = "Non-Color"
                normal_map = nodes.new("ShaderNodeNormalMap")
                normal_map.inputs["Strength"].default_value = 0.18
                links.new(normal_tex.outputs["Color"], normal_map.inputs["Color"])
                links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
            roughness = texture.get("roughness")
            if roughness and Path(roughness).is_file():
                rough_tex = nodes.new("ShaderNodeTexImage")
                rough_tex.image = bpy.data.images.load(str(roughness))
                rough_tex.image.colorspace_settings.name = "Non-Color"
                links.new(rough_tex.outputs["Color"], bsdf.inputs["Roughness"])
    return material


def material_has_image_texture(material) -> bool:
    if not material or not getattr(material, "use_nodes", False) or not material.node_tree:
        return False
    for node in material.node_tree.nodes:
        if node.bl_idname == "ShaderNodeTexImage" and getattr(node, "image", None):
            return True
    return False


def material_has_linked_bsdf_input(material, input_name: str) -> bool:
    if not material or not getattr(material, "use_nodes", False) or not material.node_tree:
        return False
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    return bool(bsdf and input_name in bsdf.inputs and bsdf.inputs[input_name].links)


def make_emissive_material(name: str, color: list[float], strength: float):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    bsdf = nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Emission Color"].default_value = (color[0], color[1], color[2], 1.0)
        bsdf.inputs["Emission Strength"].default_value = strength
    return material


def add_procedural_detail(
    material,
    color: list[float],
    noise_scale: float = 32.0,
    color_variation: float = 0.12,
    bump_strength: float = 0.035,
    bump_distance: float = 0.05,
    metallic: float = 0.0,
    affect_base_color: bool = True,
):
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    if not bsdf:
        return material
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = 0.78

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = noise_scale
    noise.inputs["Detail"].default_value = 10.0
    noise.inputs["Roughness"].default_value = 0.58

    ramp = nodes.new("ShaderNodeValToRGB")
    low = [max(0.0, channel * (1.0 - color_variation)) for channel in color]
    high = [min(1.0, channel * (1.0 + color_variation) + 0.025) for channel in color]
    ramp.color_ramp.elements[0].position = 0.22
    ramp.color_ramp.elements[0].color = (low[0], low[1], low[2], 1.0)
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (high[0], high[1], high[2], 1.0)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    if affect_base_color and not material_has_linked_bsdf_input(material, "Base Color"):
        links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])

    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = bump_strength
    bump.inputs["Distance"].default_value = bump_distance
    if bump_strength > 0.0 and not material_has_linked_bsdf_input(material, "Normal"):
        links.new(noise.outputs["Fac"], bump.inputs["Height"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return material


def make_detailed_material(
    name: str,
    color: list[float],
    noise_scale: float = 32.0,
    color_variation: float = 0.12,
    bump_strength: float = 0.035,
    metallic: float = 0.0,
):
    return add_procedural_detail(
        make_material(name, color),
        color,
        noise_scale=noise_scale,
        color_variation=color_variation,
        bump_strength=bump_strength,
        metallic=metallic,
    )


def make_semantic_materials() -> dict[str, object]:
    return {
        "aluminum": make_detailed_material("semantic_brushed_aluminum", [0.62, 0.62, 0.58], 48.0, 0.06, 0.018, 0.35),
        "black_rubber": make_detailed_material("semantic_black_rubber", [0.025, 0.024, 0.022], 42.0, 0.08, 0.045),
        "blue_plastic": make_detailed_material("semantic_blue_plastic", [0.04, 0.20, 0.48], 36.0, 0.10, 0.025),
        "cardboard": make_detailed_material("semantic_warm_cardboard", [0.58, 0.36, 0.16], 28.0, 0.22, 0.045),
        "dark_metal": make_detailed_material("semantic_dark_metal", [0.045, 0.048, 0.046], 52.0, 0.08, 0.015, 0.45),
        "forklift_orange": make_detailed_material("semantic_forklift_orange", [0.92, 0.36, 0.06], 34.0, 0.10, 0.02),
        "glass": make_detailed_material("semantic_blue_gray_glass", [0.32, 0.48, 0.56], 18.0, 0.04, 0.004),
        "green_rack": make_detailed_material("semantic_green_rack_metal", [0.02, 0.48, 0.32], 38.0, 0.08, 0.018, 0.25),
        "light_fixture": make_detailed_material("semantic_light_fixture", [0.82, 0.78, 0.68], 30.0, 0.07, 0.012),
        "safety_orange": make_detailed_material("semantic_safety_orange", [0.95, 0.28, 0.04], 30.0, 0.08, 0.016),
        "safety_yellow": make_detailed_material("semantic_safety_yellow", [0.96, 0.76, 0.04], 24.0, 0.06, 0.012),
        "wall_metal": make_detailed_material("semantic_wall_metal", [0.42, 0.47, 0.48], 42.0, 0.08, 0.018, 0.20),
        "wood": make_detailed_material("semantic_aged_wood", [0.48, 0.28, 0.12], 18.0, 0.24, 0.055),
    }


def semantic_style_for_mesh(asset: dict, obj_name: str) -> tuple[str, list[float], float, float, float, float]:
    category = str(asset.get("category", "")).lower()
    name = obj_name.lower()
    tags = {str(tag).lower() for tag in asset.get("tags", [])}
    if category == "shelf":
        if any(token in name for token in ("cardbox", "carton", "box")):
            return ("cardboard", [0.58, 0.36, 0.16], 28.0, 0.22, 0.045, 0.0)
        if "rackleg" in name or ("rack" in name and "rackshelf" not in name):
            return ("green_rack", [0.02, 0.48, 0.32], 38.0, 0.08, 0.018, 0.25)
        if any(token in name for token in ("rackshelf", "board", "wood")):
            return ("wood", [0.48, 0.28, 0.12], 18.0, 0.24, 0.055, 0.0)
        return ("green_rack", [0.02, 0.48, 0.32], 38.0, 0.08, 0.018, 0.25)
    if category == "forklift":
        if "glass" in name:
            return ("glass", [0.32, 0.48, 0.56], 18.0, 0.04, 0.004, 0.0)
        if any(token in name for token in ("tire", "rubber", "wheel")):
            return ("black_rubber", [0.025, 0.024, 0.022], 42.0, 0.08, 0.045, 0.0)
        if any(token in name for token in ("plastic", "decal", "forklift")):
            return ("forklift_orange", [0.92, 0.36, 0.06], 34.0, 0.10, 0.02, 0.0)
        return ("dark_metal", [0.045, 0.048, 0.046], 52.0, 0.08, 0.015, 0.45)
    if category == "cart":
        if any(token in name for token in ("rubber", "wheel")):
            return ("black_rubber", [0.025, 0.024, 0.022], 42.0, 0.08, 0.045, 0.0)
        return ("aluminum", [0.62, 0.62, 0.58], 48.0, 0.06, 0.018, 0.35)
    if category == "pallet":
        return ("wood", [0.48, 0.28, 0.12], 18.0, 0.24, 0.055, 0.0)
    if category == "pallet_load":
        if "pallet" in name:
            return ("wood", [0.48, 0.28, 0.12], 18.0, 0.24, 0.055, 0.0)
        return ("cardboard", [0.58, 0.36, 0.16], 28.0, 0.22, 0.045, 0.0)
    if category == "box":
        if "wood" in name or "wooden" in tags:
            return ("wood", [0.48, 0.28, 0.12], 18.0, 0.24, 0.055, 0.0)
        if "plastic" in name or "plastic" in tags:
            return ("blue_plastic", [0.04, 0.20, 0.48], 36.0, 0.10, 0.025, 0.0)
        return ("cardboard", [0.58, 0.36, 0.16], 28.0, 0.22, 0.045, 0.0)
    if category == "barrier":
        return ("safety_orange", [0.95, 0.28, 0.04], 30.0, 0.08, 0.016, 0.0)
    if category == "floor_marking":
        return ("safety_yellow", [0.96, 0.76, 0.04], 24.0, 0.06, 0.012, 0.0)
    if category == "hand_truck":
        if any(token in name for token in ("rubber", "wheel", "tire")):
            return ("black_rubber", [0.025, 0.024, 0.022], 42.0, 0.08, 0.045, 0.0)
        return ("aluminum", [0.62, 0.62, 0.58], 48.0, 0.06, 0.018, 0.35)
    if category == "ladder":
        return ("wood", [0.48, 0.28, 0.12], 18.0, 0.24, 0.055, 0.0)
    if category == "cylinder":
        return ("painted_metal", asset.get("color", [0.45, 0.48, 0.48]), 36.0, 0.10, 0.026, 0.20)
    if category in {"bin", "container"}:
        return ("container", asset.get("color", [0.35, 0.35, 0.32]), 34.0, 0.12, 0.026, 0.10)
    if category == "cabinet":
        return ("cabinet", asset.get("color", [0.45, 0.08, 0.07]), 34.0, 0.10, 0.018, 0.25)
    if category in {"scanner", "tool"}:
        return ("dark_metal", [0.045, 0.048, 0.046], 52.0, 0.08, 0.015, 0.45)
    if category in {"light", "camera"}:
        return ("light_fixture", [0.82, 0.78, 0.68], 30.0, 0.07, 0.012, 0.0)
    if category in {"cable", "duct", "pipe", "utility_box", "door"}:
        return ("wall_metal", [0.42, 0.47, 0.48], 42.0, 0.08, 0.018, 0.20)
    return ("default", asset.get("color", [0.6, 0.6, 0.6]), 30.0, 0.12, 0.026, 0.0)


def semantic_material_for_mesh(asset: dict, obj_name: str, materials: dict[str, object], default_material):
    material_key = semantic_style_for_mesh(asset, obj_name)[0]
    return materials.get(material_key, default_material)


def enhance_imported_material(material, asset: dict, obj_name: str) -> None:
    if not material:
        return
    if not material.use_nodes:
        material.use_nodes = True
    _name, color, noise_scale, color_variation, bump_strength, metallic = semantic_style_for_mesh(asset, obj_name)
    source = str(asset.get("source", "")).lower()
    force_semantic_color = source == "huggingface_simready"
    preserve_embedded_color = source == "project_authored"
    if force_semantic_color and material.node_tree:
        bsdf = material.node_tree.nodes.get("Principled BSDF")
        if bsdf and "Base Color" in bsdf.inputs:
            for link in list(bsdf.inputs["Base Color"].links):
                material.node_tree.links.remove(link)
    add_procedural_detail(
        material,
        color,
        noise_scale=noise_scale,
        color_variation=color_variation,
        bump_strength=bump_strength,
        metallic=metallic,
        affect_base_color=force_semantic_color or (not preserve_embedded_color and not material_has_image_texture(material)),
    )


def assign_semantic_materials(mesh_objects: list, asset: dict, materials: dict[str, object], default_material) -> None:
    source = str(asset.get("source", "")).lower()
    for obj in mesh_objects:
        if source == "huggingface_simready":
            material = semantic_material_for_mesh(asset, obj.name, materials, default_material)
            obj.data.materials.clear()
            obj.data.materials.append(material)
            continue
        if not obj.data.materials:
            obj.data.materials.append(semantic_material_for_mesh(asset, obj.name, materials, default_material))
            continue
        for slot in obj.data.materials:
            enhance_imported_material(slot, asset, obj.name)


def add_render_polish(mesh_objects: list) -> None:
    for obj in mesh_objects:
        if obj.type != "MESH":
            continue
        try:
            bevel = obj.modifiers.new(name="small_edge_bevel", type="BEVEL")
            bevel.width = 0.004
            bevel.segments = 1
            bevel.affect = "EDGES"
            normal = obj.modifiers.new(name="weighted_normals", type="WEIGHTED_NORMAL")
            normal.keep_sharp = True
        except Exception:
            continue


def add_cube(name: str, location: tuple[float, float, float], dimensions: tuple[float, float, float], yaw_deg: float, material) -> None:
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location, rotation=(0.0, 0.0, math.radians(yaw_deg)))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)


def add_cylinder(name: str, location: tuple[float, float, float], dimensions: tuple[float, float, float], yaw_deg: float, material) -> None:
    radius = max(dimensions[0], dimensions[1]) * 0.5
    bpy.ops.mesh.primitive_cylinder_add(vertices=32, radius=radius, depth=dimensions[2], location=location, rotation=(0.0, 0.0, math.radians(yaw_deg)))
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)


def add_procedural(obj_spec: dict, asset: dict, material) -> None:
    placement = obj_spec["placement"]
    scale = placement["scale"]
    dims = tuple(value * scale for value in asset["dimensions"])
    location = (placement["x"], placement["y"], placement["z"])
    yaw = placement["yaw_deg"]
    category = asset["category"]
    if category == "cylinder":
        add_cylinder(obj_spec["id"], location, dims, yaw, material)
    elif category == "robot_arm":
        base_dims = (dims[0] * 0.45, dims[1] * 0.45, dims[2] * 0.18)
        add_cylinder(obj_spec["id"] + "_base", (location[0], location[1], location[2] - dims[2] * 0.41), base_dims, yaw, material)
        add_cube(obj_spec["id"] + "_upright", (location[0], location[1], location[2] - dims[2] * 0.08), (dims[0] * 0.18, dims[1] * 0.18, dims[2] * 0.55), yaw, material)
        add_cube(obj_spec["id"] + "_arm", (location[0] + dims[0] * 0.2, location[1], location[2] + dims[2] * 0.18), (dims[0] * 0.62, dims[1] * 0.14, dims[2] * 0.12), yaw, material)
    elif category == "tool":
        add_cube(obj_spec["id"] + "_handle", (location[0], location[1], location[2]), (dims[0] * 0.72, dims[1] * 0.42, dims[2]), yaw, material)
        add_cube(obj_spec["id"] + "_head", (location[0] + dims[0] * 0.34, location[1], location[2]), (dims[0] * 0.25, dims[1], dims[2] * 1.15), yaw, material)
    elif category == "table":
        top_z = location[2] + dims[2] * 0.38
        add_cube(obj_spec["id"] + "_top", (location[0], location[1], top_z), (dims[0], dims[1], dims[2] * 0.12), yaw, material)
        for sx in (-0.42, 0.42):
            for sy in (-0.38, 0.38):
                add_cube(obj_spec["id"] + "_leg", (location[0] + sx * dims[0], location[1] + sy * dims[1], location[2] - dims[2] * 0.08), (0.08, 0.08, dims[2] * 0.76), yaw, material)
    elif category == "chair":
        add_cube(obj_spec["id"] + "_seat", (location[0], location[1], location[2] - dims[2] * 0.12), (dims[0], dims[1], dims[2] * 0.16), yaw, material)
        add_cube(obj_spec["id"] + "_back", (location[0], location[1] + dims[1] * 0.42, location[2] + dims[2] * 0.22), (dims[0], dims[1] * 0.12, dims[2] * 0.55), yaw, material)
    elif category == "shelf":
        shelf_thickness = max(0.035, dims[2] * 0.035)
        post_width = max(0.04, dims[0] * 0.045)
        for level in (-0.48, 0.0, 0.48):
            add_cube(
                obj_spec["id"] + "_shelf",
                (location[0], location[1], location[2] + level * dims[2]),
                (dims[0], dims[1], shelf_thickness),
                yaw,
                material,
            )
        for sx in (-0.46, 0.46):
            for sy in (-0.42, 0.42):
                add_cube(
                    obj_spec["id"] + "_post",
                    (location[0] + sx * dims[0], location[1] + sy * dims[1], location[2]),
                    (post_width, post_width, dims[2]),
                    yaw,
                    material,
                )
    elif category == "cabinet":
        panel = max(0.04, dims[2] * 0.04)
        add_cube(obj_spec["id"] + "_bottom", (location[0], location[1], location[2] - dims[2] * 0.48), (dims[0], dims[1], panel), yaw, material)
        add_cube(obj_spec["id"] + "_middle", (location[0], location[1], location[2]), (dims[0], dims[1], panel), yaw, material)
        add_cube(obj_spec["id"] + "_top", (location[0], location[1], location[2] + dims[2] * 0.48), (dims[0], dims[1], panel), yaw, material)
        add_cube(obj_spec["id"] + "_left", (location[0] - dims[0] * 0.48, location[1], location[2]), (panel, dims[1], dims[2]), yaw, material)
        add_cube(obj_spec["id"] + "_right", (location[0] + dims[0] * 0.48, location[1], location[2]), (panel, dims[1], dims[2]), yaw, material)
        add_cube(obj_spec["id"] + "_back", (location[0], location[1] + dims[1] * 0.48, location[2]), (dims[0], panel, dims[2]), yaw, material)
    else:
        add_cube(obj_spec["id"], location, dims, yaw, material)


def world_bbox(objects: list) -> tuple[Vector, Vector]:
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        evaluated = obj.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            points.append(evaluated.matrix_world @ Vector(corner))
    if not points:
        raise RuntimeError("imported asset did not contain mesh geometry")
    min_corner = Vector((min(point.x for point in points), min(point.y for point in points), min(point.z for point in points)))
    max_corner = Vector((max(point.x for point in points), max(point.y for point in points), max(point.z for point in points)))
    return min_corner, max_corner


def bbox_corners(min_corner: Vector, max_corner: Vector) -> list[Vector]:
    return [
        Vector((x, y, z))
        for x in (min_corner.x, max_corner.x)
        for y in (min_corner.y, max_corner.y)
        for z in (min_corner.z, max_corner.z)
    ]


def renderable_scene_meshes() -> list:
    ignored = {
        "ground",
        "back_wall",
        "left_wall",
        "floor_aisle_line",
        "floor_expansion_joint",
        "emissive_fluorescent_strip",
    }
    return [
        obj
        for obj in bpy.data.objects
        if obj.type == "MESH" and not any(obj.name.startswith(prefix) for prefix in ignored)
    ]


def presentation_camera_meshes(objects: list) -> list:
    selected = []
    for obj in objects:
        if obj.type != "MESH":
            continue
        try:
            min_corner, _max_corner = world_bbox([obj])
        except RuntimeError:
            continue
        if min_corner.z < 2.25:
            selected.append(obj)
    return selected or objects


def add_imported_mesh(obj_spec: dict, asset: dict, default_material, semantic_materials: dict[str, object]) -> None:
    mesh_path = Path(asset["resolved_glb_path"])
    if not mesh_path.is_file():
        raise RuntimeError(f"missing asset mesh for {asset['id']}: {mesh_path}")

    before = set(bpy.data.objects)
    try:
        bpy.ops.import_scene.gltf(filepath=str(mesh_path))
    except Exception as exc:
        raise RuntimeError(f"failed to import {mesh_path}: {exc}") from exc

    imported = [obj for obj in bpy.data.objects if obj not in before]
    for obj in list(imported):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
            imported.remove(obj)
    mesh_objects = [obj for obj in imported if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"imported asset {asset['id']} contained no mesh objects")

    for index, obj in enumerate(imported):
        obj.name = f"{obj_spec['id']}_{index}_{obj.name}"
        if obj.type == "MESH" and not obj.data.materials:
            obj.data.materials.append(default_material)
    assign_semantic_materials(mesh_objects, asset, semantic_materials, default_material)

    min_corner, max_corner = world_bbox(mesh_objects)
    source_size = max_corner - min_corner
    if min(source_size.x, source_size.y, source_size.z) <= 0:
        raise RuntimeError(f"imported asset {asset['id']} has invalid bounds")

    placement = obj_spec["placement"]
    target_dims = Vector(tuple(value * placement["scale"] for value in asset["dimensions"]))
    scale = (target_dims.x / source_size.x, target_dims.y / source_size.y, target_dims.z / source_size.z)
    source_center = (min_corner + max_corner) * 0.5
    transform = (
        Matrix.Translation(Vector((placement["x"], placement["y"], placement["z"])))
        @ Matrix.Rotation(math.radians(placement["yaw_deg"]), 4, "Z")
        @ Matrix.Diagonal((scale[0], scale[1], scale[2], 1.0))
        @ Matrix.Translation(-source_center)
    )
    for obj in imported:
        world = obj.matrix_world.copy()
        obj.parent = None
        obj.matrix_parent_inverse.identity()
        obj.matrix_world = transform @ world


def look_at(camera, target: tuple[float, float, float]) -> None:
    direction = Vector(target) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def fit_orthographic_camera(camera, objects: list, resolution: list[int], margin: float = 1.2) -> None:
    min_corner, max_corner = world_bbox(objects)
    center = (min_corner + max_corner) * 0.5
    size = max_corner - min_corner
    diagonal = max(size.length, 2.0)
    camera.location = center + Vector((diagonal * 0.72, -diagonal * 0.92, diagonal * 0.58))
    look_at(camera, (center.x, center.y, min_corner.z + max(0.45, size.z * 0.42)))
    camera.data.type = "ORTHO"
    bpy.context.view_layer.update()

    camera_inv = camera.matrix_world.inverted()
    projected = [camera_inv @ point for point in bbox_corners(min_corner, max_corner)]
    width = max(point.x for point in projected) - min(point.x for point in projected)
    height = max(point.y for point in projected) - min(point.y for point in projected)
    aspect = float(resolution[0]) / float(resolution[1])
    camera.data.ortho_scale = max(height, width / aspect, 2.8) * margin


def fit_camera_from_direction(camera, objects: list, resolution: list[int], direction: tuple[float, float, float], margin: float = 1.2) -> None:
    min_corner, max_corner = world_bbox(objects)
    center = (min_corner + max_corner) * 0.5
    size = max_corner - min_corner
    diagonal = max(size.length, 2.0)
    offset = Vector(direction).normalized() * diagonal
    camera.location = center + offset
    look_at(camera, (center.x, center.y, min_corner.z + max(0.35, size.z * 0.40)))
    camera.data.type = "ORTHO"
    bpy.context.view_layer.update()

    camera_inv = camera.matrix_world.inverted()
    projected = [camera_inv @ point for point in bbox_corners(min_corner, max_corner)]
    width = max(point.x for point in projected) - min(point.x for point in projected)
    height = max(point.y for point in projected) - min(point.y for point in projected)
    aspect = float(resolution[0]) / float(resolution[1])
    camera.data.ortho_scale = max(height, width / aspect, 1.2) * margin


def fit_presentation_camera(camera, objects: list, resolution: list[int]) -> None:
    min_corner, max_corner = world_bbox(objects)
    center = (min_corner + max_corner) * 0.5
    size = max_corner - min_corner
    diagonal = max(size.length, 2.0)
    camera.location = center + Vector((diagonal * 0.10, -diagonal * 1.08, diagonal * 0.24))
    look_at(camera, (center.x, center.y, min_corner.z + max(0.52, size.z * 0.30)))
    camera.data.type = "ORTHO"
    bpy.context.view_layer.update()

    camera_inv = camera.matrix_world.inverted()
    projected = [camera_inv @ point for point in bbox_corners(min_corner, max_corner)]
    width = max(point.x for point in projected) - min(point.x for point in projected)
    height = max(point.y for point in projected) - min(point.y for point in projected)
    aspect = float(resolution[0]) / float(resolution[1])
    camera.data.ortho_scale = max(height, width / aspect, 2.8) * 1.10


def camera_projection_metrics(camera, objects: list, resolution: list[int]) -> dict:
    min_corner, max_corner = world_bbox(objects)
    projected = [camera.matrix_world.inverted() @ point for point in bbox_corners(min_corner, max_corner)]
    min_x = min(point.x for point in projected)
    max_x = max(point.x for point in projected)
    min_y = min(point.y for point in projected)
    max_y = max(point.y for point in projected)
    frame_height = float(camera.data.ortho_scale)
    frame_width = frame_height * (float(resolution[0]) / float(resolution[1]))
    width = max_x - min_x
    height = max_y - min_y
    overflow_x = max(0.0, width - frame_width) / max(frame_width, 1e-6)
    overflow_y = max(0.0, height - frame_height) / max(frame_height, 1e-6)
    fill = min(1.0, max(0.0, (width * height) / max(frame_width * frame_height, 1e-6)))
    side_margin = min(
        (min_x + frame_width * 0.5) / max(frame_width, 1e-6),
        (frame_width * 0.5 - max_x) / max(frame_width, 1e-6),
    )
    vertical_margin = min(
        (min_y + frame_height * 0.5) / max(frame_height, 1e-6),
        (frame_height * 0.5 - max_y) / max(frame_height, 1e-6),
    )
    return {
        "fill": round(float(fill), 6),
        "overflow": round(float(overflow_x + overflow_y), 6),
        "side_margin": round(float(side_margin), 6),
        "vertical_margin": round(float(vertical_margin), 6),
        "frame_width": round(float(frame_width), 6),
        "frame_height": round(float(frame_height), 6),
        "projected_width": round(float(width), 6),
        "projected_height": round(float(height), 6),
    }


def select_presentation_camera(camera, objects: list, resolution: list[int], out_dir: Path) -> None:
    candidates = [
        {"name": "front_context", "direction": (0.10, -1.0, 0.48), "margin": 1.24, "presentation_bias": 0.62},
        {"name": "front_low", "direction": (0.10, -1.0, 0.34), "margin": 1.10, "presentation_bias": 0.18},
        {"name": "front_right", "direction": (0.58, -1.0, 0.42), "margin": 1.14, "presentation_bias": -0.20},
        {"name": "front_left", "direction": (-0.58, -1.0, 0.42), "margin": 1.14, "presentation_bias": -0.20},
        {"name": "right_aisle", "direction": (1.0, -0.35, 0.46), "margin": 1.12, "presentation_bias": -0.08},
        {"name": "left_aisle", "direction": (-1.0, -0.35, 0.46), "margin": 1.12, "presentation_bias": -0.08},
        {"name": "top_oblique", "direction": (0.18, -0.30, 0.95), "margin": 1.12, "presentation_bias": -0.24},
    ]
    scored = []
    best = None
    for candidate in candidates:
        fit_camera_from_direction(camera, objects, resolution, candidate["direction"], margin=candidate["margin"])
        metrics = camera_projection_metrics(camera, objects, resolution)
        fill = float(metrics["fill"])
        overflow = float(metrics["overflow"])
        edge_margin = min(float(metrics["side_margin"]), float(metrics["vertical_margin"]))
        direction_z = Vector(candidate["direction"]).normalized().z
        target_fill = 0.64
        fill_score = max(0.0, 1.0 - abs(fill - target_fill) / target_fill)
        margin_score = min(edge_margin / 0.065, 1.0)
        topdown_penalty = max(0.0, direction_z - 0.72)
        tight_edge_penalty = max(0.0, 0.045 - edge_margin)
        presentation_bias = float(candidate.get("presentation_bias", 0.0))
        score = (
            fill_score * 1.15
            + margin_score * 0.55
            + presentation_bias
            - overflow * 8.0
            - topdown_penalty * 0.4
            - tight_edge_penalty * 5.0
        )
        record = {
            "name": candidate["name"],
            "direction": [round(float(value), 6) for value in candidate["direction"]],
            "margin": candidate["margin"],
            "presentation_bias": round(float(presentation_bias), 6),
            "score": round(float(score), 6),
            "metrics": metrics,
        }
        scored.append(record)
        if best is None or score > best["score"]:
            best = {"score": score, "candidate": candidate, "record": record}
    assert best is not None
    fit_camera_from_direction(camera, objects, resolution, best["candidate"]["direction"], margin=best["candidate"]["margin"])
    (out_dir / "camera_selection.json").write_text(
        json.dumps(
            {
                "selected": best["record"]["name"],
                "selected_score": best["record"]["score"],
                "candidates": scored,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def configure_rendering() -> None:
    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    try:
        scene.eevee.taa_render_samples = 128
        scene.eevee.use_gtao = True
        scene.eevee.gtao_distance = 4
        scene.eevee.gtao_factor = 1.6
    except Exception:
        pass
    try:
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "Medium High Contrast"
        scene.view_settings.exposure = -0.18
        scene.view_settings.gamma = 1.0
    except Exception:
        pass
    if scene.world:
        scene.world.color = (0.50, 0.50, 0.52)


def render_png(path: Path, resolution: list[int]) -> None:
    bpy.context.scene.render.resolution_x = int(resolution[0])
    bpy.context.scene.render.resolution_y = int(resolution[1])
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def set_mesh_render_visibility(visible_meshes: set | None) -> None:
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        obj.hide_render = visible_meshes is not None and obj not in visible_meshes


def render_additional_views(camera, object_groups: dict, out_dir: Path, resolution: list[int]) -> dict:
    all_meshes = set(obj for obj in bpy.data.objects if obj.type == "MESH")
    scene_meshes = renderable_scene_meshes()
    views_dir = out_dir / "views"
    align_dir = out_dir / "alignment_views"
    views_dir.mkdir(parents=True, exist_ok=True)
    align_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict = {"scene_views": {}, "object_alignment_views": {}}
    camera_records: list[dict] = []
    scene_directions = {
        "front": (0.10, -1.0, 0.48),
        "left": (-1.0, -0.15, 0.50),
        "right": (1.0, -0.15, 0.50),
        "top_oblique": (0.18, -0.25, 1.0),
    }
    set_mesh_render_visibility(None)
    for name, direction in scene_directions.items():
        target = views_dir / f"render_{name}.png"
        fit_camera_from_direction(camera, scene_meshes, resolution, direction, margin=1.16)
        camera_records.append(camera_manifest_record(camera, f"render_{name}", resolution, "prompt2scene_blender_camera"))
        render_png(target, resolution)
        outputs["scene_views"][name] = str(target)

    object_resolution = [512, 512]
    for object_id, meshes in object_groups.items():
        visible = {obj for obj in meshes if obj.type == "MESH"}
        if not visible:
            continue
        target = align_dir / f"{object_id}.png"
        set_mesh_render_visibility(visible)
        fit_camera_from_direction(camera, list(visible), object_resolution, (0.75, -0.95, 0.55), margin=1.28)
        render_png(target, object_resolution)
        outputs["object_alignment_views"][object_id] = str(target)
    set_mesh_render_visibility(None)
    (out_dir / "render_views.json").write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "render_camera_manifest.json").write_text(
        json.dumps({"cameras": camera_records}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return outputs


def camera_manifest_record(camera, camera_name: str, resolution: list[int], source: str) -> dict:
    matrix = camera.matrix_world
    quat = matrix.to_quaternion()
    return {
        "camera_name": camera_name,
        "source": source,
        "pose_confidence": "authored",
        "position": [round(float(item), 8) for item in camera.location],
        "quaternion_wxyz": [round(float(quat.w), 8), round(float(quat.x), 8), round(float(quat.y), 8), round(float(quat.z), 8)],
        "matrix_world": [[round(float(matrix[row][col]), 8) for col in range(4)] for row in range(4)],
        "type": str(camera.data.type),
        "fov_deg": round(float(camera.data.angle) * 180.0 / math.pi, 6) if camera.data.type != "ORTHO" else None,
        "ortho_scale": round(float(camera.data.ortho_scale), 6) if camera.data.type == "ORTHO" else None,
        "resolution": [int(resolution[0]), int(resolution[1])],
    }


def add_warehouse_shell(width: float, depth: float, height: float, environment: dict) -> None:
    wall_texture = environment.get("wall_material") if environment.get("use_wall_texture", False) else None
    wall_mat = make_material("warm_plaster_warehouse_wall", [0.62, 0.46, 0.27], wall_texture)
    if wall_texture is None:
        add_procedural_detail(wall_mat, [0.62, 0.46, 0.27], noise_scale=22.0, color_variation=0.18, bump_strength=0.028)
    add_cube("back_wall", (width * 0.5, depth + 0.025, height * 0.5), (width, 0.05, height), 0.0, wall_mat)
    add_cube("left_wall", (-0.025, depth * 0.5, height * 0.5), (0.05, depth, height), 0.0, wall_mat)


def add_floor_markings(width: float, depth: float) -> None:
    yellow = make_material("safety_yellow_floor_marking", [0.95, 0.72, 0.06])
    dark = make_material("dark_expansion_joint", [0.12, 0.12, 0.11])
    for x in (width * 0.28, width * 0.62):
        add_cube("floor_aisle_line", (x, depth * 0.5, 0.002), (0.035, depth * 0.86, 0.004), 0.0, yellow)
    for y in (depth * 0.22, depth * 0.48, depth * 0.74):
        add_cube("floor_expansion_joint", (width * 0.5, y, 0.003), (width * 0.92, 0.018, 0.004), 0.0, dark)


def add_studio_lighting(width: float, depth: float, height: float) -> None:
    bpy.ops.object.light_add(type="AREA", location=(width * 0.22, depth * 0.20, height - 0.15))
    key = bpy.context.object
    key.name = "key_light"
    key.data.energy = 420
    key.data.size = 4.6
    key.data.color = (1.0, 0.82, 0.58)

    bpy.ops.object.light_add(type="AREA", location=(width * 0.72, depth * 0.68, height - 0.12))
    fill = bpy.context.object
    fill.name = "fill_light"
    fill.data.energy = 170
    fill.data.size = 5.2
    fill.data.color = (0.76, 0.84, 1.0)

    bpy.ops.object.light_add(type="POINT", location=(width * 0.46, depth * 0.20, height * 0.74))
    rim = bpy.context.object
    rim.name = "small_rim_light"
    rim.data.energy = 42
    rim.data.color = (1.0, 0.92, 0.76)

    glow = make_emissive_material("fluorescent_tube_glow", [1.0, 0.88, 0.58], 1.2)
    for x in (width * 0.30, width * 0.58):
        add_cube("emissive_fluorescent_strip", (x, depth * 0.42, height - 0.035), (1.35, 0.08, 0.018), 0.0, glow)


def object_mesh_groups(before: set) -> list:
    return [obj for obj in bpy.data.objects if obj not in before and obj.type == "MESH"]


def merge_support_planes(planes: list[float], tolerance: float) -> list[float]:
    merged: list[float] = []
    for plane in sorted(float(value) for value in planes):
        if not merged or abs(plane - merged[-1]) > tolerance:
            merged.append(plane)
        else:
            merged[-1] = (merged[-1] + plane) * 0.5
    return merged


def derive_render_support_planes(asset: dict, meshes: list, object_bounds: dict) -> list[float]:
    min_corner = Vector(tuple(object_bounds["min"]))
    max_corner = Vector(tuple(object_bounds["max"]))
    size = max_corner - min_corner
    support_kind = asset.get("support_kind", "none")
    if min(size.x, size.y, size.z) <= 0:
        return []
    if support_kind == "surface":
        return [float(max_corner.z)]
    if support_kind != "container":
        return []

    min_x_span = float(size.x * 0.45)
    min_y_span = float(size.y * 0.45)
    max_slab_height = max(0.08, float(size.z * 0.18))
    raw_planes: list[float] = []
    for mesh in meshes:
        try:
            component_min, component_max = world_bbox([mesh])
        except RuntimeError:
            continue
        component_size = component_max - component_min
        if (
            component_size.x >= min_x_span
            and component_size.y >= min_y_span
            and component_size.z <= max_slab_height
        ):
            raw_planes.append(float(component_max.z))
    if not raw_planes:
        raw_planes = derive_face_support_planes(meshes, min_corner, max_corner)
    return merge_support_planes(raw_planes, tolerance=max(0.025, float(size.z * 0.025)))


def derive_face_support_planes(meshes: list, min_corner: Vector, max_corner: Vector) -> list[float]:
    size = max_corner - min_corner
    bin_width = max(0.015, float(size.z * 0.018))
    min_area = max(0.01, float(size.x * size.y * 0.08))
    min_x_span = float(size.x * 0.35)
    min_y_span = float(size.y * 0.35)
    buckets: dict[float, dict] = {}
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for mesh_obj in meshes:
        if mesh_obj.type != "MESH":
            continue
        evaluated = mesh_obj.evaluated_get(depsgraph)
        mesh_data = evaluated.to_mesh()
        try:
            world_vertices = [evaluated.matrix_world @ vertex.co for vertex in mesh_data.vertices]
            for polygon in mesh_data.polygons:
                indices = list(polygon.vertices)
                if len(indices) < 3:
                    continue
                first = world_vertices[indices[0]]
                for offset in range(1, len(indices) - 1):
                    p0 = first
                    p1 = world_vertices[indices[offset]]
                    p2 = world_vertices[indices[offset + 1]]
                    normal = (p1 - p0).cross(p2 - p0)
                    normal_length = normal.length
                    if normal_length <= 1e-8:
                        continue
                    area = normal_length * 0.5
                    if normal.z / normal_length <= 0.85 or area <= 1e-5:
                        continue
                    centroid_z = (p0.z + p1.z + p2.z) / 3.0
                    bucket_key = round(centroid_z / bin_width) * bin_width
                    bucket = buckets.setdefault(bucket_key, {"area": 0.0, "points": []})
                    bucket["area"] += float(area)
                    bucket["points"].extend([p0, p1, p2])
        finally:
            evaluated.to_mesh_clear()

    planes: list[float] = []
    for _level, bucket in sorted(buckets.items()):
        if bucket["area"] < min_area:
            continue
        points = bucket["points"]
        span_x = max(point.x for point in points) - min(point.x for point in points)
        span_y = max(point.y for point in points) - min(point.y for point in points)
        if span_x >= min_x_span and span_y >= min_y_span:
            planes.append(float(max(point.z for point in points)))
    return planes


def support_target_for(
    obj_spec: dict,
    asset: dict,
    scene: dict,
    bounds_by_id: dict,
    support_planes_by_id: dict[str, list[float]],
) -> tuple[str, float | None]:
    tags = {str(tag).lower() for tag in asset.get("tags", [])}
    relation = obj_spec.get("relation")
    parent_id = obj_spec.get("parent_id")
    if parent_id and relation in {"on", "inside"} and parent_id in bounds_by_id:
        parent = bounds_by_id[parent_id]
        parent_asset = scene["_assets_by_object"].get(parent_id)
        planes = support_planes_by_id.get(parent_id, [])
        if planes:
            current_bottom = bounds_by_id[obj_spec["id"]]["min"][2]
            return "mesh_derived_parent_support_plane", min(planes, key=lambda value: abs(value - current_bottom))
        if parent_asset:
            if parent_asset.get("support_kind") == "container" and parent_asset.get("resolved_glb_path"):
                return "missing_mesh_support_plane", None
            heights = parent_asset.get("support_heights", [])
        else:
            heights = asset.get("support_heights", [])
        if heights:
            parent_height = parent["max"][2] - parent["min"][2]
            current_bottom = bounds_by_id[obj_spec["id"]]["min"][2]
            candidates = [parent["min"][2] + parent_height * float(height) for height in heights]
            return "registry_parent_support_plane", min(candidates, key=lambda value: abs(value - current_bottom))
        return "parent_mesh_top", parent["max"][2]
    if "ceiling" in tags and "floor" not in tags:
        return "ceiling_or_wall_mounted", None
    if "wall" in tags and asset.get("category") not in {"door"} and "floor" not in tags:
        return "ceiling_or_wall_mounted", None
    return "ground", 0.0


def build_group_bvh(meshes: list):
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    vertices: list[Vector] = []
    triangles: list[tuple[int, int, int]] = []
    triangle_centers: list[Vector] = []
    for mesh_obj in meshes:
        if mesh_obj.type != "MESH":
            continue
        evaluated = mesh_obj.evaluated_get(depsgraph)
        mesh_data = evaluated.to_mesh()
        try:
            offset = len(vertices)
            world_vertices = [evaluated.matrix_world @ vertex.co for vertex in mesh_data.vertices]
            vertices.extend(world_vertices)
            for polygon in mesh_data.polygons:
                indices = [offset + int(index) for index in polygon.vertices]
                if len(indices) < 3:
                    continue
                first = indices[0]
                for tri_index in range(1, len(indices) - 1):
                    triangle = (first, indices[tri_index], indices[tri_index + 1])
                    triangles.append(triangle)
                    triangle_centers.append((vertices[triangle[0]] + vertices[triangle[1]] + vertices[triangle[2]]) / 3.0)
        finally:
            evaluated.to_mesh_clear()
    if not vertices or not triangles:
        return None, []
    return BVHTree.FromPolygons(vertices, triangles, all_triangles=True, epsilon=0.0), triangle_centers


def bounds_overlap(left: dict, right: dict, tolerance: float = 0.0) -> bool:
    return all(
        left["min"][axis] < right["max"][axis] - tolerance
        and left["max"][axis] > right["min"][axis] + tolerance
        for axis in range(3)
    )


def support_pair(scene: dict, left_id: str, right_id: str) -> tuple[str, str] | None:
    objects = {obj["id"]: obj for obj in scene["objects"]}
    left = objects[left_id]
    right = objects[right_id]
    if left.get("parent_id") == right_id and left.get("relation") in {"on", "inside"}:
        return left_id, right_id
    if right.get("parent_id") == left_id and right.get("relation") in {"on", "inside"}:
        return right_id, left_id
    return None


def allowed_support_contact_overlap(
    scene: dict,
    child_id: str,
    parent_id: str,
    left_id: str,
    right_id: str,
    overlap_pairs: list[tuple[int, int]],
    centers_by_id: dict[str, list[Vector]],
    bounds_by_id: dict,
    support_planes_by_id: dict[str, list[float]],
    contact_band: float,
) -> bool:
    objects = {obj["id"]: obj for obj in scene["objects"]}
    child = objects[child_id]
    child_asset = scene["_assets"][child["asset_id"]]
    support_model, target_z = support_target_for(child, child_asset, scene, bounds_by_id, support_planes_by_id)
    if target_z is None or support_model == "missing_mesh_support_plane":
        return False
    for left_triangle, right_triangle in overlap_pairs:
        left_center = centers_by_id[left_id][left_triangle]
        right_center = centers_by_id[right_id][right_triangle]
        child_center = left_center if left_id == child_id else right_center
        parent_center = right_center if left_id == child_id else left_center
        child_at_contact = abs(float(child_center.z) - float(target_z)) <= contact_band
        parent_at_contact = abs(float(parent_center.z) - float(target_z)) <= contact_band
        if not (child_at_contact and parent_at_contact):
            return False
    return True


def render_collision_failures(
    scene: dict,
    object_groups: dict,
    bounds_by_id: dict,
    support_planes_by_id: dict[str, list[float]],
    contact_band: float = 0.03,
) -> list[dict]:
    bvh_by_id = {}
    centers_by_id: dict[str, list[Vector]] = {}
    for object_id, meshes in object_groups.items():
        if object_id not in bounds_by_id:
            continue
        tree, centers = build_group_bvh(meshes)
        if tree is None:
            continue
        bvh_by_id[object_id] = tree
        centers_by_id[object_id] = centers

    failures: list[dict] = []
    object_specs = {obj["id"]: obj for obj in scene["objects"]}
    for left_id, right_id in combinations(sorted(bvh_by_id), 2):
        left_asset = scene["_assets"][object_specs[left_id]["asset_id"]]
        right_asset = scene["_assets"][object_specs[right_id]["asset_id"]]
        if "floor_marking" in {left_asset.get("category"), right_asset.get("category")}:
            continue
        if not bounds_overlap(bounds_by_id[left_id], bounds_by_id[right_id], tolerance=0.0):
            continue
        overlap_pairs = bvh_by_id[left_id].overlap(bvh_by_id[right_id])
        if not overlap_pairs:
            continue
        pair = support_pair(scene, left_id, right_id)
        if pair and allowed_support_contact_overlap(
            scene,
            child_id=pair[0],
            parent_id=pair[1],
            left_id=left_id,
            right_id=right_id,
            overlap_pairs=overlap_pairs,
            centers_by_id=centers_by_id,
            bounds_by_id=bounds_by_id,
            support_planes_by_id=support_planes_by_id,
            contact_band=contact_band,
        ):
            continue
        failures.append(
            {
                "object_a": left_id,
                "object_b": right_id,
                "reason": "render_mesh_triangle_overlap",
                "overlap_triangle_pairs": len(overlap_pairs),
                "support_pair": list(pair) if pair else None,
            }
        )
    return failures


def validate_render_support(scene: dict, object_groups: dict, out_dir: Path, tolerance: float = 0.065) -> None:
    bounds_by_id: dict[str, dict] = {}
    for object_id, meshes in object_groups.items():
        if not meshes:
            continue
        min_corner, max_corner = world_bbox(meshes)
        bounds_by_id[object_id] = {
            "min": [round(min_corner.x, 6), round(min_corner.y, 6), round(min_corner.z, 6)],
            "max": [round(max_corner.x, 6), round(max_corner.y, 6), round(max_corner.z, 6)],
        }
    scene["_assets_by_object"] = {
        obj["id"]: scene["_assets"][obj["asset_id"]]
        for obj in scene["objects"]
        if obj.get("asset_id") in scene["_assets"]
    }
    support_planes_by_id: dict[str, list[float]] = {}
    for object_id, meshes in object_groups.items():
        asset = scene["_assets_by_object"].get(object_id)
        bounds = bounds_by_id.get(object_id)
        if not asset or not bounds:
            continue
        support_planes_by_id[object_id] = [
            round(float(value), 6)
            for value in derive_render_support_planes(asset, meshes, bounds)
        ]
    records = []
    failures = []
    for obj_spec in scene["objects"]:
        object_id = obj_spec["id"]
        asset = scene["_assets"][obj_spec["asset_id"]]
        bounds = bounds_by_id.get(object_id)
        if not bounds:
            failures.append({"object_id": object_id, "reason": "missing_render_mesh_group"})
            continue
        support_model, target_z = support_target_for(obj_spec, asset, scene, bounds_by_id, support_planes_by_id)
        bottom_z = bounds["min"][2]
        error = None if target_z is None else abs(bottom_z - target_z)
        status = "skipped_mounted" if target_z is None else "ok"
        if support_model == "missing_mesh_support_plane":
            status = "failed"
            failures.append(
                {
                    "object_id": object_id,
                    "support_model": support_model,
                    "reason": f"parent {obj_spec.get('parent_id')} has no mesh-derived support plane",
                }
            )
        if error is not None and error > tolerance:
            status = "failed"
            failures.append(
                {
                    "object_id": object_id,
                    "support_model": support_model,
                    "bottom_z": bottom_z,
                    "target_z": round(float(target_z), 6),
                    "error_m": round(float(error), 6),
                }
            )
        records.append(
            {
                "object_id": object_id,
                "asset_id": obj_spec["asset_id"],
                "category": asset.get("category"),
                "support_model": support_model,
                "bottom_z": bottom_z,
                "target_z": round(float(target_z), 6) if target_z is not None else None,
                "support_error_m": round(float(error), 6) if error is not None else None,
                "derived_support_planes": support_planes_by_id.get(object_id, []),
                "bounds": bounds,
                "status": status,
            }
        )
    support_failure_count = len(failures)
    collision_failures = render_collision_failures(scene, object_groups, bounds_by_id, support_planes_by_id)
    failures.extend(collision_failures)
    report = {
        "ok": not failures,
        "tolerance_m": tolerance,
        "visual_support_failure_count": support_failure_count,
        "visual_collision_failure_count": len(collision_failures),
        "failures": failures,
        "objects": records,
    }
    (out_dir / "render_validation.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        details = "; ".join(
            f"{item.get('object_id') or item.get('object_a')} error={item.get('error_m', item.get('reason', 'n/a'))}"
            for item in failures[:8]
        )
        raise RuntimeError(f"render visual validation failed: {details}")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    data = json.loads(input_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = data["scene"]
    assets = data["assets"]
    environment = data.get("environment", {})
    resolution = data.get("resolution", [1200, 900])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()

    ground_mat = make_material("textured_concrete_floor", [0.72, 0.74, 0.70], environment.get("floor_material"))
    add_procedural_detail(ground_mat, [0.58, 0.58, 0.54], noise_scale=48.0, color_variation=0.14, bump_strength=0.022)
    width, depth, _height = scene["bounds"]
    add_cube("ground", (width * 0.5, depth * 0.5, -0.025), (width, depth, 0.05), 0.0, ground_mat)
    add_warehouse_shell(width, depth, _height, environment)
    add_floor_markings(width, depth)

    materials = {}
    semantic_materials = make_semantic_materials()
    object_groups = {}
    for obj_spec in scene["objects"]:
        asset = assets[obj_spec["asset_id"]]
        material = materials.get(asset["id"])
        if material is None:
            material = make_detailed_material(asset["id"], asset.get("color", [0.6, 0.6, 0.6]), 30.0, 0.12, 0.026)
            materials[asset["id"]] = material
        before = set(bpy.data.objects)
        if asset.get("resolved_glb_path"):
            add_imported_mesh(obj_spec, asset, material, semantic_materials)
        else:
            add_procedural(obj_spec, asset, material)
        object_groups[obj_spec["id"]] = object_mesh_groups(before)
        add_render_polish(object_groups[obj_spec["id"]])

    add_studio_lighting(width, depth, _height)
    scene["_assets"] = assets
    validate_render_support(scene, object_groups, out_dir)

    bpy.ops.object.camera_add(location=(width, -depth, _height))
    camera = bpy.context.object
    select_presentation_camera(camera, presentation_camera_meshes(renderable_scene_meshes()), resolution, out_dir)
    bpy.context.scene.camera = camera

    configure_rendering()
    render_png(out_dir / "render.png", resolution)
    render_additional_views(camera, object_groups, out_dir, resolution)
    bpy.ops.export_scene.gltf(filepath=str(out_dir / "scene.glb"), export_format="GLB")
    if hasattr(bpy.ops.wm, "usd_export"):
        try:
            bpy.ops.wm.usd_export(filepath=str(out_dir / "scene.usd"))
        except Exception:
            pass


if __name__ == "__main__":
    main()
