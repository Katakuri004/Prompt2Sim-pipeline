from __future__ import annotations

import math
import shutil
import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from scenethesis_mvp.mujoco_bridge.camera_manifest import derive_mujoco_cameras
from scenethesis_mvp.mujoco_bridge.mesh_assets import prepare_mesh_assets
from scenethesis_mvp.mujoco_bridge.mesh_names import sanitize_name
from scenethesis_mvp.mujoco_bridge.schemas import CompileResult, CollisionSpec, SceneIR, SceneIRObject
from scenethesis_mvp.utils.io import write_json


def compile_scene_to_mjcf(scene_ir: SceneIR, out_dir: str | Path, config: dict[str, Any]) -> CompileResult:
    target = Path(out_dir).resolve()
    target.mkdir(parents=True, exist_ok=True)
    scene_ir, mesh_report = prepare_mesh_assets(scene_ir, target, config)
    scene_ir_path = target / "mujoco_scene_ir.json"
    write_json(scene_ir_path, scene_ir)
    camera_manifest = derive_mujoco_cameras(scene_ir, config)
    write_json(target / "camera_manifest.json", {"cameras": camera_manifest})
    _copy_robot_asset(scene_ir, target)
    xml_path = target / "scene.xml"
    _write_mjcf(scene_ir, xml_path, config, camera_manifest)
    compile_report: dict[str, Any] = {
        "scene_id": scene_ir.scene_id,
        "xml_path": str(xml_path),
        "scene_ir_path": str(scene_ir_path),
        "mesh_report": mesh_report,
        "camera_manifest_path": str(target / "camera_manifest.json"),
        "mujoco_compile_ok": False,
        "mjb_path": None,
    }
    mjb_path: Path | None = None
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(xml_path))
        mjb_path = target / "scene.mjb"
        try:
            mujoco.mj_saveModel(model, str(mjb_path))
        except TypeError:
            mujoco.mj_saveModel(model, str(mjb_path), None)
        compile_report["mujoco_compile_ok"] = True
        compile_report["mjb_path"] = str(mjb_path)
        compile_report["nq"] = int(model.nq)
        compile_report["nv"] = int(model.nv)
        compile_report["nu"] = int(model.nu)
    except Exception as exc:
        compile_report["mujoco_compile_error"] = str(exc)
        write_json(target / "compile_report.json", compile_report)
        raise
    compile_report_path = target / "compile_report.json"
    write_json(compile_report_path, compile_report)
    return CompileResult(
        scene_ir_path=str(scene_ir_path),
        xml_path=str(xml_path),
        mjb_path=str(mjb_path) if mjb_path else None,
        mesh_dir=str(target / "meshes"),
        compile_report_path=str(compile_report_path),
        object_count=len(scene_ir.objects),
        dynamic_object_count=sum(1 for obj in scene_ir.objects if obj.mobility == "dynamic"),
        visual_only_object_count=sum(1 for obj in scene_ir.objects if obj.mobility == "visual_only"),
    )


def _copy_robot_asset(scene_ir: SceneIR, target: Path) -> None:
    robot_source = Path(scene_ir.robot.mjcf_path)
    if not robot_source.is_file():
        return
    robot_dir = target / "robots" / scene_ir.robot.id
    if robot_source.parent.is_dir():
        shutil.copytree(robot_source.parent, robot_dir, dirs_exist_ok=True)
    else:
        robot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(robot_source, robot_dir / robot_source.name)


def _write_mjcf(scene_ir: SceneIR, xml_path: Path, config: dict[str, Any], camera_manifest: list[dict[str, Any]]) -> None:
    root = ET.Element("mujoco", {"model": sanitize_name(scene_ir.scene_id)})
    ET.SubElement(root, "compiler", {"angle": "radian", "meshdir": "meshes", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {
            "timestep": str(config.get("rollout", {}).get("timestep", 0.002)),
            "gravity": "0 0 -9.81",
            "integrator": "implicitfast",
        },
    )
    visual = ET.SubElement(root, "visual")
    offwidth, offheight = _offscreen_size(scene_ir, config)
    ET.SubElement(visual, "global", {"offwidth": str(offwidth), "offheight": str(offheight)})
    ET.SubElement(visual, "headlight", {"ambient": "0.34 0.34 0.34", "diffuse": "0.74 0.74 0.70", "specular": "0.18 0.18 0.18"})
    asset = ET.SubElement(root, "asset")
    _add_materials(asset, scene_ir)
    _add_mesh_assets(asset, scene_ir)
    worldbody = ET.SubElement(root, "worldbody")
    _add_world(worldbody, scene_ir)
    _add_static_visual_scene(worldbody, scene_ir)
    _add_robot(worldbody, root, asset, scene_ir)
    for obj in scene_ir.objects:
        _add_object_body(worldbody, obj)
    _add_task_sites(worldbody, scene_ir)
    _add_cameras(worldbody, scene_ir, camera_manifest)
    _add_sensors(root, scene_ir)
    _add_contacts(root, scene_ir)
    ET.indent(root, space="  ")
    tree = ET.ElementTree(root)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)


def _add_materials(asset: ET.Element, scene_ir: SceneIR) -> None:
    ET.SubElement(asset, "material", {"name": "mat_static", "rgba": "0.55 0.56 0.54 1"})
    ET.SubElement(asset, "material", {"name": "mat_dynamic", "rgba": "0.74 0.49 0.25 1"})
    ET.SubElement(asset, "material", {"name": "mat_robot", "rgba": "0.86 0.86 0.82 1"})
    ET.SubElement(asset, "material", {"name": "mat_target", "rgba": "0.25 0.55 0.95 1"})
    ET.SubElement(asset, "material", {"name": "mat_invisible_collision", "rgba": "0 0 0 0"})
    for material in scene_ir.visual_materials:
        ET.SubElement(asset, "material", {"name": material.name, "rgba": _vec(material.rgba)})


def _add_mesh_assets(asset: ET.Element, scene_ir: SceneIR) -> None:
    for part in scene_ir.static_visual_meshes:
        ET.SubElement(asset, "mesh", {"name": part.mesh_name, "file": part.file, "inertia": "shell"})
    for obj in scene_ir.objects:
        if obj.visual_mesh:
            ET.SubElement(asset, "mesh", {"name": obj.visual_mesh, "file": f"{obj.visual_mesh}.obj"})
        for part in obj.visual_parts:
            ET.SubElement(asset, "mesh", {"name": part.mesh_name, "file": part.file, "inertia": "shell"})
        for spec in obj.collision:
            if spec.kind == "mesh" and spec.mesh_name:
                ET.SubElement(asset, "mesh", {"name": spec.mesh_name, "file": f"{spec.mesh_name}.obj"})


def _offscreen_size(scene_ir: SceneIR, config: dict[str, Any]) -> tuple[int, int]:
    viz = config.get("visualization", {})
    width, height = (int(item) for item in viz.get("resolution", [640, 480]))
    for camera in scene_ir.policy.observation_cameras:
        width = max(width, int(camera.resolution[0]))
        height = max(height, int(camera.resolution[1]))
    return width, height


def _add_world(worldbody: ET.Element, scene_ir: SceneIR) -> None:
    width, depth, height = scene_ir.bounds
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "policy_key_light",
            "pos": f"{width * 0.5} {depth * 0.45} {height + 1.2}",
            "dir": "0 0 -1",
            "diffuse": "0.9 0.9 0.85",
            "specular": "0.25 0.25 0.25",
            "ambient": "0.22 0.22 0.22",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "name": "policy_fill_light",
            "pos": f"{width * 0.78} {depth * 0.18} {height + 0.8}",
            "dir": "-0.4 0.4 -1",
            "diffuse": "0.55 0.55 0.52",
            "specular": "0.12 0.12 0.12",
            "ambient": "0.12 0.12 0.12",
        },
    )
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 0",
            "size": f"{max(width, depth)} {max(width, depth)} 0.1",
            "friction": "0.9 0.02 0.001",
            "material": "mat_invisible_collision" if scene_ir.static_visual_meshes else "mat_static",
            "group": "2" if scene_ir.static_visual_meshes else "0",
        },
    )
    wall_specs = [
        ("wall_x_min", [0.0, depth * 0.5, height * 0.5], [0.03, depth * 0.5, height * 0.5]),
        ("wall_x_max", [width, depth * 0.5, height * 0.5], [0.03, depth * 0.5, height * 0.5]),
        ("wall_y_min", [width * 0.5, 0.0, height * 0.5], [width * 0.5, 0.03, height * 0.5]),
        ("wall_y_max", [width * 0.5, depth, height * 0.5], [width * 0.5, 0.03, height * 0.5]),
    ]
    for name, pos, size in wall_specs:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": name,
                "type": "box",
                "pos": _vec(pos),
                "size": _vec(size),
                "contype": "1",
                "conaffinity": "1",
                "material": "mat_invisible_collision" if scene_ir.static_visual_meshes else "mat_static",
                "group": "2",
            },
        )


def _add_static_visual_scene(worldbody: ET.Element, scene_ir: SceneIR) -> None:
    for part in scene_ir.static_visual_meshes:
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"{part.mesh_name}_visual",
                "type": "mesh",
                "mesh": part.mesh_name,
                "contype": "0",
                "conaffinity": "0",
                "group": str(part.group),
                "density": "0",
                "material": part.material,
            },
        )


def _add_robot(worldbody: ET.Element, root: ET.Element, asset: ET.Element, scene_ir: SceneIR) -> None:
    robot = scene_ir.robot
    robot_source = Path(robot.mjcf_path)
    if robot_source.is_file():
        _merge_robot_mjcf(worldbody, root, asset, scene_ir, robot_source)
        return
    base = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "panda_base",
            "pos": _vec(robot.base_position),
            "euler": f"0 0 {robot.base_yaw_rad}",
        },
    )
    ET.SubElement(base, "geom", {"name": "panda_base_geom", "type": "cylinder", "size": "0.18 0.08", "pos": "0 0 0.04", "material": "mat_robot"})
    link1 = _robot_body(base, "panda_link1", "0 0 0.12", "panda_joint1", "0 0 1", "-2.8973 2.8973", "0 0 0 0 0 0.25", "0.045")
    link2 = _robot_body(link1, "panda_link2", "0 0 0.25", "panda_joint2", "0 1 0", "-1.7628 1.7628", "0 0 0 0.23 0 0.08", "0.04")
    link3 = _robot_body(link2, "panda_link3", "0.23 0 0.08", "panda_joint3", "0 0 1", "-2.8973 2.8973", "0 0 0 0.24 0 0", "0.030")
    link4 = _robot_body(link3, "panda_link4", "0.24 0 0", "panda_joint4", "0 1 0", "-3.0718 0.0698", "0 0 0 0.16 0 -0.18", "0.022")
    link5 = _robot_body(link4, "panda_link5", "0.16 0 -0.18", "panda_joint5", "0 0 1", "-2.8973 2.8973", "0 0 0 0.14 0 0", "0.024")
    link6 = _robot_body(link5, "panda_link6", "0.14 0 0", "panda_joint6", "0 1 0", "-0.0175 3.7525", "0 0 0 0.12 0 0.08", "0.022")
    hand = ET.SubElement(link6, "body", {"name": "panda_hand", "pos": "0.12 0 0.08"})
    ET.SubElement(hand, "joint", {"name": "panda_joint7", "type": "hinge", "axis": "0 0 1", "range": "-2.8973 2.8973", "damping": "1.0"})
    ET.SubElement(hand, "geom", {"name": "panda_hand_geom", "type": "box", "size": "0.06 0.045 0.035", "material": "mat_robot"})
    ET.SubElement(hand, "site", {"name": robot.ee_site, "pos": "0.12 0 0", "size": "0.025", "rgba": "0 0.7 0.2 1"})
    ET.SubElement(hand, "camera", {"name": "wrist_rgb", "pos": "0.06 0 0.055", "euler": "0 1.570796 0", "fovy": "58"})
    left = ET.SubElement(hand, "body", {"name": "panda_leftfinger", "pos": "0.09 0.035 0"})
    ET.SubElement(left, "joint", {"name": "panda_finger_joint1", "type": "slide", "axis": "0 1 0", "range": "0 0.04", "damping": "0.2"})
    finger_geom = {
        "type": "box",
        "size": "0.045 0.008 0.025",
        "rgba": "0.06 0.06 0.06 1",
        "friction": "3.0 0.12 0.01",
        "condim": "4",
    }
    ET.SubElement(left, "geom", {"name": "panda_leftfinger_geom", **finger_geom})
    right = ET.SubElement(hand, "body", {"name": "panda_rightfinger", "pos": "0.09 -0.035 0"})
    ET.SubElement(right, "joint", {"name": "panda_finger_joint2", "type": "slide", "axis": "0 -1 0", "range": "0 0.04", "damping": "0.2"})
    ET.SubElement(right, "geom", {"name": "panda_rightfinger_geom", **finger_geom})
    actuator = ET.SubElement(root, "actuator")
    ranges = {
        "panda_joint1": "-2.8973 2.8973",
        "panda_joint2": "-1.7628 1.7628",
        "panda_joint3": "-2.8973 2.8973",
        "panda_joint4": "-3.0718 0.0698",
        "panda_joint5": "-2.8973 2.8973",
        "panda_joint6": "-0.0175 3.7525",
        "panda_joint7": "-2.8973 2.8973",
        "panda_finger_joint1": "0 0.04",
        "panda_finger_joint2": "0 0.04",
    }
    for name in robot.actuator_names:
        joint = name.removesuffix("_act")
        kp = "220" if "finger" in joint else "120"
        ET.SubElement(actuator, "position", {"name": name, "joint": joint, "kp": kp, "ctrlrange": ranges[joint]})


def _merge_robot_mjcf(worldbody: ET.Element, root: ET.Element, asset: ET.Element, scene_ir: SceneIR, robot_source: Path) -> None:
    robot_tree = ET.parse(robot_source)
    robot_root = robot_tree.getroot()
    asset_index = list(root).index(asset)
    for default in robot_root.findall("default"):
        root.insert(asset_index, copy.deepcopy(default))
        asset_index += 1

    robot_asset = robot_root.find("asset")
    if robot_asset is not None:
        for child in robot_asset:
            copied = copy.deepcopy(child)
            if copied.tag == "mesh" and copied.get("file"):
                copied.set("file", f"../robots/{scene_ir.robot.id}/assets/{copied.get('file')}")
            asset.append(copied)

    base = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "panda_base",
            "pos": _vec(scene_ir.robot.base_position),
            "euler": f"0 0 {scene_ir.robot.base_yaw_rad}",
        },
    )
    robot_world = robot_root.find("worldbody")
    if robot_world is None:
        raise RuntimeError(f"Robot MJCF is missing worldbody: {robot_source}")
    copied_robot_body = False
    for child in robot_world:
        if child.tag == "body":
            base.append(copy.deepcopy(child))
            copied_robot_body = True
    if not copied_robot_body:
        raise RuntimeError(f"Robot MJCF has no root body to merge: {robot_source}")

    for section_name in ("tendon", "equality", "actuator", "keyframe"):
        section = robot_root.find(section_name)
        if section is not None:
            root.append(copy.deepcopy(section))

    robot_contact = robot_root.find("contact")
    if robot_contact is not None:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        for child in robot_contact:
            contact.append(copy.deepcopy(child))


def _robot_body(parent: ET.Element, body_name: str, pos: str, joint_name: str, axis: str, joint_range: str, fromto: str, size: str) -> ET.Element:
    body = ET.SubElement(parent, "body", {"name": body_name, "pos": pos})
    ET.SubElement(body, "joint", {"name": joint_name, "type": "hinge", "axis": axis, "range": joint_range, "damping": "1.0"})
    ET.SubElement(body, "geom", {"name": f"{body_name}_geom", "type": "capsule", "fromto": fromto, "size": size, "material": "mat_robot"})
    return body


def _add_object_body(worldbody: ET.Element, obj: SceneIRObject) -> None:
    attrs = {
        "name": obj.id,
        "pos": _vec(obj.pose.position),
        "quat": _vec(obj.pose.quaternion),
    }
    body = ET.SubElement(worldbody, "body", attrs)
    if obj.mobility == "dynamic":
        ET.SubElement(body, "freejoint", {"name": f"{obj.id}_freejoint"})
        ET.SubElement(body, "inertial", {"pos": "0 0 0", "mass": str(obj.physics.mass_kg or 1.0), "diaginertia": _box_inertia(obj.dimensions, obj.physics.mass_kg or 1.0)})
    if obj.visual_parts:
        for part_index, part in enumerate(obj.visual_parts):
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{obj.id}_visual_{part_index}",
                    "type": "mesh",
                    "mesh": part.mesh_name,
                    "contype": "0",
                    "conaffinity": "0",
                    "group": str(part.group),
                    "density": "0",
                    "material": part.material,
                },
            )
    elif obj.visual_mesh:
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{obj.id}_visual",
                "type": "mesh",
                "mesh": obj.visual_mesh,
                "contype": "0",
                "conaffinity": "0",
                "group": "1",
                "density": "0",
                "material": "mat_target" if obj.mobility == "dynamic" else "mat_static",
            },
        )
    for index, spec in enumerate(obj.collision):
        _add_collision_geom(body, obj, spec, index)


def _add_collision_geom(body: ET.Element, obj: SceneIRObject, spec: CollisionSpec, index: int) -> None:
    if spec.kind == "visual_only":
        return
    common = {
        "name": f"{obj.id}_collision_{index}",
        "contype": "1",
        "conaffinity": "1",
        "group": str(spec.group),
        "friction": _vec(obj.physics.friction),
    }
    if obj.mobility == "dynamic":
        common["condim"] = "4"
    if obj.visual_parts:
        common["material"] = "mat_invisible_collision"
    if spec.kind == "mesh" and spec.mesh_name:
        ET.SubElement(body, "geom", {**common, "type": "mesh", "mesh": spec.mesh_name})
        return
    if spec.kind == "primitive":
        attrs = {**common, "type": spec.primitive_type or "box", "pos": _vec(spec.pos)}
        if spec.primitive_type == "cylinder":
            size = spec.size or [0.1, 0.1]
            attrs["size"] = f"{size[0]} {size[-1]}"
        else:
            attrs["size"] = _vec(spec.size or [0.1, 0.1, 0.1])
        ET.SubElement(body, "geom", attrs)


def _add_task_sites(worldbody: ET.Element, scene_ir: SceneIR) -> None:
    ET.SubElement(
        worldbody,
        "site",
        {
            "name": scene_ir.task.destination_region,
            "pos": _vec(scene_ir.task.destination_position),
            "size": _vec(scene_ir.task.destination_size),
            "rgba": "0.1 0.8 0.1 0.35",
            "type": "box",
        },
    )


def _add_cameras(worldbody: ET.Element, scene_ir: SceneIR, camera_manifest: list[dict[str, Any]]) -> None:
    existing: set[str] = set()
    for camera in camera_manifest:
        name = str(camera["camera_name"])
        if name == "wrist_rgb" or name in existing:
            continue
        if camera.get("xyaxes") is None or camera.get("pos") is None:
            continue
        existing.add(name)
        ET.SubElement(
            worldbody,
            "camera",
            {
                "name": name,
                "pos": _vec(camera["pos"]),
                "xyaxes": _vec(camera["xyaxes"]),
                "fovy": str(camera["fov_deg"]),
            },
        )


def _add_sensors(root: ET.Element, scene_ir: SceneIR) -> None:
    sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "framepos", {"name": "ee_pos", "objtype": "site", "objname": scene_ir.robot.ee_site})
    ET.SubElement(sensor, "framequat", {"name": "ee_quat", "objtype": "site", "objname": scene_ir.robot.ee_site})


def _add_contacts(root: ET.Element, scene_ir: SceneIR) -> None:
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    real_panda_names = scene_ir.robot.arm_joint_names and scene_ir.robot.arm_joint_names[0] == "joint1"
    adjacent_pairs = (
        [
            ("panda_base", "link0"),
            ("link1", "link2"),
            ("link2", "link3"),
            ("link3", "link4"),
            ("link4", "link5"),
            ("link5", "link6"),
            ("link6", "link7"),
            ("link7", "hand"),
        ]
        if real_panda_names
        else [
            ("panda_base", "panda_link1"),
            ("panda_link1", "panda_link2"),
            ("panda_link2", "panda_link3"),
            ("panda_link3", "panda_link4"),
            ("panda_link4", "panda_link5"),
            ("panda_link5", "panda_link6"),
            ("panda_link6", "panda_hand"),
        ]
    )
    for body1, body2 in adjacent_pairs:
        ET.SubElement(contact, "exclude", {"body1": body1, "body2": body2})
    if real_panda_names:
        for prefix in ("left_finger", "right_finger"):
            ET.SubElement(contact, "exclude", {"body1": prefix, "body2": "hand"})
    else:
        for prefix in ("panda_leftfinger", "panda_rightfinger"):
            ET.SubElement(contact, "exclude", {"body1": prefix, "body2": "panda_hand"})


def _box_inertia(dimensions: list[float], mass: float) -> str:
    dx, dy, dz = dimensions
    ixx = mass * (dy * dy + dz * dz) / 12.0
    iyy = mass * (dx * dx + dz * dz) / 12.0
    izz = mass * (dx * dx + dy * dy) / 12.0
    return f"{max(ixx, 1e-6):.8f} {max(iyy, 1e-6):.8f} {max(izz, 1e-6):.8f}"


def _vec(values: list[float] | tuple[float, ...]) -> str:
    return " ".join(f"{float(item):.8g}" for item in values)


def yaw_to_quat(yaw_rad: float) -> list[float]:
    half = yaw_rad * 0.5
    return [math.cos(half), 0.0, 0.0, math.sin(half)]
