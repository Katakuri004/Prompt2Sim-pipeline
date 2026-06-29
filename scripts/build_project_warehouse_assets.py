from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial
from trimesh.visual.texture import TextureVisuals

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.utils.io import write_json


BARRIER_ASSET_ID = "authored_vertical_slot_barrier_01"
BARRIER_DIMENSIONS_M = [1.25, 0.10, 1.05]
TAPE_ASSET_ID = "authored_hazard_floor_marking_01"
TAPE_DIMENSIONS_M = [2.20, 1.40, 0.015]
BOX_ASSET_ID = "authored_clean_cardboard_box_01"
BOX_DIMENSIONS_M = [0.55, 0.42, 0.38]
TABLE_ASSET_ID = "authored_wood_metal_packing_table_01"
TABLE_DIMENSIONS_M = [1.60, 0.75, 0.90]
CRATE_ASSET_ID = "authored_x_braced_wooden_crate_01"
CRATE_DIMENSIONS_M = [0.65, 0.55, 0.62]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build reproducible project-authored warehouse assets.")
    parser.add_argument("--out-dir", default="assets/library/project_authored")
    parser.add_argument("--metadata", default="assets/manifests/project_authored_warehouse_assets.json")
    args = parser.parse_args()

    output_dir = resolve(args.out_dir)
    metadata_path = resolve(args.metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    barrier_output = output_dir / f"{BARRIER_ASSET_ID}.glb"
    tape_output = output_dir / f"{TAPE_ASSET_ID}.glb"
    box_output = output_dir / f"{BOX_ASSET_ID}.glb"
    table_output = output_dir / f"{TABLE_ASSET_ID}.glb"
    crate_output = output_dir / f"{CRATE_ASSET_ID}.glb"
    build_vertical_slot_barrier(barrier_output)
    build_hazard_floor_marking(tape_output)
    build_clean_cardboard_box(box_output)
    build_wood_metal_packing_table(table_output)
    build_x_braced_wooden_crate(crate_output)

    barrier_dimensions = validate_output(barrier_output, BARRIER_DIMENSIONS_M)
    tape_dimensions = validate_output(tape_output, TAPE_DIMENSIONS_M)
    box_dimensions = validate_output(box_output, BOX_DIMENSIONS_M)
    table_dimensions = validate_output(table_output, TABLE_DIMENSIONS_M)
    crate_dimensions = validate_output(crate_output, CRATE_DIMENSIONS_M)
    barrier = trimesh.load(barrier_output, force="scene")
    if len(barrier.geometry) != 7:
        raise RuntimeError(f"Authored barrier topology is invalid: expected 7 frame members, found {len(barrier.geometry)}")

    write_json(
        metadata_path,
        {
            "assets": [
                {
                    "id": BARRIER_ASSET_ID,
                    "output_glb": str(barrier_output.relative_to(ROOT)),
                    "output_sha256": sha256(barrier_output),
                    "dimensions_m": barrier_dimensions,
                    "construction": (
                        "Seven frame members: two full-height side rails, top and bottom bands, and three "
                        "mullions forming four open vertical slots."
                    ),
                    "material": "bright orange rough plastic/painted metal PBR material",
                    "generator": "scripts/build_project_warehouse_assets.py",
                    "license": "CC0-1.0",
                    "attribution": "Prompt2Sim project-authored asset",
                },
                {
                    "id": TAPE_ASSET_ID,
                    "output_glb": str(tape_output.relative_to(ROOT)),
                    "output_sha256": sha256(tape_output),
                    "dimensions_m": tape_dimensions,
                    "construction": "Two joined floor strips forming an L-shaped aisle boundary with raised diagonal black hazard marks.",
                    "material": "yellow base tape with alternating matte black diagonal marks",
                    "generator": "scripts/build_project_warehouse_assets.py",
                    "license": "CC0-1.0",
                    "attribution": "Prompt2Sim project-authored asset",
                },
                {
                    "id": BOX_ASSET_ID,
                    "output_glb": str(box_output.relative_to(ROOT)),
                    "output_sha256": sha256(box_output),
                    "dimensions_m": box_dimensions,
                    "construction": "Clean closed corrugated shipping carton with intact faces and packing tape across the top and front.",
                    "material": "kraft cardboard with tan packing tape",
                    "generator": "scripts/build_project_warehouse_assets.py",
                    "license": "CC0-1.0",
                    "attribution": "Prompt2Sim project-authored asset",
                },
                {
                    "id": TABLE_ASSET_ID,
                    "output_glb": str(table_output.relative_to(ROOT)),
                    "output_sha256": sha256(table_output),
                    "dimensions_m": table_dimensions,
                    "construction": "Light wood rectangular top, gray metal perimeter apron, and four square legs with a fully open underside.",
                    "material": "light sealed wood tabletop and rough gray painted metal frame",
                    "generator": "scripts/build_project_warehouse_assets.py",
                    "license": "CC0-1.0",
                    "attribution": "Prompt2Sim project-authored asset",
                },
                {
                    "id": CRATE_ASSET_ID,
                    "output_glb": str(crate_output.relative_to(ROOT)),
                    "output_sha256": sha256(crate_output),
                    "dimensions_m": crate_dimensions,
                    "construction": "Closed wooden shipping crate with solid panels, perimeter frame rails, and crossed diagonal front braces.",
                    "material": "light natural wood panels with darker structural wood rails",
                    "generator": "scripts/build_project_warehouse_assets.py",
                    "license": "CC0-1.0",
                    "attribution": "Prompt2Sim project-authored asset",
                },
            ]
        },
    )
    print(f"authored asset: {barrier_output}")
    print(f"authored asset: {tape_output}")
    print(f"authored asset: {box_output}")
    print(f"authored asset: {table_output}")
    print(f"authored asset: {crate_output}")
    print(f"metadata: {metadata_path}")
    print(
        "dimensions: "
        f"barrier={barrier_dimensions}, tape={tape_dimensions}, box={box_dimensions}, "
        f"table={table_dimensions}, crate={crate_dimensions}"
    )


def build_vertical_slot_barrier(output: Path) -> None:
    width, depth, height = BARRIER_DIMENSIONS_M
    side_width = 0.09
    top_height = 0.10
    bottom_height = 0.20
    mullion_width = 0.06
    material = PBRMaterial(
        name="SafetyOrange",
        baseColorFactor=np.asarray([1.0, 0.22, 0.015, 1.0]),
        metallicFactor=0.02,
        roughnessFactor=0.34,
    )
    scene = trimesh.Scene()

    add_member(scene, material, "left_rail", (side_width, depth, height), (-(width - side_width) / 2.0, 0.0, height / 2.0))
    add_member(scene, material, "right_rail", (side_width, depth, height), ((width - side_width) / 2.0, 0.0, height / 2.0))
    add_member(scene, material, "top_band", (width - 2.0 * side_width, depth, top_height), (0.0, 0.0, height - top_height / 2.0))
    add_member(scene, material, "bottom_band", (width - 2.0 * side_width, depth, bottom_height), (0.0, 0.0, bottom_height / 2.0))

    opening_height = height - top_height - bottom_height
    inner_width = width - 2.0 * side_width
    opening_width = (inner_width - 3.0 * mullion_width) / 4.0
    first_mullion_x = -inner_width / 2.0 + opening_width + mullion_width / 2.0
    for index in range(3):
        x = first_mullion_x + index * (opening_width + mullion_width)
        add_member(
            scene,
            material,
            f"mullion_{index + 1}",
            (mullion_width, depth, opening_height),
            (x, 0.0, bottom_height + opening_height / 2.0),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output)


def build_hazard_floor_marking(output: Path) -> None:
    width, depth, height = TAPE_DIMENSIONS_M
    strip_width = 0.12
    base_height = 0.012
    mark_height = height - base_height
    yellow = PBRMaterial(
        name="HazardYellow",
        baseColorFactor=np.asarray([1.0, 0.66, 0.01, 1.0]),
        metallicFactor=0.0,
        roughnessFactor=0.42,
    )
    black = PBRMaterial(
        name="HazardBlack",
        baseColorFactor=np.asarray([0.012, 0.012, 0.012, 1.0]),
        metallicFactor=0.0,
        roughnessFactor=0.48,
    )
    scene = trimesh.Scene()
    horizontal_y = -(depth - strip_width) / 2.0
    vertical_x = (width - strip_width) / 2.0
    add_member(scene, yellow, "yellow_horizontal", (width, strip_width, base_height), (0.0, horizontal_y, base_height / 2.0))
    add_member(scene, yellow, "yellow_vertical", (strip_width, depth, base_height), (vertical_x, 0.0, base_height / 2.0))

    mark_z = base_height + mark_height / 2.0
    for index, x in enumerate(np.arange(-width / 2.0 + 0.14, width / 2.0 - 0.08, 0.24)):
        add_member(
            scene,
            black,
            f"horizontal_mark_{index:02d}",
            (0.13, 0.035, mark_height),
            (float(x), horizontal_y, mark_z),
            yaw_deg=45.0,
        )
    for index, y in enumerate(np.arange(-depth / 2.0 + 0.14, depth / 2.0 - 0.08, 0.24)):
        add_member(
            scene,
            black,
            f"vertical_mark_{index:02d}",
            (0.13, 0.035, mark_height),
            (vertical_x, float(y), mark_z),
            yaw_deg=135.0,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output)


def build_clean_cardboard_box(output: Path) -> None:
    width, depth, height = BOX_DIMENSIONS_M
    cardboard = PBRMaterial(
        name="KraftCardboard",
        baseColorFactor=np.asarray([0.57, 0.29, 0.085, 1.0]),
        metallicFactor=0.0,
        roughnessFactor=0.72,
    )
    tape = PBRMaterial(
        name="TanPackingTape",
        baseColorFactor=np.asarray([0.78, 0.57, 0.29, 1.0]),
        metallicFactor=0.0,
        roughnessFactor=0.38,
    )
    scene = trimesh.Scene()
    tape_thickness = 0.003
    body_depth = depth - 2.0 * tape_thickness
    body_height = height - tape_thickness
    add_member(
        scene,
        cardboard,
        "closed_carton_body",
        (width, body_depth, body_height),
        (0.0, 0.0, body_height / 2.0),
    )
    add_member(
        scene,
        tape,
        "top_packing_tape",
        (0.075, depth, tape_thickness),
        (0.0, 0.0, height - tape_thickness / 2.0),
    )
    add_member(
        scene,
        tape,
        "front_packing_tape",
        (0.075, tape_thickness, body_height),
        (0.0, -(depth - tape_thickness) / 2.0, body_height / 2.0),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output)


def build_wood_metal_packing_table(output: Path) -> None:
    width, depth, height = TABLE_DIMENSIONS_M
    top_thickness = 0.075
    apron_height = 0.10
    apron_thickness = 0.045
    leg_width = 0.065
    wood = PBRMaterial(
        name="LightSealedWood",
        baseColorFactor=np.asarray([0.67, 0.44, 0.21, 1.0]),
        metallicFactor=0.0,
        roughnessFactor=0.46,
    )
    metal = PBRMaterial(
        name="GrayPaintedMetal",
        baseColorFactor=np.asarray([0.42, 0.45, 0.47, 1.0]),
        metallicFactor=0.58,
        roughnessFactor=0.34,
    )
    scene = trimesh.Scene()
    add_member(
        scene,
        wood,
        "wood_tabletop",
        (width, depth, top_thickness),
        (0.0, 0.0, height - top_thickness / 2.0),
    )
    apron_z = height - top_thickness - apron_height / 2.0
    add_member(scene, metal, "front_apron", (width - 0.08, apron_thickness, apron_height), (0.0, -(depth - apron_thickness) / 2.0, apron_z))
    add_member(scene, metal, "rear_apron", (width - 0.08, apron_thickness, apron_height), (0.0, (depth - apron_thickness) / 2.0, apron_z))
    add_member(scene, metal, "left_apron", (apron_thickness, depth - 0.08, apron_height), (-(width - apron_thickness) / 2.0, 0.0, apron_z))
    add_member(scene, metal, "right_apron", (apron_thickness, depth - 0.08, apron_height), ((width - apron_thickness) / 2.0, 0.0, apron_z))

    leg_height = height - top_thickness
    leg_x = (width - 0.16) / 2.0
    leg_y = (depth - 0.16) / 2.0
    for name, x, y in (
        ("front_left_leg", -leg_x, -leg_y),
        ("front_right_leg", leg_x, -leg_y),
        ("rear_left_leg", -leg_x, leg_y),
        ("rear_right_leg", leg_x, leg_y),
    ):
        add_member(scene, metal, name, (leg_width, leg_width, leg_height), (x, y, leg_height / 2.0))
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output)


def build_x_braced_wooden_crate(output: Path) -> None:
    width, depth, height = CRATE_DIMENSIONS_M
    panel = PBRMaterial(
        name="LightCrateWood",
        baseColorFactor=np.asarray([0.60, 0.34, 0.12, 1.0]),
        metallicFactor=0.0,
        roughnessFactor=0.66,
    )
    frame = PBRMaterial(
        name="DarkCrateFrameWood",
        baseColorFactor=np.asarray([0.40, 0.19, 0.055, 1.0]),
        metallicFactor=0.0,
        roughnessFactor=0.71,
    )
    scene = trimesh.Scene()
    add_member(scene, panel, "closed_crate_body", (0.57, 0.51, 0.54), (0.0, 0.0, height / 2.0))
    add_member(
        scene,
        panel,
        "closed_crate_back",
        (0.57, 0.02, 0.54),
        (0.0, -(depth - 0.02) / 2.0, height / 2.0),
    )

    frame_width = 0.055
    frame_depth = 0.02
    front_y = (depth - frame_depth) / 2.0
    rail_x = (width - frame_width) / 2.0
    for name, x in (("front_left_rail", -rail_x), ("front_right_rail", rail_x)):
        add_member(scene, frame, name, (frame_width, frame_depth, height), (x, front_y, height / 2.0))
    for name, z in (("front_bottom_rail", frame_width / 2.0), ("front_top_rail", height - frame_width / 2.0)):
        add_member(scene, frame, name, (width - 2.0 * frame_width, frame_depth, frame_width), (0.0, front_y, z))

    inner_width = width - 2.0 * frame_width
    inner_height = height - 2.0 * frame_width
    brace_length = float(np.hypot(inner_width, inner_height))
    brace_angle = float(np.degrees(np.arctan2(inner_height, inner_width)))
    add_member(
        scene,
        frame,
        "front_brace_rising",
        (brace_length, frame_depth, 0.035),
        (0.0, front_y, height / 2.0),
        tilt_deg=brace_angle,
    )
    add_member(
        scene,
        frame,
        "front_brace_falling",
        (brace_length, frame_depth, 0.035),
        (0.0, front_y, height / 2.0),
        tilt_deg=-brace_angle,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    scene.export(output)


def add_member(
    scene: trimesh.Scene,
    material: PBRMaterial,
    name: str,
    extents: tuple[float, float, float],
    center: tuple[float, float, float],
    yaw_deg: float = 0.0,
    tilt_deg: float = 0.0,
) -> None:
    intended_x, intended_y, intended_z = extents
    center_x, center_y, center_z = center
    mesh = trimesh.creation.box(extents=(intended_x, intended_z, intended_y))
    if yaw_deg:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(-yaw_deg), [0.0, 1.0, 0.0]))
    if tilt_deg:
        mesh.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(tilt_deg), [0.0, 0.0, 1.0]))
    mesh.apply_translation((center_x, center_z, center_y))
    mesh.visual = TextureVisuals(material=material)
    scene.add_geometry(mesh, geom_name=name, node_name=name)


def validate_output(output: Path, expected_dimensions: list[float]) -> list[float]:
    loaded = trimesh.load(output, force="scene")
    source_extents = [float(value) for value in loaded.extents]
    dimensions = [round(source_extents[0], 4), round(source_extents[2], 4), round(source_extents[1], 4)]
    if any(abs(actual - expected) > 0.001 for actual, expected in zip(dimensions, expected_dimensions)):
        raise RuntimeError(f"Authored asset dimensions are invalid: {dimensions} != {expected_dimensions}")
    return dimensions


def resolve(raw_path: str) -> Path:
    path = Path(raw_path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
