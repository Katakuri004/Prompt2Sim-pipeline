from __future__ import annotations

import argparse
import sys
from pathlib import Path

import bpy


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


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.out)
    if not input_path.is_file():
        raise RuntimeError(f"USD input is missing: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    if not hasattr(bpy.ops.wm, "usd_import"):
        raise RuntimeError("This Blender build does not expose bpy.ops.wm.usd_import.")
    bpy.ops.wm.usd_import(filepath=str(input_path))
    mesh_objects = [obj for obj in bpy.data.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError(f"USD import produced no mesh objects: {input_path}")
    bpy.ops.export_scene.gltf(filepath=str(output_path), export_format="GLB")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"GLB export failed or produced an empty file: {output_path}")


if __name__ == "__main__":
    main()
