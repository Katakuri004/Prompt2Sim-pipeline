from __future__ import annotations

import argparse
import hashlib
import time
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://api.polyhaven.com"
USER_AGENT = "scenethesis-mvp-local-importer/0.1"


WAREHOUSE_ASSETS: list[dict[str, Any]] = [
    {
        "id": "real_warehouse_shelf_01",
        "polyhaven_id": "steel_frame_shelves_01",
        "category": "shelf",
        "name": "steel frame warehouse shelves",
        "dimensions": [1.65, 0.55, 1.95],
        "support_kind": "container",
        "support_heights": [0.25, 0.5, 0.75],
        "tags": ["warehouse", "storage", "rack", "shelf", "parent", "support", "anchor"],
        "color": [0.42, 0.42, 0.4],
    },
    {
        "id": "real_warehouse_shelf_02",
        "polyhaven_id": "steel_frame_shelves_02",
        "category": "shelf",
        "name": "second steel warehouse shelf",
        "dimensions": [1.45, 0.55, 1.7],
        "support_kind": "container",
        "support_heights": [0.25, 0.5, 0.75],
        "tags": ["warehouse", "storage", "rack", "shelf", "parent", "support"],
        "color": [0.42, 0.42, 0.4],
    },
    {
        "id": "real_cardboard_box_01",
        "polyhaven_id": "cardboard_box_01",
        "category": "box",
        "name": "cardboard shipping box",
        "dimensions": [0.46, 0.36, 0.34],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "cardboard", "box", "crate", "child", "storage"],
        "color": [0.68, 0.49, 0.28],
    },
    {
        "id": "real_plastic_crate_01",
        "polyhaven_id": "plastic_crate_02",
        "category": "box",
        "name": "plastic warehouse crate",
        "dimensions": [0.56, 0.4, 0.32],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "plastic", "crate", "box", "child", "storage"],
        "color": [0.15, 0.42, 0.75],
    },
    {
        "id": "real_plastic_crate_02",
        "polyhaven_id": "plastic_crate_01",
        "category": "box",
        "name": "stackable plastic crate",
        "dimensions": [0.52, 0.38, 0.30],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "plastic", "crate", "box", "child", "storage"],
        "color": [0.12, 0.36, 0.72],
    },
    {
        "id": "real_wooden_crate_01",
        "polyhaven_id": "wooden_crate_01",
        "category": "box",
        "name": "wooden warehouse crate",
        "dimensions": [0.58, 0.42, 0.38],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "wooden", "crate", "box", "child", "storage"],
        "color": [0.48, 0.31, 0.18],
    },
    {
        "id": "real_wooden_crate_02",
        "polyhaven_id": "wooden_crate_02",
        "category": "box",
        "name": "weathered wooden crate",
        "dimensions": [0.56, 0.40, 0.36],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "wooden", "crate", "box", "child", "storage"],
        "color": [0.44, 0.28, 0.16],
    },
    {
        "id": "real_old_military_crate_01",
        "polyhaven_id": "old_military_crate",
        "category": "box",
        "name": "old green supply crate",
        "dimensions": [0.68, 0.38, 0.34],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "military", "crate", "box", "child", "storage"],
        "color": [0.22, 0.28, 0.18],
    },
    {
        "id": "real_barrel_01",
        "polyhaven_id": "Barrel_02",
        "category": "cylinder",
        "name": "industrial barrel",
        "dimensions": [0.58, 0.58, 0.9],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "barrel", "drum", "floor", "industrial"],
        "color": [0.35, 0.58, 0.62],
    },
    {
        "id": "real_red_barrel_01",
        "polyhaven_id": "Barrel_01",
        "category": "cylinder",
        "name": "red industrial oil barrel",
        "dimensions": [0.58, 0.58, 0.9],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "barrel", "drum", "floor", "industrial", "oil"],
        "color": [0.64, 0.10, 0.08],
    },
    {
        "id": "real_blue_barrel_03",
        "polyhaven_id": "barrel_03",
        "category": "cylinder",
        "name": "blue painted industrial barrel",
        "dimensions": [0.58, 0.58, 0.9],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "barrel", "drum", "floor", "industrial", "painted"],
        "color": [0.10, 0.25, 0.55],
    },
    {
        "id": "real_small_lpg_tank_01",
        "polyhaven_id": "small_lpg_tank",
        "category": "cylinder",
        "name": "small LPG tank",
        "dimensions": [0.36, 0.36, 0.62],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tank", "cylinder", "floor", "industrial", "propane"],
        "color": [0.55, 0.14, 0.12],
    },
    {
        "id": "real_propane_tank_01",
        "polyhaven_id": "propane_tank",
        "category": "cylinder",
        "name": "propane tank",
        "dimensions": [0.42, 0.42, 0.72],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tank", "cylinder", "floor", "industrial", "propane"],
        "color": [0.58, 0.14, 0.12],
    },
    {
        "id": "real_plastic_storage_container_01",
        "polyhaven_id": "plastic_container",
        "category": "bin",
        "name": "plastic storage container",
        "dimensions": [0.48, 0.32, 0.30],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "storage", "container", "bin", "box", "child"],
        "color": [0.22, 0.40, 0.72],
    },
    {
        "id": "real_metal_trash_can_01",
        "polyhaven_id": "metal_trash_can",
        "category": "bin",
        "name": "metal trash can",
        "dimensions": [0.48, 0.48, 0.82],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "bin", "trash", "metal", "floor"],
        "color": [0.42, 0.42, 0.40],
    },
    {
        "id": "real_metal_jerrycan_01",
        "polyhaven_id": "metal_jerrycan",
        "category": "container",
        "name": "red metal jerry can",
        "dimensions": [0.36, 0.16, 0.46],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "jerrycan", "fuel", "container", "floor", "industrial"],
        "color": [0.55, 0.08, 0.06],
    },
    {
        "id": "real_green_jerrycan_01",
        "polyhaven_id": "metal_jerrycan_green",
        "category": "container",
        "name": "green metal jerry can",
        "dimensions": [0.36, 0.16, 0.46],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "jerrycan", "fuel", "container", "floor", "industrial"],
        "color": [0.22, 0.30, 0.18],
    },
    {
        "id": "real_cement_bag_01",
        "polyhaven_id": "cement_bag",
        "category": "bag",
        "name": "cement bag",
        "dimensions": [0.62, 0.42, 0.18],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "bag", "cement", "construction", "floor", "storage"],
        "color": [0.58, 0.54, 0.48],
    },
    {
        "id": "real_hand_truck_01",
        "polyhaven_id": "hand_truck",
        "category": "hand_truck",
        "name": "warehouse hand truck",
        "dimensions": [0.58, 0.48, 1.2],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "dolly", "cart", "floor", "logistics"],
        "color": [0.65, 0.08, 0.07],
    },
    {
        "id": "real_metal_toolbox_01",
        "polyhaven_id": "metal_toolbox",
        "category": "tool",
        "name": "metal toolbox",
        "dimensions": [0.55, 0.28, 0.25],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "toolbox", "tool", "child", "maintenance"],
        "color": [0.5, 0.08, 0.08],
    },
    {
        "id": "real_bench_vice_01",
        "polyhaven_id": "bench_vice_01",
        "category": "tool",
        "name": "bench vice",
        "dimensions": [0.42, 0.20, 0.24],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tool", "vice", "workshop", "maintenance", "child"],
        "color": [0.25, 0.26, 0.25],
    },
    {
        "id": "real_drill_01",
        "polyhaven_id": "Drill_01",
        "category": "tool",
        "name": "hand drill",
        "dimensions": [0.32, 0.10, 0.24],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tool", "drill", "workshop", "maintenance", "child"],
        "color": [0.16, 0.18, 0.18],
    },
    {
        "id": "real_pipe_wrench_01",
        "polyhaven_id": "pipe_wrench",
        "category": "tool",
        "name": "pipe wrench",
        "dimensions": [0.38, 0.10, 0.06],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tool", "wrench", "workshop", "maintenance", "child"],
        "color": [0.35, 0.08, 0.06],
    },
    {
        "id": "real_ratchet_wrench_01",
        "polyhaven_id": "ratchet_wrench",
        "category": "tool",
        "name": "ratchet wrench",
        "dimensions": [0.30, 0.08, 0.05],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tool", "wrench", "workshop", "maintenance", "child"],
        "color": [0.35, 0.35, 0.34],
    },
    {
        "id": "real_tool_chest_01",
        "polyhaven_id": "metal_tool_chest",
        "category": "cabinet",
        "name": "rolling metal tool chest",
        "dimensions": [0.78, 0.48, 0.86],
        "support_kind": "container",
        "support_heights": [0.55, 0.95],
        "tags": ["warehouse", "tool_chest", "cabinet", "parent", "maintenance", "floor"],
        "color": [0.5, 0.08, 0.08],
    },
    {
        "id": "real_packing_table_01",
        "polyhaven_id": "SchoolDesk_01",
        "category": "table",
        "name": "packing work table",
        "dimensions": [1.15, 0.62, 0.78],
        "support_kind": "surface",
        "support_heights": [1.0],
        "tags": ["warehouse", "table", "workbench", "packing", "support", "anchor"],
        "color": [0.55, 0.36, 0.2],
    },
    {
        "id": "real_warehouse_chair_01",
        "polyhaven_id": "SchoolChair_01",
        "category": "chair",
        "name": "warehouse desk chair",
        "dimensions": [0.48, 0.48, 0.86],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "chair", "seat", "floor"],
        "color": [0.25, 0.35, 0.55],
    },
    {
        "id": "real_wooden_ladder_01",
        "polyhaven_id": "wooden_ladder",
        "category": "ladder",
        "name": "wooden ladder",
        "dimensions": [0.52, 0.18, 1.75],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "ladder", "maintenance", "floor", "construction"],
        "color": [0.55, 0.35, 0.18],
    },
    {
        "id": "real_section_ladder_01",
        "polyhaven_id": "ladder_sectioned_01",
        "category": "ladder",
        "name": "sectioned metal ladder",
        "dimensions": [0.58, 0.20, 1.85],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "ladder", "maintenance", "floor", "construction", "metal"],
        "color": [0.55, 0.55, 0.50],
    },
    {
        "id": "real_power_box_01",
        "polyhaven_id": "power_box_01",
        "category": "utility_box",
        "name": "wall power box",
        "dimensions": [0.58, 0.22, 0.72],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "utility", "power", "electrical", "wall"],
        "color": [0.35, 0.35, 0.34],
    },
    {
        "id": "real_utility_box_01",
        "polyhaven_id": "utility_box_01",
        "category": "utility_box",
        "name": "industrial utility box",
        "dimensions": [0.52, 0.22, 0.68],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "utility", "power", "electrical", "wall"],
        "color": [0.32, 0.34, 0.32],
    },
    {
        "id": "real_wet_floor_sign_01",
        "polyhaven_id": "WetFloorSign_01",
        "category": "sign",
        "name": "yellow wet floor safety sign",
        "dimensions": [0.34, 0.16, 0.72],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "safety", "sign", "floor", "warning"],
        "color": [0.95, 0.78, 0.08],
    },
    {
        "id": "real_mounted_fluorescent_lights_01",
        "polyhaven_id": "mounted_fluorescent_lights",
        "category": "light",
        "name": "mounted fluorescent lights",
        "dimensions": [1.2, 0.18, 0.12],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "light", "fluorescent", "ceiling", "industrial"],
        "color": [0.85, 0.85, 0.78],
    },
    {
        "id": "real_hanging_industrial_lamp_01",
        "polyhaven_id": "hanging_industrial_lamp",
        "category": "light",
        "name": "hanging industrial lamp",
        "dimensions": [0.36, 0.36, 0.42],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "light", "lamp", "ceiling", "industrial"],
        "color": [0.16, 0.16, 0.15],
    },
    {
        "id": "real_rollup_door_01",
        "polyhaven_id": "rollershutter_door",
        "category": "door",
        "name": "warehouse roller shutter door",
        "dimensions": [1.8, 0.16, 2.2],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "door", "roller", "shutter", "wall", "industrial"],
        "color": [0.45, 0.45, 0.43],
    },
    {
        "id": "real_airduct_01",
        "polyhaven_id": "modular_airduct_rectangular_01",
        "category": "duct",
        "name": "rectangular metal air duct",
        "dimensions": [1.4, 0.34, 0.28],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "airduct", "duct", "ceiling", "industrial"],
        "color": [0.45, 0.46, 0.45],
    },
    {
        "id": "real_industrial_pipes_01",
        "polyhaven_id": "modular_industrial_pipes_01",
        "category": "pipe",
        "name": "modular industrial pipes",
        "dimensions": [1.25, 0.28, 0.44],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "pipes", "industrial", "wall", "utility"],
        "color": [0.38, 0.38, 0.36],
    },
    {
        "id": "real_modular_pipes_01",
        "polyhaven_id": "modular_pipes",
        "category": "pipe",
        "name": "wall mounted modular pipes",
        "dimensions": [1.25, 0.22, 0.32],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "pipes", "industrial", "wall", "utility"],
        "color": [0.34, 0.34, 0.33],
    },
    {
        "id": "real_circular_airduct_01",
        "polyhaven_id": "modular_airduct_circular_01",
        "category": "duct",
        "name": "round metal air duct",
        "dimensions": [1.35, 0.32, 0.32],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "airduct", "duct", "ceiling", "industrial"],
        "color": [0.46, 0.47, 0.46],
    },
    {
        "id": "real_electric_cables_01",
        "polyhaven_id": "modular_electric_cables",
        "category": "cable",
        "name": "modular electric cables",
        "dimensions": [1.25, 0.10, 0.08],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "cables", "electrical", "wall", "utility"],
        "color": [0.08, 0.08, 0.08],
    },
    {
        "id": "real_concrete_barrier_01",
        "polyhaven_id": "concrete_road_barrier",
        "category": "barrier",
        "name": "concrete safety barrier",
        "dimensions": [1.55, 0.42, 0.72],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "barrier", "safety", "floor", "industrial"],
        "color": [0.52, 0.52, 0.48],
    },
    {
        "id": "real_concrete_barrier_02",
        "polyhaven_id": "concrete_road_barrier_02",
        "category": "barrier",
        "name": "second concrete safety barrier",
        "dimensions": [1.35, 0.42, 0.68],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "barrier", "safety", "floor", "industrial"],
        "color": [0.50, 0.50, 0.47],
    },
    {
        "id": "real_chainlink_fence_01",
        "polyhaven_id": "modular_chainlink_fence",
        "category": "barrier",
        "name": "modular chainlink fence panel",
        "dimensions": [1.75, 0.08, 1.55],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "fence", "barrier", "safety", "floor", "industrial"],
        "color": [0.42, 0.42, 0.40],
    },
    {
        "id": "real_security_camera_01",
        "polyhaven_id": "security_camera_01",
        "category": "camera",
        "name": "wall security camera",
        "dimensions": [0.28, 0.18, 0.18],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "camera", "security", "wall", "utility"],
        "color": [0.78, 0.78, 0.74],
    },
    {
        "id": "real_security_light_01",
        "polyhaven_id": "security_light",
        "category": "light",
        "name": "wall mounted security light",
        "dimensions": [0.34, 0.22, 0.26],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "light", "security", "wall", "industrial"],
        "color": [0.72, 0.72, 0.68],
    },
    {
        "id": "real_industrial_wall_lamp_01",
        "polyhaven_id": "industrial_wall_lamp",
        "category": "light",
        "name": "industrial wall lamp",
        "dimensions": [0.28, 0.20, 0.34],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "light", "lamp", "wall", "industrial"],
        "color": [0.18, 0.18, 0.16],
    },
    {
        "id": "real_caged_hanging_light_01",
        "polyhaven_id": "caged_hanging_light",
        "category": "light",
        "name": "caged hanging warehouse light",
        "dimensions": [0.34, 0.34, 0.52],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "light", "lamp", "ceiling", "industrial"],
        "color": [0.12, 0.12, 0.11],
    },
    {
        "id": "real_bolt_cutters_01",
        "polyhaven_id": "bolt_cutters_01",
        "category": "tool",
        "name": "bolt cutters",
        "dimensions": [0.52, 0.12, 0.08],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tool", "cutters", "maintenance", "child"],
        "color": [0.18, 0.18, 0.16],
    },
    {
        "id": "real_crowbar_01",
        "polyhaven_id": "crowbar_01",
        "category": "tool",
        "name": "red crowbar",
        "dimensions": [0.58, 0.08, 0.06],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tool", "crowbar", "maintenance", "child"],
        "color": [0.48, 0.06, 0.05],
    },
    {
        "id": "real_sledgehammer_01",
        "polyhaven_id": "sledgehammer_01",
        "category": "tool",
        "name": "sledgehammer",
        "dimensions": [0.72, 0.16, 0.10],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tool", "hammer", "maintenance", "child"],
        "color": [0.34, 0.24, 0.15],
    },
    {
        "id": "real_measuring_tape_01",
        "polyhaven_id": "measuring_tape_01",
        "category": "tool",
        "name": "measuring tape",
        "dimensions": [0.13, 0.10, 0.06],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tool", "measuring", "maintenance", "child"],
        "color": [0.88, 0.68, 0.06],
    },
    {
        "id": "real_flathead_screwdriver_01",
        "polyhaven_id": "flathead_screwdriver",
        "category": "tool",
        "name": "flathead screwdriver",
        "dimensions": [0.24, 0.04, 0.04],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "tool", "screwdriver", "maintenance", "child"],
        "color": [0.64, 0.08, 0.06],
    },
    {
        "id": "real_plastic_jerrycan_01",
        "polyhaven_id": "plastic_jerrycan",
        "category": "container",
        "name": "plastic jerry can",
        "dimensions": [0.34, 0.18, 0.42],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "jerrycan", "fuel", "container", "floor", "industrial", "plastic"],
        "color": [0.82, 0.70, 0.16],
    },
    {
        "id": "real_small_oil_can_01",
        "polyhaven_id": "small_oil_can_01",
        "category": "container",
        "name": "small oil can",
        "dimensions": [0.22, 0.14, 0.26],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "oil", "can", "container", "floor", "industrial"],
        "color": [0.62, 0.10, 0.08],
    },
    {
        "id": "real_spray_paint_bottles_01",
        "polyhaven_id": "spray_paint_bottles",
        "category": "container",
        "name": "spray paint bottles",
        "dimensions": [0.22, 0.18, 0.24],
        "support_kind": "none",
        "support_heights": [],
        "tags": ["warehouse", "spray", "paint", "bottles", "container", "maintenance", "child"],
        "color": [0.55, 0.10, 0.08],
    },
]


def request_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch JSON from {url}")


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, target: Path, expected_md5: str | None, force: bool) -> int:
    if target.exists() and not force:
        if expected_md5 and md5sum(target) != expected_md5:
            target.unlink()
        else:
            return target.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp_target = target.with_suffix(target.suffix + ".part")
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response, tmp_target.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            tmp_target.replace(target)
            break
        except Exception:
            if tmp_target.exists():
                tmp_target.unlink()
            if attempt == 3:
                raise
            time.sleep(2.0 * (attempt + 1))
    if expected_md5 and md5sum(target) != expected_md5:
        raise RuntimeError(f"checksum mismatch for {target}")
    return target.stat().st_size


def choose_gltf_package(files: dict[str, Any], resolution: str) -> tuple[str, dict[str, Any]]:
    gltf = files.get("gltf")
    if not gltf:
        raise RuntimeError("asset has no glTF package")
    if resolution in gltf and "gltf" in gltf[resolution]:
        return resolution, gltf[resolution]["gltf"]
    for candidate in ["1k", "2k", "4k", "8k"]:
        if candidate in gltf and "gltf" in gltf[candidate]:
            return candidate, gltf[candidate]["gltf"]
    raise RuntimeError("asset has no downloadable glTF entry")


def filename_from_url(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?", 1)[0]


def import_asset(asset: dict[str, Any], resolution: str, force: bool) -> dict[str, Any]:
    polyhaven_id = asset["polyhaven_id"]
    files = request_json(f"{API_BASE}/files/{polyhaven_id}")
    selected_resolution, package = choose_gltf_package(files, resolution)

    asset_dir = ROOT / "assets" / "library" / "polyhaven" / polyhaven_id
    main_filename = filename_from_url(package["url"])
    downloaded: list[dict[str, Any]] = []
    total_bytes = 0

    total_bytes += download_file(package["url"], asset_dir / main_filename, package.get("md5"), force)
    downloaded.append({"path": main_filename, "url": package["url"], "md5": package.get("md5"), "size": package.get("size")})
    for relative_path, include in sorted(package.get("include", {}).items()):
        total_bytes += download_file(include["url"], asset_dir / relative_path, include.get("md5"), force)
        downloaded.append(
            {
                "path": relative_path,
                "url": include["url"],
                "md5": include.get("md5"),
                "size": include.get("size"),
            }
        )

    gltf_path = asset_dir / main_filename
    registry_record = {
        key: value
        for key, value in asset.items()
        if key
        in {
            "id",
            "category",
            "name",
            "dimensions",
            "support_kind",
            "support_heights",
            "tags",
            "color",
        }
    }
    registry_record.update(
        {
            "glb_path": str(Path("..") / "assets" / "library" / "polyhaven" / polyhaven_id / main_filename),
            "source": "polyhaven",
            "source_id": polyhaven_id,
            "source_url": f"https://polyhaven.com/a/{polyhaven_id}",
            "license": "CC0 1.0",
            "attribution": "Poly Haven",
        }
    )
    return {
        "registry_record": registry_record,
        "manifest_record": {
            "registry_id": asset["id"],
            "polyhaven_id": polyhaven_id,
            "resolution": selected_resolution,
            "local_gltf": str(gltf_path.relative_to(ROOT)),
            "source_url": f"https://polyhaven.com/a/{polyhaven_id}",
            "api_url": f"{API_BASE}/files/{polyhaven_id}",
            "license": "CC0 1.0",
            "downloaded_files": downloaded,
            "downloaded_bytes": total_bytes,
        },
    }


def build_registry(records: list[dict[str, Any]]) -> None:
    base_path = ROOT / "configs" / "asset_registry.yaml"
    with base_path.open("r", encoding="utf-8") as handle:
        base = yaml.safe_load(handle) or {}
    existing = [asset for asset in base.get("assets", []) if not str(asset.get("id", "")).startswith("real_")]
    output = {"assets": records + existing}
    out_path = ROOT / "configs" / "warehouse_asset_registry.yaml"
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False, allow_unicode=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small CC0 Poly Haven warehouse asset pack.")
    parser.add_argument("--resolution", default="1k", choices=["1k", "2k", "4k", "8k"])
    parser.add_argument("--force", action="store_true", help="Redownload existing files.")
    args = parser.parse_args()

    registry_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    for asset in WAREHOUSE_ASSETS:
        print(f"Importing {asset['polyhaven_id']}...")
        result = import_asset(asset, args.resolution, args.force)
        registry_records.append(result["registry_record"])
        manifest_records.append(result["manifest_record"])

    build_registry(registry_records)
    manifest_path = ROOT / "assets" / "manifests" / "polyhaven_warehouse_assets.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "source": "polyhaven",
                "license": "CC0 1.0",
                "asset_count": len(manifest_records),
                "assets": manifest_records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote configs/warehouse_asset_registry.yaml")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"asset import failed: {exc}", file=sys.stderr)
        sys.exit(1)
