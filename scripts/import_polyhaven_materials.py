from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
API_BASE = "https://api.polyhaven.com"
USER_AGENT = "scenethesis-mvp-local-material-importer/0.1"


MATERIALS = [
    {
        "id": "concrete_floor_worn_001",
        "maps": {
            "diffuse": ("Diffuse", "jpg"),
            "normal": ("nor_gl", "jpg"),
            "roughness": ("Rough", "jpg"),
        },
    },
    {
        "id": "factory_wall",
        "maps": {
            "diffuse": ("Diffuse", "jpg"),
            "normal": ("nor_gl", "jpg"),
            "roughness": ("Rough", "jpg"),
        },
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


def choose_texture(files: dict[str, Any], map_name: str, resolution: str, extension: str) -> dict[str, Any]:
    if map_name not in files:
        raise RuntimeError(f"material map is missing: {map_name}")
    map_files = files[map_name]
    if resolution in map_files and extension in map_files[resolution]:
        return map_files[resolution][extension]
    for candidate in ["1k", "2k", "4k", "8k"]:
        if candidate in map_files and extension in map_files[candidate]:
            return map_files[candidate][extension]
    raise RuntimeError(f"no downloadable {extension} texture for map {map_name}")


def import_material(material: dict[str, Any], resolution: str, force: bool) -> dict[str, Any]:
    material_id = material["id"]
    files = request_json(f"{API_BASE}/files/{material_id}")
    material_dir = ROOT / "assets" / "materials" / "polyhaven" / material_id
    downloaded = {}
    total_bytes = 0
    for alias, (map_name, extension) in material["maps"].items():
        entry = choose_texture(files, map_name, resolution, extension)
        filename = entry["url"].rsplit("/", 1)[-1].split("?", 1)[0]
        target = material_dir / filename
        total_bytes += download_file(entry["url"], target, entry.get("md5"), force)
        downloaded[alias] = {
            "path": str(target.relative_to(ROOT)),
            "url": entry["url"],
            "md5": entry.get("md5"),
            "size": entry.get("size"),
        }
    return {
        "id": material_id,
        "source": "polyhaven",
        "source_url": f"https://polyhaven.com/a/{material_id}",
        "license": "CC0 1.0",
        "resolution": resolution,
        "downloaded_bytes": total_bytes,
        "maps": downloaded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Download small CC0 Poly Haven warehouse render materials.")
    parser.add_argument("--resolution", default="1k", choices=["1k", "2k", "4k", "8k"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    records = []
    for material in MATERIALS:
        print(f"Importing material {material['id']}...")
        records.append(import_material(material, args.resolution, args.force))
    manifest_path = ROOT / "assets" / "manifests" / "polyhaven_materials.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "source": "polyhaven",
                "license": "CC0 1.0",
                "material_count": len(records),
                "materials": records,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {manifest_path.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"material import failed: {exc}", file=sys.stderr)
        sys.exit(1)
