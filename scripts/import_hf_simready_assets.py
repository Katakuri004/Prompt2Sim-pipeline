from __future__ import annotations

import argparse
import json
import posixpath
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scenethesis_mvp.render.blender_runner import resolve_blender_path
from scenethesis_mvp.schemas.asset_manifest import ControlledAssetManifest, manifest_source_dir
from scenethesis_mvp.utils.io import read_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a controlled subset of HF SimReady warehouse USD assets.")
    parser.add_argument("--manifest", default="configs/hf_simready_warehouse_manifest.yaml")
    parser.add_argument("--registry", default="configs/warehouse_asset_registry.yaml")
    parser.add_argument("--library-dir", default="assets/library/hf_simready")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-convert", action="store_true", help="Download only; do not register assets without converted GLBs.")
    args = parser.parse_args()

    manifest_path = resolve_repo_path(args.manifest)
    registry_path = resolve_repo_path(args.registry)
    library_dir = resolve_repo_path(args.library_dir)
    manifest = ControlledAssetManifest.model_validate(read_yaml(manifest_path))
    if manifest.bulk_download_allowed:
        raise RuntimeError("Manifest permits bulk download; refusing to run.")

    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    except Exception as exc:
        raise RuntimeError(f"huggingface_hub is required for controlled HF asset import: {exc}") from exc

    blender = resolve_blender_path(None)
    if not blender and not args.skip_convert:
        raise RuntimeError("Blender is required to convert HF SimReady USD assets to GLB.")

    api = HfApi()
    registry_records: list[dict[str, Any]] = []
    manifest_records: list[dict[str, Any]] = []
    for entry in manifest.entries:
        print(f"Importing {entry.id} from {entry.repo_id}:{entry.source_usd}")
        repo_files = set(api.list_repo_files(entry.repo_id, repo_type=entry.repo_type))
        if entry.source_usd not in repo_files:
            raise RuntimeError(f"Manifest source_usd does not exist in repo: {entry.source_usd}")
        if entry.thumbnail_path and entry.thumbnail_path not in repo_files:
            raise RuntimeError(f"Manifest thumbnail_path does not exist in repo: {entry.thumbnail_path}")

        source_dir = manifest_source_dir(entry)
        asset_root = library_dir / entry.id
        raw_root = asset_root / "raw"
        allow_patterns = [f"{source_dir}/**"]
        allow_patterns.extend(f"{prefix.rstrip('/')}/**" for prefix in entry.include_prefixes)
        if entry.thumbnail_path:
            allow_patterns.append(entry.thumbnail_path)
        snapshot_download(
            repo_id=entry.repo_id,
            repo_type=entry.repo_type,
            allow_patterns=allow_patterns,
            local_dir=raw_root,
        )
        source_usd = raw_root / entry.source_usd
        if not source_usd.is_file():
            raise RuntimeError(f"Controlled HF download did not produce expected USD: {source_usd}")
        dependency_prefixes = download_declared_usd_dependencies(
            repo_id=entry.repo_id,
            repo_type=entry.repo_type,
            repo_files=repo_files,
            raw_root=raw_root,
            initial_usd=source_usd,
            snapshot_download=snapshot_download,
        )

        glb_path = asset_root / f"{entry.id}.glb"
        if not args.skip_convert:
            convert_usd_to_glb(blender or "blender", source_usd, glb_path, force=args.force)
        if not glb_path.is_file():
            raise RuntimeError(f"Converted GLB is missing for {entry.id}: {glb_path}")

        thumbnail_output = asset_root / f"{entry.id}.png"
        if entry.thumbnail_path:
            source_thumb = raw_root / entry.thumbnail_path
            if source_thumb.is_file():
                shutil.copy2(source_thumb, thumbnail_output)
            else:
                raise RuntimeError(f"Downloaded thumbnail is missing for {entry.id}: {source_thumb}")

        registry_records.append(registry_record(entry, glb_path, thumbnail_output if thumbnail_output.is_file() else None))
        manifest_records.append(
            {
                "registry_id": entry.id,
                "repo_id": entry.repo_id,
                "repo_type": entry.repo_type,
                "source_usd": entry.source_usd,
                "local_usd": str(source_usd.relative_to(ROOT)),
                "local_glb": str(glb_path.relative_to(ROOT)),
                "local_thumbnail": str(thumbnail_output.relative_to(ROOT)) if thumbnail_output.is_file() else None,
                "declared_dependency_prefixes": dependency_prefixes,
                "source_url": entry.source_url or hf_file_url(entry.repo_id, entry.source_usd),
                "license": entry.license,
                "attribution": entry.attribution,
            }
        )

    merge_registry(registry_path, registry_records)
    write_import_manifest(manifest, manifest_records)
    print(f"Wrote {registry_path.relative_to(ROOT)}")
    print("HF SimReady import complete.")


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate.resolve()


def convert_usd_to_glb(blender: str, source_usd: Path, glb_path: Path, force: bool = False) -> None:
    if glb_path.is_file() and glb_path.stat().st_size > 0 and not force:
        return
    script = ROOT / "src" / "scenethesis_mvp" / "render" / "usd_to_glb_blender.py"
    command = [
        blender,
        "--background",
        "--python",
        str(script),
        "--",
        "--input",
        str(source_usd),
        "--out",
        str(glb_path),
    ]
    subprocess.run(command, check=True)


def download_declared_usd_dependencies(
    repo_id: str,
    repo_type: str,
    repo_files: set[str],
    raw_root: Path,
    initial_usd: Path,
    snapshot_download: Any,
) -> list[str]:
    discovered: set[str] = set()
    scanned: set[Path] = set()
    pending = [initial_usd]
    for _ in range(4):
        new_prefixes: set[str] = set()
        while pending:
            usd_path = pending.pop()
            if usd_path in scanned or not usd_path.is_file():
                continue
            scanned.add(usd_path)
            repo_usd_path = usd_path.relative_to(raw_root).as_posix()
            source_dir = posixpath.dirname(repo_usd_path)
            for dependency in extract_usd_dependency_paths(usd_path, source_dir):
                if dependency in repo_files:
                    new_prefixes.add(posixpath.dirname(dependency))
        new_prefixes -= discovered
        if not new_prefixes:
            break
        discovered |= new_prefixes
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            allow_patterns=[f"{prefix}/**" for prefix in sorted(new_prefixes)],
            local_dir=raw_root,
        )
        for prefix in sorted(new_prefixes):
            pending.extend(path for path in (raw_root / prefix).glob("*.usd") if path.is_file())
    return sorted(discovered)


def extract_usd_dependency_paths(usd_path: Path, source_dir: str) -> set[str]:
    raw = usd_path.read_bytes().decode("latin1", errors="ignore")
    dependencies: set[str] = set()
    patterns = [
        r"((?:\.\./)+[A-Za-z0-9_./-]+?\.usd)",
        r"(Props/[A-Za-z0-9_./-]+?\.usd)",
    ]
    for pattern in patterns:
        for match in re.findall(pattern, raw):
            normalized = posixpath.normpath(posixpath.join(source_dir, match)) if match.startswith("../") else posixpath.normpath(match)
            if normalized.startswith("Props/"):
                dependencies.add(normalized)
    return dependencies


def registry_record(entry: Any, glb_path: Path, thumbnail_path: Path | None) -> dict[str, Any]:
    record = {
        "id": entry.id,
        "category": entry.category,
        "name": entry.name,
        "dimensions": entry.dimensions,
        "support_kind": entry.support_kind,
        "support_heights": entry.support_heights,
        "tags": entry.tags,
        "color": entry.color,
        "glb_path": str(Path("..") / glb_path.relative_to(ROOT)),
        "source": "huggingface_simready",
        "source_id": entry.source_usd,
        "source_url": entry.source_url or hf_file_url(entry.repo_id, entry.source_usd),
        "license": entry.license,
        "attribution": entry.attribution,
    }
    if thumbnail_path:
        record["thumbnail_path"] = str(Path("..") / thumbnail_path.relative_to(ROOT))
    return record


def hf_file_url(repo_id: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"


def merge_registry(registry_path: Path, new_records: list[dict[str, Any]]) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_yaml(registry_path) if registry_path.exists() else {"assets": []}
    by_id = {asset["id"]: asset for asset in existing.get("assets", [])}
    for record in new_records:
        by_id[record["id"]] = record
    output = {"assets": list(by_id.values())}
    with registry_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(output, handle, sort_keys=False, allow_unicode=False)


def write_import_manifest(manifest: ControlledAssetManifest, records: list[dict[str, Any]]) -> None:
    out_path = ROOT / "assets" / "manifests" / "hf_simready_warehouse_assets.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_family": manifest.source_family,
        "bulk_download_allowed": manifest.bulk_download_allowed,
        "disk_budget_gb": manifest.disk_budget_gb,
        "license_note": "Per-entry license metadata is preserved in this manifest and registry records.",
        "asset_count": len(records),
        "assets": records,
        "unresolved_targets": [item.model_dump(mode="json") for item in manifest.unresolved_targets],
    }
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"HF SimReady import failed: {exc}", file=sys.stderr)
        sys.exit(1)
