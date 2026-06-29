from __future__ import annotations

import fnmatch
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

from pydantic import BaseModel, Field

from scenethesis_mvp.utils.io import read_json, read_yaml, write_json
from scenethesis_mvp.utils.paths import project_root, resolve_path

ArtifactKind = Literal["dataset", "checkpoint", "eval"]
Backend = Literal["rclone", "local"]


class ArtifactManifest(BaseModel):
    artifact_id: str
    artifact_type: ArtifactKind
    local_path: str
    remote_path: str
    byte_size: int
    file_count: int
    sha256_manifest: dict[str, str] = Field(default_factory=dict)
    created_time: str
    status: str
    backend: Backend
    excluded_patterns: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SyncConfig:
    root: Path
    local_cache_root: Path
    artifact_root: Path
    remote_root: str
    backend: Backend
    rclone_remote: str
    manifest_name: str
    artifact_types: dict[str, dict[str, str]]
    eval_exclude_patterns: tuple[str, ...]


def load_sync_config(config_path: str | Path = "configs/artifact_sync.yaml") -> SyncConfig:
    root = project_root()
    raw = read_yaml(resolve_path(config_path, root))
    paths = raw.get("paths", {})
    sync = raw.get("sync", {})
    retention = raw.get("retention", {})
    return SyncConfig(
        root=root,
        local_cache_root=resolve_path(paths.get("local_cache_root", "data/lerobot_cache"), root),
        artifact_root=resolve_path(paths.get("artifact_root", "outputs/lerobot_phase1"), root),
        remote_root=str(paths.get("remote_root", "gdrive:Scenethesis")).rstrip("/\\"),
        backend=str(sync.get("backend", "rclone")),
        rclone_remote=str(sync.get("rclone_remote", "gdrive")),
        manifest_name=str(sync.get("manifest_name", "artifact_manifest.json")),
        artifact_types=dict(raw.get("artifact_types", {})),
        eval_exclude_patterns=tuple(str(item) for item in retention.get("eval_exclude_patterns", [])),
    )


def push_artifact(
    *,
    artifact_id: str,
    artifact_type: ArtifactKind,
    local_path: str | Path | None = None,
    remote_path: str | Path | None = None,
    config: SyncConfig | None = None,
    backend: Backend | None = None,
    remote_root: str | Path | None = None,
    exclude_patterns: Iterable[str] | None = None,
) -> ArtifactManifest:
    cfg = config or load_sync_config()
    selected_backend: Backend = backend or cfg.backend
    source = _resolve_local_path(cfg, artifact_type, artifact_id, local_path)
    if not source.is_dir():
        raise FileNotFoundError(f"Artifact source directory does not exist: {source}")
    patterns = tuple(exclude_patterns or _default_excludes(cfg, artifact_type))
    remote = _resolve_remote_path(cfg, artifact_type, artifact_id, remote_path, remote_root)
    manifest = _build_manifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        local_path=source,
        remote_path=remote,
        backend=selected_backend,
        status="push_pending",
        manifest_name=cfg.manifest_name,
        exclude_patterns=patterns,
    )
    write_json(source / cfg.manifest_name, manifest)
    if selected_backend == "local":
        _copy_tree_local(source, Path(remote), cfg.manifest_name, patterns)
    elif selected_backend == "rclone":
        _require_rclone()
        _rclone_copy(source, str(remote), patterns)
    else:
        raise ValueError(f"Unsupported sync backend: {selected_backend}")
    manifest.status = "pushed"
    write_json(source / cfg.manifest_name, manifest)
    if selected_backend == "local":
        write_json(Path(remote) / cfg.manifest_name, manifest)
    else:
        _rclone_copy_file(source / cfg.manifest_name, str(remote))
    return manifest


def pull_artifact(
    *,
    artifact_id: str,
    artifact_type: ArtifactKind,
    local_path: str | Path | None = None,
    remote_path: str | Path | None = None,
    config: SyncConfig | None = None,
    backend: Backend | None = None,
    remote_root: str | Path | None = None,
) -> ArtifactManifest:
    cfg = config or load_sync_config()
    selected_backend: Backend = backend or cfg.backend
    target = _resolve_local_path(cfg, artifact_type, artifact_id, local_path)
    remote = _resolve_remote_path(cfg, artifact_type, artifact_id, remote_path, remote_root)
    if selected_backend == "local":
        source = Path(remote)
        if not source.is_dir():
            raise FileNotFoundError(f"Remote artifact directory does not exist: {source}")
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    elif selected_backend == "rclone":
        _require_rclone()
        target.mkdir(parents=True, exist_ok=True)
        _rclone_copy(str(remote), target, ())
    else:
        raise ValueError(f"Unsupported sync backend: {selected_backend}")
    manifest_path = target / cfg.manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pulled artifact is missing {cfg.manifest_name}: {target}")
    manifest = ArtifactManifest.model_validate(read_json(manifest_path))
    manifest.local_path = str(target)
    manifest.status = "pulled"
    write_json(manifest_path, manifest)
    verify_artifact(local_path=target, manifest_name=cfg.manifest_name)
    return manifest


def verify_artifact(
    *,
    local_path: str | Path,
    manifest_name: str = "artifact_manifest.json",
) -> ArtifactManifest:
    target = Path(local_path)
    manifest_path = target / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing artifact manifest: {manifest_path}")
    manifest = ArtifactManifest.model_validate(read_json(manifest_path))
    observed_hashes, byte_size, file_count = _hash_tree(target, manifest_name, tuple(manifest.excluded_patterns))
    if observed_hashes != manifest.sha256_manifest:
        missing = sorted(set(manifest.sha256_manifest) - set(observed_hashes))
        extra = sorted(set(observed_hashes) - set(manifest.sha256_manifest))
        changed = sorted(
            path
            for path in set(observed_hashes).intersection(manifest.sha256_manifest)
            if observed_hashes[path] != manifest.sha256_manifest[path]
        )
        raise RuntimeError(
            "Artifact checksum verification failed: "
            f"missing={missing[:5]}, extra={extra[:5]}, changed={changed[:5]}"
        )
    if byte_size != manifest.byte_size or file_count != manifest.file_count:
        raise RuntimeError(
            "Artifact size/count verification failed: "
            f"expected bytes={manifest.byte_size}, files={manifest.file_count}; "
            f"got bytes={byte_size}, files={file_count}"
        )
    return manifest


def _resolve_local_path(
    cfg: SyncConfig,
    artifact_type: ArtifactKind,
    artifact_id: str,
    local_path: str | Path | None,
) -> Path:
    if local_path is not None:
        path = Path(local_path)
        return path if path.is_absolute() else cfg.root / path
    if artifact_type == "dataset":
        return cfg.local_cache_root / "datasets" / artifact_id
    type_cfg = cfg.artifact_types.get(artifact_type, {})
    return cfg.artifact_root / type_cfg.get("local_subdir", f"{artifact_type}s") / artifact_id


def _resolve_remote_path(
    cfg: SyncConfig,
    artifact_type: ArtifactKind,
    artifact_id: str,
    remote_path: str | Path | None,
    remote_root: str | Path | None,
) -> str:
    if remote_path is not None:
        return str(remote_path).rstrip("/\\")
    root = str(remote_root if remote_root is not None else cfg.remote_root).rstrip("/\\")
    type_cfg = cfg.artifact_types.get(artifact_type, {})
    subdir = type_cfg.get("remote_subdir", f"{artifact_type}s").strip("/\\")
    return f"{root}/{subdir}/{artifact_id}"


def _default_excludes(cfg: SyncConfig, artifact_type: ArtifactKind) -> tuple[str, ...]:
    if artifact_type == "eval":
        return cfg.eval_exclude_patterns
    return ()


def _build_manifest(
    *,
    artifact_id: str,
    artifact_type: ArtifactKind,
    local_path: Path,
    remote_path: str,
    backend: Backend,
    status: str,
    manifest_name: str,
    exclude_patterns: tuple[str, ...],
) -> ArtifactManifest:
    hashes, byte_size, file_count = _hash_tree(local_path, manifest_name, exclude_patterns)
    return ArtifactManifest(
        artifact_id=artifact_id,
        artifact_type=artifact_type,
        local_path=str(local_path),
        remote_path=remote_path,
        byte_size=byte_size,
        file_count=file_count,
        sha256_manifest=hashes,
        created_time=datetime.now(timezone.utc).isoformat(),
        status=status,
        backend=backend,
        excluded_patterns=list(exclude_patterns),
    )


def _hash_tree(root: Path, manifest_name: str, exclude_patterns: tuple[str, ...]) -> tuple[dict[str, str], int, int]:
    hashes: dict[str, str] = {}
    byte_size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rel = path.relative_to(root).as_posix()
        if rel == manifest_name or _excluded(rel, exclude_patterns):
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[rel] = digest.hexdigest()
        byte_size += path.stat().st_size
    return hashes, byte_size, len(hashes)


def _excluded(rel_path: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(rel_path, pattern) for pattern in patterns)


def _copy_tree_local(source: Path, target: Path, manifest_name: str, exclude_patterns: tuple[str, ...]) -> None:
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        rel = path.relative_to(source).as_posix()
        if rel != manifest_name and _excluded(rel, exclude_patterns):
            continue
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)


def _require_rclone() -> None:
    if shutil.which("rclone") is None:
        raise RuntimeError(
            "rclone is not available on PATH. Install rclone and configure a Google Drive remote named "
            "'gdrive', or run with --backend local for credential-free tests."
        )


def _rclone_copy(source: str | Path, target: str | Path, exclude_patterns: Iterable[str]) -> None:
    command = ["rclone", "copy", str(source), str(target), "--create-empty-src-dirs"]
    for pattern in exclude_patterns:
        command.extend(["--exclude", str(pattern)])
    subprocess.run(command, check=True)


def _rclone_copy_file(source: Path, target_remote_dir: str) -> None:
    subprocess.run(["rclone", "copyto", str(source), f"{target_remote_dir.rstrip('/')}/{source.name}"], check=True)
