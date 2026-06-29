from __future__ import annotations

from pathlib import Path

import pytest

from scenethesis_mvp.artifacts.drive_sync import pull_artifact, push_artifact, verify_artifact


def test_local_backend_push_pull_verify_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "source_dataset"
    source.mkdir()
    (source / "meta").mkdir()
    (source / "meta" / "info.json").write_text('{"ok": true}', encoding="utf-8")
    (source / "frame.txt").write_text("demo", encoding="utf-8")
    remote_root = tmp_path / "remote"
    manifest = push_artifact(
        artifact_id="tiny_dataset",
        artifact_type="dataset",
        local_path=source,
        backend="local",
        remote_root=remote_root,
    )
    assert manifest.status == "pushed"
    assert (source / "artifact_manifest.json").is_file()
    assert (remote_root / "datasets" / "tiny_dataset" / "artifact_manifest.json").is_file()

    pulled = tmp_path / "pulled_dataset"
    pulled_manifest = pull_artifact(
        artifact_id="tiny_dataset",
        artifact_type="dataset",
        local_path=pulled,
        backend="local",
        remote_root=remote_root,
    )
    assert pulled_manifest.status == "pulled"
    verified = verify_artifact(local_path=pulled)
    assert verified.file_count == 2


def test_verify_detects_corruption(tmp_path: Path) -> None:
    source = tmp_path / "source_eval"
    source.mkdir()
    (source / "evaluation_report.json").write_text('{"success_rate": 1.0}', encoding="utf-8")
    push_artifact(
        artifact_id="eval_001",
        artifact_type="eval",
        local_path=source,
        backend="local",
        remote_root=tmp_path / "remote",
    )
    (source / "evaluation_report.json").write_text("corrupt", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum"):
        verify_artifact(local_path=source)


def test_eval_sync_excludes_frame_directories(tmp_path: Path) -> None:
    source = tmp_path / "eval"
    source.mkdir()
    (source / "evaluation_report.json").write_text("{}", encoding="utf-8")
    frame_dir = source / "visual_twin_frames"
    frame_dir.mkdir()
    (frame_dir / "frame_000001.png").write_text("large temporary frame", encoding="utf-8")
    remote_root = tmp_path / "remote"
    manifest = push_artifact(
        artifact_id="eval_no_frames",
        artifact_type="eval",
        local_path=source,
        backend="local",
        remote_root=remote_root,
        exclude_patterns=["visual_twin_frames/**"],
    )
    assert "visual_twin_frames/frame_000001.png" not in manifest.sha256_manifest
    assert not (remote_root / "evals" / "eval_no_frames" / "visual_twin_frames" / "frame_000001.png").exists()
