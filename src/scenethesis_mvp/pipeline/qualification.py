from __future__ import annotations

from pathlib import Path
from typing import Any

from scenethesis_mvp.schemas.metrics import Metrics
from scenethesis_mvp.schemas.qualification import QualificationCheck, QualificationReport
from scenethesis_mvp.schemas.scene_spec import SceneSpec
from scenethesis_mvp.utils.io import read_json, write_json


def build_success_qualification(
    scene: SceneSpec,
    metrics: Metrics,
    judge: dict[str, Any],
    out_dir: str | Path,
    min_objects: int = 6,
) -> QualificationReport:
    target = Path(out_dir)
    render_validation = read_json(target / "render_validation.json") if (target / "render_validation.json").exists() else {}
    correspondence = read_json(target / "correspondence_diagnostics.json") if (target / "correspondence_diagnostics.json").exists() else {}
    required_files = ["scene_spec.json", "scene.glb", "render.png", "metrics.json", "judge.json", "pipeline_diagnostics.json"]
    checks = [
        QualificationCheck(
            name="required_outputs",
            ok=all((target / name).is_file() for name in required_files),
            detail=", ".join(name for name in required_files if not (target / name).is_file()) or "ok",
        ),
        QualificationCheck(
            name="object_count",
            ok=len(scene.objects) >= min_objects,
            detail=f"{len(scene.objects)} objects; requires at least {min_objects}",
        ),
        QualificationCheck(
            name="sdf_collisions",
            ok=metrics.collision_count == 0,
            detail=f"collision_count={metrics.collision_count}",
        ),
        QualificationCheck(
            name="support_stability",
            ok=metrics.floating_count == 0 and metrics.unsupported_count == 0,
            detail=f"floating={metrics.floating_count}, unsupported={metrics.unsupported_count}",
        ),
        QualificationCheck(
            name="render_visual_support",
            ok=bool(render_validation.get("ok", False)),
            detail=f"visual_support_failure_count={render_validation.get('visual_support_failure_count', 'missing')}",
        ),
        QualificationCheck(
            name="judge",
            ok=not bool(judge.get("needs_repair")),
            detail="needs_repair=false" if not judge.get("needs_repair") else "needs_repair=true",
        ),
        QualificationCheck(
            name="roma_correspondence",
            ok=bool(correspondence.get("ok", False)),
            detail=(
                f"failed_object_count={correspondence.get('failed_object_count', 'missing')}, "
                f"applied_updates={correspondence.get('applied_updates', 'missing')}"
            ),
        ),
    ]
    reasons = [f"{check.name}: {check.detail}" for check in checks if not check.ok]
    accepted = not reasons
    return QualificationReport(
        status="accepted" if accepted else "unqualified",
        accepted=accepted,
        stage="final_validation",
        checks=checks,
        reasons=reasons,
        metadata={"min_objects": min_objects},
    )


def build_failure_qualification(stage: str, error: BaseException) -> QualificationReport:
    reason = f"{type(error).__name__}: {error}"
    return QualificationReport(
        status="unqualified",
        accepted=False,
        stage=stage,
        checks=[QualificationCheck(name=stage, ok=False, detail=reason)],
        reasons=[reason],
        metadata={"error_type": type(error).__name__},
    )


def write_qualification(out_dir: str | Path, report: QualificationReport) -> None:
    write_json(Path(out_dir) / "qualification.json", report.model_dump(mode="json"))
