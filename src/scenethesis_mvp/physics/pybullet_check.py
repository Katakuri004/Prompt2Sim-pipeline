from __future__ import annotations

from scenethesis_mvp.assets.registry import AssetRegistry
from scenethesis_mvp.schemas.scene_spec import SceneSpec


def run_pybullet_check(scene: SceneSpec, registry: AssetRegistry) -> dict[str, object]:
    """Fail-fast boundary for a future real PyBullet integration."""

    try:
        import pybullet  # noqa: F401
    except Exception as exc:
        raise RuntimeError("PyBullet is not installed; no PyBullet simulation was run.") from exc
    raise NotImplementedError(
        "PyBullet is importable, but real mesh/body simulation is not implemented in this repository. "
        "Use the configured SDF optimizer for the faithful pipeline, or add explicit PyBullet collision-shape "
        "construction before enabling this provider."
    )
