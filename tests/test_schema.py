from __future__ import annotations

import pytest
from pydantic import ValidationError

from scenethesis_mvp.schemas.scene_spec import ObjectSpec, SceneSpec


def test_scene_spec_requires_one_anchor() -> None:
    scene = SceneSpec(
        prompt="lab",
        objects=[
            ObjectSpec(id="table", category="table", role="anchor"),
            ObjectSpec(id="chair", category="chair", role="parent"),
        ],
    )
    assert scene.object_by_id("table").role == "anchor"


def test_scene_spec_rejects_duplicate_ids() -> None:
    with pytest.raises(ValidationError):
        SceneSpec(
            prompt="bad",
            objects=[
                ObjectSpec(id="table", category="table", role="anchor"),
                ObjectSpec(id="table", category="chair", role="parent"),
            ],
        )
