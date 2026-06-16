from __future__ import annotations

from scenethesis_mvp.llm.openai_client import _retry_delay_seconds, make_openai_strict_schema
from scenethesis_mvp.schemas.scene_spec import SceneSpec


def _walk_objects(node: object) -> list[dict]:
    found: list[dict] = []
    if isinstance(node, dict):
        if "properties" in node:
            found.append(node)
        for value in node.values():
            found.extend(_walk_objects(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_objects(item))
    return found


def test_openai_strict_schema_marks_all_object_properties_required() -> None:
    schema = make_openai_strict_schema(SceneSpec.model_json_schema())
    for obj_schema in _walk_objects(schema):
        properties = obj_schema["properties"]
        assert obj_schema["additionalProperties"] is False
        assert set(obj_schema["required"]) == set(properties)


def test_openai_retry_delay_honors_rate_limit_message() -> None:
    exc = RuntimeError("Rate limit reached. Please try again in 3.243s.")
    assert _retry_delay_seconds(exc, attempt=0) >= 3.7
