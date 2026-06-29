from __future__ import annotations

import base64
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from scenethesis_mvp.llm.openai_client import OpenAIClient


class FakeImages:
    def __init__(self) -> None:
        self.generate_kwargs: dict[str, Any] | None = None
        self.edit_kwargs: dict[str, Any] | None = None

    def generate(self, **kwargs: Any) -> Any:
        self.generate_kwargs = kwargs
        return _image_response()

    def edit(self, **kwargs: Any) -> Any:
        self.edit_kwargs = kwargs
        return _image_response()


def _image_response() -> Any:
    item = SimpleNamespace(b64_json=base64.b64encode(b"image-bytes").decode("ascii"), revised_prompt=None)
    return SimpleNamespace(data=[item])


def _client(images: FakeImages) -> OpenAIClient:
    client = OpenAIClient.__new__(OpenAIClient)
    client.api_key = "test"
    client._client = SimpleNamespace(images=images)
    return client


def test_gpt_image_generation_uses_supported_single_request(tmp_path: Path) -> None:
    images = FakeImages()
    output = tmp_path / "generated.png"

    metadata = _client(images).generate_image("warehouse", output, model="gpt-image-1", max_retries=1)

    assert output.read_bytes() == b"image-bytes"
    assert images.generate_kwargs is not None
    assert "response_format" not in images.generate_kwargs
    assert images.generate_kwargs["output_format"] == "png"
    assert metadata["operation"] == "generate"


def test_gpt_image_edit_uses_high_input_fidelity_without_alternate_request(tmp_path: Path) -> None:
    images = FakeImages()
    source = tmp_path / "source.png"
    source.write_bytes(b"source-image")
    reference = tmp_path / "reference.png"
    reference.write_bytes(b"reference-image")
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"mask-image")
    output = tmp_path / "edited.png"

    metadata = _client(images).edit_image(
        source,
        "preserve scene",
        output,
        model="gpt-image-1",
        reference_image_paths=[reference],
        mask_path=mask,
        max_retries=1,
    )

    assert output.read_bytes() == b"image-bytes"
    assert images.edit_kwargs is not None
    assert "response_format" not in images.edit_kwargs
    assert images.edit_kwargs["input_fidelity"] == "high"
    assert len(images.edit_kwargs["image"]) == 2
    assert "mask" in images.edit_kwargs
    assert metadata["mask_path"] == str(mask)
    assert metadata["operation"] == "edit"
