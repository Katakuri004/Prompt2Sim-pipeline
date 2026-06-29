from __future__ import annotations

import base64
import copy
import json
import os
import re
import time
import urllib.request
from contextlib import ExitStack
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional import guard
    load_dotenv = None


class OpenAIClient:
    """Small wrapper around OpenAI JSON calls with retries and .env loading."""

    def __init__(self, api_key: str | None = None):
        if load_dotenv is not None:
            load_dotenv()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client: Any | None = None
        if self.api_key:
            try:
                from openai import OpenAI

                self._client = OpenAI(api_key=self.api_key)
            except Exception:
                self._client = None

    @property
    def configured(self) -> bool:
        return self._client is not None

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        model: str,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "JsonResponse",
        max_retries: int = 3,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("OpenAI client is not configured")
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response_format: dict[str, Any]
                if json_schema:
                    response_format = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema_name,
                            "schema": make_openai_strict_schema(json_schema),
                            "strict": True,
                        },
                    }
                else:
                    response_format = {"type": "json_object"}
                request_payload: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                    "response_format": response_format,
                }
                if not _uses_default_temperature_only(model):
                    request_payload["temperature"] = temperature
                completion = self._client.chat.completions.create(**request_payload)
                content = completion.choices[0].message.content or "{}"
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(_retry_delay_seconds(exc, attempt))
        raise RuntimeError(f"OpenAI JSON call failed: {last_error}")

    def vision_json(
        self,
        system_prompt: str,
        user_prompt: str,
        image_path: str | Path,
        model: str,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "VisionResponse",
        max_retries: int = 3,
        image_detail: str = "low",
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("OpenAI client is not configured")
        image_bytes = Path(image_path).read_bytes()
        data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": image_detail}},
                ],
            },
        ]
        return self.chat_json(
            messages=messages,
            model=model,
            json_schema=json_schema,
            schema_name=schema_name,
            max_retries=max_retries,
        )

    def vision_json_multi(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str | Path],
        model: str,
        json_schema: dict[str, Any] | None = None,
        schema_name: str = "VisionResponse",
        max_retries: int = 3,
        image_detail: str = "low",
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("OpenAI client is not configured")
        if not image_paths:
            raise RuntimeError("vision_json_multi requires at least one image")
        content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for image_path in image_paths:
            path = Path(image_path)
            if not path.is_file():
                raise RuntimeError(f"vision image does not exist: {path}")
            image_bytes = path.read_bytes()
            data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": data_url, "detail": image_detail}})
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]
        return self.chat_json(
            messages=messages,
            model=model,
            json_schema=json_schema,
            schema_name=schema_name,
            max_retries=max_retries,
        )

    def generate_image(
        self,
        prompt: str,
        output_path: str | Path,
        model: str = "gpt-image-1",
        size: str = "1024x1024",
        quality: str = "low",
        max_retries: int = 3,
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("OpenAI client is not configured")
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                response = self._client.images.generate(
                    model=model,
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    n=1,
                    output_format="png",
                )
                item = response.data[0]
                _write_image_item(item, target)
                return {
                    "model": model,
                    "size": size,
                    "quality": quality,
                    "path": str(target),
                    "operation": "generate",
                    "revised_prompt": getattr(item, "revised_prompt", None),
                }
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(_retry_delay_seconds(exc, attempt, base_delay=0.75))
        raise RuntimeError(f"OpenAI image generation failed: {last_error}")

    def edit_image(
        self,
        image_path: str | Path,
        prompt: str,
        output_path: str | Path,
        model: str = "gpt-image-1",
        size: str = "1024x1024",
        quality: str = "low",
        reference_image_paths: list[str | Path] | None = None,
        mask_path: str | Path | None = None,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        if not self._client:
            raise RuntimeError("OpenAI client is not configured")
        source = Path(image_path)
        if not source.is_file():
            raise RuntimeError(f"OpenAI image edit source does not exist: {source}")
        references = [Path(path) for path in (reference_image_paths or [])]
        missing_references = [str(path) for path in references if not path.is_file()]
        if missing_references:
            raise RuntimeError("OpenAI image edit references do not exist: " + ", ".join(missing_references))
        mask = Path(mask_path) if mask_path else None
        if mask and not mask.is_file():
            raise RuntimeError(f"OpenAI image edit mask does not exist: {mask}")
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                with ExitStack() as stack:
                    image_files = [stack.enter_context(path.open("rb")) for path in [source, *references]]
                    request_payload: dict[str, Any] = {
                        "model": model,
                        "image": image_files if references else image_files[0],
                        "prompt": prompt,
                        "size": size,
                        "quality": quality,
                        "input_fidelity": "high",
                        "n": 1,
                        "output_format": "png",
                    }
                    if mask:
                        request_payload["mask"] = stack.enter_context(mask.open("rb"))
                    response = self._client.images.edit(**request_payload)
                item = response.data[0]
                _write_image_item(item, target)
                return {
                    "model": model,
                    "size": size,
                    "quality": quality,
                    "path": str(target),
                    "operation": "edit",
                    "source_image_path": str(source),
                    "reference_image_paths": [str(path) for path in references],
                    "mask_path": str(mask) if mask else None,
                    "input_fidelity": "high",
                    "revised_prompt": getattr(item, "revised_prompt", None),
                }
            except Exception as exc:
                last_error = exc
                if attempt < max_retries - 1:
                    time.sleep(_retry_delay_seconds(exc, attempt, base_delay=0.75))
        raise RuntimeError(f"OpenAI image edit failed: {last_error}")


def _write_image_item(item: Any, target: Path) -> None:
    b64_json = getattr(item, "b64_json", None)
    if b64_json:
        target.write_bytes(base64.b64decode(b64_json))
        return
    url = getattr(item, "url", None)
    if not url:
        raise RuntimeError("OpenAI image response did not include b64_json or url")
    request = urllib.request.Request(url, headers={"User-Agent": "scenethesis-mvp/0.1"})
    with urllib.request.urlopen(request, timeout=120) as result:
        target.write_bytes(result.read())


def _retry_delay_seconds(exc: Exception, attempt: int, base_delay: float = 0.5) -> float:
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", {}) if response is not None else {}
        retry_after = headers.get("retry-after") if hasattr(headers, "get") else None
    if retry_after is not None:
        try:
            return max(float(retry_after), base_delay * (attempt + 1))
        except (TypeError, ValueError):
            pass
    match = re.search(r"try again in ([0-9]+(?:\.[0-9]+)?)s", str(exc), flags=re.IGNORECASE)
    if match:
        return max(float(match.group(1)) + 0.5, base_delay * (attempt + 1))
    return min(20.0, base_delay * (2 ** attempt))


def _uses_default_temperature_only(model: str) -> bool:
    return model.startswith("gpt-5")


def make_openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert Pydantic JSON schema into the stricter shape OpenAI expects."""

    strict_schema = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            node.pop("default", None)
            if "properties" in node and isinstance(node["properties"], dict):
                node["additionalProperties"] = False
                node["required"] = list(node["properties"].keys())
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(strict_schema)
    return strict_schema
