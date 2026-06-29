from __future__ import annotations

import re


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value)
    if not cleaned:
        cleaned = "mesh"
    if cleaned[0].isdigit():
        cleaned = "m_" + cleaned
    return cleaned
