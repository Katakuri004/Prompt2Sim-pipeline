from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class QualificationCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    ok: bool
    detail: str


class QualificationReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["accepted", "unqualified"]
    accepted: bool
    stage: str
    checks: list[QualificationCheck] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
