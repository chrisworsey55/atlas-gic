"""Core schemas: agent output contract and failure types."""
from __future__ import annotations

from datetime import date as Date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field


class Pick(BaseModel):
    ticker: str
    direction: Literal["long", "short", "none"]
    conviction: float = Field(ge=0.0, le=1.0)
    horizon_days: int = Field(gt=0)
    rationale: str = ""


class AgentOutput(BaseModel):
    agent: str
    date: Date
    view: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    picks: list[Pick] = Field(default_factory=list)


class FailureReason(StrEnum):
    MALFORMED_JSON = "malformed_json"
    SCHEMA_INVALID = "schema_invalid"
    EMPTY_RESPONSE = "empty_response"
    TRUNCATED = "truncated"
    REFUSAL = "refusal"
    API_ERROR = "api_error"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"


class FailureRecord(BaseModel):
    agent: str
    date: Date
    reason: FailureReason
    detail: str
    attempt: int
    raw_response: str


class AgentResult(BaseModel):
    agent: str
    date: Date
    output: AgentOutput | None = None
    failure: FailureRecord | None = None

    @property
    def status(self) -> Literal["ok", "no_opinion"]:
        return "no_opinion" if self.failure is not None or self.output is None else "ok"
