"""Run one agent: call the LLM, validate, retry per policy, log every failed attempt.

Partial/invalid output is recorded in the FailureLog but never returned: a failed agent
yields output=None.
"""
from __future__ import annotations

import json
import re
import time
from datetime import date
from typing import Callable

from pydantic import ValidationError

from atlas.config import AgentSpec
from atlas.engine.failures import CALL_FAILURE, DROPPED_PICK, FailureLog
from atlas.engine.llm import ApiError, CachedLLM, LLMResponse, RateLimited, Timeout
from atlas.schemas import AgentOutput, AgentResult, FailureReason, FailureRecord

MAX_TOKENS = 4096
CONTENT_RETRIES = 1  # malformed_json / schema_invalid / truncated / empty_response
TRANSIENT_MAX_ATTEMPTS = 4  # api_error / rate_limited / timeout, exponential backoff
BACKOFF_BASE_S = 1.0

_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", re.DOTALL)


class _Fail(Exception):
    def __init__(self, reason: FailureReason, detail: str):
        self.reason, self.detail = reason, detail


def _parse(spec: AgentSpec, asof: date, resp: LLMResponse) -> AgentOutput:
    if resp.stop_reason == "refusal":
        raise _Fail(FailureReason.REFUSAL, "model refused")
    if resp.stop_reason == "max_tokens":
        raise _Fail(FailureReason.TRUNCATED, "stop_reason=max_tokens")
    text = resp.text.strip()
    if not text:
        raise _Fail(FailureReason.EMPTY_RESPONSE, "empty response")
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise _Fail(FailureReason.MALFORMED_JSON, f"{e.msg} at line {e.lineno} col {e.colno}") from None
    if not isinstance(data, dict):
        raise _Fail(FailureReason.SCHEMA_INVALID, "top-level JSON is not an object")
    data.setdefault("agent", spec.name)
    data.setdefault("date", asof.isoformat())
    try:
        out = AgentOutput.model_validate(data)
    except ValidationError as e:
        fields = sorted({".".join(str(p) for p in err["loc"]) or "<root>" for err in e.errors()})
        raise _Fail(FailureReason.SCHEMA_INVALID, "invalid fields: " + ", ".join(fields)) from None
    if out.agent != spec.name or out.date != asof:
        raise _Fail(FailureReason.SCHEMA_INVALID, "invalid fields: agent/date mismatch")
    return out


def run_agent(spec: AgentSpec, prompt: str, inputs: dict, asof: date, llm: CachedLLM,
              failures: FailureLog, universe: set[str], *,
              sleep: Callable[[float], None] = time.sleep) -> AgentResult:
    user = json.dumps({"asof": asof.isoformat(), "inputs": inputs}, sort_keys=True, default=str)
    attempt = 0
    content_failures = 0
    transient_failures = 0
    bypass = False

    def log(reason: FailureReason, detail: str, raw: str, *, kind: str = CALL_FAILURE,
            replayed: bool = False) -> FailureRecord:
        rec = FailureRecord(agent=spec.name, date=asof, reason=reason, detail=detail,
                            attempt=attempt, raw_response=raw)
        if not replayed:  # a cache replay was already logged by the run that made the call
            failures.record(rec, kind=kind)
        return rec

    while True:
        attempt += 1
        try:
            resp, replayed = llm.call_with_meta(prompt, user, max_tokens=MAX_TOKENS, bypass_cache=bypass)
        except (RateLimited, Timeout, ApiError) as e:
            reason = (FailureReason.RATE_LIMITED if isinstance(e, RateLimited)
                      else FailureReason.TIMEOUT if isinstance(e, Timeout) else FailureReason.API_ERROR)
            rec = log(reason, str(e), "")
            transient_failures += 1
            if transient_failures >= TRANSIENT_MAX_ATTEMPTS:
                return AgentResult(agent=spec.name, date=asof, failure=rec)
            sleep(BACKOFF_BASE_S * 2 ** (transient_failures - 1))
            continue
        try:
            out = _parse(spec, asof, resp)
        except _Fail as f:
            rec = log(f.reason, f.detail, resp.text, replayed=replayed)
            if f.reason == FailureReason.REFUSAL:
                return AgentResult(agent=spec.name, date=asof, failure=rec)
            content_failures += 1
            if content_failures > CONTENT_RETRIES:
                return AgentResult(agent=spec.name, date=asof, failure=rec)
            bypass = True  # the retry must not replay the cached bad response
            continue
        kept = []
        for p in out.picks:
            if p.ticker in universe:
                kept.append(p)
            else:
                log(FailureReason.SCHEMA_INVALID,
                    f"pick dropped: ticker {p.ticker!r} not in universe", resp.text,
                    kind=DROPPED_PICK, replayed=replayed)
        out = out.model_copy(update={"picks": kept})
        return AgentResult(agent=spec.name, date=asof, output=out)
