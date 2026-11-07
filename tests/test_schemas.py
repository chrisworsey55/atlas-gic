from datetime import date

import pytest
from pydantic import ValidationError

from atlas.schemas import (
    AgentOutput,
    AgentResult,
    FailureReason,
    FailureRecord,
    Pick,
)


def _pick(**kw):
    base = dict(ticker="AAPL", direction="long", conviction=0.7, horizon_days=5, rationale="r")
    base.update(kw)
    return base


def _output(**kw):
    base = dict(agent="a", date="2024-01-02", view="bullish", confidence=0.6, picks=[_pick()])
    base.update(kw)
    return base


def test_valid_output_parses():
    out = AgentOutput.model_validate(_output())
    assert out.date == date(2024, 1, 2)
    assert out.picks[0].ticker == "AAPL"


def test_conviction_above_one_rejected():
    with pytest.raises(ValidationError):
        Pick(**_pick(conviction=1.1))
    with pytest.raises(ValidationError):
        AgentOutput.model_validate(_output(confidence=1.5))


def test_direction_must_be_long_short_none():
    for d in ("long", "short", "none"):
        assert Pick(**_pick(direction=d)).direction == d
    with pytest.raises(ValidationError):
        Pick(**_pick(direction="hold"))


def test_horizon_must_be_positive():
    with pytest.raises(ValidationError):
        Pick(**_pick(horizon_days=0))


def test_failure_reason_values():
    assert {r.value for r in FailureReason} == {
        "malformed_json", "schema_invalid", "empty_response", "truncated",
        "refusal", "api_error", "rate_limited", "timeout",
    }


def test_agent_result_status_no_opinion_when_failure_set():
    out = AgentOutput.model_validate(_output())
    ok = AgentResult(agent="a", date=date(2024, 1, 2), output=out)
    assert ok.status == "ok"
    fail = FailureRecord(
        agent="a", date=date(2024, 1, 2), reason=FailureReason.TIMEOUT,
        detail="d", attempt=1, raw_response="",
    )
    bad = AgentResult(agent="a", date=date(2024, 1, 2), failure=fail)
    assert bad.status == "no_opinion"
