import json
from datetime import date

import pytest

from atlas.config import AgentSpec
from atlas.engine.agent import run_agent
from atlas.engine.failures import FailureLog
from atlas.engine.llm import ApiError, BudgetExceeded, CachedLLM, LLMResponse, RateLimited, StubClient, Timeout
from atlas.schemas import FailureReason

ASOF = date(2024, 3, 1)
SPEC = AgentSpec(name="macro", layer=1, kind="macro", prompt_file="macro.md")
UNIVERSE = {"AAPL", "MSFT"}


def good(picks=None):
    return json.dumps({
        "agent": "macro", "date": "2024-03-01", "view": "bullish", "confidence": 0.7,
        "picks": picks if picks is not None else [
            {"ticker": "AAPL", "direction": "long", "conviction": 0.6, "horizon_days": 20}],
    })


class Scripted:
    """Client returning scripted responses/exceptions in order."""

    def __init__(self, items):
        self.items = list(items)
        self.calls = 0

    def complete(self, system, user, max_tokens):
        self.calls += 1
        it = self.items.pop(0)
        if isinstance(it, Exception):
            raise it
        if isinstance(it, str):
            return LLMResponse(it, 100, 50, "end_turn")
        return it


def setup(tmp_path, items, budget=100.0):
    client = Scripted(items)
    llm = CachedLLM(client, tmp_path / "c.db", "m", (1.0, 1.0), budget)
    log = FailureLog(tmp_path / "f.db")
    return client, llm, log


def run(llm, log, sleeps=None, **kw):
    sleeps = sleeps if sleeps is not None else []
    return run_agent(SPEC, "system prompt", {"x": 1}, ASOF, llm, log, UNIVERSE, sleep=sleeps.append, **kw)


def rows(log):
    return log.rows()


def test_success_no_failure_rows(tmp_path):
    client, llm, log = setup(tmp_path, [good()])
    res = run(llm, log)
    assert res.status == "ok" and res.output.picks[0].ticker == "AAPL" and res.failure is None
    assert rows(log).empty
    assert log.summary().empty


def test_stub_client_end_to_end(tmp_path):
    llm = CachedLLM(StubClient(lambda s, u: good()), tmp_path / "c.db", "m", (1.0, 1.0), 10)
    res = run(llm, FailureLog(tmp_path / "f.db"))
    assert res.status == "ok"


def test_malformed_json_retried_once_then_no_opinion_with_raw_stored(tmp_path):
    client, llm, log = setup(tmp_path, ["not json {", "still { bad"])
    res = run(llm, log)
    assert client.calls == 2  # original + exactly one retry
    assert res.status == "no_opinion" and res.output is None
    assert res.failure.reason == FailureReason.MALFORMED_JSON and res.failure.attempt == 2
    r = rows(log)
    assert list(r["attempt"]) == [1, 2]
    assert list(r["raw_response"]) == ["not json {", "still { bad"]
    assert set(r["reason"]) == {"malformed_json"}
    assert set(r["agent"]) == {"macro"} and set(r["date"]) == {"2024-03-01"}


def test_retry_bypasses_cache_and_can_recover(tmp_path):
    client, llm, log = setup(tmp_path, ["garbage", good()])
    res = run(llm, log)
    assert res.status == "ok" and client.calls == 2
    assert list(rows(log)["attempt"]) == [1]
    # rerun: key now holds the good response, no client call
    res2 = run(llm, log)
    assert res2.status == "ok" and client.calls == 2


def test_cached_bad_response_not_replayed_on_retry(tmp_path):
    # first run caches bad; second run replays cache for attempt 1 then must bypass for the retry
    client, llm, log = setup(tmp_path, ["bad", "bad2", good()])
    run(llm, log)
    assert client.calls == 2
    res = run(llm, log)  # attempt 1 from cache ("bad2"), retry bypasses -> good
    assert res.status == "ok" and client.calls == 3


def test_schema_invalid_reason_lists_failing_fields(tmp_path):
    bad = json.dumps({"agent": "macro", "date": "2024-03-01", "view": "sideways", "confidence": 3,
                      "picks": []})
    client, llm, log = setup(tmp_path, [bad, bad])
    res = run(llm, log)
    assert res.failure.reason == FailureReason.SCHEMA_INVALID
    assert "view" in res.failure.detail and "confidence" in res.failure.detail
    assert client.calls == 2
    assert rows(log)["raw_response"].iloc[0] == bad


def test_truncated_stop_reason_maps_to_truncated(tmp_path):
    t = LLMResponse('{"agent": "macro", "da', 10, 4096, "max_tokens")
    client, llm, log = setup(tmp_path, [t, t])
    res = run(llm, log)
    assert res.failure.reason == FailureReason.TRUNCATED and client.calls == 2
    assert res.output is None
    assert list(rows(log)["raw_response"]) == [t.text, t.text]


def test_empty_response_retried_once(tmp_path):
    client, llm, log = setup(tmp_path, ["  ", ""])
    res = run(llm, log)
    assert res.failure.reason == FailureReason.EMPTY_RESPONSE and client.calls == 2


def test_refusal_not_retried(tmp_path):
    r = LLMResponse("I can't help with that", 10, 5, "refusal")
    client, llm, log = setup(tmp_path, [r, good()])
    res = run(llm, log)
    assert client.calls == 1
    assert res.failure.reason == FailureReason.REFUSAL and res.output is None
    assert rows(log)["raw_response"].iloc[0] == "I can't help with that"


def test_rate_limit_uses_backoff_then_gives_up(tmp_path):
    client, llm, log = setup(tmp_path, [RateLimited("429")] * 10)
    sleeps = []
    res = run(llm, log, sleeps)
    assert res.failure.reason == FailureReason.RATE_LIMITED and res.output is None
    n = client.calls
    assert n == 4  # bounded: exactly 4 attempts
    assert sleeps == [1.0, 2.0, 4.0]  # exponential backoff, sleeps only between attempts
    assert list(rows(log)["attempt"]) == list(range(1, n + 1))
    assert set(rows(log)["reason"]) == {"rate_limited"}


def test_transient_error_recovers(tmp_path):
    client, llm, log = setup(tmp_path, [Timeout("t"), ApiError("boom"), good()])
    sleeps = []
    res = run(llm, log, sleeps)
    assert res.status == "ok" and len(sleeps) == 2
    assert list(rows(log)["reason"]) == ["timeout", "api_error"]


def test_partial_json_stored_verbatim_but_result_has_no_output(tmp_path):
    partial = '{"agent": "macro", "date": "2024-03-01", "view": "bullish", "confidence": 0.7, "picks": [{"ticker": "AAPL"'
    t = LLMResponse(partial, 10, 4096, "max_tokens")
    client, llm, log = setup(tmp_path, [t, t])
    res = run(llm, log)
    assert res.output is None and res.status == "no_opinion"
    assert res.failure.raw_response == partial
    assert rows(log)["raw_response"].iloc[-1] == partial


def test_pick_outside_universe_is_dropped_and_logged(tmp_path):
    picks = [
        {"ticker": "AAPL", "direction": "long", "conviction": 0.6, "horizon_days": 20},
        {"ticker": "ZZZZ", "direction": "long", "conviction": 0.9, "horizon_days": 20},
    ]
    client, llm, log = setup(tmp_path, [good(picks)])
    res = run(llm, log)
    assert res.status == "ok" and res.failure is None
    assert [p.ticker for p in res.output.picks] == ["AAPL"]
    r = rows(log)
    assert len(r) == 1 and r["reason"].iloc[0] == "schema_invalid" and "ZZZZ" in r["detail"].iloc[0]
    assert client.calls == 1


def test_all_picks_dropped_still_valid_with_empty_picks(tmp_path):
    picks = [{"ticker": "ZZZZ", "direction": "long", "conviction": 0.9, "horizon_days": 20}]
    client, llm, log = setup(tmp_path, [good(picks)])
    res = run(llm, log)
    assert res.status == "ok" and res.output.picks == []
    assert len(rows(log)) == 1


def test_fenced_json_accepted(tmp_path):
    client, llm, log = setup(tmp_path, ["```json\n" + good() + "\n```"])
    assert run(llm, log).status == "ok"


def test_budget_exceeded_propagates_and_failed_tokens_count(tmp_path):
    # worst case per call ~0.00411 (4096 max output tokens @ $1/Mtok); budget allows one, not two
    client, llm, log = setup(tmp_path, ["bad", good()], budget=0.00415)
    with pytest.raises(BudgetExceeded):
        run(llm, log)
    assert llm.spent_usd == pytest.approx(0.00015)  # the failed attempt's tokens still count
    assert client.calls == 1  # the retry was refused before being made


def test_summary_counts_by_agent_and_reason(tmp_path):
    client, llm, log = setup(tmp_path, ["bad", "bad"])
    run(llm, log)
    s = log.summary()
    assert list(s.columns) == ["agent", "reason", "count"]
    assert s.iloc[0].to_dict() == {"agent": "macro", "reason": "malformed_json", "count": 2}


def test_rerun_of_cached_bad_day_adds_no_replay_rows(tmp_path):
    client, llm, log = setup(tmp_path, ["bad", "bad", "bad again"])
    run(llm, log)
    assert len(rows(log)) == 2 and client.calls == 2
    # re-run: attempt 1 replays the cached bad response (not re-logged); the retry is a fresh
    # live call (bypasses cache) and is the only new row
    res = run(llm, log)
    assert client.calls == 3 and res.status == "no_opinion"
    assert res.failure.raw_response == "bad again" and res.failure.attempt == 2
    assert len(rows(log)) == 3


def test_cached_bad_then_good_retry_adds_no_rows(tmp_path):
    client, llm, log = setup(tmp_path, ["bad", "bad", good()])
    run(llm, log)
    n = len(rows(log))
    assert run(llm, log).status == "ok"
    assert len(rows(log)) == n


def test_dropped_picks_not_in_call_failure_summary(tmp_path):
    picks = [{"ticker": "ZZZZ", "direction": "long", "conviction": 0.9, "horizon_days": 20}]
    client, llm, log = setup(tmp_path, [good(picks)])
    run(llm, log)
    assert log.summary().empty
    d = log.dropped_picks_summary()
    assert d.iloc[0].to_dict() == {"agent": "macro", "reason": "schema_invalid", "count": 1}
    r = rows(log)
    assert list(r["kind"]) == ["dropped_pick"]
    # replaying the cached good day does not re-log the drop
    run(llm, log)
    assert len(rows(log)) == 1


def test_run_id_recorded(tmp_path):
    client = Scripted(["bad", "bad"])
    llm = CachedLLM(client, tmp_path / "c.db", "m", (1.0, 1.0), 100.0)
    log = FailureLog(tmp_path / "f.db", run_id="run-7")
    run(llm, log)
    r = rows(log)
    assert set(r["run_id"]) == {"run-7"} and set(r["kind"]) == {"call_failure"}
