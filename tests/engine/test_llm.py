import sqlite3
import time
from types import SimpleNamespace

import pytest

from atlas.engine.llm import (
    AnthropicClient, ApiError, BudgetExceeded, CachedLLM, LLMResponse, RateLimited, StubClient, Timeout,
)


class CountingStub(StubClient):
    def __init__(self, fn):
        super().__init__(fn)
        self.calls = 0

    def complete(self, system, user, max_tokens):
        self.calls += 1
        return super().complete(system, user, max_tokens)


def make(tmp_path, fn=lambda s, u: "ok", budget=10.0, price=(1.0, 2.0)):
    client = CountingStub(fn)
    return client, CachedLLM(client, tmp_path / "c.db", "m1", price, budget)


def test_cache_hit_makes_no_client_call_and_costs_nothing(tmp_path):
    client, llm = make(tmp_path)
    r1 = llm.call("sys", "user")
    spent = llm.spent_usd
    assert spent > 0 and client.calls == 1
    r2 = llm.call("sys", "user")
    assert client.calls == 1 and llm.spent_usd == spent and r2.text == r1.text
    # persisted across instances
    llm2 = CachedLLM(client, tmp_path / "c.db", "m1", (1.0, 2.0), 10.0)
    llm2.call("sys", "user")
    assert client.calls == 1 and llm2.spent_usd == 0


def test_changed_prompt_misses_cache(tmp_path):
    client, llm = make(tmp_path)
    llm.call("sys", "user")
    llm.call("sys2", "user")
    llm.call("sys", "user2")
    assert client.calls == 3


def test_changed_model_misses_cache(tmp_path):
    client, llm = make(tmp_path)
    llm.call("s", "u")
    other = CachedLLM(client, tmp_path / "c.db", "m2", (1.0, 2.0), 10.0)
    other.call("s", "u")
    assert client.calls == 2


def test_cost_from_token_counts(tmp_path):
    class C:
        def complete(self, system, user, max_tokens):
            return LLMResponse("x", 1_000_000, 500_000, "end_turn")

    llm = CachedLLM(C(), tmp_path / "c.db", "m", (3.0, 15.0), 100.0)
    llm.call("s", "u")
    assert llm.spent_usd == pytest.approx(3.0 + 7.5)


def test_budget_exceeded_raises_before_call_that_would_exceed(tmp_path):
    class C:
        calls = 0

        def complete(self, system, user, max_tokens):
            C.calls += 1
            return LLMResponse("x", 1_000_000, 0, "end_turn")

    llm = CachedLLM(C(), tmp_path / "c.db", "m", (1.0, 1.0), 1.5)
    llm.call("a", "u", max_tokens=10)  # actual spend 1.0, under budget
    # worst case for this call is ~1.0 (1M max output tokens): 1.0 + 1.0 > 1.5 -> refused, NOT made
    with pytest.raises(BudgetExceeded):
        llm.call("b", "u", max_tokens=1_000_000)
    assert C.calls == 1 and llm.spent_usd == pytest.approx(1.0)
    llm.call("a", "u")  # cache hit still free and allowed
    assert C.calls == 1


def test_budget_exhausted_refuses_further_calls(tmp_path):
    client, llm = make(tmp_path, budget=0.0)
    with pytest.raises(BudgetExceeded):
        llm.call("s", "u")
    assert client.calls == 0


def test_concurrent_calls_never_exceed_budget(tmp_path):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    lock = threading.Lock()
    state = {"calls": 0}

    class C:
        def complete(self, system, user, max_tokens):
            with lock:
                state["calls"] += 1
            time.sleep(0.02)  # widen the race window
            return LLMResponse("x", 1000, 0, "end_turn")  # $1 per call at $1000/Mtok

    llm = CachedLLM(C(), tmp_path / "c.db", "m", (1000.0, 1000.0), 3.0)

    def go(i):
        try:
            llm.call(f"{i:02d}" + "x" * 3998, "u", max_tokens=1)  # ~1000 input tokens
            return "ok"
        except BudgetExceeded:
            return "refused"

    with ThreadPoolExecutor(16) as ex:
        results = list(ex.map(go, range(16)))
    assert results.count("ok") == 2 and results.count("refused") == 14  # worst-case reservations: 2 fit under $3, 3rd does not
    assert state["calls"] == 2 and llm.spent_usd <= 3.0
    assert sqlite3.connect(tmp_path / "c.db").execute("select count(*) from llm_cache").fetchone()[0] == 2


def test_bypass_cache_forces_call_and_overwrites_entry(tmp_path):
    seq = iter(["bad", "good"])
    client, llm = make(tmp_path, fn=lambda s, u: next(seq))
    assert llm.call("s", "u").text == "bad"
    assert llm.call("s", "u").text == "bad"  # replayed from cache
    assert client.calls == 1
    assert llm.call("s", "u", bypass_cache=True).text == "good"
    assert client.calls == 2
    assert llm.call("s", "u").text == "good"  # entry overwritten
    assert llm.spent_usd > 0


def test_client_exception_not_cached_and_no_cost(tmp_path):
    class C:
        def complete(self, system, user, max_tokens):
            raise RateLimited("slow down")

    llm = CachedLLM(C(), tmp_path / "c.db", "m", (1.0, 1.0), 5.0)
    with pytest.raises(RateLimited):
        llm.call("s", "u")
    assert llm.spent_usd == 0
    assert sqlite3.connect(tmp_path / "c.db").execute("select count(*) from llm_cache").fetchone()[0] == 0


def test_stub_client_estimates_tokens_and_stop_reason():
    r = StubClient(lambda s, u: "hello").complete("s", "u", 100)
    assert r.text == "hello" and r.stop_reason == "end_turn" and r.input_tokens > 0 and r.output_tokens > 0


class FakeSDK:
    def __init__(self, exc=None):
        self.kwargs = None
        self.exc = exc
        self.messages = self

    def create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text="hi")],
            usage=SimpleNamespace(input_tokens=3, output_tokens=4),
            stop_reason="end_turn",
        )


def test_anthropic_client_fixes_temperature_zero_and_maps_response():
    sdk = FakeSDK()
    r = AnthropicClient("claude-x", sdk=sdk).complete("sys", "usr", 321)
    assert sdk.kwargs["temperature"] == 0
    assert sdk.kwargs["model"] == "claude-x" and sdk.kwargs["max_tokens"] == 321
    assert sdk.kwargs["system"] == "sys"
    assert sdk.kwargs["messages"] == [{"role": "user", "content": "usr"}]
    assert (r.text, r.input_tokens, r.output_tokens, r.stop_reason) == ("hi", 3, 4, "end_turn")


def test_anthropic_client_maps_errors():
    import anthropic
    try:
        import httpx
    except ModuleNotFoundError:  # anthropic 2.x vendors httpx2
        import httpx2 as httpx

    req = httpx.Request("POST", "http://x")
    cases = [
        (anthropic.RateLimitError("r", response=httpx.Response(429, request=req), body=None), RateLimited),
        (anthropic.APITimeoutError(request=req), Timeout),
        (anthropic.APIConnectionError(request=req), ApiError),
        (anthropic.InternalServerError("s", response=httpx.Response(500, request=req), body=None), ApiError),
    ]
    for exc, expected in cases:
        with pytest.raises(expected):
            AnthropicClient("m", sdk=FakeSDK(exc)).complete("s", "u", 10)


def test_initial_spent_counts_toward_budget(tmp_path):
    client = CountingStub(lambda s, u: "ok")
    llm = CachedLLM(client, tmp_path / "c.db", "m1", (1.0, 2.0), 10.0, initial_spent_usd=9.999)
    assert llm.spent_usd == 9.999
    with pytest.raises(BudgetExceeded):
        llm.call("s", "u")
    assert client.calls == 0


def test_restore_spent_never_lowers_spend(tmp_path):
    client, llm = make(tmp_path)
    llm.restore_spent(3.0)
    assert llm.spent_usd == 3.0
    llm.restore_spent(1.0)  # an older checkpoint must not erase money already spent
    assert llm.spent_usd == 3.0
    with pytest.raises(ValueError):
        llm.restore_spent(float("nan"))
