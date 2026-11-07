import sqlite3
import time
from types import SimpleNamespace

import pytest

from atlas.engine.llm import (
    AnthropicClient, ApiError, BudgetExceeded, CachedLLM, FatalApiError, LLMResponse, RateLimited,
    StubClient, Timeout,
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

    content = [SimpleNamespace(type="text", text="hi")]

    def create(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return SimpleNamespace(
            content=self.content,
            usage=SimpleNamespace(input_tokens=3, output_tokens=4),
            stop_reason="end_turn",
        )


def test_anthropic_client_sends_no_sampling_params_and_maps_response():
    sdk = FakeSDK()
    r = AnthropicClient("claude-x", sdk=sdk).complete("sys", "usr", 321)
    # current models reject temperature/top_p/top_k (HTTP 400): never sent
    assert not {"temperature", "top_p", "top_k", "output_config", "thinking", "extra_body"} & set(sdk.kwargs)
    assert sdk.kwargs["model"] == "claude-x" and sdk.kwargs["max_tokens"] == 321
    assert sdk.kwargs["system"] == "sys"
    assert sdk.kwargs["messages"] == [{"role": "user", "content": "usr"}]
    assert (r.text, r.input_tokens, r.output_tokens, r.stop_reason) == ("hi", 3, 4, "end_turn")


def test_anthropic_client_sends_effort_only_when_set():
    sdk = FakeSDK()
    c = AnthropicClient("claude-x", sdk=sdk, effort="low")
    c.complete("s", "u", 10)
    assert sdk.kwargs["output_config"] == {"effort": "low"}
    assert "temperature" not in sdk.kwargs
    assert c.request_tag == "effort=low" and AnthropicClient("claude-x", sdk=FakeSDK()).request_tag == ""


def test_anthropic_client_ignores_thinking_blocks():
    sdk = FakeSDK()
    sdk.content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text="{}")]
    assert AnthropicClient("m", sdk=sdk).complete("s", "u", 10).text == "{}"


def _httpx():
    try:
        import httpx
    except ModuleNotFoundError:  # anthropic 2.x vendors httpx2
        import httpx2 as httpx
    return httpx


def test_anthropic_client_maps_errors():
    import anthropic
    httpx = _httpx()

    req = httpx.Request("POST", "http://x")
    cases = [
        (anthropic.RateLimitError("r", response=httpx.Response(429, request=req), body=None), RateLimited),
        (anthropic.APITimeoutError(request=req), Timeout),
        (anthropic.APIConnectionError(request=req), ApiError),
        (anthropic.InternalServerError("s", response=httpx.Response(500, request=req), body=None), ApiError),
        (anthropic.APIStatusError("o", response=httpx.Response(529, request=req), body=None), ApiError),
        (anthropic.APIStatusError("c", response=httpx.Response(409, request=req), body=None), ApiError),
    ]
    for exc, expected in cases:
        with pytest.raises(expected):
            AnthropicClient("m", sdk=FakeSDK(exc)).complete("s", "u", 10)


@pytest.mark.parametrize("cls,status", [
    ("BadRequestError", 400), ("AuthenticationError", 401), ("PermissionDeniedError", 403),
    ("NotFoundError", 404), ("UnprocessableEntityError", 422),
])
def test_anthropic_client_4xx_is_fatal_and_sanitised(cls, status):
    import anthropic
    httpx = _httpx()
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages?key=sk-ant-api03-SECRETSECRET")
    body = {"type": "error", "error": {"type": "invalid_request_error",
                                       "message": "temperature: bad for sk-ant-api03-SECRETSECRET see "
                                                  "https://evil.example/x?api_key=abc"}}
    exc = getattr(anthropic, cls)("raw sk-ant-api03-SECRETSECRET", response=httpx.Response(status, request=req),
                                  body=body)
    with pytest.raises(FatalApiError) as ei:
        AnthropicClient("m", sdk=FakeSDK(exc)).complete("s", "u", 10)
    e = ei.value
    assert not isinstance(e, ApiError)  # never retried as a transient api_error
    assert e.status == status
    text = str(e)
    assert f"HTTP {status}" in text and "Anthropic API rejected the request" in text
    assert "temperature: bad" in text
    assert "SECRET" not in text and "evil.example" not in text and "api_key" not in text
    assert e.__cause__ is None and e.__suppress_context__


def test_cache_key_includes_client_request_tag(tmp_path):
    class Tagged(CountingStub):
        request_tag = "effort=low"

    plain = CountingStub(lambda s, u: "a")
    tagged = Tagged(lambda s, u: "b")
    CachedLLM(plain, tmp_path / "c.db", "m", (1.0, 1.0), 10.0).call("s", "u")
    r = CachedLLM(tagged, tmp_path / "c.db", "m", (1.0, 1.0), 10.0).call("s", "u")
    assert tagged.calls == 1 and r.text == "b"  # a different request setting never replays


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
