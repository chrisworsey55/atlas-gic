"""Shared Budget: one hard spend cap across several CachedLLM instances (one per model)."""
import math
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from atlas.engine.llm import Budget, BudgetExceeded, CachedLLM, LLMResponse, RateLimited, StubClient

# Prompt sizes chosen so every call reserves just over $1 and actually costs exactly $1:
#   model "a" at $1000/Mtok: (4000 + 1) // 4 + 1 = 1001 input tokens + 1 output -> $1.002 reserved,
#     1000 tokens actually billed -> $1.000;
#   model "b" at $2000/Mtok: (2000 + 1) // 4 + 1 = 501 input tokens + 1 output -> $1.004 reserved,
#     500 tokens actually billed -> $1.000.
PRICE_A, PRICE_B = (1000.0, 1000.0), (2000.0, 2000.0)


class SleepyClient:
    """Bills a fixed number of input tokens; sleeps to keep calls in flight together."""

    def __init__(self, in_tokens: int, sleep: float = 0.02, counter=None):
        self.in_tokens, self.sleep = in_tokens, sleep
        self.calls = 0
        self._lock = threading.Lock()
        self.counter = counter

    def complete(self, system, user, max_tokens):
        with self._lock:
            self.calls += 1
        if self.counter is not None:
            self.counter()
        time.sleep(self.sleep)
        return LLMResponse("x", self.in_tokens, 0, "end_turn")


def two_models(tmp_path, cap, sleep=0.02):
    budget = Budget(cap)
    ca, cb = SleepyClient(1000, sleep), SleepyClient(500, sleep)
    a = CachedLLM(ca, tmp_path / "c.db", "a", PRICE_A, 0.0, budget=budget)
    b = CachedLLM(cb, tmp_path / "c.db", "b", PRICE_B, 0.0, budget=budget)
    return budget, (ca, a), (cb, b)


def prompt(model: str, i: int) -> str:
    return f"{i:02d}" + "x" * (3998 if model == "a" else 1998)


def test_two_models_share_one_cap_under_32_threads(tmp_path):
    budget, (ca, a), (cb, b) = two_models(tmp_path, cap=5.5)
    barrier = threading.Barrier(32)

    def go(i):
        llm, name = (a, "a") if i % 2 == 0 else (b, "b")
        barrier.wait()
        try:
            llm.call(prompt(name, i), "u", max_tokens=1)
            return "ok"
        except BudgetExceeded:
            return "refused"

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as possible to force interleaving
    try:
        with ThreadPoolExecutor(32) as ex:
            results = list(ex.map(go, range(32)))
    finally:
        sys.setswitchinterval(old)
    # every call reserves ~$1.00x: five fit under $5.5 whatever the order or model, the sixth never does
    assert results.count("ok") == 5 and results.count("refused") == 27
    assert ca.calls + cb.calls == 5  # refused calls never reached a client
    assert budget.spent_usd == pytest.approx(5.0) and budget.spent_usd <= budget.cap_usd
    assert budget.reserved_usd == pytest.approx(0.0, abs=1e-9)
    assert a.spent_usd == b.spent_usd == budget.spent_usd  # both views read the one counter
    by = budget.spent_by_model()
    assert set(by) <= {"a", "b"} and sum(by.values()) == pytest.approx(5.0)
    assert by.get("a", 0) == pytest.approx(ca.calls) and by.get("b", 0) == pytest.approx(cb.calls)
    assert sqlite3.connect(tmp_path / "c.db").execute("select count(*) from llm_cache").fetchone()[0] == 5


@pytest.mark.parametrize("rep", range(3))
def test_shared_cap_race_without_barrier(tmp_path, rep):
    """Same race with staggered starts and very short sleeps (another interleaving pattern)."""
    budget, (ca, a), (cb, b) = two_models(tmp_path, cap=7.5, sleep=0.001)

    def go(i):
        llm, name = (a, "a") if i % 3 else (b, "b")
        try:
            llm.call(prompt(name, i), "u", max_tokens=1)
            return 1
        except BudgetExceeded:
            return 0

    with ThreadPoolExecutor(32) as ex:
        ok = sum(ex.map(go, range(64)))
    # no call fails, so each admitted call ends up costing $1: exactly 7 fit under $7.5, never 8
    assert ok == 7 and ca.calls + cb.calls == 7
    assert budget.spent_usd == pytest.approx(7.0) and budget.spent_usd <= 7.5


def test_spent_by_model_sums_to_total(tmp_path):
    budget, (ca, a), (cb, b) = two_models(tmp_path, cap=100.0, sleep=0)
    a.call(prompt("a", 1), "u", max_tokens=1)
    b.call(prompt("b", 2), "u", max_tokens=1)
    b.call(prompt("b", 3), "u", max_tokens=1)
    a.call(prompt("a", 1), "u", max_tokens=1)  # cache hit: free, not attributed
    by = budget.spent_by_model()
    assert by == {"a": pytest.approx(1.0), "b": pytest.approx(2.0)}
    assert sum(by.values()) == pytest.approx(budget.spent_usd)
    by["a"] = 99.0  # a copy: mutating it never changes the budget
    assert budget.spent_by_model()["a"] == pytest.approx(1.0)


def test_restored_spend_without_attribution_has_its_own_bucket():
    budget = Budget(100.0, initial_spent_usd=4.0)
    assert budget.spent_by_model() == {Budget.UNATTRIBUTED: 4.0}
    budget.reserve(1.5)
    budget.settle(1.5, "a", 1.0)
    budget.restore_spent(7.0)  # raises total by 2.0 beyond what is attributed
    by = budget.spent_by_model()
    assert by == {Budget.UNATTRIBUTED: pytest.approx(6.0), "a": pytest.approx(1.0)}
    assert sum(by.values()) == pytest.approx(budget.spent_usd) == pytest.approx(7.0)
    assert Budget(1.0).spent_by_model() == {}  # nothing restored: no bucket


def test_restore_spent_once_on_shared_budget(tmp_path):
    budget = Budget(10.0, initial_spent_usd=3.0)
    a = CachedLLM(StubClient(lambda s, u: "x"), tmp_path / "c.db", "a", (1.0, 1.0), 0.0,
                  initial_spent_usd=3.0, budget=budget)
    b = CachedLLM(StubClient(lambda s, u: "x"), tmp_path / "c.db", "b", (1.0, 1.0), 0.0,
                  initial_spent_usd=3.0, budget=budget)
    a.restore_spent(3.0)
    b.restore_spent(3.0)  # the same checkpoint restored through each model: counted once
    assert budget.spent_usd == a.spent_usd == b.spent_usd == 3.0
    b.restore_spent(1.0)  # raise-only: never lowers
    assert budget.spent_usd == 3.0
    b.restore_spent(4.0)
    assert a.spent_usd == 4.0
    assert budget.spent_by_model() == {Budget.UNATTRIBUTED: 4.0}
    for bad in (float("nan"), float("inf"), -1.0):
        with pytest.raises(ValueError):
            a.restore_spent(bad)
        with pytest.raises(ValueError):
            Budget(1.0, initial_spent_usd=bad)
    assert budget.spent_usd == 4.0


def test_pre_call_reservation_uses_each_models_own_prices(tmp_path):
    seen = {}
    budget = Budget(1.5)

    class Peek:
        def __init__(self, name):
            self.name = name

        def complete(self, system, user, max_tokens):
            seen[self.name] = budget.reserved_usd
            # the budget lock is NOT held during the network call
            assert budget._lock.acquire(blocking=False)
            budget._lock.release()
            return LLMResponse("x", 0, 0, "end_turn")

    cheap = CachedLLM(Peek("cheap"), tmp_path / "c.db", "cheap", (1.0, 1.0), 0.0, budget=budget)
    dear = CachedLLM(Peek("dear"), tmp_path / "c.db", "dear", (1000.0, 1000.0), 0.0, budget=budget)
    s = "x" * 3999  # (3999 + 1) // 4 + 1 = 1001 input tokens
    cheap.call(s, "u", max_tokens=999)  # 2000 tokens at $1/Mtok = $0.002
    assert seen["cheap"] == pytest.approx(0.002)
    with pytest.raises(BudgetExceeded) as ei:  # same prompt at $1000/Mtok = $2.00 > $1.5
        dear.call(s, "u", max_tokens=999)
    assert "dear" not in seen  # refused before any network call
    assert str(ei.value) == ("call could cost up to $2.0000; spent $0.0000 (+$0.0000 in flight) "
                             "of $1.50 budget")
    dear.call(s, "u", max_tokens=1)  # 1002 tokens = $1.002 fits
    assert seen["dear"] == pytest.approx(1.002)
    assert budget.reserved_usd == 0.0


def test_failed_call_releases_reservation_on_shared_budget(tmp_path):
    budget = Budget(1.5)  # room for exactly one ~$1 reservation

    class Failing:
        calls = 0

        def complete(self, system, user, max_tokens):
            Failing.calls += 1
            raise RateLimited("slow down")

    ok_client = SleepyClient(500, sleep=0)
    bad = CachedLLM(Failing(), tmp_path / "c.db", "a", PRICE_A, 0.0, budget=budget)
    good = CachedLLM(ok_client, tmp_path / "c.db", "b", PRICE_B, 0.0, budget=budget)
    for i in range(3):  # a leaked reservation would block the next call
        with pytest.raises(RateLimited):
            bad.call(prompt("a", i), "u", max_tokens=1)
        assert budget.reserved_usd == 0.0 and budget.spent_usd == 0.0
    assert Failing.calls == 3
    good.call(prompt("b", 9), "u", max_tokens=1)  # the full cap is still available to the other model
    assert ok_client.calls == 1 and budget.spent_usd == pytest.approx(1.0)
    assert budget.spent_by_model() == {"b": pytest.approx(1.0)}  # failed calls attribute nothing
    assert sqlite3.connect(tmp_path / "c.db").execute("select count(*) from llm_cache").fetchone()[0] == 1


def test_reserve_and_settle_directly():
    b = Budget(2.0)
    b.reserve(1.5)
    assert b.reserved_usd == 1.5
    with pytest.raises(BudgetExceeded) as ei:
        b.reserve(0.6)
    assert str(ei.value) == "call could cost up to $0.6000; spent $0.0000 (+$1.5000 in flight) of $2.00 budget"
    assert b.reserved_usd == 1.5  # a refused reservation reserves nothing
    b.settle(1.5, "m", None)  # no usage information (the call failed): release only
    assert (b.reserved_usd, b.spent_usd, b.spent_by_model()) == (0.0, 0.0, {})
    b.reserve(2.0)  # exactly at the cap is allowed (strict >)
    b.settle(2.0, "m", 2.5)  # actual can exceed the estimate; the post-call check reports it
    assert b.spent_usd == 2.5 and b.spent_by_model() == {"m": 2.5} and b.reserved_usd == 0.0
    with pytest.raises(BudgetExceeded):
        b.reserve(0.0)


def test_post_call_overrun_on_shared_budget_raises_after_caching(tmp_path):
    budget = Budget(1.0)

    class Big:
        def complete(self, system, user, max_tokens):
            return LLMResponse("x", 2_000_000, 0, "end_turn")  # far above the estimate

    llm = CachedLLM(Big(), tmp_path / "c.db", "a", (1.0, 1.0), 0.0, budget=budget)
    with pytest.raises(BudgetExceeded, match=r"spent \$2\.0000 exceeds \$1\.00 budget"):
        llm.call("s", "u", max_tokens=10)
    assert budget.spent_usd == pytest.approx(2.0) and budget.reserved_usd == 0.0
    assert llm.call("s", "u").text == "x"  # cached: replays free


def test_single_model_behaviour_identical_to_before(tmp_path):
    """Without ``budget=`` each CachedLLM owns a private Budget built from budget_usd and
    initial_spent_usd, as before; budget_usd stays readable and settable (the CLI narrows the
    cap of an existing instance). The rest of the oracle is tests/engine/test_llm.py."""
    c1, c2 = StubClient(lambda s, u: "ok"), StubClient(lambda s, u: "ok")
    one = CachedLLM(c1, tmp_path / "c.db", "m", (1.0, 2.0), 10.0, initial_spent_usd=1.0)
    two = CachedLLM(c2, tmp_path / "c.db", "m2", (1.0, 2.0), 5.0)
    assert one.budget is not two.budget
    assert (one.budget_usd, one.spent_usd, two.budget_usd, two.spent_usd) == (10.0, 1.0, 5.0, 0.0)
    one.call("s", "u")
    assert two.spent_usd == 0.0 and one.spent_usd > 1.0
    one.budget_usd = 0.5
    assert one.budget.cap_usd == 0.5
    with pytest.raises(BudgetExceeded):
        one.call("s2", "u")
    # a shared budget ignores the budget_usd argument; setting budget_usd sets the shared cap
    shared = Budget(3.0)
    x = CachedLLM(c1, tmp_path / "c.db", "x", (1.0, 1.0), 999.0, budget=shared)
    y = CachedLLM(c2, tmp_path / "c.db", "y", (1.0, 1.0), 0.0, budget=shared)
    assert x.budget is y.budget is shared and x.budget_usd == y.budget_usd == 3.0
    y.budget_usd = 2.0
    assert x.budget_usd == 2.0 == shared.cap_usd


def test_reservation_released_even_if_costing_the_response_fails(tmp_path):
    budget = Budget(5.0)

    class Weird:
        def complete(self, system, user, max_tokens):
            return LLMResponse("x", "not-a-number", 0, "end_turn")  # type: ignore[arg-type]

    llm = CachedLLM(Weird(), tmp_path / "c.db", "a", (1.0, 1.0), 0.0, budget=budget)
    with pytest.raises(TypeError):
        llm.call("s", "u", max_tokens=10)
    assert budget.reserved_usd == 0.0 and budget.spent_usd == 0.0


def test_shared_budget_restores_initial_spent_of_each_instance(tmp_path):
    budget = Budget(10.0)
    CachedLLM(StubClient(lambda s, u: "x"), tmp_path / "c.db", "a", (1.0, 1.0),
              initial_spent_usd=2.0, budget=budget)
    assert budget.spent_usd == 2.0
    CachedLLM(StubClient(lambda s, u: "x"), tmp_path / "c.db", "b", (1.0, 1.0),
              initial_spent_usd=2.0, budget=budget)
    assert budget.spent_usd == 2.0  # raise-only: the same total through a second model counts once


class SlowFloat(float):
    """A float whose arithmetic with a plain float sleeps (releasing the GIL) before
    answering, so a read-modify-write of the budget counters that is not done under the
    budget lock interleaves with other threads and loses or over-admits updates."""

    def _slow(self, op, other):
        time.sleep(0.002)
        return op(float(other), float(self))

    def __radd__(self, other):
        return self._slow(float.__add__, other)

    def __rsub__(self, other):
        return self._slow(float.__sub__, other)


def test_reserve_check_and_add_are_one_atomic_step():
    budget = Budget(5.5)
    barrier = threading.Barrier(32)

    def go(_):
        barrier.wait()
        try:
            budget.reserve(SlowFloat(1.0))
            return 1
        except BudgetExceeded:
            return 0

    with ThreadPoolExecutor(32) as ex:
        ok = sum(ex.map(go, range(32)))
    assert ok == 5 and budget.reserved_usd == 5.0


def test_concurrent_settles_lose_no_update():
    budget = Budget(100.0)
    for _ in range(16):
        budget.reserve(1.0)
    barrier = threading.Barrier(16)

    def go(i):
        barrier.wait()
        budget.settle(SlowFloat(1.0), "a" if i % 2 else "b", SlowFloat(0.5))

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(go, range(16)))
    assert budget.reserved_usd == 0.0 and budget.spent_usd == 8.0
    assert budget.spent_by_model() == {"a": 4.0, "b": 4.0}


def test_every_counter_update_takes_the_budget_lock():
    """White-box pin for updates whose race window is too small to force from outside."""
    budget = Budget(10.0)
    real = budget._lock
    entered = []

    class Recording:
        def __enter__(self):
            real.acquire()
            entered.append(1)
            return self

        def __exit__(self, *exc):
            real.release()

    budget._lock = Recording()
    for op in (lambda: budget.restore_spent(1.0), lambda: budget.reserve(1.0),
               lambda: budget.settle(1.0, "m", 0.5), budget.spent_by_model):
        before = len(entered)
        op()
        assert len(entered) == before + 1


@pytest.mark.parametrize("bad", [float("nan"), "nan"])
def test_nan_cap_is_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="budget cap"):
        Budget(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="budget cap"):
        CachedLLM(StubClient(lambda s, u: "x"), tmp_path / "c.db", "m", (1.0, 1.0), bad)
    llm = CachedLLM(StubClient(lambda s, u: "x"), tmp_path / "c.db", "m", (1.0, 1.0), 5.0)
    with pytest.raises(ValueError, match="budget cap"):
        llm.budget_usd = bad
    assert llm.budget_usd == 5.0  # a rejected cap leaves the old one in place
    shared = Budget(2.0)
    with pytest.raises(ValueError, match="budget cap"):
        shared.cap_usd = bad
    assert shared.cap_usd == 2.0


def test_infinite_zero_and_negative_caps_still_allowed(tmp_path):
    assert Budget(math.inf).cap_usd == math.inf  # dry-run stubs use an unlimited cap
    assert Budget(0.0).cap_usd == 0.0 and Budget(-1.0).cap_usd == -1.0  # refuse every call, as before
    llm = CachedLLM(StubClient(lambda s, u: "x"), tmp_path / "c.db", "m", (1.0, 1.0), math.inf)
    llm.call("s", "u")
    llm.budget_usd = 0.0
    with pytest.raises(BudgetExceeded):
        llm.call("s2", "u")


def test_spend_exactly_at_cap_after_a_call_does_not_raise(tmp_path):
    class Exact:
        def complete(self, system, user, max_tokens):
            return LLMResponse("x", 1, 0, "end_turn")  # $1.00 at $1e6/Mtok

    budget = Budget(1.0)
    llm = CachedLLM(Exact(), tmp_path / "c.db", "a", (1e6, 0.0), budget=budget)
    resp = llm.call("", "", max_tokens=1)  # reserves (0 // 4 + 1) * $1 = $1.00 == cap: allowed
    assert resp.text == "x" and budget.spent_usd == 1.0 == budget.cap_usd  # spend == cap: no raise


def test_setting_budget_usd_on_one_instance_sets_the_shared_cap(tmp_path):
    budget = Budget(10.0)
    ca, cb = SleepyClient(1000, sleep=0), SleepyClient(500, sleep=0)
    a = CachedLLM(ca, tmp_path / "c.db", "a", PRICE_A, budget=budget)
    b = CachedLLM(cb, tmp_path / "c.db", "b", PRICE_B, budget=budget)
    a.budget_usd = 0.5  # narrowing the cap through A narrows it for B too
    assert b.budget_usd == 0.5 == budget.cap_usd
    with pytest.raises(BudgetExceeded, match=r"of \$0\.50 budget"):
        b.call(prompt("b", 1), "u", max_tokens=1)
    assert cb.calls == 0
    b.budget_usd = 3.0
    assert a.budget_usd == 3.0
    a.call(prompt("a", 2), "u", max_tokens=1)
    assert ca.calls == 1


def test_nan_estimate_or_cost_fails_closed(tmp_path):
    b = Budget(10.0)
    with pytest.raises(BudgetExceeded):
        b.reserve(float("nan"))
    assert b.reserved_usd == 0.0

    class Nan:
        def complete(self, system, user, max_tokens):
            return LLMResponse("x", 1, 0, "end_turn")

    llm = CachedLLM(Nan(), tmp_path / "c.db", "a", (1.0, 0.0), budget=b)
    llm._cost = lambda r: float("nan")  # a NaN cost must not slip past the post-call check
    with pytest.raises(BudgetExceeded):
        llm.call("s", "u", max_tokens=1)
