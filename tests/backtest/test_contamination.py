"""Contamination probe (optional): can the model name the date / ticker of anonymised data?"""
import json
import math
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from atlas.backtest.contamination import ContaminationScore, contamination_probe
from atlas.backtest.runner import anonymize_snapshot
from atlas.data.provider import PRICE_COLUMNS
from atlas.data.store import Snapshot
from atlas.engine.failures import FailureLog
from atlas.engine.llm import ApiError, BudgetExceeded, CachedLLM, StubClient

ITEMS = [(date(2023, 3, 10), "NVDA"), (date(2023, 6, 20), "XOM"), (date(2023, 9, 5), "JPM")]
SECTORS = {"NVDA": "Semiconductors", "XOM": "Energy", "JPM": "Financials"}


def snap_for(asof: date, ticker: str) -> Snapshot:
    rng = np.random.default_rng(asof.toordinal())
    days = [asof - timedelta(days=i) for i in range(6, -1, -1)]
    px = 100 * np.cumprod(1 + rng.normal(0, 0.01, len(days)))
    df = pd.DataFrame({"date": days, "ticker": ticker, "open": px, "high": px, "low": px,
                       "close": px, "adj_close": px, "volume": 1e6})[PRICE_COLUMNS]
    return Snapshot(asof=asof, prices=df, macro={"VIX": pd.Series([20.0, 21.0], index=[asof - timedelta(days=3), asof])})


def probes(items=ITEMS):
    out = []
    for d, t in items:
        anon, _ = anonymize_snapshot(snap_for(d, t), {t: SECTORS[t]})
        out.append((anon, d, t))
    return out


def make_llm(tmp_path, fn, budget=100.0, name="c.db"):
    return CachedLLM(StubClient(fn), tmp_path / name, "m", (1.0, 1.0), budget)


def reply(date_s="1999-01-01", ticker="ZZZZ"):
    return json.dumps({"date": date_s, "ticker": ticker})


def test_probe_scores_exact_date_guess_as_hit(tmp_path):
    llm = make_llm(tmp_path, lambda s, u: reply("2023-03-10", "NOPE"))
    sc = contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5)
    assert (sc.n, sc.date_hits, sc.ticker_hits, sc.rate) == (1, 1, 0, 1.0)


def test_probe_scores_guess_outside_tolerance_as_miss(tmp_path):
    llm = make_llm(tmp_path, lambda s, u: reply("2023-03-30", "NOPE"))
    sc = contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5)
    assert (sc.n, sc.date_hits, sc.ticker_hits, sc.rate) == (1, 0, 0, 0.0)


@pytest.mark.parametrize("guess,hit", [("2023-03-15", True), ("2023-03-05", True),
                                        ("2023-03-16", False), ("2023-03-04", False)])
def test_probe_tolerance_boundary(tmp_path, guess, hit):
    llm = make_llm(tmp_path, lambda s, u: reply(guess))
    sc = contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5)
    assert sc.date_hits == int(hit) and sc.rate == float(hit)


def test_probe_ticker_hit_is_case_and_space_insensitive(tmp_path):
    llm = make_llm(tmp_path, lambda s, u: reply("2000-01-01", " nvda "))
    sc = contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5)
    assert (sc.date_hits, sc.ticker_hits, sc.rate) == (0, 1, 1.0)


def test_probe_rate_counts_either_hit_once(tmp_path):
    # probe 1: both right; probe 2: date only; probe 3: neither -> rate 2/3, not 3/3
    by_day = {"2023-03-10": ("2023-03-10", "NVDA"), "2023-06-20": ("2023-06-20", "??")}

    calls = iter([by_day["2023-03-10"], by_day["2023-06-20"], ("2000-01-01", "??")])
    llm = make_llm(tmp_path, lambda s, u: reply(*next(calls)))
    sc = contamination_probe(probes(), llm, tolerance_days=5)
    assert (sc.n, sc.date_hits, sc.ticker_hits) == (3, 2, 1)
    assert sc.any_hits == 2 and sc.rate == pytest.approx(2 / 3)


def test_probe_rate_zero_when_stub_always_wrong(tmp_path):
    llm = make_llm(tmp_path, lambda s, u: reply("1990-01-01", "ZZZZ"))
    sc = contamination_probe(probes(), llm, tolerance_days=5)
    assert (sc.n, sc.date_hits, sc.ticker_hits, sc.rate) == (3, 0, 0, 0.0)
    assert not sc.empty


def test_probe_empty_input_is_nan_safe_and_flagged(tmp_path):
    llm = make_llm(tmp_path, lambda s, u: reply())
    sc = contamination_probe([], llm, tolerance_days=5)
    assert sc.n == 0 and sc.rate == 0.0 and not math.isnan(sc.rate) and sc.empty


def test_probe_uses_anonymised_snapshots_only(tmp_path):
    seen = []

    def fn(system, user):
        seen.append((system, user))
        return reply()

    llm = make_llm(tmp_path, fn)
    contamination_probe(probes(), llm, tolerance_days=5)
    assert len(seen) == 3
    real_tickers = [t for _, t in ITEMS]
    date_forms = [d.isoformat() for d, _ in ITEMS] + [d.strftime("%Y%m%d") for d, _ in ITEMS]
    for system, user in seen:
        for msg in (system, user):
            for t in real_tickers:
                assert not re.search(rf"\b{t}\b", msg, re.I)
            for ds in date_forms:
                assert ds not in msg
            assert not re.search(r"\d{4}-\d{2}-\d{2}", msg), "no calendar date in any message"
        # what the model does get: opaque ids and sector labels
        assert re.search(r"\bA\d\d\b", user)
        assert "Semiconductors" in user or "Energy" in user or "Financials" in user
    assert "YYYY-MM-DD" in seen[0][0]  # the format is requested, never an example date


def test_probe_asks_for_json_with_date_and_ticker(tmp_path):
    seen = []
    llm = make_llm(tmp_path, lambda s, u: (seen.append(s), reply())[1])
    contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5)
    assert '"date"' in seen[0] and '"ticker"' in seen[0] and "JSON" in seen[0]


@pytest.mark.parametrize("text,reason", [
    ("not json at all", "malformed_json"),
    ("", "empty_response"),
    ('["2023-03-10", "NVDA"]', "schema_invalid"),
    ('{"date": "March 2023", "ticker": "NVDA"}', "schema_invalid"),
    ('{"date": "2023-03-10"}', "schema_invalid"),
])
def test_probe_bad_reply_is_a_miss_logged_with_reason_and_never_used(tmp_path, text, reason):
    fl = FailureLog(tmp_path / "f.db")
    llm = make_llm(tmp_path, lambda s, u: text)
    sc = contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5, failures=fl)
    assert (sc.n, sc.rate, sc.failed) == (1, 0.0, 1)
    rows = fl.rows()
    assert list(rows["reason"]) == [reason] and list(rows["agent"]) == ["contamination_probe"]
    assert rows["raw_response"].iloc[0] == text  # partial output stored, verbatim


def test_probe_partial_output_with_a_correct_date_is_not_used(tmp_path):
    # a valid date but a missing ticker is schema-invalid: the date must NOT be credited
    llm = make_llm(tmp_path, lambda s, u: '{"date": "2023-03-10"}')
    sc = contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5)
    assert sc.date_hits == 0 and sc.rate == 0.0


def test_probe_fenced_json_is_accepted(tmp_path):
    llm = make_llm(tmp_path, lambda s, u: "```json\n" + reply("2023-03-10", "NVDA") + "\n```")
    sc = contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5)
    assert sc.rate == 1.0 and sc.failed == 0


def test_probe_refusal_is_a_miss_logged_as_refusal(tmp_path):
    from atlas.engine.llm import LLMResponse

    class Refuser:
        def complete(self, system, user, max_tokens):
            return LLMResponse("I can't", 10, 5, "refusal")

    fl = FailureLog(tmp_path / "f.db")
    llm = CachedLLM(Refuser(), tmp_path / "c.db", "m", (1.0, 1.0), 100)
    sc = contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5, failures=fl)
    assert sc.failed == 1 and list(fl.rows()["reason"]) == ["refusal"]


def test_probe_api_errors_retry_then_count_as_logged_miss(tmp_path):
    n = {"c": 0}

    def fn(s, u):
        n["c"] += 1
        raise ApiError("boom")

    fl = FailureLog(tmp_path / "f.db")
    sleeps = []
    llm = make_llm(tmp_path, fn)
    sc = contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5, failures=fl, sleep=sleeps.append)
    assert n["c"] == 4 and len(sleeps) == 3
    assert sc.failed == 1 and sc.n == 1 and set(fl.rows()["reason"]) == {"api_error"}


def test_probe_uses_cache_so_a_rerun_makes_no_calls(tmp_path):
    n = {"c": 0}

    def fn(s, u):
        n["c"] += 1
        return reply("2023-03-10", "NVDA")

    llm = make_llm(tmp_path, fn)
    a = contamination_probe(probes(), llm, tolerance_days=5)
    calls = n["c"]
    b = contamination_probe(probes(), llm, tolerance_days=5)
    assert n["c"] == calls and a == b


def test_probe_budget_exceeded_returns_partial_score(tmp_path):
    # price input tokens at $1 each (output free) so the cap is an exact number of tokens
    sizes = []
    llm0 = CachedLLM(StubClient(lambda s, u: (sizes.append((len(s) + len(u)) // 4), reply("2023-03-10", "NVDA"))[1]),
                     tmp_path / "c0.db", "m", (1e6, 0.0), 1e9)
    contamination_probe(probes(), llm0, tolerance_days=5)
    assert len(sizes) == 3
    llm = CachedLLM(StubClient(lambda s, u: reply("2023-03-10", "NVDA")), tmp_path / "c.db", "m",
                    (1e6, 0.0), 1.5 * max(sizes) + 2)  # room for one probe, not two
    sc = contamination_probe(probes(), llm, tolerance_days=5)
    assert sc.budget_exhausted is True
    assert sc.n == 1 and sc.date_hits == 1 and sc.rate == 1.0  # n counts completed probes only


def test_probe_budget_zero_gives_n_zero(tmp_path):
    llm = make_llm(tmp_path, lambda s, u: reply(), budget=0.0)
    sc = contamination_probe(probes(), llm, tolerance_days=5)
    assert sc.n == 0 and sc.budget_exhausted and sc.empty and sc.rate == 0.0


def test_probe_does_not_swallow_unrelated_errors(tmp_path):
    def fn(s, u):
        raise RuntimeError("bug")

    llm = make_llm(tmp_path, fn)
    with pytest.raises(RuntimeError):
        contamination_probe(probes(ITEMS[:1]), llm, tolerance_days=5)


def test_score_is_a_plain_dataclass_with_the_binding_fields():
    sc = ContaminationScore(n=4, date_hits=1, ticker_hits=2, rate=0.5)
    assert (sc.n, sc.date_hits, sc.ticker_hits, sc.rate) == (4, 1, 2, 0.5)


# ---------------------------------------------------------------------------- report section

def _report(tmp_path, **kw):
    from tests.backtest.test_report import build
    return build(tmp_path, **kw).read_text()


def test_report_includes_contamination_section_only_when_score_given(tmp_path):
    without = _report(tmp_path / "a")
    assert "Contamination probe" not in without
    sc = ContaminationScore(n=20, date_hits=6, ticker_hits=2, rate=0.35, any_hits=7)
    with_ = _report(tmp_path / "b", contamination=sc)
    assert "\n## Contamination probe (optional)\n" in with_
    sec = with_.split("## Contamination probe (optional)")[1].split("\n## ")[0]
    assert "35.00%" in sec and "20" in sec and "7 of 20" in sec
    assert "trust results for those periods less" in sec
    assert "not a measure of skill" in sec.lower() or "not evidence of skill" in sec.lower()


def test_report_contamination_accepts_a_dict_and_shows_tolerance(tmp_path):
    md = _report(tmp_path, contamination={"n": 10, "date_hits": 1, "ticker_hits": 0, "rate": 0.1,
                                          "tolerance_days": 7})
    sec = md.split("## Contamination probe (optional)")[1].split("\n## ")[0]
    assert "10.00%" in sec and "within 7 day" in sec


def test_report_contamination_empty_or_early_stop_is_stated_and_does_not_crash(tmp_path):
    md = _report(tmp_path / "a", contamination=ContaminationScore(0, 0, 0, 0.0))
    assert "no probes completed" in md.split("## Contamination probe (optional)")[1].lower()
    md = _report(tmp_path / "b", contamination=ContaminationScore(3, 0, 0, 0.0, budget_exhausted=True))
    assert "budget" in md.split("## Contamination probe (optional)")[1].lower()
    md = _report(tmp_path / "c", contamination={"garbage": 1})
    assert "unreadable" in md.split("## Contamination probe (optional)")[1].lower()


def test_report_verdict_is_unchanged_by_the_contamination_score(tmp_path):
    def verdict(md):
        return md.split("## Verdict")[1].split("\n## ")[0]
    a = _report(tmp_path / "a")
    b = _report(tmp_path / "b", contamination=ContaminationScore(10, 10, 10, 1.0, any_hits=10))
    assert verdict(a) == verdict(b)
