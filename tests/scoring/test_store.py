import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from atlas.config import AgentSpec
from atlas.schemas import AgentOutput, AgentResult, FailureReason, FailureRecord, Pick
from atlas.scoring.metrics import rank_ic
from atlas.scoring.store import ScoreStore, non_overlapping

D0 = date(2024, 1, 2)  # Tuesday
DAYS = [d.date() for d in pd.bdate_range(D0, periods=30)]


class FakeStore:
    """Stands in for PriceStore: business-day calendar, dict of price lists."""

    def __init__(self, prices):
        self.prices = prices  # ticker -> {date: price}

    def trading_days(self, start, end):
        return [d for d in DAYS if start <= d <= end]

    def outcome_prices(self, ticker, start, end):
        p = self.prices.get(ticker, {})
        items = [(d, v) for d, v in p.items() if start <= d <= end]
        return pd.Series([v for _, v in items], index=[d for d, _ in items], dtype=float)


def px(start, daily):
    return {d: start * (1 + daily) ** i for i, d in enumerate(DAYS)}


SPECS = [AgentSpec(name="tech", layer=1, kind="sector", prompt_file="x", benchmark="SMH"),
         AgentSpec(name="cro", layer=2, kind="cro", prompt_file="x")]


def result(agent="tech", day=D0, picks=()):
    return AgentResult(agent=agent, date=day, output=AgentOutput(
        agent=agent, date=day, view="bullish", confidence=0.5, picks=list(picks)))


def pick(t, d="long", c=0.6):
    return Pick(ticker=t, direction=d, conviction=c, horizon_days=5)


@pytest.fixture
def ss(tmp_path):
    return ScoreStore(tmp_path / "s.db")


def test_pick_not_scored_before_horizon_matures(ss):
    fs = FakeStore({"AAA": px(100, 0.01), "SMH": px(100, 0.0)})
    ss.record_picks(result(picks=[pick("AAA")]))
    assert ss.score_matured(DAYS[3], fs, SPECS, 0.0) == 1  # only h=1 (exit DAYS[1]) ... see below
    f5 = ss.picks_frame("tech", 5)
    assert not f5.scored.any() and math.isnan(f5.excess.iloc[0])


def test_exit_on_asof_is_scored_and_after_asof_is_not(ss):
    fs = FakeStore({"AAA": px(100, 0.01), "SMH": px(100, 0.0)})
    ss.record_picks(result(picks=[pick("AAA")]))
    assert ss.score_matured(DAYS[4], fs, SPECS, 0.0) == 1  # h=1 only
    assert ss.score_matured(DAYS[5], fs, SPECS, 0.0) == 1  # h=5 exit == asof
    f5 = ss.picks_frame("tech", 5)
    assert f5.scored.all()
    assert f5.excess.iloc[0] == pytest.approx(1.01 ** 5 - 1)
    assert not ss.picks_frame("tech", 10).scored.any()


def test_rescoring_returns_zero(ss):
    fs = FakeStore({"AAA": px(100, 0.01), "SMH": px(100, 0.0)})
    ss.record_picks(result(picks=[pick("AAA")]))
    assert ss.score_matured(DAYS[20], fs, SPECS, 0.0) == 3
    assert ss.score_matured(DAYS[20], fs, SPECS, 0.0) == 0


def test_none_direction_not_scored(ss):
    ss.record_picks(result(picks=[pick("AAA", "none")]))
    assert ss.picks_frame("tech", 1).empty


def test_no_opinion_result_records_nothing(ss):
    fail = AgentResult(agent="tech", date=D0, failure=FailureRecord(
        agent="tech", date=D0, reason=FailureReason.TIMEOUT, detail="", attempt=1, raw_response=""))
    ss.record_picks(fail)
    partial = AgentResult(agent="tech", date=D0, output=AgentOutput(
        agent="tech", date=D0, view="bullish", confidence=0.5, picks=[pick("AAA")]),
        failure=fail.failure)
    ss.record_picks(partial)
    assert ss.picks_frame(None, 1).empty


def test_record_picks_idempotent(ss):
    r = result(picks=[pick("AAA")])
    ss.record_picks(r)
    ss.record_picks(r)
    assert len(ss.picks_frame("tech", 5)) == 1


def test_short_on_falling_stock_contributes_positive_ic(ss):
    fs = FakeStore({"UP": px(100, 0.02), "DN": px(100, -0.02), "FLAT": px(100, 0.0),
                    "SMH": px(100, 0.0)})
    ss.record_picks(result(picks=[pick("UP", "long", 0.9), pick("FLAT", "long", 0.1),
                                  pick("DN", "short", 0.9)]))
    ss.score_matured(DAYS[10], fs, SPECS, 0.0)
    f = ss.picks_frame("tech", 5).set_index("ticker")
    assert f.loc["DN", "conviction"] == pytest.approx(-0.9)
    assert f.loc["DN", "excess"] < 0
    assert rank_ic(f.conviction, f.excess) == pytest.approx(1.0)


def test_missing_prices_leave_null(ss):
    fs = FakeStore({"SMH": px(100, 0.0)})
    ss.record_picks(result(picks=[pick("AAA")]))
    assert ss.score_matured(DAYS[20], fs, SPECS, 0.0) == 0
    f = ss.picks_frame("tech", 1)
    assert not f.scored.any() and f.excess.isna().all()


def test_no_benchmark_agent_uses_spy(ss):
    fs = FakeStore({"AAA": px(100, 0.01), "SPY": px(100, 0.01)})
    ss.record_picks(result("cro", picks=[pick("AAA")]))
    ss.score_matured(DAYS[10], fs, SPECS, 0.0)
    assert ss.picks_frame("cro", 1).excess.iloc[0] == pytest.approx(0.0)


def test_picks_frame_columns_and_filter(ss):
    ss.record_picks(result("tech", picks=[pick("AAA")]))
    ss.record_picks(result("cro", picks=[pick("BBB")]))
    f = ss.picks_frame(None, 5)
    assert list(f.columns) == ["agent", "date", "ticker", "conviction", "horizon", "excess", "scored"]
    assert len(f) == 2 and (f.horizon == 5).all()
    assert len(ss.picks_frame("cro", 5)) == 1


def test_non_overlapping_samples_spaced_by_horizon():
    rows = [("a", DAYS[i], "X", 0.5, 5, 0.0, True) for i in range(12)]
    rows += [("b", DAYS[i], "X", 0.5, 5, 0.0, True) for i in range(0, 12, 2)]
    df = pd.DataFrame(rows, columns=["agent", "date", "ticker", "conviction", "horizon", "excess", "scored"])
    out = non_overlapping(df, 5)
    a = sorted(out[out.agent == "a"].date.unique())
    assert a == [DAYS[0], DAYS[5], DAYS[10]]
    assert sorted(out[out.agent == "b"].date.unique()) == [DAYS[0], DAYS[10]]
    assert len(non_overlapping(df, 1)) == len(df)
