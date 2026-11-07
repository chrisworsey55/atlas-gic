"""Report generation tests: sections, labels, verdict logic, hand-computed metrics."""
import math
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from atlas.autoresearch.loop import EditRecord
from atlas.backtest.nulltest import NullDistribution
from atlas.backtest.report import (
    NOT_EVIDENCE, build_report, compute_metrics, max_drawdown, sharpe,
)
from atlas.backtest.runner import BacktestResult, Periods
from atlas.config import Settings

CUTOFF = date(2024, 1, 31)


def settings() -> Settings:
    return Settings(universe=["SPY"], commission_per_share=0.0, commission_min=0.0,
                    slippage_bps=0.0, min_valid_agents=1, rolling_ic_window=60, warmup_picks=5,
                    budget_usd=10.0, model="m", model_cutoff=CUTOFF)


def days(start: date, n: int) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def make_result(holdout_days=70, halted=False, skipped=None) -> BacktestResult:
    tr = days(date(2023, 11, 1), 20)
    va = days(tr[-1] + timedelta(days=1), 20)
    ho = days(date(2024, 2, 1), holdout_days)
    allday = tr + va + ho
    rng = np.random.default_rng(0)
    eq = pd.Series(100_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, len(allday))), index=allday)
    spy = pd.Series(400 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(allday))), index=allday)
    bsk = pd.Series(100 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(allday))), index=allday)
    periods = Periods(train=(tr[0], tr[-1]), validation=(va[0], va[-1]), holdout=(ho[0], ho[-1]))
    return BacktestResult(equity=eq, spy=spy, basket=bsk, day_results={},
                          skipped=skipped if skipped is not None else {},
                          periods=periods, halted=halted,
                          halt_reason="drawdown kill" if halted else None, spent_usd=3.0)


class FakeScores:
    def __init__(self, frame):
        self.frame = frame

    def picks_frame(self, agent, horizon):
        return self.frame if agent is None else self.frame[self.frame["agent"] == agent]


def picks(agent, n=40, ic_sign=0.0, seed=1, short=False):
    rng = np.random.default_rng(seed)
    conv = rng.uniform(0.1, 1, n)
    ex = ic_sign * conv + rng.normal(0, 0.5, n) * 0.01 if ic_sign else rng.normal(0, 0.01, n)
    direction = ["short" if short else "long"] * n
    return pd.DataFrame({
        "agent": agent, "date": [date(2024, 2, 1) + timedelta(days=i) for i in range(n)],
        "ticker": "AAA", "direction": direction, "conviction": conv, "horizon": 1,
        "excess": ex, "signed_excess": ex * (-1.0 if short else 1.0), "scored": True})


def make_scores(frame=None):
    if frame is None:
        frame = pd.concat([picks("a1", ic_sign=0.0, seed=1), picks("a2", seed=2),
                           picks("control", seed=3)], ignore_index=True)
    return FakeScores(frame)


def make_null():
    return NullDistribution(np.linspace(-0.2, 0.2, 100), deviations=np.linspace(0, 0.1, 100),
                            pair_diffs=np.linspace(0, 0.2, 100))


def edit(decision="kept"):
    return EditRecord(agent="a1", branch="b", n_days=20, old_ic=0.01, new_ic=0.2, diff=0.19,
                      keep_margin=0.15, decision=decision, reason="ok", commit_sha="abc",
                      cost_usd=0.5)


def fail_df():
    return pd.DataFrame({"agent": ["a1"], "reason": ["bad_json"], "count": [2]})


def build(tmp_path, result=None, scores=None, null="default", edits=None, ablation="default",
          control_ic=0.0, **kw):
    return build_report(
        result or make_result(), scores or make_scores(),
        make_null() if null == "default" else null,
        [edit()] if edits is None else edits,
        {"cio_total": 0.1, "baseline_total": 0.05, "cio_beats_baseline": True}
        if ablation == "default" else ablation,
        control_ic, fail_df(), 3.0, tmp_path, settings=settings(), **kw)


def test_report_contains_all_required_sections(tmp_path):
    md = build(tmp_path).read_text()
    for h in ["Verdict", "Performance", "Agents", "Control agent", "Autoresearch", "Null test",
              "CIO ablation", "Failures", "Cost", "Limitations"]:
        assert f"\n## {h}" in "\n" + md
    assert "This backtest is not evidence of future profitability." in md
    assert "bad_json" in md
    assert "SPY buy-and-hold" in md and "Equal-weight basket" in md
    assert (tmp_path / "equity.png").stat().st_size > 0
    assert (tmp_path / "agents.csv").exists() and (tmp_path / "autoresearch.csv").exists()
    assert "a prompt that abstains selectively inside a day can inflate paired IC" in md
    assert "plumbing test" in md


def test_pre_cutoff_periods_labelled_not_evidence_of_skill(tmp_path):
    md = build(tmp_path).read_text()
    assert NOT_EVIDENCE == "NOT EVIDENCE OF SKILL (pre-cutoff: the model may have seen these dates)"
    perf = md.split("## Performance")[1].split("\n## ")[0]
    tr = perf.split("### train")[1].split("### validation")[0]
    va = perf.split("### validation")[1].split("### holdout")[0]
    ho = perf.split("### holdout")[1]
    assert NOT_EVIDENCE in tr and NOT_EVIDENCE in va
    assert NOT_EVIDENCE not in ho
    assert "only uncontaminated" in ho


def test_holdout_too_short_note(tmp_path):
    md = build(tmp_path, result=make_result(holdout_days=10)).read_text()
    assert "holdout too short for reliable conclusions" in md
    assert "No evidence of skill beyond the null" in md


def test_verdict_says_no_evidence_when_ic_inside_null(tmp_path):
    md = build(tmp_path).read_text()
    verdict = md.split("## Verdict")[1].split("\n## ")[0]
    assert "No evidence of skill beyond the null" in verdict
    assert "This backtest is not evidence of future profitability." in verdict


def test_null_none_says_not_calibrated(tmp_path):
    md = build(tmp_path, null=None).read_text()
    assert "not calibrated — weights were frozen" in md.split("## Null test")[1].split("\n## ")[0]
    assert "No evidence of skill beyond the null" in md.split("## Verdict")[1].split("\n## ")[0]


def _skill_setup(tmp_path, ctrl=-0.5, ic=0.5, boost=True, null="default"):
    res = make_result(holdout_days=70)
    if boost:  # holdout strategy strongly beats SPY
        ho0 = res.periods.holdout[0]
        res.equity.loc[res.equity.index >= ho0] *= np.linspace(1, 3, int((res.equity.index >= ho0).sum()))
        res.spy.loc[res.spy.index >= ho0] *= np.linspace(1, 0.9, int((res.spy.index >= ho0).sum()))
    else:  # holdout strategy clearly loses to SPY
        ho0 = res.periods.holdout[0]
        res.equity.loc[res.equity.index >= ho0] *= np.linspace(1, 0.4, int((res.equity.index >= ho0).sum()))
    frame = pd.concat([picks("a1", ic_sign=ic, seed=1), picks("a2", ic_sign=ic, seed=2)],
                      ignore_index=True)
    return build(tmp_path, result=res, scores=make_scores(frame), control_ic=ctrl, null=null)


def verdict_of(md):
    return md.split("## Verdict")[1].split("\n## ")[0]


def test_possible_evidence_only_when_all_three_hold(tmp_path):
    v = verdict_of(_skill_setup(tmp_path / "ok", ic=5.0).read_text())
    assert "Possible evidence of skill" in v
    assert "This backtest is not evidence of future profitability." in v
    # control IC not below the real median
    v = verdict_of(_skill_setup(tmp_path / "c", ic=5.0, ctrl=2.0).read_text())
    assert "Possible evidence of skill" not in v
    # holdout excess negative
    v = verdict_of(_skill_setup(tmp_path / "h", ic=5.0, boost=False).read_text())
    assert "Possible evidence of skill" not in v
    # median IC not above dead zone (ic=0 -> noise)
    v = verdict_of(_skill_setup(tmp_path / "i", ic=0.0).read_text())
    assert "Possible evidence of skill" not in v


def test_possible_evidence_withheld_without_holdout_excess(tmp_path):
    res = make_result()
    res.equity.loc[res.equity.index >= res.periods.holdout[0]] *= 0.5
    frame = pd.concat([picks("a1", ic_sign=5.0, seed=1), picks("a2", ic_sign=5.0, seed=2)])
    v = verdict_of(build(tmp_path, result=res, scores=make_scores(frame), control_ic=-0.5).read_text())
    assert "Possible evidence of skill" not in v


def test_metrics_net_of_costs_match_hand_computed_sharpe_and_drawdown():
    d = days(date(2024, 3, 1), 6)
    eq = pd.Series([100.0, 110.0, 99.0, 108.9, 108.9, 119.79], index=d)
    # returns: .1, -.1, .1, 0, .1
    rets = np.array([0.1, -0.1, 0.1, 0.0, 0.1])
    exp_sharpe = rets.mean() / rets.std(ddof=1) * math.sqrt(252)
    m = compute_metrics(eq, d[0], d[-1], prior=None)
    assert m["sharpe"] == pytest.approx(exp_sharpe)
    assert m["vol"] == pytest.approx(rets.std(ddof=1) * math.sqrt(252))
    assert m["max_drawdown"] == pytest.approx(-0.1)  # 110 -> 99
    assert m["total_return"] == pytest.approx(119.79 / 100 - 1)
    # with a prior-day value the return uses it as the base
    m2 = compute_metrics(eq, d[1], d[-1], prior=100.0)
    assert m2["total_return"] == pytest.approx(119.79 / 100 - 1)
    assert m2["n"] == 5


def test_sharpe_edge_cases():
    assert sharpe(pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])) is None  # zero std
    assert sharpe(pd.Series([0.01, 0.02])) is None  # <5 returns
    assert max_drawdown(pd.Series([1.0, 2.0, 3.0])) == 0.0


def test_metrics_per_period_only_uses_that_period():
    d = days(date(2024, 3, 1), 10)
    eq = pd.Series([100, 200, 400, 800, 1600, 1600, 1600, 1600, 1600, 1600.0], index=d)
    m = compute_metrics(eq, d[5], d[9], prior=1600.0)
    assert m["total_return"] == 0.0 and m["max_drawdown"] == 0.0 and m["sharpe"] is None
    assert compute_metrics(eq, date(2030, 1, 1), date(2030, 2, 1), prior=None) is None


def test_empty_nan_inputs_do_not_crash(tmp_path):
    res = make_result()
    res.spy = pd.Series(dtype=float)
    res.basket = pd.Series(dtype=float)
    frame = make_scores(pd.DataFrame(columns=["agent", "date", "ticker", "direction", "conviction",
                                              "horizon", "excess", "signed_excess", "scored"]))
    p = build_report(res, frame, None, [], None, None, pd.DataFrame(), 0.0, tmp_path,
                     settings=settings())
    md = p.read_text()
    assert "not run" in md.split("## Control agent")[1]
    assert "not run" in md.split("## CIO ablation")[1]
    nan_edit = EditRecord("a", None, 0, None, float("nan"), None, None, "uncalibrated", "x", None, 0.0)
    build(tmp_path / "n", result=make_result(), edits=[nan_edit], control_ic=float("nan"))


def test_no_holdout_after_cutoff_whole_run_labelled(tmp_path):
    res = make_result()
    res.periods = None
    res.equity = res.equity[res.equity.index <= CUTOFF]
    md = build(tmp_path, result=res).read_text()
    assert NOT_EVIDENCE in md
    assert "No evidence of skill beyond the null" in verdict_of(md)


def test_halted_skipped_risk_errors_reported(tmp_path):
    res = make_result(halted=True, skipped={date(2024, 2, 5): "no data", date(2024, 2, 6): "no data"})
    res.risk_errors = {date(2024, 2, 7): "boom"}
    md = build(tmp_path, result=res).read_text()
    assert "drawdown kill" in md and "Skipped days: 2 (no data: 2)" in md and "boom" in md


def test_csv_hit_rate_uses_signed_excess_and_rank_ic_conviction_excess(tmp_path):
    n = 10
    ex = np.array([-0.02, -0.01, -0.03, -0.015, -0.005, -0.04, -0.02, -0.01, -0.03, -0.02])
    conv = -np.linspace(0.1, 1, n)  # signed conviction for shorts
    short = pd.DataFrame({"agent": "sh", "date": days(date(2024, 2, 1), n), "ticker": "T",
                          "direction": "short", "conviction": conv, "horizon": 1, "excess": ex,
                          "signed_excess": -ex, "scored": True})
    build(tmp_path, scores=make_scores(short), holdout_edit_ics={"a1": 0.3},
          )
    a = pd.read_csv(tmp_path / "agents.csv").set_index("agent")
    assert a.loc["sh", "hit_rate"] == 1.0  # every successful short is a hit
    from atlas.scoring.metrics import rank_ic
    assert a.loc["sh", "rank_ic"] == pytest.approx(rank_ic(conv, ex))
    assert a.loc["sh", "n_scored"] == n
    ar = pd.read_csv(tmp_path / "autoresearch.csv")
    assert ar.loc[0, "holdout_ic"] == pytest.approx(0.3)
    assert "holdout IC not computed" not in (tmp_path / "report.md").read_text().split("## Autoresearch")[1].split("\n## ")[0]


def test_kept_edit_without_holdout_ic_says_not_computed(tmp_path):
    md = build(tmp_path).read_text()
    assert "holdout IC not computed" in md.split("## Autoresearch")[1].split("\n## ")[0]
