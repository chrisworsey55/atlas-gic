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
    frame = pd.concat([picks("a1", ic_sign=ic, seed=1), picks("a2", ic_sign=ic, seed=2),
                       picks("a3", ic_sign=ic, seed=4)], ignore_index=True)
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
    frame = pd.concat([picks("a1", ic_sign=5.0, seed=1), picks("a2", ic_sign=5.0, seed=2),
                       picks("a3", ic_sign=5.0, seed=4)])
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


# ---- fix round 1: strict verdict rules and mutation-catching tests

def winning(res, strat=3.0, spy=0.9):
    ho0 = res.periods.holdout[0]
    m = res.equity.index >= ho0
    res.equity.loc[m] *= np.linspace(1, strat, int(m.sum()))
    res.spy.loc[m] *= np.linspace(1, spy, int(m.sum()))
    return res


def skilled(names=("a1", "a2", "a3"), ic=5.0):
    return pd.concat([picks(n, ic_sign=ic, seed=i + 1) for i, n in enumerate(names)],
                     ignore_index=True)


def verdict_line(md):
    return verdict_of(md).split("**")[1]


def run(tmp_path, res=None, frame=None, null="default", ctrl=-0.5, settings_on=True, **kw):
    res = res if res is not None else winning(make_result())
    frame = frame if frame is not None else skilled()
    if not settings_on:
        return build_report(res, make_scores(frame), make_null(), [], None, ctrl,
                            pd.DataFrame(), 1.0, tmp_path).read_text()
    return build(tmp_path, result=res, scores=make_scores(frame), control_ic=ctrl, null=null,
                 **kw).read_text()


def test_all_conditions_pass_gives_possible(tmp_path):
    assert verdict_line(run(tmp_path)) == POSSIBLE_S
    assert "PASS (a)" in run(tmp_path / "x")


POSSIBLE_S = "Possible evidence of skill"


def test_settings_none_is_safe_and_all_periods_contaminated(tmp_path):
    md = run(tmp_path, settings_on=False)
    assert verdict_line(md) == "No evidence of skill beyond the null"
    assert "model cutoff unknown" in verdict_of(md)
    perf = md.split("## Performance")[1].split("\n## ")[0]
    for name in ("train", "validation", "holdout"):
        seg = perf.split(f"### {name}")[1].split("###")[0]
        assert NOT_EVIDENCE in seg
    assert "only uncontaminated" not in perf


def test_first_day_base_inside_report(tmp_path):
    res = make_result()
    d = list(res.equity.index)
    ho0 = res.periods.holdout[0]
    i = d.index(ho0)
    vals = np.full(len(d), 100.0)
    vals[i:] = 150.0  # all of the holdout's gain happens on its first day
    res.equity = pd.Series(vals, index=d)
    md = run(tmp_path, res=res)
    ho = md.split("## Performance")[1].split("### holdout")[1].split("\n## ")[0]
    assert "50.00%" in ho.split("Strategy")[1].split("\n")[0]  # 150/100-1 incl. base day


def test_holdout_ic_drives_verdict_not_train_ic(tmp_path):
    train = pd.concat([picks(n, ic_sign=5.0, seed=i + 1).assign(
        date=lambda x: [date(2023, 11, 1) + timedelta(days=j) for j in range(len(x))])
        for i, n in enumerate(("a1", "a2", "a3"))], ignore_index=True)
    ho_noise = pd.concat([picks(n, seed=10 + i) for i, n in enumerate(("a1", "a2", "a3"))],
                         ignore_index=True)
    md = run(tmp_path, frame=pd.concat([train, ho_noise], ignore_index=True))
    assert verdict_line(md) == "No evidence of skill beyond the null"
    assert "FAIL (b)" in md


def test_excess_is_versus_spy_not_zero(tmp_path):
    res = make_result()
    idx = res.equity.index
    ho = idx >= res.periods.holdout[0]
    e = np.full(len(idx), 100_000.0)
    e[ho] = np.linspace(100_000, 101_000, int(ho.sum()))  # +1% in holdout
    s = np.full(len(idx), 400.0)
    s[ho] = np.linspace(400, 420, int(ho.sum()))  # +5%
    res.equity, res.spy = pd.Series(e, index=idx), pd.Series(s, index=idx)
    md = run(tmp_path, res=res)
    assert verdict_line(md) == "No evidence of skill beyond the null"
    assert "FAIL (d)" in md


@pytest.mark.parametrize("ctrl", [None, float("nan")])
def test_control_missing_or_nan_blocks_possible(tmp_path, ctrl):
    md = run(tmp_path, ctrl=ctrl)
    assert verdict_line(md) == "No evidence of skill beyond the null"
    assert "FAIL (e)" in md


def test_mean_shrunk_below_dead_zone_blocks_possible(tmp_path):
    # ICs (1, 1, -1): median 1 clears the raw bar but the mean shrunk IC does not clear 0.5
    frame = pd.concat([picks("a1", ic_sign=5.0, seed=1), picks("a2", ic_sign=5.0, seed=2),
                       picks("a3", ic_sign=-5.0, seed=4)], ignore_index=True)
    null = NullDistribution(np.linspace(-0.2, 0.2, 100), deviations=np.linspace(0, 0.5, 100),
                            pair_diffs=np.linspace(0, 0.2, 100))
    md = run(tmp_path, frame=frame, null=null, ctrl=-0.9)
    assert verdict_line(md) == "No evidence of skill beyond the null"
    assert "FAIL (c)" in md


def test_one_eligible_agent_blocks_possible(tmp_path):
    md = run(tmp_path, frame=skilled(names=("a1",)))
    assert verdict_line(md) == "No evidence of skill beyond the null" and "FAIL (a)" in md


def test_each_condition_failing_flips_the_verdict(tmp_path):
    assert verdict_line(run(tmp_path / "ok")) == POSSIBLE_S
    cases = {
        "a": dict(frame=skilled(names=("a1", "a2"))),
        "b": dict(frame=skilled(ic=0.0)),
        "d": dict(res=winning(make_result(), strat=0.5, spy=1.0)),
        "e": dict(ctrl=2.0),
        "f": dict(null=None),
        "g": dict(settings_on=False),
    }
    for k, kw in cases.items():
        md = run(tmp_path / k, **kw)
        assert verdict_line(md) == "No evidence of skill beyond the null", k
        assert f"FAIL ({k})" in md, k
    # (c): dead zone above any achievable mean shrunk IC
    big = NullDistribution(np.linspace(-0.2, 0.2, 100), deviations=np.full(100, 5.0),
                           pair_diffs=np.linspace(0, 0.2, 100))
    md = run(tmp_path / "c", null=big)
    assert verdict_line(md) == "No evidence of skill beyond the null" and "FAIL (c)" in md


def test_control_section_one_definition(tmp_path):
    frame = pd.concat([skilled(), picks("control", seed=9)], ignore_index=True)
    md = run(tmp_path, frame=frame, ctrl=0.777)
    sec = md.split("## Control agent")[1].split("\n## ")[0]
    assert "score store" in sec and "0.777" not in sec and "Holdout only" in sec
    sec2 = run(tmp_path / "o", ctrl=0.777).split("## Control agent")[1].split("\n## ")[0]
    assert "0.777" in sec2 and "passed-in" in sec2


def test_not_provided_markers_and_escaping(tmp_path):
    md = run(tmp_path)
    assert "not provided" in md.split("## Failures")[1].split("\n## ")[0]
    assert "not provided" in md.split("## Agents")[1].split("\n## ")[0]
    fails = pd.DataFrame({"agent": ["a|1"], "reason": ["x\ny"], "count": [1]})
    p = build_report(make_result(), make_scores(), make_null(), [], None, None, fails, 1.0,
                     tmp_path / "e", settings=settings(),
                     dropped_picks=pd.DataFrame({"agent": ["a"], "reason": ["r"], "count": [3]}),
                     turnover={"holdout": 1.5}, final_weights={"a1": 1.0})
    t = p.read_text()
    assert "a\\|1" in t and "x y" in t
    assert "not provided" not in t.split("## Failures")[1].split("\n## ")[0]


def test_nan_ic_agent_has_nan_shrunk(tmp_path):
    f = picks("a1", n=40)
    f2 = picks("flat", n=40)
    f2["conviction"] = 0.5  # constant conviction -> NaN IC
    build(tmp_path, scores=make_scores(pd.concat([f, f2], ignore_index=True)))
    a = pd.read_csv(tmp_path / "agents.csv").set_index("agent")
    assert math.isnan(a.loc["flat", "shrunk_ic"])


# ---- fix round 2: exception safety and exact threshold pins
import atlas.backtest.report as report_mod
from atlas.backtest.report import _verdict
from atlas.scoring.weights import shrunk_ic as _shrunk

NO = "No evidence of skill beyond the null"


def test_control_ic_string_does_not_crash(tmp_path):
    md = run(tmp_path, ctrl="abc")
    assert verdict_line(md) == NO


def test_empty_ics_null_does_not_crash(tmp_path):
    bad = NullDistribution(np.array([]), deviations=np.linspace(0, 0.1, 10),
                           pair_diffs=np.linspace(0, 0.2, 10))
    md = run(tmp_path, null=bad)
    assert verdict_line(md) == NO
    assert "## Null test" in md


def test_helper_raising_yields_no_evidence_and_report_written(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(report_mod, "_control_ic", boom)
    md = run(tmp_path)
    assert verdict_line(md) == NO
    assert "verdict computation failed: RuntimeError" in md
    assert "## Control agent" in md


class StubNull:
    def __init__(self, q95, dz):
        self._q, self._dz = q95, dz
        self.ics = np.zeros(10)

    def quantile(self, q):
        return self._q

    def dead_zone(self):
        return self._dz


K = 20.0
HO = (date(2024, 2, 1), date(2024, 6, 1))


def ho_frame(ics, n=100):
    return pd.DataFrame({"agent": [f"a{i}" for i in range(len(ics))],
                         "n_scored": [n] * len(ics), "rank_ic": ics})


def spy_series():
    d = days(HO[0] - timedelta(days=3), 10)
    return pd.Series([100.0] * 4 + [110.0] * 6, index=d), d


def call(q95=0.1, dz=0.1, ics=(0.5, 0.5, 0.5), ctrl=0.0, days_=60, excess_delta=1e-6, n=100):
    spy, d = spy_series()
    lo, hi = d[4], d[-1]
    spy_ret = compute_metrics(spy, lo, hi, _prior(spy, lo))["total_return"]
    hm = {"total_return": spy_ret + excess_delta}
    v, reasons = _verdict(StubNull(q95, dz), CUTOFF, (None, lo, hi), days_, hm, spy,
                          ho_frame(list(ics), n), ctrl, 5, K)
    return v, "\n".join(reasons)


from atlas.backtest.report import _prior  # noqa: E402


def test_stub_all_pass():
    assert call()[0] == POSSIBLE_S


def test_b_median_equal_threshold_fails_above_passes():
    assert call(q95=0.5)[0] == NO and "FAIL (b)" in call(q95=0.5)[1]
    assert "PASS (b)" in call(q95=0.5 - 1e-9)[1]


def test_c_mean_shrunk_equal_threshold_fails_above_passes():
    ics = [0.5, 0.5, 0.5]
    gm = float(np.mean(ics))
    ms = float(np.mean([_shrunk(i, 100, gm, K) for i in ics]))
    assert call(dz=ms)[0] == NO and "FAIL (c)" in call(dz=ms)[1]
    assert "PASS (c)" in call(dz=ms - 1e-9)[1]


def test_d_excess_zero_fails_positive_passes():
    assert call(excess_delta=0.0)[0] == NO and "FAIL (d)" in call(excess_delta=0.0)[1]
    assert "PASS (d)" in call(excess_delta=1e-9)[1]


def test_d_length_boundary_59_vs_60():
    assert call(days_=59)[0] == NO and "FAIL (d)" in call(days_=59)[1]
    assert call(days_=60)[0] == POSSIBLE_S


def test_e_control_equal_median_fails_below_passes():
    assert call(ctrl=0.5)[0] == NO and "FAIL (e)" in call(ctrl=0.5)[1]
    assert "PASS (e)" in call(ctrl=0.5 - 1e-9)[1]


def test_verdict_follows_holdout_control_not_all_period(tmp_path):
    def perfect(agent, sign, start, seed):
        f = picks(agent, seed=seed)
        f["excess"] = sign * f["conviction"] * 0.01
        f["signed_excess"] = f["excess"]
        f["date"] = [start + timedelta(days=j) for j in range(len(f))]
        return f
    ho_start = date(2024, 2, 1)
    frame = pd.concat([perfect(n, 1, ho_start, i) for i, n in enumerate(("a1", "a2", "a3"))]
                      + [perfect("control", -1, date(2023, 11, 1), 20),
                         perfect("control", 1, ho_start, 21)], ignore_index=True)
    md = run(tmp_path, frame=frame, ctrl=-0.9)
    assert verdict_line(md) == NO and "FAIL (e)" in md


def test_control_rows_not_counted_as_eligible_agents(tmp_path):
    frame = pd.concat([skilled(names=("a1", "a2")), picks("control", ic_sign=5.0, seed=22)],
                      ignore_index=True)
    md = run(tmp_path, frame=frame)
    assert verdict_line(md) == NO and "2 eligible" in md
    # direct: a control-named row must not change count or median
    h = ho_frame([0.5, 0.5, 0.5])
    h.loc[len(h)] = [CONTROL_NAME, 100, 0.99]
    v, r = _verdict(StubNull(0.1, 0.1), CUTOFF, (None, *HO), 60, None, pd.Series(dtype=float), h,
                    0.0, 5, K)
    assert "3 eligible" in "\n".join(r)


CONTROL_NAME = "control"


def test_below_warmup_agent_is_not_eligible(tmp_path):
    short = picks("a3", ic_sign=5.0, seed=4).iloc[:3]  # 3 picks < warmup_picks=5
    frame = pd.concat([skilled(names=("a1", "a2")), short], ignore_index=True)
    md = run(tmp_path, frame=frame)
    assert verdict_line(md) == NO and "2 eligible" in md


def test_ablation_by_period_headlines_holdout_and_labels_pre_cutoff(tmp_path):
    a = lambda c, b: {"cio_total": c, "baseline_total": b, "cio_beats_baseline": c > b}  # noqa: E731
    by = {"train": a(0.02, 0.01), "validation": None, "holdout": a(-0.01, 0.03), "all": a(0.0, 0.05)}
    md = build(tmp_path, ablation=by["holdout"], ablation_by_period=by).read_text()
    sec = md.split("## CIO ablation")[1].split("## Failures")[0]
    assert "Headline: holdout" in sec and "CIO total -1.00% vs baseline total 3.00%; CIO beat baseline: no" in sec
    assert "- train [NOT EVIDENCE OF SKILL" in sec and "CIO total 2.00%" in sec
    assert "- validation [NOT EVIDENCE OF SKILL" in sec and "not computable" in sec
    assert "- holdout [holdout]" in sec
    assert "- all periods [NOT EVIDENCE OF SKILL" in sec and "CIO total 0.00% vs baseline total 5.00%" in sec


def test_ablation_by_period_without_holdout_says_so(tmp_path):
    res = make_result()
    res.periods = None  # whole run is one pre-cutoff-or-unknown period: no holdout
    by = {"all": {"cio_total": 0.0, "baseline_total": 0.0, "cio_beats_baseline": False}}
    sec = build(tmp_path, result=res, ablation=None, ablation_by_period=by).read_text() \
        .split("## CIO ablation")[1].split("## Failures")[0]
    assert "Headline: no holdout" in sec


def test_single_ablation_dict_still_rendered_without_by_period(tmp_path):
    sec = build(tmp_path).read_text().split("## CIO ablation")[1].split("## Failures")[0]
    assert "CIO total 10.00% vs baseline total 5.00%; CIO beat baseline: yes." in sec


# ---- final review fixes: disclosed limitations

def _section(md, name):
    return md.split(f"## {name}")[1].split("\n## ")[0]


def test_kept_edit_holdout_ic_gap_is_explained_with_what_to_do(tmp_path):
    sec = _section(build(tmp_path).read_text(), "Autoresearch")
    assert "holdout IC not computed" in sec
    assert "known gap" in sec and "new run_dir" in sec  # why, and how to confirm a kept edit


def test_null_section_says_the_agent_sets_differ(tmp_path):
    sec = _section(build(tmp_path).read_text(), "Null test")
    assert "decision-layer" in sec and "macro/sector/superinvestor" in sec


def test_limitations_disclose_every_accepted_gap(tmp_path):
    lim = _section(build(tmp_path).read_text(), "Limitations")
    for needle in ("PLACEHOLDER GUESS", "configured per-model prices", "response cache",
                   "Single-process only", "yfinance data can be stale", "approximate (overstated by about 7/5)",
                   "multiple testing", "within-day clustering", "survivorship", "Adjusted-price leakage",
                   "No live or paper trading", "IBKR price provider is unverified",
                   "--with-control", "holdout IC of kept autoresearch edits is not computed"):
        assert needle in lim, needle
