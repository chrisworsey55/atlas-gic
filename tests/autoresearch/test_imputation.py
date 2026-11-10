"""Task 6: DayPairs and abstention imputation in paired_replay."""
import math
import random
from datetime import date, timedelta
from types import SimpleNamespace

import pandas as pd
import pytest

from atlas.autoresearch.loop import DayPairs, InsufficientReplayDays, paired_replay
from atlas.scoring.metrics import rank_ic

DAYS = [date(2024, 1, 1) + timedelta(days=i) for i in range(40)]

# the old prompt's best-convinced pick "X" is the day's worst; the new prompt omits it
OLD = DayPairs(["A", "B", "C", "X"], [0.9, 0.7, 0.5, 0.8], [0.03, 0.02, 0.01, -0.05])
NEW = DayPairs(["A", "B", "C"], [0.9, 0.7, 0.5], [0.03, 0.02, 0.01])


def runner(old, new):
    def run(day, prompt):
        return old if prompt == "old" else new
    return run


def replay(old, new, **kw):
    return paired_replay("agent_a", "old", "new", DAYS, runner(old, new), **kw)


def test_dropping_a_loser_earns_no_credit_and_stays_in_the_sample():
    plain = replay(OLD, NEW, impute=False)
    imputed = replay(OLD, NEW, impute=True)
    n = len(DAYS)
    assert plain.old_ic == pytest.approx(rank_ic(OLD.convictions * n, OLD.excess * n))
    assert plain.new_ic == pytest.approx(1.0)
    assert plain.diff > 0.1  # without imputation, dropping X looks like skill
    # imputed: X enters the new side as a copy of the old side's pick (0.8, -0.05)
    new_c = (NEW.convictions + [0.8]) * n
    new_e = (NEW.excess + [-0.05]) * n
    assert imputed.new_ic == pytest.approx(rank_ic(new_c, new_e))
    assert imputed.old_ic == pytest.approx(plain.old_ic)
    assert imputed.diff == pytest.approx(0.0, abs=1e-12)


def test_symmetric_imputation_for_picks_only_the_new_prompt_makes():
    # swap roles: the NEW prompt adds a pick the old one did not make
    res = replay(NEW, OLD, impute=True)
    n = len(DAYS)
    assert res.old_imputed == n and res.new_imputed == 0
    assert res.old_ic == pytest.approx(rank_ic(OLD.convictions * n, OLD.excess * n))
    assert res.new_ic == pytest.approx(res.old_ic)
    assert res.diff == pytest.approx(0.0, abs=1e-12)


def test_two_tuple_results_are_unchanged_and_never_imputed():
    tup_old = (OLD.convictions, OLD.excess)
    tup_new = (NEW.convictions, NEW.excess)
    res = replay(tup_old, tup_new, impute=True)
    ref = replay(OLD, NEW, impute=False)
    assert res.old_ic == pytest.approx(ref.old_ic)
    assert res.new_ic == pytest.approx(ref.new_ic)
    assert res.old_imputed == res.new_imputed == 0
    # mixed: one DayPairs and one tuple on the same day never impute either
    mixed = replay(OLD, tup_new, impute=True)
    assert mixed.new_ic == pytest.approx(ref.new_ic)
    assert mixed.old_imputed == mixed.new_imputed == 0


def test_raw_pair_counts_exclude_imputed():
    res = replay(OLD, NEW, impute=True)
    n = len(DAYS)
    assert (res.old_pairs, res.new_pairs) == (4 * n, 3 * n)
    assert (res.old_imputed, res.new_imputed) == (0, n)
    assert replay(OLD, NEW, impute=False).new_imputed == 0


def test_imputed_pairs_copy_the_other_sides_conviction_and_excess():
    # tickers on both sides keep their OWN values; only one-sided tickers are imputed
    old = DayPairs(["A", "B", "X"], [0.9, 0.1, 0.5], [0.04, 0.00, -0.06])
    new = DayPairs(["A", "B", "Y"], [0.8, 0.2, 0.6], [0.05, 0.01, 0.07])
    res = replay(old, new, impute=True)
    n = len(DAYS)
    assert (res.old_imputed, res.new_imputed) == (n, n)
    old_c, old_e = [0.9, 0.1, 0.5, 0.6], [0.04, 0.00, -0.06, 0.07]
    new_c, new_e = [0.8, 0.2, 0.6, 0.5], [0.05, 0.01, 0.07, -0.06]
    assert res.old_ic == pytest.approx(rank_ic(old_c * n, old_e * n))
    assert res.new_ic == pytest.approx(rank_ic(new_c * n, new_e * n))


def test_nan_excess_is_never_imputed_and_own_nan_is_kept_raw():
    old = DayPairs(["A", "B", "C", "X"], [0.9, 0.7, 0.5, 0.8], [0.03, 0.02, 0.01, math.nan])
    res = replay(old, NEW, impute=True)
    assert res.new_imputed == 0
    assert res.old_pairs == 4 * len(DAYS)
    # a NaN conviction is not imputed either
    nanc = DayPairs(["A", "B", "C", "X"], [0.9, 0.7, 0.5, math.nan], [0.03, 0.02, 0.01, -0.05])
    assert replay(nanc, NEW, impute=True).new_imputed == 0


def test_misaligned_day_pairs_are_rejected_not_mispaired():
    for bad in (DayPairs(["A", "B"], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3]),
                DayPairs(["A", "B", "C"], [0.1, 0.2, 0.3], [0.1, 0.2]),
                DayPairs(["A", "A", "C"], [0.1, 0.2, 0.3], [0.1, 0.2, 0.3])):
        with pytest.raises(ValueError):
            replay(bad, NEW, impute=True)
        with pytest.raises(ValueError):
            replay(NEW, bad, impute=True)


def test_failed_or_empty_side_means_no_imputation_and_day_not_comparable():
    def run(day, prompt):
        if prompt == "new" and day.day % 2 == 0:
            return None
        return OLD if prompt == "old" else NEW
    res = paired_replay("agent_a", "old", "new", DAYS, run, impute=True, min_days=5)
    assert res.new_failures > 0
    assert res.n_days == len(DAYS) - res.new_failures
    assert res.old_imputed == 0
    assert res.new_imputed == res.n_days
    assert res.old_pairs == 4 * res.n_days  # failed days contribute nothing to either side

    def run_empty(day, prompt):
        return OLD if prompt == "old" else DayPairs([], [], [])
    with pytest.raises(InsufficientReplayDays):
        paired_replay("agent_a", "old", "new", DAYS, run_empty, impute=True)


def test_min_days_parameter_lowers_the_required_days():
    short = DAYS[:10]
    with pytest.raises(InsufficientReplayDays):
        paired_replay("agent_a", "old", "new", short, runner(OLD, NEW))
    res = paired_replay("agent_a", "old", "new", short, runner(OLD, NEW), min_days=10)
    assert res.n_days == 10
    with pytest.raises(InsufficientReplayDays):
        paired_replay("agent_a", "old", "new", short, runner(OLD, NEW), min_days=11)


def _prices(day, nxt, ret):
    return pd.Series([100.0, 100.0 * (1 + ret)], index=[day, nxt])


def test_cli_scorer_returns_tickers():
    from atlas.cli import Replay
    from atlas.schemas import AgentResult

    d0, d1 = date(2024, 1, 2), date(2024, 1, 3)
    rets = {"AAA": 0.02, "BBB": -0.01, "SPY": 0.0}

    class Store:
        def outcome_prices(self, ticker, day, nxt):
            return _prices(day, nxt, rets[ticker])

    run = SimpleNamespace(store=Store(), cost_frac=0.0,
                          pipeline=SimpleNamespace(specs={"agent_a": SimpleNamespace(benchmark=None)}))
    rep = Replay(run=run, pipeline=run.pipeline, all_days=[d0, d1], history={}, results={})
    picks = [SimpleNamespace(ticker="AAA", direction="long", conviction=0.6),
             SimpleNamespace(ticker="BBB", direction="short", conviction=0.4),
             SimpleNamespace(ticker="AAA", direction="long", conviction=0.9),  # duplicate: last wins
             SimpleNamespace(ticker="BBB", direction="none", conviction=0.0)]
    res = SimpleNamespace(agent="agent_a", output=SimpleNamespace(picks=picks))
    out = rep.scorer(res, d0)
    assert isinstance(out, DayPairs)
    assert out.tickers == ["AAA", "BBB"]
    assert out.convictions == [0.9, -0.4]
    assert out.excess == pytest.approx([0.02, -0.01])  # long-frame excess
    # no next day / no output: an empty DayPairs
    assert rep.scorer(res, d1) == DayPairs([], [], [])


# ---------------------------------------------------------------- oracle-abstainer regression
# 8 tickers/day, 40 days, excess ~ N(0, 2%), mean paired diff over 10 seeds.
# Measured with the rejected conviction-0 imputation: oracle long-only +0.807, oracle signed
# +0.598 (and +0.744 with no imputation). Copying the other side's conviction gives 0.000.

def _sim(scn, signed, seed):
    rnd = random.Random(seed)
    store = {}
    for d in DAYS:
        ex = [rnd.gauss(0, 0.02) for _ in range(8)]
        cv = [rnd.uniform(0.3, 1) * (rnd.choice([1, -1]) if signed else 1) for _ in range(8)]
        if scn == "oracle":
            keep = [i for i in range(8) if ex[i] * cv[i] > 0]
        elif scn == "random":
            keep = [i for i in range(8) if rnd.random() < 0.5]
        else:  # "better": same picks, the new convictions track the (direction-adjusted) excess
            keep = list(range(8))
        sg = [1 if c > 0 else -1 for c in cv]
        ncv = [sgn * (0.3 * abs(c) + 0.7 * min(1, max(0.01, (x * sgn + 0.06) / 0.12)))
               for c, x, sgn in zip(cv, ex, sg)]
        store[d] = (cv, ex, keep, ncv)

    def run(d, prompt):
        cv, ex, keep, ncv = store[d]
        t = [f"T{i}" for i in range(8)]
        if prompt == "old":
            return DayPairs(t, cv, ex)
        use = ncv if scn == "better" else cv
        return DayPairs([t[i] for i in keep], [use[i] for i in keep], [ex[i] for i in keep])
    return run


def _mean_diff(scn, signed, impute):
    return sum(paired_replay("a", "old", "new", DAYS, _sim(scn, signed, s), impute=impute,
                             min_days=20).diff for s in range(10)) / 10


@pytest.mark.parametrize("signed", [False, True])
def test_oracle_abstainer_earns_nothing_when_imputed(signed):
    assert _mean_diff("oracle", signed, True) == pytest.approx(0.0, abs=1e-9)
    if signed:  # sanity: the simulation does reward abstention without imputation
        assert _mean_diff("oracle", True, False) > 0.4


@pytest.mark.parametrize("signed", [False, True])
def test_random_dropping_is_a_null(signed):
    assert abs(_mean_diff("random", signed, True)) < 0.02


@pytest.mark.parametrize("signed", [False, True])
def test_genuinely_better_ranking_of_shared_picks_still_registers(signed):
    assert _mean_diff("better", signed, True) > 0.3
    assert _mean_diff("better", signed, True) == pytest.approx(_mean_diff("better", signed, False))
