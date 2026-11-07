import json
import math
from datetime import date, timedelta

import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

from atlas.scoring.store import COLUMNS
from atlas.scoring.weights import (AgentStats, WeightBook, collect_stats, shrunk_ic,
                                   update_weights, W_MAX, W_MIN)

NAN = float("nan")
K, WARM, DZ = 20.0, 30, 0.05


def S(agent, ic, n=100):
    return AgentStats(agent, n, ic)


def test_shrinkage_pulls_small_samples_toward_mean():
    assert shrunk_ic(0.5, 5, 0.1, 20.0) == pytest.approx((5 * 0.5 + 20 * 0.1) / 25)
    small = shrunk_ic(0.5, 5, 0.1, 20.0)
    big = shrunk_ic(0.5, 500, 0.1, 20.0)
    assert 0.1 < small < big < 0.5
    assert abs(small - 0.1) < abs(small - 0.5)


def test_inside_dead_zone_weights_unchanged():
    stats = [S("a", 0.11), S("b", 0.10), S("c", 0.09)]
    w = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert update_weights(w, stats, 0.05, K, WARM) == w


def test_above_dead_zone_multiplies_by_1_05():
    stats = [S("a", 0.5, 10_000), S("b", 0.0, 10_000), S("c", 0.0, 10_000)]
    out = update_weights({"a": 1.0, "b": 1.0, "c": 1.0}, stats, DZ, K, WARM)
    assert out["a"] == pytest.approx(1.05)


def test_below_dead_zone_multiplies_by_0_95():
    stats = [S("a", -0.5, 10_000), S("b", 0.0, 10_000), S("c", 0.0, 10_000)]
    out = update_weights({"a": 1.0, "b": 1.0, "c": 1.0}, stats, DZ, K, WARM)
    assert out["a"] == pytest.approx(0.95)


def test_warming_up_agents_never_change():
    stats = [AgentStats("a", WARM - 1, 0.9), S("b", 0.0), S("c", 0.0), S("d", 0.3)]
    out = update_weights({"a": 1.3, "b": 1.0, "c": 1.0, "d": 1.0}, stats, DZ, K, WARM)
    assert out["a"] == 1.3
    assert out["d"] == pytest.approx(1.05)  # warming-up a excluded from group mean


def test_nan_ic_treated_as_no_signal_weight_unchanged():
    stats = [S("a", NAN), S("b", 0.4), S("c", 0.0), S("d", 0.0)]
    out = update_weights({"a": 1.2, "b": 1.0, "c": 1.0, "d": 1.0}, stats, DZ, K, WARM)
    assert out["a"] == 1.2
    assert out["b"] == pytest.approx(1.05)


def test_fewer_than_two_eligible_changes_nothing():
    out = update_weights({"a": 1.0, "b": 1.0}, [S("a", 0.9), S("b", NAN)], DZ, K, WARM)
    assert out == {"a": 1.0, "b": 1.0}


def test_dead_zone_none_freezes(caplog):
    w = {"a": 1.0, "b": 1.0}
    with caplog.at_level("INFO"):
        out = update_weights(w, [S("a", 0.9), S("b", -0.9)], None, K, WARM)
    assert out == w and out is not w
    assert "frozen" in caplog.text.lower()


def test_missing_agent_initialised_to_one_then_updated():
    stats = [S("a", 0.5, 10_000), S("b", 0.0, 10_000), S("c", 0.0, 10_000)]
    out = update_weights({"b": 1.0}, stats, DZ, K, WARM)
    assert out["a"] == pytest.approx(1.05) and out["c"] == pytest.approx(0.95)


def test_input_not_mutated():
    w = {"a": 1.0, "b": 1.0, "c": 1.0}
    snap = dict(w)
    update_weights(w, [S("a", 0.5), S("b", 0.0), S("c", 0.0)], DZ, K, WARM)
    assert w == snap


def test_bounds_clamped_at_edges():
    stats = [S("a", 0.5, 10_000), S("b", -0.5, 10_000), S("c", 0.0, 10_000)]
    out = update_weights({"a": W_MAX, "b": W_MIN, "c": 1.0}, stats, DZ, K, WARM)
    assert out["a"] == W_MAX and out["b"] == W_MIN


@settings(max_examples=100, deadline=None)
@given(rounds=st.lists(
    st.lists(st.tuples(st.floats(-1, 1) | st.just(NAN), st.sampled_from([0, 1, 29, 30, 60, 10**7])),
             min_size=1, max_size=6), min_size=1, max_size=40),
    start=st.lists(st.sampled_from([0.3, 2.5, 1.0, 0.31, 2.49]), min_size=6, max_size=6),
    dz=st.floats(0, 0.5))
def test_weights_always_within_0_3_and_2_5(rounds, start, dz):
    w = {f"a{i}": start[i] for i in range(6)}
    for r in rounds:
        stats = [AgentStats(f"a{i}", n, ic) for i, (ic, n) in enumerate(r)]
        w = update_weights(w, stats, dz, K, WARM)
        assert all(0.3 <= v <= 2.5 for v in w.values())


def test_weightbook_round_trip_and_missing(tmp_path):
    wb = WeightBook(tmp_path / "sub" / "w.json")
    assert wb.load() == {}
    wb.save({"a": 1.1, "b": 0.7})
    assert wb.load() == {"a": 1.1, "b": 0.7}


def test_weightbook_atomic_stray_temp_ignored(tmp_path):
    p = tmp_path / "w.json"
    wb = WeightBook(p)
    wb.save({"a": 1.0})
    (tmp_path / "w.json.tmp").write_text("{garbage")
    assert wb.load() == {"a": 1.0}
    wb.save({"a": 1.2})
    assert json.loads(p.read_text()) == {"a": 1.2}
    assert wb.load() == {"a": 1.2}


class FakeScores:
    def __init__(self, df):
        self.df, self.calls = df, []

    def picks_frame(self, agent, horizon):
        self.calls.append((agent, horizon))
        d = self.df[self.df.agent == agent].copy() if agent is not None else self.df.copy()
        d["horizon"] = horizon
        return d


def frame(agent, n, unscored_tail=0, anti=False):
    d0 = date(2024, 1, 1)
    rows = []
    for i in range(n + unscored_tail):
        conv = (i % 7 + 1) / 10
        ex = conv * (-1 if anti else 1) * 0.01 + i * 1e-6
        scored = i < n
        rows.append((agent, d0 + timedelta(days=i), "T", "long", conv, 1,
                     ex if scored else NAN, ex if scored else NAN, scored))
    return pd.DataFrame(rows, columns=COLUMNS)


def test_collect_stats_only_scored_rows():
    df = pd.concat([frame("a", 20, unscored_tail=15), frame("b", 10, anti=True)])
    st_ = {s.agent: s for s in collect_stats(FakeScores(df), ["a", "b"], asof_window=60)}
    assert st_["a"].n_picks == 20 and st_["b"].n_picks == 10
    assert st_["a"].ic > 0.9 and st_["b"].ic < -0.9


def test_collect_stats_rolling_window_keeps_most_recent():
    d = frame("a", 50)
    d.loc[d.index[:30], "excess"] = -d.loc[d.index[:30], "conviction"]  # old regime: negative IC
    d["signed_excess"] = d["excess"]
    s = collect_stats(FakeScores(d), ["a"], asof_window=20)[0]
    assert s.n_picks == 20 and s.ic > 0.9


def test_collect_stats_max_date_excludes_later_picks_before_windowing():
    d = frame("a", 50)
    d.loc[d.index[30:], "excess"] = -d.loc[d.index[30:], "conviction"]  # later regime: negative IC
    d["signed_excess"] = d["excess"]
    cut = d["date"].iloc[29]
    s = collect_stats(FakeScores(d), ["a"], asof_window=20, max_date=cut)[0]
    assert s.n_picks == 20 and s.ic > 0.9  # the window ends at the cut, not at the last pick
    assert collect_stats(FakeScores(d), ["a"], asof_window=20)[0].ic < -0.9


def test_collect_stats_nan_when_undefined_and_no_picks():
    d = frame("a", 2)
    out = {s.agent: s for s in collect_stats(FakeScores(d), ["a", "ghost"], asof_window=60)}
    assert math.isnan(out["a"].ic) and out["a"].n_picks == 2
    assert out["ghost"].n_picks == 0 and math.isnan(out["ghost"].ic)


def test_collect_stats_horizon_non_overlapping():
    d = frame("a", 40)
    s = collect_stats(FakeScores(d), ["a"], asof_window=60, horizon=5)[0]
    assert s.n_picks == 8


def test_identical_ics_zero_dead_zone_unchanged():
    w = {"a": 1.0, "b": 1.0}
    stats = [AgentStats("a", 100, 0.1), AgentStats("b", 100, 0.1)]
    assert update_weights(w, stats, dead_zone=0.0, k=K, warmup_picks=WARM) == w


def test_exact_dead_zone_boundary_is_strict():
    w = {"a": 1.0, "b": 1.0}
    stats = [AgentStats("a", 100, 0.5), AgentStats("b", 100, -0.5)]
    # k=0 -> shrunk == raw exactly; group mean 0.0; |deviation| == 0.5 == dead_zone
    assert update_weights(w, stats, dead_zone=0.5, k=0.0, warmup_picks=WARM) == w
    out = update_weights(w, stats, dead_zone=0.4999999, k=0.0, warmup_picks=WARM)
    assert out["a"] == pytest.approx(1.05) and out["b"] == pytest.approx(0.95)


@settings(max_examples=200, deadline=None)
@given(data=st.lists(st.tuples(st.floats(-1, 1) | st.just(NAN), st.sampled_from([0, 5, 29, 30, 100, 10**6]),
                               st.floats(0.3, 2.5)), min_size=1, max_size=8),
       dz=st.floats(0, 0.6), k=st.floats(0, 50))
def test_move_rule_exact(data, dz, k):
    w = {f"a{i}": d[2] for i, d in enumerate(data)}
    stats = [AgentStats(f"a{i}", d[1], d[0]) for i, d in enumerate(data)]
    out = update_weights(w, stats, dz, k, WARM)
    elig = [s for s in stats if s.n_picks >= WARM and not math.isnan(s.ic)]
    mean = sum(s.ic for s in elig) / len(elig) if elig else 0.0
    for s in stats:
        before, after = w[s.agent], out[s.agent]
        if len(elig) < 2 or s not in elig:
            assert after == before
            continue
        dev = (s.n_picks * s.ic + k * mean) / (s.n_picks + k) - mean
        if dev > dz:
            assert after == min(W_MAX, before * 1.05)
        elif dev < -dz:
            assert after == max(W_MIN, before * 0.95)
        else:
            assert after == before
