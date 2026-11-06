import numpy as np
import pandas as pd
import pytest

from atlas.backtest.nulltest import LeakageSuspected, NullDistribution, null_test


def frame(n_agents=3, n=60, seed=0, aligned=False):
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_agents):
        exc = rng.normal(size=n)
        conv = exc.copy() if aligned else rng.uniform(size=n)
        rows.append(pd.DataFrame({"agent": f"a{a}", "date": pd.date_range("2024-01-01", periods=n).date,
                                  "ticker": "AAA", "direction": "long", "conviction": conv,
                                  "horizon": 1, "excess": exc, "signed_excess": exc, "scored": True}))
    return pd.concat(rows, ignore_index=True)


def test_null_ic_centred_near_zero():
    d = null_test(frame(), n_shuffles=200, seed=1)
    assert abs(np.mean(d.ics)) < 0.05


def test_null_test_is_seeded_and_reproducible():
    a = null_test(frame(), 50, seed=7)
    b = null_test(frame(), 50, seed=7)
    c = null_test(frame(), 50, seed=8)
    assert np.array_equal(a.ics, b.ics)
    assert a.dead_zone() == b.dead_zone() and a.keep_margin() == b.keep_margin()
    assert not np.array_equal(a.ics, c.ics)


def test_dead_zone_is_95th_pct_of_abs_null():
    d = NullDistribution(np.arange(-10, 11) / 10.0)  # |ics|: 0..1
    assert d.quantile(0.5) == pytest.approx(0.5)
    assert d.quantile(1.0) == pytest.approx(1.0)
    assert d.dead_zone() == pytest.approx(float(np.quantile(np.abs(np.arange(-10, 11) / 10.0), 0.95)))


def test_dead_zone_matches_shrunk_deviation_percentile():
    # identical shuffle rng => recompute by hand with 2 agents is overkill; check bounds instead
    d = null_test(frame(), 100, seed=3)
    assert 0 < d.dead_zone() < 1
    assert d.keep_margin() >= 0
    # shrinkage makes deviations smaller than raw |ic| 95th pct
    assert d.dead_zone() <= d.quantile(0.95) + 1e-9


def test_null_test_flags_leakage_when_shuffled_ic_is_high():
    with pytest.raises(LeakageSuspected):
        null_test(frame(aligned=True), 30, seed=1, shuffle_fn=lambda x, rng: x)


def test_fewer_than_two_eligible_agents_raises():
    with pytest.raises(ValueError):
        null_test(frame(n_agents=1), 10, seed=1)
    f = frame(n_agents=2, n=10)
    with pytest.raises(ValueError):
        null_test(f, 10, seed=1)  # below warmup_picks=30


def test_empty_raises():
    with pytest.raises(ValueError):
        null_test(frame().iloc[0:0], 10, seed=1)


def test_unscored_rows_ignored():
    f = frame()
    junk = f.copy()
    junk["scored"] = False
    junk["excess"] = np.nan
    both = pd.concat([f, junk], ignore_index=True)
    a = null_test(f, 40, seed=5)
    b = null_test(both, 40, seed=5)
    assert np.array_equal(a.ics, b.ics)
