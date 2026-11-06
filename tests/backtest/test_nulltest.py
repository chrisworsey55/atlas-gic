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


def test_bare_distribution_dead_zone_and_keep_margin_raise():
    d = NullDistribution(np.arange(-10, 11) / 10.0)
    assert d.quantile(1.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        d.dead_zone()
    with pytest.raises(ValueError):
        d.keep_margin()


def hand_frame():
    rng = np.random.default_rng(11)
    rows = []
    for a in range(3):
        n = 6 + a
        rows.append(pd.DataFrame({"agent": f"a{a}", "date": pd.date_range("2024-01-01", periods=n).date,
                                  "ticker": "AAA", "conviction": rng.uniform(size=n),
                                  "excess": rng.normal(size=n), "scored": True}))
    return pd.concat(rows, ignore_index=True)


def roll_shuffler():
    calls = []

    def fn(x, rng):
        calls.append(1)
        return np.roll(x, (len(calls) % 4) + 1)
    return fn


def reference(f, n_shuffles, k, warmup, fn):
    from scipy.stats import spearmanr
    data = {a: (g.conviction.to_numpy(), g.excess.to_numpy()) for a, g in f.groupby("agent")}
    devs, diffs = [], []
    for _ in range(n_shuffles):
        ics = {}
        for a, (c, e) in data.items():
            ia = spearmanr(fn(c, None), e).statistic
            ib = spearmanr(fn(c, None), e).statistic
            ics[a] = ia
            diffs.append(abs(ia - ib))
        m = np.mean(list(ics.values()))
        for a, ic in ics.items():
            n = len(data[a][0])
            devs.append(abs((n * ic + k * m) / (n + k) - m))
    return np.quantile(devs, 0.95), np.quantile(diffs, 0.95), devs


def test_dead_zone_and_keep_margin_match_hand_recomputation():
    f = hand_frame()
    d = null_test(f, 4, seed=0, k=4.0, warmup_picks=5, shuffle_fn=roll_shuffler())
    dz, km, devs = reference(f, 4, 4.0, 5, roll_shuffler())
    assert d.dead_zone() == pytest.approx(dz)
    assert d.keep_margin() == pytest.approx(km)
    # shrinkage matters at this n: unshrunk deviations give a different threshold
    assert d.dead_zone() != pytest.approx(d.quantile(0.95))


def test_shrinkage_shrinks_deviations():
    f = hand_frame()
    k0 = null_test(f, 4, seed=0, k=1e-9, warmup_picks=5, shuffle_fn=roll_shuffler())
    k8 = null_test(f, 4, seed=0, k=8.0, warmup_picks=5, shuffle_fn=roll_shuffler())
    assert k8.dead_zone() < k0.dead_zone()


def test_window_uses_last_rows_and_windowed_eligibility():
    f = frame(n_agents=3, n=60)
    with pytest.raises(ValueError):
        null_test(f, 10, seed=1, window=20)  # windowed n=20 < warmup 30
    w = null_test(f, 40, seed=2, window=40)
    full = null_test(f, 40, seed=2)
    assert w.dead_zone() > full.dead_zone()  # smaller n => less shrinkage, wider null


def test_dead_zone_false_move_rate_on_fresh_null_data():
    from atlas.scoring.metrics import rank_ic
    from atlas.scoring.weights import AgentStats, update_weights
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    def make(seed, n_agents, n):
        rng = np.random.default_rng(seed)
        return pd.concat([pd.DataFrame({
            "agent": f"a{i}", "date": pd.date_range("2024-01-01", periods=n).date, "ticker": "AAA",
            "conviction": rng.choice(levels, n), "excess": rng.normal(size=n), "scored": True})
            for i in range(n_agents)], ignore_index=True)

    d = null_test(make(1, 12, 150), 100, seed=3, window=60)
    dz = d.dead_zone()
    moved = total = 0
    for r in range(60):
        fresh = make(1000 + r, 12, 60)
        stats = [AgentStats(a, len(g), rank_ic(g.conviction, g.excess)) for a, g in fresh.groupby("agent")]
        w = {s.agent: 1.0 for s in stats}
        out = update_weights(w, stats, dz, 20.0, 30)
        moved += sum(out[a] != 1.0 for a in w)
        total += len(w)
    assert 0.02 <= moved / total <= 0.09, moved / total


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
