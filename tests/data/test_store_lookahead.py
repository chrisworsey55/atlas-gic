from datetime import date, timedelta

import pandas as pd
import pytest

from atlas.data.provider import PRICE_COLUMNS
from atlas.data.store import PriceStore

START = date(2024, 1, 1)  # Monday
TICKERS = ["SPY", "AAA", "BBB"]


class FakePrices:
    """60 calendar days of weekday bars; counts fetch calls."""

    def __init__(self):
        self.calls = []

    def fetch(self, tickers, start, end):
        self.calls.append((tuple(tickers), start, end))
        rows = []
        for t_i, t in enumerate(tickers):
            for i in range(60):
                d = START + timedelta(days=i)
                if d.weekday() >= 5 or not (start <= d <= end):
                    continue
                p = 100.0 + i + t_i * 10
                rows.append((d, t, p, p + 1, p - 1, p + 0.5, p + 0.25, 1000 + i))
        return pd.DataFrame(rows, columns=PRICE_COLUMNS)


class FakeMacro:
    def fetch(self, series_id, asof):
        idx = [asof - timedelta(days=30), asof - timedelta(days=1), asof + timedelta(days=5)]
        return pd.Series([1.0, float(asof.day), 99.0], index=idx)


@pytest.fixture
def prov():
    return FakePrices()


@pytest.fixture
def store(prov, tmp_path):
    return PriceStore(prov, FakeMacro(), tmp_path / "cache")


@pytest.mark.parametrize("asof", [
    date(2024, 1, 31),   # Wednesday
    date(2024, 2, 3),    # Saturday
    date(2024, 2, 4),    # Sunday
    date(2024, 1, 15),
    date(2023, 12, 20),  # before data starts
    date(2024, 3, 25),   # after data ends
])
def test_snapshot_never_contains_rows_after_asof(store, prov, asof):
    snap = store.snapshot(asof, TICKERS, lookback_days=20)
    assert snap.asof == asof
    if not snap.prices.empty:
        assert snap.prices["date"].max() <= asof
        assert set(snap.prices["ticker"]) <= set(TICKERS)
        for t in TICKERS:
            assert snap.prices[snap.prices.ticker == t]["date"].max() <= asof
    # the provider itself was never asked for data past asof
    assert all(end <= asof for _, _, end in prov.calls)


def test_snapshot_not_polluted_by_earlier_forward_access(store):
    store.outcome_prices("AAA", date(2024, 1, 2), date(2024, 2, 20))  # caches the future
    snap = store.snapshot(date(2024, 1, 10), ["AAA"], 30)
    assert snap.prices["date"].max() <= date(2024, 1, 10)
    assert len(snap.prices) == 8  # Jan 1..10 weekdays


def test_snapshot_before_data_starts_is_empty_not_filled(store):
    snap = store.snapshot(date(2023, 12, 20), TICKERS, 10)
    assert snap.prices.empty


def test_snapshot_gaps_are_not_forward_filled(store):
    snap = store.snapshot(date(2024, 2, 4), ["AAA"], 10)  # Sunday
    assert snap.prices["date"].max() == date(2024, 2, 2)  # Friday, no synthetic rows
    assert all(d.weekday() < 5 for d in snap.prices["date"])


def test_snapshot_lookback_window(store):
    snap = store.snapshot(date(2024, 1, 31), ["AAA"], 7)
    assert snap.prices["date"].min() >= date(2024, 1, 24)


def test_snapshot_macro_uses_vintage_as_of_date(store):
    a = store.snapshot(date(2024, 1, 10), ["AAA"], 5, macro_series=["CPI"])
    b = store.snapshot(date(2024, 1, 20), ["AAA"], 5, macro_series=["CPI"])
    assert a.macro["CPI"].loc[date(2024, 1, 9)] == 10.0
    assert b.macro["CPI"].loc[date(2024, 1, 19)] == 20.0
    assert max(a.macro["CPI"].index) <= date(2024, 1, 10)  # future obs dropped


def test_next_open_is_strictly_after_given_day(store):
    d, px = store.next_open("AAA", date(2024, 1, 2))
    assert d == date(2024, 1, 3)
    assert px == 102.0  # open on day index 2
    # from a Friday, skips the weekend
    assert store.next_open("AAA", date(2024, 1, 5))[0] == date(2024, 1, 8)
    # from a non-trading day
    assert store.next_open("AAA", date(2024, 1, 6))[0] == date(2024, 1, 8)


def test_next_open_returns_none_when_no_later_bar(store):
    assert store.next_open("AAA", date(2024, 2, 29)) is None  # last bar is Feb 29
    assert store.next_open("AAA", date(2024, 6, 1)) is None


def test_outcome_prices_and_trading_days(store):
    s = store.outcome_prices("AAA", date(2024, 1, 2), date(2024, 1, 5))
    assert list(s.index) == [date(2024, 1, d) for d in (2, 3, 4, 5)]
    days = store.trading_days(date(2024, 1, 1), date(2024, 1, 7))
    assert days == [date(2024, 1, d) for d in (1, 2, 3, 4, 5)]


def test_provider_cached_to_disk_second_call_makes_no_fetch(prov, tmp_path):
    cache = tmp_path / "c"
    s1 = PriceStore(prov, FakeMacro(), cache)
    first = s1.snapshot(date(2024, 1, 31), ["AAA", "BBB"], 20)
    n = len(prov.calls)
    assert n > 0
    assert (cache / "AAA.csv").exists() and (cache / "BBB.csv").exists()
    s1.snapshot(date(2024, 1, 31), ["AAA", "BBB"], 20)
    assert len(prov.calls) == n
    # a fresh store over the same dir also hits disk only
    s2 = PriceStore(prov, FakeMacro(), cache)
    again = s2.snapshot(date(2024, 1, 31), ["AAA", "BBB"], 20)
    assert len(prov.calls) == n
    pd.testing.assert_frame_equal(first.prices, again.prices)


class GrowingPrices(FakePrices):
    """Provider whose data ends at `last` (mutable); can also fail with empty frames."""

    def __init__(self, last):
        super().__init__()
        self.last = last
        self.fail = False

    def fetch(self, tickers, start, end):
        if self.fail:
            self.calls.append((tuple(tickers), start, end))
            return pd.DataFrame(columns=PRICE_COLUMNS)
        return super().fetch(tickers, start, min(end, self.last))


def test_live_edge_not_poisoned_when_provider_gains_bars(tmp_path):
    prov = GrowingPrices(date(2024, 1, 10))
    st = PriceStore(prov, FakeMacro(), tmp_path / "c")
    assert st.next_open("AAA", date(2024, 1, 10)) is None
    prov.last = date(2024, 1, 20)
    assert st.next_open("AAA", date(2024, 1, 10))[0] == date(2024, 1, 11)
    assert not st.snapshot(date(2024, 1, 19), ["AAA"], 5).prices.empty


def test_empty_response_records_no_coverage_and_refetches(tmp_path):
    prov = GrowingPrices(date(2024, 2, 29))
    prov.fail = True
    st = PriceStore(prov, FakeMacro(), tmp_path / "c")
    assert st.snapshot(date(2024, 1, 19), ["AAA"], 5).prices.empty
    n = len(prov.calls)
    prov.fail = False
    snap = st.snapshot(date(2024, 1, 19), ["AAA"], 5)
    assert len(prov.calls) == n + 1
    assert not snap.prices.empty
    # a later failure leaves existing coverage untouched
    prov.fail = True
    assert not st.snapshot(date(2024, 1, 19), ["AAA"], 5).prices.empty
    assert len(prov.calls) == n + 1


def test_loaded_frames_are_memoised_and_invalidated_when_the_cache_file_changes(store, monkeypatch):
    """Repeated as-of reads must not re-parse the CSV (the backtest's scoring loop reads the
    same files thousands of times), but a refetch that rewrites the file must be seen."""
    store.snapshot(date(2024, 1, 20), TICKERS, lookback_days=10)
    reads = []
    real = pd.read_csv
    monkeypatch.setattr(pd, "read_csv", lambda *a, **k: reads.append(a) or real(*a, **k))
    for _ in range(3):
        store.outcome_prices("AAA", date(2024, 1, 15), date(2024, 1, 19))
    assert reads == []
    # extend coverage: the file is rewritten and the new rows must be visible
    s = store.outcome_prices("AAA", date(2024, 1, 15), date(2024, 2, 9))
    assert max(s.index) == date(2024, 2, 9)
    # callers cannot corrupt the memoised frame
    store._load("AAA").iloc[0:0]
    df = store._load("AAA")
    df.loc[df.index[0], "close"] = -1.0
    assert store._load("AAA")["close"].iloc[0] != -1.0
