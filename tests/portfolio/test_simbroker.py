from datetime import date

import numpy as np
import pandas as pd
import pytest

from atlas.data.provider import PRICE_COLUMNS
from atlas.data.store import PriceStore
from atlas.portfolio.models import Fill, Order, Portfolio
from atlas.portfolio.simbroker import SimBroker, size_order

DEC = date(2024, 1, 2)    # decision day (Tue)
NEXT = date(2024, 1, 3)   # next trading day
FRI = date(2024, 1, 5)
MON = date(2024, 1, 8)


class FakePrices:
    """bars: {ticker: {date: open}}"""

    def __init__(self, bars):
        self.bars = bars

    def fetch(self, tickers, start, end):
        rows = []
        for t in tickers:
            for d, o in self.bars.get(t, {}).items():
                if start <= d <= end:
                    rows.append((d, t, o, o + 3, o - 3, o + 0.7, o + 0.7, 1000))
        return pd.DataFrame(rows, columns=PRICE_COLUMNS)


class FakeMacro:
    def fetch(self, series_id, asof):
        return pd.Series(dtype=float)


def make_store(tmp_path, bars):
    return PriceStore(FakePrices(bars), FakeMacro(), tmp_path / "cache")


def broker(store, cps=0.005, cmin=1.0, bps=10.0):
    return SimBroker(store, commission_per_share=cps, commission_min=cmin, slippage_bps=bps)


@pytest.fixture
def store(tmp_path):
    return make_store(tmp_path, {
        "AAA": {DEC: 90.0, NEXT: 100.0},
        "GAP": {DEC: 100.0, NEXT: 120.0},
        "OLD": {DEC: 50.0},  # no later bar
        "FRI": {FRI: 70.0, MON: 80.0},
    })


def test_buy_fills_at_next_open_plus_slippage(store):
    fills, dropped = broker(store).submit([Order("AAA", "buy", 10)], DEC)
    assert dropped == []
    assert len(fills) == 1
    f = fills[0]
    assert f.date == NEXT
    assert f.price == pytest.approx(100.0 * 1.001)


def test_sell_fills_at_next_open_minus_slippage(store):
    pf = Portfolio(cash=0.0, positions={"AAA": 10})
    fills, dropped = broker(store).submit([Order("AAA", "sell", 10)], DEC, portfolio=pf)
    assert dropped == []
    assert fills[0].date == NEXT
    assert fills[0].price == pytest.approx(100.0 * 0.999)


def test_commission_charged_per_share_with_minimum(store):
    b = broker(store, cps=0.01, cmin=1.0)
    small, _ = b.submit([Order("AAA", "buy", 10)], DEC)      # 0.10 -> min 1.0
    big, _ = b.submit([Order("AAA", "buy", 500)], DEC)       # 5.00
    assert small[0].commission == pytest.approx(1.0)
    assert big[0].commission == pytest.approx(5.0)


def test_missing_next_open_drops_order_with_reason(store):
    pf = Portfolio(cash=10_000.0, positions={})
    fills, dropped = broker(store).submit(
        [Order("OLD", "buy", 5), Order("NOPE", "buy", 5)], DEC, portfolio=pf)
    assert fills == []
    assert len(dropped) == 2
    assert all(isinstance(r, str) and r for _, r in dropped)
    assert "next open" in dropped[0][1]


def test_tiny_equity_rounds_to_zero_shares_and_order_dropped():
    equity = 100.0
    shares = size_order(equity * 0.15, 400.0)
    assert shares == 0
    assert size_order(1000.0, 400.0) == 2
    assert size_order(-5.0, 10.0) == 0
    assert size_order(100.0, 0.0) == 0
    assert size_order(float("nan"), 10.0) == 0
    assert size_order(float("inf"), 10.0) == 0
    orders = [Order("AAA", "buy", s) for s in [shares] if s > 0]
    assert orders == []


def test_cash_never_negative_after_fills(store):
    # sized at decision close $100 (100 shares = $10,000), but opens at $120
    pf = Portfolio(cash=10_000.0, positions={})
    fills, dropped = broker(store).submit([Order("GAP", "buy", 100)], DEC, portfolio=pf)
    assert len(fills) == 1 and fills[0].order.shares < 100
    assert fills[0].price == pytest.approx(120.0 * 1.001)
    for f in fills:
        pf.apply(f)
    assert pf.cash >= 0
    assert pf.positions["GAP"] == fills[0].order.shares
    # one more share would not have been affordable
    px = fills[0].price
    n = fills[0].order.shares + 1
    assert n * px + max(1.0, 0.005 * n) > 10_000.0


def test_buy_dropped_when_not_even_one_share_affordable(store):
    pf = Portfolio(cash=50.0, positions={})
    fills, dropped = broker(store).submit([Order("GAP", "buy", 10)], DEC, portfolio=pf)
    assert fills == []
    assert len(dropped) == 1 and dropped[0][1]


def test_sells_processed_before_buys_free_cash(store):
    pf = Portfolio(cash=0.0, positions={"AAA": 100})
    fills, dropped = broker(store).submit(
        [Order("GAP", "buy", 50), Order("AAA", "sell", 100)], DEC, portfolio=pf)
    assert dropped == []
    assert [f.order.side for f in fills] == ["sell", "buy"]
    for f in fills:
        pf.apply(f)
    assert pf.cash >= 0 and pf.positions == {"GAP": 50}


def test_sell_exceeding_holdings_rejected(store):
    pf = Portfolio(cash=0.0, positions={"AAA": 5})
    fills, dropped = broker(store).submit(
        [Order("AAA", "sell", 6), Order("GAP", "sell", 1)], DEC, portfolio=pf)
    assert fills == []
    assert len(dropped) == 2 and all(r for _, r in dropped)


def test_multiple_sells_same_ticker_cannot_exceed_holdings(store):
    pf = Portfolio(cash=0.0, positions={"AAA": 10})
    fills, dropped = broker(store).submit(
        [Order("AAA", "sell", 6), Order("AAA", "sell", 6)], DEC, portfolio=pf)
    assert len(fills) == 1 and len(dropped) == 1


def test_numpy_int_shares_accepted(store):
    fills, _ = broker(store).submit([Order("AAA", "buy", np.int64(3))], DEC)
    assert fills[0].order.shares == 3
    assert type(fills[0].order.shares) is int


def test_zero_or_negative_shares_dropped(store):
    fills, dropped = broker(store).submit(
        [Order("AAA", "buy", 0), Order("AAA", "buy", -2)], DEC)
    assert fills == [] and len(dropped) == 2


def test_broker_does_not_mutate_portfolio(store):
    pf = Portfolio(cash=10_000.0, positions={"AAA": 10})
    broker(store).submit([Order("AAA", "sell", 10), Order("GAP", "buy", 10)], DEC, portfolio=pf)
    assert pf.cash == 10_000.0 and pf.positions == {"AAA": 10}


def test_portfolio_apply_round_trip_flat_returns_cash_minus_costs():
    pf = Portfolio(cash=1000.0, positions={})
    buy = Fill(Order("X", "buy", 10), NEXT, 10.0, 1.0)
    sell = Fill(Order("X", "sell", 10), NEXT, 10.0, 1.0)
    pf.apply(buy)
    assert pf.cash == pytest.approx(1000 - 100 - 1)
    assert pf.positions == {"X": 10}
    pf.apply(sell)
    assert pf.cash == pytest.approx(998.0)
    assert pf.positions == {}


def test_portfolio_equity_valuation():
    pf = Portfolio(cash=100.0, positions={"A": 2, "B": 3})
    assert pf.equity({"A": 10.0, "B": 20.0}) == pytest.approx(100 + 20 + 60)
    assert Portfolio(cash=5.0, positions={}).equity({}) == 5.0


def test_friday_decision_fills_monday_at_monday_open(store):
    b = broker(store, bps=25.0)
    pf = Portfolio(cash=0.0, positions={"FRI": 4})
    buys, _ = b.submit([Order("FRI", "buy", 1)], FRI)
    sells, _ = b.submit([Order("FRI", "sell", 4)], FRI, portfolio=pf)
    assert buys[0].date == MON and sells[0].date == MON
    assert buys[0].price == 80.0 * 1.0025
    assert sells[0].price == 80.0 * 0.9975


def test_decision_on_last_bar_with_data_present_is_dropped_with_reason(store):
    fills, dropped = broker(store).submit([Order("OLD", "buy", 1)], DEC)
    assert fills == []
    assert len(dropped) == 1 and dropped[0][1]
