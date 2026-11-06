import math

from hypothesis import given, settings, strategies as st

from atlas.config import Limits
from atlas.portfolio.models import Order
from atlas.risk.layer import RiskDecision, RiskLayer

LIMITS = Limits(max_position_pct=0.15, max_gross_pct=1.0, max_positions=8, min_order_usd=50,
                max_commission_frac=0.01, daily_loss_pct=0.03, drawdown_kill_pct=0.15,
                long_only=True)
PER_SHARE = 0.005
MIN_COMM = 1.0


def _layer(limits: Limits = LIMITS) -> RiskLayer:
    return RiskLayer(limits, commission_per_share=PER_SHARE, commission_min=MIN_COMM)


def _comm(shares: int) -> float:
    return max(MIN_COMM, PER_SHARE * shares)


def _apply(decision: RiskDecision, positions: dict[str, int], cash: float,
           prices: dict[str, float]) -> tuple[dict[str, int], float]:
    """Execute the decision's orders at the given prices, charging commission."""
    pos = dict(positions)
    for o in decision.orders:
        p = prices[o.ticker]
        if o.side == "sell":
            pos[o.ticker] = pos.get(o.ticker, 0) - o.shares
            cash = cash + (o.shares * p - _comm(o.shares))
        else:
            pos[o.ticker] = pos.get(o.ticker, 0) + o.shares
            cash = cash - (o.shares * p + _comm(o.shares))
    return {t: s for t, s in pos.items() if s != 0}, cash


def _equity(pos: dict[str, int], cash: float, prices: dict[str, float]) -> float:
    return cash + sum(s * prices[t] for t, s in pos.items())


def _run(orders, positions, cash, prices, prior=None, peak=None, limits=LIMITS):
    eq = _equity({t: s for t, s in positions.items() if s}, cash, prices) if all(
        t in prices for t, s in positions.items() if s) else cash
    return _layer(limits).apply(orders, positions, cash, prices,
                                prior_close_equity=eq if prior is None else prior,
                                peak_equity=eq if peak is None else peak)


def _reasons(decision: RiskDecision) -> dict[str, str]:
    return {o.ticker: r for o, r in decision.rejected}


def test_single_position_clipped_to_15pct():
    prices = {"AAPL": 100.0}
    d = _run([Order("AAPL", "buy", 500)], {}, 100_000.0, prices)
    assert not d.halted
    assert len(d.orders) == 1 and d.orders[0].ticker == "AAPL" and d.orders[0].side == "buy"
    pos, cash = _apply(d, {}, 100_000.0, prices)
    eq = _equity(pos, cash, prices)
    value = pos["AAPL"] * 100.0
    assert value <= 0.15 * eq
    assert value > 0.15 * eq - 100.0  # clipped, not zeroed: within one share of the cap


def test_single_position_cap_counts_shares_already_held():
    prices = {"AAPL": 100.0}
    # already 100 shares ($10k) of $100k equity; cap allows ~$15k total -> buy ~49
    d = _run([Order("AAPL", "buy", 300)], {"AAPL": 100}, 90_000.0, prices)
    pos, cash = _apply(d, {"AAPL": 100}, 90_000.0, prices)
    eq = _equity(pos, cash, prices)
    assert pos["AAPL"] * 100.0 <= 0.15 * eq
    assert pos["AAPL"] >= 149


def test_gross_exposure_never_above_100pct():
    limits = LIMITS.model_copy(update={"max_position_pct": 1.0})
    prices = {"AAPL": 100.0, "MSFT": 50.0}
    # weights 0.8 and 0.7 of a $100k book: sums to 1.5, must be clipped to <= 1.0
    orders = [Order("AAPL", "buy", 800), Order("MSFT", "buy", 1400)]
    d = _run(orders, {}, 100_000.0, prices, limits=limits)
    pos, cash = _apply(d, {}, 100_000.0, prices)
    eq = _equity(pos, cash, prices)
    gross = sum(s * prices[t] for t, s in pos.items())
    assert cash >= 0
    assert gross <= 1.0 * eq
    assert pos["AAPL"] == 800  # first buy fits entirely
    assert 0 < pos["MSFT"] < 1400  # second is clipped (integer floor), not dropped


def test_weights_above_one_clipped_to_position_cap():
    prices = {"AAPL": 100.0, "MSFT": 50.0}
    orders = [Order("AAPL", "buy", 800), Order("MSFT", "buy", 1400)]
    d = _run(orders, {}, 100_000.0, prices)
    pos, cash = _apply(d, {}, 100_000.0, prices)
    eq = _equity(pos, cash, prices)
    for t, s in pos.items():
        assert s * prices[t] <= 0.15 * eq
        assert s * prices[t] > 0.15 * eq - prices[t]
    assert cash >= 0


def test_gross_cap_below_one_is_respected():
    limits = LIMITS.model_copy(update={"max_position_pct": 1.0, "max_gross_pct": 0.5})
    prices = {"AAPL": 100.0, "MSFT": 50.0}
    orders = [Order("AAPL", "buy", 300), Order("MSFT", "buy", 1000)]
    d = _run(orders, {}, 100_000.0, prices, limits=limits)
    pos, cash = _apply(d, {}, 100_000.0, prices)
    gross = sum(s * prices[t] for t, s in pos.items())
    assert gross <= 0.5 * _equity(pos, cash, prices)
    assert pos["MSFT"] > 0


def test_position_cap_accounts_for_commission():
    # 120 * 124.999 = 14999.88 is under 15% of the pre-trade 100k but over 15% of the
    # post-commission equity (99_999 -> 14999.85): the commission must be reserved.
    prices = {"AAPL": 124.999}
    d = _run([Order("AAPL", "buy", 500)], {}, 100_000.0, prices)
    pos, cash = _apply(d, {}, 100_000.0, prices)
    assert pos["AAPL"] * prices["AAPL"] <= 0.15 * _equity(pos, cash, prices)
    assert pos["AAPL"] == 119


def test_gross_cap_accounts_for_commission():
    limits = LIMITS.model_copy(update={"max_position_pct": 1.0, "max_gross_pct": 0.5})
    prices = {"AAPL": 499.999}
    d = _run([Order("AAPL", "buy", 500)], {}, 100_000.0, prices, limits=limits)
    pos, cash = _apply(d, {}, 100_000.0, prices)
    assert pos["AAPL"] * prices["AAPL"] <= 0.5 * _equity(pos, cash, prices)
    assert pos["AAPL"] == 99


def test_cash_never_negative_even_if_gross_cap_allows_leverage():
    limits = LIMITS.model_copy(update={"max_position_pct": 5.0, "max_gross_pct": 2.0})
    prices = {"AAPL": 100.0}
    d = _run([Order("AAPL", "buy", 5_000)], {}, 100_000.0, prices, limits=limits)
    pos, cash = _apply(d, {}, 100_000.0, prices)
    assert cash >= 0 and pos["AAPL"] == 999  # 1000 shares + $5 commission would overdraw


def test_ninth_position_rejected():
    held = {f"T{i}": 10 for i in range(8)}
    prices = {t: 100.0 for t in held} | {"NEW": 100.0}
    orders = [Order("NEW", "buy", 10), Order("T0", "buy", 10)]
    d = _run(orders, held, 100_000.0, prices)
    assert [o.ticker for o in d.orders] == ["T0"]  # existing positions may still grow
    assert "max positions" in _reasons(d)["NEW"]


def test_ninth_position_allowed_after_a_full_sell_frees_a_slot():
    held = {f"T{i}": 10 for i in range(8)}
    prices = {t: 100.0 for t in held} | {"NEW": 100.0}
    orders = [Order("NEW", "buy", 10), Order("T0", "sell", 10)]
    d = _run(orders, held, 100_000.0, prices)
    assert [(o.ticker, o.side) for o in d.orders] == [("T0", "sell"), ("NEW", "buy")]


def test_order_under_50_dollars_rejected():
    prices = {"AAPL": 10.0}
    d = _run([Order("AAPL", "buy", 4)], {}, 100_000.0, prices)
    assert d.orders == []
    assert "minimum" in _reasons(d)["AAPL"]


def test_order_with_commission_over_1pct_rejected():
    # $60 order pays the $1 minimum commission = 1.67% > 1%
    prices = {"AAPL": 20.0}
    d = _run([Order("AAPL", "buy", 3)], {}, 100_000.0, prices)
    assert d.orders == []
    assert "commission" in _reasons(d)["AAPL"]


def test_buys_blocked_after_3pct_daily_loss_sells_allowed():
    prices = {"AAPL": 100.0, "MSFT": 100.0}
    positions = {"AAPL": 100}
    cash = 86_000.0  # equity 96k, prior close 100k -> down 4%
    orders = [Order("MSFT", "buy", 50), Order("AAPL", "sell", 50)]
    d = _run(orders, positions, cash, prices, prior=100_000.0, peak=100_000.0)
    assert not d.halted
    assert [(o.ticker, o.side, o.shares) for o in d.orders] == [("AAPL", "sell", 50)]
    assert _reasons(d)["MSFT"] == "daily loss limit"


def test_daily_loss_under_threshold_allows_buys():
    prices = {"MSFT": 100.0}
    d = _run([Order("MSFT", "buy", 50)], {}, 98_000.0, prices, prior=100_000.0, peak=100_000.0)
    assert [o.ticker for o in d.orders] == ["MSFT"]


def test_drawdown_15pct_flattens_and_halts():
    prices = {"AAPL": 100.0, "MSFT": 50.0}
    positions = {"AAPL": 100, "MSFT": 200, "ZERO": 0}
    cash = 65_000.0  # equity 85k, peak 100k -> exactly 15% down
    orders = [Order("NVDA", "buy", 10), Order("AAPL", "sell", 10)]
    d = _run(orders, positions, cash, prices | {"NVDA": 10.0}, prior=85_000.0, peak=100_000.0)
    assert d.halted and d.halt_reason
    assert sorted((o.ticker, o.side, o.shares) for o in d.orders) == [
        ("AAPL", "sell", 100), ("MSFT", "sell", 200)]
    assert all(r == "halted" for _, r in d.rejected) and len(d.rejected) == 2
    pos, cash_after = _apply(d, positions, cash, prices)
    assert pos == {} and cash_after > 0


def test_drawdown_below_threshold_does_not_halt():
    prices = {"AAPL": 100.0}
    d = _run([], {"AAPL": 100}, 76_000.0, prices, prior=86_000.0, peak=100_000.0)
    assert not d.halted and d.halt_reason is None and d.orders == []


def test_short_sell_rejected_when_long_only():
    prices = {"AAPL": 100.0}
    d = _run([Order("AAPL", "sell", 10)], {}, 100_000.0, prices)
    assert d.orders == []
    assert "short" in _reasons(d)["AAPL"]


def test_sell_more_than_held_is_clipped_to_holding():
    prices = {"AAPL": 100.0}
    d = _run([Order("AAPL", "sell", 50)], {"AAPL": 20}, 100_000.0, prices)
    assert [(o.ticker, o.side, o.shares) for o in d.orders] == [("AAPL", "sell", 20)]


def test_sells_processed_before_buys_and_fund_them():
    prices = {"AAPL": 100.0, "MSFT": 100.0}
    # no cash: the buy is only fundable from the sell's proceeds
    d = _run([Order("MSFT", "buy", 50), Order("AAPL", "sell", 100)], {"AAPL": 100}, 0.0, prices)
    assert [o.side for o in d.orders] == ["sell", "buy"]
    pos, cash = _apply(d, {"AAPL": 100}, 0.0, prices)
    assert cash >= 0 and pos["MSFT"] > 0


def test_missing_price_fails_closed_no_orders():
    prices = {"AAPL": 100.0}
    d = _run([Order("MSFT", "buy", 10), Order("AAPL", "buy", 10)], {}, 100_000.0, prices)
    assert [o.ticker for o in d.orders] == ["AAPL"]
    assert "price" in _reasons(d)["MSFT"]
    for bad in (float("nan"), 0.0, -5.0, float("inf")):
        d = _run([Order("MSFT", "buy", 10)], {}, 100_000.0, {"MSFT": bad})
        assert d.orders == [] and "price" in _reasons(d)["MSFT"]


def test_missing_price_of_held_position_fails_closed_everything():
    # equity cannot be computed -> no orders at all, not even for priced tickers
    prices = {"AAPL": 100.0}
    positions = {"AAPL": 10, "MSFT": 10}
    d = _layer().apply([Order("AAPL", "buy", 10), Order("AAPL", "sell", 1)], positions, 100_000.0,
                       prices, prior_close_equity=100_000.0, peak_equity=100_000.0)
    assert d.orders == [] and len(d.rejected) == 2
    assert all("equity" in r for _, r in d.rejected)


def test_invalid_equity_inputs_fail_closed():
    prices = {"AAPL": 100.0}
    for cash, prior, peak in [(float("nan"), 1e5, 1e5), (1e5, float("nan"), 1e5),
                              (1e5, 1e5, float("nan")), (1e5, 0.0, 1e5), (1e5, 1e5, -1.0)]:
        d = _layer().apply([Order("AAPL", "buy", 10)], {}, cash, prices,
                           prior_close_equity=prior, peak_equity=peak)
        assert d.orders == [] and len(d.rejected) == 1


def test_invalid_limits_fail_closed():
    bad = LIMITS.model_copy(update={"max_position_pct": float("nan")})
    d = _run([Order("AAPL", "buy", 10)], {}, 100_000.0, {"AAPL": 100.0}, limits=bad)
    assert d.orders == [] and "limits" in d.rejected[0][1]


def test_zero_negative_and_non_integer_shares_rejected():
    prices = {"AAPL": 100.0, "MSFT": 100.0, "NVDA": 100.0, "AMZN": 100.0}
    orders = [Order("AAPL", "buy", 0), Order("MSFT", "buy", -5), Order("NVDA", "buy", 2.5),
              Order("AMZN", "hold", 5)]
    d = _run(orders, {}, 100_000.0, prices)
    assert d.orders == [] and len(d.rejected) == 4


def test_duplicate_orders_for_one_ticker_rejected():
    prices = {"AAPL": 100.0, "MSFT": 100.0}
    orders = [Order("AAPL", "buy", 10), Order("AAPL", "buy", 10), Order("AAPL", "sell", 5),
              Order("MSFT", "buy", 10)]
    d = _run(orders, {"AAPL": 50}, 100_000.0, prices)
    assert [o.ticker for o in d.orders] == ["MSFT"]
    assert sum(1 for o, r in d.rejected if o.ticker == "AAPL" and "duplicate" in r) == 3


def test_layer_is_pure_and_deterministic():
    prices = {"AAPL": 100.0, "MSFT": 50.0}
    positions = {"AAPL": 10}
    orders = [Order("MSFT", "buy", 800), Order("AAPL", "sell", 5)]
    snap = (list(orders), dict(positions), dict(prices))
    a = _run(orders, positions, 50_000.0, prices)
    b = _run(orders, positions, 50_000.0, prices)
    assert a == b
    assert (orders, positions, prices) == snap


# ---------------------------------------------------------------- property test

TICKERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]
_good_price = st.floats(min_value=0.5, max_value=2_000.0, allow_nan=False)
_order = st.builds(Order, st.sampled_from(TICKERS), st.sampled_from(["buy", "sell"]),
                   st.one_of(st.integers(1, 5_000), st.integers(-3, 0)))


@st.composite
def _scenario(draw):
    prices = {t: draw(_good_price) for t in TICKERS}
    for t in draw(st.lists(st.sampled_from(TICKERS), unique=True, max_size=2)):  # bad/missing
        bad = draw(st.sampled_from([None, float("nan"), 0.0, -1.0, float("inf")]))
        if bad is None:
            del prices[t]
        else:
            prices[t] = bad
    held = draw(st.dictionaries(st.sampled_from(TICKERS), st.integers(0, 3_000), max_size=9))
    cash = draw(st.floats(min_value=0.0, max_value=2_000_000.0, allow_nan=False))
    orders = draw(st.lists(_order, unique_by=lambda o: o.ticker, max_size=len(TICKERS)))
    if orders and draw(st.integers(0, 9)) == 0:  # occasionally a duplicate ticker
        orders.append(Order(orders[0].ticker, draw(st.sampled_from(["buy", "sell"])), 10))
    limits = draw(st.sampled_from([
        LIMITS,
        LIMITS.model_copy(update={"max_position_pct": 1.0}),
        LIMITS.model_copy(update={"max_position_pct": 0.4, "max_gross_pct": 0.6,
                                  "max_positions": 3}),
    ]))
    prior_mult = draw(st.sampled_from([1.0, 1.0, 1.0, 1.05]))  # 1.05 trips the daily-loss rule
    peak_mult = draw(st.sampled_from([1.0, 1.0, 1.1, 1.3]))  # 1.3 trips the kill switch
    return prices, held, cash, orders, limits, prior_mult, peak_mult


@settings(max_examples=1000, deadline=None)
@given(_scenario())
def test_property_exposure_and_cash_invariants(scenario):
    prices, held, cash, orders, limits, prior_mult, peak_mult = scenario
    positions = {t: s for t, s in held.items()}
    held_live = {t: s for t, s in positions.items() if s > 0}
    priced = all(t in prices and math.isfinite(prices[t]) and prices[t] > 0 for t in held_live)
    eq0 = _equity(held_live, cash, prices) if priced else 1.0
    d = _layer(limits).apply(orders, positions, cash, prices,
                             prior_close_equity=max(eq0, 1.0) * prior_mult,
                             peak_equity=max(eq0, 1.0) * peak_mult)

    # every incoming order is either emitted (possibly clipped) or rejected, never both lost
    assert len(d.rejected) <= len(orders)
    if not priced:
        assert d.orders == [] and len(d.rejected) == len(orders)
        return

    for o in d.orders:
        assert isinstance(o.shares, int) and o.shares > 0
        assert math.isfinite(prices[o.ticker]) and prices[o.ticker] > 0
    sides = [o.side for o in d.orders]
    assert sides == sorted(sides, key=lambda s: s != "sell")  # sells strictly first

    pos, cash_after = _apply(d, held_live, cash, prices)
    assert all(s > 0 for s in pos.values())  # long only, never short
    if d.halted:
        # flatten is unconditional; commission on a dust position may exceed its proceeds
        assert pos == {} and all(o.side == "sell" for o in d.orders)
        return
    assert cash_after >= 0

    eq = _equity(pos, cash_after, prices)
    gross = sum(s * prices[t] for t, s in pos.items())
    bought = {o.ticker for o in d.orders if o.side == "buy"}
    if bought:
        assert gross <= limits.max_gross_pct * eq
    for t, s in pos.items():
        if s > held_live.get(t, 0):
            assert s * prices[t] <= limits.max_position_pct * eq
    new = set(pos) - set(held_live)
    if new:
        assert len(pos) <= limits.max_positions
