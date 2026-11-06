"""Simulated broker: next-open fills with slippage and commission."""
from __future__ import annotations

import math
from datetime import date
from typing import Protocol

from atlas.data.store import PriceStore
from atlas.portfolio.models import Fill, Order, Portfolio


def size_order(target_value: float, price: float) -> int:
    """Whole shares (floored) worth at most target_value; 0 for non-positive/non-finite inputs."""
    if not (math.isfinite(target_value) and math.isfinite(price)):
        return 0
    if target_value <= 0 or price <= 0:
        return 0
    return int(math.floor(target_value / price))


class Broker(Protocol):
    def submit(self, orders: list[Order], decision_date: date
               ) -> tuple[list[Fill], list[tuple[Order, str]]]: ...


class SimBroker:
    def __init__(self, store: PriceStore, commission_per_share: float,
                 commission_min: float, slippage_bps: float):
        self._store = store
        self._cps = commission_per_share
        self._cmin = commission_min
        self._slip = slippage_bps / 1e4

    def _commission(self, shares: int) -> float:
        return max(self._cmin, self._cps * shares)

    def _affordable(self, cash: float, px: float) -> int:
        """Largest whole share count n with n*px + commission(n) <= cash."""
        if cash <= 0:
            return 0
        n = int(math.floor(min((cash - self._cmin) / px, cash / (px + self._cps))))
        while n > 0 and n * px + self._commission(n) > cash:
            n -= 1
        return max(n, 0)

    def submit(self, orders: list[Order], decision_date: date,
               portfolio: Portfolio | None = None
               ) -> tuple[list[Fill], list[tuple[Order, str]]]:
        """Fill orders at each ticker's next open after `decision_date`.

        Sells are processed before buys. Orders with no later bar, or whose next open is
        NaN/non-positive, are dropped, never filled at a stale or missing price. The portfolio is never mutated; the caller applies
        the returned fills. If `portfolio` is omitted the broker is stateless about
        cash and holdings, so NO cash or holdings checks are possible (buys are not
        reduced for affordability and oversized sells are not rejected); the backtest
        runner always passes it.
        """
        fills: list[Fill] = []
        dropped: list[tuple[Order, str]] = []
        cash = portfolio.cash if portfolio is not None else None
        held = dict(portfolio.positions) if portfolio is not None else None

        clean: list[Order] = []
        for o in orders:
            shares = int(o.shares)
            if shares <= 0:
                dropped.append((o, "non-positive share quantity"))
                continue
            clean.append(Order(o.ticker, o.side, shares))
        ordered = ([o for o in clean if o.side == "sell"]
                   + [o for o in clean if o.side == "buy"])

        for o in ordered:
            nxt = self._store.next_open(o.ticker, decision_date)
            if nxt is None:
                dropped.append((o, f"no next open for {o.ticker} after {decision_date}"))
                continue
            fdate, open_px = nxt
            if not (isinstance(open_px, (int, float)) and math.isfinite(open_px) and open_px > 0):
                dropped.append((o, f"unusable next open {open_px!r} for {o.ticker} on {fdate}"))
                continue
            if o.side == "sell":
                if held is not None and o.shares > held.get(o.ticker, 0):
                    dropped.append((o, f"sell {o.shares} exceeds holdings "
                                       f"{held.get(o.ticker, 0)} of {o.ticker}"))
                    continue
                px = open_px * (1 - self._slip)
                comm = self._commission(o.shares)
                if held is not None:
                    held[o.ticker] -= o.shares
                    cash += o.shares * px - comm
                fills.append(Fill(o, fdate, px, comm))
            else:
                px = open_px * (1 + self._slip)
                shares = o.shares
                if cash is not None and shares * px + self._commission(shares) > cash:
                    shares = min(shares, self._affordable(cash, px))
                    if shares <= 0:
                        dropped.append((o, f"insufficient cash to buy 1 share of {o.ticker}"))
                        continue
                comm = self._commission(shares)
                if cash is not None:
                    cash -= shares * px + comm
                fills.append(Fill(Order(o.ticker, "buy", shares), fdate, px, comm))
        return fills, dropped
