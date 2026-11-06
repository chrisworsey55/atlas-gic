"""Deterministic, fail-closed pre-trade risk layer.

Every proposed order (ultimately derived from untrusted LLM output) passes through
`RiskLayer.apply` before it can reach a broker or the backtest fill model. The layer is a
pure function of its arguments: no I/O, no clock, no randomness, no mutation of inputs and
no state carried between calls, so agents cannot influence or override it and it is
trivially testable.

Policy (all thresholds come from `Limits`):

* Fail closed. If the account equity cannot be computed (non-finite cash, a held position
  with a missing / NaN / non-positive price, a malformed position) or the limits /
  reference equities cannot be evaluated, NO orders are emitted at all and every incoming
  order is rejected with a reason. An order whose own price is unusable is rejected; other
  orders are unaffected.
* Duplicate orders for one ticker are REJECTED (every order for that ticker), not netted.
  Two instructions for the same name mean the upstream intent is ambiguous; doing nothing
  for that ticker is the conservative choice and cannot create exposure nobody asked for.
* Kill switch: equity <= peak_equity * (1 - drawdown_kill_pct) => emit sells flattening
  every held position, reject every incoming order with reason "halted" and return
  halted=True. The layer is stateless: the caller must persist the halt and stop trading
  until a manual restart.
* Daily loss: equity < prior_close_equity * (1 - daily_loss_pct) => every buy is rejected
  with reason "daily loss limit"; sells are still processed.
* Sells are processed before buys (their proceeds fund the buys). A sell is clipped to the
  shares held; a sell with no shares held is a short sale and is rejected. Short selling
  is never emitted, even if `long_only` is False: margin/borrow accounting is not
  implemented, so shorts cannot be risk-checked and are rejected (fail closed).
* Buys are processed in the order given and clipped (integer floor of shares) so that,
  after all emitted orders execute at `prices` with commission charged:
    - each increased position value <= max_position_pct * post-trade equity,
    - total position value <= max_gross_pct * post-trade equity,
    - cash >= 0.
  Caps are sized against an equity floor that already deducts the worst-case commission
  of every candidate buy (commission is non-decreasing in shares, so the requested size is
  an upper bound), less a 1e-9 relative safety margin for float rounding. The invariants
  therefore hold exactly however many of the buys end up executing.
* A buy that would open a new position when `max_positions` distinct tickers are already
  held (after sells) is rejected; existing positions may still be increased.
* Any order (after clipping) worth less than `min_order_usd`, or whose commission
  max(commission_min, commission_per_share * shares) exceeds max_commission_frac of its
  value, is rejected. Orders clipped to zero shares are rejected. The kill-switch flatten
  orders are exempt from these filters: flattening must be complete.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from atlas.config import Limits
from atlas.portfolio.models import Order

_SAFETY = 1e-9  # relative margin on the equity floor, absorbs float summation-order noise


@dataclass(frozen=True)
class RiskDecision:
    orders: list[Order]
    rejected: list[tuple[Order, str]]
    halted: bool
    halt_reason: str | None


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _good_price(prices: dict, ticker) -> bool:
    p = prices.get(ticker)
    return _is_num(p) and p > 0


class RiskLayer:
    def __init__(self, limits: Limits, commission_per_share: float, commission_min: float):
        self.limits = limits
        self.commission_per_share = commission_per_share
        self.commission_min = commission_min

    def commission(self, shares: int) -> float:
        return max(self.commission_min, self.commission_per_share * shares)

    def _limits_error(self) -> str | None:
        L = self.limits
        pcts = {"max_position_pct": L.max_position_pct, "max_gross_pct": L.max_gross_pct,
                "min_order_usd": L.min_order_usd, "max_commission_frac": L.max_commission_frac,
                "daily_loss_pct": L.daily_loss_pct, "drawdown_kill_pct": L.drawdown_kill_pct,
                "commission_per_share": self.commission_per_share,
                "commission_min": self.commission_min}
        for name, v in pcts.items():
            if not _is_num(v) or v < 0:
                return f"risk limits cannot be evaluated: {name}={v!r}"
        if L.max_position_pct <= 0 or L.max_gross_pct <= 0:
            return "risk limits cannot be evaluated: non-positive exposure cap"
        if L.daily_loss_pct >= 1 or L.drawdown_kill_pct >= 1:
            return "risk limits cannot be evaluated: loss threshold >= 100%"
        if not _is_int(L.max_positions) or L.max_positions < 0:
            return f"risk limits cannot be evaluated: max_positions={L.max_positions!r}"
        return None

    def apply(self, orders: list[Order], positions: dict[str, int], cash: float,
              prices: dict[str, float], prior_close_equity: float,
              peak_equity: float) -> RiskDecision:
        incoming = list(orders)
        L = self.limits

        def reject_all(reason: str) -> RiskDecision:
            return RiskDecision([], [(o, reason) for o in incoming], False, None)

        # ---- can the limits and the account equity be evaluated at all?
        err = self._limits_error()
        if err:
            return reject_all(err)
        if not isinstance(positions, dict) or not isinstance(prices, dict):
            return reject_all("positions/prices malformed: equity cannot be computed")
        for t, s in positions.items():
            if not isinstance(t, str) or not _is_int(s) or s < 0:
                return reject_all(f"invalid position {t!r}={s!r}: equity cannot be computed")
        held = {t: s for t, s in positions.items() if s > 0}
        if not _is_num(cash):
            return reject_all(f"invalid cash {cash!r}: equity cannot be computed")
        for t in sorted(held):
            if not _good_price(prices, t):
                return reject_all(f"missing or invalid price for held {t}: "
                                  "equity cannot be computed")
        equity = cash + sum(held[t] * prices[t] for t in sorted(held))
        if not math.isfinite(equity):
            return reject_all("equity cannot be computed (non-finite)")
        for name, v in (("prior_close_equity", prior_close_equity), ("peak_equity", peak_equity)):
            if not _is_num(v) or v <= 0:
                return reject_all(f"invalid {name} {v!r}: limits cannot be evaluated")

        # ---- kill switch: flatten everything and halt
        if equity <= peak_equity * (1 - L.drawdown_kill_pct):
            reason = (f"drawdown kill switch: equity {equity:.2f} <= "
                      f"{1 - L.drawdown_kill_pct:.0%} of peak {peak_equity:.2f}; "
                      "flattened, manual restart required")
            flatten = [Order(t, "sell", held[t]) for t in sorted(held)]
            return RiskDecision(flatten, [(o, "halted") for o in incoming], True, reason)
        if equity <= 0:
            return reject_all("non-positive equity: limits cannot be evaluated")
        daily_loss = equity < prior_close_equity * (1 - L.daily_loss_pct)

        rejected: list[tuple[Order, str]] = []

        # ---- per-order validation; duplicates for a ticker are all rejected
        counts: dict[str, int] = {}
        for o in incoming:
            if isinstance(o, Order) and isinstance(o.ticker, str):
                counts[o.ticker] = counts.get(o.ticker, 0) + 1
        sells: list[Order] = []
        buys: list[Order] = []
        for o in incoming:
            if not isinstance(o, Order) or not isinstance(o.ticker, str) or not o.ticker:
                rejected.append((o, "invalid order/ticker"))
            elif counts[o.ticker] > 1:
                rejected.append((o, "duplicate orders for ticker: rejected (ambiguous intent)"))
            elif o.side not in ("buy", "sell"):
                rejected.append((o, f"invalid side {o.side!r}"))
            elif not _is_int(o.shares) or o.shares <= 0:
                rejected.append((o, f"invalid shares {o.shares!r}: must be a positive integer"))
            elif not _good_price(prices, o.ticker):
                rejected.append((o, "missing or invalid price: fail closed"))
            else:
                (sells if o.side == "sell" else buys).append(o)

        out: list[Order] = []

        def size_filter(n: int, p: float) -> str | None:
            value = n * p
            if value < L.min_order_usd:
                return f"order value {value:.2f} below minimum {L.min_order_usd}"
            if self.commission(n) > L.max_commission_frac * value:
                return f"commission exceeds {L.max_commission_frac:.2%} of order value"
            return None

        # ---- sells first
        for o in sells:
            p = prices[o.ticker]
            h = held.get(o.ticker, 0)
            if h <= 0:
                rejected.append((o, "short sell rejected: no shares held (long-only)"))
                continue
            n = min(o.shares, h)
            why = size_filter(n, p)
            if why is None and self.commission(n) > n * p:
                why = "commission exceeds sale proceeds"
            if why:
                rejected.append((o, why))
                continue
            out.append(o if n == o.shares else Order(o.ticker, "sell", n))
            cash = cash + (n * p - self.commission(n))
            held[o.ticker] = h - n
            if held[o.ticker] == 0:
                del held[o.ticker]

        # ---- buys, in the order given
        if daily_loss:
            rejected.extend((o, "daily loss limit") for o in buys)
            buys = []
        gross = sum(held[t] * prices[t] for t in sorted(held))
        reserve = sum(self.commission(o.shares) for o in buys)
        eq_floor = (cash + gross - reserve) * (1 - _SAFETY)
        pos_cap = L.max_position_pct * eq_floor
        gross_cap = L.max_gross_pct * eq_floor
        n_positions = len(held)

        def floor_room(room: float, p: float) -> int:
            return math.floor(room / p) if room > 0 else 0

        for o in buys:
            t, p = o.ticker, prices[o.ticker]
            h = held.get(t, 0)
            if eq_floor <= 0:
                rejected.append((o, "insufficient equity"))
                continue
            if h == 0 and n_positions >= L.max_positions:
                rejected.append((o, f"max positions ({L.max_positions}) reached"))
                continue
            bounds = {"requested": o.shares,
                      "position cap": floor_room(pos_cap - h * p, p),
                      "gross exposure cap": floor_room(gross_cap - gross, p),
                      "available cash": floor_room(cash, p)}
            n = min(bounds.values())

            def fits(k: int) -> bool:
                return ((h + k) * p <= pos_cap and gross + k * p <= gross_cap
                        and cash - (k * p + self.commission(k)) >= 0)

            while n > 0 and not fits(n):
                n -= 1
            if n <= 0:
                binding = min(bounds, key=lambda k: bounds[k])
                rejected.append((o, f"clipped to zero shares by {binding}"))
                continue
            why = size_filter(n, p)
            if why:
                rejected.append((o, why))
                continue
            out.append(o if n == o.shares else Order(t, "buy", n))
            cash = cash - (n * p + self.commission(n))
            gross = gross + n * p
            held[t] = h + n
            if h == 0:
                n_positions += 1

        return RiskDecision(out, rejected, False, None)
