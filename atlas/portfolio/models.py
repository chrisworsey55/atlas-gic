"""Portfolio domain models."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal


@dataclass(frozen=True)
class Order:
    ticker: str
    side: Literal["buy", "sell"]
    shares: int


@dataclass(frozen=True)
class Fill:
    order: Order
    date: date
    price: float
    commission: float


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, int] = field(default_factory=dict)

    def equity(self, prices: dict[str, float]) -> float:
        """Cash plus positions marked at `prices` (KeyError if a held ticker has no price)."""
        return self.cash + sum(n * prices[t] for t, n in self.positions.items())

    def apply(self, fill: Fill) -> None:
        o = fill.order
        value = o.shares * fill.price
        if o.side == "buy":
            self.cash -= value + fill.commission
            self.positions[o.ticker] = self.positions.get(o.ticker, 0) + o.shares
        else:
            self.cash += value - fill.commission
            self.positions[o.ticker] = self.positions.get(o.ticker, 0) - o.shares
        if self.positions[o.ticker] == 0:
            del self.positions[o.ticker]
