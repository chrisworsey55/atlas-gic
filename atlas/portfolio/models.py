"""Portfolio domain models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Order:
    ticker: str
    side: Literal["buy", "sell"]
    shares: int
