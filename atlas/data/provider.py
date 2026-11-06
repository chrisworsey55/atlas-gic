"""Data provider protocols. Concrete providers live elsewhere (task 4)."""
from __future__ import annotations

from datetime import date
from typing import Protocol

import pandas as pd

PRICE_COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]


class PriceProvider(Protocol):
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Long-format daily bars with columns PRICE_COLUMNS, start..end inclusive."""
        ...


class MacroProvider(Protocol):
    def fetch(self, series_id: str, asof: date) -> pd.Series:
        """Values as known on `asof` (real-time vintage, publication lags applied),
        indexed by observation date."""
        ...
