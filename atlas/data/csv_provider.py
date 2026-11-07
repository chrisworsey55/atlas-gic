"""Offline price provider reading one CSV per ticker (``<dir>/<TICKER>.csv``).

Columns: PRICE_COLUMNS (``ticker`` optional; it is taken from the file name). Used for
offline runs and the golden tests; a missing file yields no rows for that ticker.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from atlas.data.provider import PRICE_COLUMNS


class CsvPriceProvider:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        if not self.dir.is_dir():
            raise FileNotFoundError(f"price CSV directory {self.dir} does not exist")

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        frames = []
        for t in tickers:
            p = self.dir / f"{t}.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            df["ticker"] = t
            df["date"] = pd.to_datetime(df["date"]).dt.date
            frames.append(df[(df["date"] >= start) & (df["date"] <= end)][PRICE_COLUMNS])
        if not frames:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        return pd.concat(frames, ignore_index=True)


class CacheOnlyPrices:
    """Fetches nothing: a PriceStore built on it serves only what is already cached (used by
    the offline report/calibrate commands)."""

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        return pd.DataFrame(columns=PRICE_COLUMNS)


class NoMacro:
    """Macro provider with no series (``data.macro.source: none``)."""

    def fetch(self, series_id: str, asof: date) -> pd.Series:
        return pd.Series(dtype=float, name=series_id)
