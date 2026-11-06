"""yfinance-backed daily price provider."""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from atlas.data.provider import PRICE_COLUMNS

log = logging.getLogger(__name__)

_FIELDS = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
           "Adj Close": "adj_close", "Volume": "volume"}


class YFinanceProvider:
    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        raw = yf.download(
            list(tickers), start=start, end=end + timedelta(days=1),  # end exclusive
            auto_adjust=False, group_by="ticker", progress=False, threads=True,
        )
        frames = []
        for t in tickers:
            sub = self._slice(raw, t, len(tickers))
            if sub is None or sub.dropna(how="all").empty:
                log.warning("yfinance returned no data for ticker %s (%s..%s)", t, start, end)
                continue
            sub = sub.dropna(how="all").rename(columns=_FIELDS)[list(_FIELDS.values())].copy()
            sub.index = pd.to_datetime(sub.index).tz_localize(None).normalize()
            sub.index.name = "date"
            sub = sub.reset_index()
            sub["ticker"] = t
            frames.append(sub)
        if not frames:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        out = pd.concat(frames, ignore_index=True)[PRICE_COLUMNS]
        return out[(out["date"] >= pd.Timestamp(start)) & (out["date"] <= pd.Timestamp(end))].reset_index(drop=True)

    @staticmethod
    def _slice(raw: pd.DataFrame, ticker: str, n: int):
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            if ticker in raw.columns.get_level_values(0):
                return raw[ticker]
            return None
        return raw if n == 1 else None
