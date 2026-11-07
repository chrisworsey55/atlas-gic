"""Interactive Brokers historical daily bars (optional extra: ib_async).

UNVERIFIED LIVE: this provider is tested only against mocks, never against a real IB Gateway
/ TWS. In particular IB may reject ``whatToShow="ADJUSTED_LAST"`` combined with an explicit
``endDateTime``, and long ranges are requested in whole years (IB refuses day durations over
365 days), which fetches more history than asked (rows are filtered to the range). Prefer the
default yfinance or csv source unless you have checked it against your gateway.

Read-only market data: it connects with ``readonly=True``, disconnects after every fetch
(also on error) and contains no order-placing code.
"""
from __future__ import annotations

import logging
import math
from datetime import date, datetime, timedelta

import pandas as pd

from atlas.data.provider import PRICE_COLUMNS

log = logging.getLogger(__name__)


class IBKRProvider:
    def __init__(self, host: str, port: int, client_id: int, ib=None, contract_factory=None):
        self._host, self._port, self._client_id = host, port, client_id
        self._ib = ib
        self._contract_factory = contract_factory

    def _connect(self):
        """(ib, owned): an injected client is used as is; otherwise a new read-only connection
        that the caller must disconnect."""
        owned = self._ib is None
        if owned:
            try:
                from ib_async import IB
            except ImportError as e:
                raise ImportError("IBKRProvider needs the optional extra: pip install atlas[ibkr]") from e
            ib = IB()
            ib.connect(self._host, self._port, clientId=self._client_id, readonly=True)
        else:
            ib = self._ib
        if self._contract_factory is None:
            from ib_async import Stock
            self._contract_factory = lambda t: Stock(t, "SMART", "USD")
        return ib, owned

    def _bars(self, ib, contract, what: str, start: date, end: date) -> pd.DataFrame:
        days = (end - start).days + 1
        duration = f"{days} D" if days <= 365 else f"{math.ceil(days / 365)} Y"
        bars = ib.reqHistoricalData(
            contract, endDateTime=datetime.combine(end + timedelta(days=1), datetime.min.time()),
            durationStr=duration, barSizeSetting="1 day", whatToShow=what,
            useRTH=True, formatDate=1,
        )
        rows = [(pd.Timestamp(b.date), b.open, b.high, b.low, b.close, b.volume) for b in bars]
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        ib, owned = self._connect()
        try:
            return self._fetch(ib, tickers, start, end)
        finally:
            if owned:
                ib.disconnect()

    def _fetch(self, ib, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        frames = []
        for t in tickers:
            contract = self._contract_factory(t)
            trades = self._bars(ib, contract, "TRADES", start, end)
            if trades.empty:
                log.warning("IBKR returned no bars for ticker %s (%s..%s)", t, start, end)
                continue
            adj = self._bars(ib, contract, "ADJUSTED_LAST", start, end)[["date", "close"]]
            df = trades.merge(adj.rename(columns={"close": "adj_close"}), on="date", how="left")
            df["ticker"] = t
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=PRICE_COLUMNS)
        return pd.concat(frames, ignore_index=True)[PRICE_COLUMNS]
