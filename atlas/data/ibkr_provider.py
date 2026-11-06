"""Interactive Brokers historical daily bars (optional extra: ib_async).

Read-only market data; this module contains no order-placing code.
"""
from __future__ import annotations

import logging
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
        if self._ib is None:
            try:
                from ib_async import IB
            except ImportError as e:
                raise ImportError("IBKRProvider needs the optional extra: pip install atlas[ibkr]") from e
            ib = IB()
            ib.connect(self._host, self._port, clientId=self._client_id)
            self._ib = ib
        if self._contract_factory is None:
            from ib_async import Stock
            self._contract_factory = lambda t: Stock(t, "SMART", "USD")
        return self._ib

    def _bars(self, ib, contract, what: str, start: date, end: date) -> pd.DataFrame:
        days = (end - start).days + 1
        bars = ib.reqHistoricalData(
            contract, endDateTime=datetime.combine(end + timedelta(days=1), datetime.min.time()),
            durationStr=f"{days} D", barSizeSetting="1 day", whatToShow=what,
            useRTH=True, formatDate=1,
        )
        rows = [(pd.Timestamp(b.date), b.open, b.high, b.low, b.close, b.volume) for b in bars]
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])
        return df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))]

    def fetch(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        ib = self._connect()
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
