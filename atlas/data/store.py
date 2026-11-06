"""PriceStore: disk-cached prices with as-of snapshots (the no-lookahead core)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import pandas as pd

from atlas.data.provider import PRICE_COLUMNS, MacroProvider, PriceProvider

FORWARD_DAYS = 14  # calendar-day window fetched for next_open


@dataclass(frozen=True)
class Snapshot:
    asof: date
    prices: pd.DataFrame
    macro: dict[str, pd.Series] = field(default_factory=dict)


def _d(x) -> date:
    return pd.Timestamp(x).date()


class PriceStore:
    def __init__(self, prices: PriceProvider, macro: MacroProvider, cache_dir: Path,
                 calendar_ticker: str = "SPY"):
        self._prices = prices
        self._macro = macro
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._cal = calendar_ticker

    # ---- cache -------------------------------------------------------
    def _csv(self, t: str) -> Path:
        return self._dir / f"{t}.csv"

    def _meta(self, t: str) -> Path:
        return self._dir / f"{t}.meta.json"

    def _coverage(self, t: str) -> tuple[date, date] | None:
        p = self._meta(t)
        if not p.exists() or not self._csv(t).exists():
            return None
        m = json.loads(p.read_text())
        return date.fromisoformat(m["start"]), date.fromisoformat(m["end"])

    def _load(self, t: str) -> pd.DataFrame:
        if not self._csv(t).exists():
            return pd.DataFrame(columns=PRICE_COLUMNS)
        df = pd.read_csv(self._csv(t))
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df

    def _ensure(self, t: str, start: date, end: date) -> None:
        """Make sure the cache covers [start, end] for ticker t (fetch only if not)."""
        cov = self._coverage(t)
        if cov and cov[0] <= start and end <= cov[1]:
            return
        lo, hi = (min(start, cov[0]), max(end, cov[1])) if cov else (start, end)
        new = self._prices.fetch([t], lo, hi)
        new = new[new["ticker"] == t][PRICE_COLUMNS].copy() if len(new) else pd.DataFrame(columns=PRICE_COLUMNS)
        if new.empty:
            return  # empty/failed response: record no coverage, refetch next time
        new["date"] = pd.to_datetime(new["date"]).dt.date
        # coverage is what the provider actually returned, never the requested
        # range and never beyond today
        hi = min(hi, max(new["date"]), date.today())
        if cov:
            lo, hi = min(lo, cov[0]), max(hi, cov[1])
        old = self._load(t)
        merged = pd.concat([old, new], ignore_index=True)
        merged = merged.drop_duplicates("date", keep="last").sort_values("date")
        merged.to_csv(self._csv(t), index=False)
        self._meta(t).write_text(json.dumps({"start": lo.isoformat(), "end": hi.isoformat()}))

    # ---- as-of access ------------------------------------------------
    def snapshot(self, asof: date, tickers: list[str], lookback_days: int,
                 macro_series: Sequence[str] = ()) -> Snapshot:
        start = asof - timedelta(days=lookback_days)
        frames = []
        for t in tickers:
            self._ensure(t, start, asof)  # never request data after asof
            df = self._load(t)
            frames.append(df[(df["date"] >= start) & (df["date"] <= asof)])
        prices = (pd.concat(frames, ignore_index=True) if frames
                  else pd.DataFrame(columns=PRICE_COLUMNS))
        prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
        macro = {}
        for sid in macro_series:
            s = self._macro.fetch(sid, asof)
            macro[sid] = s[[_d(i) <= asof for i in s.index]]  # belt and braces
        return Snapshot(asof=asof, prices=prices, macro=macro)

    def trading_days(self, start: date, end: date) -> list[date]:
        self._ensure(self._cal, start, end)
        df = self._load(self._cal)
        return [d for d in df["date"] if start <= d <= end]

    # ---- explicit forward access (scoring and brokers only) ----------
    def next_open(self, ticker: str, after: date) -> tuple[date, float] | None:
        """First bar strictly after `after`; None if no later bar exists."""
        self._ensure(ticker, after, after + timedelta(days=FORWARD_DAYS))
        df = self._load(ticker)
        later = df[df["date"] > after]
        if later.empty:
            return None
        row = later.iloc[0]
        return row["date"], float(row["open"])

    def outcome_prices(self, ticker: str, start: date, end: date) -> pd.Series:
        """Adjusted closes over [start, end], indexed by date. Forward access."""
        self._ensure(ticker, start, end)
        df = self._load(ticker)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return pd.Series(df["adj_close"].to_numpy(dtype=float), index=df["date"].to_numpy(),
                         name=ticker)
