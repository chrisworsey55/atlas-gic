"""PriceStore: disk-cached prices with as-of snapshots (the no-lookahead core)."""
from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Sequence

import numpy as np
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
        # parsed CSVs keyed by ticker, valid while the file's (mtime_ns, size) is unchanged
        # NOT thread-safe: the memo assumes single-threaded access to a PriceStore
        self._frames: dict[str, tuple[tuple[int, int], pd.DataFrame]] = {}
        self._adj: dict[str, tuple[tuple[int, int], list, np.ndarray]] = {}  # sorted adj closes

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
        """The cached bars for t (a private copy)."""
        df = self._parsed(t)
        return pd.DataFrame(columns=PRICE_COLUMNS) if df is None else df.copy()

    def _parsed(self, t: str) -> pd.DataFrame | None:
        """The memoised parsed CSV (shared: never mutate), refreshed when the file's
        (mtime_ns, size) changes; None if there is no cache file. Scoring reads the same
        files thousands of times per run."""
        path = self._csv(t)
        try:
            st = path.stat()
        except FileNotFoundError:
            self._frames.pop(t, None)
            self._adj.pop(t, None)
            return None
        key = (st.st_mtime_ns, st.st_size)
        hit = self._frames.get(t)
        if hit is None or hit[0] != key:
            df = pd.read_csv(path)
            df["date"] = pd.to_datetime(df["date"]).dt.date
            hit = (key, df)
            self._frames[t] = hit
            df = df.sort_values("date", kind="stable")
            self._adj[t] = (key, list(df["date"]), df["adj_close"].to_numpy(dtype=float))
        return hit[1]

    def _ensure(self, t: str, start: date, end: date) -> None:
        """Make sure the cache covers [start, end] for ticker t (fetch only if not)."""
        cov = self._coverage(t)
        if cov and cov[0] <= start and end <= cov[1]:
            return
        lo, hi = (min(start, cov[0]), max(end, cov[1])) if cov else (start, end)
        self._merge(t, self._prices.fetch([t], lo, hi), lo, hi, cov)

    def _merge(self, t: str, new: pd.DataFrame, lo: date, hi: date, cov: tuple[date, date] | None) -> None:
        """Merge provider rows for t (requested range lo..hi) into its cache file."""
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
        self._frames.pop(t, None)
        self._adj.pop(t, None)  # never serve the pre-rewrite frame (mtime granularity)
        self._meta(t).write_text(json.dumps({"start": lo.isoformat(), "end": hi.isoformat()}))

    def prefetch(self, tickers: Sequence[str], start: date, end: date) -> list[str]:
        """Fill the cache for [start, end] for every ticker in ONE provider request (only the
        tickers whose cache does not already cover the range), so a run does not re-download a
        growing range every day. Returns the tickers that have NO bar at all in [start, end].

        Safe for the no-lookahead rule: snapshot() only ever returns rows dated <= asof, and
        forward data is already cached by scoring anyway."""
        tickers = list(dict.fromkeys(tickers))
        need = [t for t in tickers
                if not ((cov := self._coverage(t)) and cov[0] <= start and end <= cov[1])]
        if need:
            covs = {t: self._coverage(t) for t in need}
            lo = min([start] + [c[0] for c in covs.values() if c])
            hi = max([end] + [c[1] for c in covs.values() if c])
            new = self._prices.fetch(need, lo, hi)
            for t in need:
                self._merge(t, new, lo, hi, covs[t])
        missing = []
        for t in tickers:
            df = self._parsed(t)
            if df is None or not ((df["date"] >= start) & (df["date"] <= end)).any():
                missing.append(t)
        return missing

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
        df = self._parsed(self._cal)
        return [] if df is None else [d for d in df["date"] if start <= d <= end]

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
        if self._parsed(ticker) is None:  # also refreshes the sorted adj-close view
            return pd.Series(np.array([], dtype=float), index=np.array([], dtype=object), name=ticker)
        _, dates, adj = self._adj[ticker]
        lo, hi = bisect.bisect_left(dates, start), bisect.bisect_right(dates, end)
        return pd.Series(adj[lo:hi].copy(), index=np.array(dates[lo:hi], dtype=object), name=ticker)
