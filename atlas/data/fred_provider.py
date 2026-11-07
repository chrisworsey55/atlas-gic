"""FRED/ALFRED macro provider returning real-time vintages (no look-ahead)."""
from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd
import requests

log = logging.getLogger(__name__)

URL = "https://api.stlouisfed.org/fred/series/observations"
_EARLIEST = "1776-07-04"  # ALFRED's earliest realtime_start


class FredFetchError(RuntimeError):
    """FRED request failed; message carries series id and status only (never the URL/key)."""


class FredAlfredProvider:
    def __init__(self, api_key: str, publication_lag_days: dict[str, int] | None = None):
        self._api_key = api_key
        self._lags = dict(publication_lag_days or {})

    def fetch(self, series_id: str, asof: date) -> pd.Series:
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "realtime_start": _EARLIEST,
            "realtime_end": asof.isoformat(),
        }
        # The key is in the query string, so never let requests' exceptions
        # (which embed the URL) escape: raise a sanitized error outside the except block.
        failure = None
        try:
            resp = requests.get(URL, params=params, timeout=30)
            resp.raise_for_status()
            rows = resp.json().get("observations", [])
        except requests.RequestException as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            failure = f"status {status}" if status is not None else type(e).__name__
        if failure is not None:
            raise FredFetchError(f"FRED request failed for series {series_id}: {failure}") from None
        if not rows:
            return pd.Series(dtype=float, name=series_id)
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df["realtime_start"] = pd.to_datetime(df["realtime_start"])
        df = df[df["realtime_start"] <= pd.Timestamp(asof)]
        # latest value published on or before asof, per observation date
        df = df.sort_values(["date", "realtime_start"]).drop_duplicates("date", keep="last")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")  # "." -> NaN
        df = df.dropna(subset=["value"])
        lag = self._lags.get(series_id, 0)
        df = df[df["date"] <= pd.Timestamp(asof - timedelta(days=lag))]
        s = pd.Series(df["value"].to_numpy(dtype=float), index=pd.DatetimeIndex(df["date"]), name=series_id)
        return s.sort_index()
