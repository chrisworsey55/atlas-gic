from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from atlas.data import fred_provider, yf_provider
from atlas.data.fred_provider import FredAlfredProvider
from atlas.data.ibkr_provider import IBKRProvider
from atlas.data.provider import PRICE_COLUMNS
from atlas.data.yf_provider import YFinanceProvider


def _yf_frame(tickers, dates):
    cols = pd.MultiIndex.from_product(
        [tickers, ["Open", "High", "Low", "Close", "Adj Close", "Volume"]])
    return pd.DataFrame(np.arange(1, len(dates) * len(cols) + 1, dtype=float).reshape(len(dates), -1),
                        index=pd.DatetimeIndex(dates), columns=cols)


def test_yfinance_provider_returns_required_columns(monkeypatch):
    calls = {}

    def fake_download(tickers, **kw):
        calls.update(kw)
        return _yf_frame(["AAA", "BBB"], ["2024-01-02", "2024-01-03"])

    monkeypatch.setattr(yf_provider.yf, "download", fake_download)
    df = YFinanceProvider().fetch(["AAA", "BBB"], date(2024, 1, 2), date(2024, 1, 3))
    assert list(df.columns) == PRICE_COLUMNS
    assert set(df["ticker"]) == {"AAA", "BBB"}
    assert len(df) == 4
    assert calls["auto_adjust"] is False
    assert str(calls["end"]) == "2024-01-04"  # end is exclusive in yfinance


def test_yfinance_provider_warns_on_empty_ticker(monkeypatch, caplog):
    frame = _yf_frame(["AAA", "BBB"], ["2024-01-02"])
    frame["BBB"] = np.nan
    monkeypatch.setattr(yf_provider.yf, "download", lambda *a, **k: frame)
    with caplog.at_level("WARNING"):
        df = YFinanceProvider().fetch(["AAA", "BBB"], date(2024, 1, 2), date(2024, 1, 2))
    assert set(df["ticker"]) == {"AAA"}
    assert "BBB" in caplog.text


class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _fred(monkeypatch, observations, lag=None):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen["url"], seen["params"] = url, params
        return _Resp({"observations": observations})

    monkeypatch.setattr(fred_provider.requests, "get", fake_get)
    return FredAlfredProvider("KEY", lag or {}), seen


def test_fred_provider_requests_vintage_for_asof(monkeypatch):
    obs = [
        {"realtime_start": "2024-02-01", "realtime_end": "2024-02-14", "date": "2024-01-01", "value": "100"},
        {"realtime_start": "2024-02-15", "realtime_end": "2024-03-01", "date": "2024-01-01", "value": "101"},
        {"realtime_start": "2024-03-01", "realtime_end": "2024-03-01", "date": "2024-01-01", "value": "102"},
    ]
    p, seen = _fred(monkeypatch, obs)
    s = p.fetch("CPIAUCSL", date(2024, 3, 1))
    assert seen["params"]["realtime_end"] == "2024-03-01"
    assert seen["params"]["series_id"] == "CPIAUCSL"
    assert seen["params"]["realtime_start"] <= "2024-01-01"
    # latest value published on or before asof
    assert s.loc[pd.Timestamp("2024-01-01")] == 102.0


def test_fred_provider_applies_publication_lag(monkeypatch):
    obs = [
        {"realtime_start": "2024-01-02", "realtime_end": "2024-06-01", "date": "2024-01-01", "value": "1"},
        {"realtime_start": "2024-01-20", "realtime_end": "2024-06-01", "date": "2024-02-01", "value": "2"},
        {"realtime_start": "2024-01-20", "realtime_end": "2024-06-01", "date": "2024-03-01", "value": "."},
    ]
    p, _ = _fred(monkeypatch, obs, {"X": 30})
    s = p.fetch("X", date(2024, 3, 1))  # cutoff 2024-01-31 -> drops 2024-02-01
    assert list(s.index) == [pd.Timestamp("2024-01-01")]
    assert s.iloc[0] == 1.0


def test_fred_provider_api_key_not_logged(monkeypatch, caplog):
    p, _ = _fred(monkeypatch, [])
    with caplog.at_level("DEBUG"):
        s = p.fetch("X", date(2024, 3, 1))
    assert s.empty and "KEY" not in caplog.text


def test_ibkr_import_does_not_require_ib_async():
    IBKRProvider("127.0.0.1", 4001, 1)  # construction must not import/connect


def test_ibkr_provider_maps_bars_to_schema():
    def bar(d, o, c):
        return SimpleNamespace(date=d, open=o, high=o + 1, low=o - 1, close=c, volume=10)

    class FakeIB:
        def __init__(self):
            self.what = []

        def isConnected(self):
            return True

        def reqHistoricalData(self, contract, endDateTime, durationStr, barSizeSetting,
                              whatToShow, useRTH, formatDate=1):
            self.what.append(whatToShow)
            assert barSizeSetting == "1 day"
            adj = whatToShow == "ADJUSTED_LAST"
            return [bar(date(2024, 1, 2), 10, 9.5 if adj else 10.5),
                    bar(date(2024, 1, 3), 11, 10.5 if adj else 11.5),
                    bar(date(2024, 1, 4), 12, 12)]  # outside range

    ib = FakeIB()
    p = IBKRProvider("h", 1, 2, ib=ib, contract_factory=lambda t: t)
    df = p.fetch(["AAA"], date(2024, 1, 2), date(2024, 1, 3))
    assert list(df.columns) == PRICE_COLUMNS
    assert len(df) == 2 and set(df["ticker"]) == {"AAA"}
    assert df.iloc[0]["close"] == 10.5 and df.iloc[0]["adj_close"] == 9.5
    assert set(ib.what) == {"TRADES", "ADJUSTED_LAST"}


@pytest.mark.live
def test_yfinance_live():
    df = YFinanceProvider().fetch(["SPY"], date(2024, 1, 2), date(2024, 1, 10))
    assert list(df.columns) == PRICE_COLUMNS and len(df) > 0


@pytest.mark.live
def test_fred_live():
    import os
    s = FredAlfredProvider(os.environ["FRED_API_KEY"], {}).fetch("CPIAUCSL", date(2024, 6, 1))
    assert len(s) > 0


@pytest.mark.live
def test_ibkr_live():
    df = IBKRProvider("127.0.0.1", 4001, 99).fetch(["SPY"], date(2024, 1, 2), date(2024, 1, 10))
    assert list(df.columns) == PRICE_COLUMNS
