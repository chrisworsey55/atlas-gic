"""Regenerates the committed golden price CSVs (deterministic; run from anywhere).

    python tests/golden/fixtures/make_fixtures.py

45 weekdays from Monday 2024-01-01 (no holidays: a synthetic calendar), 8 tickers. The
golden run uses the first 40 days; the last 5 exist so day-40 orders can fill and the
calendar has next-day closes for scoring.
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np

TICKERS = ["SPY", "SMH", "XLF", "XLE", "XLV", "AAA", "BBB", "CCC"]
BASE = {"SPY": 470.0, "SMH": 170.0, "XLF": 37.0, "XLE": 84.0, "XLV": 135.0,
        "AAA": 52.0, "BBB": 18.0, "CCC": 240.0}
N_DAYS = 45


def main() -> None:
    out = Path(__file__).parent / "prices"
    out.mkdir(exist_ok=True)
    days, d = [], date(2024, 1, 1)
    while len(days) < N_DAYS:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rng = np.random.default_rng(20240101)
    market = rng.normal(0.0004, 0.009, N_DAYS)
    for t in TICKERS:
        beta = 1.0 if t == "SPY" else float(rng.uniform(0.6, 1.4))
        idio = np.zeros(N_DAYS) if t == "SPY" else rng.normal(0.0, 0.012, N_DAYS)
        close = BASE[t] * np.cumprod(1.0 + beta * market + idio)
        gap = rng.normal(0.0, 0.003, N_DAYS)
        lines = ["date,ticker,open,high,low,close,adj_close,volume"]
        for i, day in enumerate(days):
            c = float(close[i])
            o = c * (1.0 + float(gap[i]))
            hi, lo = max(o, c) * 1.004, min(o, c) * 0.996
            lines.append(f"{day.isoformat()},{t},{o:.4f},{hi:.4f},{lo:.4f},{c:.4f},{c:.4f},"
                         f"{1_000_000 + 1000 * i}")
        (out / f"{t}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
