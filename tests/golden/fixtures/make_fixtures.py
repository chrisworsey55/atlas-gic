"""Regenerates the committed golden price CSVs (deterministic; run from anywhere).

    python tests/golden/fixtures/make_fixtures.py

45 weekdays from Monday 2024-01-01 (no holidays: a synthetic calendar), 8 tickers. The
golden run uses the first 40 days; the last 5 exist so day-40 orders can fill and the
calendar has next-day closes for scoring.

A second, longer set (prices_long/, 85 weekdays; the long golden run uses the first 80: train
45, validation 25, holdout 10) serves the validation-confirmation end-to-end test. It is a
separate directory with its own random stream, so the 45-day set above is byte-identical.
"""
from datetime import date, timedelta
from pathlib import Path

import numpy as np

TICKERS = ["SPY", "SMH", "XLF", "XLE", "XLV", "AAA", "BBB", "CCC"]
BASE = {"SPY": 470.0, "SMH": 170.0, "XLF": 37.0, "XLE": 84.0, "XLV": 135.0,
        "AAA": 52.0, "BBB": 18.0, "CCC": 240.0}
N_DAYS = 45
LONG_DAYS = 85


def main() -> None:
    generate(Path(__file__).parent / "prices", N_DAYS)
    generate(Path(__file__).parent / "prices_long", LONG_DAYS)


def generate(out: Path, n_days: int) -> None:
    out.mkdir(exist_ok=True)
    days, d = [], date(2024, 1, 1)
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    rng = np.random.default_rng(20240101)
    market = rng.normal(0.0004, 0.009, n_days)
    for t in TICKERS:
        beta = 1.0 if t == "SPY" else float(rng.uniform(0.6, 1.4))
        idio = np.zeros(n_days) if t == "SPY" else rng.normal(0.0, 0.012, n_days)
        close = BASE[t] * np.cumprod(1.0 + beta * market + idio)
        gap = rng.normal(0.0, 0.003, n_days)
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
