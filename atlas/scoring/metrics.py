"""Scoring metrics: excess return, rank IC, hit rate.

Sign convention (defined once, here): `excess_return` is always computed for a
LONG position in the ticker. A short pick is stored with SIGNED conviction
(+conviction long, -conviction short), so rank IC correlates signed conviction
with the raw long-side excess return. A short on a stock that falls therefore
has negative conviction and negative excess: a positive contribution to IC.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

NAN = float("nan")


def rank_ic(convictions: Sequence[float], excess: Sequence[float]) -> float:
    """Spearman rank correlation; NaN if <3 points or either series is constant."""
    c = np.asarray(list(convictions), dtype=float)
    e = np.asarray(list(excess), dtype=float)
    if len(c) != len(e):
        raise ValueError("convictions and excess must have equal length")
    ok = ~(np.isnan(c) | np.isnan(e))
    c, e = c[ok], e[ok]
    if len(c) < 3 or np.ptp(c) == 0 or np.ptp(e) == 0:
        return NAN
    rho = spearmanr(c, e).statistic
    return NAN if rho is None or math.isnan(rho) else float(rho)


def hit_rate(excess: Sequence[float]) -> float:
    """Fraction of observations with excess > 0; NaN if empty."""
    e = np.asarray(list(excess), dtype=float)
    e = e[~np.isnan(e)]
    return NAN if len(e) == 0 else float((e > 0).mean())


def excess_return(ticker_prices: pd.Series, bench_prices: pd.Series,
                  horizon: int, cost_frac: float) -> float:
    """Long-side return over `horizon` bars minus benchmark return minus cost.

    Both series start at the entry bar (position 0); exit is position `horizon`.
    `cost_frac` (round trip) is charged to the pick leg only. NaN if a series
    is too short or an endpoint is missing.
    """
    if len(ticker_prices) <= horizon or len(bench_prices) <= horizon:
        return NAN
    t0, t1 = float(ticker_prices.iloc[0]), float(ticker_prices.iloc[horizon])
    b0, b1 = float(bench_prices.iloc[0]), float(bench_prices.iloc[horizon])
    if any(math.isnan(x) for x in (t0, t1, b0, b1)) or t0 == 0 or b0 == 0:
        return NAN
    return (t1 / t0 - 1.0) - cost_frac - (b1 / b0 - 1.0)
