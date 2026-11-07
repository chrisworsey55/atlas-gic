"""Null test: how large can an agent's rank IC look by pure chance?

Convictions are permuted within each agent (real excess returns untouched), which
destroys any conviction/outcome relationship while keeping each agent's marginal
distributions. Repeating this gives the null distributions used to calibrate the
weight dead zone and the autoresearch keep margin.

Leakage rule: with a correct permutation the shuffled raw ICs are centred on zero.
`LeakageSuspected` is raised when |mean(ics)| > 4 * max(std(ics), STD_FLOOR) / sqrt(N);
the floor keeps a degenerate all-equal distribution (std = 0) from dividing by zero,
and any non-zero mean of such a distribution is then flagged.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from atlas.scoring.metrics import rank_ic
from atlas.scoring.weights import shrunk_ic

STD_FLOOR = 1e-9
LEAK_SIGMAS = 4.0


class LeakageSuspected(Exception):
    """The shuffled-IC null is not centred on zero: shuffle or scoring is broken/leaking."""


class NullDistribution:
    """`ics` are all raw shuffled ICs (histogram, `quantile`). `deviations` and `pair_diffs`
    are filled by `null_test`; `dead_zone()` / `keep_margin()` raise if they are absent."""

    def __init__(self, ics: np.ndarray, deviations: np.ndarray | None = None,
                 pair_diffs: np.ndarray | None = None):
        self.ics = np.asarray(ics, dtype=float)
        self.deviations = None if deviations is None else np.asarray(deviations, dtype=float)
        self.pair_diffs = None if pair_diffs is None else np.asarray(pair_diffs, dtype=float)

    def quantile(self, q: float) -> float:
        return float(np.quantile(np.abs(self.ics), q))

    def dead_zone(self) -> float:
        """95th percentile of |shrunk_ic - group_mean| under the null (the production rule)."""
        if self.deviations is None or len(self.deviations) == 0:
            raise ValueError("no shrunk-IC deviations: build this distribution with null_test")
        return float(np.quantile(self.deviations, 0.95))

    def keep_margin(self) -> float:
        """95th percentile of |IC_a - IC_b| for two independent shuffles of one agent.

        Independent shuffles OVERSTATE the null spread of a paired replay whose old and new
        picks are correlated (measured 95th pct of |dIC|: 0.223 at 0% overlap, 0.156 at 50%,
        0.103 at 80%), so this margin is conservative: no false keeps, lower power."""
        if self.pair_diffs is None or len(self.pair_diffs) == 0:
            raise ValueError("no paired-IC differences: build this distribution with null_test")
        return float(np.quantile(self.pair_diffs, 0.95))


def _check_centred(ics: np.ndarray) -> None:
    if len(ics) == 0:
        raise ValueError("null test produced no ICs")
    se = max(float(np.std(ics)), STD_FLOOR) / math.sqrt(len(ics))
    if abs(float(np.mean(ics))) > LEAK_SIGMAS * se:
        raise LeakageSuspected(
            f"shuffled IC mean {np.mean(ics):.4f} is not centred on 0 (se {se:.4g}, n={len(ics)})")


def null_test(picks: pd.DataFrame, n_shuffles: int, seed: int, *, k: float = 20.0,
              warmup_picks: int = 30, window: int | None = None,
              shuffle_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None
              ) -> NullDistribution:
    """`picks` is a picks_frame-like frame, already non-overlapping 1-day-horizon samples.

    `window` keeps each agent's last `window` rows (by date, ticker) as `collect_stats` does;
    it MUST equal Settings.rolling_ic_window. The default None uses all rows, which
    calibrates a too-tight dead zone for production's shorter window (more false moves)."""
    shuffle = shuffle_fn or (lambda x, rng: rng.permutation(x))
    df = picks[picks["scored"].astype(bool)].dropna(subset=["conviction", "excess"])
    data = {}
    for agent, g in df.groupby("agent", sort=True):
        g = g.sort_values(["date", "ticker"], kind="stable")
        if window is not None:
            g = g.tail(window)
        if len(g) >= warmup_picks:
            data[agent] = (g["conviction"].to_numpy(float), g["excess"].to_numpy(float))
    if len(data) < 2:
        raise ValueError(f"null test needs >= 2 agents with >= {warmup_picks} scored picks; "
                         f"got {len(data)}")
    rng = np.random.default_rng(seed)
    raw: list[float] = []
    devs: list[float] = []
    diffs: list[float] = []
    for _ in range(n_shuffles):
        ics = {}
        for agent, (conv, exc) in data.items():
            ic_a = rank_ic(shuffle(conv, rng), exc)
            ic_b = rank_ic(shuffle(conv, rng), exc)
            if not math.isnan(ic_a):
                ics[agent] = ic_a
                raw.append(ic_a)
                if not math.isnan(ic_b):
                    diffs.append(abs(ic_a - ic_b))
        if len(ics) >= 2:
            mean = sum(ics.values()) / len(ics)
            for agent, ic in ics.items():
                devs.append(abs(shrunk_ic(ic, len(data[agent][0]), mean, k) - mean))
    arr = np.asarray(raw, dtype=float)
    _check_centred(arr)
    if not devs or not diffs:
        raise ValueError("null test produced no usable shuffles (constant convictions/excess?)")
    return NullDistribution(arr, np.asarray(devs), np.asarray(diffs))
