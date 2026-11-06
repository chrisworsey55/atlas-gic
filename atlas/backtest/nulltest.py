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
    def __init__(self, ics: np.ndarray, dead_zone: float | None = None,
                 keep_margin: float | None = None):
        self.ics = np.asarray(ics, dtype=float)
        self._dead_zone = dead_zone
        self._keep_margin = keep_margin

    def quantile(self, q: float) -> float:
        return float(np.quantile(np.abs(self.ics), q))

    def dead_zone(self) -> float:
        """95th percentile of |shrunk_ic - group_mean| under the null (the production rule)."""
        return self.quantile(0.95) if self._dead_zone is None else self._dead_zone

    def keep_margin(self) -> float:
        """95th percentile of |IC_a - IC_b| for two independent shuffles of one agent."""
        return self.quantile(0.95) if self._keep_margin is None else self._keep_margin


def _check_centred(ics: np.ndarray) -> None:
    if len(ics) == 0:
        raise ValueError("null test produced no ICs")
    se = max(float(np.std(ics)), STD_FLOOR) / math.sqrt(len(ics))
    if abs(float(np.mean(ics))) > LEAK_SIGMAS * se:
        raise LeakageSuspected(
            f"shuffled IC mean {np.mean(ics):.4f} is not centred on 0 (se {se:.4g}, n={len(ics)})")


def null_test(picks: pd.DataFrame, n_shuffles: int, seed: int, *, k: float = 20.0,
              warmup_picks: int = 30,
              shuffle_fn: Callable[[np.ndarray, np.random.Generator], np.ndarray] | None = None
              ) -> NullDistribution:
    shuffle = shuffle_fn or (lambda x, rng: rng.permutation(x))
    df = picks[picks["scored"].astype(bool)].dropna(subset=["conviction", "excess"])
    data = {}
    for agent, g in df.groupby("agent", sort=True):
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
    return NullDistribution(arr, float(np.quantile(devs, 0.95)), float(np.quantile(diffs, 0.95)))
