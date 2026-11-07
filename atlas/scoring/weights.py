"""Darwinian agent weights: shrunk IC, dead zone, bounds, JSON persistence.

Each round an agent's rank IC is shrunk toward the group mean (over eligible
agents) in proportion to how few observations it has. Weights move only when
the shrunk IC is further than `dead_zone` from the group mean: x1.05 above,
x0.95 below. Warming-up and NaN-IC agents never change. Weights are clamped to
[W_MIN, W_MAX] after every step. `dead_zone is None` (not yet calibrated by the
null test) freezes all weights.
"""
from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from atlas.scoring.metrics import rank_ic
from atlas.scoring.store import ScoreStore, non_overlapping

log = logging.getLogger(__name__)

W_MIN, W_MAX, W_INIT = 0.3, 2.5, 1.0
UP_FACTOR, DOWN_FACTOR = 1.05, 0.95


@dataclass
class AgentStats:
    agent: str
    n_picks: int
    ic: float


def shrunk_ic(ic: float, n: int, group_mean: float, k: float) -> float:
    return (n * ic + k * group_mean) / (n + k)


def _clamp(w: float) -> float:
    return min(W_MAX, max(W_MIN, w))


def update_weights(weights: dict[str, float], stats: list[AgentStats], dead_zone: float | None,
                   k: float, warmup_picks: int) -> dict[str, float]:
    if dead_zone is None:
        log.info("dead_zone not calibrated (null test pending): weights frozen")
        return dict(weights)
    out = {a: _clamp(w) for a, w in weights.items()}
    for s in stats:
        out.setdefault(s.agent, W_INIT)
    eligible = [s for s in stats if s.n_picks >= warmup_picks and not math.isnan(s.ic)]
    if len(eligible) < 2:
        return out
    mean = sum(s.ic for s in eligible) / len(eligible)
    for s in eligible:
        dev = shrunk_ic(s.ic, s.n_picks, mean, k) - mean
        if dev > dead_zone:
            out[s.agent] = _clamp(out[s.agent] * UP_FACTOR)
        elif dev < -dead_zone:
            out[s.agent] = _clamp(out[s.agent] * DOWN_FACTOR)
    return out


class WeightBook:
    def __init__(self, path: Path):
        self._path = Path(path)

    def load(self) -> dict[str, float]:
        if not self._path.exists():
            return {}
        return {str(a): float(w) for a, w in json.loads(self._path.read_text()).items()}

    def save(self, weights: dict[str, float]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_name(self._path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(weights, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self._path)


def window_frame(scores: ScoreStore, agent: str, asof_window: int, horizon: int = 1,
                 max_date: date | None = None):
    """The scored, non-overlapping rows behind an agent's rolling IC (the last `asof_window`
    rows). `max_date` drops picks dated after it BEFORE the window is taken."""
    df = scores.picks_frame(agent, horizon)
    df = df[df["scored"]]
    if max_date is not None:
        df = df[df["date"] <= max_date]
    df = df.sort_values(["date", "ticker"], kind="stable")
    df = non_overlapping(df, horizon).sort_values(["date", "ticker"], kind="stable")
    return df.tail(asof_window)


def collect_stats(scores: ScoreStore, agents: list[str], asof_window: int,
                  horizon: int = 1, max_date: date | None = None) -> list[AgentStats]:
    out = []
    for agent in agents:
        df = window_frame(scores, agent, asof_window, horizon, max_date)
        out.append(AgentStats(agent, len(df), rank_ic(df["conviction"], df["excess"])))
    return out
