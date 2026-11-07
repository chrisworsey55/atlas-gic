"""CIO ablation: does the CIO beat a plain weight-averaged long-only baseline?"""
from __future__ import annotations

import pandas as pd

from atlas.schemas import AgentResult

NON_BASELINE = frozenset({"cro", "cio", "execution", "control"})


def _total(r: pd.Series) -> float:
    return float((1.0 + r).prod() - 1.0)


def cio_ablation(cio_returns: pd.Series, baseline_returns: pd.Series) -> dict:
    j = pd.concat([cio_returns.rename("cio"), baseline_returns.rename("base")],
                  axis=1, join="inner").dropna()
    if len(j) < 2:
        raise ValueError(f"CIO and baseline share {len(j)} dates; need >= 2")
    cio, base = _total(j["cio"]), _total(j["base"])
    return {"cio_total": cio, "baseline_total": base, "cio_beats_baseline": bool(cio > base)}


def baseline_target_weights(day_results: dict[str, AgentResult], weights: dict[str, float],
                            max_positions: int = 8,
                            layers: dict[str, int] | None = None) -> dict[str, float]:
    """Sum agent_weight * conviction over LONG picks of valid layer 1-3 agents, keep the
    top `max_positions`, normalise to 1.0. Empty dict if nothing positive."""
    score: dict[str, float] = {}
    for name, res in day_results.items():
        if res.status != "ok":
            continue
        if layers is not None:
            if not 1 <= layers.get(name, 0) <= 3:
                continue
        elif name in NON_BASELINE:
            continue
        w = weights.get(name, 1.0)
        for p in res.output.picks:
            if p.direction == "long":
                score[p.ticker] = score.get(p.ticker, 0.0) + w * p.conviction
    top = sorted(((t, s) for t, s in score.items() if s > 0), key=lambda x: (-x[1], x[0]))[:max_positions]
    total = sum(s for _, s in top)
    return {t: s / total for t, s in top} if total > 0 else {}
