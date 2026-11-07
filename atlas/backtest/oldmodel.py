"""Older-model check (optional validity control).

Rerun the backtest with an OLDER model over ONLY the part of the run that lies after that
model's training cutoff. For that window the older model cannot have seen the data, so a
result that holds there is not explained by memory of the period.

Model availability is the user's responsibility: ``Settings.older_model`` is only a name and
nothing here lists or probes models. A model that cannot be reached surfaces as
``OlderModelUnavailable`` with a clear message, never a traceback or an empty result.

The check runs in ``<run_dir>/older_model/<model>/`` with its own checkpoint, prediction log,
score store, weights, failure log and LLM cache, so it never touches the main run's state.
"""
from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Callable

from atlas.backtest.checkpoint import Checkpoint
from atlas.backtest.predlog import PredictionLog
from atlas.backtest.runner import (
    HISTORY_NAME, BacktestConfig, BacktestResult, Periods, RunPaused, run_backtest,
)
from atlas.data.store import PriceStore
from atlas.engine.failures import FailureLog
from atlas.engine.llm import ApiError, BudgetExceeded, CachedLLM, LLMClient, RateLimited, Timeout
from atlas.engine.pipeline import Pipeline
from atlas.scoring.store import ScoreStore
from atlas.scoring.weights import WeightBook

log = logging.getLogger("atlas.backtest.oldmodel")

MIN_WINDOW_DAYS = 3  # one trading day per period of the runner's three-period contract


class OlderModelUnavailable(RuntimeError):
    """The older model could not be used (unreachable, unknown name, or budget exhausted)."""


@dataclass
class OlderModelSetup:
    """What the caller has already built for the main run, reused for the older model.

    ``make_pipeline(llm, failures)`` returns the pipeline wired to the older model's LLM;
    ``client`` is the raw client for the older model (``AnthropicClient(older_model)`` in
    production, a ``StubClient`` in tests); ``broker``/``risk`` are stateless simulators."""
    store: PriceStore
    make_pipeline: Callable[[CachedLLM, FailureLog], Pipeline]
    client: LLMClient
    broker: object
    risk: object
    initial_cash: float = 100_000.0
    lookback_days: int = 60
    macro_series: tuple = ()
    universe_sectors: dict[str, str] | None = None


def period_after_cutoff(periods: Periods, model_cutoff: date) -> tuple[date, date] | None:
    """The part of the run strictly after ``model_cutoff``: (max(first day, cutoff + 1 day),
    last day), or None if every date precedes it (the last day is on or before the cutoff)."""
    first, last = periods.train[0], periods.holdout[1]
    if last <= model_cutoff:
        return None
    return max(first, model_cutoff + timedelta(days=1)), last


def _window_periods(periods: Periods, older_cutoff: date, store: PriceStore) -> Periods | None:
    """Periods covering only post-cutoff trading days. Each original period is clipped to start
    after the cutoff; if all three keep trading days the clipped periods are used as they are.
    If the cutoff swallowed a whole period, the post-cutoff trading days are split into three
    chronological chunks (train, validation, holdout) cut inside the surviving period(s), never
    across a gap between periods, because the runner needs all three; those roles then only
    label chunks of the window. None if fewer than MIN_WINDOW_DAYS days."""
    start = older_cutoff + timedelta(days=1)
    clipped, days = {}, []
    for name, (lo, hi) in periods.items():  # the run's own days only: gaps between periods stay out
        lo = max(lo, start)
        ds = store.trading_days(lo, hi) if lo <= hi else []
        clipped[name] = (lo, hi, ds)
        days += ds
    days = sorted(set(days))
    if len(days) < MIN_WINDOW_DAYS:
        return None
    if all(ds for _, _, ds in clipped.values()):
        return Periods(**{n: (lo, hi) for n, (lo, hi, _) in clipped.items()})
    # a period's range must not span dates that were never part of the run (gaps between the
    # original periods), so chunks are cut inside the surviving period(s), never across a gap
    segs = [ds for _, _, ds in clipped.values() if ds]
    if len(segs) == 1:
        d, a, b = segs[0], len(segs[0]) // 3, 2 * len(segs[0]) // 3
        chunks = [d[:a], d[a:b], d[b:]]
    else:  # two surviving periods: split the longer one in half
        i = max(range(2), key=lambda k: (len(segs[k]), -k))
        h = len(segs[i]) // 2
        parts = [segs[i][:h], segs[i][h:]]
        chunks = parts + [segs[1]] if i == 0 else [segs[0]] + parts
    return Periods(*[(c[0], c[-1]) for c in chunks])


def _dirname(model: str) -> str:
    """A single safe path component for `model`. A name that needed any change gets a short
    hash of the raw name, so 'a/b' and 'a_b' (or '../x') never collide or leave older_model/."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model).strip(".")
    if safe and safe == model:
        return safe
    return f"{safe or 'model'}-{hashlib.sha256(model.encode('utf-8')).hexdigest()[:8]}"


def run_older_model_check(cfg: BacktestConfig, older_model: str, older_cutoff: date,
                          setup: OlderModelSetup, *, run_dir: Path, budget_usd: float | None = None,
                          confirm: bool = False) -> BacktestResult | None:
    """Run ``run_backtest`` with ``older_model`` over only the post-``older_cutoff`` window.

    Returns None (and logs why) when no part of the run lies after the cutoff. State goes to
    ``<run_dir>/older_model/<model>/``; re-running resumes from its checkpoint.
    ``budget_usd`` is this check's own hard cap (default ``cfg.settings.budget_usd``); the
    caller decides how much of the run's overall budget to give it.
    Raises OlderModelUnavailable if the model cannot be reached or the budget is exhausted."""
    window = period_after_cutoff(cfg.periods, older_cutoff)
    if window is None:
        log.warning("older-model check skipped: no period lies after %s's training cutoff %s "
                    "(the run ends %s)", older_model, older_cutoff, cfg.periods.holdout[1])
        return None
    periods = _window_periods(cfg.periods, older_cutoff, setup.store)
    if periods is None:
        log.warning("older-model check skipped: fewer than %d trading days lie after %s's training "
                    "cutoff %s (window %s..%s)", MIN_WINDOW_DAYS, older_model, older_cutoff, *window)
        return None

    s = cfg.settings
    settings = s.model_copy(update={"model": older_model, "model_cutoff": older_cutoff})
    ocfg = BacktestConfig(periods, settings, cfg.limits, False, cfg.dry_run)
    sub = Path(run_dir) / "older_model" / _dirname(older_model)
    if budget_usd is None:
        # the run's ONE hard cap: what remains after the backtest, control, autoresearch, probe and
        # every other older-model check; this model's own earlier spend is restored by the runner
        from atlas.cli import _load_checkpoint, spend_so_far
        own = float((_load_checkpoint(Checkpoint(sub / "checkpoint.json")) or {}).get("spent_usd", 0.0))
        spent = sum(spend_so_far(Path(run_dir)).values()) - own
        cap = s.budget_usd - spent
        if cap <= 0:
            raise OlderModelUnavailable(
                f"older-model check with {older_model} not run: LLM budget exhausted (${spent:.2f} of "
                f"${s.budget_usd:.2f} already spent on this run); raise budget_usd")
    else:
        cap = budget_usd
    sub.mkdir(parents=True, exist_ok=True)
    llm = CachedLLM(setup.client, sub / "llm_cache.sqlite", older_model,
                    (s.usd_per_mtok_input, s.usd_per_mtok_output), cap)
    failures = FailureLog(sub / "failures.sqlite")
    pipeline = setup.make_pipeline(llm, failures)
    hint = (f"Check that '{older_model}' is a model name your API key can use "
            "(availability is not checked here)")
    try:
        return run_backtest(
            ocfg, setup.store, pipeline, setup.broker, setup.risk, ScoreStore(sub / "scores.sqlite"),
            WeightBook(sub / "weights.json"), PredictionLog(sub / "prediction_log.jsonl"),
            Checkpoint(sub / "checkpoint.json"), confirm=confirm, initial_cash=setup.initial_cash,
            universe_sectors=setup.universe_sectors, lookback_days=setup.lookback_days,
            macro_series=setup.macro_series, run_dir=sub, history_path=sub / HISTORY_NAME)
    except RunPaused as e:
        if e.kind == "budget":
            raise OlderModelUnavailable(
                f"older-model check with {older_model} stopped: LLM budget exhausted ({e.reason}); "
                f"raise the budget and re-run to resume from {sub}") from None
        codes = ", ".join(sorted(set(e.failed.values()))) or "unknown"
        raise OlderModelUnavailable(
            f"older-model check with {older_model} stopped on {e.day}: agents failed with {codes}. "
            f"{hint}. The check can be resumed from {sub}") from None
    except (ApiError, RateLimited, Timeout) as e:
        raise OlderModelUnavailable(f"older model {older_model} is not reachable ({e}). {hint}") from None
    except BudgetExceeded as e:
        raise OlderModelUnavailable(f"older-model check with {older_model} stopped: LLM budget "
                                    f"exhausted ({e})") from None
