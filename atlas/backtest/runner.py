"""Backtest runner: the day loop over chronological periods, checkpoint/resume, cost
control and input anonymisation.

For each trading day T (spec section 4, in this order):

1. as-of-T snapshot (no data after T);
2. the 4-layer pipeline, with weights from the WeightBook;
3. a skipped day (too few valid agents, CIO/execution failed) proposes no trades and is
   recorded in ``skipped``. Exception (ruling H): if agents failed with TRANSIENT
   infrastructure errors (api_error / rate_limited / timeout) and those failures decide
   the skip, the run PAUSES instead: the last completed day stays the commit point (its
   checkpoint is re-saved with the spend so far) and ``RunPaused`` is raised; nothing of
   day T is traded, logged or scored, and resuming re-runs T;
4. otherwise the proposed orders go through the RiskLayer (prices = last-known closes as
   of T; prior_close_equity = previous recorded equity; peak = running max of recorded
   equity) and the approved orders are submitted to the broker with ``portfolio=``; they
   fill at T+1's open and the fills are applied to the portfolio. Any exception from the
   risk layer sends NO orders. A halt (kill switch) is STICKY (rulings J/L): its flatten
   orders are sent on the day it fires, afterwards no orders at all until the checkpoint's
   flag is cleared with ``clear_halt``. On skipped days the risk layer is still evaluated
   with no orders so the kill switch cannot be starved by LLM failures;
5. the day's agent results are appended to the PredictionLog (fsynced) BEFORE
   ``record_picks`` and ``score_matured(asof=T)``;
6. Darwinian weights are updated (``dead_zone=None`` freezes them, ruling O) and saved;
7. equity for T is the portfolio marked at T's close BEFORE T's own fills (those happen at
   T+1's open); SPY buy-and-hold and an equal-weight universe basket, both normalised to
   the starting equity, are recorded alongside;
8. the checkpoint is saved: it is the commit point. Mid-day state is never checkpointed.

Crash safety: a crash after T's log append / fills / scoring but before the checkpoint
re-runs T on resume from the checkpoint of T-1. LLM outputs come from the cache, so the
idempotent ``PredictionLog.append`` no-ops, ``record_picks`` and ``score_matured`` are
idempotent, weights are restored from the checkpoint, and fills apply exactly once because
the portfolio comes from the checkpoint. A ``BudgetExceeded`` from the LLM layer also
pauses: the checkpoint is re-saved with the spend and ``RunPaused`` (kind "budget") is
raised from it.

``BacktestResult.day_results`` only holds days executed by this invocation (a resumed run
does not reconstruct earlier days' agent results; they are in the PredictionLog).
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import tempfile
from contextlib import ExitStack
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from atlas.backtest.checkpoint import Checkpoint, dump_date, load_date
from atlas.backtest.predlog import PredictionLog
from atlas.config import Limits, Settings
from atlas.data.store import PriceStore, Snapshot
from atlas.engine.llm import BudgetExceeded, CachedLLM, StubClient
from atlas.engine.pipeline import DayResult, Pipeline, _last_close, orders_from_targets
from atlas.portfolio.models import Fill, Order, Portfolio
from atlas.risk.layer import RiskLayer
from atlas.schemas import AgentResult, FailureReason
from atlas.scoring.store import ScoreStore
from atlas.scoring.weights import WeightBook, collect_stats, update_weights

log = logging.getLogger(__name__)

BENCHMARK = "SPY"
PERIOD_NAMES = ("train", "validation", "holdout")
TRANSIENT = frozenset({FailureReason.API_ERROR, FailureReason.RATE_LIMITED, FailureReason.TIMEOUT})
STATE_VERSION = 1
WEIGHTED_KINDS = ("macro", "sector", "superinvestor")  # CIO/execution are scored by ablation
DRY_RUN_RESPONSE = json.dumps({"view": "neutral", "confidence": 0.0, "picks": []})

LIMITATIONS = [
    "Prices come from the provider adjusted as of today: adj_close and split-adjusted bars embed "
    "splits/dividends that happened after each simulated day (accepted data-level leak, ruling G); "
    "results are slightly optimistic.",
    "Pre-cutoff periods are a plumbing test, not evidence of skill: the model may have seen the "
    "period in training.",
    "Fixed universe of current liquid names: survivorship bias from delisted names is not modelled.",
    "Anonymisation is partial: macro levels and return patterns can still identify an era, and "
    "agent prompt files may name real tickers.",
    "Portfolio and benchmarks are marked at close prices; dividends are not credited to the "
    "portfolio.",
]


# ---------------------------------------------------------------------------- config

def _check_range(name: str, rng) -> tuple[date, date]:
    if not (isinstance(rng, tuple) and len(rng) == 2 and all(isinstance(d, date) for d in rng)):
        raise ValueError(f"{name} period must be a (start, end) tuple of dates, got {rng!r}")
    if rng[0] > rng[1]:
        raise ValueError(f"{name} period starts after it ends: {rng[0]} > {rng[1]}")
    return rng


@dataclass(frozen=True)
class Periods:
    """Chronological, non-overlapping train < validation < holdout (inclusive ranges)."""
    train: tuple[date, date]
    validation: tuple[date, date]
    holdout: tuple[date, date]

    def __post_init__(self):
        for name, rng in self.items():
            _check_range(name, rng)
        if not self.train[1] < self.validation[0]:
            raise ValueError("validation must start after train ends (chronological, non-overlapping)")
        if not self.validation[1] < self.holdout[0]:
            raise ValueError("holdout must start after validation ends (chronological, non-overlapping)")

    def items(self) -> list[tuple[str, tuple[date, date]]]:
        return [(n, getattr(self, n)) for n in PERIOD_NAMES]

    def period_of(self, day: date) -> str:
        for name, (lo, hi) in self.items():
            if lo <= day <= hi:
                return name
        raise ValueError(f"{day} is in no configured period")


@dataclass(frozen=True)
class BacktestConfig:
    periods: Periods
    settings: Settings
    limits: Limits
    anonymize: bool
    dry_run: bool

    def __post_init__(self):
        cutoff = self.settings.model_cutoff
        if not self.periods.holdout[0] > cutoff:
            raise ValueError(f"holdout must start after the model's training cutoff {cutoff} "
                             f"(starts {self.periods.holdout[0]})")


class CostConfirmationRequired(RuntimeError):
    def __init__(self, estimate: float, ceiling: float):
        self.estimate = estimate
        super().__init__(f"estimated LLM cost ${estimate:.2f} exceeds the confirmation ceiling "
                         f"${ceiling:.2f}; re-run with confirm=True to proceed")


class RunPaused(RuntimeError):
    """The run stopped at a resumable point; the checkpoint holds the last completed day.

    kind: "transient" (api_error/rate_limited/timeout decided the day) or "budget".
    failed: agent -> failure reason code for every failed agent on `day` (empty for budget).
    """

    def __init__(self, reason: str, day: date, failed: dict[str, str], kind: str):
        self.reason, self.day, self.failed, self.kind = reason, day, failed, kind
        super().__init__(f"run paused on {day}: {reason}")


class ResumeRefused(RuntimeError):
    pass


@dataclass
class BacktestResult:
    equity: pd.Series
    spy: pd.Series
    basket: pd.Series
    day_results: dict[date, DayResult]
    skipped: dict[date, str]
    limitations: list[str] = field(default_factory=lambda: list(LIMITATIONS))
    risk_errors: dict[date, str] = field(default_factory=dict)
    halted: bool = False
    halt_reason: str | None = None
    final_portfolio: Portfolio | None = None
    spent_usd: float = 0.0
    periods: Periods | None = None


def estimate_cost(cfg: BacktestConfig, n_days: int, n_agents: int) -> float:
    """Upper-level estimate from Settings pricing assumptions; 0 for a dry run. Ignores
    cache hits and retries (cached days are free, retries cost extra)."""
    if cfg.dry_run:
        return 0.0
    s = cfg.settings
    per_call = (s.est_input_tokens_per_call * s.usd_per_mtok_input
                + s.est_output_tokens_per_call * s.usd_per_mtok_output) / 1e6
    return float(n_days) * float(n_agents) * per_call


# ---------------------------------------------------------------------------- anonymisation

def _opaque_ids(tickers) -> dict[str, str]:
    """real -> opaque id, stable across days; ordered by hash so ids don't leak alphabet order."""
    order = sorted(set(tickers), key=lambda t: hashlib.sha256(t.encode("utf-8")).hexdigest())
    width = max(2, len(str(len(order))))
    return {t: f"A{i + 1:0{width}d}" for i, t in enumerate(order)}


def anonymize_snapshot(snapshot: Snapshot, universe_sectors: dict[str, str]
                       ) -> tuple[Snapshot, dict[str, str]]:
    """Opaque ticker ids, each ticker's prices rebased to 100 at its first bar in the window,
    integer relative day index (0 = last bar, -1 the bar before, ...) instead of dates, a
    ``sector`` column, volume removed. Macro series are re-indexed by calendar-day offset
    from asof (<= 0). The returned Snapshot's ``asof`` is the relative index 0, not a date.
    Tickers absent from ``universe_sectors`` are dropped. Returns (snapshot, opaque->real)."""
    fwd = _opaque_ids(universe_sectors)
    df = snapshot.prices
    df = df[df["ticker"].isin(fwd)].copy()
    parts = []
    for _, g in df.groupby("ticker", sort=True):
        g = g.sort_values("date").copy()
        for cols, ref in ((["open", "high", "low", "close"], "close"), (["adj_close"], "adj_close")):
            first = g[ref].astype(float)
            first = first[(first > 0) & first.notna()]
            base = 100.0 / float(first.iloc[0]) if len(first) else float("nan")
            for c in cols:
                g[c] = g[c].astype(float) * base
        parts.append(g)
    p = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].copy()
    dates = sorted(set(p["date"]))
    rel = {d: i - (len(dates) - 1) for i, d in enumerate(dates)}
    p["date"] = [rel[d] for d in p["date"]]
    p["sector"] = [str(universe_sectors[t]) for t in p["ticker"]]
    p["ticker"] = [fwd[t] for t in p["ticker"]]
    p["volume"] = float("nan")
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    asof = pd.Timestamp(snapshot.asof).date()
    macro = {sid: pd.Series(s.to_numpy(), index=[(pd.Timestamp(i).date() - asof).days for i in s.index],
                            name=None)
             for sid, s in snapshot.macro.items()}
    return Snapshot(asof=0, prices=p, macro=macro), {v: k for k, v in fwd.items()}  # type: ignore[arg-type]


def _demap_results(results: dict[str, AgentResult], back: dict[str, str]) -> dict[str, AgentResult]:
    out = {}
    for n, r in results.items():
        if r.output is not None:
            picks = [p.model_copy(update={"ticker": back.get(p.ticker, p.ticker)}) for p in r.output.picks]
            r = r.model_copy(update={"output": r.output.model_copy(update={"picks": picks})})
        out[n] = r
    return out


# ---------------------------------------------------------------------------- state

@dataclass
class _State:
    fingerprint: dict
    cash: float
    positions: dict[str, int]
    peak_equity: float
    prev_equity: float
    weights: dict[str, float]
    last_completed_day: date | None = None
    equity: dict[date, float] = field(default_factory=dict)
    spy: dict[date, float] = field(default_factory=dict)
    basket: dict[date, float] = field(default_factory=dict)
    bench_ref: dict[str, float] = field(default_factory=dict)
    spent_usd: float = 0.0
    predlog_head: tuple[int, str | None] = (0, None)
    halted: bool = False
    halt_reason: str | None = None
    skipped: dict[date, str] = field(default_factory=dict)
    risk_errors: dict[date, str] = field(default_factory=dict)
    last_known: dict[str, float] = field(default_factory=dict)

    def to_json(self) -> dict:
        ds = lambda m: {dump_date(k): v for k, v in sorted(m.items())}  # noqa: E731
        return {
            "version": STATE_VERSION, "fingerprint": self.fingerprint,
            "last_completed_day": dump_date(self.last_completed_day) if self.last_completed_day else None,
            "portfolio": {"cash": self.cash, "positions": dict(sorted(self.positions.items()))},
            "peak_equity": self.peak_equity, "prev_equity": self.prev_equity,
            "weights": dict(sorted(self.weights.items())),
            "equity": ds(self.equity), "spy": ds(self.spy), "basket": ds(self.basket),
            "bench_ref": self.bench_ref, "spent_usd": self.spent_usd,
            "predlog_head": list(self.predlog_head),
            "halted": self.halted, "halt_reason": self.halt_reason,
            "skipped": ds(self.skipped), "risk_errors": ds(self.risk_errors),
            "last_known_prices": self.last_known,
        }

    @classmethod
    def from_json(cls, s: dict) -> "_State":
        if s.get("version") != STATE_VERSION:
            raise ResumeRefused(f"checkpoint version {s.get('version')!r} != {STATE_VERSION}")
        dl = lambda m: {load_date(k): v for k, v in m.items()}  # noqa: E731
        lcd = s["last_completed_day"]
        return cls(
            fingerprint=s["fingerprint"], cash=float(s["portfolio"]["cash"]),
            positions={t: int(n) for t, n in s["portfolio"]["positions"].items()},
            peak_equity=float(s["peak_equity"]), prev_equity=float(s["prev_equity"]),
            weights={a: float(w) for a, w in s["weights"].items()},
            last_completed_day=load_date(lcd) if lcd else None,
            equity=dl(s["equity"]), spy=dl(s["spy"]), basket=dl(s["basket"]),
            bench_ref=dict(s["bench_ref"]), spent_usd=float(s["spent_usd"]),
            predlog_head=(int(s["predlog_head"][0]), s["predlog_head"][1]),
            halted=bool(s["halted"]), halt_reason=s["halt_reason"],
            skipped=dl(s["skipped"]), risk_errors=dl(s["risk_errors"]),
            last_known=dict(s["last_known_prices"]),
        )


def clear_halt(checkpoint: Checkpoint) -> None:
    """Manual restart after a kill-switch halt: clears the sticky flag and resets the peak
    to the last recorded equity (otherwise the drawdown from the old peak would re-halt the
    run on its next day)."""
    s = checkpoint.load()
    if s is None:
        raise ValueError(f"no checkpoint at {checkpoint.path}")
    s["halted"], s["halt_reason"] = False, None
    if s.get("equity"):
        s["peak_equity"] = s["equity"][max(s["equity"])]
    checkpoint.save(s)


def _check_predlog(predlog: PredictionLog, head: tuple[int, str | None]) -> None:
    path = predlog.path
    try:
        predlog.head()
    except ValueError as e:
        raw = path.read_bytes() if path.exists() else b""
        if raw and not raw.endswith(b"\n"):
            raise ResumeRefused(
                f"prediction log {path} has a torn final line — repair it manually (remove the "
                "incomplete last line, which was never committed) before running") from e
        raise ResumeRefused(f"prediction log {path} failed hash-chain verification; refusing to run") from e
    if not predlog.contains_head(*head):
        raise ResumeRefused(
            f"prediction log {path} does not match the checkpoint (expected line {head[0]} with hash "
            f"{head[1]}, at most one uncommitted line after it): truncated or rewritten; refusing to resume")


def _pause_failed(pipeline: Pipeline, day: DayResult) -> dict[str, str] | None:
    """Agents' failure codes if transient failures decide this skipped day, else None.

    Decisive means: had the transient failures (and any layer-4 agents not run because of
    the early skip) succeeded, the day could have reached min_valid_agents. A day that
    falls short through content failures alone is an ordinary skip."""
    if day.skipped_reason is None:
        return None
    res = day.results
    transient = [n for n, r in res.items() if r.failure is not None and r.failure.reason in TRANSIENT]
    if not transient:
        return None
    valid = sum(1 for r in res.values() if r.status == "ok")
    not_run = sum(1 for s in pipeline.specs.values() if s.kind != "control" and s.name not in res)
    if valid + len(transient) + not_run < pipeline.min_valid_agents:
        return None
    return {n: (r.failure.reason.value if r.failure is not None else "no_output")
            for n, r in sorted(res.items()) if r.status != "ok"}


def _filter(snap: Snapshot, tickers: set[str]) -> Snapshot:
    p = snap.prices
    return Snapshot(asof=snap.asof, prices=p[p["ticker"].isin(tickers)].reset_index(drop=True),
                    macro=snap.macro)


def _closes_on(snap: Snapshot, day: date) -> dict[str, float]:
    p = snap.prices
    out = {}
    for t, c in zip(p.loc[p["date"] == day, "ticker"], p.loc[p["date"] == day, "close"]):
        c = float(c)
        if math.isfinite(c) and c > 0:
            out[str(t)] = c
    return out


# ---------------------------------------------------------------------------- the run

def run_backtest(cfg: BacktestConfig, store: PriceStore, pipeline: Pipeline, broker, risk: RiskLayer,
                 scores: ScoreStore, weightbook: WeightBook, predlog: PredictionLog,
                 checkpoint: Checkpoint, *, confirm: bool = False, llm: CachedLLM | None = None,
                 initial_cash: float = 100_000.0, universe_sectors: dict[str, str] | None = None,
                 lookback_days: int = 60, macro_series=(), run_dir: Path | None = None
                 ) -> BacktestResult:
    """Run (or resume) the backtest over train, validation and holdout in date order.

    One continuous portfolio spans all three periods; use ``cfg.periods.period_of`` to split
    the result per period. ``llm`` defaults to ``pipeline.llm`` (its spend is restored from
    the checkpoint). ``run_dir`` holds the dry-run stub cache (a temp dir if omitted).
    Raises CostConfirmationRequired, RunPaused, ResumeRefused (see module docstring).
    """
    s = cfg.settings
    universe = list(dict.fromkeys(s.universe))
    snap_tickers = universe + ([BENCHMARK] if BENCHMARK not in universe else [])
    period_days = {n: store.trading_days(lo, hi) for n, (lo, hi) in cfg.periods.items()}
    for n, ds in period_days.items():
        if not ds:
            raise ValueError(f"{n} period {getattr(cfg.periods, n)} contains no trading days")
    all_days = sorted({d for ds in period_days.values() for d in ds})
    fingerprint = {"periods": [[dump_date(a), dump_date(b)] for _, (a, b) in cfg.periods.items()],
                   "anonymize": cfg.anonymize, "dry_run": cfg.dry_run,
                   "initial_cash": float(initial_cash), "universe": sorted(universe)}

    # ---- resume validation (before any spend)
    raw = checkpoint.load()
    if raw is not None:
        st = _State.from_json(raw)
        diff = sorted(k for k in fingerprint if fingerprint[k] != st.fingerprint.get(k))
        if diff:
            raise ResumeRefused(f"checkpoint was written by a different configuration ({', '.join(diff)}); "
                                "refusing to resume")
    else:
        st = _State(fingerprint=fingerprint, cash=float(initial_cash), positions={},
                    peak_equity=float(initial_cash), prev_equity=float(initial_cash),
                    weights=weightbook.load())
    _check_predlog(predlog, st.predlog_head)
    todo = [d for d in all_days if st.last_completed_day is None or d > st.last_completed_day]

    agent_specs = [sp for sp in pipeline.specs.values() if sp.kind != "control"]
    est = estimate_cost(cfg, len(todo), len(agent_specs))
    if est > s.confirm_cost_usd and not confirm:
        raise CostConfirmationRequired(est, s.confirm_cost_usd)

    with ExitStack() as stack:
        # ---- the pipeline as run: dry-run stub LLM and/or anonymised universe
        active = copy.copy(pipeline)
        active_llm = llm if llm is not None else pipeline.llm
        if cfg.dry_run:
            d = Path(run_dir) if run_dir is not None else Path(stack.enter_context(tempfile.TemporaryDirectory()))
            active_llm = CachedLLM(StubClient(lambda _s, _u: DRY_RUN_RESPONSE), d / "dry_run_llm.db",
                                   "dry-run", (0.0, 0.0), math.inf)
            active.llm = active_llm
        limitations = list(LIMITATIONS)
        sectors: dict[str, str] = {}
        fwd: dict[str, str] = {}
        if cfg.anonymize:
            sectors = {t: str((universe_sectors or {}).get(t, "unknown")) for t in universe}
            if "unknown" in sectors.values():
                limitations.append("Anonymised run: some tickers had no sector label ('unknown').")
            fwd = _opaque_ids(sectors)
            active.universe = set(fwd.values())
            active.specs = {n: sp.model_copy(update={"benchmark": fwd.get(sp.benchmark, "unavailable")})
                            if sp.benchmark else sp for n, sp in pipeline.specs.items()}

        if raw is not None:
            active_llm.restore_spent(st.spent_usd)
            weightbook.save(st.weights)  # undo any update made by an uncommitted day
            if st.halted:
                log.warning("run is HALTED (%s): no orders until clear_halt()", st.halt_reason)

        portfolio = Portfolio(cash=st.cash, positions=dict(st.positions))
        cost_frac = 2 * s.slippage_bps / 1e4
        weighted = [sp.name for sp in agent_specs if sp.kind in WEIGHTED_KINDS]
        day_results: dict[date, DayResult] = {}
        committed = copy.deepcopy(st)

        def pause(reason: str, day: date, failed: dict[str, str], kind: str) -> RunPaused:
            committed.spent_usd = active_llm.spent_usd
            checkpoint.save(committed.to_json())
            return RunPaused(reason, day, failed, kind)

        for T in todo:
            st = copy.deepcopy(committed)
            portfolio = Portfolio(cash=st.cash, positions=dict(st.positions))
            i = all_days.index(T)
            snap = store.snapshot(T, snap_tickers, lookback_days, macro_series)
            today = _closes_on(snap, T)
            st.last_known.update(today)
            marks = dict(st.last_known)
            missing = sorted(t for t in portfolio.positions if t not in marks)
            if missing:
                raise RuntimeError(f"held tickers {missing} have never been priced as of {T}")
            equity_T = portfolio.equity(marks)

            # ---- (1-2) snapshot -> pipeline
            pipe_snap = _filter(snap, set(universe))
            weights = weightbook.load()
            agent_state = {"cash": portfolio.cash, "positions": dict(portfolio.positions), "equity": equity_T}
            label, back = None, {}
            if cfg.anonymize:
                pipe_in, back = anonymize_snapshot(pipe_snap, sectors)
                agent_state["positions"] = {fwd[t]: n for t, n in portfolio.positions.items()}
                label = f"day {i}"
            else:
                pipe_in = pipe_snap
            try:
                day = active.run_day(T, pipe_in, weights, agent_state, asof_label=label)
            except BudgetExceeded as e:
                raise pause(f"LLM budget exceeded: {e}", T, {}, "budget") from e
            failed = _pause_failed(active, day)
            if failed is not None:
                raise pause(f"transient infrastructure failures decided the day ({day.skipped_reason})",
                            T, failed, "transient")
            results = _demap_results(day.results, back) if cfg.anonymize else day.results

            # ---- (3-4) orders -> risk layer -> broker
            proposed: list[dict] = []
            if day.skipped_reason is None:
                if cfg.anonymize:
                    ex = next((results[sp.name] for sp in agent_specs if sp.kind == "execution"), None)
                    targets = {p.ticker: (p.conviction if p.direction == "long" else 0.0)
                               for p in ex.output.picks}
                    proposed = orders_from_targets(targets, dict(portfolio.positions), equity_T,
                                                   _last_close(pipe_snap))
                else:
                    proposed = list(day.proposed_orders or [])
            else:
                st.skipped[T] = day.skipped_reason
            to_send: list[Order] = []
            if st.halted:
                if proposed:
                    log.info("%s: halted (%s); %d proposed orders suppressed", T, st.halt_reason, len(proposed))
            else:
                try:
                    orders = [Order(str(o["ticker"]), o["side"], o["shares"]) for o in proposed]
                    gap = [o for o in orders if o.ticker not in today]
                    if gap:
                        log.info("%s: no bar today for %s: excluded", T, sorted({o.ticker for o in gap}))
                    orders = [o for o in orders if o.ticker in today]
                    decision = risk.apply(orders, dict(portfolio.positions), portfolio.cash, marks,
                                          st.prev_equity, st.peak_equity)
                except Exception as e:  # noqa: BLE001 - fail closed on ANY risk-layer error
                    log.exception("%s: risk layer failed; no orders sent", T)
                    st.risk_errors[T] = f"{type(e).__name__}: {e}"
                else:
                    if decision.halted:
                        st.halted, st.halt_reason = True, decision.halt_reason
                        log.error("%s: HALT: %s", T, decision.halt_reason)
                    for o, why in decision.rejected:
                        log.info("%s: risk rejected %s: %s", T, o, why)
                    to_send = list(decision.orders)
            if to_send:
                fills, dropped = broker.submit(to_send, T, portfolio=portfolio)
                for o, why in dropped:
                    log.info("%s: broker dropped %s: %s", T, o, why)
                for f in fills:
                    _apply_fill(portfolio, f, st.last_known)

            # ---- (5) log before scoring, then score matured horizons (data <= T only)
            predlog.append(T, results)
            for n in sorted(results):
                scores.record_picks(results[n])
            scores.score_matured(T, store, list(pipeline.specs.values()), cost_frac)

            # ---- (6) Darwinian weights
            stats = collect_stats(scores, weighted, s.rolling_ic_window, horizon=1)
            new_w = update_weights(weights, stats, s.dead_zone, s.shrinkage_k, s.warmup_picks)
            weightbook.save(new_w)

            # ---- (7) equity at T's close before T's own fills; benchmarks
            if not st.bench_ref:
                st.bench_ref = {t: marks[t] for t in snap_tickers if t in marks}
            st.equity[T] = equity_T
            st.spy[T] = (initial_cash * marks[BENCHMARK] / st.bench_ref[BENCHMARK]
                         if BENCHMARK in marks and BENCHMARK in st.bench_ref else float("nan"))
            rel = [marks[t] / st.bench_ref[t] for t in universe if t in st.bench_ref and t in marks]
            st.basket[T] = initial_cash * sum(rel) / len(rel) if rel else float("nan")
            st.peak_equity = max(st.peak_equity, equity_T)
            st.prev_equity = equity_T

            # ---- (8) commit point
            st.cash, st.positions = portfolio.cash, dict(portfolio.positions)
            st.weights = new_w
            st.spent_usd = active_llm.spent_usd
            st.predlog_head = predlog.head()
            st.last_completed_day = T
            day_results[T] = DayResult(T, day.regime, results,
                                       proposed if day.skipped_reason is None else None, day.skipped_reason)
            checkpoint.save(st.to_json())
            committed = st

        st = committed
        series = lambda m, name: pd.Series(  # noqa: E731
            [m[d] for d in sorted(m)], index=sorted(m), name=name, dtype=float)
        return BacktestResult(
            equity=series(st.equity, "equity"), spy=series(st.spy, "spy"), basket=series(st.basket, "basket"),
            day_results=day_results, skipped=dict(st.skipped), limitations=limitations,
            risk_errors=dict(st.risk_errors), halted=st.halted, halt_reason=st.halt_reason,
            final_portfolio=Portfolio(cash=st.cash, positions=dict(st.positions)),
            spent_usd=st.spent_usd, periods=cfg.periods)


def _apply_fill(portfolio: Portfolio, f: Fill, last_known: dict[str, float]) -> None:
    if not (math.isfinite(f.price) and f.price > 0):  # belt and braces: never book a NaN fill
        log.error("refusing fill at unusable price %r: %s", f.price, f)
        return
    portfolio.apply(f)
    last_known.setdefault(f.order.ticker, f.price)  # mark a never-closed ticker at its fill
