"""The ``atlas`` command line: backtest, calibrate, autoresearch, report.

Commands (``atlas <command> --help`` for options)::

    atlas backtest --config RUN.yaml [--dry-run] [--yes] [--with-control]
    atlas calibrate --config RUN.yaml
    atlas autoresearch --config RUN.yaml
    atlas report --run-dir RUN_DIR

The run config (documented by ``atlas/config/run.example.yaml``) names a run directory that
holds everything a run writes::

    checkpoint.json        runner commit point (resume state, equity, spend)
    prediction_log.jsonl   hash-chained agent outputs, appended before scoring
    scores.sqlite          picks and matured excess returns (ScoreStore)
    failures.sqlite        failed LLM attempts and dropped picks of the backtest
    llm_cache.sqlite       raw LLM response cache (dry runs use dry_run_llm.db)
    weights.json           Darwinian weights after the last completed day
    weights_history.jsonl  per day: weights and state given to the pipeline, the day's fills
    run_config.json        the resolved run config, settings, limits and agents (for report)
    calibration.json       null-test dead zone / keep margin (``atlas calibrate``)
    control.json           spend of the optional date-only control agent (--with-control)
    autoresearch_log.jsonl one line per autoresearch attempt
    autoresearch_failures.sqlite  failures of autoresearch replay calls (kept separate)
    prices/                price cache (unless data.cache_dir says otherwise)
    dry_run/               ALL state of `backtest --dry-run` (same layout): never the real run's
    report/                report.md, equity.png, agents.csv, autoresearch.csv

Keys: ANTHROPIC_API_KEY and (for ``data.macro.source: fred``) FRED_API_KEY are read from the
environment only. A missing key is a usage error (exit 2) raised before any work; key values
are never printed, logged or written to the run directory. ``--dry-run`` needs no keys and
makes zero LLM calls (the runner's built-in neutral stub) while exercising the full loop.

Two-pass workflow (rulings O/T): with no calibrated dead zone the backtest runs with FROZEN
weights (pass 1). ``atlas calibrate`` runs the null test on pass 1's train-period picks and
writes ``calibration.json``. Pass 2 is a new run directory whose config sets
``calibration: <pass-1 run_dir>/calibration.json``; it runs with adaptive weights. Resuming
pass 1 with a calibration is refused (the checkpoint's fingerprint changes).

Stored run = single source of truth: run_config.json holds the VALIDATED config of the run
(written on a fresh run only). A later `backtest` on the same run_dir that disagrees with it
(except ALLOWED_DRIFT: budget, confirmation ceiling, null-test/autoresearch options, cache
location) is refused with exit 1 and the file is left untouched; `calibrate` and
`autoresearch` refuse a disagreeing --config with exit 2 and use the stored periods and
settings; `report` reads only the stored record. Unknown config keys are rejected with a
did-you-mean hint. One LLM hard cap (budget_usd) spans the backtest, the control agent and
autoresearch of a run directory.

Anonymised runs are refused (ruling W): anonymised prompt variants do not exist yet and the
real prompt files name tickers and ask for dates.

Exit codes: 0 success; 1 operational failure (paused run, refused resume, dirty repo, too
little data, ...: the message says how to fix it); 2 usage/config/key errors. Expected errors
print a message, never a stack trace.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import logging
import math
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from atlas.backtest.checkpoint import Checkpoint, load_date
from atlas.backtest.nulltest import LeakageSuspected, NullDistribution, null_test
from atlas.backtest.predlog import PredictionLog
from atlas.backtest.runner import (
    BENCHMARK, DRY_RUN_RESPONSE, HISTORY_NAME, LIMITATIONS, WEIGHTED_KINDS, BacktestConfig,
    BacktestResult, CostConfirmationRequired, Periods, ResumeRefused, RunPaused, _filter,
    estimate_cost, read_day_history, run_backtest,
)
from atlas.config import AgentSpec, Limits, Settings, load_agents, load_limits
from atlas.data.csv_provider import CacheOnlyPrices, CsvPriceProvider, NoMacro
from atlas.data.fred_provider import FredFetchError
from atlas.data.store import PriceStore
from atlas.engine.failures import CALL_FAILURE, FailureLog
from atlas.engine.llm import BudgetExceeded, CachedLLM, LLMResponse, StubClient
from atlas.engine.pipeline import Pipeline
from atlas.portfolio.models import Portfolio
from atlas.portfolio.simbroker import SimBroker
from atlas.risk.layer import RiskLayer
from atlas.schemas import AgentOutput, AgentResult
from atlas.scoring.metrics import excess_return
from atlas.scoring.store import ScoreStore, non_overlapping
from atlas.scoring.weights import W_INIT, WeightBook

log = logging.getLogger("atlas.cli")

EXIT_OK, EXIT_FAILED, EXIT_USAGE = 0, 1, 2
PKG = Path(__file__).resolve().parent
DEFAULT_REPO = PKG.parent
ANTHROPIC_KEY, FRED_KEY = "ANTHROPIC_API_KEY", "FRED_API_KEY"

CHECKPOINT, PREDLOG, SCORES, FAILURES = "checkpoint.json", "prediction_log.jsonl", "scores.sqlite", "failures.sqlite"
LLM_CACHE, WEIGHTS, RUN_CONFIG, CALIBRATION = "llm_cache.sqlite", "weights.json", "run_config.json", "calibration.json"
CONTROL_STATE, AR_LOG, AR_FAILURES = "control.json", "autoresearch_log.jsonl", "autoresearch_failures.sqlite"
DRY_RUN_CACHE = "dry_run_llm.db"
ABLATION_NOTE = ("CIO ablation: the decision layer's portfolio is the execution agent's stated long "
                 "target weights (scaled down to max_gross_pct); the baseline is the weight-averaged "
                 "portfolio (baseline_target_weights) of the same day's valid layer 1-3 outputs and "
                 "Darwinian weights, scaled to the SAME gross exposure as that day's execution targets, "
                 "so only selection and weighting differ. Both are held one day, adjusted close to next "
                 "adjusted close, with no costs; skipped days are excluded. It is reported per period, "
                 "with the holdout as the headline; pre-cutoff periods are not evidence of skill.")


class UsageError(Exception):
    """Bad command line, config or missing key: exit 2."""


class OperationalError(Exception):
    """The command could not do its work (data missing, run not found, ...): exit 1."""


# ============================================================================ run config

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PeriodRange(_Strict):
    start: date
    end: date


class PeriodsConfig(_Strict):
    train: PeriodRange
    validation: PeriodRange
    holdout: PeriodRange


class PricesConfig(_Strict):
    source: Literal["yfinance", "csv", "ibkr"] = "yfinance"
    dir: Path | None = None  # csv: one <TICKER>.csv per ticker
    host: str = "127.0.0.1"  # ibkr: a running IB Gateway / TWS
    port: int = 4002
    client_id: int = 1


class MacroConfig(_Strict):
    source: Literal["fred", "none"] = "fred"
    series: list[str] = Field(default_factory=list)
    publication_lag_days: dict[str, int] = Field(default_factory=dict)


class DataConfig(_Strict):
    prices: PricesConfig = Field(default_factory=PricesConfig)
    macro: MacroConfig = Field(default_factory=MacroConfig)
    lookback_days: int = Field(60, gt=0)
    cache_dir: Path | None = None  # default <run_dir>/prices


class NullTestConfig(_Strict):
    n_shuffles: int = Field(200, gt=0)
    seed: int = 0


class AutoresearchConfig(_Strict):
    n_replay_days: int = 60


class RunConfig(_Strict):
    """A run: where it lives, its periods and what overrides the packaged config files."""
    run_dir: Path
    periods: PeriodsConfig
    initial_cash: float = Field(100_000.0, gt=0)
    anonymize: bool = False
    universe: list[str] | None = None  # overrides settings.universe
    budget_usd: float | None = Field(None, gt=0)  # overrides settings.budget_usd (hard cap)
    settings: dict = Field(default_factory=dict)  # any other Settings field overrides
    settings_file: Path | None = None
    limits_file: Path | None = None
    agents_file: Path | None = None
    calibration: Path | None = None  # calibration.json of a pass-1 run: sets dead_zone
    llm_cache: Path | None = None  # default <run_dir>/llm_cache.sqlite
    repo: Path | None = None  # repository holding atlas/prompts/ (default: this checkout)
    data: DataConfig = Field(default_factory=DataConfig)
    null_test: NullTestConfig = Field(default_factory=NullTestConfig)
    autoresearch: AutoresearchConfig = Field(default_factory=AutoresearchConfig)

    def path(self, name: str) -> Path:
        return self.run_dir / name

    @property
    def cache_dir(self) -> Path:
        return self.data.cache_dir or self.run_dir / "prices"

    @property
    def llm_cache_path(self) -> Path:
        return self.llm_cache or self.run_dir / LLM_CACHE

    @property
    def prompt_repo(self) -> Path:
        return self.repo or DEFAULT_REPO


def _errors(e: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['msg']}"
                     for err in e.errors())


_NORMAL_KEY = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _safe_key(key) -> str:
    """An unknown key as printable text: a long or key-like name (maybe a pasted secret) is
    shown as its first 12 characters + '…' only."""
    k = str(key)
    if len(k) > 24 or k.startswith("sk-") or not _NORMAL_KEY.fullmatch(k):
        return k[:12] + "…"
    return k


def _did_you_mean(key: str, valid) -> str:
    close = difflib.get_close_matches(str(key), sorted(valid), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""


def _unknown_keys(raw, model: type[BaseModel], prefix: str = "") -> list[str]:
    """Unknown keys anywhere in a run config mapping (strict sections and `settings:`),
    with a did-you-mean hint. A misspelt hard cap must never fall back to its default."""
    if not isinstance(raw, dict):
        return []
    out = []
    fields = model.model_fields
    for k, v in raw.items():
        name = f"{prefix}{k}"
        if k not in fields:
            out.append(f"{prefix}{_safe_key(k)}{_did_you_mean(k, fields)}")
            continue
        if model is RunConfig and k == "settings" and isinstance(v, dict):
            out += [f"settings.{_safe_key(sk)}{_did_you_mean(sk, Settings.model_fields)}"
                    for sk in v if sk not in Settings.model_fields]
            continue
        ann = fields[k].annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            out += _unknown_keys(v, ann, f"{name}.")
    return out


def load_run_config(path: str | Path) -> RunConfig:
    """Parse and validate a run config; relative paths are relative to the config file."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise UsageError(f"run config {path} not found") from None
    except (OSError, yaml.YAMLError) as e:
        raise UsageError(f"cannot read run config {path}: {type(e).__name__}") from None
    if not isinstance(raw, dict):
        raise UsageError(f"run config {path} must be a YAML mapping")
    unknown = _unknown_keys(raw, RunConfig)
    if unknown:
        raise UsageError(f"invalid run config {path}: unknown key(s): {', '.join(unknown)}")
    try:
        rc = RunConfig.model_validate(raw)
    except ValidationError as e:
        raise UsageError(f"invalid run config {path}: {_errors(e)}") from None
    base = path.resolve().parent
    rel = lambda p: None if p is None else (p if p.is_absolute() else base / p)  # noqa: E731
    upd = {k: rel(getattr(rc, k)) for k in ("settings_file", "limits_file", "agents_file",
                                             "calibration", "llm_cache", "repo")}
    upd["run_dir"] = rel(rc.run_dir)
    data = rc.data.model_copy(update={
        "cache_dir": rel(rc.data.cache_dir),
        "prices": rc.data.prices.model_copy(update={"dir": rel(rc.data.prices.dir)})})
    return rc.model_copy(update={**upd, "data": data})


def read_calibration(path: Path) -> dict:
    try:
        cal = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise UsageError(f"calibration file {path} not found (run `atlas calibrate` on the pass-1 run)") from None
    except (OSError, ValueError) as e:
        raise UsageError(f"cannot read calibration file {path}: {type(e).__name__}") from None
    if not isinstance(cal, dict) or not isinstance(cal.get("dead_zone"), (int, float)):
        raise UsageError(f"calibration file {path} has no numeric dead_zone")
    return cal


def resolve_settings(rc: RunConfig) -> Settings:
    path = rc.settings_file or PKG / "config" / "settings.yaml"
    try:
        base = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as e:
        raise UsageError(f"cannot read settings file {path}: {type(e).__name__}") from None
    merged = {**base, **rc.settings}
    if rc.universe is not None:
        merged["universe"] = list(rc.universe)
    if rc.budget_usd is not None:
        merged["budget_usd"] = rc.budget_usd
    if rc.calibration is not None:
        if "dead_zone" in rc.settings:
            raise UsageError("set either settings.dead_zone or calibration in the run config, not both")
        merged["dead_zone"] = float(read_calibration(rc.calibration)["dead_zone"])
    try:
        return Settings(**merged)
    except ValidationError as e:
        raise UsageError(f"invalid settings: {_errors(e)}") from None


def resolve_periods(rc: RunConfig) -> Periods:
    p = rc.periods
    try:
        return Periods(train=(p.train.start, p.train.end), validation=(p.validation.start, p.validation.end),
                       holdout=(p.holdout.start, p.holdout.end))
    except ValueError as e:
        raise UsageError(f"invalid periods: {e}") from None


def make_backtest_config(periods: Periods, settings: Settings, limits: Limits, anonymize: bool,
                         dry_run: bool) -> BacktestConfig:
    try:
        return BacktestConfig(periods=periods, settings=settings, limits=limits, anonymize=anonymize,
                              dry_run=dry_run)
    except ValueError as e:
        raise UsageError(
            f"{e}. Fix: the holdout must lie entirely after the model's training cutoff; move "
            "periods.holdout.start after settings.model_cutoff, or set settings.model_cutoff in the "
            "run config to the documented cutoff of the model you use") from None


def refuse_anonymize(rc: RunConfig, command: str) -> None:
    if rc.anonymize:
        raise UsageError(
            f"`atlas {command}` refuses anonymize: true. Anonymised prompt variants do not exist yet: "
            "the real prompt files name tickers and ask the model to echo dates, so an anonymised "
            "run would not be anonymous. Set anonymize: false.")


def require_keys(rc: RunConfig, *, anthropic: bool, fred: bool) -> str | None:
    """Check the environment for the keys this command needs; returns the FRED key (or None)
    for the provider. Never prints a value."""
    names = []
    if anthropic:
        names.append(ANTHROPIC_KEY)
    if fred and rc.data.macro.source == "fred":
        names.append(FRED_KEY)
    missing = [n for n in names if not os.environ.get(n, "").strip()]
    if missing:
        raise UsageError(
            f"missing environment variable(s): {', '.join(missing)}. Export them in the shell that "
            "runs atlas (keys are read from the environment only, never from config files)"
            + (", or use --dry-run, which needs no keys" if ANTHROPIC_KEY in missing else "") + ".")
    return os.environ.get(FRED_KEY) if rc.data.macro.source == "fred" else None


# ============================================================================ components

class _NoCalls:
    """LLM client for dry runs: the runner swaps in its stub, so any call here is a bug."""

    def complete(self, system: str, user: str, max_tokens: int) -> LLMResponse:
        raise RuntimeError("dry run attempted a real LLM call")


@dataclass
class Run:
    rc: RunConfig
    settings: Settings
    limits: Limits
    specs: list[AgentSpec]
    prompts: dict[str, str]
    periods: Periods
    store: PriceStore
    macro_series: list[str]
    llm: CachedLLM | None = None
    failures: FailureLog | None = None
    pipeline: Pipeline | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def universe(self) -> list[str]:
        return list(dict.fromkeys(self.settings.universe))

    def trading_days(self) -> list[date]:
        days = []
        for name, (lo, hi) in self.periods.items():
            ds = self.store.trading_days(lo, hi)
            if not ds:
                raise UsageError(f"{name} period {lo}..{hi} contains no trading days in the price data")
            days += ds
        return sorted(set(days))

    @property
    def cost_frac(self) -> float:
        return 2 * self.settings.slippage_bps / 1e4


def load_specs(rc: RunConfig) -> list[AgentSpec]:
    path = rc.agents_file or PKG / "config" / "agents.yaml"
    try:
        return load_agents(path)
    except (OSError, yaml.YAMLError, KeyError, TypeError, ValidationError) as e:
        raise UsageError(f"cannot load agents file {path}: {type(e).__name__}") from None


def load_prompts(rc: RunConfig, specs: list[AgentSpec]) -> dict[str, str]:
    out = {}
    for sp in specs:
        p = rc.prompt_repo / "atlas" / sp.prompt_file
        try:
            out[sp.name] = p.read_text(encoding="utf-8")
        except OSError:
            raise UsageError(f"prompt file for agent {sp.name} not found: {p}") from None
    return out


def _price_provider(rc: RunConfig, offline: bool):
    if offline:
        return CacheOnlyPrices()
    src = rc.data.prices
    if src.source == "csv":
        if src.dir is None:
            raise UsageError("data.prices.source csv needs data.prices.dir")
        try:
            return CsvPriceProvider(src.dir)
        except FileNotFoundError as e:
            raise UsageError(str(e)) from None
    if src.source == "ibkr":
        from atlas.data.ibkr_provider import IBKRProvider
        return IBKRProvider(src.host, src.port, src.client_id)
    from atlas.data.yf_provider import YFinanceProvider
    return YFinanceProvider()


def build_run(rc: RunConfig, *, offline: bool = False, fred_key: str | None = None,
              llm_client=None, dry_run: bool = False, failures_name: str = FAILURES,
              settings: Settings | None = None) -> Run:
    """Assemble the components the commands share. `offline` serves prices from the cache
    only and no macro data (report/calibrate)."""
    settings = settings or resolve_settings(rc)
    lim_path = rc.limits_file or PKG / "config" / "limits.yaml"
    try:
        limits = load_limits(lim_path)
    except (OSError, yaml.YAMLError, TypeError, ValidationError) as e:
        raise UsageError(f"cannot load limits file {lim_path}: {type(e).__name__}") from None
    specs = load_specs(rc)
    prompts = load_prompts(rc, specs)
    periods = resolve_periods(rc)
    notes = []
    macro, series = NoMacro(), []
    if not offline and rc.data.macro.source == "fred":
        if fred_key:
            from atlas.data.fred_provider import FredAlfredProvider
            macro, series = FredAlfredProvider(fred_key, rc.data.macro.publication_lag_days), list(rc.data.macro.series)
        else:
            notes.append(f"dry run: {FRED_KEY} is not set, so the run uses no macro series")
    store = PriceStore(_price_provider(rc, offline), macro, rc.cache_dir)
    run = Run(rc, settings, limits, specs, prompts, periods, store, series, notes=notes)
    if llm_client is not None or not offline:
        client = llm_client
        if client is None:
            if dry_run:
                client = _NoCalls()
            else:
                from atlas.engine.llm import AnthropicClient
                client = AnthropicClient(settings.model)
        run.llm = CachedLLM(client, rc.llm_cache_path, settings.model,
                            (settings.usd_per_mtok_input, settings.usd_per_mtok_output), settings.budget_usd)
        run.failures = FailureLog(rc.path(failures_name))
        run.pipeline = Pipeline(specs, prompts, run.llm, run.failures, run.universe, settings.min_valid_agents)
    return run


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _load_checkpoint(checkpoint: Checkpoint) -> dict | None:
    try:
        return checkpoint.load()
    except ValueError:
        raise OperationalError(f"checkpoint is corrupt: {checkpoint.path}") from None


def _last_completed(checkpoint: Checkpoint) -> date | None:
    raw = _load_checkpoint(checkpoint)
    return load_date(raw["last_completed_day"]) if raw and raw.get("last_completed_day") else None


# ============================================================================ stored run record

# Keys of the stored record that may legitimately change between invocations on one run
# directory (prefix match on the flattened key): where the directory is, how the null test /
# autoresearch are run, the response-cache location, the spend cap and confirmation ceiling.
ALLOWED_DRIFT = ("run_config.run_dir", "run_config.null_test", "run_config.autoresearch",
                 "run_config.llm_cache", "settings.budget_usd", "settings.confirm_cost_usd", "macro_series")


def run_record(rc: RunConfig, run: Run) -> dict:
    """What run_config.json stores: the VALIDATED config (the raw `settings:` dict, universe and
    budget overrides are folded into the validated Settings dump, never copied verbatim)."""
    return {"run_config": rc.model_dump(mode="json", exclude={"settings", "universe", "budget_usd"}),
            "settings": run.settings.model_dump(mode="json"),
            "limits": run.limits.model_dump(mode="json"),
            "agents": [sp.model_dump(mode="json") for sp in run.specs],
            "macro_series": list(run.macro_series)}


def _flat(obj, prefix: str = "") -> dict:
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}{k}."))
        return out or {prefix.rstrip("."): {}}
    if isinstance(obj, list) and obj and all(isinstance(x, dict) and "name" in x for x in obj):
        return _flat({x["name"]: x for x in obj}, prefix)  # agents: diff by name
    return {prefix.rstrip("."): obj}


# raw override fields of records written before the validated-only format; their effect is
# compared through the validated "settings" dump
LEGACY_RAW = ("run_config.settings", "run_config.universe", "run_config.budget_usd")


def record_diff(stored: dict, current: dict) -> list[str]:
    a, b = _flat(stored), _flat(current)
    skip = ALLOWED_DRIFT + LEGACY_RAW
    return sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")
                  and not any(k == p or k.startswith(p + ".") for p in skip))


def read_stored(run_dir: Path) -> dict | None:
    p = Path(run_dir) / RUN_CONFIG
    if not p.exists():
        return None
    try:
        rec = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise OperationalError(f"stored run config is corrupt: {p}") from None
    if not isinstance(rec, dict) or not {"run_config", "settings", "limits", "agents"} <= set(rec):
        raise OperationalError(f"stored run config is corrupt: {p}")
    return rec


def stored_run(rc: RunConfig, run: Run, command: str) -> dict:
    """Make the stored run the single source of truth for `command`: the supplied config must
    agree with run_config.json on everything but ALLOWED_DRIFT (exit 2 otherwise); periods and
    settings are then taken from the stored record."""
    stored = read_stored(rc.run_dir)
    if stored is None:
        raise OperationalError(f"no stored run in {rc.run_dir} (no {RUN_CONFIG}): run "
                               f"`atlas backtest --config ...` first")
    diff = record_diff(stored, run_record(rc, run))
    if diff:
        raise UsageError(f"`atlas {command}`: the config disagrees with the stored run in {rc.run_dir} on "
                         f"{', '.join(diff)}; use the config the run was made with (its record is "
                         f"{rc.path(RUN_CONFIG)})")
    src = RunConfig.model_validate(stored["run_config"]).model_copy(update={"run_dir": rc.run_dir})
    run.periods = resolve_periods(src)
    run.settings = Settings(**{**stored["settings"], "budget_usd": run.settings.budget_usd,
                               "confirm_cost_usd": run.settings.confirm_cost_usd})
    return stored


_WARNED_TORN: set[str] = set()


def read_ar_log(path: Path) -> list[dict]:
    """autoresearch_log.jsonl records. A malformed FINAL line (a crash mid-write) is ignored
    with one warning; a malformed earlier line is corruption: OperationalError."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    last = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
            if not isinstance(rec, dict):
                raise ValueError("not an object")
        except ValueError:
            if i == last:
                if str(path) not in _WARNED_TORN:
                    _WARNED_TORN.add(str(path))
                    print(f"warning: ignoring a truncated final line {i + 1} of {path} (crash mid-write)",
                          file=sys.stderr)
                continue
            raise OperationalError(f"autoresearch log is corrupt at line {i + 1}: {path}") from None
        out.append(rec)
    return out


def _drop_torn_tail(path: Path) -> None:
    """Before appending to the autoresearch log: set a truncated final line aside (to
    <log>.torn) so the next record does not glue onto it and corrupt the log."""
    path = Path(path)
    if not path.exists():
        return
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return
    try:
        json.loads(lines[-1])
        if not lines[-1].endswith("\n"):
            with open(path, "a", encoding="utf-8") as f:
                f.write("\n")
        return
    except ValueError:
        pass
    with open(path.with_name(path.name + ".torn"), "a", encoding="utf-8") as f:
        f.write(lines[-1].rstrip("\n") + "\n")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(lines[:-1]), encoding="utf-8")
    os.replace(tmp, path)
    print(f"note: moved a truncated final line of {path} to {path.name}.torn", file=sys.stderr)


def spend_so_far(run_dir: Path) -> dict[str, float]:
    """LLM spend recorded in a run directory: backtest (checkpoint), control, autoresearch."""
    run_dir = Path(run_dir)
    raw = _load_checkpoint(Checkpoint(run_dir / CHECKPOINT)) or {}
    ctl = run_dir / CONTROL_STATE
    ar_cost = sum((float(r.get("cost_usd") or 0.0) for r in read_ar_log(run_dir / AR_LOG)), 0.0)
    return {"backtest": float(raw.get("spent_usd", 0.0)),
            "control": float(json.loads(ctl.read_text())["spent_usd"]) if ctl.exists() else 0.0,
            "autoresearch": ar_cost}


# ============================================================================ backtest

def _confirm(estimate: float, ceiling: float, yes: bool) -> bool:
    if yes:
        return True
    msg = (f"estimated LLM cost ${estimate:.2f} exceeds the confirmation ceiling ${ceiling:.2f} "
           "(settings.confirm_cost_usd)")
    if not sys.stdin.isatty():
        print(f"{msg}. Not running: stdin is not a terminal, re-run with --yes to proceed.", file=sys.stderr)
        return False
    ans = input(f"{msg}. Proceed? [y/N] ")
    if ans.strip().lower() in ("y", "yes"):
        return True
    print("Not confirmed; nothing was run. Re-run with --yes to skip this question.", file=sys.stderr)
    return False


DRY_RUN_DIR = "dry_run"


def cmd_backtest(args, llm_client=None) -> int:
    rc = load_run_config(args.config)
    refuse_anonymize(rc, "backtest")
    fred_key = require_keys(rc, anthropic=not args.dry_run, fred=not args.dry_run)
    if args.dry_run:
        fred_key = os.environ.get(FRED_KEY) or None
        # a dry run keeps ALL its state in <run_dir>/dry_run/: it never touches the real run
        rc = rc.model_copy(update={"run_dir": rc.run_dir / DRY_RUN_DIR, "llm_cache": None})
    run = build_run(rc, fred_key=fred_key, llm_client=llm_client, dry_run=args.dry_run)
    cfg = make_backtest_config(run.periods, run.settings, run.limits, False, args.dry_run)
    control = [sp for sp in run.specs if sp.kind == "control"]
    if args.with_control and not control:
        raise UsageError("--with-control needs a control agent (kind: control) in the agents file")
    for n in run.notes:
        print(f"note: {n}")
    if args.dry_run:
        print(f"dry run: zero LLM calls (built-in neutral stub); no keys needed; state in {rc.run_dir}")
    checkpoint = Checkpoint(rc.path(CHECKPOINT))
    lcd = _last_completed(checkpoint)
    stored = read_stored(rc.run_dir)
    record = run_record(rc, run)
    if stored is not None:
        diff = record_diff(stored, record)
        if diff:
            raise OperationalError(
                f"{rc.run_dir} holds a run made with a different configuration ({', '.join(diff)}); "
                f"refusing to run. Restore the original config (recorded in {rc.path(RUN_CONFIG)}) to "
                "resume, or use a new run_dir for the new configuration.")
    fresh = stored is None and not checkpoint.path.exists()
    if not args.dry_run:  # one hard cap: control and autoresearch spend count against the backtest too
        other = spend_so_far(rc.run_dir)
        run.llm.budget_usd = max(0.0, run.settings.budget_usd - other["control"] - other["autoresearch"])
        if sum(other.values()) >= run.settings.budget_usd and (lcd is None or lcd < max(run.trading_days())):
            raise OperationalError(
                f"LLM budget exhausted: ${sum(other.values()):.2f} of ${run.settings.budget_usd:.2f} already "
                "spent on this run (backtest + control + autoresearch); raise budget_usd to continue")
    all_days = run.trading_days()
    todo = [d for d in all_days if lcd is None or d > lcd]
    n_agents = sum(1 for sp in run.specs if sp.kind != "control") + (1 if args.with_control else 0)
    est = estimate_cost(cfg, len(todo), n_agents)
    confirmed = args.yes
    if est > run.settings.confirm_cost_usd:
        if not _confirm(est, run.settings.confirm_cost_usd, args.yes):
            return EXIT_USAGE
        confirmed = True
    if lcd is not None:
        print(f"resuming after {lcd} ({len(todo)} trading days left)")
    if fresh:  # a fresh run: nothing to disagree with
        _write_json(rc.path(RUN_CONFIG), record)
        stored = record
    broker = SimBroker(run.store, run.settings.commission_per_share, run.settings.commission_min,
                       run.settings.slippage_bps)
    risk = RiskLayer(run.limits, run.settings.commission_per_share, run.settings.commission_min)
    scores = ScoreStore(rc.path(SCORES))
    try:
        result = run_backtest(cfg, run.store, run.pipeline, broker, risk, scores, WeightBook(rc.path(WEIGHTS)),
                              PredictionLog(rc.path(PREDLOG)), checkpoint, confirm=confirmed,
                              initial_cash=rc.initial_cash, universe_sectors=None,
                              lookback_days=rc.data.lookback_days, macro_series=run.macro_series,
                              run_dir=rc.run_dir, history_path=rc.path(HISTORY_NAME))
    except RunPaused as e:
        if stored != record:  # the runner accepted the checkpoint: record the (allowed) changes
            _write_json(rc.path(RUN_CONFIG), record)
        print(f"{e}\nThe checkpoint is intact (it holds the last completed day); fix the cause if "
              "needed and re-run the same command to resume.", file=sys.stderr)
        if e.failed:
            print("failed agents: " + ", ".join(f"{a} ({r})" for a, r in e.failed.items()), file=sys.stderr)
        return EXIT_FAILED
    except CostConfirmationRequired as e:  # belt and braces: the estimate is checked above
        print(f"{e} (use --yes)", file=sys.stderr)
        return EXIT_USAGE
    if stored != record:  # resumed with only ALLOWED_DRIFT changes, or a pre-record checkpoint
        _write_json(rc.path(RUN_CONFIG), record)
    spent = result.spent_usd
    if args.with_control:
        try:
            spent += run_control_pass(run, scores, result, all_days, dry_run=args.dry_run)
        except BudgetExceeded as e:
            print(f"control agent stopped: LLM budget exceeded ({e}). The backtest itself is complete; "
                  "raise budget_usd and re-run the same command to finish the control agent "
                  "(completed days are cache hits).", file=sys.stderr)
            return EXIT_FAILED
    eq = result.equity
    print(f"backtest complete: {len(eq)} trading days {eq.index[0]}..{eq.index[-1]}; final equity "
          f"{eq.iloc[-1]:,.2f} (start {rc.initial_cash:,.2f}); {len(result.skipped)} skipped days; "
          f"LLM spend ${spent:.2f}" + ("; HALTED: " + str(result.halt_reason) if result.halted else ""))
    print(f"next: atlas report --run-dir {rc.run_dir}")
    return EXIT_OK


def run_control_pass(run: Run, scores: ScoreStore, result: BacktestResult, all_days: list[date], *,
                     dry_run: bool) -> float:
    """The date-only control agent on every completed day, recorded under agent 'control'
    in the score store. It runs after the trading loop and only adds its own rows: the
    portfolio, prediction log, weights (control is never a weighted agent) and checkpoint
    are untouched. Idempotent: re-running replays the LLM cache. Returns its total spend."""
    from atlas.backtest.control import run_control_agent
    spec = next(sp for sp in run.specs if sp.kind == "control")
    state_path = run.rc.path(CONTROL_STATE)
    prev = json.loads(state_path.read_text())["spent_usd"] if state_path.exists() else 0.0
    if dry_run:
        llm = CachedLLM(StubClient(lambda _s, _u: DRY_RUN_RESPONSE), run.rc.path(DRY_RUN_CACHE), "dry-run",
                        (0.0, 0.0), math.inf)
    else:
        other = spend_so_far(run.rc.run_dir)  # one hard cap across backtest, control, autoresearch
        remaining = max(0.0, run.settings.budget_usd - result.spent_usd - prev - other["autoresearch"])
        llm = CachedLLM(run.llm.client, run.rc.llm_cache_path, run.settings.model,
                        (run.settings.usd_per_mtok_input, run.settings.usd_per_mtok_output), remaining)
    last = max(result.equity.index)
    days = [d for d in all_days if d <= last]
    done = 0
    try:
        for d in days:
            scores.record_picks(run_control_agent(d, run.universe, run.prompts[spec.name], llm,
                                                  run.failures, spec=spec))
            done += 1
    finally:
        _write_json(state_path, {"spent_usd": prev + llm.spent_usd, "days": done})
    scores.score_matured(last, run.store, run.specs, run.cost_frac)
    return prev + llm.spent_usd


# ============================================================================ null / calibrate

def train_null(scores: ScoreStore, run: Run, all_days: list[date]) -> tuple[NullDistribution, date]:
    """Null test on the non-overlapping 1-day TRAIN picks of the weighted agents (ruling T):
    dated on or before the last train day whose scoring exit is still inside train, with
    window=rolling_ic_window. Raises ValueError (too few agents) or LeakageSuspected."""
    from atlas.autoresearch.loop import last_safe_train_day
    safe_end = last_safe_train_day(all_days, run.periods, 1)
    if safe_end is None:
        raise ValueError("no train day has its 1-day scoring exit inside the train period")
    weighted = {sp.name for sp in run.specs if sp.kind in WEIGHTED_KINDS}
    df = scores.picks_frame(None, 1)
    df = df[df["agent"].isin(weighted) & (df["date"] >= run.periods.train[0]) & (df["date"] <= safe_end)]
    s = run.settings
    nt = run.rc.null_test
    null = null_test(non_overlapping(df, 1), nt.n_shuffles, nt.seed, k=s.shrinkage_k,
                     warmup_picks=s.warmup_picks, window=s.rolling_ic_window)
    return null, safe_end


def _cached_days(run: Run) -> list[date]:
    try:
        return run.trading_days()
    except UsageError as e:
        raise OperationalError(f"{e}; has the backtest been run for this config?") from None


def cmd_calibrate(args, llm_client=None) -> int:
    rc = load_run_config(args.config)
    run = build_run(rc, offline=True)
    stored_run(rc, run, "calibrate")  # periods and settings of the STORED run (ruling T)
    if not rc.path(SCORES).exists():
        raise OperationalError(f"no stored run in {rc.run_dir}: run `atlas backtest --config {args.config}` first")
    _last_completed(Checkpoint(rc.path(CHECKPOINT)))  # a corrupt checkpoint fails clearly
    scores = ScoreStore(rc.path(SCORES))
    try:
        null, safe_end = train_null(scores, run, _cached_days(run))
    except LeakageSuspected as e:
        raise OperationalError(f"cannot calibrate: the null test suspects leakage ({e}); do not trust "
                               "this run's scores") from None
    except ValueError as e:
        raise OperationalError(f"cannot calibrate: {e}. Run a longer train period.") from None
    s = run.settings
    cal = {"dead_zone": null.dead_zone(), "keep_margin": null.keep_margin(),
           "n_shuffles": rc.null_test.n_shuffles, "seed": rc.null_test.seed, "window": s.rolling_ic_window,
           "warmup_picks": s.warmup_picks, "shrinkage_k": s.shrinkage_k,
           "train_max_date": safe_end.isoformat(), "mean_ic": float(np.mean(null.ics)),
           "ics": null.ics.tolist(), "deviations": null.deviations.tolist(), "pair_diffs": null.pair_diffs.tolist()}
    _write_json(rc.path(CALIBRATION), cal)
    print(f"calibrated on train picks up to {safe_end}: dead_zone {cal['dead_zone']:.4f}, keep_margin "
          f"{cal['keep_margin']:.4f} (null mean IC {cal['mean_ic']:+.4f}, {len(null.ics)} shuffled ICs)")
    print(f"wrote {rc.path(CALIBRATION)}. For pass 2 use a NEW run_dir with `calibration: "
          f"{rc.path(CALIBRATION)}` in its config.")
    return EXIT_OK


NULL_META_KEYS = ("seed", "n_shuffles", "window", "train_max_date")


def null_test_meta(rc: RunConfig, settings: Settings, safe_end: date | None) -> dict:
    return {"seed": rc.null_test.seed, "n_shuffles": rc.null_test.n_shuffles,
            "window": settings.rolling_ic_window,
            "train_max_date": safe_end.isoformat() if safe_end else None}


def null_from_calibration(cal: dict) -> NullDistribution:
    return NullDistribution(np.asarray(cal["ics"], float), np.asarray(cal["deviations"], float),
                            np.asarray(cal["pair_diffs"], float))


# ============================================================================ replay (autoresearch)

def predlog_results(path: Path) -> dict[date, dict[str, AgentResult]]:
    """day -> agent -> AgentResult as logged (failed agents: output None)."""
    pl = PredictionLog(path)
    if not pl.verify():
        raise OperationalError(f"prediction log {path} failed hash-chain verification")
    out: dict[date, dict[str, AgentResult]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        d = date.fromisoformat(rec["day"])
        day = {}
        for a in rec["agents"]:
            out_ = AgentOutput.model_validate(a["output"]) if a["status"] == "ok" else None
            day[a["agent"]] = AgentResult(agent=a["agent"], date=d, output=out_)
        out[d] = day
    return out


@dataclass
class Replay:
    """Re-runs single agents on recorded days exactly as the backtest called them."""
    run: Run
    pipeline: Pipeline
    all_days: list[date]
    history: dict[date, dict]
    results: dict[date, dict[str, AgentResult]]
    _snaps: dict = field(default_factory=dict)

    @property
    def prompts(self) -> dict[str, str]:
        return self.run.prompts

    def context(self, day: date):
        from atlas.autoresearch.loop import DayContext
        h, res = self.history.get(day), self.results.get(day)
        if h is None or res is None:
            return None
        if day not in self._snaps:
            r = self.run
            tickers = r.universe + ([BENCHMARK] if BENCHMARK not in r.universe else [])
            snap = r.store.snapshot(day, tickers, r.rc.data.lookback_days, r.macro_series)
            self._snaps[day] = _filter(snap, set(r.universe))
        return DayContext(self._snaps[day], copy.deepcopy(h["weights"]), copy.deepcopy(h["state"]), dict(res))

    def scorer(self, res: AgentResult, day: date) -> tuple[list[float], list[float]]:
        """Horizon-1 (signed conviction, excess) pairs, as ScoreStore scores them."""
        i = self.all_days.index(day)
        if i + 1 >= len(self.all_days) or res.output is None:
            return [], []
        nxt = self.all_days[i + 1]
        bench = self.run.pipeline.specs[res.agent].benchmark or BENCHMARK
        rows = {}
        for p in res.output.picks:
            if p.direction == "none":
                continue
            sign = 1 if p.direction == "long" else -1
            tp = self.run.store.outcome_prices(p.ticker, day, nxt).reindex([day, nxt])
            bp = self.run.store.outcome_prices(bench, day, nxt).reindex([day, nxt])
            ex = excess_return(tp, bp, 1, self.run.cost_frac, sign)
            if ex == ex:
                rows[p.ticker] = (sign * p.conviction, ex)
        return [c for c, _ in rows.values()], [e for _, e in rows.values()]

    def run_agent(self, name: str, day: date, prompt: str) -> AgentResult:
        ctx = self.context(day)
        if ctx is None:
            raise KeyError(f"no recorded context for {day}")
        return self.pipeline.run_single(name, day, ctx.snapshot, ctx.weights, ctx.state, ctx.upstream,
                                        prompt_override=prompt)

    def run_one_day(self, agent: str):
        from atlas.autoresearch.loop import build_run_one_day
        return build_run_one_day(self.pipeline, agent, self.context, self.scorer, horizon=1)


def _make_replay(run: Run) -> Replay:
    rc = run.rc
    ckpt = Checkpoint(rc.path(CHECKPOINT))
    if _last_completed(ckpt) is None:
        raise OperationalError(f"no stored run in {rc.run_dir}: run `atlas backtest --config ...` first")
    ar_failures = FailureLog(rc.path(AR_FAILURES))
    pipe = Pipeline(run.specs, run.prompts, run.llm, ar_failures, run.universe, run.settings.min_valid_agents)
    return Replay(run, pipe, run.trading_days(), read_day_history(rc.path(HISTORY_NAME), ckpt),
                  predlog_results(rc.path(PREDLOG)))


def build_replay(config: str | Path, *, llm_client=None, fred_key: str | None = None) -> Replay:
    rc = load_run_config(config)
    return _make_replay(build_run(rc, fred_key=fred_key, llm_client=llm_client))


def train_failure_stats(failures: FailureLog, specs: list[AgentSpec], periods: Periods, safe_end: date,
                        committed: list[date]) -> tuple[dict[str, float], dict[str, str]]:
    """Per non-control agent: share of committed train days (<= safe_end) on which it had at
    least one failed call (conservative: recovered retries count), and its failure reasons."""
    days = [d for d in committed if periods.train[0] <= d <= safe_end]
    df = failures.rows(CALL_FAILURE)
    if len(df):
        dd = [date.fromisoformat(x) for x in df["date"]]
        df = df[[periods.train[0] <= d <= safe_end for d in dd]]
    rates, notes = {}, {}
    for sp in specs:
        if sp.kind == "control":
            continue
        g = df[df["agent"] == sp.name] if len(df) else df
        rates[sp.name] = (g["date"].nunique() / len(days)) if days and len(g) else 0.0
        if len(g):
            vc = g["reason"].value_counts()
            notes[sp.name] = ", ".join(f"{r} x{n}" for r, n in sorted(vc.items(), key=lambda x: (-x[1], x[0])))
    return rates, notes


def cmd_autoresearch(args, llm_client=None) -> int:
    from atlas.autoresearch.gitops import DirtyWorkingTree, GitError, GitOps
    from atlas.autoresearch.loop import (
        HoldoutViolation, InsufficientReplayDays, last_safe_train_day, run_autoresearch,
    )
    rc = load_run_config(args.config)
    refuse_anonymize(rc, "autoresearch")
    fred_key = require_keys(rc, anthropic=True, fred=True)
    try:
        GitOps(rc.prompt_repo).require_clean()
    except GitError as e:
        raise UsageError(f"repo {rc.prompt_repo} is not usable as a git repository: {e}") from None
    except DirtyWorkingTree as e:
        raise OperationalError(f"autoresearch refused: {e} (in {rc.prompt_repo})") from None
    run = build_run(rc, fred_key=fred_key, llm_client=llm_client)
    stored_run(rc, run, "autoresearch")  # train/holdout split and settings of the STORED run (ruling U)
    replay = _make_replay(run)
    spent = spend_so_far(rc.run_dir)
    remaining = run.settings.budget_usd - sum(spent.values())
    if remaining <= 0:
        raise OperationalError(f"LLM budget exhausted: ${sum(spent.values()):.2f} of ${run.settings.budget_usd:.2f} "
                               "already spent on this run (backtest + control + autoresearch); raise budget_usd")
    run.llm.budget_usd = remaining  # one hard cap across the run directory
    print(f"autoresearch budget: ${remaining:.2f} remaining of ${run.settings.budget_usd:.2f}")
    scores = ScoreStore(rc.path(SCORES))
    try:
        null, _ = train_null(scores, run, replay.all_days)
        keep_margin = null.keep_margin()
        print(f"keep_margin {keep_margin:.4f} (null test on train picks, window {run.settings.rolling_ic_window})")
    except (ValueError, LeakageSuspected) as e:
        keep_margin = None
        print(f"keep_margin: not calibrated ({type(e).__name__}: {e}); attempts will be recorded as "
              "'uncalibrated' and no edit can be kept")
    safe_end = last_safe_train_day(replay.all_days, run.periods, 1)
    meta = {**null_test_meta(rc, run.settings, safe_end), "keep_margin": keep_margin}
    _drop_torn_tail(rc.path(AR_LOG))
    rates, notes = train_failure_stats(run.failures, run.specs, run.periods, safe_end or run.periods.train[0],
                                       sorted(replay.history))
    try:
        records = run_autoresearch(run.settings, repo=rc.prompt_repo, periods=run.periods,
                                   trading_days=replay.all_days, specs=run.specs, llm=run.llm, scores=scores,
                                   failure_rates=rates, make_run_one_day=replay.run_one_day,
                                   keep_margin=keep_margin, run_dir=rc.run_dir, horizon=1,
                                   failure_notes=lambda a: notes.get(a, ""),
                                   n_replay_days=rc.autoresearch.n_replay_days, null_test_meta=meta)
    except DirtyWorkingTree as e:
        raise OperationalError(f"autoresearch refused: {e}") from None
    except InsufficientReplayDays as e:
        raise OperationalError(f"autoresearch cannot run: {e}. A paired replay needs a longer train "
                               "period.") from None
    except HoldoutViolation as e:
        raise OperationalError(f"autoresearch stopped: {e}") from None
    for r in records:
        print(f"{r.agent}: {r.decision} ({r.reason})" + (f" commit {r.commit_sha}" if r.commit_sha else ""))
    if not records:
        print("autoresearch: no eligible agent to edit")
    print(f"log: {rc.path(AR_LOG)}. Edits touch only atlas/prompts/ and are never pushed. A backtest "
          "with edited prompts needs a new run_dir (the old checkpoint fingerprints the prompts).")
    return EXIT_OK


# ============================================================================ report

def result_from_checkpoint(raw: dict, periods: Periods) -> BacktestResult:
    ser = lambda m, name: pd.Series([m[k] for k in sorted(m)], index=[load_date(k) for k in sorted(m)],  # noqa: E731
                                    name=name, dtype=float)
    return BacktestResult(
        equity=ser(raw["equity"], "equity"), spy=ser(raw["spy"], "spy"), basket=ser(raw["basket"], "basket"),
        day_results={}, skipped={load_date(k): v for k, v in raw["skipped"].items()},
        limitations=list(LIMITATIONS) + [ABLATION_NOTE],
        risk_errors={load_date(k): v for k, v in raw["risk_errors"].items()},
        halted=bool(raw["halted"]), halt_reason=raw["halt_reason"],
        final_portfolio=Portfolio(cash=float(raw["portfolio"]["cash"]),
                                  positions={t: int(n) for t, n in raw["portfolio"]["positions"].items()}),
        spent_usd=float(raw["spent_usd"]), periods=periods)


def turnover_by_period(history: dict[date, dict], equity: pd.Series, periods: Periods) -> dict[str, float] | None:
    """Traded notional (fills booked to their decision day) / mean equity, per period."""
    if not history:
        return None
    out = {}
    for name, (lo, hi) in periods.items():
        eq = equity[[lo <= d <= hi for d in equity.index]]
        if not len(eq) or not float(eq.mean()) > 0:
            continue
        notional = sum(f["shares"] * f["price"] for d, h in history.items() if lo <= d <= hi for f in h["fills"])
        out[name] = notional / float(eq.mean())
    return out


def ablation_from_run(history: dict[date, dict], results: dict[date, dict[str, AgentResult]],
                      store: PriceStore, specs: list[AgentSpec], limits: Limits,
                      periods: Periods) -> dict[str, dict | None] | None:
    """CIO ablation (see ABLATION_NOTE) per period name plus "all"; a period's entry is None
    if it has fewer than 2 comparable days. None if there is no history at all."""
    from atlas.backtest.ablation import baseline_target_weights, cio_ablation
    if not history:
        return None
    layers = {sp.name: sp.layer for sp in specs}
    ex_names = [sp.name for sp in specs if sp.kind == "execution"]
    days = sorted(history)
    cal = store.trading_days(days[0], days[-1] + timedelta(days=14))
    cio_r, base_r = {}, {}
    for T in days:
        h, res = history[T], results.get(T)
        if h.get("skipped") or res is None or not ex_names or ex_names[0] not in res:
            continue
        ex = res[ex_names[0]]
        if ex.status != "ok":
            continue
        cio_w = {}
        for p in ex.output.picks:  # last pick per ticker wins, as in the pipeline
            cio_w[p.ticker] = p.conviction if p.direction == "long" else 0.0
        cio_w = {t: w for t, w in cio_w.items() if w > 0}
        gross = sum(cio_w.values())
        if gross > limits.max_gross_pct > 0:
            cio_w = {t: w * limits.max_gross_pct / gross for t, w in cio_w.items()}
        base_w = baseline_target_weights(res, h["weights"], limits.max_positions, layers)
        gross = sum(cio_w.values())  # gross-match: only selection/weighting may differ
        base_w = {t: w * gross for t, w in base_w.items()}
        later = [d for d in cal if d > T]
        if not later:
            continue
        nxt = later[0]
        rets = {}
        for t in set(cio_w) | set(base_w):
            s = store.outcome_prices(t, T, nxt).reindex([T, nxt])
            if s.isna().any() or float(s.iloc[0]) == 0:
                break
            rets[t] = float(s.iloc[1]) / float(s.iloc[0]) - 1.0
        else:
            cio_r[T] = sum(w * rets[t] for t, w in cio_w.items())
            base_r[T] = sum(w * rets[t] for t, w in base_w.items())
    def one(keep) -> dict | None:
        c = pd.Series({d: v for d, v in cio_r.items() if keep(d)}, dtype=float)
        b = pd.Series({d: v for d, v in base_r.items() if keep(d)}, dtype=float)
        try:
            return cio_ablation(c, b)
        except ValueError:
            return None

    out = {name: one(lambda d, lo=lo, hi=hi: lo <= d <= hi) for name, (lo, hi) in periods.items()}
    out["all"] = one(lambda d: True)
    return out


def cmd_report(args, llm_client=None) -> int:
    from atlas.autoresearch.loop import EditRecord
    from atlas.backtest.report import build_report
    run_dir = Path(args.run_dir)
    stored_path = run_dir / RUN_CONFIG
    if not stored_path.exists():
        raise UsageError(f"{run_dir} is not a run directory (no {RUN_CONFIG}); run `atlas backtest` first")
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    rc = RunConfig.model_validate(stored["run_config"]).model_copy(update={"run_dir": run_dir})
    if rc.data.cache_dir is None:
        rc = rc.model_copy(update={"data": rc.data.model_copy(update={"cache_dir": run_dir / "prices"})})
    settings = Settings(**stored["settings"])
    limits = Limits(**stored["limits"])
    specs = [AgentSpec(**a) for a in stored["agents"]]
    periods = resolve_periods(rc)
    ckpt = Checkpoint(run_dir / CHECKPOINT)
    raw = _load_checkpoint(ckpt)
    if not raw or not raw.get("last_completed_day"):
        raise OperationalError(f"no completed day in {run_dir}: nothing to report")
    run = Run(rc, settings, limits, specs, {}, periods, PriceStore(CacheOnlyPrices(), NoMacro(), rc.cache_dir), [])
    result = result_from_checkpoint(raw, periods)
    scores = ScoreStore(run_dir / SCORES)

    null, null_src, null_meta = None, "not calibrated", None
    for p in (run_dir / CALIBRATION, rc.calibration):
        if p is not None and Path(p).exists():
            cal = json.loads(Path(p).read_text())
            null, null_src = null_from_calibration(cal), str(p)
            null_meta = {k: cal.get(k) for k in NULL_META_KEYS}
            break
    if null is None:
        try:
            null, safe_end = train_null(scores, run, _cached_days(run))
            null_src = "computed from this run's train picks"
            null_meta = null_test_meta(rc, settings, safe_end)
        except (ValueError, LeakageSuspected, OperationalError) as e:
            null_src = f"not computable ({type(e).__name__})"
    known = set(EditRecord.__dataclass_fields__)
    ar_records = read_ar_log(run_dir / AR_LOG)
    edits = [EditRecord(**{k: v for k, v in r.items() if k in known}) for r in ar_records]
    ar_metas = [r.get("null_test") for r in ar_records]
    failures = FailureLog(run_dir / FAILURES)
    spent = spend_so_far(run_dir)  # backtest + control + autoresearch: one total
    weights = WeightBook(run_dir / WEIGHTS).load()
    final_weights = {sp.name: weights.get(sp.name, W_INIT) for sp in specs if sp.kind in WEIGHTED_KINDS}
    history = read_day_history(run_dir / HISTORY_NAME, ckpt)
    turnover = turnover_by_period(history, result.equity, periods)
    by_period = ablation_from_run(history, predlog_results(run_dir / PREDLOG), run.store, specs, limits, periods)
    ablation = by_period.get("holdout") if by_period else None
    path = build_report(result, scores, null, edits, ablation, None, failures.summary(),
                        sum(spent.values()), run_dir / "report", settings=settings,
                        ablation_by_period=by_period, null_meta=null_meta, null_source=null_src,
                        autoresearch_null_meta=ar_metas,
                        holdout_edit_ics=None, final_weights=final_weights,
                        dropped_picks=failures.dropped_picks_summary(), turnover=turnover)
    print(f"null distribution: {null_src}; CIO ablation: {'computed' if by_period else 'not provided'}; "
          f"turnover: {'computed' if turnover is not None else 'not provided'}")
    print(f"wrote {path}")
    return EXIT_OK


# ============================================================================ entry point

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="atlas", description="ATLAS backtest system: multi-agent LLM "
                                "trading research with honest, leakage-aware evaluation.")
    sub = p.add_subparsers(dest="command", required=True, metavar="{backtest,calibrate,autoresearch,report}")
    b = sub.add_parser("backtest", help="run or resume a backtest over train, validation and holdout")
    b.add_argument("--config", required=True, help="run config YAML (see atlas/config/run.example.yaml)")
    b.add_argument("--dry-run", action="store_true", help="no keys, zero LLM calls: plumbing check")
    b.add_argument("--yes", action="store_true", help="do not ask before spending above confirm_cost_usd")
    b.add_argument("--with-control", action="store_true",
                   help="also run the date-only control agent each day (one extra LLM call per day)")
    c = sub.add_parser("calibrate", help="null test on the run's train picks -> calibration.json")
    c.add_argument("--config", required=True)
    a = sub.add_parser("autoresearch", help="targeted prompt edits judged by paired replay on train days")
    a.add_argument("--config", required=True)
    r = sub.add_parser("report", help="write report.md, equity.png, agents.csv, autoresearch.csv")
    r.add_argument("--run-dir", required=True)
    return p


COMMANDS = {"backtest": cmd_backtest, "calibrate": cmd_calibrate, "autoresearch": cmd_autoresearch,
            "report": cmd_report}


def main(argv: list[str] | None = None, *, llm_client=None) -> int:
    """`llm_client` replaces the Anthropic client (tests inject a StubClient)."""
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else EXIT_USAGE
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        return COMMANDS[args.command](args, llm_client=llm_client)
    except UsageError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except (OperationalError, ResumeRefused, FredFetchError) as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
