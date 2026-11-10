"""Backtest runner tests on a synthetic weekday market with a scripted StubClient.

All 12 agents return valid JSON by default; the execution agent emits target weights so the
CIO/execution path produces real orders and fills.
"""
import hashlib
import json
import math
import re
import threading
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

import atlas.backtest.runner as runner_mod
import atlas.engine.agent as agent_mod
from atlas.backtest.checkpoint import Checkpoint
from atlas.backtest.predlog import PredictionLog
from atlas.backtest.runner import (
    BacktestConfig, CostConfirmationRequired, Periods, ResumeRefused, RunPaused,
    anonymize_snapshot, clear_halt, estimate_cost, run_backtest,
)
from atlas.config import AgentSpec, Limits, Settings
from atlas.data.provider import PRICE_COLUMNS
from atlas.data.store import PriceStore, Snapshot
from atlas.engine.failures import FailureLog
from atlas.engine.llm import ApiError, CachedLLM, StubClient
from atlas.engine.pipeline import Pipeline
from atlas.portfolio.models import Order
from atlas.portfolio.simbroker import SimBroker
from atlas.risk.layer import RiskDecision, RiskLayer
from atlas.scoring.store import HORIZONS, ScoreStore
from atlas.scoring.weights import WeightBook

START = date(2024, 1, 1)  # Monday
WEEKDAYS = [d for d in (START + timedelta(days=i) for i in range(120)) if d.weekday() < 5]
UNIVERSE = ["SPY", "XAA", "XBB", "XCC"]
SECTORS = {"SPY": "broad_market", "XAA": "semis", "XBB": "banks", "XCC": "energy"}
BASE = {"SPY": 400.0, "XAA": 100.0, "XBB": 50.0, "XCC": 20.0}
SLIP_BPS = 5.0
INITIAL = 100_000.0

SPECS = (
    [AgentSpec(name=f"m{i}", layer=1, kind="macro", prompt_file="x", benchmark="SPY") for i in (1, 2, 3)]
    + [AgentSpec(name="s1", layer=2, kind="sector", prompt_file="x", benchmark="XCC")]
    + [AgentSpec(name=f"s{i}", layer=2, kind="sector", prompt_file="x", benchmark="SPY") for i in (2, 3, 4)]
    + [AgentSpec(name=f"si{i}", layer=3, kind="superinvestor", prompt_file="x", benchmark="SPY")
       for i in (1, 2)]
    + [AgentSpec(name="cro", layer=4, kind="cro", prompt_file="x"),
       AgentSpec(name="cio", layer=4, kind="cio", prompt_file="x"),
       AgentSpec(name="execution", layer=4, kind="execution", prompt_file="x")]
)
assert len(SPECS) == 12
PROMPTS = {s.name: f"PROMPT<{s.name}>" for s in SPECS}


def px(ticker: str, i: int) -> tuple[float, float]:
    """(open, close) on weekday index i: distinct, deterministic, wiggly."""
    k = UNIVERSE.index(ticker) + 1
    close = BASE[ticker] * (1 + 0.002 * i * (1 if k % 2 else -1) + 0.02 * math.sin(i * 0.9 * k + k))
    return close * (1 - 0.003 * math.cos(i + k)), close


class FakePrices:
    def __init__(self, gaps=()):
        self.gaps = set(gaps)  # {(ticker, date)} with no bar

    def fetch(self, tickers, start, end):
        rows = []
        for t in tickers:
            for i, d in enumerate(WEEKDAYS):
                if start <= d <= end and (t, d) not in self.gaps:
                    o, c = px(t, i)
                    rows.append((d, t, o, max(o, c) + 1, min(o, c) - 1, c, c, 1000 + i))
        return pd.DataFrame(rows, columns=PRICE_COLUMNS)


class FakeMacro:
    def fetch(self, series_id, asof):
        return pd.Series(dtype=float)


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16)


def default_handler(name: str, u: dict) -> str:
    inputs = u["inputs"]
    tickers = sorted((inputs.get("market") or {}).get("prices") or inputs.get("last_close") or {})
    s = _seed(name, u["asof"])
    if name == "execution":
        picks = [{"ticker": t, "direction": "long", "conviction": round(0.05 + 0.05 * ((s >> (4 * k)) % 4), 2),
                  "horizon_days": 1} for k, t in enumerate(tickers[:3])]
    else:
        picks = [{"ticker": t, "direction": "long" if (s >> k) % 3 else "short",
                  "conviction": round(((s >> (8 * k)) % 100) / 100, 2), "horizon_days": 1}
                 for k, t in enumerate(tickers[1:4])]
    view = ["bullish", "bearish", "neutral"][s % 3]
    return json.dumps({"view": view, "confidence": 0.6, "picks": picks})


class RecordingBroker(SimBroker):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.submits = []  # (decision_date, orders, fills, dropped)

    def submit(self, orders, decision_date, portfolio=None):
        fills, dropped = super().submit(orders, decision_date, portfolio=portfolio)
        self.submits.append((decision_date, list(orders), fills, dropped))
        return fills, dropped


class CrashingCheckpoint(Checkpoint):
    """Simulates a process kill: raises from save() on the n-th call (after the day's fills,
    log append and scoring, before the commit point)."""

    def __init__(self, path, crash_on):
        super().__init__(path)
        self.n, self.crash_on = 0, crash_on

    def save(self, state):
        self.n += 1
        if self.n == self.crash_on:
            raise KeyboardInterrupt("simulated kill before checkpoint")
        super().save(state)


def settings(**over):
    base = dict(universe=UNIVERSE, commission_per_share=0.005, commission_min=1.0, slippage_bps=SLIP_BPS,
                min_valid_agents=8, rolling_ic_window=60, warmup_picks=3, budget_usd=1e9, model="m",
                usd_per_mtok_input=3.0, usd_per_mtok_output=15.0,
                model_cutoff=date(2023, 6, 1), dead_zone=0.0, confirm_cost_usd=1e9)
    base.update(over)
    return Settings(**base)


def limits(**over):
    base = dict(max_position_pct=0.3, max_gross_pct=1.0, max_positions=8, min_order_usd=50,
                max_commission_frac=0.01, daily_loss_pct=0.5, drawdown_kill_pct=0.9, long_only=True)
    base.update(over)
    return Limits(**base)


def periods_for(n_days: int) -> Periods:
    d = WEEKDAYS[:n_days]
    a, b = n_days // 3, 2 * n_days // 3
    return Periods(train=(d[0], d[a - 1]), validation=(d[a], d[b - 1]), holdout=(d[b], d[-1]))


class Env:
    def __init__(self, root, handler=default_handler, n_days=10, anonymize=False, dry_run=False,
                 set_over=None, lim_over=None, price=(0.0, 0.0), gaps=(), checkpoint=None,
                 shared_cache=None):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self.calls = []
        self.lock = threading.Lock()
        self.handler = handler
        self.settings = settings(**(set_over or {}))
        self.cfg = BacktestConfig(periods=periods_for(n_days), settings=self.settings,
                                  limits=limits(**(lim_over or {})), anonymize=anonymize, dry_run=dry_run)
        self.store = PriceStore(FakePrices(gaps), FakeMacro(), (shared_cache or root) / "prices")
        self.price = price
        self.pipeline = self._pipeline()
        self.broker = RecordingBroker(self.store, self.settings.commission_per_share,
                                      self.settings.commission_min, self.settings.slippage_bps)
        self.risk = RiskLayer(self.cfg.limits, self.settings.commission_per_share, self.settings.commission_min)
        self.scores = ScoreStore(root / "scores.db")
        self.weightbook = WeightBook(root / "weights.json")
        self.predlog = PredictionLog(root / "predictions.jsonl")
        self.checkpoint = checkpoint or Checkpoint(root / "checkpoint.json")

    def _fn(self, system, user):
        name = system[len("PROMPT<"):-1]
        u = json.loads(user)
        with self.lock:
            self.calls.append((system, user))
        return self.handler(name, u)

    def _pipeline(self, budget=1e9):
        self.llm = CachedLLM(StubClient(self._fn), self.root / "llm.db", "m", self.price, budget)
        return Pipeline(SPECS, PROMPTS, self.llm, FailureLog(self.root / "f.db"), UNIVERSE,
                        self.settings.min_valid_agents)

    def run(self, **kw):
        kw.setdefault("universe_sectors", SECTORS)
        kw.setdefault("initial_cash", INITIAL)
        return run_backtest(self.cfg, self.store, self.pipeline, self.broker, self.risk, self.scores,
                            self.weightbook, self.predlog, self.checkpoint, **kw)

    def score_rows(self):
        return pd.concat([self.scores.picks_frame(None, h) for h in HORIZONS], ignore_index=True)


def open_close(ticker, d):
    return px(ticker, WEEKDAYS.index(d))


# ---------------------------------------------------------------------------- periods / config

def test_periods_must_be_chronological_and_non_overlapping():
    d = WEEKDAYS
    p = Periods(train=(d[0], d[4]), validation=(d[5], d[9]), holdout=(d[10], d[14]))
    assert p.period_of(d[0]) == "train" and p.period_of(d[4]) == "train"
    assert p.period_of(d[7]) == "validation" and p.period_of(d[14]) == "holdout"
    with pytest.raises(ValueError):
        p.period_of(d[20])
    with pytest.raises(ValueError):  # overlap train/validation
        Periods(train=(d[0], d[5]), validation=(d[5], d[9]), holdout=(d[10], d[14]))
    with pytest.raises(ValueError):  # out of order
        Periods(train=(d[10], d[14]), validation=(d[5], d[9]), holdout=(d[15], d[19]))
    with pytest.raises(ValueError):  # start after end
        Periods(train=(d[4], d[0]), validation=(d[5], d[9]), holdout=(d[10], d[14]))
    with pytest.raises(ValueError):  # holdout overlapping validation
        Periods(train=(d[0], d[4]), validation=(d[5], d[12]), holdout=(d[10], d[14]))
    # the holdout must lie after the model's training cutoff
    with pytest.raises(ValueError, match="cutoff"):
        BacktestConfig(periods=p, settings=settings(model_cutoff=d[12]), limits=limits(),
                       anonymize=False, dry_run=False)


def test_period_without_trading_days_is_rejected(tmp_path):
    env = Env(tmp_path)
    sat, sun = date(2024, 1, 6), date(2024, 1, 7)
    env.cfg = BacktestConfig(periods=Periods(train=(WEEKDAYS[0], WEEKDAYS[2]), validation=(sat, sun),
                                             holdout=(WEEKDAYS[6], WEEKDAYS[8])),
                             settings=env.settings, limits=env.cfg.limits, anonymize=False, dry_run=False)
    with pytest.raises(ValueError, match="validation"):
        env.run()


# ---------------------------------------------------------------------------- core loop

def test_decisions_on_day_t_fill_at_t_plus_1_open(tmp_path):
    env = Env(tmp_path, n_days=6)
    env.run()
    n_fills = 0
    for decision, _orders, fills, _dropped in env.broker.submits:
        nxt = WEEKDAYS[WEEKDAYS.index(decision) + 1]
        for f in fills:
            n_fills += 1
            assert f.date == nxt
            o, _ = open_close(f.order.ticker, nxt)
            want = o * (1 + SLIP_BPS / 1e4) if f.order.side == "buy" else o * (1 - SLIP_BPS / 1e4)
            assert f.price == pytest.approx(want, rel=1e-12)
    assert n_fills >= 3


def test_equity_for_day_t_excludes_t_own_fills(tmp_path):
    env = Env(tmp_path, n_days=3)
    res = env.run()
    d0, d1, d2 = WEEKDAYS[:3]
    assert res.equity[d0] == INITIAL  # day-0 decisions fill at d1's open: not in d0's equity
    fills0 = [f for dd, _, fs, _ in env.broker.submits if dd == d0 for f in fs]
    assert fills0
    cash, pos = INITIAL, {}
    for f in fills0:
        sgn = 1 if f.order.side == "buy" else -1
        cash -= sgn * f.order.shares * f.price + f.commission
        pos[f.order.ticker] = pos.get(f.order.ticker, 0) + sgn * f.order.shares
    # d1 equity: d0's fills (at d1 open) marked at d1 close, d1's own fills excluded
    want = cash + sum(n * open_close(t, d1)[1] for t, n in pos.items())
    assert res.equity[d1] == pytest.approx(want, rel=1e-12)
    assert list(res.equity.index) == [d0, d1, d2]
    # benchmarks normalised to the starting equity
    assert res.spy[d0] == pytest.approx(INITIAL) and res.basket[d0] == pytest.approx(INITIAL)
    assert res.spy[d2] == pytest.approx(INITIAL * open_close("SPY", d2)[1] / open_close("SPY", d0)[1])
    assert any("adjusted" in s for s in res.limitations)


def test_skipped_day_makes_no_trades_and_is_logged(tmp_path):
    bad_day = WEEKDAYS[3]

    def handler(name, u):
        if u["asof"] == bad_day.isoformat() and name in ("m1", "m2", "m3", "s1", "s2"):
            return "this is not json"
        return default_handler(name, u)

    env = Env(tmp_path, handler, n_days=6)
    res = env.run()
    assert set(res.skipped) == {bad_day}
    reason = res.skipped[bad_day]
    assert "malformed_json" in reason and "m1" in reason and "s2" in reason
    assert bad_day not in [dd for dd, *_ in env.broker.submits]
    assert bad_day in res.equity.index  # still marked to market
    days_logged = [json.loads(line)["day"] for line in env.predlog.path.read_text().splitlines()]
    assert bad_day.isoformat() in days_logged  # predictions still recorded
    assert res.day_results[bad_day].skipped_reason == reason
    assert json.loads(env.checkpoint.path.read_text())["skipped"] == {bad_day.isoformat(): reason}


def test_prediction_log_written_before_outcomes_scored(tmp_path):
    env = Env(tmp_path, n_days=5)
    events = []
    orig_append, orig_record, orig_score = env.predlog.append, env.scores.record_picks, env.scores.score_matured

    def append(day, results):
        events.append(("append", day))
        return orig_append(day, results)

    def record(result):
        events.append(("record", result.date))
        return orig_record(result)

    def score(asof, *a, **k):
        events.append(("score", asof))
        return orig_score(asof, *a, **k)

    env.predlog.append, env.scores.record_picks, env.scores.score_matured = append, record, score
    env.run()
    for d in WEEKDAYS[:5]:
        i_append = events.index(("append", d))
        assert i_append < events.index(("score", d))
        assert all(i_append < i for i, e in enumerate(events) if e == ("record", d))
        # nothing scored at or after day d before d's predictions are logged
        assert not any(k == "score" and x >= d for k, x in events[:i_append])


# ---------------------------------------------------------------------------- cost control

def test_cost_estimate_over_ceiling_requires_confirmation(tmp_path):
    env = Env(tmp_path, n_days=4, set_over={"confirm_cost_usd": 0.01})
    est = estimate_cost(env.cfg, 4, 12)
    assert est > 0.01
    assert estimate_cost(env.cfg, 8, 12) == pytest.approx(2 * est)
    assert estimate_cost(env.cfg, 4, 6) == pytest.approx(est / 2)
    with pytest.raises(CostConfirmationRequired) as ei:
        env.run()
    assert ei.value.estimate == pytest.approx(est)
    assert env.calls == [] and not env.predlog.path.exists()
    res = env.run(confirm=True)
    assert len(res.equity) == 4 and env.calls


def test_dry_run_makes_zero_llm_calls(tmp_path):
    env = Env(tmp_path, n_days=6, dry_run=True, set_over={"confirm_cost_usd": 0.0})
    assert estimate_cost(env.cfg, 6, 12) == 0.0
    res = env.run()
    assert env.calls == []
    assert len(res.equity) == 6
    assert len(env.predlog.path.read_text().splitlines()) == 6
    assert res.skipped == {}
    assert all(r.status == "ok" for dr in res.day_results.values() for r in dr.results.values())


# ---------------------------------------------------------------------------- resume

def _final(env, res):
    return (res.final_portfolio.cash, dict(res.final_portfolio.positions), res.equity.to_dict(),
            res.spy.to_dict(), res.basket.to_dict(), env.score_rows(), env.weightbook.load(),
            env.predlog.path.read_bytes())


def test_resume_after_interruption_does_not_double_fill_or_double_score(tmp_path):
    ref_env = Env(tmp_path / "ref", n_days=10)
    ref = _final(ref_env, ref_env.run())
    assert any(abs(w - 1.0) > 1e-12 for w in ref[6].values())  # weights actually moved

    env = Env(tmp_path / "crash", n_days=10, checkpoint=CrashingCheckpoint(tmp_path / "crash" / "cp.json", 6))
    with pytest.raises(KeyboardInterrupt):
        env.run()
    day6 = WEEKDAYS[5]
    assert day6 in [dd for dd, *_ in env.broker.submits]  # day 6 had filled before the "kill"
    assert json.loads(env.predlog.path.read_text().splitlines()[-1])["day"] == day6.isoformat()
    # restart: fresh objects on the same files, normal checkpoint
    env2 = Env(tmp_path / "crash", n_days=10, checkpoint=Checkpoint(tmp_path / "crash" / "cp.json"))
    res2 = env2.run()
    got = _final(env2, res2)
    assert got[0] == ref[0] and got[1] == ref[1]
    assert got[2] == ref[2] and got[3] == ref[3] and got[4] == ref[4]
    pd.testing.assert_frame_equal(got[5], ref[5])
    assert got[6] == ref[6]
    assert got[7] == ref[7]
    assert min(res2.day_results) == day6  # only the uncommitted day and later were re-run


def test_resume_refused_when_predlog_tail_truncated(tmp_path):
    env = Env(tmp_path, n_days=6)
    env.run()
    lines = env.predlog.path.read_text().splitlines(keepends=True)
    env.predlog.path.write_text("".join(lines[:-2]))  # drop the last two (checkpointed) lines
    with pytest.raises(ResumeRefused, match="prediction log"):
        env.run()


def test_torn_predlog_line_gives_repair_message(tmp_path):
    env = Env(tmp_path, n_days=4)
    env.run()
    with open(env.predlog.path, "a") as f:
        f.write('{"day": "2024-01-0')
    with pytest.raises(ResumeRefused, match="torn final line.*repair it manually"):
        env.run()


def test_resume_refused_on_config_mismatch(tmp_path):
    env = Env(tmp_path, n_days=4)
    env.run()
    env.cfg = BacktestConfig(periods=env.cfg.periods, settings=env.settings, limits=env.cfg.limits,
                             anonymize=True, dry_run=False)
    with pytest.raises(ResumeRefused, match="anonymize"):
        env.run()


# ---------------------------------------------------------------------------- pauses

def test_transient_failures_pause_run_but_content_failures_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    bad_day = WEEKDAYS[3]
    broken = {"on": True}

    def handler(name, u):
        if broken["on"] and u["asof"] == bad_day.isoformat() and name in ("m1", "m2", "m3", "s1", "s2"):
            raise ApiError("503 upstream")
        return default_handler(name, u)

    env = Env(tmp_path, handler, n_days=6)
    with pytest.raises(RunPaused) as ei:
        env.run()
    p = ei.value
    assert p.day == bad_day
    assert p.failed == {n: "api_error" for n in ("m1", "m2", "m3", "s1", "s2")}
    assert bad_day not in [dd for dd, *_ in env.broker.submits]
    cp = json.loads(env.checkpoint.path.read_text())
    assert cp["last_completed_day"] == WEEKDAYS[2].isoformat()
    assert len(env.predlog.path.read_text().splitlines()) == 3  # paused day not logged

    broken["on"] = False  # the outage is over: resume
    res = Env(tmp_path, handler, n_days=6).run()
    assert res.skipped == {} and bad_day in res.equity.index


def test_budget_exceeded_pauses_and_spend_is_restored_on_resume(tmp_path):
    price = (1.0, 0.0)  # input-only pricing keeps the worst-case reservation close to actual
    probe = Env(tmp_path / "probe", n_days=6, price=price)
    probe.run()
    per_run = probe.llm.spent_usd
    assert per_run > 0

    env = Env(tmp_path / "run", n_days=6, price=price)
    env.pipeline = env._pipeline(budget=per_run * 0.5)
    with pytest.raises(RunPaused, match="budget") as ei:
        env.run()
    assert ei.value.failed == {}
    cp = json.loads(env.checkpoint.path.read_text())
    assert 0 < cp["spent_usd"] <= per_run * 0.5
    last = cp["last_completed_day"]

    # a restart with a fresh CachedLLM (spend 0) inherits the spend: the cap spans resumes
    env2 = Env(tmp_path / "run", n_days=6, price=price)
    env2.pipeline = env2._pipeline(budget=cp["spent_usd"])  # no headroom left after restore
    with pytest.raises(RunPaused, match="budget"):
        env2.run()
    assert env2.calls == []  # restored spend blocked the very first live call of the resumed day
    assert env2.llm.spent_usd >= cp["spent_usd"]
    assert json.loads(env2.checkpoint.path.read_text())["last_completed_day"] == last

    env3 = Env(tmp_path / "run", n_days=6, price=price)
    env3.pipeline = env3._pipeline(budget=1e9)
    res = env3.run()
    assert len(res.equity) == 6
    assert res.spent_usd >= cp["spent_usd"]
    assert json.loads(env3.checkpoint.path.read_text())["spent_usd"] == res.spent_usd


# ---------------------------------------------------------------------------- risk

class HaltOnDay(RiskLayer):
    def __init__(self, base: RiskLayer, day):
        super().__init__(base.limits, base.commission_per_share, base.commission_min)
        self.day, self.calls = day, []

    def apply(self, orders, positions, cash, prices, prior_close_equity, peak_equity):
        self.calls.append(list(orders))
        if len(self.calls) == self.day:
            flat = [Order(t, "sell", n) for t, n in sorted(positions.items())]
            return RiskDecision(flat, [(o, "halted") for o in orders], True, "test kill switch")
        return super().apply(orders, positions, cash, prices, prior_close_equity, peak_equity)


def test_halt_is_sticky_across_resume_and_suppresses_orders(tmp_path):
    env = Env(tmp_path, n_days=9, checkpoint=CrashingCheckpoint(tmp_path / "cp.json", 6))
    env.risk = HaltOnDay(env.risk, 3)  # the 3rd risk evaluation (day index 2) halts
    with pytest.raises(KeyboardInterrupt):
        env.run()
    halt_day = WEEKDAYS[2]
    by_day = {dd: (orders, fills) for dd, orders, fills, _ in env.broker.submits}
    assert by_day[halt_day][0] and all(o.side == "sell" for o in by_day[halt_day][0])  # flatten sent
    assert max(by_day) == halt_day  # nothing traded after the halt
    cp = json.loads(env.checkpoint.path.read_text())
    assert cp["halted"] is True and cp["halt_reason"] == "test kill switch"

    env2 = Env(tmp_path, n_days=9, checkpoint=Checkpoint(tmp_path / "cp.json"))
    res = env2.run()
    assert env2.broker.submits == []  # still halted after resume
    assert res.halted and res.halt_reason == "test kill switch"
    assert res.final_portfolio.positions == {}
    assert len(res.equity) == 9  # still marked to market
    assert len(env2.predlog.path.read_text().splitlines()) == 9  # predictions still recorded

    clear_halt(env2.checkpoint)
    cp = json.loads(env2.checkpoint.path.read_text())
    assert cp["halted"] is False and cp["halt_reason"] is None
    assert cp["peak_equity"] == res.equity.iloc[-1]


def test_risk_layer_exception_sends_no_orders(tmp_path):
    class Boom(RiskLayer):
        def apply(self, *a, **k):
            raise RuntimeError("risk bug")

    env = Env(tmp_path, n_days=4)
    env.risk = Boom(env.cfg.limits, 0.005, 1.0)
    res = env.run()
    assert env.broker.submits == []
    assert res.final_portfolio.positions == {} and res.final_portfolio.cash == INITIAL
    assert len(res.risk_errors) == 4 and all("risk bug" in v for v in res.risk_errors.values())


def test_held_ticker_with_missing_bar_marked_at_last_known_price(tmp_path):
    gap_day = WEEKDAYS[3]
    env = Env(tmp_path, n_days=6, gaps={("XAA", gap_day)})
    res = env.run()
    assert len(res.equity) == 6 and res.equity.notna().all()
    # no order for the gapped ticker on its gap day
    assert all(o.ticker != "XAA" for dd, orders, *_ in env.broker.submits if dd == gap_day for o in orders)


# ---------------------------------------------------------------------------- anonymisation

def test_anonymize_snapshot_unit():
    rows = []
    for i, d in enumerate(WEEKDAYS[:4]):
        for t in ("SPY", "XAA"):
            o, c = px(t, i)
            rows.append({"date": d, "ticker": t, "open": o, "high": o, "low": c, "close": c,
                         "adj_close": c, "volume": 1})
    snap = Snapshot(asof=WEEKDAYS[3], prices=pd.DataFrame(rows),
                    macro={"DGS10": pd.Series([4.0, 4.1], index=[WEEKDAYS[0], WEEKDAYS[3]])})
    anon, mapping = anonymize_snapshot(snap, {"SPY": "broad_market", "XAA": "semis"})
    assert set(mapping.values()) == {"SPY", "XAA"}
    assert all(re.fullmatch(r"A\d\d", k) for k in mapping)
    p = anon.prices
    assert set(p["ticker"]) == set(mapping)
    assert sorted(set(p["date"])) == [-3, -2, -1, 0]
    for oid, real in mapping.items():
        g = p[p["ticker"] == oid].sort_values("date")
        assert g["close"].iloc[0] == pytest.approx(100.0)
        assert g["close"].iloc[-1] == pytest.approx(100 * px(real, 3)[1] / px(real, 0)[1])
        assert set(g["sector"]) == {SECTORS[real]}
    assert list(anon.macro["DGS10"].index) == [-(WEEKDAYS[3] - WEEKDAYS[0]).days, 0]
    assert not isinstance(anon.asof, date)
    _, mapping2 = anonymize_snapshot(snap, {"SPY": "broad_market", "XAA": "semis"})
    assert mapping2 == mapping  # stable across days


def _strings(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k)
            yield from _strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v)
    elif isinstance(obj, str):
        yield obj


def test_anonymize_hides_tickers_and_dates_but_keeps_sector(tmp_path):
    env = Env(tmp_path, n_days=6, anonymize=True)
    res = env.run()
    assert env.calls
    year_re = re.compile(r"(19|20)\d\d")
    sectors_seen = set()
    for system, user in env.calls:
        for t in UNIVERSE:
            assert t not in user and t not in system
        assert not re.search(r"\d{4}-\d{2}-\d{2}", user), user[:200]
        assert re.fullmatch(r"day \d+", json.loads(user)["asof"])
        for s in _strings(json.loads(user)):  # numbers aside, no text looks like a year
            assert not year_re.search(s), s
        sectors_seen |= {s for s in SECTORS.values() if s in user}
    assert sectors_seen == set(SECTORS.values())
    # but the log, the score store and the fills use REAL tickers and dates
    log = [json.loads(line) for line in env.predlog.path.read_text().splitlines()]
    assert [r["day"] for r in log] == [d.isoformat() for d in WEEKDAYS[:6]]
    picked = {p["ticker"] for r in log for a in r["agents"] if a["status"] == "ok" for p in a["output"]["picks"]}
    assert picked and picked <= set(UNIVERSE)
    rows = env.score_rows()
    assert set(rows["ticker"]) <= set(UNIVERSE) and set(rows["date"]) <= set(WEEKDAYS[:6])
    fills = [f for *_, fs, _ in env.broker.submits for f in fs]
    assert fills and {f.order.ticker for f in fills} <= set(UNIVERSE)
    # orders were sized from real prices: every fill is within the 30% position cap
    for f in fills:
        if f.order.side == "buy":
            assert f.order.shares * f.price <= 0.31 * INITIAL * 1.2


# ---------------------------------------------------------------------------- fix round 1

DAY6 = WEEKDAYS[5]


def _crash_then_resume(root, phase1, phase2, n_days=10, crash_on=6):
    """Run with `phase1` until the kill at the crash_on-th checkpoint save, then resume with
    fresh objects on the same files using `phase2`."""
    env = Env(root, phase1, n_days=n_days, checkpoint=CrashingCheckpoint(root / "cp.json", crash_on))
    with pytest.raises(KeyboardInterrupt):
        env.run()
    assert json.loads(env.predlog.path.read_text().splitlines()[-1])["day"] == WEEKDAYS[crash_on - 1].isoformat()
    env2 = Env(root, phase2, n_days=n_days, checkpoint=Checkpoint(root / "cp.json"))
    return env2, env2.run()


def _assert_same_run(env, res, ref_env, ref_res):
    a, b = _final(env, res), _final(ref_env, ref_res)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2] and a[3] == b[3] and a[4] == b[4]
    pd.testing.assert_frame_equal(a[5], b[5])
    assert a[6] == b[6] and a[7] == b[7]


def _on(day, names, action):
    def handler(name, u):
        if u["asof"] == day.isoformat() and name in names:
            return action()
        return default_handler(name, u)
    return handler


def _raise_api():
    raise ApiError("503 upstream")


def test_resume_when_crashed_day_had_content_failure_that_now_succeeds(tmp_path):
    # phase 1: s3's response is bad (and cached); on resume the replay is bad but the live
    # retry succeeds, so the re-run's results differ from the logged (uncommitted) line
    env, res = _crash_then_resume(tmp_path / "a", _on(DAY6, {"s3"}, lambda: "not json"), default_handler)
    ref_env = Env(tmp_path / "ref", n_days=10)
    _assert_same_run(env, res, ref_env, ref_env.run())
    assert res.day_results[DAY6].results["s3"].status == "ok"


def test_resume_when_crashed_day_had_transient_failure_that_now_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    env, res = _crash_then_resume(tmp_path / "b", _on(DAY6, {"s3"}, _raise_api), default_handler)
    ref_env = Env(tmp_path / "ref", n_days=10)
    _assert_same_run(env, res, ref_env, ref_env.run())


def test_resume_leaves_no_stale_picks_for_agent_that_fails_on_rerun(tmp_path, monkeypatch):
    # phase 1: m1 fails transiently (not decisive), s3 is ok and its picks are recorded.
    # phase 2: m1 succeeds, so s3's inputs change (cache miss) and s3 now fails.
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    phase2 = _on(DAY6, {"s3"}, lambda: "not json")
    env, res = _crash_then_resume(tmp_path / "c", _on(DAY6, {"m1"}, _raise_api), phase2)
    assert res.day_results[DAY6].results["s3"].status == "no_opinion"
    rows = env.score_rows()
    assert rows[(rows["agent"] == "s3") & (rows["date"] == DAY6)].empty
    ref_env = Env(tmp_path / "ref", phase2, n_days=10)
    _assert_same_run(env, res, ref_env, ref_env.run())


def test_resume_refused_when_uncommitted_line_is_not_for_the_rerun_day(tmp_path):
    env = Env(tmp_path, n_days=6, checkpoint=CrashingCheckpoint(tmp_path / "cp.json", 4))
    with pytest.raises(KeyboardInterrupt):
        env.run()
    # tamper: make the checkpoint claim day 4 was committed while the log's extra line is day 4
    cp = json.loads((tmp_path / "cp.json").read_text())
    cp["last_completed_day"] = WEEKDAYS[3].isoformat()
    (tmp_path / "cp.json").write_text(json.dumps(cp))
    with pytest.raises(ResumeRefused, match="uncommitted"):
        Env(tmp_path, n_days=6, checkpoint=Checkpoint(tmp_path / "cp.json")).run()


def test_cio_content_failure_with_unrelated_transient_failure_is_ordinary_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    d = WEEKDAYS[3]

    def handler(name, u):
        if u["asof"] == d.isoformat():
            if name == "s3":
                raise ApiError("503")
            if name == "cio":
                return "not json"
        return default_handler(name, u)

    res = Env(tmp_path, handler, n_days=6).run()
    assert set(res.skipped) == {d} and "cio" in res.skipped[d]


def test_cio_transient_failure_pauses(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    d = WEEKDAYS[3]
    with pytest.raises(RunPaused) as ei:
        Env(tmp_path, _on(d, {"cio"}, _raise_api), n_days=6).run()
    assert ei.value.day == d and ei.value.failed == {"cio": "api_error"}


def test_too_few_valid_with_enough_transient_failures_pauses(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    d = WEEKDAYS[3]
    calls = {"n": 0}

    def handler(name, u):
        if u["asof"] == d.isoformat():
            if name in ("m1", "m2", "m3"):
                return "not json"   # 3 content failures
            if name in ("s1", "s2"):
                raise ApiError("503")   # 2 transient: 4 valid + 3 layer-4 = 7; +2 = 9 >= 8
        return default_handler(name, u)

    with pytest.raises(RunPaused) as ei:
        Env(tmp_path, handler, n_days=6).run()
    assert ei.value.failed["s1"] == "api_error" and ei.value.failed["m1"] == "malformed_json"


def test_too_few_valid_where_transients_cannot_reach_minimum_is_skip(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    d = WEEKDAYS[3]

    def handler(name, u):
        if u["asof"] == d.isoformat():
            if name in ("m1", "m2", "m3", "s1", "s2"):
                return "not json"   # 5 content failures: 4 valid + 3 layer-4 = 7
            if name == "s3":
                raise ApiError("503")   # +1 transient -> 3 valid + 1 + 3 = 7 < 8
        return default_handler(name, u)

    res = Env(tmp_path, handler, n_days=6).run()
    assert set(res.skipped) == {d}


def _resume_with(tmp_path, mutate):
    env = Env(tmp_path, n_days=4)
    env.run()
    env2 = Env(tmp_path, n_days=4)
    mutate(env2)
    with pytest.raises(ResumeRefused) as ei:
        env2.run()
    return str(ei.value)


def test_resume_refused_when_a_limit_changed(tmp_path):
    def mutate(e):
        e.cfg = BacktestConfig(periods=e.cfg.periods, settings=e.settings,
                               limits=limits(daily_loss_pct=0.4), anonymize=False, dry_run=False)
        e.risk = RiskLayer(e.cfg.limits, 0.005, 1.0)
    assert "limits.daily_loss_pct" in _resume_with(tmp_path, mutate)


def test_resume_refused_when_a_prompt_changed(tmp_path):
    def mutate(e):
        e.pipeline.prompts = dict(PROMPTS, s2="PROMPT<s2> be bolder")
    msg = _resume_with(tmp_path, mutate)
    assert "prompt.s2" in msg and "prompt.s1" not in msg


def test_resume_refused_when_settings_or_sectors_changed(tmp_path):
    def mutate(e):
        e.settings = settings(slippage_bps=9.0)
        e.cfg = BacktestConfig(periods=e.cfg.periods, settings=e.settings, limits=e.cfg.limits,
                               anonymize=False, dry_run=False)
    assert "slippage_bps" in _resume_with(tmp_path / "s", mutate)
    env = Env(tmp_path / "u", n_days=4)
    env.run()
    with pytest.raises(ResumeRefused, match="universe_sectors"):
        Env(tmp_path / "u", n_days=4).run(universe_sectors=dict(SECTORS, XAA="software"))


def test_cfg_limits_must_match_risk_layer(tmp_path):
    env = Env(tmp_path, n_days=4)
    env.risk = RiskLayer(limits(max_position_pct=0.5), 0.005, 1.0)
    with pytest.raises(ValueError, match="limits"):
        env.run()


def test_run_backtest_has_no_separate_llm_parameter(tmp_path):
    env = Env(tmp_path, n_days=4)
    with pytest.raises(TypeError):
        env.run(llm=env.llm)


def test_resume_leaves_no_stale_picks_for_layer4_agents_not_run_on_rerun(tmp_path, monkeypatch):
    # phase 1: m1 fails transiently (not decisive), the day trades and cro/cio/execution picks
    # are recorded. phase 2: m1 succeeds, so layers 2-3 see new inputs (cache misses) and five
    # of them now fail on content -> early skip, layer 4 never runs on the re-run.
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    phase2 = _on(DAY6, {"s1", "s2", "s3", "s4", "si1"}, lambda: "not json")
    env, res = _crash_then_resume(tmp_path / "c", _on(DAY6, {"m1"}, _raise_api), phase2)
    assert DAY6 in res.skipped and "cio" not in res.day_results[DAY6].results
    rows = env.score_rows()
    assert rows[(rows["date"] == DAY6) & rows["agent"].isin(["cro", "cio", "execution"])].empty
    ref_env = Env(tmp_path / "ref", phase2, n_days=10)
    _assert_same_run(env, res, ref_env, ref_env.run())


# ---------------------------------------------------------------------------- day history

def test_day_history_records_weights_state_and_fills_of_committed_days(tmp_path):
    from atlas.backtest.runner import read_day_history
    env = Env(tmp_path, n_days=6)
    seen = []
    real_run_day = env.pipeline.run_day

    def spy_run_day(asof, snap, weights, state, **kw):
        seen.append((asof, dict(weights), json.loads(json.dumps(state))))
        return real_run_day(asof, snap, weights, state, **kw)

    env.pipeline.run_day = spy_run_day
    env.run(history_path=tmp_path / "weights_history.jsonl")
    hist = read_day_history(tmp_path / "weights_history.jsonl", env.checkpoint)
    assert sorted(hist) == WEEKDAYS[:6]
    for asof, weights, state in seen:
        assert hist[asof]["weights"] == weights
        assert hist[asof]["state"] == state
    for decision, _orders, fills, _dropped in env.broker.submits:
        assert hist[decision]["fills"] == [
            {"ticker": f.order.ticker, "side": f.order.side, "shares": f.order.shares,
             "date": f.date.isoformat(), "price": f.price, "commission": f.commission} for f in fills]


def test_day_history_after_crash_and_resume_equals_uninterrupted(tmp_path):
    from atlas.backtest.runner import read_day_history
    ref = Env(tmp_path / "ref", n_days=8)
    ref.run(history_path=tmp_path / "ref" / "h.jsonl")
    want = read_day_history(tmp_path / "ref" / "h.jsonl", ref.checkpoint)
    root = tmp_path / "crash"
    env = Env(root, n_days=8, checkpoint=CrashingCheckpoint(root / "cp.json", 5))
    with pytest.raises(KeyboardInterrupt):
        env.run(history_path=root / "h.jsonl")
    # the crashed day's line is on disk but not committed: readers ignore it
    assert WEEKDAYS[4] not in read_day_history(root / "h.jsonl", Checkpoint(root / "cp.json"))
    env2 = Env(root, n_days=8, checkpoint=Checkpoint(root / "cp.json"))
    env2.run(history_path=root / "h.jsonl")
    assert read_day_history(root / "h.jsonl", env2.checkpoint) == want


def test_day_history_survives_a_torn_line(tmp_path):
    from atlas.backtest.runner import read_day_history
    env = Env(tmp_path, n_days=3)
    h = tmp_path / "h.jsonl"
    h.write_text('{"day": "2024-01-0')  # crash mid-write of an earlier attempt
    env.run(history_path=h)
    assert sorted(read_day_history(h, env.checkpoint)) == WEEKDAYS[:3]


# ---------------------------------------------------------------------------- final review fixes

def test_fatal_api_error_propagates_is_not_a_pause_and_keeps_the_checkpoint(tmp_path):
    from atlas.engine.llm import FatalApiError

    def handler(name, u):
        if u["asof"] == WEEKDAYS[2].isoformat():
            raise FatalApiError(401, "authentication_error", "invalid x-api-key")
        return default_handler(name, u)

    env = Env(tmp_path, handler=handler, price=(1.0, 1.0))
    with pytest.raises(FatalApiError):
        env.run()
    raw = env.checkpoint.load()
    assert raw["last_completed_day"] == WEEKDAYS[1].isoformat()
    assert raw["spent_usd"] == pytest.approx(env.llm.spent_usd) and raw["spent_usd"] > 0
    assert env.predlog.head()[0] == 2  # nothing of the rejected day was logged


class CountingPrices(FakePrices):
    def __init__(self, gaps=(), drop=()):
        super().__init__(gaps)
        self.calls, self.drop = [], set(drop)

    def fetch(self, tickers, start, end):
        self.calls.append((tuple(tickers), start, end))
        df = super().fetch([t for t in tickers if t not in self.drop], start, end)
        return df


def test_missing_ticker_fails_preflight_before_any_llm_call(tmp_path):
    from atlas.backtest.runner import MissingPriceData
    env = Env(tmp_path)
    env.store = PriceStore(CountingPrices(drop={"XBB"}), FakeMacro(), tmp_path / "p2")
    with pytest.raises(MissingPriceData) as ei:
        env.run()
    assert ei.value.tickers == ["XBB"] and "no price data for: XBB over" in str(ei.value)
    assert env.calls == [] and not env.checkpoint.path.exists()


def test_run_prefetches_the_whole_range_once_not_daily(tmp_path):
    prov = CountingPrices()
    env = Env(tmp_path, n_days=12)
    env.store = PriceStore(prov, FakeMacro(), tmp_path / "p2")
    env.broker._store = env.store
    env.run()
    multi = [i for i, c in enumerate(prov.calls) if len(c[0]) > 1]
    assert len(multi) == 1 and set(prov.calls[multi[0]][0]) >= set(UNIVERSE)
    # before it: the SPY calendar of each period; after it, only forward fetches past the run's
    # last day (scoring / next-open fills at the tail), never a daily snapshot re-download
    assert all(c[0] == ("SPY",) for c in prov.calls[:multi[0]])
    last = WEEKDAYS[11]
    assert all(end > last for _, _, end in prov.calls[multi[0] + 1:])


def test_prefetch_leaves_the_run_identical(tmp_path):
    a, b = Env(tmp_path / "a", n_days=9), Env(tmp_path / "b", n_days=9)
    ra = a.run()
    import atlas.backtest.runner as runner_mod
    orig = runner_mod.preflight_prices
    try:
        runner_mod.preflight_prices = lambda *x, **k: None  # the pre-fix behaviour: no pre-fetch
        rb = b.run()
    finally:
        runner_mod.preflight_prices = orig
    pd.testing.assert_series_equal(ra.equity, rb.equity)
    assert a.predlog.path.read_bytes() == b.predlog.path.read_bytes()
    assert sorted(a.calls) == sorted(b.calls)  # identical inputs (snapshots) to every agent


# ---------------------------------------------------------------------------- fingerprint compat

def _old_fingerprint_fn():
    """The ``_fingerprint`` of commit 9ce69b3 (before AgentSpec.model / spoke_model existed),
    executed from its own source so the test cannot drift with the current implementation."""
    import subprocess
    src = subprocess.run(["git", "show", "9ce69b3:atlas/backtest/runner.py"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parents[2], check=True).stdout
    start = src.index("def _sha(")
    end = src.index("# ----", src.index("def _fingerprint("))
    ns = {k: v for k, v in vars(runner_mod).items() if not k.startswith("__")}
    exec(compile(src[start:end], "old_runner_fingerprint", "exec"), ns)
    old = ns["_fingerprint"]

    class _OldSpec:  # an AgentSpec as it was then: no ``model`` field
        def __init__(self, sp):
            self.sp = sp

        def model_dump(self, **kw):
            d = self.sp.model_dump(**kw)
            d.pop("model")
            return d

    def wrapped(cfg, pipeline, *a):
        shim = type("P", (), {"specs": {n: _OldSpec(sp) for n, sp in pipeline.specs.items()},
                              "prompts": pipeline.prompts})()
        return old(cfg, shim, *a)
    return wrapped


def test_fingerprint_is_identical_to_pre_hubspoke_algorithm(tmp_path):
    env = Env(tmp_path, n_days=4)
    args = (env.cfg, env.pipeline, INITIAL, list(UNIVERSE), SECTORS, 60, ())
    assert runner_mod._fingerprint(*args) == _old_fingerprint_fn()(*args)


def test_old_checkpoint_still_resumes_with_new_code(tmp_path, monkeypatch):
    env = Env(tmp_path, n_days=4)
    env.run()
    # rewrite the checkpoint's fingerprint as the old algorithm produced it
    monkeypatch.setattr(runner_mod, "_fingerprint", _old_fingerprint_fn())
    env_old = Env(tmp_path, n_days=4)
    cp = json.loads(env_old.checkpoint.path.read_text())
    cp["fingerprint"] = runner_mod._fingerprint(env_old.cfg, env_old.pipeline, INITIAL, list(UNIVERSE),
                                                SECTORS, 60, ())
    env_old.checkpoint.path.write_text(json.dumps(cp))
    monkeypatch.undo()
    res = Env(tmp_path, n_days=4).run()  # must not raise ResumeRefused
    assert res.skipped == {}


def test_resume_refused_when_an_agent_model_changed(tmp_path):
    def mutate(e):
        specs = {n: sp.model_copy() for n, sp in e.pipeline.specs.items()}
        specs["s2"] = specs["s2"].model_copy(update={"model": "other-model"})
        e.pipeline.specs = specs
    assert "agents" in _resume_with(tmp_path, mutate)


def test_resume_refused_when_spoke_model_set(tmp_path):
    def mutate(e):
        e.settings = settings(spoke_model="spoke-m", spoke_usd_per_mtok_input=1.0, spoke_usd_per_mtok_output=2.0)
        e.cfg = BacktestConfig(periods=e.cfg.periods, settings=e.settings, limits=e.cfg.limits,
                               anonymize=False, dry_run=False)
    assert "settings.spoke_model" in _resume_with(tmp_path, mutate)
