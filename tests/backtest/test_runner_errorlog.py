"""run_backtest writes its notable events to the optional ErrorLog."""
import pytest

import atlas.engine.agent as agent_mod
from atlas.backtest.checkpoint import Checkpoint
from atlas.backtest.runner import RunPaused
from atlas.engine.llm import ApiError, FatalApiError
from atlas.errorlog import ErrorLog, read_events
from atlas.risk.layer import RiskLayer
from tests.backtest.test_runner import (
    WEEKDAYS, CrashingCheckpoint, Env, HaltOnDay, default_handler,
)

FIVE = ("m1", "m2", "m3", "s1", "s2")


def test_skipped_day_event_lists_failed_agents_and_codes(tmp_path):
    bad_day = WEEKDAYS[3]

    def handler(name, u):
        if u["asof"] == bad_day.isoformat() and name in FIVE:
            return "this is not json"
        return default_handler(name, u)

    path = tmp_path / "errors.log"
    env = Env(tmp_path, handler, n_days=6)
    res = env.run(error_log=ErrorLog(path))
    (ev,) = read_events(path, kind="day_skipped")
    assert ev["level"] == "warning" and ev["day"] == bad_day.isoformat()
    assert ev["fields"]["failed"] == {n: "malformed_json" for n in FIVE}
    assert ev["message"].startswith(res.skipped[bad_day][:40])


def test_transient_pause_writes_run_paused_before_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    bad_day = WEEKDAYS[3]

    def handler(name, u):
        if u["asof"] == bad_day.isoformat() and name in FIVE:
            raise ApiError("503 upstream")
        return default_handler(name, u)

    path = tmp_path / "errors.log"
    env = Env(tmp_path, handler, n_days=6)
    with pytest.raises(RunPaused):
        env.run(error_log=ErrorLog(path))
    (ev,) = read_events(path, kind="run_paused")
    assert ev["day"] == bad_day.isoformat()
    assert ev["fields"]["pause_kind"] == "transient"
    assert ev["fields"]["failed"] == {n: "api_error" for n in FIVE}
    assert "transient" in ev["message"]


def test_budget_pause_writes_budget_exceeded(tmp_path):
    price = (1.0, 0.0)
    probe = Env(tmp_path / "probe", n_days=6, price=price)
    probe.run()
    path = tmp_path / "errors.log"
    env = Env(tmp_path / "run", n_days=6, price=price)
    env.pipeline = env._pipeline(budget=probe.llm.spent_usd * 0.5)
    with pytest.raises(RunPaused, match="budget"):
        env.run(error_log=ErrorLog(path))
    (ev,) = read_events(path, kind="budget_exceeded")
    assert ev["level"] == "error" and ev["day"]
    (paused,) = read_events(path, kind="run_paused")
    assert paused["fields"]["pause_kind"] == "budget"


def test_fatal_api_error_event_is_sanitised(tmp_path):
    def handler(name, u):
        if u["asof"] == WEEKDAYS[2].isoformat():
            raise FatalApiError(401, "authentication_error", "bad key sk-ant-CANARY123abc")
        return default_handler(name, u)

    path = tmp_path / "errors.log"
    env = Env(tmp_path, handler=handler, price=(1.0, 1.0))
    with pytest.raises(FatalApiError):
        env.run(error_log=ErrorLog(path))
    (ev,) = read_events(path, kind="fatal_api_error")
    assert ev["day"] == WEEKDAYS[2].isoformat() and ev["fields"]["exc_type"] == "FatalApiError"
    assert "CANARY123" not in path.read_text()


def test_risk_layer_exception_is_logged_and_sends_no_orders(tmp_path, monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("risk bug")

    monkeypatch.setattr(RiskLayer, "apply", boom)
    path = tmp_path / "errors.log"
    env = Env(tmp_path, n_days=4)
    env.run(error_log=ErrorLog(path))
    assert env.broker.submits == []
    evs = read_events(path, kind="risk_error")
    assert len(evs) == 4 and all(e["fields"]["exc_type"] == "RuntimeError" for e in evs)
    assert evs[0]["message"] == "risk bug" and evs[0]["day"] == WEEKDAYS[0].isoformat()


def test_kill_switch_writes_halted_event_once(tmp_path):
    path = tmp_path / "errors.log"
    env = Env(tmp_path, n_days=9)
    env.risk = HaltOnDay(env.risk, 3)
    env.run(error_log=ErrorLog(path))
    (ev,) = read_events(path, kind="halted")
    assert ev["level"] == "error" and ev["day"] == WEEKDAYS[2].isoformat()
    assert ev["fields"]["halt_reason"] == "test kill switch"


def test_resumed_run_appends_to_the_same_log(tmp_path):
    path = tmp_path / "errors.log"
    env = Env(tmp_path, n_days=9, checkpoint=CrashingCheckpoint(tmp_path / "cp.json", 6))
    env.risk = HaltOnDay(env.risk, 3)
    with pytest.raises(KeyboardInterrupt):
        env.run(error_log=ErrorLog(path))
    assert len(read_events(path, kind="halted")) == 1
    Env(tmp_path, n_days=9, checkpoint=Checkpoint(tmp_path / "cp.json")).run(
        error_log=ErrorLog(path))
    assert len(read_events(path, kind="halted")) == 1  # resume does not re-announce the halt
    assert read_events(path)[0]["kind"] == "halted"  # first run's events are still there
