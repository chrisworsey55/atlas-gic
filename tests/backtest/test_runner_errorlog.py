"""run_backtest writes its notable events to the optional ErrorLog."""
import json

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
    assert read_events(path, kind="run_paused") == []  # an ordinary skip is not a pause


class OrderedCheckpoint(Checkpoint):
    def __init__(self, path, order):
        super().__init__(path)
        self.order = order

    def save(self, state):
        super().save(state)
        self.order.append("checkpoint.save")


class OrderedLog(ErrorLog):
    def __init__(self, path, order):
        super().__init__(path)
        self.order = order

    def event(self, kind, *a, **k):
        self.order.append(f"event:{kind}")
        super().event(kind, *a, **k)


def _transient_handler(bad_day):
    def handler(name, u):
        if u["asof"] == bad_day.isoformat() and name in FIVE:
            raise ApiError("503 upstream")
        return default_handler(name, u)
    return handler


def test_transient_pause_writes_run_paused_before_raising(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    bad_day = WEEKDAYS[3]
    path = tmp_path / "errors.log"
    order = []
    env = Env(tmp_path, _transient_handler(bad_day), n_days=6,
              checkpoint=OrderedCheckpoint(tmp_path / "cp.json", order))
    seen = None
    with pytest.raises(RunPaused):
        try:
            env.run(error_log=OrderedLog(path, order))
        except RunPaused:
            seen = read_events(path, kind="run_paused")  # already on disk inside the handler
            raise
    assert len(seen) == 1
    (ev,) = read_events(path, kind="run_paused")
    assert ev["day"] == bad_day.isoformat()
    assert ev["fields"]["pause_kind"] == "transient"
    assert ev["fields"]["failed"] == {n: "api_error" for n in FIVE}
    assert "transient" in ev["message"]
    assert read_events(path, kind="day_skipped") == []  # a pause day is not also a skipped day
    assert order[-2:] == ["checkpoint.save", "event:run_paused"]  # checkpoint first, then the event


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


def test_fatal_api_error_saves_the_spend_before_logging(tmp_path):
    def handler(name, u):
        if u["asof"] == WEEKDAYS[2].isoformat():
            raise FatalApiError(401, "authentication_error", "invalid")
        return default_handler(name, u)

    order = []
    env = Env(tmp_path, handler=handler, price=(1.0, 1.0),
              checkpoint=OrderedCheckpoint(tmp_path / "cp.json", order))
    with pytest.raises(FatalApiError):
        env.run(error_log=OrderedLog(tmp_path / "errors.log", order))
    assert order[-2:] == ["checkpoint.save", "event:fatal_api_error"]


def test_replayed_cached_failure_writes_no_new_call_failure_event(tmp_path):
    from tests.engine.test_agent import good, run, setup
    path = tmp_path / "errors.log"
    client, llm, log = setup(tmp_path, ["bad", "bad", "bad again"])
    log.error_log = ErrorLog(path)
    run(llm, log)
    assert len(read_events(path, kind="call_failure")) == 2
    run(llm, log)  # attempt 1 replays the cache (not logged); only the live retry is new
    assert len(read_events(path, kind="call_failure")) == 3

    picks = [{"ticker": "ZZZZ", "direction": "long", "conviction": 0.9, "horizon_days": 20}]
    client2, llm2, log2 = setup(tmp_path / "b", [good(picks)])
    log2.error_log = ErrorLog(tmp_path / "b.log")
    run(llm2, log2)
    run(llm2, log2)  # cached good day: the drop is not re-logged
    assert len(read_events(tmp_path / "b.log", kind="dropped_pick")) == 1


def _fingerprint_run(env, res):
    return (res.equity.to_json(), env.score_rows().to_json(), env.predlog.path.read_bytes(),
            json.loads(env.checkpoint.path.read_text()))


def test_run_with_error_log_equals_run_without(tmp_path):
    a = Env(tmp_path / "a", n_days=8)
    ra = a.run()
    b = Env(tmp_path / "b", n_days=8)
    rb = b.run(error_log=ErrorLog(tmp_path / "errors.log"))
    assert _fingerprint_run(a, ra) == _fingerprint_run(b, rb)
    assert "errors.log" not in b.checkpoint.path.read_text()


@pytest.fixture
def broken_log(monkeypatch):
    def boom(self, *a, **k):
        raise RuntimeError("log is broken")
    monkeypatch.setattr(ErrorLog, "event", boom)
    monkeypatch.setattr(ErrorLog, "exception", boom)


def test_broken_error_log_does_not_change_a_normal_run(tmp_path, broken_log):
    a = Env(tmp_path / "a", n_days=8)
    b = Env(tmp_path / "b", n_days=8)
    b.risk = HaltOnDay(b.risk, 3)  # also exercises the halted call site
    a.risk = HaltOnDay(a.risk, 3)
    ra = a.run()
    rb = b.run(error_log=ErrorLog(tmp_path / "errors.log"))
    assert _fingerprint_run(a, ra) == _fingerprint_run(b, rb)


def test_broken_error_log_transient_pause_still_raises_runpaused(tmp_path, monkeypatch, broken_log):
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    bad_day = WEEKDAYS[3]
    env = Env(tmp_path, _transient_handler(bad_day), n_days=6)
    with pytest.raises(RunPaused) as ei:
        env.run(error_log=ErrorLog(tmp_path / "errors.log"))
    assert ei.value.day == bad_day and ei.value.kind == "transient"
    assert env.checkpoint.load()["last_completed_day"] == WEEKDAYS[2].isoformat()


def test_broken_error_log_budget_pause_still_raises_runpaused(tmp_path, broken_log):
    price = (1.0, 0.0)
    probe = Env(tmp_path / "probe", n_days=6, price=price)
    probe.run()
    env = Env(tmp_path / "run", n_days=6, price=price)
    env.pipeline = env._pipeline(budget=probe.llm.spent_usd * 0.5)
    with pytest.raises(RunPaused, match="budget"):
        env.run(error_log=ErrorLog(tmp_path / "errors.log"))
    assert env.checkpoint.load()["spent_usd"] > 0


def test_broken_error_log_fatal_error_still_saves_spend_and_reraises(tmp_path, broken_log):
    def handler(name, u):
        if u["asof"] == WEEKDAYS[2].isoformat():
            raise FatalApiError(401, "authentication_error", "invalid")
        return default_handler(name, u)

    env = Env(tmp_path, handler=handler, price=(1.0, 1.0))
    with pytest.raises(FatalApiError):
        env.run(error_log=ErrorLog(tmp_path / "errors.log"))
    raw = env.checkpoint.load()
    assert raw["last_completed_day"] == WEEKDAYS[1].isoformat()
    assert raw["spent_usd"] == pytest.approx(env.llm.spent_usd) and raw["spent_usd"] > 0


def test_broken_error_log_risk_exception_still_sends_no_orders(tmp_path, monkeypatch, broken_log):
    monkeypatch.setattr(RiskLayer, "apply", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    env = Env(tmp_path, n_days=4)
    res = env.run(error_log=ErrorLog(tmp_path / "errors.log"))
    assert env.broker.submits == [] and len(res.risk_errors) == 4


def test_broken_error_log_skipped_day_still_skips(tmp_path, broken_log):
    bad_day = WEEKDAYS[3]

    def handler(name, u):
        if u["asof"] == bad_day.isoformat() and name in FIVE:
            return "not json"
        return default_handler(name, u)

    res = Env(tmp_path, handler, n_days=6).run(error_log=ErrorLog(tmp_path / "errors.log"))
    assert set(res.skipped) == {bad_day}
