import json
import threading
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from atlas.config import AgentSpec, load_agents
from atlas.data.store import Snapshot
from atlas.engine.failures import FailureLog
from atlas.engine.llm import CachedLLM, StubClient
from atlas.engine.pipeline import Pipeline, aggregate_regime, orders_from_targets
from atlas.schemas import AgentOutput, AgentResult, FailureReason

ASOF = date(2024, 3, 1)
UNIVERSE = {"AAA", "BBB", "SPY"}
SPECS = [
    AgentSpec(name="m1", layer=1, kind="macro", prompt_file="x"),
    AgentSpec(name="m2", layer=1, kind="macro", prompt_file="x"),
    AgentSpec(name="m3", layer=1, kind="macro", prompt_file="x"),
    AgentSpec(name="s1", layer=2, kind="sector", prompt_file="x"),
    AgentSpec(name="si1", layer=3, kind="superinvestor", prompt_file="x"),
    AgentSpec(name="cro", layer=4, kind="cro", prompt_file="x"),
    AgentSpec(name="cio", layer=4, kind="cio", prompt_file="x"),
    AgentSpec(name="execution", layer=4, kind="execution", prompt_file="x"),
    AgentSpec(name="control", layer=4, kind="control", prompt_file="x"),
]
PROMPTS = {s.name: f"PROMPT<{s.name}>" for s in SPECS}
STATE = {"cash": 50_000.0, "positions": {"AAA": 100}, "equity": 100_000.0}
WEIGHTS = {s.name: 1.0 for s in SPECS}


def out(agent, view="bullish", picks=None, conf=0.8):
    return json.dumps({"agent": agent, "date": "2024-03-01", "view": view, "confidence": conf,
                       "picks": picks if picks is not None else
                       [{"ticker": "AAA", "direction": "long", "conviction": 0.5, "horizon_days": 20}]})


def snapshot():
    rows = []
    for t, px in (("AAA", 100.0), ("BBB", 50.0), ("SPY", 400.0)):
        for d, p in ((date(2024, 2, 29), px * 0.99), (ASOF, px)):
            rows.append({"date": d, "ticker": t, "open": p, "high": p, "low": p, "close": p,
                         "adj_close": p, "volume": 1})
    return Snapshot(asof=ASOF, prices=pd.DataFrame(rows),
                    macro={"DGS10": pd.Series([4.0, 4.1], index=[date(2024, 2, 29), ASOF])})


class Harness:
    def __init__(self, tmp_path, handler, specs=SPECS, min_valid=3):
        self.calls = []  # (agent, user dict)
        self.lock = threading.Lock()
        self.handler = handler

        def fn(system, user):
            name = system[len("PROMPT<"):-1]
            with self.lock:
                self.calls.append((name, json.loads(user)))
            return self.handler(name, json.loads(user))

        llm = CachedLLM(StubClient(fn), tmp_path / "c.db", "m", (0.0, 0.0), 1e9)
        self.pipe = Pipeline(specs, PROMPTS, llm, FailureLog(tmp_path / "f.db"), UNIVERSE, min_valid)

    def run(self, weights=WEIGHTS, state=STATE):
        return self.pipe.run_day(ASOF, snapshot(), weights, state)

    def users(self, name):
        return [u for n, u in self.calls if n == name]


def default_handler(name, user):
    if name == "execution":
        return out("execution", picks=[{"ticker": "AAA", "direction": "long", "conviction": 0.2,
                                        "horizon_days": 5}])
    return out(name)


def test_layers_run_in_order_1_to_4(tmp_path):
    h = Harness(tmp_path, default_handler)
    res = h.run()
    order = [n for n, _ in h.calls]
    layer = {s.name: s.layer for s in SPECS}
    seq = [layer[n] for n in order]
    assert seq == sorted(seq)
    assert order[-3:] == ["cro", "cio", "execution"]
    assert "control" not in order
    assert res.skipped_reason is None and res.regime == "RISK_ON"


def test_macro_agents_in_one_layer_run_concurrently(tmp_path):
    barrier = threading.Barrier(3, timeout=5)

    def handler(name, user):
        if name.startswith("m"):
            barrier.wait()  # raises BrokenBarrierError if the three macro agents are not concurrent
        return default_handler(name, user)

    h = Harness(tmp_path, handler)
    res = h.run()
    assert all(res.results[n].status == "ok" for n in ("m1", "m2", "m3"))


def res_ok(agent, view, conf=1.0):
    return AgentResult(agent=agent, date=ASOF, output=AgentOutput(
        agent=agent, date=ASOF, view=view, confidence=conf))


def test_regime_weighted_vote():
    a, b = res_ok("a", "bullish"), res_ok("b", "bearish")
    assert aggregate_regime([a, b], {"a": 1.0, "b": 1.0}) == "NEUTRAL"
    assert aggregate_regime([a, b], {"a": 1.5, "b": 1.0}) == "RISK_ON"
    assert aggregate_regime([a, b], {"a": 1.0, "b": 1.5}) == "RISK_OFF"
    assert aggregate_regime([], {}) == "NEUTRAL"
    failed = AgentResult(agent="c", date=ASOF)
    assert aggregate_regime([failed], {"c": 9.0}) == "NEUTRAL"
    # confidence scales the vote
    assert aggregate_regime([res_ok("a", "bullish", 0.2), b], {"a": 1.0, "b": 1.0}) == "RISK_OFF"


def test_day_skipped_when_valid_agents_below_minimum(tmp_path):
    def handler(name, user):
        if name in ("m1", "m2", "s1", "si1"):
            return "not json"
        return default_handler(name, user)

    h = Harness(tmp_path, handler, min_valid=5)
    res = h.run()
    assert res.proposed_orders is None
    for n in ("m1", "m2", "s1", "si1"):
        assert n in res.skipped_reason
    assert FailureReason.MALFORMED_JSON.value in res.skipped_reason
    assert "m3" not in res.skipped_reason


def test_failed_agent_output_not_passed_to_next_layer(tmp_path):
    def handler(name, user):
        if name == "m1":  # schema-invalid (confidence 9) but carries a marker
            return json.dumps({"agent": "m1", "date": "2024-03-01", "view": "bullish", "confidence": 9,
                               "picks": [{"ticker": "AAA", "direction": "long", "conviction": 0.5,
                                          "horizon_days": 5, "rationale": "LEAKMARK"}]})
        return default_handler(name, user)

    h = Harness(tmp_path, handler)
    res = h.run()
    assert res.results["m1"].status == "no_opinion"
    for name in ("s1", "si1", "cro", "cio", "execution"):
        for u in h.users(name):
            assert "LEAKMARK" not in json.dumps(u)
            assert "m1" not in json.dumps(u["inputs"].get("views", []))
    cio_views = {v["agent"]: v for v in h.users("cio")[0]["inputs"]["views"]}
    assert "m1" not in cio_views and "m2" in cio_views


def test_execution_agent_receives_cio_targets_and_equity(tmp_path):
    def handler(name, user):
        if name == "cio":
            return out("cio", picks=[{"ticker": "BBB", "direction": "long", "conviction": 0.3,
                                      "horizon_days": 10}])
        return default_handler(name, user)

    w = dict(WEIGHTS, m1=2.5)
    h = Harness(tmp_path, handler)
    h.run(weights=w)
    ex = h.users("execution")[0]["inputs"]
    assert ex["cio_targets"] == [{"ticker": "BBB", "direction": "long", "conviction": 0.3,
                                  "horizon_days": 10, "rationale": ""}]
    assert ex["state"] == STATE
    assert ex["last_close"]["BBB"] == 50.0
    cio = {v["agent"]: v for v in h.users("cio")[0]["inputs"]["views"]}
    assert cio["m1"]["weight"] == 2.5 and cio["m2"]["weight"] == 1.0


def test_run_day_converts_targets_to_integer_share_orders(tmp_path):
    def handler(name, user):
        if name == "execution":
            return out("execution", picks=[
                {"ticker": "AAA", "direction": "long", "conviction": 0.05, "horizon_days": 5},
                {"ticker": "BBB", "direction": "long", "conviction": 0.1, "horizon_days": 5}])
        return default_handler(name, user)

    res = Harness(tmp_path, handler).run()
    # AAA: target 5000/100=50 shares, hold 100 -> sell 50; BBB: 10000/50=200 -> buy 200
    assert res.proposed_orders == [{"ticker": "AAA", "side": "sell", "shares": 50},
                                   {"ticker": "BBB", "side": "buy", "shares": 200}]


def test_orders_from_targets_rules():
    eq, px = 10_000.0, {"A": 33.0, "B": 10.0, "C": 7.0, "D": 20.0}
    pos = {"A": 10, "B": 100, "C": 4}
    targets = {"A": 0.5, "B": 0.0, "C": 0.0028, "D": 0.0}
    orders = orders_from_targets(targets, pos, eq, px)
    # A: floor(5000/33)=151 -> buy 141; B exit -> sell 100; C: floor(28/7)=4 == held -> none; D none
    assert orders == [{"ticker": "B", "side": "sell", "shares": 100},
                      {"ticker": "A", "side": "buy", "shares": 141}]
    assert all(isinstance(o["shares"], int) and o["shares"] > 0 for o in orders)
    assert orders_from_targets({"A": 0.5}, {}, eq, {}) == []  # no price -> no order


def test_exit_via_direction_none(tmp_path):
    def handler(name, user):
        if name == "execution":
            return out("execution", picks=[
                {"ticker": "AAA", "direction": "none", "conviction": 0.0, "horizon_days": 5}])
        return default_handler(name, user)

    res = Harness(tmp_path, handler).run()
    assert res.proposed_orders == [{"ticker": "AAA", "side": "sell", "shares": 100}]


def test_no_orders_when_cio_or_execution_failed(tmp_path):
    for bad in ("cio", "execution"):
        def handler(name, user, bad=bad):
            return "garbage" if name == bad else default_handler(name, user)

        res = Harness(tmp_path, handler).run()
        assert res.proposed_orders is None
        assert bad in res.skipped_reason


def test_run_single_matches_run_day(tmp_path):
    h = Harness(tmp_path, default_handler)
    day = h.run()
    n_before = len(h.calls)
    for name in ("m2", "s1", "si1", "cro", "cio", "execution"):
        single = h.pipe.run_single(name, ASOF, snapshot(), WEIGHTS, STATE, day.results)
        assert single == day.results[name]
    # identical inputs -> cache hits, no new LLM calls
    assert len(h.calls) == n_before


def test_run_single_prompt_override(tmp_path):
    h = Harness(tmp_path, default_handler)
    day = h.run()

    def alt(system, user):
        assert system == "ALT PROMPT"
        return out("s1", view="bearish", picks=[])

    h.pipe.llm.client = StubClient(alt)
    r = h.pipe.run_single("s1", ASOF, snapshot(), WEIGHTS, STATE, day.results, prompt_override="ALT PROMPT")
    assert r.output.view == "bearish"


def test_control_agent_gets_only_date_and_universe(tmp_path):
    h = Harness(tmp_path, default_handler)
    r = h.pipe.run_single("control", ASOF, snapshot(), WEIGHTS, STATE, {})
    assert r.status == "ok"
    u = h.users("control")[0]
    assert u == {"asof": "2024-03-01", "inputs": {"universe": ["AAA", "BBB", "SPY"]}}


def test_prompt_file_exists_for_every_agent():
    import atlas
    root = Path(atlas.__file__).parent
    for spec in load_agents(root / "config" / "agents.yaml"):
        text = (root / spec.prompt_file).read_text(encoding="utf-8")
        assert text.strip(), spec.name
        assert "JSON" in text and "picks" in text and "horizon_days" in text, spec.name
    assert (root / "prompts" / "control_date_only.md").read_text().strip()
