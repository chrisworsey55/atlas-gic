import json
import threading

from atlas.engine.failures import FailureLog
from atlas.engine.llm import CachedLLM, StubClient
from atlas.engine.pipeline import Pipeline
from tests.engine.test_pipeline import (ASOF, PROMPTS, SPECS, STATE, UNIVERSE, WEIGHTS, default_handler,
                                        snapshot)

HUB_KINDS = {"cro", "cio", "execution"}
MODELS = {s.name: ("hub-m" if s.kind in HUB_KINDS else "spoke-m") for s in SPECS if s.kind != "control"}


class Routed:
    def __init__(self, tmp_path, with_llms=True, models=MODELS, only=("hub-m", "spoke-m")):
        self.served = []  # (client label, agent)
        self.lock = threading.Lock()

        def client(label):
            def fn(system, user):
                name = system[len("PROMPT<"):-1]
                with self.lock:
                    self.served.append((label, name))
                return default_handler(name, json.loads(user))
            return StubClient(fn)

        self.hub = CachedLLM(client("hub"), tmp_path / "hub.db", "hub-m", (0.0, 0.0), 1e9)
        self.spoke = CachedLLM(client("spoke"), tmp_path / "spoke.db", "spoke-m", (0.0, 0.0), 1e9)
        self.default = CachedLLM(client("default"), tmp_path / "def.db", "def-m", (0.0, 0.0), 1e9)
        llms = {m: {"hub-m": self.hub, "spoke-m": self.spoke}[m] for m in only}
        self.pipe = Pipeline(SPECS, PROMPTS, self.default, FailureLog(tmp_path / "f.db"), UNIVERSE, 3,
                             llms=llms if with_llms else None, models=models)

    def labels(self, agent):
        return {c for c, n in self.served if n == agent}


def test_each_agent_call_goes_to_its_models_client(tmp_path):
    r = Routed(tmp_path)
    res = r.pipe.run_day(ASOF, snapshot(), WEIGHTS, STATE)
    assert res.skipped_reason is None
    for s in SPECS:
        if s.kind == "control":
            continue
        want = "hub" if s.kind in HUB_KINDS else "spoke"
        assert r.labels(s.name) == {want}, s.name
    assert "default" not in {c for c, _ in r.served}


def test_run_single_routes_like_run_day(tmp_path):
    r = Routed(tmp_path)
    for name in ("m1", "cio"):
        r.pipe.run_single(name, ASOF, snapshot(), WEIGHTS, STATE, {})
    assert r.labels("m1") == {"spoke"} and r.labels("cio") == {"hub"}


def test_missing_model_mapping_falls_back_to_default_llm(tmp_path):
    # agent absent from `models`, and a model with no client in `llms`
    models = dict(MODELS)
    del models["m1"]
    r = Routed(tmp_path, models=models, only=("hub-m",))
    r.pipe.run_single("m1", ASOF, snapshot(), WEIGHTS, STATE, {})
    r.pipe.run_single("m2", ASOF, snapshot(), WEIGHTS, STATE, {})  # spoke-m has no client
    r.pipe.run_single("cio", ASOF, snapshot(), WEIGHTS, STATE, {})
    assert r.labels("m1") == {"default"} and r.labels("m2") == {"default"}
    assert r.labels("cio") == {"hub"}


def test_no_routing_uses_single_llm(tmp_path):
    r = Routed(tmp_path, with_llms=False)
    r.pipe.run_day(ASOF, snapshot(), WEIGHTS, STATE)
    assert {c for c, _ in r.served} == {"default"}


def test_models_none_disables_routing_even_with_llms(tmp_path):
    r = Routed(tmp_path, models=None)
    r.pipe.run_single("m1", ASOF, snapshot(), WEIGHTS, STATE, {})
    assert r.labels("m1") == {"default"}


def test_cache_keys_differ_between_models_for_identical_prompts(tmp_path):
    r = Routed(tmp_path)
    assert r.hub._key("S", "U") != r.spoke._key("S", "U")
    # SPEC: the same prompt through each model's client is a live call each (no shared cache hit)
    for llm in (r.hub, r.spoke):
        _, replayed = llm.call_with_meta("PROMPT<m1>", "{}", max_tokens=100)
        assert not replayed
    assert [c for c, _ in r.served] == ["hub", "spoke"]
