"""Deterministic scripted stub model for the golden tests (fixtures/stub_script.json).

The agent is identified by the first line of its system prompt ("# <agent name>", as in
atlas/prompts/*.md); the day by the user message's "asof". Same (agent, day) -> same text.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path

SCRIPT = json.loads((Path(__file__).parent / "fixtures" / "stub_script.json").read_text())


class SimulatedCrash(BaseException):
    """Raised by the stub to simulate the process dying mid-day. A BaseException, like KeyboardInterrupt:
    a dying process is not an `Exception` that the CLI catch-all may turn into an exit code."""


def agent_of(system: str) -> str:
    first = system.strip().splitlines()[0] if system.strip() else ""
    return first.lstrip("#").strip()


def respond(agent: str, asof: str) -> str:
    cfg = SCRIPT["agents"].get(agent, SCRIPT["default"])
    seed = int(hashlib.sha256(f"{agent}|{asof}".encode()).hexdigest(), 16)
    tickers = cfg["tickers"]
    start = seed % len(tickers)
    picks = []
    for j in range(min(cfg["n_picks"], len(tickers))):
        bits = seed >> (8 * (j + 1))
        t = tickers[(start + j) % len(tickers)]
        if cfg.get("targets"):
            lv = SCRIPT["target_levels"]
            picks.append({"ticker": t, "direction": "long", "conviction": lv[bits % len(lv)],
                          "horizon_days": 1, "rationale": "stub target"})
        else:
            lv = SCRIPT["conv_levels"]
            short = cfg.get("allow_short") and bits % 3 == 0
            picks.append({"ticker": t, "direction": "short" if short else "long",
                          "conviction": lv[(bits >> 2) % len(lv)], "horizon_days": 1,
                          "rationale": "stub pick"})
    view = SCRIPT["views"][seed % len(SCRIPT["views"])]
    return json.dumps({"view": view, "confidence": 0.6, "picks": picks})


class GoldenStub:
    """fn(system, user) for StubClient; counts calls; optionally crashes after N calls."""

    def __init__(self, crash_after_calls: int | None = None):
        self.calls = 0
        self.crash_after = crash_after_calls
        self._lock = threading.Lock()

    def __call__(self, system: str, user: str) -> str:
        with self._lock:
            self.calls += 1
            n = self.calls
        if self.crash_after is not None and n > self.crash_after:
            raise SimulatedCrash(f"simulated crash at call {n}")
        return respond(agent_of(system), json.loads(user)["asof"])
