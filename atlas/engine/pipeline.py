"""The 4-layer agent pipeline: macro -> sector -> superinvestor -> CRO/CIO/execution.

Only agents whose AgentResult is ok feed later layers; a failed agent's partial output never
leaves run_agent. The execution agent keeps the AgentOutput contract (conviction = target
portfolio weight); this module converts targets into integer-share orders.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from collections.abc import Mapping
from datetime import date
from typing import Literal

import pandas as pd

from atlas.config import AgentSpec
from atlas.data.store import Snapshot
from atlas.engine.agent import MAX_TOKENS, run_agent
from atlas.engine.failures import FailureLog
from atlas.engine.llm import CachedLLM
from atlas.schemas import AgentResult

Regime = Literal["RISK_ON", "RISK_OFF", "NEUTRAL"]
_VIEW_SIGN = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}


@dataclass
class DayResult:
    asof: date
    regime: Regime | None
    results: dict[str, AgentResult] = field(default_factory=dict)
    proposed_orders: list[dict] | None = None
    skipped_reason: str | None = None


def aggregate_regime(macro_outputs: list[AgentResult], weights: dict[str, float]) -> Regime:
    """Weighted vote of valid macro views, each vote = sign * confidence * agent weight."""
    score = 0.0
    for r in macro_outputs:
        if r.status != "ok":
            continue
        score += _VIEW_SIGN[r.output.view] * r.output.confidence * weights.get(r.agent, 1.0)
    if abs(score) < 1e-12:
        return "NEUTRAL"
    return "RISK_ON" if score > 0 else "RISK_OFF"


def orders_from_targets(targets: dict[str, float], positions: dict[str, int], equity: float,
                        last_close: dict[str, float]) -> list[dict]:
    """Target portfolio weights -> whole-share orders (sells first). Tickers with no usable
    price are skipped; a no-change target yields no order."""
    sells, buys = [], []
    for t in sorted(targets):
        px = last_close.get(t)
        if not px or px <= 0 or math.isnan(px):
            continue
        held = positions.get(t, 0)
        want = max(0, math.floor(max(0.0, targets[t]) * equity / px))
        if want < held:
            sells.append({"ticker": t, "side": "sell", "shares": held - want})
        elif want > held:
            buys.append({"ticker": t, "side": "buy", "shares": want - held})
    return sells + buys


def _num(x) -> float | None:
    return None if x is None or pd.isna(x) else round(float(x), 6)


def _market_summary(snap: Snapshot) -> dict:
    prices: dict[str, dict] = {}
    for t, df in snap.prices.groupby("ticker"):
        c = df.sort_values("date")["close"].astype(float).tolist()
        ret = lambda n: _num(c[-1] / c[-1 - n] - 1) if len(c) > n and c[-1 - n] else None  # noqa: E731
        prices[t] = {"last_close": _num(c[-1]), "ret_5d": ret(5), "ret_20d": ret(20)}
        if "sector" in df.columns:  # anonymised snapshots carry a sector label per ticker
            prices[t]["sector"] = str(df["sector"].iloc[-1])
    macro = {}
    for sid, s in sorted(snap.macro.items()):
        s = s.dropna()
        if len(s):
            macro[sid] = {"last": _num(s.iloc[-1]), "chg_5obs": _num(s.iloc[-1] - s.iloc[max(0, len(s) - 6)])}
    return {"prices": prices, "macro": macro}


def _last_close(snap: Snapshot) -> dict[str, float]:
    out = {}
    for t, df in snap.prices.groupby("ticker"):
        v = _num(df.sort_values("date")["close"].iloc[-1])
        if v is not None:
            out[t] = v
    return out


class Pipeline:
    def __init__(self, specs: list[AgentSpec], prompts: dict[str, str], llm: CachedLLM,
                 failures: FailureLog, universe, min_valid_agents: int, *, max_tokens: int = MAX_TOKENS,
                 llms: Mapping[str, CachedLLM] | None = None, models: Mapping[str, str] | None = None):
        """`llm` is the default client. For a hub-and-spoke run, `models` maps agent name ->
        effective model and `llms` maps model -> its CachedLLM; both None means every agent
        uses `llm` (single-model run, unchanged). An agent missing from `models` falls back to
        `llm`; an agent whose mapped model has no client in `llms` is a configuration error and
        raises (a silent fallback would run a different model than the one mapped and
        misattribute its spend)."""
        self.specs = {s.name: s for s in specs}
        self.max_tokens = max_tokens  # per agent call (Settings.max_output_tokens in the CLI)
        self.prompts = prompts
        self.llm = llm
        self.llms = llms
        self.models = models
        self.failures = failures
        self.universe = set(universe)
        self.min_valid_agents = min_valid_agents

    def llm_for(self, spec: AgentSpec) -> CachedLLM:
        """The client serving this agent's model; `self.llm` when there is no routing or the
        agent is not in `models`. A mapped model with no client raises (see __init__)."""
        if self.llms is None or self.models is None:
            return self.llm
        model = self.models.get(spec.name)
        if model is None:
            return self.llm
        try:
            return self.llms[model]
        except KeyError:
            raise RuntimeError(f"agent '{spec.name}' is mapped to model '{model}' but no client was "
                               f"built for it (clients: {sorted(self.llms)})") from None

    # ---- per-agent building blocks (shared by run_day and run_single) ----
    def _view(self, spec: AgentSpec, r: AgentResult, weights: dict[str, float]) -> dict:
        o = r.output
        return {"agent": spec.name, "kind": spec.kind, "view": o.view, "confidence": o.confidence,
                "weight": weights.get(spec.name, 1.0),
                "picks": [p.model_dump() for p in o.picks]}

    def _valid(self, upstream: dict[str, AgentResult], layers: range, weights) -> list[dict]:
        """Views from ok upstream agents in the given layers, ordered by (layer, name)."""
        specs = sorted((s for s in self.specs.values() if s.layer in layers and s.kind != "control"),
                       key=lambda s: (s.layer, s.name))
        return [self._view(s, upstream[s.name], weights) for s in specs
                if s.name in upstream and upstream[s.name].status == "ok"]

    def _regime(self, upstream: dict[str, AgentResult], weights) -> Regime:
        macro = [upstream[s.name] for s in self.specs.values()
                 if s.kind == "macro" and s.name in upstream]
        return aggregate_regime(macro, weights)

    def _inputs(self, spec: AgentSpec, snap: Snapshot, weights: dict[str, float], state: dict,
                upstream: dict[str, AgentResult]) -> dict:
        if spec.kind == "control":
            return {"universe": sorted(self.universe)}
        base = {"market": _market_summary(snap)}
        if spec.kind == "macro":
            return base
        regime = self._regime(upstream, weights)
        if spec.kind == "sector":
            return {**base, "regime": regime, "benchmark": spec.benchmark,
                    "views": self._valid(upstream, range(1, 2), weights)}
        if spec.kind == "superinvestor":
            return {**base, "regime": regime, "benchmark": spec.benchmark,
                    "views": self._valid(upstream, range(1, 3), weights)}
        views = self._valid(upstream, range(1, 4), weights)
        cro = upstream.get("cro")
        # the review's date is omitted: it duplicates asof and would defeat anonymised labels
        review = (cro.output.model_dump(mode="json", exclude={"date"})
                  if cro is not None and cro.status == "ok" else None)
        if spec.kind == "cro":
            return {**base, "regime": regime, "views": views, "state": state}
        if spec.kind == "cio":
            return {"regime": regime, "views": views, "risk_review": review, "state": state,
                    "last_close": _last_close(snap)}
        # execution
        cio = upstream.get("cio")
        targets = [p.model_dump() for p in cio.output.picks] if cio is not None and cio.status == "ok" else []
        return {"cio_targets": targets, "state": state, "last_close": _last_close(snap)}

    def run_single(self, name: str, asof: date, snapshot: Snapshot, weights: dict[str, float],
                   state: dict, upstream: dict[str, AgentResult],
                   prompt_override: str | None = None, *, asof_label: str | None = None) -> AgentResult:
        spec = self.specs[name]
        prompt = prompt_override if prompt_override is not None else self.prompts[name]
        inputs = self._inputs(spec, snapshot, weights, state, upstream)
        return run_agent(spec, prompt, inputs, asof, self.llm_for(spec), self.failures, self.universe,
                         asof_label=asof_label, max_tokens=self.max_tokens)

    # ---- the day ----
    def _run_layer(self, layer: int, asof, snapshot, weights, state, results, asof_label=None) -> None:
        specs = sorted((s for s in self.specs.values() if s.layer == layer and s.kind != "control"),
                       key=lambda s: s.name)
        if not specs:
            return
        snap_results = dict(results)  # layer members never see each other
        if layer <= 3:
            with ThreadPoolExecutor(max_workers=len(specs)) as ex:
                futs = {s.name: ex.submit(self.run_single, s.name, asof, snapshot, weights, state,
                                          snap_results, asof_label=asof_label) for s in specs}
                for n, f in futs.items():
                    results[n] = f.result()
        else:  # layer 4 is sequential CRO -> CIO -> execution
            order = {"cro": 0, "cio": 1, "execution": 2}
            for s in sorted(specs, key=lambda s: order.get(s.kind, 9)):
                results[s.name] = self.run_single(s.name, asof, snapshot, weights, state, results,
                                                  asof_label=asof_label)

    def _failed_text(self, results: dict[str, AgentResult]) -> str:
        return ", ".join(f"{n} ({r.failure.reason.value})" for n, r in sorted(results.items())
                         if r.status != "ok" and r.failure is not None)

    def _too_few(self, results: dict[str, AgentResult]) -> str | None:
        valid = sum(1 for r in results.values() if r.status == "ok")
        if valid < self.min_valid_agents:
            return (f"only {valid} valid agents (< {self.min_valid_agents}); failed: "
                    f"{self._failed_text(results)}")
        return None

    def run_day(self, asof: date, snapshot: Snapshot, weights: dict[str, float], state: dict, *,
                asof_label: str | None = None) -> DayResult:
        """`asof_label` replaces the date shown to agents (anonymised runs)."""
        results: dict[str, AgentResult] = {}
        for layer in (1, 2, 3):
            self._run_layer(layer, asof, snapshot, weights, state, results, asof_label)
        regime = self._regime(results, weights)
        # fail fast: layer 4 cannot raise the count above the minimum by more than its 3 agents
        remaining = sum(1 for s in self.specs.values() if s.layer == 4 and s.kind != "control")
        valid = sum(1 for r in results.values() if r.status == "ok")
        if valid + remaining < self.min_valid_agents:
            return DayResult(asof, regime, results, None, self._too_few(results))
        self._run_layer(4, asof, snapshot, weights, state, results, asof_label)
        reason = self._too_few(results)
        if reason is None:
            for kind in ("cio", "execution"):
                for s in self.specs.values():
                    if s.kind == kind and results.get(s.name) is not None and results[s.name].status != "ok":
                        f = results[s.name].failure
                        reason = f"{s.name} failed ({f.reason.value if f else 'no output'}); no decision to trade on"
                        break
                if reason:
                    break
        if reason is None:
            ex = next((results[s.name] for s in self.specs.values() if s.kind == "execution"), None)
            if ex is None:
                reason = "no execution agent configured"
        if reason is not None:
            return DayResult(asof, regime, results, None, reason)
        targets = {p.ticker: (p.conviction if p.direction == "long" else 0.0) for p in ex.output.picks}
        orders = orders_from_targets(targets, state.get("positions", {}), state["equity"],
                                     _last_close(snapshot))
        return DayResult(asof, regime, results, orders, None)
