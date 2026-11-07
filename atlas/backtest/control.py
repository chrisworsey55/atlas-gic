"""Date-only control agent: sees the date and ticker list, no market data.

Measures how much apparent skill is just model memory of the period.
"""
from __future__ import annotations

from datetime import date

from atlas.config import AgentSpec
from atlas.engine.agent import MAX_TOKENS, run_agent
from atlas.engine.failures import FailureLog
from atlas.engine.llm import CachedLLM
from atlas.schemas import AgentResult

CONTROL_SPEC = AgentSpec(name="control", layer=4, kind="control", prompt_file="prompts/control.md")


def run_control_agent(asof: date, universe: list[str], prompt: str, llm: CachedLLM,
                      failures: FailureLog, *, spec: AgentSpec | None = None,
                      max_tokens: int = MAX_TOKENS) -> AgentResult:
    spec = spec or CONTROL_SPEC
    return run_agent(spec, prompt, {"universe": sorted(universe)}, asof, llm, failures, set(universe),
                     max_tokens=max_tokens)
