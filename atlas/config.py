"""Typed loaders for the YAML files in atlas/config/."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class AgentSpec(BaseModel):
    name: str
    layer: int
    kind: Literal["macro", "sector", "superinvestor", "cro", "cio", "execution", "control"]
    prompt_file: str
    benchmark: str | None = None


class Limits(BaseModel):
    max_position_pct: float
    max_gross_pct: float
    max_positions: int
    min_order_usd: float
    max_commission_frac: float
    daily_loss_pct: float
    drawdown_kill_pct: float
    long_only: bool


class Settings(BaseModel):
    universe: list[str]
    commission_per_share: float
    commission_min: float
    slippage_bps: float
    min_valid_agents: int
    rolling_ic_window: int
    warmup_picks: int
    budget_usd: float
    model: str
    model_cutoff: date
    shrinkage_k: float = 20.0
    dead_zone: float | None = None
    max_edits: int = 5
    max_failure_rate: float = 0.2
    confirm_cost_usd: float = 25.0
    older_model: str | None = None
    tolerance_days: int = 5
    # pricing assumptions for the up-front backtest cost estimate (USD per million tokens and
    # typical tokens per agent call); the hard cap is enforced by CachedLLM on actual usage
    usd_per_mtok_input: float = 3.0
    usd_per_mtok_output: float = 15.0
    est_input_tokens_per_call: int = 4000
    est_output_tokens_per_call: int = 800


def _load(path: str | Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_agents(path: str | Path) -> list[AgentSpec]:
    return [AgentSpec(**a) for a in _load(path)["agents"]]


def load_limits(path: str | Path) -> Limits:
    return Limits(**_load(path))


def load_settings(path: str | Path) -> Settings:
    return Settings(**_load(path))
