"""Typed loaders for the YAML files in atlas/config/."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator

# Anthropic first-party list prices, USD per million tokens (input, output), from the Claude API
# reference (models table cached 2026-10-06). Only models that reference confirms are listed;
# any other model needs explicit usd_per_mtok_input/output. These are ESTIMATES used for the
# budget cap: Anthropic bills actual usage (cache writes/reads, Bedrock/Vertex pricing, and
# price changes are not modelled), so also set a spend limit on the Anthropic account.
# claude-haiku-5-5 uses its higher (>100K-token prompt) rate so the cap errs on the safe side.
MODEL_PRICES_USD_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-1": (10.0, 50.0),
    "claude-mythos-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-5-5": (0.5, 2.5),
    "claude-haiku-4-5": (1.0, 5.0),
}

# the model_cutoff shipped in atlas/config/settings.yaml: a PLACEHOLDER, not a documented cutoff
SHIPPED_MODEL_CUTOFF = date(2026, 6, 1)

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class UnknownModelPrice(ValueError):
    """No price is known for a model and none was configured."""


def model_prices(model: str, explicit: tuple[float | None, float | None] = (None, None), *,
                 fields: tuple[str, str] = ("usd_per_mtok_input", "usd_per_mtok_output")) -> tuple[float, float]:
    """(input, output) USD per million tokens: the explicit pair if set, else the table entry
    for `model`. Raises UnknownModelPrice naming the settings to set."""
    if explicit[0] is not None and explicit[1] is not None:
        return float(explicit[0]), float(explicit[1])
    if model in MODEL_PRICES_USD_PER_MTOK:
        return MODEL_PRICES_USD_PER_MTOK[model]
    raise UnknownModelPrice(
        f"no price is known for model '{model}', so the budget cap cannot be enforced. Set "
        f"settings.{fields[0]} and settings.{fields[1]} (USD per million tokens, from Anthropic's "
        f"pricing page) in the run config. Known models: {', '.join(sorted(MODEL_PRICES_USD_PER_MTOK))}")


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
    # request settings. No sampling parameter exists: current models reject temperature/top_p/
    # top_k with HTTP 400, and replay determinism comes from the response cache. `effort` is
    # sent as output_config.effort when set (None = the model's default); thinking stays at the
    # model default and its tokens count against max_output_tokens.
    max_output_tokens: int = Field(8192, gt=0)
    effort: Effort | None = None
    # prices in USD per million tokens. None = look up `model` (`older_model`) in
    # MODEL_PRICES_USD_PER_MTOK; an unknown model with no explicit prices is refused. Set both
    # of a pair or neither. They price the up-front estimate AND the budget cap, which counts
    # tokens at these prices: an estimate of the bill, not the bill.
    usd_per_mtok_input: float | None = Field(None, ge=0)
    usd_per_mtok_output: float | None = Field(None, ge=0)
    older_usd_per_mtok_input: float | None = Field(None, ge=0)
    older_usd_per_mtok_output: float | None = Field(None, ge=0)
    est_input_tokens_per_call: int = 4000
    est_output_tokens_per_call: int = 800

    @model_validator(mode="after")
    def _price_pairs(self):
        for a, b in (("usd_per_mtok_input", "usd_per_mtok_output"),
                     ("older_usd_per_mtok_input", "older_usd_per_mtok_output")):
            if (getattr(self, a) is None) != (getattr(self, b) is None):
                raise ValueError(f"set both {a} and {b}, or neither")
        return self

    def prices(self) -> tuple[float, float]:
        """(input, output) USD per million tokens for `model`; raises UnknownModelPrice."""
        return model_prices(self.model, (self.usd_per_mtok_input, self.usd_per_mtok_output))

    def older_prices(self, model: str | None = None) -> tuple[float, float]:
        """Prices of the older model (`model` or `older_model`): its own explicit pair or table
        entry, never the main model's prices; raises UnknownModelPrice."""
        return model_prices(model or self.older_model or "", (self.older_usd_per_mtok_input,
                                                                self.older_usd_per_mtok_output),
                            fields=("older_usd_per_mtok_input", "older_usd_per_mtok_output"))


def _load(path: str | Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_agents(path: str | Path) -> list[AgentSpec]:
    return [AgentSpec(**a) for a in _load(path)["agents"]]


def load_limits(path: str | Path) -> Limits:
    return Limits(**_load(path))


def load_settings(path: str | Path) -> Settings:
    return Settings(**_load(path))
