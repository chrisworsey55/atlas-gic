"""ONE live smoke test of the configured model (skipped by default; run with --live and
ANTHROPIC_API_KEY set). It makes a single minimal real call with the packaged settings
(model, effort) and the same request shape every agent call uses, so a rejected parameter
or model name shows up here instead of on day 1 of a backtest. Cost: well under $0.01."""
import os
from pathlib import Path

import pytest

from atlas.config import load_settings
from atlas.engine.llm import AnthropicClient

SETTINGS = Path(__file__).resolve().parents[2] / "atlas" / "config" / "settings.yaml"


@pytest.mark.live
def test_live_minimal_call_with_the_configured_model_returns_text():
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        pytest.skip("ANTHROPIC_API_KEY is not set")
    s = load_settings(SETTINGS)
    r = AnthropicClient(s.model, effort=s.effort).complete(
        "Reply with exactly one word.", "Say: ok", max_tokens=1024)
    assert r.stop_reason == "end_turn" and r.text.strip()
    assert r.input_tokens > 0 and r.output_tokens > 0
