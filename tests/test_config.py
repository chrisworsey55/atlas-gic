from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.config import AgentSpec, load_agents, load_limits, load_settings

CFG = Path(__file__).resolve().parent.parent / "atlas" / "config"


def test_agents_yaml_has_12_agents_in_expected_layers():
    agents = load_agents(CFG / "agents.yaml")
    twelve = [a for a in agents if a.kind != "control"]
    assert len(twelve) == 12
    layers = {}
    for a in twelve:
        layers.setdefault(a.layer, []).append(a.kind)
    assert sorted(layers[1]) == ["macro"] * 3
    assert sorted(layers[2]) == ["sector"] * 4
    assert sorted(layers[3]) == ["superinvestor"] * 2
    assert sorted(layers[4]) == ["cio", "cro", "execution"]
    assert [a.kind for a in agents].count("control") == 1


def test_limits_defaults_match_spec():
    lim = load_limits(CFG / "limits.yaml")
    assert (lim.max_position_pct, lim.max_gross_pct, lim.max_positions, lim.min_order_usd,
            lim.max_commission_frac, lim.daily_loss_pct, lim.drawdown_kill_pct) == (
        0.15, 1.0, 8, 50, 0.01, 0.03, 0.15)
    assert lim.long_only is True


def test_sector_benchmarks():
    by_name = {a.name: a for a in load_agents(CFG / "agents.yaml")}
    expected = {"semiconductors": "SMH", "financials": "XLF", "energy": "XLE", "healthcare": "XLV"}
    for key, bench in expected.items():
        (a,) = [x for n, x in by_name.items() if key in n]
        assert a.kind == "sector" and a.benchmark == bench
    for a in by_name.values():
        if a.kind in ("macro", "superinvestor"):
            assert a.benchmark == "SPY"
        if a.kind in ("cro", "cio", "execution", "control"):
            assert a.benchmark is None


def test_unknown_kind_rejected():
    with pytest.raises(ValidationError):
        AgentSpec(name="x", layer=1, kind="wizard", prompt_file="x.md", benchmark=None)


def test_settings_load():
    s = load_settings(CFG / "settings.yaml")
    assert {"SPY", "SMH", "XLF", "XLE", "XLV"} <= set(s.universe)
    assert (s.commission_per_share, s.commission_min, s.slippage_bps) == (0.005, 1.0, 5)
    assert (s.min_valid_agents, s.rolling_ic_window, s.warmup_picks, s.budget_usd) == (8, 60, 30, 50)
    assert s.model == "claude-sonnet-5-5" and s.model_cutoff.isoformat() == "2026-06-01"
    assert (s.shrinkage_k, s.dead_zone, s.max_edits, s.max_failure_rate,
            s.confirm_cost_usd, s.older_model, s.tolerance_days) == (20.0, None, 5, 0.2, 25.0, None, 5)
