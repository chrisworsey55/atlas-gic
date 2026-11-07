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


def _settings(**over):
    from atlas.config import Settings
    base = dict(universe=["SPY"], commission_per_share=0.005, commission_min=1.0, slippage_bps=5,
                min_valid_agents=8, rolling_ic_window=60, warmup_picks=30, budget_usd=50,
                model="claude-sonnet-5-5", model_cutoff="2026-06-01")
    base.update(over)
    return Settings(**base)


def test_prices_looked_up_by_model_when_not_set():
    from atlas.config import MODEL_PRICES_USD_PER_MTOK
    s = _settings()
    assert s.usd_per_mtok_input is None and s.usd_per_mtok_output is None
    assert s.prices() == MODEL_PRICES_USD_PER_MTOK["claude-sonnet-5-5"] == (2.0, 10.0)
    assert _settings(model="claude-opus-5-5").prices() == (4.0, 20.0)


def test_unknown_model_without_prices_is_refused_with_a_fix():
    from atlas.config import UnknownModelPrice
    s = _settings(model="claude-imaginary-9")
    with pytest.raises(UnknownModelPrice) as ei:
        s.prices()
    msg = str(ei.value)
    assert "claude-imaginary-9" in msg and "usd_per_mtok_input" in msg and "usd_per_mtok_output" in msg


def test_explicit_prices_win_over_the_table():
    s = _settings(usd_per_mtok_input=7.0, usd_per_mtok_output=70.0)
    assert s.prices() == (7.0, 70.0)
    assert _settings(model="claude-imaginary-9", usd_per_mtok_input=1.0, usd_per_mtok_output=2.0).prices() == (1.0, 2.0)


def test_half_set_or_invalid_prices_rejected():
    with pytest.raises(ValidationError):
        _settings(usd_per_mtok_input=1.0)
    with pytest.raises(ValidationError):
        _settings(usd_per_mtok_input=-1.0, usd_per_mtok_output=1.0)
    with pytest.raises(ValidationError):
        _settings(older_usd_per_mtok_output=1.0)


def test_older_model_priced_with_its_own_entry_or_explicit_prices():
    from atlas.config import UnknownModelPrice
    s = _settings(usd_per_mtok_input=9.0, usd_per_mtok_output=99.0, older_model="claude-haiku-4-5")
    assert s.prices() == (9.0, 99.0)
    assert s.older_prices() == (1.0, 5.0)  # the older model's table entry, not the main model's prices
    s2 = _settings(older_model="claude-old-x", older_usd_per_mtok_input=0.5, older_usd_per_mtok_output=1.5)
    assert s2.older_prices() == (0.5, 1.5)
    with pytest.raises(UnknownModelPrice) as ei:
        _settings(older_model="claude-old-x").older_prices()
    assert "older_usd_per_mtok_input" in str(ei.value)


def test_request_settings_defaults():
    s = _settings()
    assert s.max_output_tokens == 8192 and s.effort is None
    assert not hasattr(s, "temperature")  # never sent: current models reject sampling params
    with pytest.raises(ValidationError):
        _settings(effort="extreme")
    with pytest.raises(ValidationError):
        _settings(max_output_tokens=0)


def test_packaged_settings_set_low_effort_and_known_model():
    s = load_settings(CFG / "settings.yaml")
    assert s.effort == "low" and s.max_output_tokens == 8192
    assert s.prices() == (2.0, 10.0)
