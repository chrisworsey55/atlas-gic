from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from atlas.config import (
    AgentSpec, ModelCutoffError, Settings, UnknownModelPrice, check_model_cutoffs, load_agents,
    load_limits, load_settings)

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
    assert s.model == "claude-sonnet-5-5" and s.model_cutoff.isoformat() == "2026-06-30"
    assert (s.shrinkage_k, s.dead_zone, s.max_edits, s.max_failure_rate,
            s.confirm_cost_usd, s.older_model, s.tolerance_days) == (20.0, None, 5, 0.2, 25.0, None, 5)


def _settings(**over):
    from atlas.config import Settings
    base = dict(universe=["SPY"], commission_per_share=0.005, commission_min=1.0, slippage_bps=5,
                min_valid_agents=8, rolling_ic_window=60, warmup_picks=30, budget_usd=50,
                model="claude-sonnet-5-5", model_cutoff="2026-06-30")
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


def test_new_settings_defaults():
    s = _settings()
    assert (s.confirm_on_validation, s.confirm_min_days, s.min_pair_ratio, s.impute_abstentions) == (
        True, 20, 0.95, True)
    for bad in (dict(confirm_min_days=4), dict(min_pair_ratio=0), dict(min_pair_ratio=1.01)):
        with pytest.raises(ValidationError):
            _settings(**bad)


def test_old_run_config_still_loads_without_new_fields():
    from atlas.cli import record_diff
    from atlas.config import Settings
    new = _settings().model_dump(mode="json")
    old = {k: v for k, v in new.items() if k not in (
        "confirm_on_validation", "confirm_min_days", "min_pair_ratio", "impute_abstentions")}
    assert Settings(**old).min_pair_ratio == 0.95
    # a run directory stored before these fields existed is not refused merely for the added defaults
    assert record_diff({"settings": old}, {"settings": new}) == []
    # the four autoresearch-only gate settings are allowed drift (the live config wins in stored_run)
    for k, v in (("min_pair_ratio", 0.8), ("confirm_on_validation", False), ("confirm_min_days", 30),
                 ("impute_abstentions", False)):
        assert record_diff({"settings": old}, {"settings": {**new, k: v}}) == []
    # ...but a real change to anything else is still a difference
    assert record_diff({"settings": old}, {"settings": {**new, "warmup_picks": 9}}) == ["settings.warmup_picks"]
    assert record_diff({"settings": old}, {"settings": {**new, "budget_usd": 1}}) == []  # allowed drift
    assert record_diff({"settings": old}, {"settings": {**new, "slippage_bps": 9}}) == ["settings.slippage_bps"]


# ---------------------------------------------------------------------------- hub and spoke models

def _spec(kind, name=None, model=None):
    return AgentSpec(name=name or kind, layer=1, kind=kind, prompt_file=f"{kind}.md", model=model)


def test_agent_model_precedence_override_then_spoke_then_main():
    s = _settings(model="claude-opus-5-5", spoke_model="claude-haiku-5-5")
    for kind in ("macro", "sector", "superinvestor"):
        assert s.agent_model(_spec(kind)) == "claude-haiku-5-5"
    for kind in ("cro", "cio", "execution"):
        assert s.agent_model(_spec(kind)) == "claude-opus-5-5"
    assert s.agent_model(_spec("macro", model="claude-sonnet-5-5")) == "claude-sonnet-5-5"
    assert s.agent_model(_spec("cio", model="claude-sonnet-5-5")) == "claude-sonnet-5-5"


def test_control_uses_spoke_model():
    assert _settings(spoke_model="claude-haiku-5-5").agent_model(_spec("control")) == "claude-haiku-5-5"
    assert _settings().agent_model(_spec("control")) == "claude-sonnet-5-5"


def test_unset_spoke_model_changes_nothing():
    s = _settings()
    assert s.spoke_model is None
    for kind in ("macro", "sector", "superinvestor", "cro", "cio", "execution", "control"):
        assert s.agent_model(_spec(kind)) == s.model
    assert check_model_cutoffs(s, [_spec("macro"), _spec("cio")]) == []


def test_prices_for_uses_table_and_explicit_pair_and_refuses_unknown():
    s = _settings(model="m-main", usd_per_mtok_input=1.0, usd_per_mtok_output=2.0,
                  spoke_model="claude-haiku-5-5")
    assert s.prices_for("m-main") == (1.0, 2.0) == s.prices()
    assert s.prices_for("claude-haiku-5-5") == (0.5, 2.5)          # table
    assert s.prices_for("claude-opus-5-5") == (4.0, 20.0)          # per-agent override: table
    s2 = _settings(spoke_model="m-spoke", spoke_usd_per_mtok_input=0.1, spoke_usd_per_mtok_output=0.2)
    assert s2.prices_for("m-spoke") == (0.1, 0.2)
    assert s2.prices_for("claude-sonnet-5-5") == (2.0, 10.0)       # main model still from the table
    with pytest.raises(UnknownModelPrice, match="spoke_usd_per_mtok_input"):
        _settings(spoke_model="m-spoke").prices_for("m-spoke")
    with pytest.raises(UnknownModelPrice, match="m-other"):
        s2.prices_for("m-other")


def test_spoke_price_pair_both_or_neither():
    with pytest.raises(ValidationError):
        _settings(spoke_usd_per_mtok_input=1.0)
    with pytest.raises(ValidationError):
        _settings(spoke_usd_per_mtok_output=1.0)
    with pytest.raises(ValidationError):
        _settings(spoke_usd_per_mtok_input=-1.0, spoke_usd_per_mtok_output=1.0)


def test_cutoff_table_and_shipped_default():
    from atlas.config import MODEL_TRAINING_CUTOFFS, SHIPPED_MODEL_CUTOFF
    for m in ("claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5", "claude-fable-5-1"):
        assert MODEL_TRAINING_CUTOFFS[m] == date(2026, 6, 30)
    assert SHIPPED_MODEL_CUTOFF == date(2026, 6, 30)


def test_shipped_default_cutoff_is_2026_06_30():
    assert load_settings(CFG / "settings.yaml").model_cutoff == date(2026, 6, 30)


def test_cutoff_check_refuses_a_used_model_with_a_later_documented_cutoff():
    specs = [_spec("macro"), _spec("cio")]
    # equal is fine; one day earlier refuses (off-by-one pin)
    assert check_model_cutoffs(_settings(spoke_model="claude-haiku-5-5", model_cutoff="2026-06-30"), specs) == []
    with pytest.raises(ModelCutoffError, match="claude-haiku-5-5"):
        check_model_cutoffs(_settings(spoke_model="claude-haiku-5-5", model_cutoff="2026-06-29"), specs)
    # the main model alone is checked as well
    with pytest.raises(ModelCutoffError, match="claude-sonnet-5-5"):
        check_model_cutoffs(_settings(model_cutoff="2026-01-01"), specs)
    assert ModelCutoffError.__mro__[1] is ValueError


def test_per_agent_override_model_checked_too():
    s = _settings(model="m-stub", usd_per_mtok_input=1.0, usd_per_mtok_output=1.0, model_cutoff="2024-01-31")
    with pytest.raises(ModelCutoffError, match="claude-opus-5-5"):
        check_model_cutoffs(s, [_spec("cio"), _spec("sector", model="claude-opus-5-5")])
    # model unused by any agent (the only agent that would use it is overridden away): no refusal
    check_model_cutoffs(s, [_spec("cio", model="m-stub")])


def test_unknown_model_warns_and_requires_explicit_cutoff_at_least_max_known():
    s = _settings(model="m-stub", usd_per_mtok_input=1.0, usd_per_mtok_output=1.0, model_cutoff="2026-06-30")
    w = check_model_cutoffs(s, [_spec("cio")])
    assert len(w) == 1 and "m-stub" in w[0] and "shipped default" in w[0]
    # an explicit cutoff earlier than the newest documented one: warns it may be too early, never refuses
    s = _settings(model="m-stub", usd_per_mtok_input=1.0, usd_per_mtok_output=1.0, model_cutoff="2025-01-31")
    w = check_model_cutoffs(s, [_spec("cio")])
    assert len(w) == 1 and "m-stub" in w[0] and "2026-06-30" in w[0] and "shipped default" not in w[0]
    # known models never warn
    assert check_model_cutoffs(_settings(), [_spec("cio")]) == []


def test_date_suffixed_ids_use_their_base_model_cutoff():
    # claude-sonnet-5-5-20260101 is a snapshot id of a table model: refuse like the base model
    s = _settings(model="claude-sonnet-5-5-20260101", usd_per_mtok_input=1.0, usd_per_mtok_output=1.0,
                  model_cutoff="2026-01-01")
    with pytest.raises(ModelCutoffError):
        check_model_cutoffs(s, [_spec("cio")])
    assert check_model_cutoffs(_settings(model="claude-sonnet-5-5-20260101", usd_per_mtok_input=1.0,
                                         usd_per_mtok_output=1.0), [_spec("cio")]) == []


def test_old_run_config_without_spoke_and_agent_models_is_not_refused():
    from atlas.cli import record_diff
    new_s = _settings().model_dump(mode="json")
    old_s = {k: v for k, v in new_s.items() if not k.startswith("spoke_")}
    assert Settings(**old_s).spoke_model is None
    new_a = [_spec("macro", "a").model_dump(mode="json")]
    old_a = [{k: v for k, v in a.items() if k != "model"} for a in new_a]
    assert record_diff({"settings": old_s, "agents": old_a}, {"settings": new_s, "agents": new_a}) == []
    # a real change is still caught, including setting the new fields
    assert record_diff({"settings": old_s, "agents": old_a},
                       {"settings": {**new_s, "spoke_model": "claude-haiku-5-5"}, "agents": new_a}) == ["settings.spoke_model"]
    changed = [{**new_a[0], "model": "claude-opus-5-5"}]
    assert record_diff({"settings": old_s, "agents": old_a}, {"settings": new_s, "agents": changed}) == ["agents.a.model"]


@pytest.mark.parametrize("field,bad", [("budget_usd", float("nan")), ("budget_usd", "nan"),
                                       ("budget_usd", float("inf")), ("budget_usd", 0), ("budget_usd", -1),
                                       ("confirm_cost_usd", float("nan")), ("confirm_cost_usd", float("inf")),
                                       ("confirm_cost_usd", -0.01)])
def test_money_caps_must_be_real_numbers(field, bad):
    # a NaN cap makes every `spend > cap` comparison False, i.e. no cap at all
    with pytest.raises(ValidationError, match=field):
        _settings(**{field: bad})


def test_money_caps_keep_accepting_current_values():
    assert _settings(budget_usd=0.2).budget_usd == 0.2
    assert _settings(confirm_cost_usd=0.0).confirm_cost_usd == 0.0  # 0 = always ask (used by tests)
    assert _settings(confirm_cost_usd=1e9).confirm_cost_usd == 1e9
