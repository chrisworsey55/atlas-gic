from datetime import date

import pytest

from atlas.backtest.runner import BacktestConfig, Periods, estimate_cost
from atlas.config import Settings
from tests.backtest.test_runner import limits


def _cfg(dry_run=False, **over):
    base = dict(universe=["SPY"], commission_per_share=0.005, commission_min=1.0, slippage_bps=5,
                min_valid_agents=8, rolling_ic_window=60, warmup_picks=30, budget_usd=50,
                model="hub-m", model_cutoff="2026-06-30", usd_per_mtok_input=10.0,
                usd_per_mtok_output=50.0, spoke_model="spoke-m", spoke_usd_per_mtok_input=1.0,
                spoke_usd_per_mtok_output=5.0, est_input_tokens_per_call=1000,
                est_output_tokens_per_call=2000)
    base.update(over)
    s = Settings(**base)
    p = Periods(train=(date(2026, 1, 5), date(2026, 3, 1)), validation=(date(2026, 3, 2), date(2026, 5, 1)),
                holdout=(date(2026, 7, 1), date(2026, 9, 1)))
    lim = limits()
    return BacktestConfig(periods=p, settings=s, limits=lim, anonymize=False, dry_run=dry_run)


HUB, SPOKE = 10.0 * 1000 / 1e6 + 50.0 * 2000 / 1e6, 1.0 * 1000 / 1e6 + 5.0 * 2000 / 1e6


def test_estimate_matches_hand_computed_mixed_cost():
    cfg = _cfg()
    models = ["spoke-m"] * 9 + ["hub-m"] * 3
    assert estimate_cost(cfg, 4, 12, models) == pytest.approx(4 * (9 * SPOKE + 3 * HUB))
    assert 9 * SPOKE + 3 * HUB < 12 * HUB  # not priced at the main rate for everyone


def test_estimate_unchanged_without_agent_models():
    cfg = _cfg()
    assert estimate_cost(cfg, 4, 12) == pytest.approx(4 * 12 * HUB)
    assert estimate_cost(cfg, 4, 12, None) == estimate_cost(cfg, 4, 12)
    assert estimate_cost(cfg, 4, 12, ["hub-m"] * 12) == pytest.approx(estimate_cost(cfg, 4, 12))


def test_agent_models_length_must_match_n_agents():
    cfg = _cfg()
    with pytest.raises(ValueError):
        estimate_cost(cfg, 4, 12, ["hub-m"] * 11)
    with pytest.raises(ValueError):
        estimate_cost(_cfg(dry_run=True), 4, 12, ["hub-m"] * 13)  # checked even for a dry run


def test_dry_run_is_free_with_agent_models():
    assert estimate_cost(_cfg(dry_run=True), 4, 12, ["spoke-m"] * 12) == 0.0


def test_unpriced_model_raises():
    from atlas.config import UnknownModelPrice
    with pytest.raises(UnknownModelPrice):
        estimate_cost(_cfg(), 4, 2, ["hub-m", "no-such-model"])


def test_run_backtest_estimates_with_the_pipelines_models(tmp_path):
    from atlas.backtest.runner import CostConfirmationRequired
    from tests.backtest.test_runner import Env
    env = Env(tmp_path, n_days=4, set_over={"confirm_cost_usd": 0.0001, "usd_per_mtok_input": 10.0,
                                            "usd_per_mtok_output": 50.0, "spoke_model": "spoke-m",
                                            "spoke_usd_per_mtok_input": 1.0, "spoke_usd_per_mtok_output": 5.0})
    s = env.settings
    per = lambda p: (s.est_input_tokens_per_call * p[0] + s.est_output_tokens_per_call * p[1]) / 1e6  # noqa: E731
    names = [sp.name for sp in env.pipeline.specs.values() if sp.kind != "control"]
    hub = {"cro", "cio", "execution"}
    env.pipeline.models = {n: (s.model if n in hub else "spoke-m") for n in names}
    with pytest.raises(CostConfirmationRequired) as ei:
        env.run()
    want = 4 * (9 * per((1.0, 5.0)) + 3 * per((10.0, 50.0)))
    assert ei.value.estimate == pytest.approx(want)


def test_dry_run_never_reaches_routed_clients(tmp_path):
    from atlas.engine.llm import CachedLLM, StubClient
    from tests.backtest.test_runner import Env

    def boom(_s, _u):
        raise AssertionError("a routed client was called in a dry run")

    env = Env(tmp_path, n_days=6, dry_run=True)
    other = CachedLLM(StubClient(boom), tmp_path / "o.db", "spoke-m", (0.0, 0.0), 1e9)
    env.pipeline.llms = {"spoke-m": other}
    env.pipeline.models = {n: "spoke-m" for n in env.pipeline.specs}
    res = env.run()
    assert len(res.equity) == 6 and env.calls == []
