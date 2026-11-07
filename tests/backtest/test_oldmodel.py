"""Older-model check (optional): rerun only the window after an older model's cutoff."""
import hashlib
import json
import logging
from datetime import date
from pathlib import Path

import pytest

from atlas.backtest.oldmodel import (
    OlderModelSetup, OlderModelUnavailable, period_after_cutoff, run_older_model_check,
)
from atlas.backtest.runner import BacktestConfig, Periods
from atlas.cli import build_run, load_run_config, main
from atlas.engine.llm import ApiError, StubClient
from atlas.engine.pipeline import Pipeline
from atlas.portfolio.simbroker import SimBroker
from atlas.risk.layer import RiskLayer
from tests.golden.stub import GoldenStub
from tests.golden.test_golden_backtest import ROOT, write_config

P = Periods(train=(date(2024, 1, 1), date(2024, 1, 19)),
            validation=(date(2024, 1, 22), date(2024, 2, 2)),
            holdout=(date(2024, 2, 5), date(2024, 2, 23)))


# ---------------------------------------------------------------------------- period_after_cutoff

def test_period_after_cutoff_none_when_all_dates_precede_cutoff():
    assert period_after_cutoff(P, date(2024, 2, 23)) is None   # cutoff == last day: nothing after
    assert period_after_cutoff(P, date(2025, 1, 1)) is None


def test_period_after_cutoff_before_everything_is_the_whole_run():
    assert period_after_cutoff(P, date(2023, 6, 1)) == (date(2024, 1, 1), date(2024, 2, 23))


def test_period_after_cutoff_inside_train_clips_the_start_to_cutoff_plus_one():
    assert period_after_cutoff(P, date(2024, 1, 10)) == (date(2024, 1, 11), date(2024, 2, 23))


def test_period_after_cutoff_exactly_at_a_boundary_is_strictly_after():
    assert period_after_cutoff(P, date(2024, 1, 19)) == (date(2024, 1, 20), date(2024, 2, 23))
    # cutoff the day before the run starts: the run starts at its own first day
    assert period_after_cutoff(P, date(2023, 12, 31)) == (date(2024, 1, 1), date(2024, 2, 23))
    # cutoff the day before the last day
    assert period_after_cutoff(P, date(2024, 2, 22)) == (date(2024, 2, 23), date(2024, 2, 23))


def test_period_after_cutoff_in_a_gap_between_periods():
    assert period_after_cutoff(P, date(2024, 2, 3)) == (date(2024, 2, 4), date(2024, 2, 23))


# ---------------------------------------------------------------------------- the check

@pytest.fixture(scope="module")
def main_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("old")
    run_dir = root / "run"
    conf = write_config(root / "run.yaml", run_dir, ROOT)
    import os
    os.environ["ANTHROPIC_API_KEY"] = "sk-ant-TEST-do-not-print"
    try:
        assert main(["backtest", "--config", str(conf)], llm_client=StubClient(GoldenStub())) == 0
    finally:
        del os.environ["ANTHROPIC_API_KEY"]
    return {"conf": conf, "run_dir": run_dir}


def tree_hash(root: Path, skip: str = "older_model") -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and rel.parts[0] != skip:
            out[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def setup_for(conf, client) -> tuple[BacktestConfig, OlderModelSetup]:
    rc = load_run_config(conf)
    run = build_run(rc, offline=False, llm_client=StubClient(GoldenStub()))

    def make_pipeline(llm, failures):
        return Pipeline(run.specs, run.prompts, llm, failures, run.universe, run.settings.min_valid_agents)

    cfg = BacktestConfig(run.periods, run.settings, run.limits, False, False)
    setup = OlderModelSetup(
        store=run.store, make_pipeline=make_pipeline, client=client,
        broker=SimBroker(run.store, run.settings.commission_per_share, run.settings.commission_min,
                         run.settings.slippage_bps),
        risk=RiskLayer(run.limits, run.settings.commission_per_share, run.settings.commission_min),
        initial_cash=rc.initial_cash, lookback_days=rc.data.lookback_days)
    return cfg, setup


def test_older_model_check_skipped_with_clear_message_when_no_window(main_run, caplog):
    cfg, setup = setup_for(main_run["conf"], StubClient(GoldenStub()))
    with caplog.at_level(logging.WARNING, logger="atlas.backtest.oldmodel"):
        res = run_older_model_check(cfg, "old-model", date(2024, 3, 1), setup, run_dir=main_run["run_dir"])
    assert res is None
    msg = " ".join(r.getMessage() for r in caplog.records)
    assert "old-model" in msg and "2024-03-01" in msg and "skipped: no period lies after" in msg
    assert not (main_run["run_dir"] / "older_model").exists()


def test_older_model_check_runs_only_post_cutoff_days_in_its_own_dir(main_run):
    cfg, setup = setup_for(main_run["conf"], StubClient(GoldenStub()))
    before = tree_hash(main_run["run_dir"])
    res = run_older_model_check(cfg, "old-model", date(2024, 1, 12), setup, run_dir=main_run["run_dir"])
    assert res is not None
    assert min(res.equity.index) == date(2024, 1, 15) and max(res.equity.index) == date(2024, 2, 23)
    assert all(d > date(2024, 1, 12) for d in res.equity.index)
    sub = main_run["run_dir"] / "older_model" / "old-model"
    for name in ("checkpoint.json", "prediction_log.jsonl", "scores.sqlite", "llm_cache.sqlite"):
        assert (sub / name).exists(), name
    assert tree_hash(main_run["run_dir"]) == before  # the main run is byte-identical
    cp = json.loads((sub / "checkpoint.json").read_text())
    assert cp["last_completed_day"] == "2024-02-23"


def test_older_model_check_uses_the_older_model_name_and_cutoff(main_run):
    cfg, setup = setup_for(main_run["conf"], StubClient(GoldenStub()))
    seen = {}
    orig = setup.make_pipeline

    def spy(llm, failures):
        seen["model"] = llm.model
        return orig(llm, failures)

    setup.make_pipeline = spy
    run_older_model_check(cfg, "claude-old-1", date(2024, 1, 26), setup, run_dir=main_run["run_dir"])
    assert seen["model"] == "claude-old-1"


def test_older_model_check_window_with_dropped_periods_resplits_into_three(main_run):
    # cutoff 2024-01-26 leaves only validation (partly) and holdout: the post-cutoff trading
    # days are split in three contiguous parts so the runner's three-period contract holds
    cfg, setup = setup_for(main_run["conf"], StubClient(GoldenStub()))
    res = run_older_model_check(cfg, "m2", date(2024, 1, 26), setup, run_dir=main_run["run_dir"])
    assert res is not None
    assert min(res.equity.index) == date(2024, 1, 29) and max(res.equity.index) == date(2024, 2, 23)
    assert res.periods.train[0] == date(2024, 1, 29) and res.periods.holdout[1] == date(2024, 2, 23)


def test_older_model_check_window_too_short_is_skipped(main_run, caplog):
    cfg, setup = setup_for(main_run["conf"], StubClient(GoldenStub()))
    with caplog.at_level(logging.WARNING, logger="atlas.backtest.oldmodel"):
        res = run_older_model_check(cfg, "m3", date(2024, 2, 21), setup, run_dir=main_run["run_dir"])
    assert res is None and "trading days" in caplog.text


def test_older_model_unreachable_is_a_clear_error_not_a_traceback_or_empty_result(main_run, monkeypatch):
    import atlas.engine.agent as agent_mod
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)

    def down(system, user):
        raise ApiError("APIStatusError status=404")

    cfg, setup = setup_for(main_run["conf"], StubClient(down))
    with pytest.raises(OlderModelUnavailable) as ei:
        run_older_model_check(cfg, "ghost-model", date(2024, 1, 12), setup, run_dir=main_run["run_dir"])
    msg = str(ei.value)
    assert "ghost-model" in msg and "api_error" in msg and "model name" in msg


def test_older_model_budget_is_its_own_hard_cap(main_run):
    cfg, setup = setup_for(main_run["conf"], StubClient(GoldenStub()))
    with pytest.raises(OlderModelUnavailable) as ei:
        run_older_model_check(cfg, "m-budget", date(2024, 1, 12), setup, run_dir=main_run["run_dir"],
                              budget_usd=0.0)
    assert "budget" in str(ei.value).lower()


# ---------------------------------------------------------------------------- fix round 1

import dataclasses  # noqa: E402
import shutil  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from atlas.backtest.oldmodel import _dirname, _window_periods  # noqa: E402
from atlas.cli import spend_so_far  # noqa: E402


@pytest.fixture
def run_copy(main_run, tmp_path):
    dest = tmp_path / "run"
    shutil.copytree(main_run["run_dir"], dest, ignore=shutil.ignore_patterns("older_model"))
    return dest


def with_budget(cfg, usd):
    return dataclasses.replace(cfg, settings=cfg.settings.model_copy(update={"budget_usd": usd}))


def test_older_model_default_cap_is_what_remains_of_the_runs_hard_cap(main_run, run_copy):
    cfg, setup = setup_for(main_run["conf"], StubClient(GoldenStub()))
    own = spend_so_far(run_copy)["backtest"]
    cfg = with_budget(cfg, own + 0.3)  # $0.30 left; the check needs about $1
    with pytest.raises(OlderModelUnavailable) as ei:
        run_older_model_check(cfg, "old", date(2024, 1, 12), setup, run_dir=run_copy)
    assert "budget" in str(ei.value).lower()
    total = sum(spend_so_far(run_copy).values())
    assert total <= own + 0.3 + 1e-9 and spend_so_far(run_copy)["older_model"] > 0


def test_older_model_refuses_when_nothing_remains(main_run, run_copy):
    calls = []
    cfg, setup = setup_for(main_run["conf"], StubClient(lambda s, u: calls.append(1)))
    cfg = with_budget(cfg, spend_so_far(run_copy)["backtest"])
    with pytest.raises(OlderModelUnavailable) as ei:
        run_older_model_check(cfg, "old", date(2024, 1, 12), setup, run_dir=run_copy)
    assert "exhausted" in str(ei.value) and not calls
    assert not (run_copy / "older_model").exists()


def test_older_model_spend_is_its_own_bucket_counted_once_in_the_report_total(main_run, run_copy):
    from atlas.cli import main
    cfg, setup = setup_for(main_run["conf"], StubClient(GoldenStub()))
    own = spend_so_far(run_copy)["backtest"]
    res = run_older_model_check(cfg, "old", date(2024, 1, 12), setup, run_dir=run_copy)
    spent = spend_so_far(run_copy)
    assert spent["older_model"] == pytest.approx(res.spent_usd) and spent["older_model"] > 0
    assert main(["report", "--run-dir", str(run_copy)]) == 0
    md = (run_copy / "report" / "report.md").read_text()
    assert f"Total API cost: ${own + res.spent_usd:.2f}" in md


def test_model_names_never_leave_older_model_and_never_collide(main_run, run_copy):
    names = ["../evil", "a/b", "a_b", "..", ".", "x y"]
    dirs = [_dirname(n) for n in names]
    assert len(set(dirs)) == len(names)
    for d in dirs:
        assert d not in ("", ".", "..") and "/" not in d and "\\" not in d and not d.startswith(".")
    assert _dirname("claude-3-opus_20240229.v1") == "claude-3-opus_20240229.v1"  # plain names stay readable
    cfg, setup = setup_for(main_run["conf"], StubClient(GoldenStub()))
    for name in ("../evil", "a/b", "a_b"):
        res = run_older_model_check(cfg, name, date(2024, 2, 12), setup, run_dir=run_copy)
        assert res is not None
    kids = sorted(p.name for p in (run_copy / "older_model").iterdir())
    assert len(kids) == 3 and all((run_copy / "older_model" / k / "checkpoint.json").exists() for k in kids)
    assert sorted(p.name for p in run_copy.parent.iterdir()) == ["run"]  # nothing escaped to ../
    assert not (run_copy / "evil").exists()


def test_resplit_never_spans_dates_that_were_not_part_of_the_run():
    def weekdays(lo, hi):
        out, d = [], lo
        while d <= hi:
            if d.weekday() < 5:
                out.append(d)
            d = date.fromordinal(d.toordinal() + 1)
        return out

    store = SimpleNamespace(trading_days=weekdays)
    per = Periods(train=(date(2024, 1, 1), date(2024, 1, 3)), validation=(date(2024, 1, 4), date(2024, 1, 10)),
                  holdout=(date(2024, 2, 5), date(2024, 2, 9)))
    new = _window_periods(per, date(2024, 1, 3), store)
    own = set(weekdays(date(2024, 1, 4), date(2024, 1, 10))) | set(weekdays(date(2024, 2, 5), date(2024, 2, 9)))
    covered = [d for _, (lo, hi) in new.items() for d in weekdays(lo, hi)]
    assert set(covered) == own and len(covered) == len(own)  # no gap day (01-11..02-02), no overlap
    # one surviving period is split in thirds
    one = _window_periods(per, date(2024, 2, 1), store)
    assert [d for _, (lo, hi) in one.items() for d in weekdays(lo, hi)] == weekdays(date(2024, 2, 5), date(2024, 2, 9))


def test_older_model_is_priced_with_its_own_table_entry_not_the_main_models(main_run, run_copy):
    from atlas.engine.llm import LLMResponse
    cfg, setup = setup_for(main_run["conf"], None)
    # main model priced at $3/$15 explicitly; the older model is a known table entry ($1/$5)
    cfg = dataclasses.replace(cfg, settings=cfg.settings.model_copy(
        update={"older_usd_per_mtok_input": None, "older_usd_per_mtok_output": None}))

    class Fixed:
        def complete(self, system, user, max_tokens):
            r = StubClient(GoldenStub()).complete(system, user, max_tokens)
            return LLMResponse(r.text, 1_000, 100, r.stop_reason)

    setup.client = Fixed()
    res = run_older_model_check(cfg, "claude-haiku-4-5", date(2024, 2, 12), setup, run_dir=run_copy)
    calls = sum(len(d.results) for d in res.day_results.values())
    assert res.spent_usd == pytest.approx(calls * (1_000 * 1.0 + 100 * 5.0) / 1e6)


def test_unknown_older_model_without_prices_is_refused_before_any_call(main_run, run_copy):
    calls = []
    cfg, setup = setup_for(main_run["conf"], StubClient(lambda s, u: calls.append(1)))
    cfg = dataclasses.replace(cfg, settings=cfg.settings.model_copy(
        update={"older_usd_per_mtok_input": None, "older_usd_per_mtok_output": None}))
    with pytest.raises(OlderModelUnavailable) as ei:
        run_older_model_check(cfg, "claude-old-unknown", date(2024, 1, 12), setup, run_dir=run_copy)
    assert "older_usd_per_mtok_input" in str(ei.value) and not calls
    assert not (run_copy / "older_model").exists()


def test_older_model_fatal_api_error_is_a_clear_error(main_run, run_copy):
    from atlas.engine.llm import FatalApiError

    def rejected(system, user):
        raise FatalApiError(404, "not_found_error", "model: claude-ghost")

    cfg, setup = setup_for(main_run["conf"], StubClient(rejected))
    with pytest.raises(OlderModelUnavailable) as ei:
        run_older_model_check(cfg, "claude-ghost", date(2024, 1, 12), setup, run_dir=run_copy)
    assert "HTTP 404" in str(ei.value) and "claude-ghost" in str(ei.value)
