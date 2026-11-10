"""CLI-level tests of the final review fixes: fatal API errors, per-model pricing, the price
data preflight, the model-cutoff placeholder warning, logging hygiene and small robustness fixes."""
import json
import logging

import pytest

from atlas.engine.llm import FatalApiError
from tests.golden.stub import GoldenStub
from tests.golden.test_golden_backtest import (  # noqa: F401 - fixtures
    FAKE_ANTHROPIC, SHORT_PERIODS, cli, copy_run, golden, keys, write_config,
)


# ---------------------------------------------------------------------------- C1 fatal API errors

def test_fatal_api_error_aborts_with_a_clear_message_not_a_transient_pause(golden, tmp_path, keys, capsys):
    over = {"periods": SHORT_PERIODS}
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], **over)
    good = GoldenStub()
    calls = []

    def rejecting(system, user):
        calls.append(1)
        if json.loads(user)["asof"] == "2024-01-04":
            raise FatalApiError(400, "invalid_request_error", "temperature: not supported for this model")
        return good(system, user)

    assert cli(["backtest", "--config", conf], rejecting) == 1
    text = "".join(capsys.readouterr())
    assert ("error: Anthropic API rejected the request (HTTP 400 invalid_request_error): temperature: "
            "not supported for this model") in text
    assert "transient" not in text and "paused" not in text and "Traceback" not in text
    assert FAKE_ANTHROPIC not in text
    cp = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    assert cp["last_completed_day"] == "2024-01-03"  # the checkpoint is intact
    # 3 good days of 12 agents, then each layer-1 agent of 2024-01-04 tried ONCE: nothing retried
    assert len(calls) == 3 * 12 + 3
    from atlas.engine.failures import FailureLog
    assert FailureLog(tmp_path / "run" / "failures.sqlite").rows().empty  # not an agent failure
    # fixing the cause and re-running resumes and matches an uninterrupted run
    assert cli(["backtest", "--config", conf], GoldenStub()) == 0


def test_fatal_api_error_in_the_control_pass_also_aborts_clearly(golden, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], periods=SHORT_PERIODS)
    good = GoldenStub()

    def control_rejected(system, user):
        if "universe" in json.loads(user)["inputs"] and len(json.loads(user)["inputs"]) == 1:
            raise FatalApiError(404, "not_found_error", "model: x")
        return good(system, user)

    assert cli(["backtest", "--config", conf, "--with-control"], control_rejected) == 1
    text = "".join(capsys.readouterr())
    assert "error: Anthropic API rejected the request (HTTP 404 not_found_error)" in text
    assert "Traceback" not in text


# ---------------------------------------------------------------------------- I1 pricing

def test_unknown_model_without_prices_is_refused_before_any_call(golden, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"],
                        settings={"model": "claude-imaginary-9", "usd_per_mtok_input": None,
                                  "usd_per_mtok_output": None})
    stub = GoldenStub()
    assert cli(["backtest", "--config", conf], stub) == 2
    text = "".join(capsys.readouterr())
    assert "claude-imaginary-9" in text and "usd_per_mtok_input" in text and "usd_per_mtok_output" in text
    assert stub.calls == 0 and not (tmp_path / "run" / "checkpoint.json").exists()


def test_known_model_is_priced_from_the_table(golden, tmp_path, monkeypatch):
    import atlas.config as cfg
    from atlas.cli import build_run, load_run_config
    # a stub id (not in the cutoff table, so the golden 2024 cutoff stays valid) priced via the table
    monkeypatch.setitem(cfg.MODEL_PRICES_USD_PER_MTOK, "golden-priced-model", (4.0, 20.0))
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"],
                        settings={"model": "golden-priced-model", "usd_per_mtok_input": None,
                                  "usd_per_mtok_output": None})
    run = build_run(load_run_config(conf), llm_client=GoldenStub())
    assert run.llm.usd_per_mtok == (4.0, 20.0)


def test_build_run_wires_effort_and_max_output_tokens(golden, tmp_path, monkeypatch):
    import atlas.engine.llm as llm_mod
    from atlas.cli import build_run, load_run_config
    seen = {}

    class FakeClient:
        request_tag = "effort=medium"

        def __init__(self, model, sdk=None, *, effort=None):
            seen.update(model=model, effort=effort)

    monkeypatch.setattr(llm_mod, "AnthropicClient", FakeClient)
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"],
                        settings={"effort": "medium", "max_output_tokens": 5000})
    run = build_run(load_run_config(conf))
    assert seen == {"model": "golden-stub-model", "effort": "medium"}
    assert run.pipeline.max_tokens == 5000 and run.llm.request_tag == "effort=medium"


# ---------------------------------------------------------------------------- I4 preflight

def test_universe_ticker_with_no_price_data_fails_before_any_llm_call(golden, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"],
                        universe=["SPY", "SMH", "XLF", "XLE", "XLV", "AAA", "BBB", "CCC", "ZZZ"])
    stub = GoldenStub()
    assert cli(["backtest", "--config", conf], stub) == 1
    text = "".join(capsys.readouterr())
    assert "error: no price data for: ZZZ over 2024-01-01..2024-02-23" in text and "Traceback" not in text
    assert stub.calls == 0 and not (tmp_path / "run" / "checkpoint.json").exists()


def test_dry_run_also_runs_the_preflight(golden, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"],
                        universe=["SPY", "AAA", "QQQ_NOPE"])
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 1
    assert "no price data for: QQQ_NOPE" in "".join(capsys.readouterr())


# ---------------------------------------------------------------------------- I2 cutoff warning

def test_cli_warns_when_an_unlisted_model_uses_the_shipped_default_cutoff(golden, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"],
                        settings={"model_cutoff": "2026-06-30"})
    cli(["backtest", "--config", conf, "--dry-run"])
    err = capsys.readouterr().err
    assert "golden-stub-model" in err and "model_cutoff is the shipped default; re-check it on the Models " \
        "overview page" in err
    assert err.count("shipped default") == 1
    conf2 = write_config(tmp_path / "run2.yaml", tmp_path / "run2", golden["repo"])
    cli(["backtest", "--config", conf2, "--dry-run"])
    assert "shipped default" not in capsys.readouterr().err


# ---------------------------------------------------------------------------- logging hygiene

def test_urllib3_logger_is_pinned_to_warning_even_under_debug_logging(golden, tmp_path, monkeypatch):
    root = logging.getLogger()
    monkeypatch.setattr(root, "level", logging.DEBUG)
    lg = logging.getLogger("urllib3")
    monkeypatch.setattr(lg, "level", logging.DEBUG)
    cli(["report", "--run-dir", tmp_path / "nothing"])
    assert lg.level == logging.WARNING and not lg.isEnabledFor(logging.DEBUG)


# ---------------------------------------------------------------------------- M2 corrupt calibration

def test_corrupt_calibration_is_a_clear_report_error(golden, tmp_path, capsys):
    run_dir, _ = copy_run(golden, tmp_path)
    (run_dir / "calibration.json").write_text("{not json")
    assert cli(["report", "--run-dir", run_dir]) == 1
    text = "".join(capsys.readouterr())
    assert f"error: calibration.json is corrupt: {run_dir / 'calibration.json'}" in text
    assert "Traceback" not in text


# ---------------------------------------------------------------------------- M4 branch print

def test_autoresearch_prints_the_branch_it_will_merge_into(golden, tmp_path, keys, monkeypatch, capsys):
    import atlas.autoresearch.loop as loop_mod
    monkeypatch.setattr(loop_mod, "run_autoresearch", lambda *a, **k: [])
    run_dir, conf = copy_run(golden, tmp_path)
    assert cli(["autoresearch", "--config", conf], GoldenStub()) == 0
    out = capsys.readouterr().out
    assert "autoresearch: kept edits are merged into the checked-out branch 'main'" in out


def test_autoresearch_refuses_a_detached_head(golden, tmp_path, keys, capsys):
    import shutil
    from tests.autoresearch.conftest import git
    repo = tmp_path / "repo"
    shutil.copytree(golden["repo"], repo)
    git(repo, "checkout", "-q", "--detach")
    shutil.copytree(golden["run_dir"], tmp_path / "run")
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo)
    assert cli(["autoresearch", "--config", conf], GoldenStub()) == 1
    text = "".join(capsys.readouterr())
    assert "detached HEAD" in text and "Traceback" not in text
