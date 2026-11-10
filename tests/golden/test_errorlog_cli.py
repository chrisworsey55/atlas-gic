"""Error log wiring in the CLI and `atlas errors`. Golden fixtures, injected clients, no network."""
import json
import os
import shutil

import pytest

import atlas.cli as cli_mod
from atlas.engine.llm import ApiError, FatalApiError, sanitise
from atlas.errorlog import read_events
from tests.golden.stub import GoldenStub
from tests.golden.test_golden_backtest import (  # noqa: F401 (fixtures)
    FAKE_ANTHROPIC, SHORT_PERIODS, cli, golden, keys, write_config,
)

CANARY = "sk-ant-CANARY"


def _conf(tmp_path, golden):  # noqa: F811
    return write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], periods=SHORT_PERIODS)


def _log(tmp_path):
    return tmp_path / "run" / "errors.log"


def test_rejected_request_writes_fatal_api_error_event_and_exits_1(golden, tmp_path, keys, capsys):  # noqa: F811
    def reject(system, user):
        raise FatalApiError(400, "invalid_request_error", "bad model name")

    assert cli(["backtest", "--config", _conf(tmp_path, golden)], reject) == 1
    err = capsys.readouterr().err
    assert "Anthropic API rejected the request (HTTP 400 invalid_request_error): bad model name" in err
    assert "Nothing was retried" in err and "Traceback" not in err
    ev = read_events(_log(tmp_path), kind="fatal_api_error")
    # the runner logs it with its day; main() adds the command-level record (no day)
    assert ev and all("bad model name" in e["message"] for e in ev)
    assert ev[-1]["fields"]["exc_type"] == "FatalApiError" and ev[-1]["day"] is None


def test_unexpected_exception_is_caught_logged_and_exits_1_without_traceback(golden, tmp_path, keys, capsys):  # noqa: F811
    def boom(system, user):
        raise RuntimeError(f"boom {CANARY}")

    conf = _conf(tmp_path, golden)
    assert cli(["backtest", "--config", conf], boom) == 1
    cap = capsys.readouterr()
    lines = [ln for ln in cap.err.splitlines() if ln.strip()]
    unexpected = [ln for ln in lines if ln.startswith("error: unexpected ")]
    assert len(unexpected) == 1
    assert unexpected[0].startswith("error: unexpected RuntimeError: boom <redacted>")
    assert unexpected[0].endswith(f"(details: {_log(tmp_path)})")
    assert "Traceback" not in cap.err and CANARY not in cap.err
    ev = read_events(_log(tmp_path), kind="unexpected_exception")
    assert len(ev) == 1 and ev[0]["fields"]["exc_type"] == "RuntimeError"
    assert ev[0]["fields"]["frames"] and CANARY not in json.dumps(ev[0])


def test_catch_all_handles_exception_but_not_base_exception(golden, tmp_path, keys, monkeypatch, capsys):  # noqa: F811
    conf = _conf(tmp_path, golden)

    def explode(args, llm_client=None):
        raise RuntimeError(f"boom {CANARY}")

    monkeypatch.setitem(cli_mod.COMMANDS, "backtest", explode)
    assert cli(["backtest", "--config", conf]) == 1
    err = capsys.readouterr().err
    assert err.strip() == "error: unexpected RuntimeError: boom <redacted>"  # no log active: no path
    assert not (tmp_path / "run").exists()

    def interrupt(args, llm_client=None):
        raise KeyboardInterrupt

    monkeypatch.setitem(cli_mod.COMMANDS, "backtest", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cli(["backtest", "--config", conf])


def test_unexpected_exception_after_build_run_logs_and_names_the_log(golden, tmp_path, keys, monkeypatch, capsys):  # noqa: F811
    conf = _conf(tmp_path, golden)

    def bad(*a, **k):
        raise ValueError(f"oops {CANARY}")

    monkeypatch.setattr(cli_mod, "make_backtest_config", bad)
    assert cli(["backtest", "--config", conf], GoldenStub()) == 1
    err = capsys.readouterr().err
    assert err.strip() == f"error: unexpected ValueError: oops <redacted> (details: {_log(tmp_path)})"
    assert len(read_events(_log(tmp_path), kind="unexpected_exception")) == 1


def test_paused_run_leaves_run_paused_event_and_resume_appends(golden, tmp_path, keys, monkeypatch, capsys):  # noqa: F811
    import atlas.engine.agent as agent_mod
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)
    conf = _conf(tmp_path, golden)
    good = GoldenStub()

    def flaky(system, user):
        if json.loads(user)["asof"] == "2024-01-04":
            raise ApiError("APIStatusError status=529")
        return good(system, user)

    assert cli(["backtest", "--config", conf], flaky) == 1
    out = capsys.readouterr().err
    assert "paused" in out and "re-run the same command to resume" in out
    first = read_events(_log(tmp_path))
    assert len([e for e in first if e["kind"] == "run_paused"]) == 1
    n = len(first)
    assert cli(["backtest", "--config", conf], GoldenStub()) == 0
    after = read_events(_log(tmp_path))
    assert len(after) >= n and after[:n] == first  # appended to the same file, nothing rewritten
    assert len([e for e in after if e["kind"] == "run_paused"]) == 1


def test_dry_run_writes_log_under_dry_run_dir_only(golden, tmp_path, monkeypatch, capsys):  # noqa: F811
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    conf = _conf(tmp_path, golden)
    # a failing dry run: break the stored record so the second run is refused
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 0
    other = write_config(tmp_path / "run2.yaml", tmp_path / "run", golden["repo"],
                         periods={**SHORT_PERIODS, "train": {"start": "2024-01-01", "end": "2024-01-02"}})
    assert cli(["backtest", "--config", other, "--dry-run"]) == 1
    assert "different configuration" in capsys.readouterr().err
    assert not _log(tmp_path).exists()
    ev = read_events(tmp_path / "run" / "dry_run" / "errors.log", kind="operational_error")
    assert len(ev) == 1 and "different configuration" in ev[0]["message"]
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["dry_run"]


def _seed(tmp_path):
    from atlas.errorlog import ErrorLog
    d = tmp_path / "rd"
    log = ErrorLog(d / "errors.log")
    for i in range(5):
        log.event("call_failure", f"msg {i}", level="warning", day="2024-01-0%d" % (i + 1),
                  agent=f"a{i}", reason="timeout")
    log.event("run_paused", "paused here", level="warning", day="2024-01-09")
    return d


def test_errors_subcommand_prints_tail_and_filters_by_kind(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    d = _seed(tmp_path)
    assert cli(["errors", "--run-dir", d, "--tail", "3"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 3
    assert "msg 3" in lines[0] and "msg 4" in lines[1] and "paused here" in lines[2]  # oldest first
    parts = lines[0].split()
    assert parts[1] == "warning" and parts[2] == "call_failure" and "2024-01-04" in lines[0]
    assert "a3" in lines[0] and "timeout" in lines[0]
    assert cli(["errors", "--run-dir", d, "--kind", "run_paused"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "run_paused" in lines[0]
    assert cli(["errors", "--run-dir", d]) == 0  # default tail 20 shows all 6
    assert len(capsys.readouterr().out.strip().splitlines()) == 6


def test_errors_subcommand_accepts_dry_run_subdir(tmp_path, capsys):
    d = _seed(tmp_path)
    shutil.copytree(d, tmp_path / "x" / "dry_run")
    assert cli(["errors", "--run-dir", tmp_path / "x" / "dry_run"]) == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 6


def test_errors_subcommand_missing_log_exits_1(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    assert cli(["errors", "--run-dir", tmp_path / "empty"]) == 1
    assert capsys.readouterr().err.strip() == f"error: no error log in {tmp_path / 'empty'}"
    assert sorted(p.name for p in (tmp_path / "empty").iterdir()) == []


def test_errors_subcommand_creates_no_files(tmp_path, capsys):
    d = _seed(tmp_path)
    before = sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*"))
    mtime = os.stat(d / "errors.log").st_mtime_ns
    assert cli(["errors", "--run-dir", d]) == 0
    assert sorted(str(p.relative_to(tmp_path)) for p in tmp_path.rglob("*")) == before
    assert os.stat(d / "errors.log").st_mtime_ns == mtime


def test_errors_is_listed_in_help_and_commands(capsys):
    assert "errors" in cli_mod.COMMANDS
    with pytest.raises(SystemExit):
        cli_mod.build_parser().parse_args(["--help"])
    assert "errors" in capsys.readouterr().out


def test_canary_key_absent_from_errors_log_and_stdout_stderr(golden, tmp_path, keys, monkeypatch, capsys):  # noqa: F811
    def reject(system, user):
        raise FatalApiError(401, "authentication_error", sanitise(f"bad key {CANARY}-xyz"))

    assert cli(["backtest", "--config", _conf(tmp_path, golden)], reject) == 1
    cap = capsys.readouterr()
    assert CANARY not in cap.out + cap.err
    raw = _log(tmp_path).read_text()
    assert CANARY not in raw and FAKE_ANTHROPIC not in raw
    assert cli(["errors", "--run-dir", tmp_path / "run"]) == 0
    assert CANARY not in capsys.readouterr().out


def test_config_error_before_run_known_prints_error_but_creates_no_log(golden, tmp_path, capsys):  # noqa: F811
    p = tmp_path / "bad.yaml"
    p.write_text("run_dir: x\nperiods: {train: {start: 2024-01-01}}\n")
    assert cli(["backtest", "--config", p, "--dry-run"]) == 2
    assert capsys.readouterr().err.startswith("error: ")
    assert cli_mod.ACTIVE_ERRORLOG is None
    assert not list(tmp_path.rglob("errors.log"))


def test_active_errorlog_is_reset_at_the_start_of_main(golden, tmp_path, keys, capsys):  # noqa: F811
    assert cli(["backtest", "--config", _conf(tmp_path, golden), "--dry-run"]) == 0
    assert cli_mod.ACTIVE_ERRORLOG is not None
    cli(["errors", "--run-dir", tmp_path])
    assert cli_mod.ACTIVE_ERRORLOG is None


def test_usage_error_after_build_run_is_logged_and_message_unchanged(golden, tmp_path, keys, capsys):  # noqa: F811
    conf = _conf(tmp_path, golden)
    import atlas.cli as c
    orig = c.make_backtest_config

    def no_prices(*a, **k):
        raise c.UsageError("synthetic usage problem")

    c.make_backtest_config = no_prices
    try:
        assert cli(["backtest", "--config", conf, "--dry-run"]) == 2
    finally:
        c.make_backtest_config = orig
    assert capsys.readouterr().err == "error: synthetic usage problem\n"
    ev = read_events(tmp_path / "run" / "dry_run" / "errors.log", kind="usage_error")
    assert len(ev) == 1 and ev[0]["message"] == "synthetic usage problem"
