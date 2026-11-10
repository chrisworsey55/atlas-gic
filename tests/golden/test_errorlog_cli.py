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
    FAKE_ANTHROPIC, SHORT_PERIODS, cli, copy_run, golden, keys, write_config,
)

CANARY = "sk-ant-CANARY"


def _conf(tmp_path, golden):  # noqa: F811
    return write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], periods=SHORT_PERIODS)


def _no_cutoff_warning(err: str) -> str:
    """Drop the stub model's 'no documented training cutoff' warning (the golden stub model is not
    in the cutoff table) so these tests pin the error line only."""
    return "".join(ln for ln in err.splitlines(True) if not ln.startswith("warning: model 'golden-stub-model'"))


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
    err = _no_cutoff_warning(capsys.readouterr().err)
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
    assert _no_cutoff_warning(capsys.readouterr().err) == "error: synthetic usage problem\n"
    ev = read_events(tmp_path / "run" / "dry_run" / "errors.log", kind="usage_error")
    assert len(ev) == 1 and ev[0]["message"] == "synthetic usage problem"


# ---------------------------------------------------------------------------- fix round 1

import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
from pathlib import Path  # noqa: E402

from atlas.engine.failures import FailureLog  # noqa: E402
from atlas.errorlog import ErrorLog  # noqa: E402
from atlas.schemas import FailureRecord, FailureReason  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NASTY = "hi \x1b[2J\x1b]0;pwned\x07 cr\rbs\bdel\x7f c1\x9b end"


def _no_ctrl(raw: bytes) -> bool:
    return not any(b < 0x20 and b != 0x0A or b == 0x7F for b in raw)


def test_control_characters_are_escaped_on_the_write_side(tmp_path):
    path = tmp_path / "errors.log"
    ErrorLog(path).event("k\x1bind", NASTY, day="2024\x07", agent="a\rb", reason="r\x08",
                         **{"f\x1b": NASTY, "nested": [NASTY, {"x": NASTY}]})
    try:
        raise ValueError(NASTY)
    except ValueError as e:
        ErrorLog(path).exception("unexpected_exception", e)
    raw = path.read_bytes()
    assert _no_ctrl(raw) and b"\xc2\x9b" not in raw
    # JSON doubles the backslash of the visible escape; CR/LF are whitespace and collapse to a space
    assert b"\\\\x1b" in raw and b"\\\\x07" in raw and b"\\\\x08" in raw and b"\\\\x7f" in raw
    assert len(raw.splitlines()) == 2
    ev = read_events(path)[0]
    assert ev["message"].startswith("hi \\x1b[2J\\x1b]0;pwned\\x07")
    assert "\x1b" not in json.dumps(ev, ensure_ascii=False)


def test_errors_command_reescapes_and_redacts_an_edited_log(tmp_path, capsys):
    d = tmp_path / "rd"
    d.mkdir()
    rec = {"ts_utc": "2024-01-01T00:00:00+\x1b[31m", "level": "error", "kind": "k\x1b[0m", "day": "2024-01-01",
           "agent": "a\rb", "reason": "r\x08", "fields": {},
           "message": NASTY + " sk-ant-api03-ABCDEFGHIJKLMNOP"}
    (d / "errors.log").write_text(json.dumps(rec) + "\n")
    assert cli(["errors", "--run-dir", d]) == 0
    out = capsys.readouterr().out
    assert _no_ctrl(out.rstrip("\n").encode()) and "\x9b" not in out
    assert "\\x1b[2J" in out and "\\x07" in out and "sk-ant-api03" not in out and "<redacted>" in out
    assert len(out.strip().splitlines()) == 1


def test_errors_command_keeps_valid_events_and_warns_about_malformed_lines(tmp_path, capsys):
    d = _seed(tmp_path)
    lines = (d / "errors.log").read_text().splitlines()
    lines.insert(2, "{not json")
    lines.insert(4, "[1, 2]")
    (d / "errors.log").write_text("\n".join(lines) + "\n")
    assert cli(["errors", "--run-dir", d]) == 0
    cap = capsys.readouterr()
    assert len(cap.out.strip().splitlines()) == 6
    assert cap.err.strip() == f"warning: skipped 2 malformed line(s) in {d / 'errors.log'}"
    assert [e["kind"] for e in read_events(d / "errors.log", strict=False)].count("call_failure") == 5
    with pytest.raises(ValueError):
        read_events(d / "errors.log")  # strict default unchanged


def test_errors_command_on_all_garbage_prints_only_the_warning(tmp_path, capsys):
    d = tmp_path / "rd"
    d.mkdir()
    (d / "errors.log").write_text("garbage\nmore garbage\nand more\n")
    assert cli(["errors", "--run-dir", d]) == 0
    cap = capsys.readouterr()
    assert cap.out == "" and "skipped 3 malformed line(s)" in cap.err


def test_closed_pipe_is_not_an_error_subprocess(tmp_path):
    d = tmp_path / "rd"
    d.mkdir()
    rec = json.dumps({"ts_utc": "t", "level": "error", "kind": "call_failure", "day": "2024-01-01",
                      "agent": "a", "reason": "r", "message": "m" * 60, "fields": {}})
    (d / "errors.log").write_text((rec + "\n") * 30000)
    p = subprocess.Popen([sys.executable, "-m", "atlas", "errors", "--run-dir", str(d), "--tail", "100000"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=ROOT)
    assert p.stdout.readline().startswith(b"t error call_failure")
    p.stdout.close()
    err = p.stderr.read()
    assert p.wait(timeout=60) == 0
    assert err == b""


class _ClosedStdout:
    def write(self, s):
        raise BrokenPipeError(32, "Broken pipe")

    def flush(self):
        raise BrokenPipeError(32, "Broken pipe")


def test_broken_pipe_on_stdout_is_quiet_in_errors(tmp_path, monkeypatch, capsys):
    d = _seed(tmp_path)
    monkeypatch.setattr(sys, "stdout", _ClosedStdout())
    assert cli(["errors", "--run-dir", d]) == 0
    assert capsys.readouterr().err == ""
    assert read_events(d / "errors.log", kind="unexpected_exception") == []


def test_broken_pipe_at_the_final_stdout_flush_is_quiet_for_any_command(tmp_path, monkeypatch, capsys):
    class FlushFails:
        def write(self, s):
            return len(s)

        def flush(self):
            raise BrokenPipeError(32, "Broken pipe")

    monkeypatch.setitem(cli_mod.COMMANDS, "report", lambda args, llm_client=None: cli_mod.EXIT_OK)
    monkeypatch.setattr(sys, "stdout", FlushFails())
    assert cli(["report", "--run-dir", tmp_path]) == 0
    assert capsys.readouterr().err == ""


def test_broken_pipe_from_a_non_stdout_source_is_an_unexpected_failure(golden, tmp_path, keys, monkeypatch,  # noqa: F811
                                                                    capsys):
    conf = _conf(tmp_path, golden)

    def socket_dies(args, llm_client=None):
        cli_mod.build_run(cli_mod.load_run_config(args.config), offline=True)  # a run log is active
        raise BrokenPipeError(32, "net")

    monkeypatch.setitem(cli_mod.COMMANDS, "backtest", socket_dies)
    assert cli(["backtest", "--config", conf]) == 1
    err = _no_cutoff_warning(capsys.readouterr().err)
    assert err.strip() == (f"error: unexpected BrokenPipeError: [Errno 32] net "
                           f"(details: {tmp_path / 'run' / 'errors.log'})")
    assert len(read_events(tmp_path / "run" / "errors.log", kind="unexpected_exception")) == 1


def test_broken_pipe_raised_by_the_computation_of_a_backtest_is_a_failure(golden, tmp_path, keys, monkeypatch,  # noqa: F811
                                                                         capsys):
    def dies(*a, **k):
        raise BrokenPipeError(32, "net")

    monkeypatch.setattr(cli_mod, "run_backtest", dies)
    assert cli(["backtest", "--config", _conf(tmp_path, golden)], GoldenStub()) == 1
    assert "error: unexpected BrokenPipeError" in capsys.readouterr().err
    assert len(read_events(tmp_path / "run" / "errors.log", kind="unexpected_exception")) == 1


def test_invisible_and_bidi_characters_are_escaped_idempotently(tmp_path, capsys):
    from atlas.errorlog import escape_controls
    chars = [chr(c) for c in (*range(0x200B, 0x2010), *range(0x202A, 0x202F), *range(0x2066, 0x206A), 0xFEFF)]
    for ch in chars:
        assert escape_controls("a" + ch + "b") == "a\\u%04xb" % ord(ch)
    keep = "caf\u00e9 \u65e5\u672c\u8a9e \U0001f600 \u00a0"
    assert escape_controls(keep) == keep
    once = escape_controls("x\x1b" + "".join(chars))
    assert escape_controls(once) == once
    # write side
    path = tmp_path / "errors.log"
    ErrorLog(path).event("k", "m\u202eevil\u200b" + keep.rstrip(), **{"f\ufeff": "\u2066v"})
    raw = path.read_text(encoding="utf-8")
    assert not any(ch in raw for ch in chars) and "\\\\u202e" in raw and keep.rstrip() in json.loads(raw.splitlines()[0])["message"]
    # read side: an edited log
    d = tmp_path / "rd"
    d.mkdir()
    rec = {"ts_utc": "t", "level": "error", "kind": "k\u202e", "day": None, "agent": "a\u200f", "reason": None,
           "message": "m\ufeff\u2069 " + keep.rstrip(), "fields": {}}
    (d / "errors.log").write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    assert cli(["errors", "--run-dir", d]) == 0
    out = capsys.readouterr().out
    assert not any(ch in out for ch in chars) and "\\u202e" in out and "\\ufeff" in out and keep.rstrip() in out


class _OrderedStderr:
    """stderr that notes, at the first write of an error line, which events are already on disk."""

    def __init__(self, log):
        self.log, self.seen, self.text = log, None, ""

    def write(self, s):
        if self.seen is None and s.startswith("error:"):
            self.seen = [e["kind"] for e in read_events(self.log, strict=False)] if self.log.exists() else []
        self.text += s
        return len(s)

    def flush(self):
        pass


def test_known_errors_are_recorded_before_they_are_printed(golden, tmp_path, keys, monkeypatch):  # noqa: F811
    conf = _conf(tmp_path, golden)
    log = tmp_path / "run" / "dry_run" / "errors.log"
    # operational error (run record mismatch) and usage error, both after build_run
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 0
    other = write_config(tmp_path / "run2.yaml", tmp_path / "run", golden["repo"],
                         periods={**SHORT_PERIODS, "train": {"start": "2024-01-01", "end": "2024-01-02"}})
    fake = _OrderedStderr(log)
    monkeypatch.setattr(sys, "stderr", fake)
    assert cli(["backtest", "--config", other, "--dry-run"]) == 1
    assert fake.seen == ["operational_error"]

    def usage(*a, **k):
        raise cli_mod.UsageError("synthetic usage problem")

    monkeypatch.setattr(cli_mod, "make_backtest_config", usage)
    fake = _OrderedStderr(log)
    monkeypatch.setattr(sys, "stderr", fake)
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 2
    assert fake.seen == ["operational_error", "usage_error"]


def test_fatal_and_unexpected_are_recorded_before_they_are_printed(golden, tmp_path, keys, monkeypatch):  # noqa: F811
    conf = _conf(tmp_path, golden)
    log = tmp_path / "run" / "errors.log"

    def bad(*a, **k):
        raise ValueError("oops")

    monkeypatch.setattr(cli_mod, "make_backtest_config", bad)
    fake = _OrderedStderr(log)
    monkeypatch.setattr(sys, "stderr", fake)
    assert cli(["backtest", "--config", conf], GoldenStub()) == 1
    assert fake.seen == ["unexpected_exception"]


def test_one_error_log_instance_is_shared_by_every_failure_log_and_the_runner(golden, tmp_path, keys,  # noqa: F811
                                                                           monkeypatch):
    made, runner_got = [], []

    class Spy(FailureLog):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    real_run = cli_mod.run_backtest

    def spy_run(*a, **k):
        runner_got.append(k.get("error_log"))
        return real_run(*a, **k)

    monkeypatch.setattr(cli_mod, "FailureLog", Spy)
    monkeypatch.setattr(cli_mod, "run_backtest", spy_run)
    assert cli(["backtest", "--config", _conf(tmp_path, golden)], GoldenStub()) == 0
    active = cli_mod.ACTIVE_ERRORLOG
    assert active is not None and active.path == tmp_path / "run" / "errors.log"
    assert len(made) == 1 and made[0].error_log is active
    assert runner_got == [active]


def test_build_run_wires_run_errors_into_failures(golden, tmp_path):  # noqa: F811
    from atlas.engine.llm import StubClient
    rc = cli_mod.load_run_config(_conf(tmp_path, golden))
    run = cli_mod.build_run(rc, llm_client=StubClient(GoldenStub()))
    assert run.failures.error_log is run.errors and run.errors is cli_mod.ACTIVE_ERRORLOG
    assert run.errors.path == rc.path(cli_mod.ERRORS_NAME)


def test_replay_and_probe_failure_logs_get_the_same_instance(golden, tmp_path, keys, monkeypatch):  # noqa: F811
    from atlas.engine.llm import StubClient
    from tests.golden.test_golden_backtest import copy_run
    run_dir, conf = copy_run(golden, tmp_path)
    made = []

    class Spy(FailureLog):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(cli_mod, "FailureLog", Spy)
    replay = cli_mod.build_replay(conf, llm_client=StubClient(GoldenStub()))
    assert len(made) == 2 and all(f.error_log is replay.run.errors for f in made)
    assert made[1].db_path.name == cli_mod.AR_FAILURES
    made.clear()
    wrong = json.dumps({"date": "1990-01-01", "ticker": "ZZZZ"})
    assert cli(["probe", "--run-dir", run_dir, "--samples", "3", "--yes"], lambda s, u: wrong) == 0
    assert len(made) == 1 and made[0].db_path.name == cli_mod.PROBE_FAILURES
    assert made[0].error_log is cli_mod.ACTIVE_ERRORLOG and made[0].error_log.path == run_dir / "errors.log"


def test_threads_logging_through_two_failure_logs_never_interleave(tmp_path):
    shared = ErrorLog(tmp_path / "errors.log")
    logs = [FailureLog(tmp_path / f"f{i}.sqlite", error_log=shared) for i in range(2)]
    from datetime import date

    def work(fl, tag):
        for i in range(100):
            fl.record(FailureRecord(agent=tag, date=date(2024, 1, 2), reason=FailureReason.TIMEOUT,
                                    detail="d" * 200 + str(i), attempt=1, raw_response="raw"))

    ts = [threading.Thread(target=work, args=(logs[i % 2], f"t{i}")) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lines = (tmp_path / "errors.log").read_text().splitlines()
    assert len(lines) == 400 and all(json.loads(ln)["kind"] == "call_failure" for ln in lines)


def test_dry_run_log_lands_only_in_dry_run_dir(golden, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    conf = _conf(tmp_path, golden)
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 0
    assert cli_mod.ACTIVE_ERRORLOG.path == tmp_path / "run" / "dry_run" / "errors.log"
    assert not (tmp_path / "run" / "errors.log").exists()


def test_report_errors_section_missing_empty_and_torn_log(golden, tmp_path):  # noqa: F811
    run_dir, _ = copy_run(golden, tmp_path)
    log = run_dir / "errors.log"
    log.unlink(missing_ok=True)

    def errors_section():
        assert cli(["report", "--run-dir", run_dir]) == 0
        md = (run_dir / "report" / "report.md").read_text()
        return md.split("## Errors")[1].split("\n## ")[0]

    assert "not provided" in errors_section()
    log.write_text("")
    assert "no errors logged" in errors_section()
    good = ('{"ts_utc":"t","level":"error","kind":"agent_failure","day":null,"agent":"cro",'
            '"reason":"r","message":"hello","fields":{},"run_id":null}')
    log.write_text("garbage\n" + good + "\n" + '{"torn":')  # bad middle line, torn tail
    s = errors_section()
    assert "agent_failure" in s and "hello" in s
