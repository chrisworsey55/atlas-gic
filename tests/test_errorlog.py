import json
import os
import stat
import threading
from datetime import date

import pytest

from atlas.errorlog import ErrorLog, NullErrorLog, read_events

CANARY = "sk-ant-api03-CANARY"
KEYS = {"ts_utc", "level", "kind", "day", "agent", "reason", "message", "fields", "run_id"}


def test_event_appends_one_json_line_with_all_keys(tmp_path):
    p = tmp_path / "errors.log"
    log = ErrorLog(p, run_id="r1")
    log.event("llm_failed", "boom", day=date(2026, 1, 5), agent="macro", reason="timeout", n=3)
    lines = p.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert set(rec) == KEYS
    assert rec["kind"] == "llm_failed" and rec["level"] == "error"
    assert rec["day"] == "2026-01-05" and rec["agent"] == "macro" and rec["reason"] == "timeout"
    assert rec["fields"] == {"n": 3} and rec["run_id"] == "r1" and rec["message"] == "boom"
    assert read_events(p, kind="llm_failed") == [rec] and read_events(p, kind="x") == []


def test_strings_are_sanitised_and_truncated(tmp_path):
    p = tmp_path / "errors.log"
    log = ErrorLog(p)
    log.event("k", f"key {CANARY} " + "x" * 5000,
              url=f"https://h.example/a?api_key=CANARY", nested={"a": [f"Bearer {CANARY}", 1]},
              obj=object(), day="2026-01-05")
    raw = p.read_text()
    assert "CANARY" not in raw
    rec = json.loads(raw)
    assert len(rec["message"]) <= 500
    assert rec["fields"]["nested"]["a"][1] == 1
    assert isinstance(rec["fields"]["obj"], str)


def test_exception_has_sanitised_traceback_without_locals(tmp_path):
    p = tmp_path / "errors.log"
    log = ErrorLog(p)

    def inner():
        secret = CANARY  # noqa: F841
        raise ValueError(f"bad {CANARY}")

    try:
        inner()
    except ValueError as e:
        log.exception("crash", e, agent="a")
    raw = p.read_text()
    assert "CANARY" not in raw
    rec = json.loads(raw)
    assert rec["fields"]["exc_type"] == "ValueError"
    frames = rec["fields"]["frames"]
    assert 1 <= len(frames) <= 8
    assert any(f.startswith("test_errorlog.py:") and f.endswith(" inner") for f in frames)
    assert all("/" not in f for f in frames)


def test_exception_frames_capped_at_8(tmp_path):
    p = tmp_path / "errors.log"

    def rec_(n):
        if n == 0:
            raise RuntimeError("deep")
        rec_(n - 1)

    try:
        rec_(30)
    except RuntimeError as e:
        ErrorLog(p).exception("deep", e)
    assert len(json.loads(p.read_text())["fields"]["frames"]) == 8


def test_never_raises_when_directory_is_read_only_or_path_is_a_directory(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("atlas.errorlog._warned", False)
    d = tmp_path / "errors.log"
    d.mkdir()
    log = ErrorLog(d)
    log.event("a", "x")
    log.event("b", "y")
    err = capsys.readouterr().err
    assert err.count("warning: could not write error log (") == 1
    assert err.startswith("warning: could not write error log (IsADirectoryError)")
    # unserialisable / bad inputs also never raise
    ErrorLog(tmp_path / "ok.log").event("k", None, level=None, day=object())  # type: ignore


def test_concurrent_threads_write_whole_lines(tmp_path):
    p = tmp_path / "errors.log"
    log = ErrorLog(p)

    def work(t):
        for i in range(50):
            log.event("k", "m" * 200, t=t, i=i)

    ts = [threading.Thread(target=work, args=(t,)) for t in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    lines = p.read_text().splitlines()
    assert len(lines) == 800
    assert all(json.loads(ln)["kind"] == "k" for ln in lines)


def test_rotates_at_max_bytes_keeping_one_backup(tmp_path):
    p = tmp_path / "errors.log"
    log = ErrorLog(p, max_bytes=1000)
    for i in range(60):
        log.event("k", "m" * 100, i=i)
    b = tmp_path / "errors.log.1"
    assert b.exists() and not (tmp_path / "errors.log.2").exists()
    assert p.stat().st_size <= 1000 and b.stat().st_size <= 1000
    ids = [e["fields"]["i"] for e in read_events(b)] + [e["fields"]["i"] for e in read_events(p)]
    assert ids == sorted(ids) and ids[-1] == 59
    assert stat.S_IMODE(b.stat().st_mode) == 0o600


def test_file_mode_is_0600(tmp_path):
    p = tmp_path / "errors.log"
    ErrorLog(p).event("k")
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    q = tmp_path / "pre.log"
    q.write_text("")
    os.chmod(q, 0o644)
    ErrorLog(q).event("k")
    assert stat.S_IMODE(q.stat().st_mode) == 0o600


def test_read_events_skips_torn_final_line_but_raises_on_earlier_corruption(tmp_path):
    p = tmp_path / "errors.log"
    good = json.dumps({"kind": "k"})
    p.write_text(good + "\n" + '{"kind": "tor')
    assert read_events(p) == [{"kind": "k"}]
    p.write_text(good + "\n" + "garbage\n" + good + "\n")
    with pytest.raises(ValueError):
        read_events(p)
    assert read_events(tmp_path / "missing.log") == []


def test_null_error_log_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for log in (NullErrorLog(), ErrorLog(None)):
        log.event("k", "m")
        log.exception("k", ValueError("x"))
    assert list(tmp_path.iterdir()) == []


def test_unknown_level_is_coerced_to_error(tmp_path):
    p = tmp_path / "errors.log"
    log = ErrorLog(p)
    log.event("a", level="warning")
    log.event("b", level="bogus")
    assert [e["level"] for e in read_events(p)] == ["warning", "error"]


class Bad:
    def __str__(self):
        raise RuntimeError("no str")


class BadExc(Exception):
    def __str__(self):
        raise RuntimeError("no str")


def test_unprintable_values_degrade_but_record_is_written(tmp_path):
    p = tmp_path / "errors.log"
    log = ErrorLog(p)
    log.event("k", "m", bad=Bad(), nested={Bad(): [Bad()]})
    log.event("k2", Bad())
    log.event(Bad(), "m", agent=Bad(), reason=Bad(), day=Bad())
    try:
        raise BadExc()
    except BadExc as e:
        log.exception("x", e)
    recs = read_events(p)
    assert len(recs) == 4
    assert recs[0]["fields"]["bad"] == "<unprintable Bad>"
    assert recs[1]["message"] == "<unprintable Bad>"
    assert recs[2]["kind"] == "<unprintable Bad>" and recs[2]["day"] == "<unprintable Bad>"
    assert recs[3]["fields"]["exc_type"] == "BadExc"
    assert recs[3]["message"] == "<unprintable BadExc>"


def test_every_string_field_is_sanitised(tmp_path):
    p = tmp_path / "errors.log"
    ErrorLog(p).event(f"k-{CANARY}", f"m-{CANARY}", agent=f"a-{CANARY}", reason=f"r-{CANARY}",
                      day=f"d-{CANARY}", **{f"f-{CANARY}": f"v-{CANARY}"})
    assert "CANARY" not in p.read_text()


def test_exception_message_present_and_redacted(tmp_path):
    p = tmp_path / "errors.log"
    ErrorLog(p).exception("x", ValueError(f"disk full key {CANARY} at https://h.example/p?api_key=CANARY"))
    rec = read_events(p)[0]
    assert rec["message"].startswith("disk full key <redacted>")
    assert "CANARY" not in p.read_text()


def test_each_event_is_fsynced(tmp_path, monkeypatch):
    calls = []
    real = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (calls.append(fd), real(fd))[1])
    log = ErrorLog(tmp_path / "errors.log")
    for _ in range(3):
        log.event("k")
    assert len(calls) == 3


def test_rotation_boundary_is_exact(tmp_path):
    probe = tmp_path / "probe.log"
    ErrorLog(probe).event("k", "m" * 50)
    n = probe.stat().st_size
    p = tmp_path / "errors.log"
    log = ErrorLog(p, max_bytes=2 * n)
    log.event("k", "m" * 50)
    log.event("k", "m" * 50)
    assert not (tmp_path / "errors.log.1").exists() and p.stat().st_size == 2 * n
    log.event("k", "m" * 50)
    assert (tmp_path / "errors.log.1").exists() and p.stat().st_size == n


def test_concurrent_writes_across_rotation_lose_nothing(tmp_path):
    p = tmp_path / "errors.log"
    log = ErrorLog(p, max_bytes=200_000)
    errs = []

    def work(t):
        try:
            for i in range(50):
                log.event("k", "m" * 100, t=t, i=i)
        except Exception as e:  # pragma: no cover
            errs.append(e)

    ts = [threading.Thread(target=work, args=(t,)) for t in range(16)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs
    b = tmp_path / "errors.log.1"
    assert b.exists()
    recs = read_events(b) + read_events(p)  # raises on any malformed line
    assert len(recs) == 800
    assert len({(r["fields"]["t"], r["fields"]["i"]) for r in recs}) == 800


def test_torn_line_is_repaired_before_next_append(tmp_path):
    p = tmp_path / "errors.log"
    good = json.dumps({"kind": "old"})
    p.write_text(good + "\n" + '{"kind":"tor')
    os.chmod(p, 0o600)
    ErrorLog(p).event("a")
    ErrorLog(p).event("b")
    kinds = [e["kind"] for e in read_events(p)]
    assert kinds == ["old", "log_repaired", "a", "b"]
    assert read_events(p, kind="b")[0]["kind"] == "b"
    # a malformed line NOT followed by a marker is still corruption
    p.write_text(good + "\nnot json\n" + good + "\n")
    with pytest.raises(ValueError):
        read_events(p)


def test_rotation_failure_falls_back_to_live_file(tmp_path, monkeypatch):
    monkeypatch.setattr("atlas.errorlog._warned", False)
    p = tmp_path / "errors.log"
    (tmp_path / "errors.log.1").mkdir()
    log = ErrorLog(p, max_bytes=300)
    for i in range(10):
        log.event("k", "m" * 100, i=i)
    assert [e["fields"]["i"] for e in read_events(p)] == list(range(10))


def test_read_events_directory_and_blank_lines(tmp_path):
    d = tmp_path / "d"
    d.mkdir()
    assert read_events(d) == []
    p = tmp_path / "errors.log"
    g = json.dumps({"kind": "k"})
    p.write_text("\n" + g + "\n\n   \n" + g + "\n\n")
    assert len(read_events(p)) == 2
