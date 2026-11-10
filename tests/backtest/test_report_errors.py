"""The report's `## Errors` section: not provided / none / counts and last ten, escaping, and a
Verdict section that never depends on the error log."""
from tests.backtest.test_report import build


def ev(i=0, **kw):
    rec = {"ts_utc": f"2026-01-01T00:00:{i:02d}+00:00", "level": "error", "kind": "agent_failure",
           "day": "2026-01-02", "agent": "cro", "reason": "boom", "message": f"m{i}",
           "fields": {}, "run_id": "r"}
    rec.update(kw)
    return rec


def section(md):
    return md.split("## Errors")[1].split("\n## ")[0]


def test_errors_section_absent_value_says_not_provided(tmp_path):
    md = build(tmp_path / "a").read_text()
    assert "not provided" in section(md)
    assert md.index("## Failures") < md.index("## Errors") < md.index("## Cost")
    md = build(tmp_path / "b", errors=[]).read_text()
    assert "no errors logged" in section(md)
    assert "not provided" not in section(md)


def test_errors_section_counts_and_last_ten(tmp_path):
    events = [ev(i, kind="a" if i % 2 else "b", level="warning" if i < 3 else "error")
              for i in range(15)]
    s = section(build(tmp_path, errors=events).read_text())
    assert "errors.log" in s
    assert "| a | 7 |" in s and "| b | 8 |" in s
    assert "| warning | 3 |" in s and "| error | 12 |" in s
    assert "m14" in s and "m5" in s and "m4" not in s


def test_errors_section_escapes_pipes_and_newlines(tmp_path):
    bad = ev(0, message="a|b\nc\x1b[31m‮x​y", kind="k|1", agent="sk-ant-" + "A" * 40)
    s = section(build(tmp_path, errors=[bad, "junk", {"level": ["x"], "kind": None}]).read_text())
    assert "a\\|b c" in s and "k\\|1" in s
    assert "\x1b" not in s and "‮" not in s and "​" not in s
    assert "\\x1b" in s and "\\u202e" in s
    assert "sk-ant-" + "A" * 40 not in s
    assert all(len(ln) < 400 for ln in s.splitlines())


def test_verdict_section_unchanged_with_and_without_errors(tmp_path):
    def verdict(md):
        return md.split("## Verdict")[1].split("\n## ")[0]
    a = verdict(build(tmp_path / "a").read_text())
    b = verdict(build(tmp_path / "b", errors=[ev(i) for i in range(30)]).read_text())
    c = verdict(build(tmp_path / "c", errors=[]).read_text())
    assert a == b == c

