"""`atlas probe` (optional): contamination probe CLI on the golden run. No network."""
import json
import re
import shutil
from pathlib import Path

import pytest

from atlas.cli import build_parser, spend_so_far
from tests.golden.test_golden_backtest import (
    FAKE_ANTHROPIC, cli, golden, keys, write_config,  # noqa: F401 (fixtures)
)

CANARY = "sk-ant-api03-CANARYCANARYCANARY0123456789"


def wrong(system, user):
    return json.dumps({"date": "1990-01-01", "ticker": "ZZZZ"})


@pytest.fixture
def run(golden, tmp_path):  # noqa: F811
    dest = tmp_path / "run"
    shutil.copytree(golden["run_dir"], dest)
    return dest


def test_probe_help_marks_the_command_optional(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["probe", "--help"])
    out = capsys.readouterr().out
    assert "(optional)" in out and "--run-dir" in out and "--samples" in out and "--yes" in out
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--help"])
    assert "(optional)" in capsys.readouterr().out  # the top-level listing says it too


def test_probe_writes_contamination_json_and_spend_file(run, keys, capsys):  # noqa: F811
    assert cli(["probe", "--run-dir", run, "--samples", "6", "--yes"], wrong) == 0
    c = json.loads((run / "contamination.json").read_text())
    assert c["n"] == 6 and c["rate"] == 0.0 and c["date_hits"] == 0 and c["ticker_hits"] == 0
    assert c["tolerance_days"] == 5 and c["model"] == "golden-stub-model" and isinstance(c["seed"], int)
    assert c["spent_usd"] > 0 and len(c["samples"]) == 6
    assert json.loads((run / "probe.json").read_text())["spent_usd"] == pytest.approx(c["spent_usd"])
    assert spend_so_far(run)["probe"] == pytest.approx(c["spent_usd"])
    out = capsys.readouterr().out
    assert "contamination probe" in out.lower() and "6" in out and "(optional)" in out
    assert FAKE_ANTHROPIC not in out


def test_probe_selection_is_deterministic_and_spread(run, keys, tmp_path):  # noqa: F811
    assert cli(["probe", "--run-dir", run, "--samples", "8", "--yes"], wrong) == 0
    a = json.loads((run / "contamination.json").read_text())
    other = tmp_path / "run2"
    shutil.copytree(run, other)
    (other / "contamination.json").unlink()
    assert cli(["probe", "--run-dir", other, "--samples", "8", "--yes"], wrong) == 0
    b = json.loads((other / "contamination.json").read_text())
    assert a["samples"] == b["samples"] and a["seed"] == b["seed"]
    days = {s["date"] for s in a["samples"]}
    tickers = {s["ticker"] for s in a["samples"]}
    assert len(days) == 8 and len(tickers) >= 4  # spread over days and over the universe


def test_probe_prompts_carry_no_real_ticker_or_date(run, keys):  # noqa: F811
    seen = []

    def spy(system, user):
        seen.append(system + "\n" + user)
        return wrong(system, user)

    assert cli(["probe", "--run-dir", run, "--samples", "5", "--yes"], spy) == 0
    assert len(seen) == 5
    c = json.loads((run / "contamination.json").read_text())
    for msg in seen:
        assert not re.search(r"\d{4}-\d{2}-\d{2}", msg)
        for t in ("SPY", "SMH", "XLF", "XLE", "XLV", "AAA", "BBB", "CCC"):
            assert not re.search(rf"\b{t}\b", msg)
        for s in c["samples"]:
            assert s["date"] not in msg


def test_probe_stops_at_the_shared_budget_cap_with_prior_spend(run, keys, capsys):  # noqa: F811
    left = 50.0 - spend_so_far(run)["backtest"]
    (run / "control.json").write_text(json.dumps({"spent_usd": left - 0.0001, "days": 0}))  # $0.0001 left
    rc = cli(["probe", "--run-dir", run, "--samples", "6", "--yes"], wrong)
    c = json.loads((run / "contamination.json").read_text())
    assert c["budget_exhausted"] is True and c["n"] < 6
    assert sum(spend_so_far(run).values()) <= 50.0 + 1e-9
    assert rc == 0 and "budget" in capsys.readouterr().out.lower()


def test_probe_refuses_before_any_call_when_budget_already_spent(run, keys, capsys):  # noqa: F811
    (run / "autoresearch_log.jsonl").write_text(json.dumps({"cost_usd": 60.0}) + "\n")
    calls = []
    assert cli(["probe", "--run-dir", run, "--samples", "3", "--yes"],
               lambda s, u: (calls.append(1), wrong(s, u))[1]) == 1
    assert not calls and "budget exhausted" in "".join(capsys.readouterr())
    assert not (run / "contamination.json").exists()


def test_probe_spend_counts_for_a_later_backtest_resume_and_report_total(run, keys):  # noqa: F811
    assert cli(["probe", "--run-dir", run, "--samples", "4", "--yes"], wrong) == 0
    spent = spend_so_far(run)
    assert set(spent) == {"backtest", "control", "autoresearch", "probe"} and spent["probe"] > 0


def test_probe_never_prints_or_stores_the_key(run, monkeypatch, capsys):  # noqa: F811
    monkeypatch.setenv("ANTHROPIC_API_KEY", CANARY)
    assert cli(["probe", "--run-dir", run, "--samples", "3", "--yes"], wrong) == 0
    assert CANARY not in "".join(capsys.readouterr())
    for p in run.rglob("*"):
        if p.is_file() and p.suffix in (".json", ".jsonl", ".md", ".csv"):
            assert CANARY not in p.read_text(errors="ignore")


def test_probe_missing_key_is_a_usage_error(run, monkeypatch, capsys):  # noqa: F811
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main_probe(run, client=None) == 2
    assert "ANTHROPIC_API_KEY" in "".join(capsys.readouterr())


def main_probe(run, client):
    from atlas.cli import main
    return main(["probe", "--run-dir", str(run), "--samples", "2", "--yes"], llm_client=client)


def test_probe_refuses_when_run_config_is_missing(tmp_path, keys, capsys):  # noqa: F811
    d = tmp_path / "empty"
    d.mkdir()
    assert cli(["probe", "--run-dir", d, "--yes"], wrong) == 1
    err = "".join(capsys.readouterr())
    assert "run_config.json" in err and "Traceback" not in err
    assert cli(["probe", "--run-dir", tmp_path / "nope", "--yes"], wrong) == 1


def test_probe_all_unusable_replies_is_an_error_not_a_zero_rate(run, keys, capsys):  # noqa: F811
    assert cli(["probe", "--run-dir", run, "--samples", "3", "--yes"], lambda s, u: "garbage") == 1
    assert "unusable" in "".join(capsys.readouterr()).lower()
    c = json.loads((run / "contamination.json").read_text())
    assert c["failed"] == 3 and c["n"] == 3
    assert "malformed_json" in set(__import__("atlas.engine.failures", fromlist=["FailureLog"])
                                   .FailureLog(run / "probe_failures.sqlite").rows()["reason"])


def test_probe_asks_for_confirmation_above_the_ceiling(run, keys, capsys, monkeypatch):  # noqa: F811
    st = json.loads((run / "run_config.json").read_text())
    st["settings"]["confirm_cost_usd"] = 0.0
    (run / "run_config.json").write_text(json.dumps(st))
    calls = []
    from atlas.cli import main
    from atlas.engine.llm import StubClient
    rc = main(["probe", "--run-dir", str(run), "--samples", "3"],
              llm_client=StubClient(lambda s, u: (calls.append(1), wrong(s, u))[1]))
    assert rc == 2 and not calls and "--yes" in "".join(capsys.readouterr())


def test_report_includes_the_contamination_section_after_a_probe_and_omits_it_before(run, keys):  # noqa: F811
    assert cli(["report", "--run-dir", run]) == 0
    assert "Contamination probe" not in (run / "report" / "report.md").read_text()
    assert cli(["probe", "--run-dir", run, "--samples", "4", "--yes"], wrong) == 0
    assert cli(["report", "--run-dir", run]) == 0
    md = (run / "report" / "report.md").read_text()
    assert "## Contamination probe (optional)" in md and "n = 4 probes" in md
    # the section is the only difference in the verdict-bearing part
    assert md.split("## Verdict")[1].split("\n## ")[0].count("No evidence of skill") >= 1
