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
    assert set(spent) == {"backtest", "control", "autoresearch", "probe", "older_model"} and spent["probe"] > 0


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
    assert "## Contamination probe (optional)" in md and "n = 4 probes" in md and "Chance level" in md
    # the section is the only difference in the verdict-bearing part
    assert md.split("## Verdict")[1].split("\n## ")[0].count("No evidence of skill") >= 1


# ---------------------------------------------------------------------------- fix round 1

def test_unique_sector_labels_are_never_sent_only_shared_ones():
    from atlas.cli import probe_sector_labels
    assert set(probe_sector_labels(["SPY", "SMH", "XLF", "XLE", "XLV", "AAA"]).values()) == {"Unclassified"}
    lab = probe_sector_labels(["SMH", "XLF", "NVDA"])
    assert lab == {"SMH": "Unclassified", "XLF": "Unclassified", "NVDA": "Unclassified"}
    from atlas import cli as c
    old = dict(c.PROBE_SECTORS)
    try:
        c.PROBE_SECTORS.update({"AAA": "Tech", "BBB": "Tech", "CCC": "Energy"})
        assert c.probe_sector_labels(["AAA", "BBB", "CCC"]) == {"AAA": "Tech", "BBB": "Tech", "CCC": "Unclassified"}
    finally:
        c.PROBE_SECTORS.clear()
        c.PROBE_SECTORS.update(old)


def test_probe_cli_sends_only_unclassified_and_headlines_the_date_rate(run, keys, capsys):  # noqa: F811
    seen = []

    def spy(system, user):
        seen.append(user)
        return wrong(system, user)

    assert cli(["probe", "--run-dir", run, "--samples", "5", "--yes"], spy) == 0
    assert {json.loads(u)["instruments"][0]["sector"] for u in seen} == {"Unclassified"}
    c = json.loads((run / "contamination.json").read_text())
    assert c["date_rate"] == 0.0 and c["ticker_rate"] == 0.0
    assert c["chance_date_rate"] == pytest.approx(11 / c["n_trading_days"]) and c["chance_ticker_rate"] == 1 / 8
    out = capsys.readouterr().out
    assert "date-hit rate 0.0%" in out and "about 27.5%" in out and "approximate" in out
    assert "not evidence of date memory" in out and "trust results for those periods less" in out
    assert sum(v["n"] for v in c["by_period"].values()) == c["n"]


def test_probe_selection_avoids_thin_history_days(run, keys):  # noqa: F811
    assert cli(["probe", "--run-dir", run, "--samples", "6", "--yes"], wrong) == 0
    c = json.loads((run / "contamination.json").read_text())
    # the price history starts 2024-01-01: earlier days have fewer than 20 bars
    assert all(s["date"] >= "2024-01-26" for s in c["samples"])


def test_probe_refuses_a_dry_run_directory(golden, tmp_path, keys, capsys):  # noqa: F811
    conf = write_config(tmp_path / "r.yaml", tmp_path / "run", golden["repo"])
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 0
    capsys.readouterr()
    assert cli(["probe", "--run-dir", tmp_path / "run" / "dry_run", "--yes"], wrong) == 2
    assert "dry run" in "".join(capsys.readouterr())
    other = tmp_path / "copy"  # renamed, but the dry-run marker file gives it away
    shutil.copytree(tmp_path / "run" / "dry_run", other)
    (other / "dry_run_llm.db").write_bytes(b"")  # the marker the runner leaves in a dry-run dir
    assert cli(["probe", "--run-dir", other, "--yes"], wrong) == 2


@pytest.mark.parametrize("name", ["probe.json", "control.json", "run_config.json"])
@pytest.mark.parametrize("content", ["{not json", "[]", '{"spent_usd": "x"}'])
def test_corrupt_state_files_are_a_clear_exit_1_without_traceback(run, keys, capsys, name, content):  # noqa: F811
    (run / name).write_text(content)
    assert cli(["probe", "--run-dir", run, "--samples", "2", "--yes"], wrong) == 1
    err = "".join(capsys.readouterr())
    assert f"{name} is corrupt" in err and "Traceback" not in err
    if name == "run_config.json":
        assert cli(["report", "--run-dir", run]) == 1


# --- the one hard cap, handed to every consumer (mutation-guarded)

def _plant_spend(run: Path) -> dict:
    """control 1.0, autoresearch 2.0, probe 3.0, older-model 4.0 on top of the backtest's own."""
    (run / "control.json").write_text(json.dumps({"spent_usd": 1.0, "days": 0}))
    row = {"agent": "a", "branch": "b", "n_days": 20, "old_ic": 0.0, "new_ic": 0.0, "diff": 0.0,
           "keep_margin": 0.1, "decision": "rejected", "reason": "x", "commit_sha": None, "cost_usd": 2.0}
    (run / "autoresearch_log.jsonl").write_text(json.dumps(row) + "\n")
    (run / "probe.json").write_text(json.dumps({"spent_usd": 3.0}))
    (run / "older_model" / "m").mkdir(parents=True)
    (run / "older_model" / "m" / "checkpoint.json").write_text(json.dumps({"spent_usd": 4.0}))
    return {"control": 1.0, "autoresearch": 2.0, "probe": 3.0, "older_model": 4.0}


def test_spend_buckets_are_exact_and_each_counted_once(run):  # noqa: F811
    own = spend_so_far(run)["backtest"]
    planted = _plant_spend(run)
    assert spend_so_far(run) == {"backtest": own, **planted}
    assert sum(spend_so_far(run).values()) == pytest.approx(own + 10.0)


def test_every_consumer_gets_the_budget_minus_everyone_elses_spend(golden, tmp_path, keys, monkeypatch):  # noqa: F811
    import atlas.cli as c
    run = tmp_path / "run"
    shutil.copytree(golden["run_dir"], run)
    own = spend_so_far(run)["backtest"]
    _plant_spend(run)
    budget = 100.0
    conf = write_config(tmp_path / "c.yaml", run, golden["repo"], budget_usd=budget)
    seen = {"resume": None, "ctor": []}
    real_run, real_llm = c.run_backtest, c.CachedLLM

    def spy_run(cfg, store, pipeline, *a, **k):
        seen["resume"] = pipeline.llm.budget_usd
        return real_run(cfg, store, pipeline, *a, **k)

    class SpyLLM(real_llm):
        def __init__(self, client, cache, model, pricing, budget_usd, **k):
            seen["ctor"].append(budget_usd)
            super().__init__(client, cache, model, pricing, budget_usd, **k)

    monkeypatch.setattr(c, "run_backtest", spy_run)
    monkeypatch.setattr(c, "CachedLLM", SpyLLM)
    assert cli(["backtest", "--config", conf, "--yes", "--with-control"], wrong) in (0, 1)
    # backtest resume: everything but its own spend is subtracted (control 1 + ar 2 + probe 3 + older 4)
    assert seen["resume"] == pytest.approx(budget - 10.0)
    # control pass: cap minus backtest, its own previous spend and the other three buckets
    assert seen["ctor"][-1] == pytest.approx(budget - own - 1.0 - (2.0 + 3.0 + 4.0))
    # probe: cap minus the whole run's spend, once
    seen["ctor"].clear()
    before = sum(spend_so_far(run).values())
    assert cli(["probe", "--run-dir", run, "--samples", "2", "--yes"], wrong) == 0
    assert seen["ctor"][-1] == pytest.approx(budget - before)


def test_report_total_counts_older_model_and_probe_spend_exactly_once(golden, tmp_path, keys):  # noqa: F811
    run = tmp_path / "run"
    shutil.copytree(golden["run_dir"], run)
    own = spend_so_far(run)["backtest"]
    _plant_spend(run)
    assert cli(["report", "--run-dir", run]) == 0
    md = (run / "report" / "report.md").read_text()
    assert f"Total API cost: ${own + 10.0:.2f}" in md
