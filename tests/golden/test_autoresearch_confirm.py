"""Validation confirmation through the CLI and the report: the short-validation refusal, an
end-to-end run on an 80-day golden market (train 45, validation 25, holdout 10) with a stub that
can "edit" a prompt, and the report's confirmation columns and count."""
import csv
import json
import re
from datetime import date
from pathlib import Path

import pytest

from atlas.backtest.report import build_report
from atlas.autoresearch.loop import EditRecord
from tests.backtest.test_report import build
from tests.golden.stub import GoldenStub, agent_of, respond
from tests.golden.test_golden_backtest import (
    FAKE_ANTHROPIC, FIX, cli, copy_run, golden, keys, make_prompt_repo, run_state, write_config,  # noqa: F401
)

LONG = FIX / "prices_long"
LONG_OVER = {
    "periods": {"train": {"start": "2024-01-01", "end": "2024-03-01"},
                "validation": {"start": "2024-03-04", "end": "2024-04-05"},
                "holdout": {"start": "2024-04-08", "end": "2024-04-19"}},
    "data": {"prices": {"dir": str(LONG)}},
    "settings": {"max_edits": 1, "confirm_on_validation": True},
}
MARK = "HINDSIGHT-STUB-EDIT"


def _closes() -> dict[str, dict[str, float]]:
    out = {}
    for f in LONG.glob("*.csv"):
        with open(f) as fh:
            rows = list(csv.DictReader(fh))
        out[f.stem] = {r["date"]: float(r["close"]) for r in rows}
    return out


class HindsightStub(GoldenStub):
    """GoldenStub plus an LLM 'proposer': a proposal appends MARK to the prompt, and an agent whose
    prompt carries MARK picks the same tickers but signs/sizes convictions by the realised next-day
    excess over SPY (hindsight, a deliberately cheating stub). With ``train_only`` the hindsight
    only exists on days up to ``train_end`` (the edit then fails on validation)."""

    def __init__(self, train_only: bool = False, train_end: str = "2024-03-01"):
        super().__init__()
        self.closes = _closes()
        self.dates = sorted(self.closes["SPY"])
        self.train_only, self.train_end = train_only, train_end
        self.proposals = 0

    def __call__(self, system: str, user: str) -> str:
        if system.startswith("You improve the system prompt"):
            self.proposals += 1
            cur = user.split("<<<PROMPT\n", 1)[1].rsplit("\nPROMPT>>>", 1)[0]
            return cur.rstrip("\n") + f"\n\n{MARK}: size convictions by what you expect.\n"
        out = super().__call__(system, user)
        if MARK not in system:
            return out
        asof = json.loads(user)["asof"]
        if self.train_only and asof > self.train_end:
            return out
        i = self.dates.index(asof)
        if i + 1 >= len(self.dates):
            return out
        nxt = self.dates[i + 1]
        data = json.loads(out)
        if any(not 0 < p["conviction"] <= 1 for p in data["picks"]):
            return out
        r = lambda t: self.closes[t][nxt] / self.closes[t][asof] - 1  # noqa: E731
        for p in data["picks"]:
            if p["ticker"] not in self.closes:
                continue
            ex = r(p["ticker"]) - r("SPY")
            p["direction"] = "long" if ex >= 0 else "short"
            p["conviction"] = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9][min(5, int(abs(ex) / 0.004))]
        return json.dumps(data)


@pytest.fixture(scope="module")
def long_golden(tmp_path_factory):
    root = tmp_path_factory.mktemp("long_golden")
    repo = make_prompt_repo(root / "repo")
    run_dir = root / "run"
    conf = write_config(root / "run.yaml", run_dir, repo, **LONG_OVER)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
        assert cli(["backtest", "--config", conf], HindsightStub()) == 0
    from tests.autoresearch.conftest import git
    return {"root": root, "repo": repo, "run_dir": run_dir, "sha": git(repo, "rev-parse", "HEAD")}


def _copy(long_golden, dest: Path, **over):
    import shutil
    shutil.copytree(long_golden["run_dir"], dest / "run")
    from tests.autoresearch.conftest import git
    repo = long_golden["repo"]  # the stored run is bound to this path: reset it instead of copying
    git(repo, "checkout", "-q", "-f", "main")
    git(repo, "reset", "-q", "--hard", long_golden["sha"])
    for b in git(repo, "branch", "--list", "autoresearch/*").split():
        if b != "*":
            git(repo, "branch", "-q", "-D", b)
    conf = write_config(dest / "run.yaml", dest / "run", repo,
                        **{**LONG_OVER, **over, "settings": {**LONG_OVER["settings"], **over.get("settings", {})}})
    return dest / "run", repo, conf


# ---------------------------------------------------------------------------- CLI refusal

@pytest.fixture(scope="module")
def short_validation_run(tmp_path_factory):
    """The 40-day golden run with confirmation ON (golden.yaml turns it off): 10 validation days,
    9 whose exit is also inside validation."""
    root = tmp_path_factory.mktemp("short_val")
    repo = make_prompt_repo(root / "repo")
    conf = write_config(root / "run.yaml", root / "run", repo, settings={"confirm_on_validation": True})
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
        assert cli(["backtest", "--config", conf], GoldenStub()) == 0
    return {"root": root, "repo": repo, "run_dir": root / "run", "config": conf}


def test_cli_refuses_short_validation_with_actionable_message(short_validation_run, tmp_path, keys, capsys):
    stub = GoldenStub()
    head_before = (short_validation_run["run_dir"] / "run_config.json").read_bytes()
    assert cli(["autoresearch", "--config", short_validation_run["config"]], stub) == 1
    text = "".join(capsys.readouterr())
    assert "Traceback" not in text and stub.calls == 0
    assert "9 usable trading days" in text and "confirm_min_days 20" in text
    assert "settings.confirm_on_validation: false" in text
    assert "new run_dir" in text  # the setting is part of the stored run
    assert "rolling_ic_window / picks_per_day" in text and "reuses (spends) the same validation days" in text
    assert not (short_validation_run["run_dir"] / "autoresearch_log.jsonl").exists()
    assert (short_validation_run["run_dir"] / "run_config.json").read_bytes() == head_before


def test_cli_does_not_refuse_short_validation_when_confirmation_is_off(golden, tmp_path, keys, capsys):
    """golden.yaml has confirm_on_validation false: the command gets past the validation check and
    stops at the (separate) train-days requirement."""
    run_dir, conf = copy_run(golden, tmp_path)
    assert cli(["autoresearch", "--config", conf], GoldenStub()) == 1
    text = "".join(capsys.readouterr())
    assert "validation period has" not in text and "train days" in text


# ---------------------------------------------------------------------------- end to end

def _records(run_dir):
    return [json.loads(x) for x in (run_dir / "autoresearch_log.jsonl").read_text().splitlines()]


def test_cli_runs_confirmation_end_to_end_with_stub_model_on_a_long_golden_fixture(long_golden, tmp_path, keys, capsys):
    run_dir, repo, conf = _copy(long_golden, tmp_path)
    stub = HindsightStub()
    assert cli(["autoresearch", "--config", conf], stub) == 0
    text = "".join(capsys.readouterr())
    (rec,) = _records(run_dir)
    c = rec["confirm"]
    assert stub.proposals == 1
    assert c["n_days_requested"] == 24 and c["n_days"] >= 20 and c["passed"] is True
    assert c["diff"] > c["margin"] >= rec["keep_margin"] > 0
    assert rec["decision"] == "kept" and rec["commit_sha"]
    assert "confirmed on validation" in rec["reason"]
    prompt = (repo / "atlas" / "prompts" / (rec["agent"] + ".md")).read_text()
    assert MARK in prompt
    assert "Traceback" not in text
    # the report shows it
    assert cli(["report", "--run-dir", run_dir]) == 0
    md = (run_dir / "report" / "report.md").read_text()
    sec = md.split("## Autoresearch")[1].split("\n## ")[0]
    assert "Validation confirmations run: 1" in sec and rec["agent"] in sec and "kept" in sec


def test_edit_that_only_works_on_train_is_unconfirmed_and_the_prompt_is_untouched(long_golden, tmp_path, keys):
    run_dir, repo, conf = _copy(long_golden, tmp_path)
    from tests.autoresearch.conftest import git
    head = git(repo, "rev-parse", "HEAD")
    assert cli(["autoresearch", "--config", conf], HindsightStub(train_only=True)) == 0
    (rec,) = _records(run_dir)
    assert rec["decision"] == "unconfirmed" and rec["confirm"]["passed"] is False
    assert rec["confirm"]["n_days_requested"] == 24
    assert git(repo, "rev-parse", "HEAD") == head and git(repo, "status", "--porcelain") == ""


# ---------------------------------------------------------------------------- report

def _rec(agent="a1", decision="kept", confirm="default", **kw):
    if confirm == "default":
        confirm = {"n_days_requested": 24, "n_days": 24, "old_ic": 0.01, "new_ic": 0.4, "diff": 0.39,
                   "margin": 0.15, "new_pairs": 72, "passed": True, "reason": "validation diff 0.39 > margin 0.15"}
    return EditRecord(agent=agent, branch="b", n_days=40, old_ic=0.01, new_ic=0.3, diff=0.29, keep_margin=0.15,
                      decision=decision, reason="ok", commit_sha="abc", cost_usd=0.5, confirm=confirm, **kw)


def _ar(md):
    return md.split("## Autoresearch")[1].split("\n## ")[0]


def test_report_autoresearch_section_lists_confirmation_numbers(tmp_path):
    s = _ar(build(tmp_path, edits=[_rec("sector_energy")]).read_text())
    for col in ("validation days", "old IC", "new IC", "diff", "margin", "decision"):
        assert col in s
    row = next(x for x in s.splitlines() if x.startswith("| sector_energy"))
    assert "| kept |" in row and "| 24 |" in row and "0.010" in row and "0.400" in row and "0.390" in row
    assert "0.150" in row
    assert "repeatedly used sample" in s


def test_unconfirmed_decision_is_listed(tmp_path):
    c = {"n_days_requested": 0, "n_days": 0, "old_ic": None, "new_ic": None, "diff": None, "margin": None,
         "new_pairs": 0, "passed": False, "reason": "validation too short"}
    s = _ar(build(tmp_path, edits=[_rec("x", "unconfirmed", c)]).read_text())
    row = next(x for x in s.splitlines() if x.startswith("| x "))
    assert "unconfirmed" in row and "n/a" in row and "validation too short" in row
    assert "Validation confirmations run: 0 " in s  # requested 0 days: no replay ran


def test_confirmation_count_stated(tmp_path):
    edits = [_rec("a"), _rec("b", "unconfirmed"), _rec("c", "rejected", confirm=None), _rec("d", "reverted", None)]
    s = _ar(build(tmp_path, edits=edits).read_text())
    assert "Validation confirmations run: 2 (of 4 logged attempts" in s
    # whole log, not just the edits that were kept
    only_unconfirmed = _ar(build(tmp_path / "u", edits=[_rec("b", "unconfirmed")]).read_text())
    assert "Validation confirmations run: 1 " in only_unconfirmed


def test_old_logs_without_confirm_still_render(tmp_path):
    s = _ar(build(tmp_path, edits=[_rec("a", confirm=None)]).read_text())
    assert "Validation confirmations run: 0 (of 1" in s and "No logged edit has a validation confirmation" in s
    assert "a: **kept**" in s


def test_confirmation_cells_are_escaped(tmp_path):
    c = {"n_days_requested": 24, "n_days": 24, "old_ic": 0.0, "new_ic": 0.1, "diff": 0.1, "margin": 0.1,
         "new_pairs": 72, "passed": False, "reason": "a|b\nc\x1b[31m‮x sk-ant-" + "A" * 40}
    s = _ar(build(tmp_path, edits=[_rec("ag|ent", "unconfirmed", c)]).read_text())
    assert "ag\\|ent" in s and "a\\|b" in s
    assert "\x1b" not in s and "‮" not in s and "sk-ant-" + "A" * 40 not in s
    assert all(ln.count("\n") == 0 for ln in s.splitlines())


def test_verdict_section_unchanged_by_confirmation_records(tmp_path):
    v = lambda md: md.split("## Verdict")[1].split("\n## ")[0]  # noqa: E731
    a = v(build(tmp_path / "a", edits=[_rec("a", confirm=None)]).read_text())
    b = v(build(tmp_path / "b", edits=[_rec("a"), _rec("b", "unconfirmed")]).read_text())
    assert a == b
