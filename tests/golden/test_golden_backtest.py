"""Golden end-to-end tests: the `atlas` CLI on a committed 40-day synthetic market with a
scripted stub model (tests/golden/fixtures). No network, no real keys.

One full golden run is shared per module (``golden``); tests that write into a run directory
work on a copy.
"""
import io
import json
import shutil
import sqlite3
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pytest
import yaml

from atlas.backtest.checkpoint import Checkpoint
from atlas.backtest.nulltest import null_test
from atlas.backtest.runner import read_day_history
from atlas.cli import main
from atlas.engine.llm import StubClient
from atlas.scoring.store import ScoreStore
from tests.autoresearch.conftest import git
from tests.golden.stub import GoldenStub, SimulatedCrash

ROOT = Path(__file__).resolve().parents[2]
FIX = Path(__file__).parent / "fixtures"
N_AGENTS = 12
GOLDEN_DAYS = 40
HOLDOUT_END = date(2024, 2, 23)
FAKE_ANTHROPIC = "sk-ant-TEST-do-not-print-0123456789"
FAKE_FRED = "fred-TEST-do-not-print-abcdef"
SHORT_PERIODS = {"train": {"start": "2024-01-01", "end": "2024-01-03"},
                 "validation": {"start": "2024-01-04", "end": "2024-01-05"},
                 "holdout": {"start": "2024-02-05", "end": "2024-02-06"}}


# ---------------------------------------------------------------------------- helpers

def make_prompt_repo(root: Path) -> Path:
    """A throwaway git repo holding a copy of the real prompts (autoresearch edits it)."""
    (root / "atlas" / "prompts").mkdir(parents=True)
    for p in sorted((ROOT / "atlas" / "prompts").glob("*.md")):
        shutil.copy(p, root / "atlas" / "prompts" / p.name)
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "prompts")
    return root


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def write_config(path: Path, run_dir: Path, repo: Path, **over) -> Path:
    cfg = yaml.safe_load((FIX / "golden.yaml").read_text())
    cfg["run_dir"] = str(run_dir)
    cfg["repo"] = str(repo)
    cfg["data"]["prices"]["dir"] = str(FIX / "prices")
    cfg = _merge(cfg, over)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def cli(argv, stub=None) -> int:
    return main([str(a) for a in argv], llm_client=StubClient(stub) if stub is not None else None)


def run_state(run_dir: Path) -> dict:
    """Everything that must be identical between equivalent runs."""
    cp = json.loads((run_dir / "checkpoint.json").read_text())
    with sqlite3.connect(run_dir / "scores.sqlite") as c:
        rows = set(c.execute("SELECT * FROM picks").fetchall())
    return {"equity": cp["equity"], "spy": cp["spy"], "basket": cp["basket"],
            "portfolio": cp["portfolio"], "weights": cp["weights"], "skipped": cp["skipped"],
            "last_completed_day": cp["last_completed_day"], "scores": rows,
            "predlog": (run_dir / "prediction_log.jsonl").read_bytes(),
            "history": read_day_history(run_dir / "weights_history.jsonl", Checkpoint(run_dir / "checkpoint.json"))}


class FakeTTY(io.StringIO):
    def isatty(self):
        return True


class NotTTY(io.StringIO):
    def isatty(self):
        return False


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    monkeypatch.delenv("FRED_API_KEY", raising=False)


@pytest.fixture(scope="module")
def golden(tmp_path_factory):
    root = tmp_path_factory.mktemp("golden")
    repo = make_prompt_repo(root / "repo")
    run_dir = root / "run"
    conf = write_config(root / "run.yaml", run_dir, repo)
    stub = GoldenStub()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
        assert cli(["backtest", "--config", conf], stub) == 0
    return {"root": root, "repo": repo, "run_dir": run_dir, "config": conf, "calls": stub.calls,
            "state": run_state(run_dir)}


def copy_run(golden, dest: Path) -> tuple[Path, Path]:
    shutil.copytree(golden["run_dir"], dest / "run")
    conf = write_config(dest / "run.yaml", dest / "run", golden["repo"])
    return dest / "run", conf


# ---------------------------------------------------------------------------- golden runs

def test_golden_run_covers_every_day_and_trades(golden):
    st = golden["state"]
    assert len(st["equity"]) == GOLDEN_DAYS
    assert st["last_completed_day"] == HOLDOUT_END.isoformat()
    assert golden["calls"] == GOLDEN_DAYS * N_AGENTS  # every agent valid every day
    assert not st["skipped"]
    assert any(h["fills"] for h in st["history"].values())
    assert len(st["scores"]) > 0
    for name in ("checkpoint.json", "prediction_log.jsonl", "scores.sqlite", "failures.sqlite",
                 "llm_cache.sqlite", "weights.json", "weights_history.jsonl", "run_config.json"):
        assert (golden["run_dir"] / name).exists(), name


def test_golden_backtest_is_identical_across_two_runs(golden, tmp_path, keys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"])
    assert cli(["backtest", "--config", conf], GoldenStub()) == 0
    other = run_state(tmp_path / "run")
    ref = golden["state"]
    assert other["equity"] == ref["equity"]
    assert other["scores"] == ref["scores"]
    assert other["predlog"] == ref["predlog"]
    assert other == ref


def test_golden_backtest_resume_matches_uninterrupted_run(golden, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"])
    crash_day = 7
    with pytest.raises(SimulatedCrash):  # the process "dies" during day 8's agent calls
        cli(["backtest", "--config", conf], GoldenStub(crash_after_calls=crash_day * N_AGENTS))
    cp = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    assert len(cp["equity"]) == crash_day
    resumed = GoldenStub()
    assert cli(["backtest", "--config", conf], resumed) == 0  # same command resumes
    assert resumed.calls < GOLDEN_DAYS * N_AGENTS  # completed days were not re-run
    assert run_state(tmp_path / "run") == golden["state"]


# ---------------------------------------------------------------------------- CLI behaviour

def test_cli_backtest_dry_run_exits_zero_and_makes_no_llm_calls(golden, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"])
    stub = GoldenStub()
    assert cli(["backtest", "--config", conf, "--dry-run"], stub) == 0
    assert stub.calls == 0
    cp = json.loads((tmp_path / "run" / "dry_run" / "checkpoint.json").read_text())
    assert len(cp["equity"]) == GOLDEN_DAYS  # the full loop ran
    assert sorted(p.name for p in (tmp_path / "run").iterdir()) == ["dry_run"]  # real run untouched
    assert cp["fingerprint"]["dry_run"] is True
    assert cp["spent_usd"] == 0.0
    assert "dry run" in capsys.readouterr().out.lower()


def test_cli_requires_confirmation_above_cost_ceiling_unless_yes(golden, tmp_path, keys, monkeypatch, capsys):
    over = {"periods": SHORT_PERIODS, "settings": {"confirm_cost_usd": 0.5}}
    # non-interactive: refuse with a clear message, before any work
    conf = write_config(tmp_path / "a.yaml", tmp_path / "a", golden["repo"], **over)
    monkeypatch.setattr(sys, "stdin", NotTTY(""))
    stub = GoldenStub()
    assert cli(["backtest", "--config", conf], stub) == 2
    out = capsys.readouterr()
    assert "--yes" in out.out + out.err and "exceeds" in out.out + out.err
    assert stub.calls == 0 and not (tmp_path / "a" / "checkpoint.json").exists()
    # interactive "no": refused, nothing spent
    monkeypatch.setattr(sys, "stdin", FakeTTY("n\n"))
    assert cli(["backtest", "--config", conf], stub) == 2
    assert stub.calls == 0
    # interactive "yes": runs
    monkeypatch.setattr(sys, "stdin", FakeTTY("y\n"))
    assert cli(["backtest", "--config", conf], stub) == 0
    assert stub.calls > 0
    # --yes: runs without asking (stdin not a terminal)
    conf_b = write_config(tmp_path / "b.yaml", tmp_path / "b", golden["repo"], **over)
    monkeypatch.setattr(sys, "stdin", NotTTY(""))
    stub_b = GoldenStub()
    assert cli(["backtest", "--config", conf_b, "--yes"], stub_b) == 0
    assert stub_b.calls == 7 * N_AGENTS


def test_cli_refuses_anonymize_for_backtest_and_autoresearch(golden, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], anonymize=True)
    stub = GoldenStub()
    for cmd in (["backtest", "--config", conf], ["backtest", "--config", conf, "--dry-run"],
                ["autoresearch", "--config", conf]):
        assert cli(cmd, stub) == 2
        out = capsys.readouterr()
        assert "anonymi" in (out.out + out.err).lower()
    assert stub.calls == 0 and not (tmp_path / "run").exists()


@pytest.mark.parametrize("present", ["ANTHROPIC_API_KEY", "FRED_API_KEY", None])
def test_cli_missing_keys_exit_2_and_never_echo_a_key(golden, tmp_path, monkeypatch, capsys, present):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    if present == "ANTHROPIC_API_KEY":
        monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    if present == "FRED_API_KEY":
        monkeypatch.setenv("FRED_API_KEY", FAKE_FRED)
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"],
                        data={"macro": {"source": "fred", "series": ["DGS10"]}})
    stub = GoldenStub()
    for cmd in (["backtest", "--config", conf], ["autoresearch", "--config", conf]):
        assert cli(cmd, stub) == 2
        out = capsys.readouterr()
        text = out.out + out.err
        missing = {"ANTHROPIC_API_KEY", "FRED_API_KEY"} - {present}
        for name in missing:
            assert name in text
        assert FAKE_ANTHROPIC not in text and FAKE_FRED not in text
        assert "Traceback" not in text
    assert stub.calls == 0 and not (tmp_path / "run").exists()


def test_holdout_not_after_model_cutoff_is_a_config_error(golden, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"],
                        settings={"model_cutoff": "2024-03-01"})
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 2
    text = "".join(capsys.readouterr())
    assert "model_cutoff" in text and "holdout" in text and "Traceback" not in text


def test_bad_config_file_is_a_usage_error(tmp_path, capsys):
    p = tmp_path / "bad.yaml"
    p.write_text("run_dir: x\nperiods: {train: {start: 2024-01-01}}\n")
    assert cli(["backtest", "--config", p, "--dry-run"]) == 2
    assert cli(["backtest", "--config", tmp_path / "missing.yaml", "--dry-run"]) == 2
    assert "Traceback" not in "".join(capsys.readouterr())


def test_python_dash_m_atlas_help_works():
    p = subprocess.run([sys.executable, "-m", "atlas", "--help"], capture_output=True, text=True,
                       cwd=ROOT, timeout=60)
    assert p.returncode == 0
    for cmd in ("backtest", "autoresearch", "report", "calibrate"):
        assert cmd in p.stdout


def test_golden_fixtures_need_no_network(golden, tmp_path, monkeypatch):
    """A fresh dry run with every network entry point patched to fail still succeeds."""
    import socket

    def no_net(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", no_net)
    monkeypatch.setattr(socket.socket, "connect", no_net)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], periods=SHORT_PERIODS)
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 0


# ---------------------------------------------------------------------------- null test / calibration

def _train_picks(run_dir: Path):
    df = ScoreStore(run_dir / "scores.sqlite").picks_frame(None, 1)
    agents = [a for a in df["agent"].unique()
              if a.startswith(("macro_", "sector_", "superinvestor_"))]
    return df[df["agent"].isin(agents) & (df["date"] <= date(2024, 1, 18))]


def test_null_test_on_golden_data_gives_ic_near_zero(golden):
    picks = _train_picks(golden["run_dir"])
    null = null_test(picks, 200, 7, k=20.0, warmup_picks=5, window=20)  # no LeakageSuspected
    assert abs(float(np.mean(null.ics))) < 0.05
    assert null.dead_zone() > 0 and null.keep_margin() > 0


def test_calibrate_writes_calibration_and_second_pass_reads_it(golden, tmp_path, keys, capsys):
    run_dir, conf = copy_run(golden, tmp_path)
    assert cli(["calibrate", "--config", conf]) == 0
    cal = json.loads((run_dir / "calibration.json").read_text())
    assert cal["window"] == 20 and cal["n_shuffles"] == 200 and cal["seed"] == 7
    assert cal["train_max_date"] == "2024-01-18"  # last train day whose 1-day exit is in train
    assert abs(float(np.mean(cal["ics"]))) < 0.05
    direct = null_test(_train_picks(golden["run_dir"]), 200, 7, k=20.0, warmup_picks=5, window=20)
    assert cal["dead_zone"] == pytest.approx(direct.dead_zone())
    assert cal["keep_margin"] == pytest.approx(direct.keep_margin())
    assert "dead_zone" in capsys.readouterr().out
    # pass 2: a new run_dir pointed at pass 1's calibration runs with adaptive weights
    conf2 = write_config(tmp_path / "pass2.yaml", tmp_path / "pass2", golden["repo"],
                         periods=SHORT_PERIODS, calibration=str(run_dir / "calibration.json"))
    assert cli(["backtest", "--config", conf2], GoldenStub()) == 0
    cp = json.loads((tmp_path / "pass2" / "checkpoint.json").read_text())
    assert cp["fingerprint"]["settings.dead_zone"] == pytest.approx(cal["dead_zone"])
    # pass 1's frozen run cannot silently be resumed with the calibrated dead zone
    conf3 = write_config(tmp_path / "pass1b.yaml", run_dir, golden["repo"],
                         calibration=str(run_dir / "calibration.json"))
    assert cli(["backtest", "--config", conf3], GoldenStub()) == 1


def test_calibrate_with_too_few_picks_fails_cleanly(golden, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], periods=SHORT_PERIODS,
                        settings={"warmup_picks": 30})  # 2 safe train days cannot reach warm-up
    assert cli(["backtest", "--config", conf], GoldenStub()) == 0
    assert cli(["calibrate", "--config", conf]) == 1
    assert "cannot calibrate" in "".join(capsys.readouterr()).lower()
    assert not (tmp_path / "run" / "calibration.json").exists()


# ---------------------------------------------------------------------------- report

def test_report_writes_all_artefacts_and_verdict_on_golden_run(golden, tmp_path, capsys):
    run_dir, _ = copy_run(golden, tmp_path)
    assert cli(["report", "--run-dir", run_dir]) == 0
    rep = run_dir / "report"
    for name in ("report.md", "equity.png", "agents.csv", "autoresearch.csv"):
        assert (rep / name).exists() and (rep / name).stat().st_size > 0, name
    md = (rep / "report.md").read_text()
    assert "**No evidence of skill beyond the null**" in md
    assert "NOT EVIDENCE OF SKILL" in md  # train is pre-cutoff
    assert "- Dead zone:" in md and "not calibrated" not in md  # null computed from train picks
    assert "CIO total" in md  # ablation from the weights history and prediction log
    assert "Total API cost: $0.00" not in md  # stub spend is accounted
    assert "Control agent: not run." in md
    perf = md.split("## Performance")[1].split("## Agents")[0]
    assert "not provided" not in perf  # turnover computed from the recorded fills
    assert "No edits were attempted." in md


def test_report_says_not_provided_for_what_it_cannot_compute(golden, tmp_path):
    run_dir, _ = copy_run(golden, tmp_path)
    (run_dir / "weights_history.jsonl").unlink()  # no fills/weights history: no turnover, no ablation
    assert cli(["report", "--run-dir", run_dir]) == 0
    md = (run_dir / "report" / "report.md").read_text()
    assert "not provided" in md.split("## Performance")[1].split("## Agents")[0]
    assert "## CIO ablation\n\nnot run." in md
    assert "**No evidence of skill beyond the null**" in md


def test_report_includes_autoresearch_log_and_calibration(golden, tmp_path):
    run_dir, conf = copy_run(golden, tmp_path)
    assert cli(["calibrate", "--config", conf]) == 0
    rec = {"agent": "sector_energy", "branch": "autoresearch/sector_energy-0", "n_days": 41,
           "old_ic": 0.01, "new_ic": 0.02, "diff": 0.01, "keep_margin": 0.2, "decision": "reverted",
           "reason": "diff 0.0100 <= keep_margin 0.2000", "commit_sha": None, "cost_usd": 0.5}
    (run_dir / "autoresearch_log.jsonl").write_text(json.dumps(rec) + "\n")
    assert cli(["report", "--run-dir", run_dir]) == 0
    md = (run_dir / "report" / "report.md").read_text()
    assert "sector_energy: **reverted**" in md
    cal = json.loads((run_dir / "calibration.json").read_text())
    assert f"- Dead zone: {cal['dead_zone']:.3f}" in md
    assert "autoresearch/sector_energy-0" in (run_dir / "report" / "autoresearch.csv").read_text()


def test_report_on_missing_run_dir_is_a_usage_error(tmp_path, capsys):
    assert cli(["report", "--run-dir", tmp_path / "nope"]) == 2
    assert "Traceback" not in "".join(capsys.readouterr())


def test_with_control_records_control_picks_in_the_score_store(golden, tmp_path, keys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], periods=SHORT_PERIODS)
    stub = GoldenStub()
    assert cli(["backtest", "--config", conf, "--with-control"], stub) == 0
    assert stub.calls == 7 * (N_AGENTS + 1)
    df = ScoreStore(tmp_path / "run" / "scores.sqlite").picks_frame("control", 1)
    assert len(df) > 0 and df["scored"].any()
    # trading is unaffected by the control agent
    ref = write_config(tmp_path / "ref.yaml", tmp_path / "ref", golden["repo"], periods=SHORT_PERIODS)
    assert cli(["backtest", "--config", ref], GoldenStub()) == 0
    a, b = run_state(tmp_path / "run"), run_state(tmp_path / "ref")
    for k in ("equity", "portfolio", "predlog", "history"):
        assert a[k] == b[k], k
    assert cli(["report", "--run-dir", tmp_path / "run"]) == 0
    md = (tmp_path / "run" / "report" / "report.md").read_text()
    assert "Source: score store" in md


# ---------------------------------------------------------------------------- autoresearch

def test_autoresearch_refuses_a_dirty_prompt_repo(golden, tmp_path, keys, capsys):
    repo = make_prompt_repo(tmp_path / "repo")
    (repo / "atlas" / "prompts" / "cio.md").write_text("# cio\n\nedited by hand\n")
    run_dir, _ = copy_run(golden, tmp_path)
    conf = write_config(tmp_path / "ar.yaml", run_dir, repo)
    stub = GoldenStub()
    assert cli(["autoresearch", "--config", conf], stub) == 1
    text = "".join(capsys.readouterr())
    assert "uncommitted changes" in text and "Traceback" not in text
    assert stub.calls == 0
    assert (repo / "atlas" / "prompts" / "cio.md").read_text() == "# cio\n\nedited by hand\n"


def test_autoresearch_without_a_stored_run_fails_cleanly(golden, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"])
    assert cli(["autoresearch", "--config", conf], GoldenStub()) == 1
    assert "atlas backtest" in "".join(capsys.readouterr())


def test_autoresearch_on_golden_run_needs_more_train_days(golden, tmp_path, keys, capsys):
    """The golden train period (15 days) is too short for a paired replay (>= 40 days): the
    command builds its inputs, says so and exits 1 without calling the model or touching git."""
    run_dir, conf = copy_run(golden, tmp_path)
    head = git(golden["repo"], "rev-parse", "HEAD")
    stub = GoldenStub()
    assert cli(["autoresearch", "--config", conf], stub) == 1
    out = "".join(capsys.readouterr())
    assert "keep_margin" in out and "train days" in out
    assert stub.calls == 0
    assert git(golden["repo"], "rev-parse", "HEAD") == head
    assert git(golden["repo"], "status", "--porcelain") == ""


def test_autoresearch_replay_reconstructs_the_runs_inputs_exactly(golden, tmp_path, keys):
    """The old-prompt replay of a recorded day is a pure cache hit of the backtest and gives
    back the picks in the prediction log: the day context is rebuilt faithfully."""
    from atlas.cli import build_replay
    run_dir, conf = copy_run(golden, tmp_path)
    stub = GoldenStub()
    replay = build_replay(conf, llm_client=StubClient(stub))
    lines = [json.loads(x) for x in (run_dir / "prediction_log.jsonl").read_text().splitlines()]
    for line in (lines[3], lines[10]):
        day = date.fromisoformat(line["day"])
        for rec in line["agents"]:
            name = rec["agent"]
            res = replay.run_agent(name, day, replay.prompts[name])
            assert res.output.model_dump(mode="json") == rec["output"], (name, day)
    assert stub.calls == 0
    run = replay.run_one_day("sector_energy")
    conv, exc = run(date(2024, 1, 10), replay.prompts["sector_energy"])
    assert len(conv) == len(exc) > 0
    assert run.horizon == 1
    assert stub.calls == 0


def test_transient_failures_pause_with_exit_1_and_the_same_command_resumes(golden, tmp_path, keys,
                                                                           monkeypatch, capsys):
    import atlas.engine.agent as agent_mod
    from atlas.engine.llm import ApiError
    monkeypatch.setattr(agent_mod, "BACKOFF_BASE_S", 0.0)  # no real backoff sleeps
    over = {"periods": SHORT_PERIODS}
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], **over)
    good = GoldenStub()

    def flaky(system, user):
        if json.loads(user)["asof"] == "2024-01-04":
            raise ApiError("APIStatusError status=529")
        return good(system, user)

    assert cli(["backtest", "--config", conf], flaky) == 1
    text = "".join(capsys.readouterr())
    assert "paused" in text and "re-run the same command to resume" in text and "Traceback" not in text
    cp = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    assert cp["last_completed_day"] == "2024-01-03"  # the checkpoint is intact
    assert cli(["backtest", "--config", conf], GoldenStub()) == 0
    ref = write_config(tmp_path / "ref.yaml", tmp_path / "ref", golden["repo"], **over)
    assert cli(["backtest", "--config", ref], GoldenStub()) == 0
    a, b = run_state(tmp_path / "run"), run_state(tmp_path / "ref")
    assert a == b


def test_example_run_config_is_valid():
    from atlas.cli import load_run_config, resolve_periods, resolve_settings
    rc = load_run_config(ROOT / "atlas" / "config" / "run.example.yaml")
    assert rc.anonymize is False
    resolve_periods(rc)
    s = resolve_settings(rc)
    assert s.dead_zone is None  # pass 1: frozen weights


# ---------------------------------------------------------------------------- fix round 1

ALT_PERIODS = {"train": {"start": "2024-01-01", "end": "2024-02-02"},
               "validation": {"start": "2024-02-05", "end": "2024-02-09"},
               "holdout": {"start": "2024-02-12", "end": "2024-02-23"}}


def test_readme_order_dry_run_then_real_run_both_succeed(golden, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    conf = write_config(tmp_path / "my_run.yaml", tmp_path / "run", golden["repo"], periods=SHORT_PERIODS)
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 0
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    assert cli(["backtest", "--config", conf], GoldenStub()) == 0
    real = json.loads((tmp_path / "run" / "checkpoint.json").read_text())
    dry = json.loads((tmp_path / "run" / "dry_run" / "checkpoint.json").read_text())
    assert real["fingerprint"]["dry_run"] is False and dry["fingerprint"]["dry_run"] is True
    with sqlite3.connect(tmp_path / "run" / "scores.sqlite") as c:  # only the real run's picks
        assert c.execute("SELECT COUNT(*) FROM picks").fetchone()[0] > 0
    assert len((tmp_path / "run" / "prediction_log.jsonl").read_text().splitlines()) == 7


def test_refused_rerun_with_changed_periods_leaves_stored_config_untouched(golden, tmp_path, keys, capsys):
    run_dir, _ = copy_run(golden, tmp_path)
    before = (run_dir / "run_config.json").read_bytes()
    conf = write_config(tmp_path / "changed.yaml", run_dir, golden["repo"], periods=ALT_PERIODS,
                        settings={"model_cutoff": "2024-02-09"})
    stub = GoldenStub()
    assert cli(["backtest", "--config", conf, "--yes"], stub) == 1
    text = "".join(capsys.readouterr())
    assert "periods.holdout.start" in text and "settings.model_cutoff" in text
    assert stub.calls == 0
    assert (run_dir / "run_config.json").read_bytes() == before
    assert cli(["report", "--run-dir", run_dir]) == 0
    md = (run_dir / "report" / "report.md").read_text()
    assert "### holdout (2024-02-05 to 2024-02-23)" in md  # the periods the run traded with


def test_calibrate_refuses_a_config_that_disagrees_with_the_stored_run(golden, tmp_path, capsys):
    run_dir, conf = copy_run(golden, tmp_path)
    for key, over in (("periods.train.end", {"periods": ALT_PERIODS, "settings": {"model_cutoff": "2024-02-09"}}),
                      ("settings.rolling_ic_window", {"settings": {"rolling_ic_window": 7}}),
                      ("settings.warmup_picks", {"settings": {"warmup_picks": 1}})):
        bad = write_config(tmp_path / "bad.yaml", run_dir, golden["repo"], **over)
        assert cli(["calibrate", "--config", bad]) == 2, key
        text = "".join(capsys.readouterr())
        assert "disagrees with the stored run" in text and key in text and "Traceback" not in text
        assert not (run_dir / "calibration.json").exists()
    assert cli(["calibrate", "--config", conf]) == 0  # the matching config is accepted


def test_autoresearch_refuses_a_config_that_disagrees_with_the_stored_run(golden, tmp_path, keys, capsys):
    run_dir, _ = copy_run(golden, tmp_path)
    bad = write_config(tmp_path / "bad.yaml", run_dir, golden["repo"], periods=ALT_PERIODS,
                       settings={"model_cutoff": "2024-02-09"})
    stub = GoldenStub()
    assert cli(["autoresearch", "--config", bad], stub) == 2
    text = "".join(capsys.readouterr())
    assert "periods.train.end" in text and stub.calls == 0
    assert not (run_dir / "autoresearch_log.jsonl").exists()


def test_calibrate_without_stored_config_fails_clearly(golden, tmp_path, capsys):
    run_dir, conf = copy_run(golden, tmp_path)
    (run_dir / "run_config.json").unlink()
    assert cli(["calibrate", "--config", conf]) == 1
    assert "run_config.json" in "".join(capsys.readouterr())


@pytest.mark.parametrize("over,needle", [
    ({"settings": {"budget_usdd": 1}}, "settings.budget_usdd (did you mean 'budget_usd'?)"),
    ({"settings": {"confirm_cost": 0}}, "settings.confirm_cost (did you mean 'confirm_cost_usd'?)"),
    ({"periods": {"train": {"start": "2024-01-01", "endd": "2024-01-19"}}}, "periods.train.endd (did you mean 'end'?)"),
    ({"budgett_usd": 1}, "budgett_usd (did you mean 'budget_usd'?)"),
    ({"data": {"macro": {"sereis": []}}}, "data.macro.sereis (did you mean 'series'?)"),
])
def test_typos_in_the_run_config_are_rejected_with_a_hint(golden, tmp_path, capsys, over, needle):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], **over)
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 2
    assert needle in "".join(capsys.readouterr())
    assert not (tmp_path / "run").exists()


def test_stored_run_config_holds_validated_settings_not_raw_overrides(golden):
    rec = json.loads((golden["run_dir"] / "run_config.json").read_text())
    assert "settings" not in rec["run_config"]
    assert rec["settings"]["warmup_picks"] == 5 and rec["settings"]["budget_usd"] == 50


def test_keys_never_reach_output_or_run_dir_on_success(golden, tmp_path, monkeypatch, capsys):
    from atlas.data.fred_provider import FredAlfredProvider
    import pandas as pd
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    monkeypatch.setenv("FRED_API_KEY", FAKE_FRED)
    monkeypatch.setattr(FredAlfredProvider, "fetch", lambda self, sid, asof: pd.Series(dtype=float, name=sid))
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], periods=SHORT_PERIODS,
                        data={"macro": {"source": "fred", "series": ["DGS10"]}})
    assert cli(["backtest", "--config", conf, "--with-control"], GoldenStub()) == 0
    assert cli(["calibrate", "--config", conf]) in (0, 1)
    assert cli(["report", "--run-dir", tmp_path / "run"]) == 0
    text = "".join(capsys.readouterr())
    for secret in (FAKE_ANTHROPIC, FAKE_FRED):
        assert secret not in text
        for f in (tmp_path / "run").rglob("*"):
            if f.is_file():
                assert secret.encode() not in f.read_bytes(), f


def test_control_gets_only_the_remaining_budget(golden, tmp_path, keys, monkeypatch):
    import atlas.cli as cli_mod
    made = []

    class Recording(cli_mod.CachedLLM):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            made.append(self)

    monkeypatch.setattr(cli_mod, "CachedLLM", Recording)
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], periods=SHORT_PERIODS,
                        budget_usd=10)
    assert cli(["backtest", "--config", conf, "--with-control"], GoldenStub()) == 0
    bt_spent = json.loads((tmp_path / "run" / "checkpoint.json").read_text())["spent_usd"]
    assert bt_spent > 0
    control_llm = made[-1]
    assert control_llm.budget_usd == pytest.approx(10 - bt_spent)
    ctl = json.loads((tmp_path / "run" / "control.json").read_text())
    assert ctl["days"] == 7 and ctl["spent_usd"] == pytest.approx(control_llm.spent_usd)


def test_autoresearch_gets_what_remains_of_the_hard_cap(golden, tmp_path, keys, monkeypatch, capsys):
    import atlas.autoresearch.loop as loop_mod
    seen = {}

    def fake_loop(cfg, *, llm, **kw):
        seen["budget"] = llm.budget_usd
        return []

    monkeypatch.setattr(loop_mod, "run_autoresearch", fake_loop)
    run_dir, conf = copy_run(golden, tmp_path)
    rec = {"agent": "sector_energy", "branch": None, "n_days": 0, "old_ic": None, "new_ic": None,
           "diff": None, "keep_margin": 0.2, "decision": "rejected", "reason": "x", "commit_sha": None,
           "cost_usd": 0.75}
    (run_dir / "autoresearch_log.jsonl").write_text(json.dumps(rec) + "\n")
    (run_dir / "control.json").write_text(json.dumps({"spent_usd": 0.25, "days": 40}))
    bt = json.loads((run_dir / "checkpoint.json").read_text())["spent_usd"]
    assert cli(["autoresearch", "--config", conf], GoldenStub()) == 0
    assert seen["budget"] == pytest.approx(50 - bt - 0.25 - 0.75)
    assert cli(["report", "--run-dir", run_dir]) == 0
    md = (run_dir / "report" / "report.md").read_text()
    assert f"Total API cost: ${bt + 1.0:.2f}" in md
    # an exhausted cap refuses before any call
    (run_dir / "control.json").write_text(json.dumps({"spent_usd": 60.0, "days": 40}))
    assert cli(["autoresearch", "--config", conf], GoldenStub()) == 1
    assert "budget exhausted" in "".join(capsys.readouterr())


def test_corrupt_checkpoint_is_a_clear_error_for_every_command(golden, tmp_path, keys, capsys):
    run_dir, conf = copy_run(golden, tmp_path)
    (run_dir / "checkpoint.json").write_text("{not json")
    for cmd in (["report", "--run-dir", run_dir], ["calibrate", "--config", conf],
                ["backtest", "--config", conf], ["autoresearch", "--config", conf]):
        assert cli(cmd, GoldenStub()) == 1, cmd
        text = "".join(capsys.readouterr())
        assert f"checkpoint is corrupt: {run_dir / 'checkpoint.json'}" in text and "Traceback" not in text


def test_report_ablation_is_per_period_with_holdout_headline(golden, tmp_path):
    run_dir, _ = copy_run(golden, tmp_path)
    assert cli(["report", "--run-dir", run_dir]) == 0
    sec = (run_dir / "report" / "report.md").read_text().split("## CIO ablation")[1].split("## Failures")[0]
    assert "Headline: holdout" in sec
    assert "- train [NOT EVIDENCE OF SKILL" in sec and "- all periods [NOT EVIDENCE OF SKILL" in sec
    assert "SAME gross exposure" in (run_dir / "report" / "report.md").read_text()


# ---------------------------------------------------------------------------- ablation unit

def _abl_inputs(tmp_path):
    from atlas.backtest.runner import Periods
    from atlas.config import AgentSpec, Limits
    from atlas.data.csv_provider import CsvPriceProvider, NoMacro
    from atlas.data.store import PriceStore
    from atlas.schemas import AgentOutput, AgentResult, Pick
    store = PriceStore(CsvPriceProvider(FIX / "prices"), NoMacro(), tmp_path / "prices")
    specs = [AgentSpec(name="m1", layer=1, kind="macro", prompt_file="x"),
             AgentSpec(name="m2", layer=1, kind="macro", prompt_file="x"),
             AgentSpec(name="execution", layer=4, kind="execution", prompt_file="x")]
    limits = Limits(max_position_pct=0.5, max_gross_pct=1.0, max_positions=8, min_order_usd=1,
                    max_commission_frac=0.01, daily_loss_pct=0.5, drawdown_kill_pct=0.9, long_only=True)
    days = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
    nxt = {days[0]: days[1], days[1]: days[2], days[2]: date(2024, 1, 5)}

    def res(d, name, ticker, conv):
        return AgentResult(agent=name, date=d, output=AgentOutput(
            agent=name, date=d, view="neutral", confidence=0.5,
            picks=[Pick(ticker=ticker, direction="long", conviction=conv, horizon_days=1)]))

    results = {d: {"m1": res(d, "m1", "AAA", 0.5), "m2": res(d, "m2", "BBB", 0.5),
                   "execution": res(d, "execution", "CCC", 0.4)} for d in days}
    w = {"m1": 2.5, "m2": 0.3}
    history = {days[0]: {"weights": w, "skipped": None}, days[1]: {"weights": w, "skipped": None},
               days[2]: {"weights": w, "skipped": "cio failed (malformed_json); no decision to trade on"}}
    periods = Periods(train=(days[0], days[2]), validation=(date(2024, 1, 8), date(2024, 1, 9)),
                      holdout=(date(2024, 2, 5), date(2024, 2, 6)))

    def r(t, d):
        s = store.outcome_prices(t, d, nxt[d])
        return float(s.iloc[-1] / s.iloc[0] - 1)

    return store, specs, limits, history, results, periods, days, r


def test_ablation_uses_the_days_weights_gross_match_and_skips_skipped_days(tmp_path):
    from atlas.cli import ablation_from_run
    store, specs, limits, history, results, periods, days, r = _abl_inputs(tmp_path)
    out = ablation_from_run(history, results, store, specs, limits, periods)
    wa, wb = 2.5 * 0.5 / (2.5 * 0.5 + 0.3 * 0.5), 0.3 * 0.5 / (2.5 * 0.5 + 0.3 * 0.5)
    base = [0.4 * (wa * r("AAA", d) + wb * r("BBB", d)) for d in days[:2]]  # gross-matched to 0.4
    cio = [0.4 * r("CCC", d) for d in days[:2]]                              # day 3 skipped
    tot = lambda xs: (1 + xs[0]) * (1 + xs[1]) - 1  # noqa: E731
    assert out["train"]["baseline_total"] == pytest.approx(tot(base), rel=1e-12)
    assert out["train"]["cio_total"] == pytest.approx(tot(cio), rel=1e-12)
    assert out["all"] == out["train"]
    assert out["validation"] is None and out["holdout"] is None
