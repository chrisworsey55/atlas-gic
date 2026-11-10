"""Hub-and-spoke runs through the `atlas` CLI on the golden fixture: one client and one
CachedLLM per model, ONE shared budget, per-model spend in the checkpoint and the report.

The hub (cro, cio, execution) runs `golden-stub-model` at $3/$15 per million tokens; the
spokes (macro, sector, superinvestor) and the control agent run `golden-stub-spoke` at $1/$5.
Each model gets its own recording stub client, so a test can see which model served which
agent and recompute every dollar from the recorded token counts.
"""
import hashlib
import json
import sys
import threading
from pathlib import Path

import pytest
import yaml

from atlas.cli import anthropic_client, build_run, load_run_config, main
from atlas.config import Settings, load_settings
from atlas.engine.llm import Budget, StubClient
from tests.golden.stub import GoldenStub, SimulatedCrash, agent_of
from tests.golden.test_golden_backtest import (
    GOLDEN_DAYS, NotTTY, SHORT_PERIODS, make_prompt_repo, run_state, write_config,
)

HUB, SPOKE = "golden-stub-model", "golden-stub-spoke"
HUB_P, SPOKE_P = (3.0, 15.0), (1.0, 5.0)
HUB_AGENTS = {"cro", "cio", "execution"}
SPOKE_AGENTS = {"macro_rates", "macro_volatility", "macro_dollar_commodities", "sector_semiconductors",
                "sector_financials", "sector_energy", "sector_healthcare", "superinvestor_druckenmiller",
                "superinvestor_ackman"}
MIXED = {"settings": {"spoke_model": SPOKE, "spoke_usd_per_mtok_input": SPOKE_P[0],
                      "spoke_usd_per_mtok_output": SPOKE_P[1]}}
FAKE_ANTHROPIC = "sk-ant-TEST-do-not-print-0123456789"
UNATTRIBUTED = Budget.UNATTRIBUTED
# sha256 of the golden run's prediction_log.jsonl made on the commit BEFORE hub-and-spoke
# construction (single model, spoke_model unset): the single-model run must not change at all
PRE_CHANGE_PREDLOG_SHA256 = "88a86363f3de40f8d4f15a30fad96ec7ebedf8a6f143d8def92ccbf522d98039"


class Recorder:
    """A stub client for one model: records (agent, input_tokens, output_tokens) per call."""

    def __init__(self, stub: GoldenStub | None = None):
        self.inner = StubClient(stub or GoldenStub())
        self.calls: list[tuple[str, int, int]] = []
        self._lock = threading.Lock()

    def complete(self, system, user, max_tokens):
        r = self.inner.complete(system, user, max_tokens)
        with self._lock:
            self.calls.append((agent_of(system), r.input_tokens, r.output_tokens))
        return r

    def agents(self) -> set[str]:
        return {a for a, _, _ in self.calls}

    def cost(self, prices, agents=None) -> float:
        return sum((i * prices[0] + o * prices[1]) / 1e6 for a, i, o in self.calls
                   if agents is None or a in agents)


def clients(hub_stub=None, spoke_stub=None) -> dict[str, Recorder]:
    return {HUB: Recorder(hub_stub), SPOKE: Recorder(spoke_stub)}


def pkg_settings(**over) -> Settings:
    base = load_settings(Path(__file__).resolve().parents[2] / "atlas" / "config" / "settings.yaml")
    return base.model_copy(update=over)


def run_cli(argv, llm_client=None) -> int:
    return main([str(a) for a in argv], llm_client=llm_client)


def ckpt(run_dir: Path) -> dict:
    return json.loads((run_dir / "checkpoint.json").read_text())


@pytest.fixture(scope="module")
def repo(tmp_path_factory):
    return make_prompt_repo(tmp_path_factory.mktemp("hubspoke") / "repo")


@pytest.fixture(scope="module")
def mixed(repo, tmp_path_factory):
    """One full mixed run (40 days) with the control agent, then calibrate + report."""
    root = tmp_path_factory.mktemp("mixed")
    conf = write_config(root / "run.yaml", root / "run", repo, **MIXED)
    cl = clients()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
        assert run_cli(["backtest", "--config", conf, "--with-control"], cl) == 0
        assert run_cli(["report", "--run-dir", root / "run"]) == 0
    return {"root": root, "conf": conf, "run_dir": root / "run", "clients": cl}


@pytest.fixture(scope="module")
def single(repo, tmp_path_factory):
    """The golden single-model run (spoke_model unset) with the control agent, + report."""
    root = tmp_path_factory.mktemp("single")
    conf = write_config(root / "run.yaml", root / "run", repo)
    stub = GoldenStub()
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
        assert main(["backtest", "--config", str(conf), "--with-control"], llm_client=StubClient(stub)) == 0
        assert run_cli(["report", "--run-dir", root / "run"]) == 0
    return {"root": root, "conf": conf, "run_dir": root / "run", "calls": stub.calls}


@pytest.fixture
def keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_ANTHROPIC)
    monkeypatch.delenv("FRED_API_KEY", raising=False)


# ---------------------------------------------------------------------------- construction

def test_build_run_one_cached_llm_per_model_sharing_one_budget_and_cache(repo, tmp_path, keys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo, **MIXED)
    cl = clients()
    run = build_run(load_run_config(conf), llm_client=cl)
    assert set(run.llms) == {HUB, SPOKE}
    assert run.llm is run.llms[HUB]  # proposals / restore / cap narrowing go through the hub
    assert all(llm.budget is run.llm.budget for llm in run.llms.values())  # ONE budget
    assert run.llm.budget.cap_usd == run.settings.budget_usd
    assert {llm.cache_path for llm in run.llms.values()} == {run.rc.llm_cache_path}  # one cache file
    assert run.llms[HUB].model == HUB and run.llms[HUB].usd_per_mtok == HUB_P
    assert run.llms[SPOKE].model == SPOKE and run.llms[SPOKE].usd_per_mtok == SPOKE_P
    assert run.llms[HUB].client is cl[HUB] and run.llms[SPOKE].client is cl[SPOKE]
    # the pipeline routes EVERY agent, the control agent included
    assert run.pipeline.llms is run.llms
    assert run.pipeline.models == {sp.name: run.settings.agent_model(sp) for sp in run.specs}
    assert run.pipeline.models["control"] == SPOKE and run.pipeline.models["cio"] == HUB
    # narrowing the cap on any one instance narrows the one shared cap
    run.llms[SPOKE].budget_usd = 1.25
    assert run.llms[HUB].budget_usd == 1.25


def test_hub_and_spoke_calls_with_identical_prompts_do_not_collide_in_the_real_cache(repo, tmp_path, keys):
    import sqlite3
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo, **MIXED)
    cl = clients()
    run = build_run(load_run_config(conf), llm_client=cl)
    sysmsg, user = "# cio\nsame system", '{"asof": "2024-01-02"}'
    for _ in range(2):  # the second round is served from the cache, per model
        run.llms[HUB].call(sysmsg, user, 100)
        run.llms[SPOKE].call(sysmsg, user, 100)
    assert len(cl[HUB].calls) == 1 and len(cl[SPOKE].calls) == 1
    with sqlite3.connect(run.rc.llm_cache_path) as c:
        assert c.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0] == 2
    by = run.llm.budget.spent_by_model()
    assert set(by) == {HUB, SPOKE} and by[HUB] > by[SPOKE] > 0


def test_anthropic_client_is_built_per_model_with_the_run_effort():
    s = pkg_settings(model="claude-sonnet-5-5", spoke_model="claude-haiku-5-5", effort="low")
    hub, spoke = anthropic_client(s), anthropic_client(s, "claude-haiku-5-5")
    assert (hub.model, hub.effort) == ("claude-sonnet-5-5", "low")
    assert (spoke.model, spoke.effort) == ("claude-haiku-5-5", "low")


def test_every_model_in_use_must_be_priced_before_any_spend(repo, tmp_path, keys, capsys):
    agents = yaml.safe_load((Path(__file__).resolve().parents[2] / "atlas" / "config" / "agents.yaml").read_text())
    for a in agents["agents"]:
        if a["name"] == "control":
            a["model"] = "unpriced-control-model"  # only the control agent uses it
    (tmp_path / "agents.yaml").write_text(yaml.safe_dump(agents))
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo, agents_file=str(tmp_path / "agents.yaml"),
                        **MIXED)
    cl = clients()
    assert run_cli(["backtest", "--config", conf], cl) == 2
    err = capsys.readouterr().err
    assert "no price is known for the per-agent model 'unpriced-control-model'" in err
    assert cl[HUB].calls == [] and cl[SPOKE].calls == []
    assert not (tmp_path / "run" / "checkpoint.json").exists()
    # an unpriced spoke model is refused the same way, naming its own price fields
    conf2 = write_config(tmp_path / "run2.yaml", tmp_path / "run2", repo,
                         settings={"spoke_model": "unpriced-spoke-model"})
    assert run_cli(["backtest", "--config", conf2], clients()) == 2
    assert "spoke_usd_per_mtok_input" in capsys.readouterr().err


# ---------------------------------------------------------------------------- dry run

def test_mixed_run_dry_run_makes_zero_calls_and_prints_models(repo, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    def no_client(*a, **k):
        raise AssertionError("a dry run must not build a real client")
    monkeypatch.setattr("atlas.cli.anthropic_client", no_client)
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo, **MIXED)
    assert run_cli(["backtest", "--config", conf, "--dry-run", "--with-control"]) == 0
    out = capsys.readouterr().out
    assert "models in use:" in out
    hub_line = next(ln for ln in out.splitlines() if ln.strip().startswith(HUB))
    spoke_line = next(ln for ln in out.splitlines() if ln.strip().startswith(SPOKE))
    assert "hub" in hub_line and all(a in hub_line for a in HUB_AGENTS)
    assert "spoke" in spoke_line and all(a in spoke_line for a in SPOKE_AGENTS | {"control"})
    cp = ckpt(tmp_path / "run" / "dry_run")
    assert len(cp["equity"]) == GOLDEN_DAYS and cp["spent_usd"] == 0.0
    # stub clients passed in are never called either
    cl = clients()
    assert run_cli(["backtest", "--config", conf, "--dry-run", "--with-control"], cl) == 0
    assert cl[HUB].calls == [] and cl[SPOKE].calls == []


# ---------------------------------------------------------------------------- routing and money

def test_mixed_run_calls_reach_the_right_stub_clients_and_spend_is_split(mixed):
    hub, spoke = mixed["clients"][HUB], mixed["clients"][SPOKE]
    assert hub.agents() == HUB_AGENTS
    assert spoke.agents() == SPOKE_AGENTS | {"control"}
    assert len(hub.calls) == GOLDEN_DAYS * 3
    assert len(spoke.calls) == GOLDEN_DAYS * (9 + 1)  # 9 spokes + the control agent
    cp = ckpt(mixed["run_dir"])
    by = cp["spent_by_model"]
    assert set(by) == {HUB, SPOKE}
    assert by[HUB] == pytest.approx(hub.cost(HUB_P))
    assert by[SPOKE] == pytest.approx(spoke.cost(SPOKE_P, SPOKE_AGENTS))  # backtest only: no control
    assert cp["spent_usd"] == pytest.approx(by[HUB] + by[SPOKE])
    control = json.loads((mixed["run_dir"] / "control.json").read_text())
    assert control["spent_usd"] == pytest.approx(spoke.cost(SPOKE_P, {"control"}))


def test_control_agent_uses_spoke_model(mixed):
    assert "control" not in mixed["clients"][HUB].agents()
    ctl = [c for c in mixed["clients"][SPOKE].calls if c[0] == "control"]
    assert len(ctl) == GOLDEN_DAYS
    control = json.loads((mixed["run_dir"] / "control.json").read_text())
    # priced at the SPOKE model's prices (hub prices would be 3x)
    assert control["spent_usd"] == pytest.approx(sum((i * 1.0 + o * 5.0) / 1e6 for _, i, o in ctl))


def test_estimate_uses_each_agents_model_including_control(repo, tmp_path, keys, monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", NotTTY(""))
    over = {"settings": {**MIXED["settings"], "confirm_cost_usd": 0.0}}
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo, **over)
    s = pkg_settings()
    hub_call = (s.est_input_tokens_per_call * 3.0 + s.est_output_tokens_per_call * 15.0) / 1e6
    spoke_call = (s.est_input_tokens_per_call * 1.0 + s.est_output_tokens_per_call * 5.0) / 1e6
    for flags, n_spoke in (([], 9), (["--with-control"], 10)):
        cl = clients()
        assert run_cli(["backtest", "--config", conf, *flags], cl) == 2
        err = capsys.readouterr().err
        want = GOLDEN_DAYS * (3 * hub_call + n_spoke * spoke_call)
        assert f"estimated LLM cost ${want:.2f} exceeds" in err, err
        assert cl[HUB].calls == [] and cl[SPOKE].calls == []


def test_hard_cap_spans_both_models(repo, tmp_path, keys, capsys):
    cap = 0.5
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo, budget_usd=cap, **MIXED)
    cl = clients()
    assert run_cli(["backtest", "--config", conf], cl) == 1  # paused: budget
    assert "budget" in capsys.readouterr().err.lower()
    cp = ckpt(tmp_path / "run")
    spent_all = cl[HUB].cost(HUB_P) + cl[SPOKE].cost(SPOKE_P)  # every call made, committed or not
    assert cl[HUB].calls and cl[SPOKE].calls
    assert spent_all <= cap
    assert cp["spent_usd"] <= cap
    assert cp["spent_usd"] == pytest.approx(spent_all)  # the pause persists the spend of the paused day
    assert sum(cp["spent_by_model"].values()) == pytest.approx(cp["spent_usd"])
    assert len(cp["equity"]) < GOLDEN_DAYS


def test_resume_restores_spend_once_for_all_models(repo, tmp_path, keys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo, **MIXED)
    crash_day = 7
    with pytest.raises(SimulatedCrash):  # dies during day 8's spoke calls
        run_cli(["backtest", "--config", conf], clients(spoke_stub=GoldenStub(crash_after_calls=crash_day * 9)))
    before = ckpt(tmp_path / "run")
    assert len(before["equity"]) == crash_day
    assert set(before["spent_by_model"]) == {HUB, SPOKE}
    cl = clients()
    assert run_cli(["backtest", "--config", conf], cl) == 0
    after = ckpt(tmp_path / "run")
    assert len(after["equity"]) == GOLDEN_DAYS
    new_hub, new_spoke = cl[HUB].cost(HUB_P), cl[SPOKE].cost(SPOKE_P)
    # the checkpoint's total is restored ONCE into the shared budget, not once per model
    assert after["spent_usd"] == pytest.approx(before["spent_usd"] + new_hub + new_spoke)
    assert after["spent_by_model"][HUB] == pytest.approx(before["spent_by_model"][HUB] + new_hub)
    assert after["spent_by_model"][SPOKE] == pytest.approx(before["spent_by_model"][SPOKE] + new_spoke)
    assert UNATTRIBUTED not in after["spent_by_model"]


def test_resume_of_a_checkpoint_without_spent_by_model_keeps_it_unattributed(repo, tmp_path, keys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo, **MIXED)
    with pytest.raises(SimulatedCrash):
        run_cli(["backtest", "--config", conf], clients(spoke_stub=GoldenStub(crash_after_calls=5 * 9)))
    path = tmp_path / "run" / "checkpoint.json"
    old = json.loads(path.read_text())
    del old["spent_by_model"]  # a checkpoint written before per-model tracking
    path.write_text(json.dumps(old, sort_keys=True, indent=2))
    cl = clients()
    assert run_cli(["backtest", "--config", conf], cl) == 0
    after = ckpt(tmp_path / "run")
    by = after["spent_by_model"]
    assert by[UNATTRIBUTED] == pytest.approx(old["spent_usd"])
    assert by[HUB] == pytest.approx(cl[HUB].cost(HUB_P)) and by[SPOKE] == pytest.approx(cl[SPOKE].cost(SPOKE_P))
    assert after["spent_usd"] == pytest.approx(old["spent_usd"] + cl[HUB].cost(HUB_P) + cl[SPOKE].cost(SPOKE_P))


def test_changing_spoke_model_on_an_existing_run_is_refused_by_the_stored_record(repo, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo, periods=SHORT_PERIODS, **MIXED)
    assert run_cli(["backtest", "--config", conf], clients()) == 0
    stored = (tmp_path / "run" / "run_config.json").read_bytes()
    changed = {"settings": {**MIXED["settings"], "spoke_model": "golden-stub-spoke-2"}}
    conf2 = write_config(tmp_path / "run2.yaml", tmp_path / "run", repo, periods=SHORT_PERIODS, **changed)
    cl = {HUB: Recorder(), "golden-stub-spoke-2": Recorder()}
    capsys.readouterr()
    assert run_cli(["backtest", "--config", conf2], cl) == 1
    assert "settings.spoke_model" in capsys.readouterr().err
    assert (tmp_path / "run" / "run_config.json").read_bytes() == stored
    assert all(not c.calls for c in cl.values())
    # unsetting it (back to a single model) is refused too; calibrate refuses with exit 2
    conf3 = write_config(tmp_path / "run3.yaml", tmp_path / "run", repo, periods=SHORT_PERIODS)
    assert run_cli(["backtest", "--config", conf3], clients()) == 1
    assert run_cli(["calibrate", "--config", conf2]) == 2
    assert "settings.spoke_model" in capsys.readouterr().err


# ---------------------------------------------------------------------------- report

def test_report_shows_model_column_and_per_model_cost_only_when_mixed(mixed, single):
    md = (mixed["run_dir"] / "report" / "report.md").read_text()
    agents = md.split("## Agents\n", 1)[1].split("\n## ", 1)[0]
    assert "| agent | model |" in agents
    rows = {ln.split("|")[1].strip(): ln.split("|")[2].strip() for ln in agents.splitlines()
            if ln.startswith("| ") and not ln.startswith("| agent")}
    assert rows["cio"] == HUB and rows["macro_rates"] == SPOKE and rows["control"] == SPOKE
    cost = md.split("## Cost\n", 1)[1].split("\n## ", 1)[0]
    cp = ckpt(mixed["run_dir"])
    control = json.loads((mixed["run_dir"] / "control.json").read_text())["spent_usd"]
    assert f"- {HUB}: ${cp['spent_by_model'][HUB]:.2f}" in cost
    assert f"- {SPOKE}: ${cp['spent_by_model'][SPOKE] + control:.2f}" in cost
    assert "mixed models: results cannot be attributed to the hub or the spokes" in md.lower()
    assert "model" in (mixed["run_dir"] / "report" / "agents.csv").read_text().splitlines()[0].split(",")
    # a single-model report has none of it
    smd = (single["run_dir"] / "report" / "report.md").read_text()
    sagents = smd.split("## Agents\n", 1)[1].split("\n## ", 1)[0]
    assert "| agent | n_scored |" in sagents and "| model |" not in sagents
    scost = smd.split("## Cost\n", 1)[1].split("\n## ", 1)[0]
    assert f"- {HUB}:" not in scost and "unattributed" not in scost
    assert "mixed models" not in smd.lower()
    assert "model" not in (single["run_dir"] / "report" / "agents.csv").read_text().splitlines()[0].split(",")


def test_single_model_golden_run_is_byte_identical_to_before(single, repo, tmp_path, keys):
    st = run_state(single["run_dir"])
    # the prediction log of the single-model run is byte-identical to the pre-change run
    assert hashlib.sha256(st["predlog"]).hexdigest() == PRE_CHANGE_PREDLOG_SHA256
    cp = ckpt(single["run_dir"])
    assert "spent_by_model" not in cp  # the single-model checkpoint format is unchanged
    # and spoke_model set to the hub model itself is the same single-model run
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", repo,
                        settings={"spoke_model": HUB})
    stub = GoldenStub()
    assert main(["backtest", "--config", str(conf), "--with-control"], llm_client=StubClient(stub)) == 0
    assert stub.calls == single["calls"]
    other = run_state(tmp_path / "run")
    assert other == st
    assert json.loads((tmp_path / "run" / "control.json").read_text()) == \
        json.loads((single["run_dir"] / "control.json").read_text())



def test_report_guard_is_in_the_report_too(single, tmp_path, monkeypatch):
    """build_report adds model column / per-model cost only for MORE than one model, even when the
    caller passes a (single-model) agent_models map."""
    import shutil
    run = tmp_path / "run"
    shutil.copytree(single["run_dir"], run)
    before = (run / "report" / "report.md").read_text()
    monkeypatch.setattr("atlas.cli.mixed_models", lambda *a: True)  # cli passes the maps anyway
    assert run_cli(["report", "--run-dir", run]) == 0
    assert (run / "report" / "report.md").read_text().replace(str(run), str(single["run_dir"])) == before


def test_autoresearch_replays_route_each_agent_to_its_model(mixed, tmp_path, keys):
    """Old-prompt replays of recorded days are pure cache hits of the mixed backtest (so each
    agent is replayed on the model that served it); a new prompt goes to the agent's model."""
    import shutil
    from datetime import date
    from atlas.cli import build_replay
    shutil.copytree(mixed["run_dir"], tmp_path / "run")
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run",
                        Path(yaml.safe_load(mixed["conf"].read_text())["repo"]), **MIXED)
    cl = clients()
    replay = build_replay(conf, llm_client=cl)
    lines = [json.loads(x) for x in (tmp_path / "run" / "prediction_log.jsonl").read_text().splitlines()]
    for line in (lines[3], lines[10]):
        day = date.fromisoformat(line["day"])
        for rec in line["agents"]:
            res = replay.run_agent(rec["agent"], day, replay.prompts[rec["agent"]])
            assert res.output.model_dump(mode="json") == rec["output"], rec["agent"]
    assert cl[HUB].calls == [] and cl[SPOKE].calls == []
    day = date.fromisoformat(lines[5]["day"])
    replay.run_agent("sector_energy", day, replay.prompts["sector_energy"] + "\nedited")
    replay.run_agent("cio", day, replay.prompts["cio"] + "\nedited")
    assert [c[0] for c in cl[SPOKE].calls] == ["sector_energy"]
    assert [c[0] for c in cl[HUB].calls] == ["cio"]
    assert replay.pipeline.llm is replay.run.llm  # proposals: the hub
    assert all(llm.budget is replay.run.llm.budget for llm in replay.run.llms.values())
