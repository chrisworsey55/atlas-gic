"""CLI-level model training-cutoff checks (hub-and-spoke): a used model whose documented cutoff is
later than settings.model_cutoff refuses the run (exit 2); unknown models only warn."""
from tests.golden.stub import GoldenStub
from tests.golden.test_golden_backtest import (  # noqa: F401 - fixtures
    cli, golden, keys, write_config,
)


def test_cutoff_later_than_model_cutoff_refuses_run_exit_2(golden, tmp_path, keys, capsys):
    # golden model_cutoff is 2024-01-31; haiku-5-5 is documented as 2026-06-30
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"],
                        settings={"spoke_model": "claude-haiku-5-5"})
    stub = GoldenStub()
    assert cli(["backtest", "--config", conf], stub) == 2
    text = "".join(capsys.readouterr())
    assert "claude-haiku-5-5" in text and "2026-06-30" in text and "model_cutoff" in text
    assert "Traceback" not in text and stub.calls == 0 and not (tmp_path / "run" / "checkpoint.json").exists()
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 2   # the dry run refuses as well


def test_per_agent_override_model_checked_too(golden, tmp_path, keys, capsys):
    import yaml
    agents = yaml.safe_load(open("atlas/config/agents.yaml"))
    for a in agents["agents"]:
        if a["name"] == "sector_energy":
            a["model"] = "claude-opus-5-5"
    af = tmp_path / "agents.yaml"
    af.write_text(yaml.safe_dump(agents))
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"], agents_file=str(af))
    assert cli(["backtest", "--config", conf], GoldenStub()) == 2
    assert "claude-opus-5-5" in "".join(capsys.readouterr())


def test_unknown_model_only_warns_about_the_cutoff(golden, tmp_path, keys, capsys):
    conf = write_config(tmp_path / "run.yaml", tmp_path / "run", golden["repo"])
    assert cli(["backtest", "--config", conf, "--dry-run"]) == 0
    err = capsys.readouterr().err
    assert "golden-stub-model" in err and "no documented training cutoff" in err
