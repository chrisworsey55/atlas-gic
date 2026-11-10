"""Docs stay truthful: documented CLI flags exist, limitations carry the new caveats, the example
config loads and README.md keeps its CRLF line endings."""
from __future__ import annotations

import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _atlas_commands() -> list[list[str]]:
    text = README.read_bytes().decode("utf-8").replace("\r\n", "\n")
    cmds = []
    for block in re.findall(r"```[a-z]*\n(.*?)```", text, flags=re.S):
        for line in block.splitlines():
            line = line.split("#", 1)[0].strip()
            if line.startswith("atlas "):
                cmds.append(shlex.split(line))
    return cmds


def _help(sub: str) -> str:
    r = subprocess.run([sys.executable, "-m", "atlas", sub, "--help"], capture_output=True, text=True,
                       cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_every_documented_cli_flag_exists():
    cmds = _atlas_commands()
    assert cmds, "README has no atlas command blocks"
    helps: dict[str, str] = {}
    seen_flags = 0
    for argv in cmds:
        sub = argv[1]
        helps.setdefault(sub, _help(sub))
        for tok in argv[2:]:
            if tok.startswith("--"):
                seen_flags += 1
                assert tok.split("=")[0] in helps[sub], f"`atlas {sub}` has no {tok}"
    assert seen_flags >= 5


def test_limitations_mention_cutoff_table_and_probe_models():
    from atlas.backtest.runner import LIMITATIONS
    text = " ".join(LIMITATIONS).lower()
    for needle in ("month", "guard, not a guarantee", "every model in use", "number of models"):
        assert needle in text, needle


def test_run_example_yaml_loads(tmp_path):
    from atlas.cli import load_run_config
    path = ROOT / "atlas" / "config" / "run.example.yaml"
    raw = path.read_text(encoding="utf-8")
    assert "# spoke_model: claude-haiku-5-5" in raw
    assert load_run_config(path) is not None
    # the commented example, once uncommented, is still a valid config
    tmp = tmp_path / "uncommented.yaml"
    tmp.write_text(raw.replace("# spoke_model: claude-haiku-5-5", "spoke_model: claude-haiku-5-5"),
                   encoding="utf-8")
    assert load_run_config(tmp) is not None


def test_readme_is_crlf_consistent():
    data = README.read_bytes()
    assert data.count(b"\r\n") == data.count(b"\n") == data.count(b"\r")
