import subprocess
from pathlib import Path

import pytest

PROMPT_A = ("# agent_a\n\nYou rank stocks. Respond with ONLY one JSON object with picks; each pick has "
            "a ticker, a direction and a conviction.\n")
PROMPT_B = ("# agent_b\n\nYou rank stocks too. Respond with ONLY one JSON object with picks; each pick has "
            "a ticker, a direction and a conviction.\n")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def prompt_repo(tmp_path) -> Path:
    """A real throwaway git repo with atlas/prompts/ and an out-of-bounds config file."""
    repo = tmp_path / "repo"
    (repo / "atlas" / "prompts").mkdir(parents=True)
    (repo / "atlas" / "config").mkdir(parents=True)
    (repo / "atlas" / "prompts" / "agent_a.md").write_text(PROMPT_A)
    (repo / "atlas" / "prompts" / "agent_b.md").write_text(PROMPT_B)
    (repo / "atlas" / "config" / "limits.yaml").write_text("max_position_pct: 0.1\n")
    git(repo, "init", "-q", "-b", "main")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test@example.com")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "init")
    return repo
