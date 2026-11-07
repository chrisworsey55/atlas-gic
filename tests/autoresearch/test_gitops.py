import os

import pytest

from atlas.autoresearch.gitops import DirtyWorkingTree, ForbiddenPath, GitError, GitOps
from tests.autoresearch.conftest import PROMPT_A, git

P_A = "atlas/prompts/agent_a.md"


def test_gitops_rejects_paths_outside_prompts_dir(prompt_repo):
    g = GitOps(prompt_repo)
    g.create_branch("autoresearch/agent_a-0")
    for bad in ("atlas/config/limits.yaml", "README.md", "atlas/prompts_evil/x.md",
                "atlas/prompts"):
        with pytest.raises(ForbiddenPath):
            g.write_file(bad, "x")
        with pytest.raises(ForbiddenPath):
            g.commit_file(bad, "msg")
    # nothing outside the prompts dir was touched
    assert (prompt_repo / "atlas/config/limits.yaml").read_text() == "max_position_pct: 0.1\n"
    assert not (prompt_repo / "README.md").exists()


@pytest.mark.parametrize("bad", ["atlas/prompts/../config/limits.yaml",
                                 "atlas/prompts/sub/../../config/limits.yaml",
                                 "./atlas/prompts/../../outside.md"])
def test_gitops_rejects_path_traversal(prompt_repo, bad):
    g = GitOps(prompt_repo)
    with pytest.raises(ForbiddenPath):
        g.write_file(bad, "x")
    with pytest.raises(ForbiddenPath):
        g.commit_file(bad, "msg")


def test_gitops_rejects_absolute_paths(prompt_repo):
    g = GitOps(prompt_repo)
    absolute = str(prompt_repo / P_A)
    with pytest.raises(ForbiddenPath):
        g.write_file(absolute, "x")
    with pytest.raises(ForbiddenPath):
        g.commit_file(absolute, "msg")


def test_gitops_rejects_symlinks_resolving_outside(prompt_repo, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret\n")
    os.symlink(outside, prompt_repo / "atlas/prompts/link.md")
    os.symlink(prompt_repo / "atlas/config", prompt_repo / "atlas/prompts/cfgdir")
    git(prompt_repo, "add", "-A")
    git(prompt_repo, "commit", "-q", "-m", "links")
    g = GitOps(prompt_repo)
    for bad in ("atlas/prompts/link.md", "atlas/prompts/cfgdir/limits.yaml"):
        with pytest.raises(ForbiddenPath):
            g.write_file(bad, "x")
        with pytest.raises(ForbiddenPath):
            g.commit_file(bad, "msg")
    assert outside.read_text() == "secret\n"
    assert (prompt_repo / "atlas/config/limits.yaml").read_text() == "max_position_pct: 0.1\n"


def test_commit_merge_roundtrip_only_touches_prompt(prompt_repo):
    g = GitOps(prompt_repo)
    base = git(prompt_repo, "rev-parse", "main")
    g.create_branch("autoresearch/agent_a-0")
    assert g.current_branch() == "autoresearch/agent_a-0"
    g.write_file(P_A, PROMPT_A + "extra\n")
    sha = g.commit_file(P_A, "edit a")
    assert git(prompt_repo, "diff", "--name-only", base, sha) == P_A
    merged = g.merge_branch("autoresearch/agent_a-0")
    assert g.current_branch() == "main"
    assert git(prompt_repo, "rev-parse", "main") == merged
    assert (prompt_repo / P_A).read_text() == PROMPT_A + "extra\n"
    assert g.is_clean()
    assert not g.branch_exists("autoresearch/agent_a-0")


def test_discard_restores_original_branch_byte_for_byte(prompt_repo):
    g = GitOps(prompt_repo)
    base = git(prompt_repo, "rev-parse", "main")
    g.create_branch("autoresearch/agent_a-0")
    g.write_file(P_A, "changed\n")
    g.commit_file(P_A, "edit a")
    g.write_file(P_A, "changed again, uncommitted\n")  # mid-flight state
    (prompt_repo / "atlas/prompts/new_untracked.md").write_text("junk")
    g.discard_branch("autoresearch/agent_a-0")
    assert g.current_branch() == "main"
    assert git(prompt_repo, "rev-parse", "main") == base
    assert (prompt_repo / P_A).read_text() == PROMPT_A
    assert not (prompt_repo / "atlas/prompts/new_untracked.md").exists()
    assert g.is_clean()
    assert not g.branch_exists("autoresearch/agent_a-0")


def test_commit_only_commits_the_named_file(prompt_repo):
    g = GitOps(prompt_repo)
    g.create_branch("autoresearch/agent_a-0")
    g.write_file(P_A, "a2\n")
    g.write_file("atlas/prompts/agent_b.md", "b2\n")
    sha = g.commit_file(P_A, "edit a")
    assert git(prompt_repo, "show", "--name-only", "--format=", sha) == P_A


def test_merge_refuses_branch_touching_files_outside_prompts(prompt_repo):
    g = GitOps(prompt_repo)
    g.create_branch("autoresearch/agent_a-0")
    (prompt_repo / "atlas/config/limits.yaml").write_text("max_position_pct: 1.0\n")
    git(prompt_repo, "commit", "-q", "-am", "sneaky")  # bypasses GitOps
    base = git(prompt_repo, "rev-parse", "main")
    with pytest.raises(ForbiddenPath):
        g.merge_branch("autoresearch/agent_a-0")
    assert git(prompt_repo, "rev-parse", "main") == base


def test_dirty_working_tree_detected(prompt_repo):
    g = GitOps(prompt_repo)
    assert g.is_clean()
    (prompt_repo / "atlas/config/limits.yaml").write_text("changed\n")
    assert not g.is_clean()
    with pytest.raises(DirtyWorkingTree):
        g.require_clean()


def test_untracked_file_in_prompts_counts_as_dirty(prompt_repo):
    g = GitOps(prompt_repo)
    (prompt_repo / "atlas/prompts/draft.md").write_text("draft")
    assert not g.is_clean()


def test_create_branch_refuses_existing_or_invalid_names(prompt_repo):
    g = GitOps(prompt_repo)
    git(prompt_repo, "branch", "autoresearch/x-0")
    with pytest.raises(GitError):
        g.create_branch("autoresearch/x-0")
    with pytest.raises(GitError):
        g.create_branch("bad name..lock")
    assert g.current_branch() == "main"
