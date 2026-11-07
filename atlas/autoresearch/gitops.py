"""Prompt-only git operations for autoresearch.

Every file operation is confined to ``atlas/prompts/`` inside the repository: absolute
paths, any ``..`` component, symlinks (the file itself, or a directory on the way that
resolves outside the prompts dir) and anything else raise ``ForbiddenPath``. Merges are
refused if the branch changes any file outside the prompts dir, even one committed behind
GitOps' back. Nothing here ever pushes.

"Clean" means: no tracked change anywhere (staged or not) and no untracked file under
``atlas/prompts/`` (one could be overwritten or deleted when a branch is discarded).
Untracked files elsewhere (e.g. a run directory) are not touched and do not count.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath

PROMPTS_DIR = PurePosixPath("atlas/prompts")


class ForbiddenPath(Exception):
    pass


class DirtyWorkingTree(Exception):
    pass


class GitError(Exception):
    pass


class GitOps:
    def __init__(self, repo: Path):
        self.repo = Path(repo).resolve()
        self.prompts_root = (self.repo / PROMPTS_DIR).resolve()
        if not self.prompts_root.is_relative_to(self.repo):
            raise ForbiddenPath(f"{PROMPTS_DIR} resolves outside the repository")
        self._bases: dict[str, str] = {}

    # ---------------------------------------------------------------- plumbing
    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
        p = subprocess.run(["git", *args], cwd=self.repo, capture_output=True, text=True, env=env)
        if check and p.returncode != 0:
            raise GitError(f"git {' '.join(args)} failed: {p.stderr.strip() or p.stdout.strip()}")
        return p

    def _check_path(self, path: str | os.PathLike) -> str:
        """Validate a repo-relative path to a file under atlas/prompts/; return it as posix."""
        raw = str(path)
        if os.path.isabs(raw) or raw.startswith(("/", "\\")):
            raise ForbiddenPath(f"absolute path not allowed: {raw}")
        rel = PurePosixPath(raw.replace("\\", "/"))
        if ".." in rel.parts:
            raise ForbiddenPath(f"path traversal not allowed: {raw}")
        rel = PurePosixPath(*[p for p in rel.parts if p != "."]) if rel.parts else rel
        if not rel.is_relative_to(PROMPTS_DIR) or rel == PROMPTS_DIR:
            raise ForbiddenPath(f"only files under {PROMPTS_DIR}/ may be changed: {raw}")
        full = self.repo / rel
        if full.is_symlink():
            raise ForbiddenPath(f"symlinks not allowed: {raw}")
        resolved = full.resolve()
        if not resolved.is_relative_to(self.prompts_root) or resolved == self.prompts_root:
            raise ForbiddenPath(f"path resolves outside {PROMPTS_DIR}/: {raw}")
        return rel.as_posix()

    # ---------------------------------------------------------------- queries
    def current_branch(self) -> str:
        name = self._git("branch", "--show-current").stdout.strip()
        if not name:
            raise GitError("detached HEAD: autoresearch needs a branch checked out")
        return name

    def head_sha(self, ref: str = "HEAD") -> str:
        return self._git("rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()

    def branch_exists(self, name: str) -> bool:
        return self._git("rev-parse", "--verify", "--quiet", f"refs/heads/{name}", check=False).returncode == 0

    def is_clean(self) -> bool:
        out = self._git("status", "--porcelain", "--untracked-files=all").stdout
        for line in out.splitlines():
            if not line.startswith("??"):
                return False
            if PurePosixPath(line[3:].strip().strip('"')).is_relative_to(PROMPTS_DIR):
                return False
        return True

    def require_clean(self) -> None:
        if not self.is_clean():
            raise DirtyWorkingTree("working tree has uncommitted changes; commit or stash them first")

    # ---------------------------------------------------------------- operations
    def create_branch(self, name: str) -> None:
        """Create and check out `name` from the current branch (remembered as its base)."""
        if self._git("check-ref-format", "--branch", name, check=False).returncode != 0:
            raise GitError(f"invalid branch name: {name!r}")
        if self.branch_exists(name):
            raise GitError(f"branch already exists: {name}")
        base = self.current_branch()
        self._git("checkout", "-q", "-b", name)
        self._bases[name] = base

    def write_file(self, path: str, text: str) -> None:
        rel = self._check_path(path)
        full = self.repo / rel
        full.parent.mkdir(parents=True, exist_ok=True)
        with open(full, "w", encoding="utf-8", newline="") as f:
            f.write(text)

    def commit_file(self, path: str, message: str) -> str:
        """Commit only `path` (nothing else staged is included); return the commit sha."""
        rel = self._check_path(path)
        self._git("add", "--", rel)
        self._git("commit", "-q", "-m", message, "--only", "--", rel)
        sha = self.head_sha()
        changed = self._git("show", "--name-only", "--format=", sha).stdout.split()
        if changed != [rel]:
            raise GitError(f"commit {sha} touched unexpected files: {changed}")
        return sha

    def _base_of(self, name: str) -> str:
        if name not in self._bases:
            raise GitError(f"branch {name} was not created by this GitOps")
        return self._bases[name]

    def merge_branch(self, name: str) -> str:
        """Merge `name` into the branch it was created from (fast-forward if possible), delete
        it, and return the new head sha of the base branch."""
        base = self._base_of(name)
        changed = self._git("diff", "--name-only", f"{base}...{name}").stdout.split()
        bad = [c for c in changed if not PurePosixPath(c).is_relative_to(PROMPTS_DIR)]
        if bad:
            raise ForbiddenPath(f"branch {name} changes files outside {PROMPTS_DIR}/: {bad}")
        self._git("checkout", "-q", base)
        if self._git("merge", "-q", "--ff-only", name, check=False).returncode != 0:
            p = self._git("merge", "-q", "--no-ff", "--no-edit", name, check=False)
            if p.returncode != 0:
                self._git("merge", "--abort", check=False)
                raise GitError(f"merge of {name} into {base} failed: {p.stderr.strip()}")
        self._git("branch", "-q", "-d", name)
        del self._bases[name]
        return self.head_sha()

    def discard_branch(self, name: str) -> None:
        """Drop `name` and any uncommitted prompt changes; return to the base branch."""
        base = self._base_of(name)
        prompts = PROMPTS_DIR.as_posix()
        if self.current_branch() == name:
            self._git("reset", "-q", "--", prompts, check=False)
            self._git("checkout", "-q", "--", prompts, check=False)
            self._git("clean", "-q", "-f", "-d", "--", prompts)
        self._git("checkout", "-q", base)
        self._git("branch", "-q", "-D", name)
        del self._bases[name]
