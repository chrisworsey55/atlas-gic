"""The autoresearch loop: one targeted prompt edit per attempt, evaluated by paired replay
on TRAIN days only, kept via a git merge or reverted.

Per attempt (at most ``Settings.max_edits`` per call):

1. ``select_target`` picks the eligible agent with the LOWEST shrunk IC (past warm-up, IC
   not NaN, failure rate <= ``max_failure_rate``, never ``control``). Cooldown: the agent
   targeted by the previous attempt is skipped when any other agent is eligible, so one
   prompt is not hammered repeatedly while others are never looked at.
2. ``propose_edit`` asks the LLM for ONE targeted modification and the COMPLETE revised
   prompt; an edit that is empty, unchanged, too long or breaks the output contract is
   recorded as ``rejected`` and never evaluated.
3. On a feature branch ``autoresearch/<agent>-<n>`` the prompt file is written and committed,
   then ``paired_replay`` re-runs the agent under the old and the new prompt on the same
   train days (other agents' LLM calls are cache hits: see ``build_run_one_day``).
4. KEEP only if ``diff > keep_margin`` (the null test's ``keep_margin()``, calibrated with
   window == ``Settings.rolling_ic_window``). ``keep_margin is None`` (not yet calibrated)
   never keeps: the attempt is recorded as ``uncalibrated``. Kept => merged into the
   branch that was checked out; otherwise the branch is discarded and the original branch
   and prompt file stay byte-for-byte unchanged, including when evaluation raises.

Holdout safety: replay days are the most recent train days whose scoring exit (``horizon``
trading days later) also lies inside train; ``check_train_only`` raises ``HoldoutViolation``
otherwise, and the run_one_day callable is wrapped so it can never be asked for another day.

Each attempt appends one ``EditRecord`` JSON line to ``<run_dir>/autoresearch_log.jsonl``
(flushed and fsynced immediately). ``BudgetExceeded`` stops the loop cleanly. Replay calls
through the pipeline record their failures in the pipeline's FailureLog: give autoresearch
its own FailureLog so replay failures do not mix with the backtest's.
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Literal, Sequence

from atlas.autoresearch.gitops import GitOps
from atlas.backtest.runner import Periods
from atlas.config import AgentSpec, Settings
from atlas.engine.llm import BudgetExceeded, CachedLLM
from atlas.engine.pipeline import Pipeline
from atlas.schemas import AgentResult
from atlas.scoring.metrics import rank_ic
from atlas.scoring.weights import AgentStats, shrunk_ic

log = logging.getLogger(__name__)

MIN_REPLAY_DAYS = 40
MAX_PROMPT_CHARS = 20_000
CONTRACT_WORDS = ("picks", "conviction", "direction", "JSON")
PROPOSAL_MAX_TOKENS = 8192
LOG_NAME = "autoresearch_log.jsonl"

PROPOSAL_SYSTEM_PROMPT = """\
You improve the system prompt of one agent in a multi-agent equity research system. The
agent's picks are scored by rank IC: the rank correlation between its signed conviction
(+ for long, - for short) and the next-day excess return over its benchmark, net of costs.

You are shown the agent's CURRENT PROMPT and DIAGNOSTICS (its shrunk IC, hit rate, recent
failure reasons and some of its recent worst picks). Make exactly ONE targeted modification
that addresses the most likely cause of weak or failing output: add, change or remove one
rule, step or emphasis. Do not rewrite the prompt wholesale, do not change the agent's role
or sector, and do not add knowledge of specific future events or dates.

The output contract must keep working unchanged: the agent must still answer with ONLY one
JSON object with the same fields (view, confidence, picks with ticker, direction,
conviction, horizon_days, rationale).

Return the COMPLETE revised prompt text and nothing else: no commentary, no explanation,
no markdown code fences around it.
"""

_FENCE = re.compile(r"^```[A-Za-z0-9_-]*[ \t]*\n(.*?)\n?```\s*$", re.DOTALL)

RunOneDay = Callable[[date, str], "tuple[list[float], list[float]] | None"]
Decision = Literal["kept", "reverted", "rejected", "uncalibrated", "error"]


class InsufficientReplayDays(Exception):
    pass


class HoldoutViolation(Exception):
    pass


class InvalidEdit(Exception):
    pass


# ---------------------------------------------------------------- target selection

def select_target(stats: list[AgentStats], failure_rates: dict[str, float], max_failure_rate: float,
                  k: float, warmup_picks: int, *, avoid: str | None = None) -> str | None:
    """Eligible agent with the lowest shrunk IC (ties broken by name), or None.

    The shrinkage group mean is taken over agents past warm-up with a non-NaN IC (as in
    update_weights). An agent missing from `failure_rates` counts as 0% failures. `avoid`
    (the previous target) is skipped if any other agent is eligible (cooldown).
    """
    pool = [s for s in stats if s.agent != "control" and s.n_picks >= warmup_picks and not math.isnan(s.ic)]
    if not pool:
        return None
    mean = sum(s.ic for s in pool) / len(pool)
    eligible = [s for s in pool if failure_rates.get(s.agent, 0.0) <= max_failure_rate]
    ranked = sorted(eligible, key=lambda s: (shrunk_ic(s.ic, s.n_picks, mean, k), s.agent))
    if not ranked:
        return None
    others = [s for s in ranked if s.agent != avoid]
    return (others or ranked)[0].agent


# ---------------------------------------------------------------- proposal

def _strip_fences(text: str) -> str:
    t = text.strip()
    m = _FENCE.match(t)
    return m.group(1) if m else text


def validate_edit(current: str, new: str) -> None:
    if not new.strip():
        raise InvalidEdit("empty edit")
    if new.strip() == current.strip():
        raise InvalidEdit("edit is identical to the current prompt")
    if len(new) > 2 * len(current):
        raise InvalidEdit(f"edit is {len(new)} chars, more than 2x the original {len(current)}")
    if len(new) > MAX_PROMPT_CHARS:
        raise InvalidEdit(f"edit is {len(new)} chars, over the {MAX_PROMPT_CHARS} limit")
    missing = [w for w in CONTRACT_WORDS if w not in new]
    if missing:
        raise InvalidEdit(f"edit breaks the output contract: missing {', '.join(missing)}")


def propose_edit(llm: CachedLLM, spec: AgentSpec, current_prompt: str, diagnostics: str) -> str:
    """Ask for one targeted modification; return the validated COMPLETE revised prompt."""
    user = (f"AGENT: {spec.name} (layer {spec.layer}, kind {spec.kind}, benchmark {spec.benchmark})\n\n"
            f"DIAGNOSTICS:\n{diagnostics}\n\n"
            f"CURRENT PROMPT (between the markers):\n<<<PROMPT\n{current_prompt}\nPROMPT>>>\n")
    resp = llm.call(PROPOSAL_SYSTEM_PROMPT, user, max_tokens=PROPOSAL_MAX_TOKENS)
    if resp.stop_reason == "max_tokens":
        raise InvalidEdit("proposal truncated (max_tokens)")
    if resp.stop_reason == "refusal":
        raise InvalidEdit("model refused to propose an edit")
    new = _strip_fences(resp.text)
    if current_prompt.endswith("\n") and new.strip() and not new.endswith("\n"):
        new += "\n"
    validate_edit(current_prompt, new)
    return new


# ---------------------------------------------------------------- paired replay

@dataclass(frozen=True)
class PairedResult:
    old_ic: float
    new_ic: float
    n_days: int
    diff: float


def paired_replay(agent: str, old_prompt: str, new_prompt: str, days: list[date],
                  run_one_day: RunOneDay) -> PairedResult:
    """Rank IC of the agent's picks under each prompt, over the SAME days only.

    A day counts when both prompts returned non-empty (convictions, excess) pairs; at least
    MIN_REPLAY_DAYS distinct such days are required (InsufficientReplayDays otherwise).
    """
    oc, oe, nc, ne = [], [], [], []
    n = 0
    for d in sorted(set(days)):
        old = run_one_day(d, old_prompt)
        new = run_one_day(d, new_prompt)
        if old is None or new is None or not old[0] or not new[0]:
            continue
        for c, e in (old, new):
            if len(c) != len(e):
                raise ValueError(f"{agent} {d}: {len(c)} convictions vs {len(e)} excess returns")
        oc += old[0]; oe += old[1]; nc += new[0]; ne += new[1]
        n += 1
    if n < MIN_REPLAY_DAYS:
        raise InsufficientReplayDays(
            f"{agent}: only {n} replay days with data under both prompts (< {MIN_REPLAY_DAYS})")
    old_ic, new_ic = rank_ic(oc, oe), rank_ic(nc, ne)
    return PairedResult(old_ic, new_ic, n, new_ic - old_ic)


@dataclass
class DayContext:
    snapshot: object
    weights: dict[str, float]
    state: dict
    upstream: dict[str, AgentResult]


def build_run_one_day(pipeline: Pipeline, agent_name: str,
                      day_context: Callable[[date], DayContext | None],
                      scorer: Callable[[AgentResult, date], tuple[list[float], list[float]]]) -> RunOneDay:
    """run_one_day for paired_replay: re-run only `agent_name` with the given prompt on the
    day's recorded upstream results (unchanged agents are cache hits). None if the day has
    no context or the agent produced no valid output."""

    def run(day: date, prompt: str):
        ctx = day_context(day)
        if ctx is None:
            return None
        res = pipeline.run_single(agent_name, day, ctx.snapshot, ctx.weights, ctx.state, ctx.upstream,
                                  prompt_override=prompt)
        if res.status != "ok":
            return None
        return scorer(res, day)

    return run


# ---------------------------------------------------------------- holdout safety

def _exit_day(day: date, trading_days: Sequence[date], horizon: int) -> date | None:
    td = sorted(set(trading_days))
    if day not in td:
        return None
    i = td.index(day) + horizon
    return td[i] if i < len(td) else None


def check_train_only(days: Sequence[date], trading_days: Sequence[date], periods: Periods,
                     horizon: int = 1) -> None:
    lo, hi = periods.train
    for d in days:
        if not lo <= d <= hi:
            raise HoldoutViolation(f"replay day {d} is outside the train period {lo}..{hi}")
        ex = _exit_day(d, trading_days, horizon)
        if ex is None or ex > hi:
            raise HoldoutViolation(f"replay day {d}: scoring exit {ex} is not inside train (ends {hi})")


def replay_days(trading_days: Sequence[date], periods: Periods, horizon: int = 1,
                n: int = 60) -> list[date]:
    """The `n` most recent train days whose `horizon`-day exit is also in train."""
    if n < MIN_REPLAY_DAYS:
        raise ValueError(f"n={n}: paired replay needs at least {MIN_REPLAY_DAYS} days")
    lo, hi = periods.train
    td = sorted(set(trading_days))
    ok = [d for i, d in enumerate(td)
          if lo <= d <= hi and i + horizon < len(td) and td[i + horizon] <= hi]
    return ok[-n:]


# ---------------------------------------------------------------- the loop

@dataclass
class EditRecord:
    agent: str
    branch: str | None
    n_days: int
    old_ic: float | None
    new_ic: float | None
    diff: float | None
    keep_margin: float | None
    decision: Decision
    reason: str
    commit_sha: str | None
    cost_usd: float


def _num(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else x


def _append(path: Path, rec: EditRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(rec)
    for f in ("old_ic", "new_ic", "diff", "keep_margin"):
        d[f] = _num(d[f])
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def run_autoresearch(cfg: Settings, *, repo: Path, periods: Periods, trading_days: Sequence[date],
                     specs: list[AgentSpec], llm: CachedLLM, stats: list[AgentStats],
                     failure_rates: dict[str, float], diagnostics: Callable[[str], str],
                     make_run_one_day: Callable[[str], RunOneDay], keep_margin: float | None,
                     run_dir: Path, horizon: int = 1, n_replay_days: int = 60) -> list[EditRecord]:
    """Run up to cfg.max_edits attempts; see the module docstring. `stats`/`failure_rates`/
    `diagnostics` must be built from TRAIN-period data only by the caller."""
    if keep_margin is not None and not (math.isfinite(keep_margin) and keep_margin >= 0):
        raise ValueError(f"keep_margin must be a finite non-negative number or None, got {keep_margin!r}")
    git = GitOps(repo)
    git.require_clean()
    original = git.current_branch()
    days = replay_days(trading_days, periods, horizon, n_replay_days)
    if len(days) < MIN_REPLAY_DAYS:
        raise InsufficientReplayDays(
            f"only {len(days)} train days with in-train exits (< {MIN_REPLAY_DAYS})")
    check_train_only(days, trading_days, periods, horizon)
    allowed = frozenset(days)
    by_name = {s.name: s for s in specs}
    log_path = Path(run_dir) / LOG_NAME
    records: list[EditRecord] = []
    last: str | None = None

    def emit(rec: EditRecord) -> None:
        _append(log_path, rec)
        records.append(rec)

    for n in range(cfg.max_edits):
        target = select_target(stats, failure_rates, cfg.max_failure_rate, cfg.shrinkage_k,
                               cfg.warmup_picks, avoid=last)
        if target is None:
            log.info("autoresearch: no eligible agent")
            break
        last = target
        spec = by_name[target]
        prompt_path = f"atlas/{spec.prompt_file}"
        current = (git.repo / prompt_path).read_text(encoding="utf-8")
        spent0 = llm.spent_usd
        cost = lambda: llm.spent_usd - spent0  # noqa: E731
        try:
            new = propose_edit(llm, spec, current, diagnostics(target))
        except BudgetExceeded as e:
            log.info("autoresearch: budget exhausted before proposal: %s", e)
            break
        except InvalidEdit as e:
            emit(EditRecord(target, None, 0, None, None, None, keep_margin, "rejected", str(e), None, cost()))
            continue

        branch, i = f"autoresearch/{target}-{n}", n
        while git.branch_exists(branch):
            i += 1
            branch = f"autoresearch/{target}-{i}"
        base_sha = git.head_sha(original)
        raw_run = make_run_one_day(target)

        def guarded(day: date, prompt: str, _run=raw_run):
            if day not in allowed:
                raise HoldoutViolation(f"replay requested non-train day {day}")
            return _run(day, prompt)

        result: PairedResult | None = None
        decision: Decision = "error"
        reason, sha, stop, created = "", None, False, False
        try:
            git.create_branch(branch)
            created = True
            git.write_file(prompt_path, new)
            git.commit_file(prompt_path, f"autoresearch: targeted edit of {target} prompt")
            result = paired_replay(target, current, new, days, guarded)
            if keep_margin is None:
                decision, reason = "uncalibrated", "keep_margin not calibrated by the null test: never keep"
            elif math.isfinite(result.diff) and result.diff > keep_margin:
                decision, reason = "kept", f"diff {result.diff:.4f} > keep_margin {keep_margin:.4f}"
            else:
                decision, reason = "reverted", f"diff {result.diff:.4f} <= keep_margin {keep_margin:.4f}"
            if decision == "kept":
                sha = git.merge_branch(branch)
                created = False
        except BudgetExceeded as e:
            reason, stop = f"budget exceeded during evaluation: {e}", True
        except InsufficientReplayDays as e:
            reason = f"replay: {e}"
        except BaseException as e:
            decision, reason = "error", f"{type(e).__name__}: {e}"
            emit(EditRecord(target, branch, result.n_days if result else 0,
                            result.old_ic if result else None, result.new_ic if result else None,
                            result.diff if result else None, keep_margin, "error", reason, None, cost()))
            raise
        finally:
            if created:
                git.discard_branch(branch)
            if git.current_branch() != original:
                raise RuntimeError(f"autoresearch left {git.current_branch()} checked out, not {original}")
            if decision != "kept" and git.head_sha(original) != base_sha:
                raise RuntimeError(f"{original} moved during a non-kept attempt")
        emit(EditRecord(target, branch, result.n_days if result else 0,
                        result.old_ic if result else None, result.new_ic if result else None,
                        result.diff if result else None, keep_margin, decision, reason, sha, cost()))
        if stop:
            break
    return records
