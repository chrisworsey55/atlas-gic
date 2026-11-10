"""The autoresearch loop: one targeted prompt edit per attempt, evaluated by paired replay
on TRAIN days, confirmed on VALIDATION days, kept via a git merge or reverted.

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
4. ``edit_gate`` first REJECTS edits that win by survivorship: more failed (None) or
   empty-pick days than the old prompt beyond max(2, 5% of days), a failure rate above
   ``max_failure_rate``, fewer pairs than ``rolling_ic_window`` (the n keep_margin was
   calibrated on) or than 80% of the old prompt's, or < 40 comparable days. Then
   KEEP only if ``diff > keep_margin`` (the null test's ``keep_margin()``, calibrated with
   window == ``Settings.rolling_ic_window``). ``keep_margin is None`` (not yet calibrated)
   never keeps: the attempt is recorded as ``uncalibrated``.
5. Validation confirmation (``Settings.confirm_on_validation``, default on): a train win is
   NOT merged yet. The same agent is replayed under both prompts on the ``n_confirm_days``
   most recent validation days whose scoring exit is also inside validation
   (``validation_days``), and the edit is kept only if there are at least
   ``confirm_min_days`` such days, ``edit_gate`` passes on the validation result (with
   ``min_days=confirm_min_days``) and the finite validation diff exceeds
   ``confirm_margin(keep_margin, rolling_ic_window, new_pairs)``. Otherwise the decision is
   ``unconfirmed`` (also when validation is too short: fail safe; no replay is run then).
   Note: the gate already requires ``new_pairs >= rolling_ic_window``, so the sqrt factor of
   ``confirm_margin`` is 1 whenever the gate passes (effective margin = keep_margin), and the
   effective minimum validation length is about rolling_ic_window / picks_per_day comparable
   days, which can exceed ``confirm_min_days`` (window 60 at 1-2 picks/day: 30-60 days).
   Kept => merged into the branch that was checked out; otherwise the branch is discarded
   and the original branch and prompt file stay byte-for-byte unchanged, including when
   evaluation or confirmation raises.

Holdout safety: replay days are the most recent train days whose scoring exit (``horizon``
trading days later) also lies inside train; ``check_train_only`` raises ``HoldoutViolation``
otherwise, and the run_one_day callable is wrapped so it can never be asked for another day.
Confirmation days get their own check (``check_validation_only``, run once before any edit)
and their own wrapper: neither guard accepts the other period's days, and a validation day
whose exit falls in the holdout (the last ``horizon`` validation days) is never used.

Train-only inputs: target selection (AgentStats) and the proposal's diagnostics are built
by run_autoresearch itself from the ScoreStore, using only picks dated on or before the last
train day whose scoring exit is still inside train (``collect_stats(max_date=...)``). They are
built once, before the first attempt: confirmation results never feed back into them.
Anonymised runs: ``DayContext.asof_label``/``ticker_map`` make the replay call the agent
exactly as the runner did (same label, so cache keys match) and map tickers back for scoring.

Each confirmation spends part of the validation set (the same most recent days each time), so
it is recorded: ``EditRecord.confirm`` holds the validation numbers (None when no confirmation
was attempted; ``n_days_requested == 0`` when validation was too short and nothing was
replayed). The number of confirmations run is the count of log lines with
``confirm.n_days_requested > 0``; ``Settings.max_edits`` caps attempts and so confirmations
PER CALL of run_autoresearch (per ``atlas autoresearch`` invocation), not per run directory:
repeated invocations keep consuming the same validation days, so count them across the log.
The validation replay's LLM calls go through the same ``CachedLLM`` (one budget) and are
included in ``cost_usd``.

Each attempt appends one ``EditRecord`` JSON line to ``<run_dir>/autoresearch_log.jsonl``
(flushed and fsynced immediately), written BEFORE git cleanup so a failing cleanup never
loses it; cleanup errors are logged and attached as a note to the original exception, which
is re-raised. ``BudgetExceeded`` stops the loop cleanly (a line is written for it too). Replay calls
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
from typing import Callable, Literal, NamedTuple, Sequence

from atlas.autoresearch.gitops import GitOps
from atlas.backtest.runner import Periods, _demap_results
from atlas.config import AgentSpec, Settings
from atlas.engine.llm import BudgetExceeded, CachedLLM
from atlas.engine.pipeline import Pipeline
from atlas.schemas import AgentResult
from atlas.scoring.metrics import hit_rate, rank_ic
from atlas.scoring.weights import AgentStats, collect_stats, shrunk_ic, window_frame

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



class DayPairs(NamedTuple):
    """One day's scored picks with their tickers, so abstentions can be imputed.
    The three lists are aligned by position; tickers must be unique within a day."""
    tickers: list[str]
    convictions: list[float]
    excess: list[float]


RunOneDay = Callable[[date, str], "DayPairs | tuple[list[float], list[float]] | None"]
Decision = Literal["kept", "reverted", "rejected", "uncalibrated", "unconfirmed", "error"]


class MalformedReplayOutput(ValueError):
    """A replay result whose lists are misaligned or whose tickers repeat. `side` is
    "old" or "new"; a malformed NEW result rejects the edit, a malformed OLD one is a bug."""
    def __init__(self, msg: str, side: str):
        super().__init__(msg)
        self.side = side


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
    update_weights). An agent missing from `failure_rates` is INELIGIBLE (unknown failure
    rate: fail safe). `avoid`
    (the previous target) is skipped if any other agent is eligible (cooldown).
    """
    pool = [s for s in stats if s.agent != "control" and s.n_picks >= warmup_picks and not math.isnan(s.ic)]
    if not pool:
        return None
    mean = sum(s.ic for s in pool) / len(pool)
    eligible = [s for s in pool if s.agent in failure_rates
                and failure_rates[s.agent] <= max_failure_rate]
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
    """`old_ic`/`new_ic` are over the `n_days` comparable days only (both prompts gave picks).
    The counts cover every requested day so survivorship can be checked (``edit_gate``):
    `*_failures` = days the prompt could not be run (None), `*_empty` = days it returned no
    picks, `*_pairs` = (conviction, excess) pairs on the comparable days."""
    old_ic: float
    new_ic: float
    n_days: int
    diff: float
    old_failures: int = 0
    new_failures: int = 0
    old_pairs: int = 0
    new_pairs: int = 0
    n_days_requested: int = 0
    old_empty: int = 0
    new_empty: int = 0
    old_imputed: int = 0
    new_imputed: int = 0


def _split(agent: str, d: date, out, side: str = "old") -> tuple[list[str] | None, list[float], list[float]]:
    """(tickers or None for a plain 2-tuple, convictions, excess); ValueError if misaligned."""
    if isinstance(out, DayPairs):
        t, c, e = out
        if not len(t) == len(c) == len(e):
            raise MalformedReplayOutput(
                f"{agent} {d}: {len(t)} tickers, {len(c)} convictions, {len(e)} excess returns", side)
        if len(set(t)) != len(t):
            raise MalformedReplayOutput(f"{agent} {d}: duplicate tickers in one day's picks", side)
        return list(t), list(c), list(e)
    if len(out[0]) != len(out[1]):
        raise MalformedReplayOutput(
            f"{agent} {d}: {len(out[0])} convictions vs {len(out[1])} excess returns", side)
    return None, list(out[0]), list(out[1])


def _impute(own: tuple[list[str], list[float], list[float]],
            other: tuple[list[str], list[float], list[float]]) -> tuple[list[float], list[float], int]:
    """Samples for `own` side: its own pairs plus, for every ticker only `other` picked, a
    COPY of `other`'s pair (its conviction and its cost-adjusted excess; the excess is an
    approximation). Copying the conviction makes abstaining neutral: the dropped pick is
    the same observation on both sides, so it cannot help or hurt the difference. (A fixed
    conviction of 0 was tried and rewards abstention: it ranks lowest when picks are mostly
    positive, next to the lowest excess.) A pick with NaN conviction or excess is never
    imputed. Returns (convictions, excess, number imputed)."""
    have = set(own[0])
    c, e = list(own[1]), list(own[2])
    k = 0
    for t, cv, ex in zip(*other):
        if t not in have and ex == ex and cv == cv:
            c.append(cv); e.append(ex); k += 1
    return c, e, k


def paired_replay(agent: str, old_prompt: str, new_prompt: str, days: list[date],
                  run_one_day: RunOneDay, *, impute: bool = False,
                  min_days: int = MIN_REPLAY_DAYS) -> PairedResult:
    """Rank IC of the agent's picks under each prompt, over the SAME days only.

    A day is comparable when both prompts returned non-empty (convictions, excess) pairs; at
    least MIN_REPLAY_DAYS distinct comparable days are required (InsufficientReplayDays
    otherwise). Restricting to comparable days alone is gameable (a prompt that abstains on
    its bad days drops them from both sides): ``edit_gate`` must be applied to the result.

    With ``impute`` and ``DayPairs`` results on both sides of a comparable day, a ticker only
    one prompt picked enters the other side as a copy of that pick (same conviction, same
    excess), so dropping a loser neither removes it from the sample nor earns credit. The IC uses the imputed samples;
    ``old_pairs``/``new_pairs`` stay RAW counts (what ``edit_gate`` compares). Plain 2-tuples
    are never imputed.
    """
    oc, oe, nc, ne = [], [], [], []
    n = 0
    imputed = {"old": 0, "new": 0}
    fails = {"old": 0, "new": 0}
    empty = {"old": 0, "new": 0}
    requested = sorted(set(days))
    raw = {"old": 0, "new": 0}
    for d in requested:
        old = run_one_day(d, old_prompt)
        new = run_one_day(d, new_prompt)
        parts = {}
        for key, out in (("old", old), ("new", new)):
            if out is None:
                fails[key] += 1
                continue
            parts[key] = _split(agent, d, out, key)
            if not parts[key][1]:
                empty[key] += 1
        if "old" not in parts or "new" not in parts or not parts["old"][1] or not parts["new"][1]:
            continue
        o, nw = parts["old"], parts["new"]
        raw["old"] += len(o[1]); raw["new"] += len(nw[1])
        if impute and o[0] is not None and nw[0] is not None:
            c1, e1, k1 = _impute(o, nw)
            c2, e2, k2 = _impute(nw, o)
            imputed["old"] += k1; imputed["new"] += k2
        else:
            c1, e1, c2, e2 = o[1], o[2], nw[1], nw[2]
        oc += c1; oe += e1; nc += c2; ne += e2
        n += 1
    if n < min_days:
        raise InsufficientReplayDays(
            f"{agent}: only {n} of {len(requested)} replay days with picks under both prompts "
            f"(< {min_days}); failures old {fails['old']} new {fails['new']}, "
            f"empty old {empty['old']} new {empty['new']}")
    old_ic, new_ic = rank_ic(oc, oe), rank_ic(nc, ne)
    return PairedResult(old_ic, new_ic, n, new_ic - old_ic, fails["old"], fails["new"], raw["old"], raw["new"],
                        len(requested), empty["old"], empty["new"], imputed["old"], imputed["new"])


def edit_gate(r: PairedResult, cfg: Settings, *, min_days: int = MIN_REPLAY_DAYS) -> str | None:
    """Reasons to REJECT an edit regardless of its IC gain (None if it may be judged on IC).

    Tolerance for extra failures/empty days: max(2, 5% of requested days). Rejects when the
    new prompt fails or abstains on more days than the old one beyond that tolerance, fails
    on more than ``max_failure_rate`` of the days, produces fewer pairs than
    ``rolling_ic_window`` (the n keep_margin is calibrated on) or fewer than ``min_pair_ratio`` of the
    old prompt's pairs, or there are fewer than `min_days` comparable days (MIN_REPLAY_DAYS on
    train, ``confirm_min_days`` on validation)."""
    tol = max(2, math.ceil(0.05 * r.n_days_requested))
    why = []
    if r.new_failures > r.old_failures + tol:
        why.append(f"new prompt failed on {r.new_failures} days vs {r.old_failures} (tolerance {tol})")
    if r.new_empty > r.old_empty + tol:
        why.append(f"new prompt gave no picks on {r.new_empty} days vs {r.old_empty} (tolerance {tol})")
    if r.n_days_requested and r.new_failures / r.n_days_requested > cfg.max_failure_rate:
        why.append(f"new prompt failure rate {r.new_failures}/{r.n_days_requested} > {cfg.max_failure_rate}")
    if r.new_pairs < cfg.rolling_ic_window:
        why.append(f"new prompt gave {r.new_pairs} pairs < rolling_ic_window {cfg.rolling_ic_window}")
    if r.new_pairs < cfg.min_pair_ratio * r.old_pairs:
        why.append(f"new prompt gave {r.new_pairs} pairs < min_pair_ratio {cfg.min_pair_ratio} of old {r.old_pairs}")
    if r.n_days < min_days:
        why.append(f"only {r.n_days} comparable days < {min_days}")
    return "; ".join(why) or None


@dataclass
class DayContext:
    """What the backtest showed the agents on a day. For an anonymised run pass the
    ANONYMISED snapshot/state/upstream exactly as the runner gave them to ``run_day``, the
    runner's label (``f"day {i}"``) as `asof_label` and the opaque->real map as `ticker_map`
    (picks are mapped back to real tickers before the scorer sees them)."""
    snapshot: object
    weights: dict[str, float]
    state: dict
    upstream: dict[str, AgentResult]
    asof_label: str | None = None
    ticker_map: dict[str, str] | None = None


def build_run_one_day(pipeline: Pipeline, agent_name: str,
                      day_context: Callable[[date], DayContext | None],
                      scorer: Callable[[AgentResult, date], "DayPairs | tuple[list[float], list[float]]"],
                      *, horizon: int = 1) -> RunOneDay:
    """run_one_day for paired_replay: re-run only `agent_name` with the given prompt on the
    day's recorded upstream results, called exactly as the runner calls it (same asof_label,
    so old-prompt calls are cache hits of the backtest; unchanged agents are not re-run).
    None if the day has no context or the agent produced no valid output.

    `horizon` is the scorer's horizon; a scorer with a ``horizon`` attribute must agree. The
    returned callable carries ``.horizon`` so run_autoresearch can check it against its own."""
    sh = getattr(scorer, "horizon", horizon)
    if sh != horizon:
        raise ValueError(f"scorer horizon {sh} != replay horizon {horizon}")

    def run(day: date, prompt: str):
        ctx = day_context(day)
        if ctx is None:
            return None
        res = pipeline.run_single(agent_name, day, ctx.snapshot, ctx.weights, ctx.state, ctx.upstream,
                                  prompt_override=prompt, asof_label=ctx.asof_label)
        if res.status != "ok":
            return None
        if ctx.ticker_map:
            res = _demap_results({agent_name: res}, ctx.ticker_map)[agent_name]
        return scorer(res, day)

    run.horizon = horizon  # type: ignore[attr-defined]
    return run


# ---------------------------------------------------------------- holdout safety

def _exit_day(day: date, trading_days: Sequence[date], horizon: int) -> date | None:
    td = sorted(set(trading_days))
    if day not in td:
        return None
    i = td.index(day) + horizon
    return td[i] if i < len(td) else None


def _check_inside(days: Sequence[date], trading_days: Sequence[date], rng: tuple[date, date],
                  name: str, horizon: int) -> None:
    lo, hi = rng
    for d in days:
        if not lo <= d <= hi:
            raise HoldoutViolation(f"replay day {d} is outside the {name} period {lo}..{hi}")
        ex = _exit_day(d, trading_days, horizon)
        if ex is None or ex > hi:
            raise HoldoutViolation(f"replay day {d}: scoring exit {ex} is not inside {name} (ends {hi})")


def _days_inside(trading_days: Sequence[date], rng: tuple[date, date], horizon: int) -> list[date]:
    """Trading days in `rng` whose `horizon`-day exit is a known trading day also in `rng`."""
    lo, hi = rng
    td = sorted(set(trading_days))
    return [d for i, d in enumerate(td) if lo <= d <= hi and i + horizon < len(td) and td[i + horizon] <= hi]


def check_train_only(days: Sequence[date], trading_days: Sequence[date], periods: Periods,
                     horizon: int = 1) -> None:
    _check_inside(days, trading_days, periods.train, "train", horizon)


def check_validation_only(days: Sequence[date], trading_days: Sequence[date], periods: Periods,
                          horizon: int = 1) -> None:
    """HoldoutViolation unless every day AND its `horizon`-day scoring exit lie in validation
    (so neither train days nor the last validation days, whose exit is in holdout, pass)."""
    _check_inside(days, trading_days, periods.validation, "validation", horizon)


def replay_days(trading_days: Sequence[date], periods: Periods, horizon: int = 1,
                n: int = 60) -> list[date]:
    """The `n` most recent train days whose `horizon`-day exit is also in train."""
    if n < MIN_REPLAY_DAYS:
        raise ValueError(f"n={n}: paired replay needs at least {MIN_REPLAY_DAYS} days")
    return _days_inside(trading_days, periods.train, horizon)[-n:]


def validation_days(trading_days: Sequence[date], periods: Periods, horizon: int = 1,
                    n: int = 60) -> list[date]:
    """The `n` most recent validation days whose `horizon`-day exit is also in validation
    (never a holdout day or exit). May be shorter than ``confirm_min_days``: the caller decides."""
    if n < 1:
        raise ValueError(f"n={n}: need at least one validation day")
    return _days_inside(trading_days, periods.validation, horizon)[-n:]


def confirm_margin(keep_margin: float, window: int, new_pairs: int) -> float:
    """The validation keep margin: ``keep_margin`` (calibrated on `window` pairs) scaled by
    sqrt(window / new_pairs) when the validation sample is smaller, never below keep_margin.
    A heuristic: the null spread of an IC shrinks roughly with the square root of the sample.

    Inert as used (a deliberate, stricter literal reading of the spec): the confirmation also
    applies ``edit_gate``, which requires ``new_pairs >= rolling_ic_window``, so whenever the
    gate passes the factor is 1 and the effective margin equals ``keep_margin``. The scaling
    only takes effect if the gate's pair floor is ever relaxed for validation."""
    return keep_margin * max(1.0, math.sqrt(window / max(1, new_pairs)))


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
    # how keep_margin was calibrated (seed, n_shuffles, window, train_max_date, ...): recorded so
    # re-running with different null-test seeds until a lucky margin appears is visible
    null_test: dict | None = None
    # validation confirmation of a train win (None when none was attempted): n_days_requested
    # (validation days replayed; 0 = no replay ran), n_days, old_ic, new_ic, diff, margin,
    # new_pairs, passed, reason. The top-level n_days/old_ic/new_ic/diff stay the TRAIN numbers.
    confirm: dict | None = None


def _num(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else x


_CONFIRM_FLOATS = ("old_ic", "new_ic", "diff", "margin")


def _confirm_info(n_requested: int, reason: str, val: PairedResult | None = None,
                  margin: float | None = None, passed: bool = False) -> dict:
    return {"n_days_requested": n_requested, "n_days": val.n_days if val else 0,
            "old_ic": val.old_ic if val else None, "new_ic": val.new_ic if val else None,
            "diff": val.diff if val else None, "margin": margin,
            "new_pairs": val.new_pairs if val else 0, "passed": passed, "reason": reason}


def _append(path: Path, rec: EditRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(rec)
    for f in ("old_ic", "new_ic", "diff", "keep_margin"):
        d[f] = _num(d[f])
    if d["confirm"] is not None:
        d["confirm"] = {k: (_num(v) if k in _CONFIRM_FLOATS else v) for k, v in d["confirm"].items()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def last_safe_train_day(trading_days: Sequence[date], periods: Periods, horizon: int = 1) -> date | None:
    """The last train day whose `horizon`-day exit is also inside train."""
    ok = _days_inside(trading_days, periods.train, horizon)
    return ok[-1] if ok else None


def train_stats(scores, agents: list[str], cfg: Settings, max_date: date, horizon: int = 1) -> list[AgentStats]:
    """Rolling-window AgentStats from picks dated <= max_date only (never validation/holdout)."""
    return collect_stats(scores, agents, cfg.rolling_ic_window, horizon, max_date=max_date)


def build_diagnostics(scores, agent: str, stats: list[AgentStats], cfg: Settings, max_date: date,
                      failure_rate: float, failure_notes: str = "", horizon: int = 1,
                      n_worst: int = 5) -> str:
    """Diagnostics shown to the proposal model, from train-period picks (<= max_date) only.
    Worst picks carry no dates (anonymised runs must not leak them into prompts)."""
    df = window_frame(scores, agent, cfg.rolling_ic_window, horizon, max_date)
    if len(df) and max(df["date"]) > max_date:
        raise HoldoutViolation(f"diagnostics for {agent} would use a pick after {max_date}")
    pool = [s for s in stats if s.agent != "control" and s.n_picks >= cfg.warmup_picks and not math.isnan(s.ic)]
    mean = sum(s.ic for s in pool) / len(pool) if pool else 0.0
    me = next(s for s in stats if s.agent == agent)
    lines = [f"agent: {agent}",
             f"rolling window: {me.n_picks} scored picks (train period only)",
             f"rank IC {me.ic:.4f}; shrunk IC {shrunk_ic(me.ic, me.n_picks, mean, cfg.shrinkage_k):.4f} "
             f"(group mean {mean:.4f}, k={cfg.shrinkage_k})",
             f"hit rate {hit_rate(df['signed_excess']):.3f}",
             f"failure rate {failure_rate:.3f}"]
    if failure_notes:
        lines.append(f"recent failure reasons: {failure_notes}")
    worst = df.sort_values(["signed_excess", "ticker"], kind="stable").head(n_worst)
    if len(worst):
        lines.append("recent worst picks (ticker, direction, signed conviction, direction-adjusted excess):")
        lines += [f"- {r.ticker} {r.direction} {r.conviction:+.3f} {r.signed_excess:+.4f}"
                  for r in worst.itertuples()]
    return "\n".join(lines)


def _confirm(agent: str, old_prompt: str, new_prompt: str, vdays: list[date], run_one_day: RunOneDay,
             keep_margin: float, cfg: Settings) -> dict:
    """Replay a train win on the validation days and judge it (``EditRecord.confirm``).
    Passed only with >= confirm_min_days days, the edit gate passing on the validation result
    (min_days=confirm_min_days) and a finite ``diff > confirm_margin(...)``. BudgetExceeded,
    MalformedReplayOutput and anything else propagate to the caller's handlers."""
    if len(vdays) < cfg.confirm_min_days:
        return _confirm_info(0, f"only {len(vdays)} validation days with in-validation exits "
                                f"(< confirm_min_days {cfg.confirm_min_days})")
    try:
        val = paired_replay(agent, old_prompt, new_prompt, vdays, run_one_day,
                            impute=cfg.impute_abstentions, min_days=cfg.confirm_min_days)
    except InsufficientReplayDays as e:
        return _confirm_info(len(vdays), f"replay: {e}")
    margin = confirm_margin(keep_margin, cfg.rolling_ic_window, val.new_pairs)
    gate = edit_gate(val, cfg, min_days=cfg.confirm_min_days)
    if gate is not None:
        return _confirm_info(len(vdays), f"gate: {gate}", val, margin)
    if math.isfinite(val.diff) and val.diff > margin:
        return _confirm_info(len(vdays), f"diff {val.diff:.4f} > margin {margin:.4f}", val, margin, True)
    return _confirm_info(len(vdays), f"diff {val.diff:.4f} <= margin {margin:.4f}", val, margin)


def run_autoresearch(cfg: Settings, *, repo: Path, periods: Periods, trading_days: Sequence[date],
                     specs: list[AgentSpec], llm: CachedLLM, scores,
                     failure_rates: dict[str, float],
                     make_run_one_day: Callable[[str], RunOneDay], keep_margin: float | None,
                     run_dir: Path, horizon: int = 1,
                     failure_notes: Callable[[str], str] | None = None,
                     n_replay_days: int = 60, null_test_meta: dict | None = None,
                     n_confirm_days: int = 60) -> list[EditRecord]:
    """Run up to cfg.max_edits attempts; see the module docstring.

    `scores` is the backtest's ScoreStore (anything with ``picks_frame``): target selection
    and the proposal's diagnostics are built HERE from picks whose scoring exit lies inside
    train. `failure_rates` (and optional `failure_notes`) must also come from train-period
    data; an agent missing from `failure_rates` is never targeted. Every
    ``make_run_one_day(agent)`` result must carry ``.horizon == horizon`` (as
    ``build_run_one_day`` does), else ValueError. With ``cfg.confirm_on_validation`` the same
    run_one_day is also asked for the `n_confirm_days` most recent validation days (whose exit
    is inside validation), behind its own guard."""
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
    # confirmation days: chosen and checked once, up front (a holdout day refuses the run before
    # any edit); fewer than confirm_min_days makes every train win "unconfirmed" without a replay
    vdays = validation_days(trading_days, periods, horizon, n_confirm_days) if cfg.confirm_on_validation else []
    check_validation_only(vdays, trading_days, periods, horizon)
    safe_end = last_safe_train_day(trading_days, periods, horizon)
    agents = [s.name for s in specs if s.kind != "control"]
    stats = train_stats(scores, agents, cfg, safe_end, horizon)
    allowed_train = frozenset(days)
    allowed_validation = frozenset(vdays)
    by_name = {s.name: s for s in specs}
    log_path = Path(run_dir) / LOG_NAME
    records: list[EditRecord] = []
    last: str | None = None

    def emit(rec: EditRecord) -> None:
        if null_test_meta is not None:
            rec.null_test = dict(null_test_meta)
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
        raw_run = make_run_one_day(target)
        rh = getattr(raw_run, "horizon", None)
        if rh != horizon:
            raise ValueError(f"run_one_day for {target} declares horizon {rh!r}, loop uses {horizon}")
        prompt_path = f"atlas/{spec.prompt_file}"
        current = (git.repo / prompt_path).read_text(encoding="utf-8")
        spent0 = llm.spent_usd
        cost = lambda: llm.spent_usd - spent0  # noqa: E731
        diag = build_diagnostics(scores, target, stats, cfg, safe_end, failure_rates[target],
                                 failure_notes(target) if failure_notes else "", horizon)
        try:
            new = propose_edit(llm, spec, current, diag)
        except BudgetExceeded as e:
            emit(EditRecord(target, None, 0, None, None, None, keep_margin, "error",
                            f"budget exceeded before proposal: {e}", None, cost()))
            break
        except InvalidEdit as e:
            emit(EditRecord(target, None, 0, None, None, None, keep_margin, "rejected", str(e), None, cost()))
            continue

        branch, i = f"autoresearch/{target}-{n}", n
        while git.branch_exists(branch):
            i += 1
            branch = f"autoresearch/{target}-{i}"
        base_sha = git.head_sha(original)

        def guarded(day: date, prompt: str, _run=raw_run):
            if day not in allowed_train:
                raise HoldoutViolation(f"replay requested non-train day {day}")
            return _run(day, prompt)

        def vguarded(day: date, prompt: str, _run=raw_run):
            if day not in allowed_validation:
                raise HoldoutViolation(f"confirmation replay requested non-validation day {day}")
            return _run(day, prompt)

        result: PairedResult | None = None
        confirm: dict | None = None
        pending: dict | None = None
        decision: Decision = "error"
        reason, sha, stop, created, merged = "", None, False, False, False
        err: BaseException | None = None
        try:
            git.create_branch(branch)
            created = True
            git.write_file(prompt_path, new)
            git.commit_file(prompt_path, f"autoresearch: targeted edit of {target} prompt")
            result = paired_replay(target, current, new, days, guarded,
                                    impute=cfg.impute_abstentions)
            gate = edit_gate(result, cfg)
            if gate is not None:
                decision, reason = "rejected", gate
            elif keep_margin is None:
                decision, reason = "uncalibrated", "keep_margin not calibrated by the null test: never keep"
            elif math.isfinite(result.diff) and result.diff > keep_margin:
                won = f"diff {result.diff:.4f} > keep_margin {keep_margin:.4f}"
                if cfg.confirm_on_validation:
                    # NOT merged yet: replay on validation days; the merge waits for confirmation
                    confirm = pending = _confirm_info(len(vdays), "confirmation replay did not complete")
                    confirm = _confirm(target, current, new, vdays, vguarded, keep_margin, cfg)
                if confirm is not None and not confirm["passed"]:
                    decision = "unconfirmed"
                    reason = f"{won} on train; not confirmed on validation: {confirm['reason']}"
                else:
                    sha = git.merge_branch(branch)
                    merged = True
                    decision = "kept"
                    reason = won + (f"; confirmed on validation: {confirm['reason']}" if confirm else "")
            else:
                decision, reason = "reverted", f"diff {result.diff:.4f} <= keep_margin {keep_margin:.4f}"
        except BudgetExceeded as e:
            decision, reason, stop = "error", f"budget exceeded during evaluation: {e}", True
        except InsufficientReplayDays as e:
            decision, reason = "rejected", f"replay: {e}"
        except MalformedReplayOutput as e:
            if e.side != "new":
                decision, reason, sha, err = "error", f"{type(e).__name__}: {e}", None, e
            else:
                decision, reason = "rejected", f"new prompt gave malformed replay output: {e}"
        except BaseException as e:  # noqa: BLE001 - recorded, cleaned up, then re-raised
            decision, reason, sha, err = "error", f"{type(e).__name__}: {e}", None, e
        if confirm is not None and confirm is pending:  # the confirmation replay raised
            confirm["reason"] = f"{confirm['reason']}: {reason}"
        # record BEFORE cleanup: a failing cleanup must never lose the attempt's record
        emit(EditRecord(target, branch, result.n_days if result else 0,
                        result.old_ic if result else None, result.new_ic if result else None,
                        result.diff if result else None, keep_margin, decision, reason, sha, cost(),
                        confirm=confirm))
        cleanup_err: BaseException | None = None
        try:
            if created and not merged:
                git.discard_branch(branch)
            if git.current_branch() != original:
                raise RuntimeError(f"autoresearch left {git.current_branch()} checked out, not {original}")
            if not merged and git.head_sha(original) != base_sha:
                raise RuntimeError(f"{original} moved during a non-kept attempt")
        except BaseException as ce:  # noqa: BLE001
            log.error("autoresearch cleanup after %s failed: %r", branch, ce)
            cleanup_err = ce
        if err is not None:
            if cleanup_err is not None:
                err.add_note(f"autoresearch cleanup also failed: {cleanup_err!r}")
            raise err
        if cleanup_err is not None:
            raise cleanup_err
        if stop:
            break
    return records
