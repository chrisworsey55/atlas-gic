import hashlib
import json
import math
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from atlas.autoresearch.gitops import DirtyWorkingTree, GitOps
from atlas.autoresearch.loop import (
    MIN_REPLAY_DAYS,
    PROPOSAL_SYSTEM_PROMPT,
    edit_gate,
    DayContext,
    EditRecord,
    HoldoutViolation,
    InsufficientReplayDays,
    InvalidEdit,
    PairedResult,
    build_run_one_day,
    check_train_only,
    paired_replay,
    propose_edit,
    replay_days,
    run_autoresearch,
    select_target,
)
from atlas.backtest.runner import Periods
from atlas.config import AgentSpec, Settings
from atlas.data.store import Snapshot
from atlas.engine.failures import FailureLog
from atlas.engine.llm import BudgetExceeded, CachedLLM, StubClient
from atlas.engine.pipeline import Pipeline
from atlas.scoring.metrics import rank_ic
from atlas.scoring.weights import AgentStats
from tests.autoresearch.conftest import PROMPT_A, PROMPT_B, git

TRAIN = (date(2024, 1, 1), date(2024, 6, 28))
PERIODS = Periods(train=TRAIN, validation=(date(2024, 7, 1), date(2024, 9, 30)),
                  holdout=(date(2024, 10, 1), date(2024, 12, 31)))
TRADING_DAYS = [d.date() for d in pd.bdate_range("2024-01-02", "2024-12-31")]
LAST_TRAIN_DAY = max(d for d in TRADING_DAYS if d <= TRAIN[1])
SPECS = [AgentSpec(name="agent_a", layer=2, kind="sector", prompt_file="prompts/agent_a.md", benchmark="SPY"),
         AgentSpec(name="agent_b", layer=1, kind="macro", prompt_file="prompts/agent_b.md", benchmark="SPY")]
TICKERS = [f"T{i}" for i in range(5)]
NEW_SUFFIX = "Rank by the hidden signal.\n"


def settings(**kw) -> Settings:
    base = dict(universe=TICKERS + ["SPY"], commission_per_share=0.0, commission_min=0.0,
                slippage_bps=0.0, min_valid_agents=1, rolling_ic_window=60, warmup_picks=30,
                budget_usd=100.0, model="m", model_cutoff=date(2023, 1, 1), max_edits=1)
    base.update(kw)
    return Settings(**base)


def _u(*parts) -> float:
    """Deterministic uniform(-1, 1) from a key."""
    h = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 63 - 1.0


def excess(day, t) -> float:
    return _u("ex", day, t) * 0.02


def fake_run_one_day(skill_new: float = 1.0, skill_old: float = 0.0, seen=None, fail=None):
    """run_one_day factory: the new prompt's convictions track excess with `skill_new`."""

    def make(agent):
        def run(day, prompt):
            if seen is not None:
                seen.append((agent, day, prompt))
            if fail is not None:
                fail(day, prompt)
            skill = skill_new if prompt.endswith(NEW_SUFFIX) else skill_old
            conv = [skill * excess(day, t) / 0.02 + (1 - skill) * _u("cv", day, t) for t in TICKERS]
            return conv, [excess(day, t) for t in TICKERS]
        run.horizon = 1
        return run
    return make


def with_horizon(fn, h=1):
    fn.horizon = h
    return fn


class FakeScores:
    """Duck-typed ScoreStore: picks_frame(agent, horizon) over a fixed frame."""

    def __init__(self, df):
        self.df = df

    def picks_frame(self, agent, horizon):
        d = self.df[self.df.agent == agent].copy()
        d["horizon"] = horizon
        return d


def pick_rows(agent, days, ic_sign, ticker="TR"):
    rows = []
    for i, d in enumerate(days):
        conv = ((i % 7) + 1) / 10
        ex = ic_sign * conv * 0.01 + i * 1e-7
        rows.append((agent, d, ticker, "long", conv, 1, ex, ex, True))
    return rows


TRAIN_PICK_DAYS = [d for d in TRADING_DAYS if d < LAST_TRAIN_DAY][-100:]
VAL_PICK_DAYS = [d for d in TRADING_DAYS if PERIODS.validation[0] <= d <= PERIODS.validation[1]]


def scores(a=(100, -1.0), b=(100, 1.0), val=None):
    """Train picks per agent (n, IC sign); `val` adds validation-period picks {agent: sign}."""
    from atlas.scoring.store import COLUMNS
    rows = pick_rows("agent_a", TRAIN_PICK_DAYS[-a[0]:], a[1]) + pick_rows("agent_b", TRAIN_PICK_DAYS[-b[0]:], b[1])
    for agent, sign in (val or {}).items():
        rows += pick_rows(agent, VAL_PICK_DAYS, sign, ticker="VALT")
    return FakeScores(pd.DataFrame(rows, columns=COLUMNS))


def proposal_stub(new_text=None):
    def fn(system, user):
        assert system == PROPOSAL_SYSTEM_PROMPT
        if new_text is not None:
            return new_text
        cur = PROMPT_A if "# agent_a" in user else PROMPT_B
        return "```markdown\n" + cur + NEW_SUFFIX + "```"
    return fn


def make_llm(tmp_path, fn, budget=1e6, price=(1.0, 1.0)) -> CachedLLM:
    return CachedLLM(StubClient(fn), tmp_path / "llm.db", "m", price, budget)


def run(tmp_path, repo, *, llm=None, make=None, keep_margin=0.1, store=None, cfg=None,
        trading_days=TRADING_DAYS, rates=None, **kw):
    return run_autoresearch(
        cfg or settings(), repo=repo, periods=PERIODS, trading_days=trading_days, specs=SPECS,
        llm=llm or make_llm(tmp_path, proposal_stub()), scores=store or scores(),
        failure_rates=rates if rates is not None else {"agent_a": 0.0, "agent_b": 0.0},
        make_run_one_day=make or fake_run_one_day(), keep_margin=keep_margin,
        run_dir=tmp_path / "run", **kw)


def log_lines(tmp_path):
    p = tmp_path / "run" / "autoresearch_log.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def branches(repo):
    return sorted(git(repo, "branch", "--format=%(refname:short)").split())


# ---------------------------------------------------------------- select_target

def test_select_target_skips_warming_up_and_high_failure_agents():
    stats = [AgentStats("warm", 5, -0.9),        # warming up: lowest raw IC but not eligible
             AgentStats("flaky", 100, -0.5),     # failure rate too high
             AgentStats("nan", 100, math.nan),
             AgentStats("weak", 100, -0.1),
             AgentStats("strong", 100, 0.2),
             AgentStats("control", 100, -0.8)]   # never a target
    rates = {"flaky": 0.5, "weak": 0.1, "strong": 0.0, "warm": 0.0, "nan": 0.0, "control": 0.0}
    assert select_target(stats, rates, 0.2, k=20.0, warmup_picks=30) == "weak"


def test_select_target_uses_shrunk_ic():
    # few picks with a very low IC shrink toward the mean; many picks with a mild IC do not
    stats = [AgentStats("few", 31, -0.30), AgentStats("many", 2000, -0.20), AgentStats("good", 2000, 0.3)]
    rates = {"few": 0.0, "many": 0.0, "good": 0.0}
    assert select_target(stats, rates, 0.2, k=500.0, warmup_picks=30) == "many"
    assert select_target(stats, rates, 0.2, k=0.0, warmup_picks=30) == "few"


def test_select_target_none_when_all_ineligible():
    stats = [AgentStats("a", 5, -0.5), AgentStats("b", 100, math.nan), AgentStats("c", 100, -0.2),
             AgentStats("control", 100, -0.9)]
    assert select_target(stats, {"a": 0.0, "b": 0.0, "c": 0.9, "control": 0.0}, 0.2, 20.0, 30) is None
    assert select_target([], {}, 0.2, 20.0, 30) is None


def test_select_target_cooldown_avoids_last_agent_when_alternative_exists():
    stats = [AgentStats("a", 100, -0.3), AgentStats("b", 100, -0.1)]
    rates = {"a": 0.0, "b": 0.0}
    assert select_target(stats, rates, 0.2, 20.0, 30, avoid="a") == "b"
    assert select_target(stats, rates, 0.2, 20.0, 30, avoid="b") == "a"
    only_a = [AgentStats("a", 100, -0.3), AgentStats("b", 5, -0.1)]
    assert select_target(only_a, rates, 0.2, 20.0, 30, avoid="a") == "a"


# ---------------------------------------------------------------- propose_edit

def test_propose_edit_strips_fences_and_returns_complete_prompt(tmp_path):
    seen = []

    def fn(system, user):
        seen.append((system, user))
        return "```markdown\n" + PROMPT_A + NEW_SUFFIX + "```"

    out = propose_edit(make_llm(tmp_path, fn), SPECS[0], PROMPT_A, "shrunk IC -0.05; hit rate 0.41")
    assert out == PROMPT_A + NEW_SUFFIX
    system, user = seen[0]
    assert system == PROPOSAL_SYSTEM_PROMPT
    assert "one targeted" in PROPOSAL_SYSTEM_PROMPT.lower()
    assert PROMPT_A in user and "shrunk IC -0.05; hit rate 0.41" in user and "agent_a" in user


@pytest.mark.parametrize("reply", [
    "",                                                     # empty
    PROMPT_A,                                               # unchanged
    "```\n" + PROMPT_A + "```",                             # unchanged inside fences
    PROMPT_A.replace("JSON", "json") + NEW_SUFFIX,          # output contract broken
    PROMPT_A.replace("conviction", "score") + NEW_SUFFIX,
    PROMPT_A.replace("direction", "side") + NEW_SUFFIX,
    PROMPT_A.replace("picks", "ideas") + NEW_SUFFIX,
    PROMPT_A + "x" * (len(PROMPT_A) + 1),                   # more than 2x the original
])
def test_propose_edit_rejects_invalid_edits(tmp_path, reply):
    with pytest.raises(InvalidEdit):
        propose_edit(make_llm(tmp_path, lambda s, u: reply), SPECS[0], PROMPT_A, "d")


def test_propose_edit_rejects_over_20000_chars(tmp_path):
    big = PROMPT_A + "y" * 15_000
    with pytest.raises(InvalidEdit):
        propose_edit(make_llm(tmp_path, lambda s, u: big + "z" * 6_000), SPECS[0], big, "d")


# ---------------------------------------------------------------- paired_replay

def test_paired_replay_requires_at_least_40_days():
    days = TRADING_DAYS[:39]
    with pytest.raises(InsufficientReplayDays):
        paired_replay("agent_a", "old", "old" + NEW_SUFFIX, days, fake_run_one_day()("agent_a"))
    # 45 days offered but 6 cannot be run under one of the prompts -> 39 paired days
    days = TRADING_DAYS[:45]
    bad = set(days[:6])
    base = fake_run_one_day()("agent_a")

    def flaky(day, prompt):
        if day in bad and prompt.endswith(NEW_SUFFIX):
            return None
        return base(day, prompt)

    with pytest.raises(InsufficientReplayDays):
        paired_replay("agent_a", "old", "old" + NEW_SUFFIX, days, flaky)
    # duplicates do not count twice
    with pytest.raises(InsufficientReplayDays):
        paired_replay("agent_a", "old", "old" + NEW_SUFFIX, TRADING_DAYS[:20] * 3, base)
    assert MIN_REPLAY_DAYS == 40


def test_paired_replay_computes_rank_ic_on_identical_days():
    days = TRADING_DAYS[:50]
    fn = fake_run_one_day(skill_new=0.8)("agent_a")
    res = paired_replay("agent_a", "old", "old" + NEW_SUFFIX, days, fn)
    old_c, old_e, new_c, new_e = [], [], [], []
    for d in days:
        c, e = fn(d, "old"); old_c += c; old_e += e
        c, e = fn(d, "old" + NEW_SUFFIX); new_c += c; new_e += e
    assert isinstance(res, PairedResult)
    assert res.n_days == 50
    assert res.old_ic == pytest.approx(rank_ic(old_c, old_e))
    assert res.new_ic == pytest.approx(rank_ic(new_c, new_e))
    assert res.diff == pytest.approx(res.new_ic - res.old_ic)
    assert res.diff > 0.5


# ---------------------------------------------------------------- holdout safety

def test_holdout_days_never_passed_to_replay(tmp_path, prompt_repo):
    seen = []
    run(tmp_path, prompt_repo, make=fake_run_one_day(seen=seen))
    days = sorted({d for _, d, _ in seen})
    assert days, "replay ran"
    assert all(TRAIN[0] <= d <= TRAIN[1] for d in days)
    # the 1-day exit of every replay day must also lie in train: the last train day is dropped
    for d in days:
        exit_day = TRADING_DAYS[TRADING_DAYS.index(d) + 1]
        assert exit_day <= TRAIN[1]
    assert LAST_TRAIN_DAY not in days
    # the most recent eligible train days are used
    eligible = [d for d in TRADING_DAYS if d < LAST_TRAIN_DAY]
    assert days == eligible[-60:]


def test_replay_days_selection_and_train_only_check():
    days = replay_days(TRADING_DAYS, PERIODS, horizon=1, n=60)
    assert len(days) == 60 and days[-1] < LAST_TRAIN_DAY
    check_train_only(days, TRADING_DAYS, PERIODS, horizon=1)
    with pytest.raises(HoldoutViolation):
        check_train_only(days + [date(2024, 7, 1)], TRADING_DAYS, PERIODS, horizon=1)
    with pytest.raises(HoldoutViolation):  # in train, but its exit (next trading day) is not
        check_train_only(days + [LAST_TRAIN_DAY], TRADING_DAYS, PERIODS, horizon=1)
    with pytest.raises(HoldoutViolation):
        check_train_only([date(2024, 11, 1)], TRADING_DAYS, PERIODS, horizon=1)
    with pytest.raises(HoldoutViolation):  # exit unknown -> cannot prove it is in train
        check_train_only([TRADING_DAYS[-1]], TRADING_DAYS, PERIODS, horizon=1)
    # trading days that end inside train: the final train day has no known exit and is dropped
    only_train = [d for d in TRADING_DAYS if d <= TRAIN[1]]
    assert LAST_TRAIN_DAY not in replay_days(only_train, PERIODS, horizon=1, n=60)
    with pytest.raises(ValueError):
        replay_days(TRADING_DAYS, PERIODS, horizon=1, n=39)


def test_too_few_train_days_refuses_before_any_edit(tmp_path, prompt_repo):
    short = [d for d in TRADING_DAYS if d >= date(2024, 5, 20)]  # ~29 train days
    llm_calls = []
    llm = make_llm(tmp_path, lambda s, u: llm_calls.append(1) or "x")
    with pytest.raises(InsufficientReplayDays):
        run(tmp_path, prompt_repo, llm=llm, trading_days=short)
    assert llm_calls == []


def test_replay_guard_rejects_non_train_day_requests(tmp_path, prompt_repo, monkeypatch):
    import atlas.autoresearch.loop as loop

    monkeypatch.setattr(loop, "replay_days", lambda *a, **k: [date(2024, 10, 1)] * 40)
    with pytest.raises(HoldoutViolation):
        run(tmp_path, prompt_repo)
    assert git(prompt_repo, "branch", "--show-current") == "main"
    assert branches(prompt_repo) == ["main"]


# ---------------------------------------------------------------- keep / revert

def test_edit_kept_only_when_diff_exceeds_keep_margin(tmp_path, prompt_repo):
    days = replay_days(TRADING_DAYS, PERIODS, horizon=1, n=60)
    expected = paired_replay("agent_a", PROMPT_A, PROMPT_A + NEW_SUFFIX, days,
                             fake_run_one_day(skill_new=0.3)("agent_a"))
    base = git(prompt_repo, "rev-parse", "main")

    recs = run(tmp_path, prompt_repo, make=fake_run_one_day(skill_new=0.3), keep_margin=expected.diff)
    assert [r.decision for r in recs] == ["reverted"]  # diff == margin is not enough
    assert git(prompt_repo, "rev-parse", "main") == base

    recs = run(tmp_path, prompt_repo, make=fake_run_one_day(skill_new=0.3),
               keep_margin=expected.diff - 1e-9)
    r = recs[0]
    assert r.decision == "kept" and r.agent == "agent_a"
    assert r.diff == pytest.approx(expected.diff) and r.n_days == 60
    head = git(prompt_repo, "rev-parse", "main")
    assert head != base and r.commit_sha == head
    assert (prompt_repo / "atlas/prompts/agent_a.md").read_text() == PROMPT_A + NEW_SUFFIX
    assert git(prompt_repo, "diff", "--name-only", base, head) == "atlas/prompts/agent_a.md"
    assert git(prompt_repo, "branch", "--show-current") == "main"
    assert GitOps(prompt_repo).is_clean()
    assert branches(prompt_repo) == ["main"]


def test_edit_reverted_when_diff_below_margin_main_unchanged(tmp_path, prompt_repo):
    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=fake_run_one_day(skill_new=0.0), keep_margin=0.1)
    assert [r.decision for r in recs] == ["reverted"]
    assert recs[0].diff == pytest.approx(0.0) and recs[0].commit_sha is None
    assert git(prompt_repo, "rev-parse", "main") == base
    assert (prompt_repo / "atlas/prompts/agent_a.md").read_bytes() == PROMPT_A.encode()
    assert git(prompt_repo, "branch", "--show-current") == "main"
    assert GitOps(prompt_repo).is_clean()
    assert branches(prompt_repo) == ["main"]


def test_uncalibrated_keep_margin_never_keeps(tmp_path, prompt_repo):
    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=fake_run_one_day(skill_new=1.0), keep_margin=None)
    assert recs[0].decision == "uncalibrated" and recs[0].diff > 0.5
    assert recs[0].keep_margin is None
    assert git(prompt_repo, "rev-parse", "main") == base
    assert (prompt_repo / "atlas/prompts/agent_a.md").read_text() == PROMPT_A
    assert branches(prompt_repo) == ["main"]


def test_invalid_edit_rejected_and_recorded_without_evaluation(tmp_path, prompt_repo):
    seen = []
    llm = make_llm(tmp_path, proposal_stub(new_text=PROMPT_A.replace("JSON", "text") + NEW_SUFFIX))
    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, llm=llm, make=fake_run_one_day(seen=seen))
    assert [r.decision for r in recs] == ["rejected"]
    assert "JSON" in recs[0].reason
    assert seen == []
    assert git(prompt_repo, "rev-parse", "main") == base
    assert branches(prompt_repo) == ["main"]
    assert log_lines(tmp_path)[0]["decision"] == "rejected"


def test_dirty_working_tree_refused(tmp_path, prompt_repo):
    (prompt_repo / "atlas/config/limits.yaml").write_text("user edit in progress\n")
    calls = []
    llm = make_llm(tmp_path, lambda s, u: calls.append(1) or "x")
    with pytest.raises(DirtyWorkingTree):
        run(tmp_path, prompt_repo, llm=llm)
    assert calls == []
    assert (prompt_repo / "atlas/config/limits.yaml").read_text() == "user edit in progress\n"


def test_invalid_keep_margin_refused(tmp_path, prompt_repo):
    for bad in (math.nan, -0.1, math.inf):
        with pytest.raises(ValueError):
            run(tmp_path, prompt_repo, keep_margin=bad)


# ---------------------------------------------------------------- caps and failures

def test_loop_stops_at_max_edits_and_at_budget(tmp_path, prompt_repo):
    # max_edits: 3 attempts, cooldown alternates between the two eligible agents
    recs = run(tmp_path, prompt_repo, cfg=settings(max_edits=3), keep_margin=10.0)
    assert [r.agent for r in recs] == ["agent_a", "agent_b", "agent_a"]
    assert len(log_lines(tmp_path)) == 3

    # budget exhausted before the first proposal: clean stop, no records, no branches
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    llm = make_llm(tmp2, proposal_stub(), budget=1e-9)
    recs = run(tmp2, prompt_repo, llm=llm, cfg=settings(max_edits=3))
    assert [(r.decision, r.branch) for r in recs] == [("error", None)]
    assert "budget" in recs[0].reason and len(log_lines(tmp2)) == 1
    assert branches(prompt_repo) == ["main"]

    # budget exhausted during the second attempt's replay: that attempt is recorded as an
    # error, its branch is discarded and the loop stops cleanly
    tmp3 = tmp_path / "c"
    tmp3.mkdir()
    calls = {"n": 0}

    def make(agent):
        inner = fake_run_one_day()(agent)

        def run_one(day, prompt):
            calls["n"] += 1
            if calls["n"] > 130:  # first attempt = 120 calls; second attempt dies early
                raise BudgetExceeded("over budget")
            return inner(day, prompt)
        return with_horizon(run_one)

    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp3, prompt_repo, make=make, cfg=settings(max_edits=5), keep_margin=10.0)
    assert [r.decision for r in recs] == ["reverted", "error"]
    assert "budget" in recs[1].reason.lower()
    assert git(prompt_repo, "rev-parse", "main") == base
    assert branches(prompt_repo) == ["main"]
    assert len(log_lines(tmp3)) == 2


def test_failed_evaluation_leaves_main_unchanged(tmp_path, prompt_repo):
    base = git(prompt_repo, "rev-parse", "main")

    def boom(day, prompt):
        if prompt.endswith(NEW_SUFFIX):
            raise RuntimeError("pipeline crashed")

    with pytest.raises(RuntimeError, match="pipeline crashed"):
        run(tmp_path, prompt_repo, make=fake_run_one_day(fail=boom))
    assert git(prompt_repo, "rev-parse", "main") == base
    assert git(prompt_repo, "branch", "--show-current") == "main"
    assert (prompt_repo / "atlas/prompts/agent_a.md").read_bytes() == PROMPT_A.encode()
    assert GitOps(prompt_repo).is_clean()
    assert branches(prompt_repo) == ["main"]
    lines = log_lines(tmp_path)
    assert [x["decision"] for x in lines] == ["error"] and "pipeline crashed" in lines[0]["reason"]


def test_insufficient_replay_data_is_recorded_and_loop_continues(tmp_path, prompt_repo):
    def mostly_none(agent):
        inner = fake_run_one_day()(agent)
        return with_horizon(
            lambda day, prompt: None if (day.day % 2 and prompt.endswith(NEW_SUFFIX)) else inner(day, prompt))

    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=mostly_none, cfg=settings(max_edits=2))
    assert [r.decision for r in recs] == ["rejected", "rejected"]
    assert all("replay" in r.reason.lower() for r in recs)
    assert git(prompt_repo, "rev-parse", "main") == base
    assert branches(prompt_repo) == ["main"]


def test_original_branch_restored_and_used_as_merge_base(tmp_path, prompt_repo):
    git(prompt_repo, "checkout", "-q", "-b", "work")
    main_sha = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, cfg=settings(max_edits=2), keep_margin=0.1)
    assert [r.decision for r in recs] == ["kept", "kept"]
    assert git(prompt_repo, "branch", "--show-current") == "work"
    assert git(prompt_repo, "rev-parse", "main") == main_sha  # untouched
    assert recs[1].commit_sha == git(prompt_repo, "rev-parse", "work")
    assert (prompt_repo / "atlas/prompts/agent_a.md").read_text() == PROMPT_A + NEW_SUFFIX
    assert (prompt_repo / "atlas/prompts/agent_b.md").read_text() == PROMPT_B + NEW_SUFFIX
    assert branches(prompt_repo) == ["main", "work"]


def test_log_line_written_per_attempt(tmp_path, prompt_repo):
    recs = run(tmp_path, prompt_repo, cfg=settings(max_edits=2), keep_margin=10.0)
    lines = log_lines(tmp_path)
    assert len(lines) == 2
    assert set(lines[0]) == {"agent", "branch", "n_days", "old_ic", "new_ic", "diff", "keep_margin",
                             "decision", "reason", "commit_sha", "cost_usd", "null_test"}
    assert lines[0]["null_test"] is None  # optional: set when run_autoresearch gets null_test_meta
    assert lines[0]["agent"] == recs[0].agent and lines[0]["branch"] == "autoresearch/agent_a-0"
    assert lines[1]["branch"] == "autoresearch/agent_b-1"
    assert lines[0]["keep_margin"] == 10.0 and lines[0]["cost_usd"] > 0
    assert isinstance(recs[0], EditRecord)


def test_branch_name_collision_gets_fresh_suffix(tmp_path, prompt_repo):
    git(prompt_repo, "branch", "autoresearch/agent_a-0")  # left over from an earlier run
    recs = run(tmp_path, prompt_repo, keep_margin=10.0)
    assert recs[0].branch != "autoresearch/agent_a-0"
    assert recs[0].branch.startswith("autoresearch/agent_a-")
    assert branches(prompt_repo) == ["autoresearch/agent_a-0", "main"]


# ---------------------------------------------------------------- end to end with the real pipeline

def test_end_to_end_with_stub_model_and_real_pipeline(tmp_path, prompt_repo):
    calls = []

    def signal(day, t):
        return _u("sig", day, t)

    def fn(system, user):
        if system == PROPOSAL_SYSTEM_PROMPT:
            return "```markdown\n" + PROMPT_A + NEW_SUFFIX + "```"
        u = json.loads(user)
        asof = u["asof"]
        agent = "agent_a" if "# agent_a" in system else "agent_b"
        calls.append((agent, asof, system.endswith(NEW_SUFFIX)))
        if agent == "agent_b":
            return json.dumps({"agent": agent, "date": asof, "view": "neutral", "confidence": 0.5,
                               "picks": []})
        smart = system.endswith(NEW_SUFFIX)
        picks = []
        for t in TICKERS:
            s = signal(asof, t) if smart else _u("noise", asof, t)
            picks.append({"ticker": t, "direction": "long" if s >= 0 else "short",
                          "conviction": round(abs(s), 4), "horizon_days": 1})
        return json.dumps({"agent": agent, "date": asof, "view": "bullish", "confidence": 0.6,
                           "picks": picks})

    llm = make_llm(tmp_path, fn)
    prompts = {"agent_a": PROMPT_A, "agent_b": PROMPT_B}
    pipe = Pipeline(SPECS, prompts, llm, FailureLog(tmp_path / "f.db"), TICKERS + ["SPY"], 1)

    def snap(day):
        rows = [{"date": day, "ticker": t, "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                 "adj_close": 10.0, "volume": 1} for t in TICKERS + ["SPY"]]
        return Snapshot(asof=day, prices=pd.DataFrame(rows), macro={})

    def day_context(day):
        s = snap(day)
        up = {"agent_b": pipe.run_single("agent_b", day, s, {}, {}, {})}
        return DayContext(snapshot=s, weights={}, state={}, upstream=up)

    def scorer(result, day):
        conv, ex = [], []
        for p in result.output.picks:
            sign = 1.0 if p.direction == "long" else -1.0 if p.direction == "short" else 0.0
            conv.append(sign * p.conviction)
            ex.append(0.01 * signal(day.isoformat(), p.ticker) + 0.002 * _u("e", day, p.ticker))
        return conv, ex

    make = lambda agent: build_run_one_day(pipe, agent, day_context, scorer)  # noqa: E731
    store = scores(b=(3, 1.0))  # agent_b warming up
    recs = run(tmp_path, prompt_repo, llm=llm, make=make, store=store, keep_margin=0.2)
    assert len(recs) == 1
    r = recs[0]
    assert r.agent == "agent_a" and r.decision == "kept"
    assert r.new_ic > 0.8 and abs(r.old_ic) < 0.3 and r.n_days == 60
    assert (prompt_repo / "atlas/prompts/agent_a.md").read_text() == PROMPT_A + NEW_SUFFIX
    # unchanged agent: one live call per replay day despite two replays (the rest are cache hits)
    b_calls = [c for c in calls if c[0] == "agent_b"]
    assert len(b_calls) == 60 and len({c[1] for c in b_calls}) == 60
    a_new = [c for c in calls if c[0] == "agent_a" and c[2]]
    a_old = [c for c in calls if c[0] == "agent_a" and not c[2]]
    assert len(a_new) == 60 and len(a_old) == 60
    assert all(date.fromisoformat(c[1]) < LAST_TRAIN_DAY for c in calls)
    assert r.cost_usd > 0


def test_build_run_one_day_returns_none_when_day_or_agent_unavailable(tmp_path):
    llm = make_llm(tmp_path, lambda s, u: "not json")
    pipe = Pipeline(SPECS, {"agent_a": PROMPT_A, "agent_b": PROMPT_B}, llm,
                    FailureLog(tmp_path / "f.db"), TICKERS + ["SPY"], 1)
    snap = Snapshot(asof=TRADING_DAYS[0], prices=pd.DataFrame(
        [{"date": TRADING_DAYS[0], "ticker": "T0", "open": 1, "high": 1, "low": 1, "close": 1,
          "adj_close": 1, "volume": 1}]), macro={})
    fn = build_run_one_day(pipe, "agent_a",
                           lambda d: None if d != TRADING_DAYS[0] else DayContext(snap, {}, {}, {}),
                           lambda r, d: pytest.fail("scorer must not see failed results"))
    assert fn(TRADING_DAYS[1], PROMPT_A) is None  # no context for that day
    assert fn(TRADING_DAYS[0], PROMPT_A) is None  # agent failed (malformed JSON)


# ---------------------------------------------------------------- fix round 1

def test_survivorship_gaming_edit_is_rejected(tmp_path, prompt_repo):
    """demo1: both prompts are pure noise, but the new one abstains (None) on days its picks
    would score badly. The comparable-day IC gain is large; the edit must still be rejected."""
    def make(agent):
        def run_one(day, prompt):
            conv = [_u("noise", day, t, prompt) for t in TICKERS]
            ex = [excess(day, t) for t in TICKERS]
            if prompt.endswith(NEW_SUFFIX) and sum(c * e for c, e in zip(conv, ex)) < 0:
                return None
            return conv, ex
        return with_horizon(run_one)

    days = replay_days(TRADING_DAYS, PERIODS, horizon=1, n=100)
    gamed = paired_replay("agent_a", PROMPT_A, PROMPT_A + NEW_SUFFIX, days, make("agent_a"))
    assert gamed.diff > 0.2 and gamed.new_failures > 10 and gamed.old_failures == 0
    assert gamed.n_days_requested == 100 and gamed.n_days >= 40
    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=make, keep_margin=0.1, n_replay_days=100)
    assert recs[0].decision == "rejected" and "failed on" in recs[0].reason
    assert git(prompt_repo, "rev-parse", "main") == base and branches(prompt_repo) == ["main"]


def test_empty_pick_days_gaming_is_rejected(tmp_path, prompt_repo):
    def make(agent):
        inner = fake_run_one_day(skill_new=1.0)(agent)

        def run_one(day, prompt):
            c, e = inner(day, prompt)
            if prompt.endswith(NEW_SUFFIX) and day.day % 7 == 0:
                return [], []
            return c, e
        return with_horizon(run_one)

    recs = run(tmp_path, prompt_repo, make=make, keep_margin=0.1)
    assert recs[0].decision == "rejected" and "no picks" in recs[0].reason


def test_pair_count_collapse_is_rejected(tmp_path, prompt_repo):
    def make(agent):
        inner = fake_run_one_day(skill_new=1.0)(agent)

        def run_one(day, prompt):
            c, e = inner(day, prompt)
            return (c[:2], e[:2]) if prompt.endswith(NEW_SUFFIX) else (c, e)  # 2 picks/day vs 5
        return with_horizon(run_one)

    recs = run(tmp_path, prompt_repo, make=make, keep_margin=0.1)
    assert recs[0].decision == "rejected" and "80%" in recs[0].reason
    # fewer pairs than the window keep_margin was calibrated on
    recs = run(tmp_path, prompt_repo, cfg=settings(rolling_ic_window=400), keep_margin=0.1)
    assert recs[-1].decision == "rejected" and "rolling_ic_window" in recs[-1].reason


def test_equal_failures_with_real_gain_still_kept(tmp_path, prompt_repo):
    def make(agent):
        inner = fake_run_one_day(skill_new=1.0)(agent)
        return with_horizon(lambda day, prompt: None if day.day == 15 else inner(day, prompt))

    recs = run(tmp_path, prompt_repo, make=make, keep_margin=0.1)
    assert recs[0].decision == "kept"


def test_edit_gate_thresholds():
    cfg = settings(rolling_ic_window=60, max_failure_rate=0.2)
    ok = PairedResult(0.0, 0.3, 60, 0.3, old_failures=1, new_failures=4, old_pairs=300, new_pairs=240,
                      n_days_requested=60)
    assert edit_gate(ok, cfg) is None  # tolerance max(2, 3) = 3 extra failures, 80% pairs exactly
    from dataclasses import replace
    assert edit_gate(replace(ok, new_failures=5), cfg) is not None
    assert edit_gate(replace(ok, new_pairs=239), cfg) is not None
    assert edit_gate(replace(ok, new_pairs=59, old_pairs=59), cfg) is not None
    assert edit_gate(replace(ok, old_failures=13, new_failures=13), cfg) is not None  # > 20%
    assert edit_gate(replace(ok, new_empty=4), cfg) is not None
    assert edit_gate(replace(ok, n_days=39), cfg) is not None


def test_target_and_diagnostics_use_train_picks_only(tmp_path, prompt_repo):
    # train: a weak, b strong. validation: a strong, b terrible (would flip the target).
    store = scores(a=(100, -1.0), b=(100, 1.0), val={"agent_a": 1.0, "agent_b": -1.0})
    users = []

    def fn(system, user):
        users.append(user)
        return "```markdown\n" + PROMPT_A + NEW_SUFFIX + "```"

    recs = run(tmp_path, prompt_repo, llm=make_llm(tmp_path, fn), store=store, keep_margin=10.0)
    assert recs[0].agent == "agent_a"
    assert "VALT" not in users[0] and "TR" in users[0]  # worst picks come from train
    assert "train period only" in users[0] and "rank IC -" in users[0]
    # sanity: with the validation picks counted, b would have been chosen
    from atlas.scoring.weights import collect_stats
    full = collect_stats(store, ["agent_a", "agent_b"], 60)
    assert select_target(full, {"agent_a": 0.0, "agent_b": 0.0}, 0.2, 20.0, 30) == "agent_b"


def test_stats_exclude_last_train_day_whose_exit_is_in_validation(tmp_path, prompt_repo):
    from atlas.scoring.store import COLUMNS
    rows = pick_rows("agent_a", TRAIN_PICK_DAYS, -1.0) + pick_rows("agent_b", TRAIN_PICK_DAYS, 1.0)
    rows += pick_rows("agent_a", [LAST_TRAIN_DAY], -10.0, ticker="EDGE")  # worst pick if counted
    users = []

    def fn(system, user):
        users.append(user)
        return "```markdown\n" + PROMPT_A + NEW_SUFFIX + "```"

    run(tmp_path, prompt_repo, llm=make_llm(tmp_path, fn),
        store=FakeScores(pd.DataFrame(rows, columns=COLUMNS)), keep_margin=10.0)
    assert "EDGE" not in users[0] and "rolling window: 60 scored picks" in users[0]


def test_missing_failure_rate_is_ineligible():
    stats = [AgentStats("a", 100, -0.3), AgentStats("b", 100, -0.1)]
    assert select_target(stats, {"b": 0.0}, 0.2, 20.0, 30) == "b"
    assert select_target(stats, {}, 0.2, 20.0, 30) is None


def test_horizon_mismatch_raises(tmp_path, prompt_repo):
    inner = fake_run_one_day()("agent_a")
    with pytest.raises(ValueError):
        run(tmp_path, prompt_repo, make=lambda a: with_horizon(lambda d, p: inner(d, p), 5))
    with pytest.raises(ValueError):  # undeclared horizon fails safe
        run(tmp_path, prompt_repo, make=lambda a: (lambda d, p: inner(d, p)))
    pipe = Pipeline(SPECS, {"agent_a": PROMPT_A, "agent_b": PROMPT_B}, make_llm(tmp_path, lambda s, u: "x"),
                    FailureLog(tmp_path / "f.db"), TICKERS, 1)
    scorer = lambda r, d: ([], [])  # noqa: E731
    scorer.horizon = 5
    with pytest.raises(ValueError):
        build_run_one_day(pipe, "agent_a", lambda d: None, scorer, horizon=1)
    assert build_run_one_day(pipe, "agent_a", lambda d: None, scorer, horizon=5).horizon == 5
    assert branches(prompt_repo) == ["main"]


def test_cleanup_failure_keeps_record_and_original_exception(tmp_path, prompt_repo, monkeypatch):
    from atlas.autoresearch.gitops import GitError

    def boom(day, prompt):
        if prompt.endswith(NEW_SUFFIX):
            raise RuntimeError("pipeline crashed")

    def broken_discard(self, name):
        raise GitError("discard failed")

    monkeypatch.setattr(GitOps, "discard_branch", broken_discard)
    with pytest.raises(RuntimeError, match="pipeline crashed") as ei:
        run(tmp_path, prompt_repo, make=fake_run_one_day(fail=boom))
    assert any("discard failed" in n for n in getattr(ei.value, "__notes__", []))
    lines = log_lines(tmp_path)
    assert [x["decision"] for x in lines] == ["error"] and "pipeline crashed" in lines[0]["reason"]
    # without an evaluation error the cleanup error itself surfaces, after the record
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    git(prompt_repo, "checkout", "-q", "-f", "main")
    for b in branches(prompt_repo):
        if b != "main":
            git(prompt_repo, "branch", "-q", "-D", b)
    with pytest.raises(GitError, match="discard failed"):
        run(tmp2, prompt_repo, keep_margin=10.0)
    assert [x["decision"] for x in log_lines(tmp2)] == ["reverted"]


def test_anonymised_replay_uses_label_and_opaque_tickers_and_hits_backtest_cache(tmp_path):
    from atlas.backtest.runner import anonymize_snapshot

    real = ["XOMX", "CVXX", "COPX", "SLBX", "SPY"]
    sectors = {t: "energy" for t in real}
    live = []

    def fn(system, user):
        u = json.loads(user)
        for t in real[:-1]:
            assert t not in user, "real ticker leaked into an anonymised prompt"
        assert "2024-" not in user, "real date leaked into an anonymised prompt"
        live.append((system[:9], u["asof"], system.endswith(NEW_SUFFIX)))
        if "# agent_b" in system:
            return json.dumps({"view": "neutral", "confidence": 0.5, "picks": []})
        ids = sorted(u["inputs"]["market"]["prices"])
        picks = [{"ticker": t, "direction": "long", "conviction": round((i + 1) / 10, 2), "horizon_days": 1}
                 for i, t in enumerate(ids)]
        return json.dumps({"view": "bullish", "confidence": 0.5, "picks": picks})

    llm = make_llm(tmp_path, fn)
    days = TRADING_DAYS[:3]
    ctx = {}
    pipe = Pipeline(SPECS, {"agent_a": PROMPT_A, "agent_b": PROMPT_B}, llm,
                    FailureLog(tmp_path / "f.db"), real, 1)
    for i, d in enumerate(days):  # the backtest, as the runner calls it
        rows = [{"date": d, "ticker": t, "open": 10.0 + j, "high": 10.0 + j, "low": 10.0 + j,
                 "close": 10.0 + j, "adj_close": 10.0 + j, "volume": 1} for j, t in enumerate(real)]
        anon, back = anonymize_snapshot(Snapshot(asof=d, prices=pd.DataFrame(rows), macro={}), sectors)
        pipe.universe = set(back)
        res = pipe.run_day(d, anon, {}, {"equity": 1.0}, asof_label=f"day {i}")
        ctx[d] = DayContext(anon, {}, {"equity": 1.0}, {"agent_b": res.results["agent_b"]},
                            asof_label=f"day {i}", ticker_map=back)
    n_backtest = len(live)
    seen = []

    def scorer(result, day):
        seen.extend(p.ticker for p in result.output.picks)
        return [p.conviction for p in result.output.picks], [0.0] * len(result.output.picks)

    fn1 = build_run_one_day(pipe, "agent_a", ctx.get, scorer)
    for d in days:
        assert fn1(d, PROMPT_A) is not None
    assert len(live) == n_backtest  # old-prompt replay = cache hits of the backtest calls
    for d in days:
        fn1(d, PROMPT_A + NEW_SUFFIX)
    assert len(live) == n_backtest + 3 and all(x[1].startswith("day ") for x in live)
    assert set(seen) <= set(real) and seen  # the scorer sees real tickers


def test_null_test_meta_is_recorded_in_every_log_line(tmp_path, prompt_repo):
    meta = {"seed": 3, "n_shuffles": 50, "window": 60, "train_max_date": "2024-03-01", "keep_margin": 10.0}
    recs = run(tmp_path, prompt_repo, cfg=settings(max_edits=2), keep_margin=10.0, null_test_meta=meta)
    lines = log_lines(tmp_path)
    assert len(lines) == 2 and all(line["null_test"] == meta for line in lines)
    assert all(r.null_test == meta for r in recs)
