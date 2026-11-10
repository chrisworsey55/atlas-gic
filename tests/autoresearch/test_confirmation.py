"""Task 7: an edit that wins on train is kept only if it is confirmed on validation days."""
import copy
import json
import math
from datetime import date

import pytest

import atlas.autoresearch.loop as loop
from atlas.autoresearch.gitops import GitOps
from atlas.autoresearch.loop import (
    DayPairs,
    HoldoutViolation,
    check_train_only,
    check_validation_only,
    confirm_margin,
    edit_gate,
    paired_replay,
    replay_days,
    validation_days,
)
from atlas.backtest.runner import Periods
from atlas.engine.llm import BudgetExceeded
from tests.autoresearch.conftest import PROMPT_A, git
from tests.autoresearch.test_loop import (
    NEW_SUFFIX,
    PERIODS,
    TICKERS,
    TRADING_DAYS,
    _u,
    branches,
    excess,
    log_lines,
    make_llm,
    proposal_stub,
    run,
    scores,
    settings,
)

VAL = PERIODS.validation
VAL_TRADING = [d for d in TRADING_DAYS if VAL[0] <= d <= VAL[1]]
LAST_VAL_DAY = VAL_TRADING[-1]
CONFIRM_KEYS = {"n_days_requested", "n_days", "old_ic", "new_ic", "diff", "margin", "new_pairs",
                "passed", "reason"}


def in_val(d) -> bool:
    return VAL[0] <= d <= VAL[1]


def split_run_one_day(train_skill=1.0, val_skill=1.0, seen=None, fail=None):
    """The new prompt's convictions track excess with `train_skill` on train days and
    `val_skill` on validation days; the old prompt is noise."""
    def make(agent):
        def run_one(day, prompt):
            if seen is not None:
                seen.append((agent, day, prompt))
            if fail is not None:
                out = fail(day, prompt)
                if out is not None:
                    return out
            new = prompt.endswith(NEW_SUFFIX)
            skill = (val_skill if in_val(day) else train_skill) if new else 0.0
            conv = [skill * excess(day, t) / 0.02 + (1 - skill) * _u("cv", day, t) for t in TICKERS]
            return DayPairs(list(TICKERS), conv, [excess(day, t) for t in TICKERS])
        run_one.horizon = 1
        return run_one
    return make


def val_seen(seen):
    return sorted({d for _, d, _ in seen if in_val(d)})


def unchanged(repo, base):
    assert git(repo, "rev-parse", "main") == base
    assert (repo / "atlas/prompts/agent_a.md").read_bytes() == PROMPT_A.encode()
    assert git(repo, "branch", "--show-current") == "main"
    assert GitOps(repo).is_clean()
    assert branches(repo) == ["main"]


# ---------------------------------------------------------------- keep / unconfirmed

def test_train_win_and_validation_win_is_kept_and_merged(tmp_path, prompt_repo):
    seen = []
    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=split_run_one_day(seen=seen), keep_margin=0.1)
    r = recs[0]
    assert r.decision == "kept" and r.confirm["passed"] is True
    assert r.confirm["n_days"] == 60 and r.confirm["n_days_requested"] == 60
    assert r.confirm["diff"] > r.confirm["margin"] >= 0.1
    assert "validation" in r.reason
    head = git(prompt_repo, "rev-parse", "main")
    assert head != base and r.commit_sha == head
    assert (prompt_repo / "atlas/prompts/agent_a.md").read_text() == PROMPT_A + NEW_SUFFIX
    assert branches(prompt_repo) == ["main"]
    # the train numbers stay the train replay's; validation days were replayed under both prompts
    assert r.n_days == 60
    vs = val_seen(seen)
    assert len(vs) == 60
    for d in vs:
        assert {(p.endswith(NEW_SUFFIX)) for a, dd, p in seen if dd == d} == {True, False}


def test_train_win_validation_loss_is_unconfirmed_and_branch_discarded(tmp_path, prompt_repo):
    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=split_run_one_day(train_skill=1.0, val_skill=0.0),
               keep_margin=0.1)
    r = recs[0]
    assert r.decision == "unconfirmed" and r.commit_sha is None
    assert r.diff > 0.5  # it did win on train
    assert r.confirm["passed"] is False and r.confirm["n_days"] == 60
    assert r.confirm["diff"] == pytest.approx(0.0, abs=1e-12)
    assert "<=" in r.confirm["reason"]
    unchanged(prompt_repo, base)


def test_validation_too_short_is_unconfirmed_not_kept(tmp_path, prompt_repo):
    # trading days end after the 20th validation day: 19 validation days have an in-validation exit
    short = [d for d in TRADING_DAYS if d <= VAL_TRADING[19]]
    assert len(validation_days(short, PERIODS, 1, 60)) == 19
    seen = []
    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=split_run_one_day(seen=seen), keep_margin=0.1, trading_days=short)
    r = recs[0]
    assert r.decision == "unconfirmed" and r.commit_sha is None
    assert r.confirm["passed"] is False and r.confirm["n_days_requested"] == 0
    assert "19" in r.confirm["reason"] and "confirm_min_days" in r.confirm["reason"]
    assert val_seen(seen) == []  # nothing replayed: validation data is not spent
    unchanged(prompt_repo, base)
    # one more day reaches confirm_min_days (20): the confirmation runs and keeps the edit
    tmp2 = tmp_path / "b"
    tmp2.mkdir()
    longer = [d for d in TRADING_DAYS if d <= VAL_TRADING[20]]
    recs = run(tmp2, prompt_repo, make=split_run_one_day(), keep_margin=0.1, trading_days=longer)
    assert recs[0].decision == "kept" and recs[0].confirm["n_days"] == 20


def test_validation_replay_with_too_few_comparable_days_is_unconfirmed(tmp_path, prompt_repo):
    # the new prompt fails on most validation days: fewer than confirm_min_days comparable days
    def fail(day, prompt):
        if in_val(day) and prompt.endswith(NEW_SUFFIX) and day.day % 4:
            return "none"

    def make(agent):
        inner = split_run_one_day()(agent)

        def run_one(day, prompt):
            if fail(day, prompt):
                return None
            return inner(day, prompt)
        run_one.horizon = 1
        return run_one

    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=make, keep_margin=0.1)
    assert recs[0].decision == "unconfirmed" and "replay" in recs[0].confirm["reason"]
    assert recs[0].confirm["n_days_requested"] == 60
    unchanged(prompt_repo, base)


def test_confirmation_disabled_keeps_old_behaviour(tmp_path, prompt_repo):
    seen = []
    recs = run(tmp_path, prompt_repo, make=split_run_one_day(val_skill=0.0, seen=seen), keep_margin=0.1,
               cfg=settings(confirm_on_validation=False))
    assert recs[0].decision == "kept" and recs[0].confirm is None
    assert val_seen(seen) == []
    assert (prompt_repo / "atlas/prompts/agent_a.md").read_text() == PROMPT_A + NEW_SUFFIX
    assert log_lines(tmp_path)[0]["confirm"] is None


def test_no_confirmation_without_a_train_win(tmp_path, prompt_repo):
    for margin in (None, 10.0):  # uncalibrated, reverted
        seen = []
        d = tmp_path / str(margin)
        d.mkdir()
        recs = run(d, prompt_repo, make=split_run_one_day(seen=seen), keep_margin=margin)
        assert recs[0].decision in ("uncalibrated", "reverted") and recs[0].confirm is None
        assert val_seen(seen) == []


def test_selective_abstention_learned_on_train_fails_validation(tmp_path, prompt_repo):
    """The new prompt 'learned' which train-period picks lose: on train days it drops the day's
    worst pick and raises the conviction of the rest (credited even with imputation). On
    validation days it has no such knowledge: it drops an arbitrary pick and boosts the rest."""
    tickers = [f"S{i}" for i in range(10)]

    def ex(day, t):
        return _u("sx", day, t) * 0.02

    def make(agent):
        def run_one(day, prompt):
            conv = [_u("sc", day, t) for t in tickers]
            exs = [ex(day, t) for t in tickers]
            if not prompt.endswith(NEW_SUFFIX):
                return DayPairs(list(tickers), conv, exs)
            if in_val(day):
                drop = min(tickers, key=lambda t: _u("drop", day, t))
            else:
                drop = min(tickers, key=lambda t: ex(day, t))
            keep = [i for i, t in enumerate(tickers) if t != drop]
            return DayPairs([tickers[i] for i in keep], [conv[i] + 2.0 for i in keep], [exs[i] for i in keep])
        run_one.horizon = 1
        return run_one

    cfg = settings(min_pair_ratio=0.85)
    tdays = replay_days(TRADING_DAYS, PERIODS, 1, 60)
    train = paired_replay("agent_a", PROMPT_A, PROMPT_A + NEW_SUFFIX, tdays, make("agent_a"), impute=True)
    assert edit_gate(train, cfg) is None and train.diff > 0.1  # it wins on train
    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=make, keep_margin=0.1, cfg=cfg)
    r = recs[0]
    assert r.diff == pytest.approx(train.diff)
    assert r.decision == "unconfirmed" and r.confirm["passed"] is False
    assert abs(r.confirm["diff"]) < 0.1
    unchanged(prompt_repo, base)


# ---------------------------------------------------------------- holdout safety

def test_last_validation_day_with_holdout_exit_is_excluded(tmp_path, prompt_repo):
    vd = validation_days(TRADING_DAYS, PERIODS, horizon=1, n=60)
    assert LAST_VAL_DAY not in vd and vd[-1] == VAL_TRADING[-2]
    assert vd == VAL_TRADING[:-1][-60:]
    check_validation_only(vd, TRADING_DAYS, PERIODS, horizon=1)
    # longer horizon: the last h validation days are dropped
    assert validation_days(TRADING_DAYS, PERIODS, horizon=3, n=100) == VAL_TRADING[:-3]
    seen = []
    run(tmp_path, prompt_repo, make=split_run_one_day(seen=seen), keep_margin=0.1)
    vs = val_seen(seen)
    assert vs == vd and LAST_VAL_DAY not in vs
    assert all(d < PERIODS.holdout[0] for _, d, _ in seen)
    for d in vs:
        assert TRADING_DAYS[TRADING_DAYS.index(d) + 1] <= VAL[1]


def test_holdout_day_in_confirmation_raises_holdout_violation(tmp_path, prompt_repo, monkeypatch):
    vd = validation_days(TRADING_DAYS, PERIODS, horizon=1, n=60)
    for bad in ([PERIODS.holdout[0]], [LAST_VAL_DAY], [TRADING_DAYS[-1]], [date(2024, 7, 6)]):
        with pytest.raises(HoldoutViolation):
            check_validation_only(vd + bad, TRADING_DAYS, PERIODS, horizon=1)
    # the guards are separate: train days never pass the validation check and vice versa
    with pytest.raises(HoldoutViolation):
        check_validation_only(replay_days(TRADING_DAYS, PERIODS, 1, 60), TRADING_DAYS, PERIODS, 1)
    with pytest.raises(HoldoutViolation):
        check_train_only(vd, TRADING_DAYS, PERIODS, 1)
    with pytest.raises(ValueError):  # periods cannot overlap
        Periods(train=(date(2024, 1, 1), date(2024, 7, 15)), validation=VAL, holdout=PERIODS.holdout)
    # a validation-day selection that includes holdout days is refused before any edit
    calls = []
    llm = make_llm(tmp_path, lambda s, u: calls.append(1) or proposal_stub()(s, u))
    monkeypatch.setattr(loop, "validation_days", lambda *a, **k: vd[-30:] + [PERIODS.holdout[0]])
    base = git(prompt_repo, "rev-parse", "main")
    with pytest.raises(HoldoutViolation):
        run(tmp_path, prompt_repo, llm=llm, make=split_run_one_day(), keep_margin=0.1)
    assert calls == []
    unchanged(prompt_repo, base)


def test_confirmation_replay_guard_rejects_non_validation_days(tmp_path, prompt_repo, monkeypatch):
    """If the confirmation replay ever asked for a day outside the checked validation days (here:
    a train day), the guard raises; the attempt is recorded and nothing is merged."""
    vd = validation_days(TRADING_DAYS, PERIODS, horizon=1, n=60)
    real = loop.paired_replay

    def sneaky(agent, old, new, days, run_one_day, **kw):
        if days and in_val(days[0]):
            run_one_day(replay_days(TRADING_DAYS, PERIODS, 1, 60)[0], new)
        return real(agent, old, new, days, run_one_day, **kw)

    monkeypatch.setattr(loop, "paired_replay", sneaky)
    base = git(prompt_repo, "rev-parse", "main")
    with pytest.raises(HoldoutViolation):
        run(tmp_path, prompt_repo, make=split_run_one_day(), keep_margin=0.1)
    lines = log_lines(tmp_path)
    assert [x["decision"] for x in lines] == ["error"] and "HoldoutViolation" in lines[0]["reason"]
    assert vd  # sanity
    unchanged(prompt_repo, base)


# ---------------------------------------------------------------- margin

def test_confirm_margin_grows_with_fewer_pairs_and_never_below_keep_margin():
    assert confirm_margin(0.1, 60, 60) == pytest.approx(0.1)
    assert confirm_margin(0.1, 60, 600) == pytest.approx(0.1)  # never below keep_margin
    assert confirm_margin(0.1, 60, 15) == pytest.approx(0.2)   # sqrt(60/15) = 2
    assert confirm_margin(0.1, 60, 30) == pytest.approx(0.1 * math.sqrt(2))
    assert confirm_margin(0.1, 60, 0) == pytest.approx(0.1 * math.sqrt(60))  # no pairs: max(1, n)
    assert confirm_margin(0.0, 60, 10) == 0.0
    ms = [confirm_margin(0.05, 100, n) for n in (5, 10, 25, 50, 100, 200)]
    assert ms == sorted(ms, reverse=True) and min(ms) == pytest.approx(0.05)


def test_validation_gate_uses_confirm_min_days():
    cfg = settings(rolling_ic_window=60)
    from atlas.autoresearch.loop import PairedResult
    r = PairedResult(0.0, 0.3, 25, 0.3, old_pairs=125, new_pairs=125, n_days_requested=25)
    assert edit_gate(r, cfg) is not None  # 25 < 40 train days
    assert edit_gate(r, cfg, min_days=20) is None
    assert edit_gate(r, cfg, min_days=30) is not None


# ---------------------------------------------------------------- logging, caps, leakage

def test_confirmation_numbers_are_logged_in_the_jsonl_record(tmp_path, prompt_repo):
    cfg = settings()
    recs = run(tmp_path, prompt_repo, make=split_run_one_day(val_skill=0.3), keep_margin=0.1, cfg=cfg)
    line = log_lines(tmp_path)[0]
    assert set(line["confirm"]) == CONFIRM_KEYS
    vd = validation_days(TRADING_DAYS, PERIODS, 1, 60)
    val = paired_replay("agent_a", PROMPT_A, PROMPT_A + NEW_SUFFIX, vd, split_run_one_day(val_skill=0.3)("agent_a"),
                        impute=True, min_days=20)
    c = line["confirm"]
    assert c["n_days"] == val.n_days == 60 and c["new_pairs"] == val.new_pairs
    assert c["old_ic"] == pytest.approx(val.old_ic) and c["new_ic"] == pytest.approx(val.new_ic)
    assert c["diff"] == pytest.approx(val.diff)
    assert c["margin"] == pytest.approx(confirm_margin(0.1, cfg.rolling_ic_window, val.new_pairs))
    assert c["passed"] is (val.diff > c["margin"]) and line["decision"] == ("kept" if c["passed"] else "unconfirmed")
    assert recs[0].confirm == c
    # train numbers are still the top-level fields
    assert line["n_days"] == 60 and line["diff"] == pytest.approx(recs[0].diff)


def test_non_finite_validation_ic_is_unconfirmed_and_logged_as_null(tmp_path, prompt_repo):
    def fail(day, prompt):  # constant convictions on validation under both prompts: IC is NaN
        if in_val(day):
            return DayPairs(list(TICKERS), [0.5] * len(TICKERS), [excess(day, t) for t in TICKERS])

    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=split_run_one_day(fail=fail), keep_margin=0.1)
    assert recs[0].decision == "unconfirmed"
    raw = (tmp_path / "run" / "autoresearch_log.jsonl").read_text()
    assert "NaN" not in raw
    c = json.loads(raw)["confirm"]
    assert c["diff"] is None and c["passed"] is False
    unchanged(prompt_repo, base)


def test_every_attempt_is_logged_and_max_edits_caps_confirmations(tmp_path, prompt_repo):
    for max_edits in (2, 3):
        seen = []
        d = tmp_path / str(max_edits)
        d.mkdir()
        recs = run(d, prompt_repo, make=split_run_one_day(val_skill=0.0, seen=seen), keep_margin=0.1,
                   cfg=settings(max_edits=max_edits))
        lines = log_lines(d)
        assert len(recs) == len(lines) == max_edits
        assert [x["decision"] for x in lines] == ["unconfirmed"] * max_edits
        assert all(x["confirm"]["n_days_requested"] == 60 for x in lines)
        # each confirmation replays 60 validation days under both prompts, and there are no more
        # confirmations than attempts
        assert sum(1 for _, day, _ in seen if in_val(day)) == max_edits * 2 * 60
        assert branches(prompt_repo) == ["main"]


def test_confirmation_result_never_reaches_proposal_diagnostics_or_target_selection(
        tmp_path, prompt_repo, monkeypatch):
    sel, diag, users = [], [], []
    real_sel, real_diag = loop.select_target, loop.build_diagnostics

    def spy_sel(stats, rates, *a, **k):
        sel.append((copy.deepcopy(stats), copy.deepcopy(rates), a, k))
        return real_sel(stats, rates, *a, **k)

    def spy_diag(scores, agent, stats, *a, **k):
        diag.append((agent, copy.deepcopy(stats), a, k, scores.df.copy()))
        return real_diag(scores, agent, stats, *a, **k)

    real_prop = loop.propose_edit

    def spy_prop(llm, spec, current, diagnostics):  # what the proposal model is shown
        users.append(current + "\n" + diagnostics)
        return real_prop(llm, spec, current, diagnostics)

    monkeypatch.setattr(loop, "select_target", spy_sel)
    monkeypatch.setattr(loop, "build_diagnostics", spy_diag)
    monkeypatch.setattr(loop, "propose_edit", spy_prop)
    # attempt 1 (agent_a) wins on train, loses on validation; attempt 3 targets agent_a again
    # the store also holds validation-period picks (VALT): they must never be used
    store = scores(val={"agent_a": 1.0, "agent_b": -1.0})
    recs = run(tmp_path, prompt_repo, keep_margin=0.1, cfg=settings(max_edits=3), store=store,
               make=split_run_one_day(val_skill=-1.0))
    assert [r.agent for r in recs] == ["agent_a", "agent_b", "agent_a"]
    assert all(r.decision == "unconfirmed" and r.confirm["diff"] < 0 for r in recs)
    # target selection saw the same train-only stats every time
    assert sel[0][0] == sel[1][0] == sel[2][0] and sel[0][1] == sel[2][1]
    # diagnostics for agent_a: identical inputs and identical proposal text before and after
    a = [x for x in diag if x[0] == "agent_a"]
    assert len(a) == 2 and a[0][1] == a[1][1] and a[0][2] == a[1][2] and a[0][3] == a[1][3]
    assert a[0][4].equals(a[1][4])
    assert all(max_date <= PERIODS.train[1] for _, _, (_, max_date, *_), _, _ in diag)
    assert len(users) == 3 and users[0] == users[2]
    for u in users:
        assert "validation" not in u.lower() and "confirm" not in u.lower()
        assert f"{recs[0].confirm['diff']:.4f}" not in u and "VALT" not in u


# ---------------------------------------------------------------- failures during confirmation

def test_budget_exceeded_during_confirmation_stops_and_discards(tmp_path, prompt_repo):
    def fail(day, prompt):
        if in_val(day) and prompt.endswith(NEW_SUFFIX):
            raise BudgetExceeded("over budget")

    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=split_run_one_day(fail=fail), keep_margin=0.1,
               cfg=settings(max_edits=3))
    assert [r.decision for r in recs] == ["error"]
    assert "budget" in recs[0].reason and "did not complete: budget" in recs[0].confirm["reason"]
    assert recs[0].confirm["passed"] is False
    unchanged(prompt_repo, base)


def test_crash_during_confirmation_is_recorded_cleaned_up_and_reraised(tmp_path, prompt_repo):
    def fail(day, prompt):
        if in_val(day) and prompt.endswith(NEW_SUFFIX):
            raise KeyboardInterrupt

    base = git(prompt_repo, "rev-parse", "main")
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, prompt_repo, make=split_run_one_day(fail=fail), keep_margin=0.1)
    lines = log_lines(tmp_path)
    assert [x["decision"] for x in lines] == ["error"] and lines[0]["confirm"]["passed"] is False
    unchanged(prompt_repo, base)


def test_malformed_new_output_on_validation_rejects(tmp_path, prompt_repo):
    def fail(day, prompt):
        if in_val(day) and prompt.endswith(NEW_SUFFIX):
            return DayPairs(["A", "A"], [0.1, 0.2], [0.1, 0.2])

    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=split_run_one_day(fail=fail), keep_margin=0.1)
    assert recs[0].decision == "rejected" and "malformed" in recs[0].reason
    assert recs[0].confirm["passed"] is False
    unchanged(prompt_repo, base)


def test_edit_gate_applies_to_the_validation_result(tmp_path, prompt_repo):
    # on validation the new prompt wins on the days it answers but fails on a third of them:
    # survivorship on validation is rejected by the gate even though its diff clears the margin
    def fail(day, prompt):
        if in_val(day) and prompt.endswith(NEW_SUFFIX) and day.day % 3 == 0:
            return "fail"

    def make(agent):
        inner = split_run_one_day()(agent)

        def run_one(day, prompt):
            return None if fail(day, prompt) else inner(day, prompt)
        run_one.horizon = 1
        return run_one

    base = git(prompt_repo, "rev-parse", "main")
    recs = run(tmp_path, prompt_repo, make=make, keep_margin=0.1)
    c = recs[0].confirm
    assert recs[0].decision == "unconfirmed" and c["passed"] is False
    assert c["diff"] > c["margin"] and c["n_days"] >= 20
    assert c["reason"].startswith("gate:") and "failed on" in c["reason"]
    unchanged(prompt_repo, base)
