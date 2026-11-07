from datetime import date

import pytest

from atlas.backtest.predlog import PredictionLog
from atlas.schemas import AgentOutput, AgentResult, FailureReason, FailureRecord, Pick


def ok(agent="a1", day=date(2024, 1, 2), ticker="AAA"):
    out = AgentOutput(agent=agent, date=day, view="bullish", confidence=0.6, picks=[
        Pick(ticker=ticker, direction="long", conviction=0.5, horizon_days=5)])
    return AgentResult(agent=agent, date=day, output=out)


def no_op(agent="a2", day=date(2024, 1, 2)):
    return AgentResult(agent=agent, date=day, failure=FailureRecord(
        agent=agent, date=day, reason=FailureReason.TIMEOUT, detail="d", attempt=3, raw_response="SECRET"))


def _res(day, ticker="AAA"):
    return {"a1": ok("a1", day, ticker), "a2": no_op("a2", day)}


D1, D2, D3 = date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)


def make(tmp_path, days=(D1, D2, D3)):
    log = PredictionLog(tmp_path / "p.jsonl")
    hashes = [log.append(d, _res(d)) for d in days]
    return log, hashes


def test_log_hash_chain_verifies(tmp_path):
    log, hashes = make(tmp_path)
    assert log.verify() is True
    assert len(set(hashes)) == 3
    import json
    lines = [json.loads(x) for x in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert lines[1]["prev_hash"] == hashes[0] and lines[1]["hash"] == hashes[1]


def test_empty_or_missing_log_verifies(tmp_path):
    assert PredictionLog(tmp_path / "none.jsonl").verify() is True


def test_tampering_with_a_past_line_fails_verify(tmp_path):
    log, _ = make(tmp_path)
    p = tmp_path / "p.jsonl"
    p.write_text(p.read_text().replace("AAA", "BBB", 1))
    assert log.verify() is False


def test_deleted_middle_line_fails_verify(tmp_path):
    log, _ = make(tmp_path)
    p = tmp_path / "p.jsonl"
    lines = p.read_text().splitlines(keepends=True)
    p.write_text(lines[0] + lines[2])
    assert log.verify() is False


def test_reordered_and_malformed_fail_verify(tmp_path):
    log, _ = make(tmp_path)
    p = tmp_path / "p.jsonl"
    lines = p.read_text().splitlines(keepends=True)
    p.write_text(lines[1] + lines[0] + lines[2])
    assert log.verify() is False
    p.write_text("".join(lines) + "not json\n")
    assert log.verify() is False


def test_log_append_only_never_rewrites(tmp_path):
    p = tmp_path / "p.jsonl"
    log = PredictionLog(p)
    log.append(D1, _res(D1))
    before = p.read_bytes()
    log.append(D2, _res(D2))
    assert p.read_bytes().startswith(before)
    # reopening continues the chain
    log2 = PredictionLog(p)
    log2.append(D3, _res(D3))
    assert p.read_bytes().startswith(before)
    assert log2.verify() is True


def test_no_opinion_recorded_with_reason_and_no_partial_output(tmp_path):
    import json
    log = PredictionLog(tmp_path / "p.jsonl")
    log.append(D1, _res(D1))
    line = json.loads((tmp_path / "p.jsonl").read_text())
    rec = next(a for a in line["agents"] if a["agent"] == "a2")
    assert rec == {"agent": "a2", "status": "no_opinion", "reason": "timeout"}
    assert "SECRET" not in (tmp_path / "p.jsonl").read_text()


def test_append_refuses_to_extend_tampered_log(tmp_path):
    log, _ = make(tmp_path, (D1, D2))
    p = tmp_path / "p.jsonl"
    p.write_text(p.read_text().replace("AAA", "BBB", 1))
    with pytest.raises(ValueError):
        PredictionLog(p).append(D3, _res(D3))


def test_idempotent_reappend_same_day(tmp_path):
    log, hashes = make(tmp_path, (D1, D2))
    p = tmp_path / "p.jsonl"
    before = p.read_bytes()
    assert log.append(D2, _res(D2)) == hashes[1]
    assert PredictionLog(p).append(D2, _res(D2)) == hashes[1]
    assert p.read_bytes() == before


def test_different_content_for_existing_day_raises(tmp_path):
    log, _ = make(tmp_path, (D1, D2))
    with pytest.raises(ValueError):
        log.append(D2, _res(D2, ticker="ZZZ"))


def test_older_day_cannot_be_reappended(tmp_path):
    log, _ = make(tmp_path, (D1, D2))
    with pytest.raises(ValueError):
        log.append(D1, _res(D1))


def test_head_empty_and_nonempty(tmp_path):
    assert PredictionLog(tmp_path / "x.jsonl").head() == (0, None)
    log, hashes = make(tmp_path)
    assert log.head() == (3, hashes[-1])


def test_tail_deletion_passes_verify_but_caught_by_contains_head(tmp_path):
    log, _ = make(tmp_path)
    count, h = log.head()
    p = tmp_path / "p.jsonl"
    lines = p.read_text().splitlines(keepends=True)
    p.write_text("".join(lines[:2]))
    assert log.verify() is True  # documented limitation
    assert log.contains_head(count, h) is False
    # re-appending different picks for the deleted day is still caught
    log.append(D3, _res(D3, ticker="ZZZ"))
    assert log.verify() is True
    assert log.contains_head(count, h) is False


def test_contains_head_accepts_exact_and_one_ahead(tmp_path):
    log, hashes = make(tmp_path, (D1, D2))
    count, h = log.head()
    assert log.contains_head(count, h) is True
    log.append(D3, _res(D3))
    assert log.contains_head(count, h) is True
    assert log.contains_head(count, h, max_ahead=0) is False
    assert log.contains_head(0, None) is False  # 3 lines ahead of empty


def test_contains_head_empty_log(tmp_path):
    log = PredictionLog(tmp_path / "x.jsonl")
    assert log.contains_head(0, None) is True
    assert log.contains_head(1, "abc") is False


def test_contains_head_rejects_replaced_history(tmp_path):
    log, _ = make(tmp_path, (D1, D2))
    count, h = log.head()
    other = PredictionLog(tmp_path / "q.jsonl")
    other.append(D1, _res(D1, ticker="ZZZ"))
    other.append(D2, _res(D2, ticker="ZZZ"))
    assert other.verify() is True
    assert other.contains_head(count, h) is False


def test_torn_last_line_fails_verify_and_append_raises(tmp_path):
    log, _ = make(tmp_path, (D1, D2))
    p = tmp_path / "p.jsonl"
    p.write_bytes(p.read_bytes()[:-1])
    assert log.verify() is False
    with pytest.raises(ValueError):
        log.append(D3, _res(D3))


def test_drop_uncommitted_tail_removes_only_the_single_uncheckpointed_line(tmp_path):
    log, hashes = make(tmp_path)
    before = log.path.read_text().splitlines(keepends=True)
    log.drop_uncommitted_tail(D3, 2, hashes[1])
    assert log.path.read_text() == "".join(before[:2])
    assert log.verify() and log.head() == (2, hashes[1])
    assert not (tmp_path / "p.jsonl.tmp").exists()
    # re-append different content for the dropped day is now allowed
    log.append(D3, _res(D3, "BBB"))
    assert log.head()[0] == 3


@pytest.mark.parametrize("args", [
    (D3, 1, None),          # log is two lines ahead
    (D3, 3, "x"),           # log is not ahead at all
    (D2, 2, None),          # wrong expected day   (hash replaced below)
    (D3, 2, "0" * 64),      # committed prefix does not match the checkpoint
])
def test_drop_uncommitted_tail_refuses_anything_else(tmp_path, args):
    log, hashes = make(tmp_path)
    day, count, h = args
    if day == D2:
        h = hashes[1]
    if count == 1:
        h = hashes[0]
    raw = log.path.read_bytes()
    with pytest.raises(ValueError):
        log.drop_uncommitted_tail(day, count, h)
    assert log.path.read_bytes() == raw


def test_drop_uncommitted_tail_from_empty_checkpoint(tmp_path):
    log, _ = make(tmp_path, days=(D1,))
    log.drop_uncommitted_tail(D1, 0, None)
    assert log.path.read_text() == "" and log.head() == (0, None)
