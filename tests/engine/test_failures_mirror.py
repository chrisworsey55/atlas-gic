from datetime import date

from atlas.engine.failures import DROPPED_PICK, FailureLog
from atlas.errorlog import ErrorLog, read_events
from atlas.schemas import FailureReason, FailureRecord

CANARY = "RAW-MODEL-REPLY-CANARY"


def _rec(reason=FailureReason.MALFORMED_JSON):
    return FailureRecord(agent="m1", date=date(2024, 1, 3), reason=reason,
                         detail="bad json at char 4", attempt=2, raw_response=CANARY)


def test_failure_record_is_mirrored_without_raw_response(tmp_path):
    path = tmp_path / "errors.log"
    fl = FailureLog(tmp_path / "f.db", error_log=ErrorLog(path))
    fl.record(_rec())
    (ev,) = read_events(path, kind="call_failure")
    assert ev["level"] == "warning" and ev["agent"] == "m1" and ev["day"] == "2024-01-03"
    assert ev["reason"] == "malformed_json" and ev["message"] == "bad json at char 4"
    assert ev["fields"]["attempt"] == 2
    assert CANARY not in path.read_text()
    assert fl.rows().raw_response.tolist() == [CANARY]  # still verbatim in the DB


def test_dropped_pick_mirrored_as_info(tmp_path):
    path = tmp_path / "errors.log"
    fl = FailureLog(tmp_path / "f.db", error_log=ErrorLog(path))
    fl.record(_rec(FailureReason.SCHEMA_INVALID), kind=DROPPED_PICK)
    (ev,) = read_events(path, kind="dropped_pick")
    assert ev["level"] == "info" and ev["reason"] == "schema_invalid"
    assert CANARY not in path.read_text()


def test_no_error_log_is_unchanged(tmp_path):
    fl = FailureLog(tmp_path / "f.db")
    fl.record(_rec())
    assert len(fl.rows()) == 1
