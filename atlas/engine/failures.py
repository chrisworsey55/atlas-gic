"""Persistent failure log: one row per failed LLM attempt, raw response verbatim."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from atlas.errorlog import ErrorLog, NullErrorLog
from atlas.schemas import FailureRecord

_COLS = ["agent", "date", "reason", "detail", "attempt", "raw_response"]


CALL_FAILURE = "call_failure"
DROPPED_PICK = "dropped_pick"


class FailureLog:
    """kind='call_failure' rows are failed LLM attempts; kind='dropped_pick' rows are
    out-of-universe picks removed from an otherwise valid output."""

    def __init__(self, db_path: Path, run_id: str | None = None, error_log: ErrorLog | None = None):
        self.db_path = Path(db_path)
        self.run_id = run_id
        self.error_log = error_log if error_log is not None else NullErrorLog()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._conn()) as c, c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS failures (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "agent TEXT, date TEXT, reason TEXT, detail TEXT, attempt INTEGER, raw_response TEXT, "
                "kind TEXT NOT NULL DEFAULT 'call_failure', run_id TEXT)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=30)

    def record(self, rec: FailureRecord, kind: str = CALL_FAILURE, run_id: str | None = None) -> None:
        with closing(self._conn()) as c, c:
            c.execute(
                "INSERT INTO failures (agent, date, reason, detail, attempt, raw_response, kind, run_id) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (rec.agent, rec.date.isoformat(), rec.reason.value, rec.detail, rec.attempt,
                 rec.raw_response, kind, run_id if run_id is not None else self.run_id),
            )
        # mirror after the insert; only the detail text, never the raw model reply
        self.error_log.event(kind, rec.detail, level="info" if kind == DROPPED_PICK else "warning",
                             day=rec.date, agent=rec.agent, reason=rec.reason.value,
                             attempt=rec.attempt)

    def rows(self, kind: str | None = None) -> pd.DataFrame:
        q = f"SELECT {', '.join(_COLS)}, kind, run_id FROM failures"
        args: tuple = ()
        if kind is not None:
            q += " WHERE kind=?"
            args = (kind,)
        with closing(self._conn()) as c:
            return pd.read_sql_query(q + " ORDER BY id", c, params=args)

    def _count(self, kind: str) -> pd.DataFrame:
        df = self.rows(kind)
        if df.empty:
            return pd.DataFrame(columns=["agent", "reason", "count"])
        return (
            df.groupby(["agent", "reason"]).size().reset_index(name="count")
            .sort_values(["agent", "reason"]).reset_index(drop=True)
        )

    def summary(self) -> pd.DataFrame:
        """Counts of failed calls only (dropped picks excluded)."""
        return self._count(CALL_FAILURE)

    def dropped_picks_summary(self) -> pd.DataFrame:
        return self._count(DROPPED_PICK)
