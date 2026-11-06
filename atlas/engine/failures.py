"""Persistent failure log: one row per failed LLM attempt, raw response verbatim."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from atlas.schemas import FailureRecord

_COLS = ["agent", "date", "reason", "detail", "attempt", "raw_response"]


class FailureLog:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(
                "CREATE TABLE IF NOT EXISTS failures (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                "agent TEXT, date TEXT, reason TEXT, detail TEXT, attempt INTEGER, raw_response TEXT)"
            )

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def record(self, rec: FailureRecord) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO failures (agent, date, reason, detail, attempt, raw_response) VALUES (?,?,?,?,?,?)",
                (rec.agent, rec.date.isoformat(), rec.reason.value, rec.detail, rec.attempt, rec.raw_response),
            )

    def rows(self) -> pd.DataFrame:
        with self._conn() as c:
            return pd.read_sql_query(f"SELECT {', '.join(_COLS)} FROM failures ORDER BY id", c)

    def summary(self) -> pd.DataFrame:
        df = self.rows()
        if df.empty:
            return pd.DataFrame(columns=["agent", "reason", "count"])
        return (
            df.groupby(["agent", "reason"]).size().reset_index(name="count")
            .sort_values(["agent", "reason"]).reset_index(drop=True)
        )
