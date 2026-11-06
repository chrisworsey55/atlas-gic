"""SQLite pick store and maturity-gated scoring.

One row per (agent, date, ticker) with excess-return columns for the 1, 5 and
10 trading-day horizons (NULL until matured). `conviction` is SIGNED (+ long,
- short); excess is the raw long-side excess return (see metrics docstring).
Rank IC uses (conviction, excess); hit rate MUST use `signed_excess`
(= +1 long / -1 short times excess, i.e. the net direction-adjusted return;
hit = signed_excess > 0). Cost is charged to the pick leg in both directions.
Picks with direction "none" and results with status != ok are never recorded.
A connection is opened per operation and closed after.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import date
from pathlib import Path

import pandas as pd

from atlas.config import AgentSpec
from atlas.schemas import AgentResult
from atlas.scoring.metrics import excess_return

HORIZONS = (1, 5, 10)
DEFAULT_BENCHMARK = "SPY"
COLUMNS = ["agent", "date", "ticker", "direction", "conviction", "horizon", "excess",
           "signed_excess", "scored"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS picks (
    agent TEXT NOT NULL, date TEXT NOT NULL, ticker TEXT NOT NULL,
    direction TEXT NOT NULL, conviction REAL NOT NULL,
    excess_1 REAL, excess_5 REAL, excess_10 REAL,
    PRIMARY KEY (agent, date, ticker)
)"""


class ScoreStore:
    def __init__(self, db_path: Path):
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.execute(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._path, timeout=30)
        try:
            yield conn
            conn.commit()
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()

    def discard_unscored(self, agent: str, day: date) -> None:
        """Delete (agent, day) pick rows that have no scored horizon (left by an earlier,
        uncommitted attempt at the day). Rows with any scored horizon are never touched."""
        with self._conn() as c:
            c.execute("DELETE FROM picks WHERE agent=? AND date=? AND excess_1 IS NULL "
                      "AND excess_5 IS NULL AND excess_10 IS NULL", (agent, day.isoformat()))

    def record_picks(self, result: AgentResult) -> None:
        """Replace the (agent, date) picks unless any is scored. A non-ok result records
        nothing and discards that agent's unscored rows for the date (a re-run day where
        the agent now fails must not keep the crashed attempt's picks)."""
        if result.status != "ok" or result.output is None:
            self.discard_unscored(result.agent, result.date)
            return
        rows = {}
        for p in result.output.picks:
            if p.direction == "none":
                continue
            sign = 1.0 if p.direction == "long" else -1.0
            rows[p.ticker] = (result.agent, result.date.isoformat(), p.ticker,
                              p.direction, sign * p.conviction)
        with self._conn() as c:  # one transaction per (agent, date)
            c.execute("BEGIN IMMEDIATE")
            scored = c.execute("SELECT 1 FROM picks WHERE agent=? AND date=? AND (excess_1 IS NOT NULL "
                               "OR excess_5 IS NOT NULL OR excess_10 IS NOT NULL) LIMIT 1",
                               (result.agent, result.date.isoformat())).fetchone()
            if scored:  # never rewrite scored history
                return
            c.execute("DELETE FROM picks WHERE agent=? AND date=?",
                      (result.agent, result.date.isoformat()))
            c.executemany("INSERT INTO picks "
                          "(agent, date, ticker, direction, conviction) VALUES (?,?,?,?,?)",
                          list(rows.values()))

    def score_matured(self, asof: date, store, specs: list[AgentSpec], cost_frac: float) -> int:
        bench = {s.name: (s.benchmark or DEFAULT_BENCHMARK) for s in specs}
        with self._conn() as c:
            pending = c.execute("SELECT agent, date, ticker, direction, excess_1, excess_5, excess_10 "
                                "FROM picks WHERE excess_1 IS NULL OR excess_5 IS NULL "
                                "OR excess_10 IS NULL").fetchall()
        scored = 0
        cal: dict[date, list[date]] = {}
        for agent, d, ticker, direction, *cur in pending:
            t = date.fromisoformat(d)
            if t > asof:
                continue
            if t not in cal:  # calendar capped at asof: no data after asof is touched
                cal[t] = store.trading_days(t, asof)
            days = cal[t]
            if not days or days[0] != t:
                continue
            b = bench.get(agent, DEFAULT_BENCHMARK)
            for h, existing in zip(HORIZONS, cur):
                if existing is not None or len(days) <= h:
                    continue
                window = days[: h + 1]
                tp = store.outcome_prices(ticker, window[0], window[-1]).reindex(window)
                bp = store.outcome_prices(b, window[0], window[-1]).reindex(window)
                ex = excess_return(tp, bp, h, cost_frac, 1 if direction == "long" else -1)
                if ex != ex:  # NaN: missing data, leave NULL
                    continue
                with self._conn() as c:
                    cur_ = c.execute(f"UPDATE picks SET excess_{h}=? WHERE agent=? AND date=? "
                                     f"AND ticker=? AND excess_{h} IS NULL", (ex, agent, d, ticker))
                    scored += cur_.rowcount
        return scored

    def picks_frame(self, agent: str | None, horizon: int) -> pd.DataFrame:
        if horizon not in HORIZONS:
            raise ValueError(f"horizon must be one of {HORIZONS}")
        q = f"SELECT agent, date, ticker, direction, conviction, excess_{horizon} FROM picks"
        args: tuple = ()
        if agent is not None:
            q += " WHERE agent=?"
            args = (agent,)
        q += " ORDER BY agent, date, ticker"
        with self._conn() as c:
            rows = c.execute(q, args).fetchall()
        df = pd.DataFrame(rows, columns=["agent", "date", "ticker", "direction", "conviction", "excess"])
        df["date"] = [date.fromisoformat(x) for x in df["date"]]
        df["excess"] = df["excess"].astype(float)  # NULL -> NaN
        df["horizon"] = horizon
        df["signed_excess"] = df["excess"] * df["direction"].map({"long": 1.0, "short": -1.0})
        df["scored"] = df["excess"].notna()
        return df[COLUMNS]


def non_overlapping(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Per agent keep rows on every horizon-th unique date so samples are >= horizon apart."""
    if horizon <= 1 or df.empty:
        return df.copy()
    parts = []
    for _, g in df.groupby("agent", sort=False):
        keep = sorted(g["date"].unique())[::horizon]
        parts.append(g[g["date"].isin(keep)])
    return pd.concat(parts).reset_index(drop=True)
