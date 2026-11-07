"""Contamination probe (optional validity control): can the model recognise the data?

Each probe shows the model ONE anonymised snapshot (opaque instrument id, prices rebased to
100, relative day index, sector label; no calendar date, no real ticker) and asks it to guess
the calendar date and the real ticker symbol. The true date and ticker are used only locally,
to score the reply. A high hit rate means the model can identify these periods from price
shape alone, so results for periods like them deserve less trust. It is NOT a measure of skill
and a low rate is not proof of an uncontaminated model: it is a lower bound on recognisability.

Scoring (per probe, all local):

* date hit   = the guessed date is within ``tolerance_days`` (inclusive) of the true date;
* ticker hit = the guessed symbol equals the true ticker (case and surrounding spaces ignored);
* probe hit  = date hit OR ticker hit (counted once);
* ``rate``   = probe hits / n, where n is the number of COMPLETED probes (a malformed, refused
  or unreachable reply is a completed miss; a probe never started because the budget ran out
  is not counted). n = 0 gives rate 0.0 and ``empty`` True instead of NaN.

Each snapshot should hold the single instrument under test; if it holds several, the guess is
asked for the first listed (lowest opaque id).
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

from atlas.data.store import Snapshot
from atlas.engine.agent import BACKOFF_BASE_S, TRANSIENT_MAX_ATTEMPTS
from atlas.engine.failures import FailureLog
from atlas.engine.llm import ApiError, BudgetExceeded, CachedLLM, LLMResponse, RateLimited, Timeout
from atlas.schemas import FailureReason, FailureRecord

PROBE_AGENT = "contamination_probe"
MAX_TOKENS = 200

PROBE_SYSTEM = """# contamination_probe
You are shown anonymised market data for one instrument: an opaque id, a sector label, daily
bars [relative_day, open, high, low, close] with prices rebased to 100 at the first bar (the
last bar is relative day 0, earlier bars are negative), and optional macro series
[calendar_day_offset, value] with offsets counted back from the last bar.

Using only what you recognise about this data, guess (a) the calendar date of the last bar and
(b) the real ticker symbol of the instrument. Always give your best guess.

Reply with ONLY a JSON object, no other text:
{"date": "YYYY-MM-DD", "ticker": "SYMBOL"}
"""

_FENCE = re.compile(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```$", re.DOTALL)


@dataclass(frozen=True)
class ContaminationScore:
    n: int
    date_hits: int
    ticker_hits: int
    rate: float
    any_hits: int = 0          # probes with a date hit OR a ticker hit (rate = any_hits / n)
    failed: int = 0            # completed probes whose reply was unusable (counted as misses)
    budget_exhausted: bool = False  # stopped early by the hard budget cap

    @property
    def empty(self) -> bool:
        return self.n == 0


def _user_message(snap: Snapshot) -> str:
    p = snap.prices
    insts = []
    for tid, g in p.groupby("ticker", sort=True):
        g = g.sort_values("date")
        bars = [[int(r.date), round(float(r.open), 4), round(float(r.high), 4), round(float(r.low), 4),
                 round(float(r.close), 4)] for r in g.itertuples()]
        sector = str(g["sector"].iloc[0]) if "sector" in g.columns else ""
        insts.append({"id": str(tid), "sector": sector, "bars": bars})
    macro = {f"M{i + 1}": [[int(k), round(float(v), 4)] for k, v in s.items()]
             for i, (_, s) in enumerate(sorted(snap.macro.items()))}  # real series ids are not sent
    return json.dumps({"instruments": insts, "macro": macro}, sort_keys=True)


class _Bad(Exception):
    def __init__(self, reason: FailureReason, detail: str):
        self.reason, self.detail = reason, detail


def _parse(resp: LLMResponse) -> tuple[date, str]:
    if resp.stop_reason == "refusal":
        raise _Bad(FailureReason.REFUSAL, "model refused")
    if resp.stop_reason == "max_tokens":
        raise _Bad(FailureReason.TRUNCATED, "stop_reason=max_tokens")
    text = resp.text.strip()
    if not text:
        raise _Bad(FailureReason.EMPTY_RESPONSE, "empty response")
    m = _FENCE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise _Bad(FailureReason.MALFORMED_JSON, f"{e.msg} at line {e.lineno} col {e.colno}") from None
    if not isinstance(data, dict):
        raise _Bad(FailureReason.SCHEMA_INVALID, "top-level JSON is not an object")
    d, t = data.get("date"), data.get("ticker")
    try:
        if not isinstance(d, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", d):
            raise ValueError
        guess = date.fromisoformat(d)
    except ValueError:
        raise _Bad(FailureReason.SCHEMA_INVALID, "invalid fields: date") from None
    if not isinstance(t, str) or not t.strip():
        raise _Bad(FailureReason.SCHEMA_INVALID, "invalid fields: ticker")
    return guess, t.strip()


def contamination_probe(snapshots: list[tuple[Snapshot, date, str]], llm: CachedLLM,
                        tolerance_days: int, *, failures: FailureLog | None = None,
                        sleep: Callable[[float], None] = time.sleep) -> ContaminationScore:
    """`snapshots` are (ANONYMISED snapshot, true date, true real ticker). Only the snapshot
    is sent to the model; the date and ticker are used to score the reply. Temperature 0
    (client) and caching (CachedLLM) as everywhere else; spend counts against `llm`'s budget
    and BudgetExceeded stops the probe cleanly with the score so far."""
    n = date_hits = ticker_hits = any_hits = failed = 0
    exhausted = False
    for snap, true_date, true_ticker in snapshots:
        user = _user_message(snap)
        attempt = transient = 0
        outcome: tuple[date, str] | None = None
        try:
            while True:
                attempt += 1

                def log(reason: FailureReason, detail: str, raw: str, replayed: bool = False) -> None:
                    if failures is not None and not replayed:
                        failures.record(FailureRecord(agent=PROBE_AGENT, date=true_date, reason=reason,
                                                      detail=detail, attempt=attempt, raw_response=raw))

                try:
                    resp, replayed = llm.call_with_meta(PROBE_SYSTEM, user, max_tokens=MAX_TOKENS)
                except (RateLimited, Timeout, ApiError) as e:
                    reason = (FailureReason.RATE_LIMITED if isinstance(e, RateLimited)
                              else FailureReason.TIMEOUT if isinstance(e, Timeout) else FailureReason.API_ERROR)
                    log(reason, str(e), "")
                    transient += 1
                    if transient >= TRANSIENT_MAX_ATTEMPTS:
                        break
                    sleep(BACKOFF_BASE_S * 2 ** (transient - 1))
                    continue
                try:
                    outcome = _parse(resp)
                except _Bad as b:  # partial output is stored in the log, never used
                    log(b.reason, b.detail, resp.text, replayed)
                break
        except BudgetExceeded:
            exhausted = True
            break
        n += 1
        if outcome is None:
            failed += 1
            continue
        guess_date, guess_ticker = outcome
        d_hit = abs(guess_date - true_date) <= timedelta(days=tolerance_days)
        t_hit = guess_ticker.upper() == true_ticker.strip().upper()
        date_hits += d_hit
        ticker_hits += t_hit
        any_hits += d_hit or t_hit
    rate = any_hits / n if n else 0.0
    return ContaminationScore(n=n, date_hits=date_hits, ticker_hits=ticker_hits, rate=rate,
                              any_hits=any_hits, failed=failed, budget_exhausted=exhausted)
