"""Append-only, hash-chained JSONL prediction log.

Each line: {"day", "agents", "prev_hash", "hash"}; hash = sha256 of the canonical
JSON (sorted keys, no whitespace) of the line without the "hash" field.
Task 12 must append (fsynced) before scoring outcomes.

LIMITATION: a pure hash chain cannot detect deletion of the *tail* line(s):
after dropping the last line the log still verifies, and one could re-append
different picks for that day. Mitigation: the runner stores ``head()`` (line
count, last hash) in its checkpoint and, on resume, requires
``contains_head(count, hash)`` -- the log must still hold the checkpointed line
at that position and be at most ``max_ahead`` (default 1: crash after append,
before checkpoint) lines beyond it. ``verify()`` alone does not catch this.

REWRITES: the log is append-only with exactly ONE sanctioned rewrite,
``drop_uncommitted_tail``. The runner's checkpoint is the commit point; a line
appended for day T after the last checkpoint (crash between append and
checkpoint) is uncommitted. When T is re-run its agent results may legitimately
differ (a live retry now succeeds, a transient failure has cleared), so the
runner drops that single line before re-running T. It only ever removes the
last line, only when the log is exactly one line beyond the checkpointed head,
the committed prefix matches the checkpointed hash and the line is for the day
being re-run; anything else raises. Committed lines are still protected by
``contains_head`` (the tail-truncation limits above are unchanged).
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

from atlas.schemas import AgentResult

GENESIS = "0" * 64


def _canon(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _hash(content: dict) -> str:
    return hashlib.sha256(_canon(content).encode("utf-8")).hexdigest()


def _agent_record(name: str, r: AgentResult) -> dict:
    if r.status == "ok":
        return {"agent": name, "status": "ok", "output": r.output.model_dump(mode="json")}
    reason = r.failure.reason.value if r.failure is not None else "no_output"
    return {"agent": name, "status": "no_opinion", "reason": reason}


class PredictionLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def _read(self) -> list[dict] | None:
        """Parsed lines if the whole chain is valid, else None."""
        if not self.path.exists():
            return []
        lines: list[dict] = []
        prev = GENESIS
        try:
            raw = self.path.read_text(encoding="utf-8")
            if raw and not raw.endswith("\n"):
                return None
            for text in raw.splitlines():
                rec = json.loads(text)
                if not isinstance(rec, dict) or rec.get("prev_hash") != prev:
                    return None
                h = rec.get("hash")
                body = {k: v for k, v in rec.items() if k != "hash"}
                if h != _hash(body):
                    return None
                lines.append(rec)
                prev = h
        except (ValueError, UnicodeDecodeError):
            return None
        return lines

    def verify(self) -> bool:
        return self._read() is not None

    def head(self) -> tuple[int, str | None]:
        """(number of lines, hash of last line); (0, None) if empty/missing.

        Raises ValueError if the log fails verification.
        """
        lines = self._read()
        if lines is None:
            raise ValueError(f"prediction log {self.path} failed verification")
        return (len(lines), lines[-1]["hash"] if lines else None)

    def contains_head(self, count: int, head_hash: str | None, max_ahead: int = 1) -> bool:
        """True iff the log verifies, line number ``count`` has hash ``head_hash``
        (count 0 <=> head_hash None), and it has at most ``max_ahead`` further lines.
        Detects tail truncation and replaced history relative to a checkpointed head.
        """
        lines = self._read()
        if lines is None or count < 0 or len(lines) < count or len(lines) - count > max_ahead:
            return False
        if count == 0:
            return head_hash is None
        return lines[count - 1]["hash"] == head_hash

    def drop_uncommitted_tail(self, expected_day: date, checkpoint_count: int,
                              checkpoint_hash: str | None) -> None:
        """Remove the single uncommitted last line (see module docstring). Raises
        ValueError unless the log verifies, has exactly ``checkpoint_count + 1`` lines,
        line ``checkpoint_count`` has ``checkpoint_hash`` (0 lines <=> None) and the last
        line's day is ``expected_day``. Atomic: temp file, fsync, os.replace."""
        lines = self._read()
        if lines is None:
            raise ValueError(f"prediction log {self.path} failed verification")
        if checkpoint_count < 0 or len(lines) != checkpoint_count + 1:
            raise ValueError(f"log has {len(lines)} lines; expected exactly {checkpoint_count + 1} "
                             "(one uncommitted line beyond the checkpoint)")
        committed = lines[checkpoint_count - 1]["hash"] if checkpoint_count else None
        if committed != checkpoint_hash:
            raise ValueError("committed prefix of the log does not match the checkpoint head")
        if lines[-1]["day"] != expected_day.isoformat():
            raise ValueError(f"uncommitted line is for {lines[-1]['day']}, not the re-run day "
                             f"{expected_day.isoformat()}")
        keep = "".join(self.path.read_text(encoding="utf-8").splitlines(keepends=True)[:-1])
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(keep)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        try:
            fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def append(self, day: date, results: dict[str, AgentResult]) -> str:
        lines = self._read()
        if lines is None:
            raise ValueError(f"prediction log {self.path} failed verification; refusing to extend")
        agents = [_agent_record(n, results[n]) for n in sorted(results)]
        day_s = day.isoformat()
        if lines:
            last = lines[-1]
            if last["day"] == day_s:
                if last["agents"] == agents:
                    return last["hash"]
                raise ValueError(f"log already has different content for {day_s}")
            if last["day"] > day_s:
                raise ValueError(f"cannot append {day_s}: log already past it ({last['day']})")
        prev = lines[-1]["hash"] if lines else GENESIS
        body = {"day": day_s, "agents": agents, "prev_hash": prev}
        rec = {**body, "hash": _hash(body)}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(_canon(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec["hash"]
