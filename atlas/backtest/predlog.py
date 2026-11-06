"""Append-only, hash-chained JSONL prediction log.

Each line: {"day", "agents", "prev_hash", "hash"}; hash = sha256 of the canonical
JSON (sorted keys, no whitespace) of the line without the "hash" field.
Task 12 must append (fsynced) before scoring outcomes.
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
