"""Run-level error log: append-only JSON Lines, fsynced per event, mode 0600, rotated at a size cap.

Writing never raises into the caller (one stderr warning per process, then silence) and is
thread-safe. Every string goes through ``atlas.engine.llm.sanitise`` and is truncated; tracebacks
carry only ``file:line func`` frames (no locals, no argument reprs)."""
from __future__ import annotations

import json
import os
import sys
import threading
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

from atlas.engine.llm import sanitise

_LEVELS = frozenset({"error", "warning", "info"})
_MSG_LIMIT = 500
_FIELD_LIMIT = 300
_MAX_FRAMES = 8
_MAX_DEPTH = 6
_warned = False
_warn_lock = threading.Lock()


def _warn_once(exc: BaseException) -> None:
    global _warned
    with _warn_lock:
        if _warned:
            return
        _warned = True
    try:
        print(f"warning: could not write error log ({type(exc).__name__})", file=sys.stderr)
    except Exception:
        pass


def _clean(v, depth: int = 0):
    """JSON-serialisable, sanitised copy of ``v``."""
    if isinstance(v, str):
        return sanitise(v, _FIELD_LIMIT)
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else str(v)
    if depth >= _MAX_DEPTH:
        return sanitise(str(v), _FIELD_LIMIT)
    if isinstance(v, dict):
        return {sanitise(str(k), _FIELD_LIMIT): _clean(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_clean(x, depth + 1) for x in v]
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return sanitise(str(v), _FIELD_LIMIT)


class ErrorLog:
    def __init__(self, path: Path | None, *, run_id: str | None = None,
                 max_bytes: int = 5_000_000) -> None:
        self.path = Path(path) if path is not None else None
        self.run_id = run_id
        self.max_bytes = max_bytes
        self._lock = threading.Lock()

    def event(self, kind: str, message: str = "", *, level: str = "error",
              day: date | str | None = None, agent: str | None = None,
              reason: str | None = None, **fields) -> None:
        if self.path is None:
            return
        try:
            rec = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "level": level if level in _LEVELS else "error",
                "kind": sanitise(str(kind), _FIELD_LIMIT),
                "day": None if day is None else (day.isoformat() if isinstance(day, (date, datetime))
                                                 else sanitise(str(day), _FIELD_LIMIT)),
                "agent": None if agent is None else sanitise(str(agent), _FIELD_LIMIT),
                "reason": None if reason is None else sanitise(str(reason), _FIELD_LIMIT),
                "message": sanitise("" if message is None else str(message), _MSG_LIMIT),
                "fields": _clean(fields),
                "run_id": self.run_id,
            }
            line = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
            self._write(line)
        except Exception as e:  # never raise into the caller
            _warn_once(e)

    def exception(self, kind: str, exc: BaseException, *, level: str = "error",
                  day=None, agent: str | None = None, **fields) -> None:
        if self.path is None:
            return
        try:
            frames = [f"{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}"
                      for fr in traceback.extract_tb(exc.__traceback__)[-_MAX_FRAMES:]]
            fields = {**fields, "exc_type": type(exc).__name__, "frames": frames}
            self.event(kind, str(exc), level=level, day=day, agent=agent, **fields)
        except Exception as e:
            _warn_once(e)

    def _write(self, line: bytes) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            try:
                size = self.path.stat().st_size
            except FileNotFoundError:
                size = 0
            if size > 0 and size + len(line) > self.max_bytes:
                os.replace(self.path, self.path.with_name(self.path.name + ".1"))
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.fchmod(fd, 0o600)
                view = memoryview(line)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)


class NullErrorLog(ErrorLog):
    def __init__(self) -> None:
        super().__init__(None)


def read_events(path: Path, *, kind: str | None = None) -> list[dict]:
    """Records of an error log. A malformed FINAL line (crash mid-write) is skipped; a malformed
    earlier line raises ValueError. Missing file -> []."""
    path = Path(path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    last = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
            if not isinstance(rec, dict):
                raise ValueError("not an object")
        except ValueError:
            if i == last:
                continue
            raise ValueError(f"error log is corrupt at line {i + 1}: {path}") from None
        if kind is None or rec.get("kind") == kind:
            out.append(rec)
    return out
