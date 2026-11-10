"""Run-level error log: append-only JSON Lines, fsynced per event, mode 0600, rotated at a size cap.

Writing never raises into the caller (one stderr warning per process, then silence) and is
thread-safe. Every string goes through ``atlas.engine.llm.sanitise`` and is truncated; tracebacks
carry only ``file:line func`` frames (no locals, no argument reprs).

Every value is converted defensively: one whose ``__str__`` raises degrades to
``"<unprintable TypeName>"`` so the record is still written.

The lock is per instance: two ErrorLog objects on one path (or two processes) can race, so
callers must share ONE instance per run.

Torn lines: a crash mid-write can leave a final line without a newline. The next append
detects this (last byte is not ``\\n``) and writes ``\\n`` plus a ``log_repaired`` marker record
before the event, so the fragment sits on its own line. ``read_events`` skips a malformed line
if it is the final line OR is immediately followed (next non-blank line) by a ``log_repaired``
marker; any other malformed line raises ValueError. Marker records are returned like any event
(filter with ``kind``). A directory or missing path reads as ``[]``.

If rotating to ``errors.log.1`` fails, the event is appended to the live file anyway (the cap is
advisory) and one warning is printed."""
from __future__ import annotations

import json
import os
import re
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


_CTRL = re.compile("[\x00-\x1f\x7f-\x9f]")


def escape_controls(text: str) -> str:
    """Every control character becomes a visible escape (``\\x1b``): one line, nothing a
    terminal can interpret."""
    return _CTRL.sub(lambda m: "\\x%02x" % ord(m.group()), text)


def _s(x, limit: int = _FIELD_LIMIT) -> str:
    """escape_controls(sanitise(str(x))) that never raises."""
    try:
        return escape_controls(sanitise(str(x), limit))
    except Exception:
        try:
            name = type(x).__name__
        except Exception:
            name = "object"
        return f"<unprintable {name}>"


def _clean(v, depth: int = 0):
    """JSON-serialisable, sanitised copy of ``v``."""
    if isinstance(v, str):
        return _s(v)
    if v is None or isinstance(v, (bool, int)):
        return v
    if isinstance(v, float):
        return v if v == v and v not in (float("inf"), float("-inf")) else str(v)
    if depth >= _MAX_DEPTH:
        return _s(v)
    if isinstance(v, dict):
        return {_s(k): _clean(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple, set, frozenset)):
        return [_clean(x, depth + 1) for x in v]
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    return _s(v)


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
            rec = self._record(kind, message, level, day, agent, reason, fields)
            line = (json.dumps(rec, separators=(",", ":")) + "\n").encode("utf-8")
            self._write(line)
        except Exception as e:  # never raise into the caller
            _warn_once(e)

    def _record(self, kind, message, level, day, agent, reason, fields) -> dict:
        return {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "level": level if isinstance(level, str) and level in _LEVELS else "error",
            "kind": _s(kind),
            "day": None if day is None else (day.isoformat() if isinstance(day, (date, datetime))
                                             else _s(day)),
            "agent": None if agent is None else _s(agent),
            "reason": None if reason is None else _s(reason),
            "message": "" if message is None else _s(message, _MSG_LIMIT),
            "fields": _clean(fields),
            "run_id": self.run_id,
        }

    def exception(self, kind: str, exc: BaseException, *, level: str = "error",
                  day=None, agent: str | None = None, **fields) -> None:
        if self.path is None:
            return
        try:
            frames = [f"{os.path.basename(fr.filename)}:{fr.lineno} {fr.name}"
                      for fr in traceback.extract_tb(exc.__traceback__)[-_MAX_FRAMES:]]
            fields = {**fields, "exc_type": type(exc).__name__, "frames": frames}
            self.event(kind, exc, level=level, day=day, agent=agent, **fields)
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
                try:
                    os.replace(self.path, self.path.with_name(self.path.name + ".1"))
                except OSError as e:  # cap is advisory: keep logging to the live file
                    _warn_once(e)
            fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.fchmod(fd, 0o600)
                end = os.fstat(fd).st_size
                if end > 0 and os.pread(fd, 1, end - 1) != b"\n":
                    marker = self._record("log_repaired", "torn final line set aside", "warning",
                                          None, None, None, {})
                    line = (b"\n" + json.dumps(marker, separators=(",", ":")).encode("utf-8")
                            + b"\n" + line)
                view = memoryview(line)
                while view:
                    view = view[os.write(fd, view):]
                os.fsync(fd)
            finally:
                os.close(fd)


class NullErrorLog(ErrorLog):
    def __init__(self) -> None:
        super().__init__(None)


def read_events(path: Path, *, kind: str | None = None, strict: bool = True,
                skipped: list[int] | None = None) -> list[dict]:
    """Records of an error log (see the module docstring for torn-line rules). Missing file or a
    non-file path -> []. ``strict=False`` skips every malformed line instead of raising; the
    1-based numbers of all skipped lines are appended to ``skipped`` if given."""
    path = Path(path)
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    parsed: list[tuple[int, dict | None]] = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            rec = json.loads(ln)
            if not isinstance(rec, dict):
                raise ValueError("not an object")
        except ValueError:
            rec = None
        parsed.append((i, rec))
    out = []
    for n, (i, rec) in enumerate(parsed):
        if rec is None:
            nxt = parsed[n + 1][1] if n + 1 < len(parsed) else None
            if n == len(parsed) - 1 or (nxt is not None and nxt.get("kind") == "log_repaired") \
                    or not strict:
                if skipped is not None:
                    skipped.append(i + 1)
                continue
            raise ValueError(f"error log is corrupt at line {i + 1}: {path}")
        if kind is None or rec.get("kind") == kind:
            out.append(rec)
    return out
