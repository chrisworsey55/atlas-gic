"""Atomic JSON checkpoint store.

State must be JSON-serialisable. Dates are stored as ISO strings: use
``dump_date`` / ``load_date`` when writing and reading them.
"""
from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path


def dump_date(d: date) -> str:
    return d.isoformat()


def load_date(s: str) -> date:
    return date.fromisoformat(s)


class Checkpoint:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def _tmp(self) -> Path:
        return self.path.with_name(self.path.name + ".tmp")

    def save(self, state: dict) -> None:
        data = json.dumps(state, sort_keys=True, indent=2)  # raises before touching disk
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._tmp, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(self._tmp, self.path)
        try:  # persist the rename itself
            fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeDecodeError) as e:
            raise ValueError(f"corrupt checkpoint {self.path}: {e}") from e
        if not isinstance(state, dict):
            raise ValueError(f"corrupt checkpoint {self.path}: expected a JSON object")
        return state
