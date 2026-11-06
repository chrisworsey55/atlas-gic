import os
from datetime import date

import pytest

from atlas.backtest.checkpoint import Checkpoint, dump_date, load_date


def test_checkpoint_roundtrip(tmp_path):
    cp = Checkpoint(tmp_path / "cp.json")
    state = {"last_completed_day": dump_date(date(2024, 1, 2)), "portfolio": {"cash": 1.5},
             "weights_path": "w.json", "spend": 3.2, "halted": False}
    cp.save(state)
    assert cp.load() == state
    assert load_date(cp.load()["last_completed_day"]) == date(2024, 1, 2)


def test_missing_checkpoint_returns_none(tmp_path):
    assert Checkpoint(tmp_path / "cp.json").load() is None


def test_corrupt_checkpoint_raises(tmp_path):
    p = tmp_path / "cp.json"
    p.write_text("{not json")
    with pytest.raises(ValueError):
        Checkpoint(p).load()


def test_non_dict_checkpoint_raises(tmp_path):
    p = tmp_path / "cp.json"
    p.write_text("[1]")
    with pytest.raises(ValueError):
        Checkpoint(p).load()


def test_checkpoint_atomic_write_survives_partial_write(tmp_path, monkeypatch):
    cp = Checkpoint(tmp_path / "cp.json")
    cp.save({"v": 1})

    def boom(*a, **k):
        raise OSError("crash")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        cp.save({"v": 2})
    monkeypatch.undo()
    assert cp.load() == {"v": 1}
    # stray temp file does not break load
    (tmp_path / "cp.json.tmp").write_text("{partial")
    assert cp.load() == {"v": 1}
    cp.save({"v": 3})
    assert cp.load() == {"v": 3}


def test_unserialisable_state_does_not_clobber(tmp_path):
    cp = Checkpoint(tmp_path / "cp.json")
    cp.save({"v": 1})
    with pytest.raises(TypeError):
        cp.save({"d": date(2024, 1, 1)})
    assert cp.load() == {"v": 1}
