import pandas as pd
import pytest

from atlas.backtest.ablation import baseline_target_weights, cio_ablation
from atlas.schemas import AgentOutput, AgentResult, Pick
from datetime import date

IDX = pd.date_range("2024-01-01", periods=3)


def test_ablation_reports_cio_not_beating_baseline():
    cio = pd.Series([0.01, 0.0, -0.01], index=IDX)
    base = pd.Series([0.02, 0.01, 0.0], index=IDX)
    r = cio_ablation(cio, base)
    assert r["cio_total"] == pytest.approx(1.01 * 1.0 * 0.99 - 1)
    assert r["baseline_total"] == pytest.approx(1.02 * 1.01 - 1)
    assert r["cio_beats_baseline"] is False


def test_ablation_fewer_than_two_dates_raises():
    cio = pd.Series([0.1, 0.1], index=pd.date_range("2024-01-01", periods=2))
    base = pd.Series([0.0, 0.0], index=pd.date_range("2024-01-02", periods=2))
    with pytest.raises(ValueError):
        cio_ablation(cio, base)


def test_ablation_join_values():
    cio = pd.Series([0.1, 0.1, 0.1, 0.5], index=pd.date_range("2024-01-01", periods=4))
    base = pd.Series([0.0, float("nan"), 0.0, 0.0], index=pd.date_range("2024-01-01", periods=4))
    r = cio_ablation(cio, base)  # dates 01-01, 01-03, 01-04
    assert r["cio_total"] == pytest.approx(1.1 * 1.1 * 1.5 - 1)
    assert r["baseline_total"] == pytest.approx(0.0)
    assert r["cio_beats_baseline"] is True


def res(name, picks):
    d = date(2024, 1, 2)
    return AgentResult(agent=name, date=d, output=AgentOutput(
        agent=name, date=d, view="bullish", confidence=0.5,
        picks=[Pick(ticker=t, direction=dr, conviction=c, horizon_days=5) for t, dr, c in picks]))


def test_baseline_target_weights_hand_computed():
    days = {
        "a": res("a", [("X", "long", 0.8), ("Y", "long", 0.4), ("Z", "short", 0.9)]),
        "b": res("b", [("X", "long", 0.5), ("Y", "none", 0.9)]),
        "cio": res("cio", [("W", "long", 1.0)]),
        "bad": AgentResult(agent="bad", date=date(2024, 1, 2)),
    }
    w = baseline_target_weights(days, {"a": 2.0, "b": 1.0, "cio": 5.0, "bad": 1.0})
    # X = 2*0.8 + 1*0.5 = 2.1 ; Y = 2*0.4 = 0.8 ; total 2.9
    assert w == pytest.approx({"X": 2.1 / 2.9, "Y": 0.8 / 2.9})
    assert sum(w.values()) == pytest.approx(1.0)


def test_baseline_max_positions_and_empty():
    picks = [(f"T{i}", "long", 0.1 * (i + 1)) for i in range(9)]
    w = baseline_target_weights({"a": res("a", picks)}, {"a": 1.0}, max_positions=3)
    assert set(w) == {"T8", "T7", "T6"} and sum(w.values()) == pytest.approx(1.0)
    assert baseline_target_weights({"a": res("a", [("X", "short", 0.5)])}, {"a": 1.0}) == {}


def test_baseline_layers_arg_excludes_non_layer_1_3():
    days = {"a": res("a", [("X", "long", 0.5)]), "z": res("z", [("Y", "long", 0.5)])}
    w = baseline_target_weights(days, {"a": 1.0, "z": 1.0}, layers={"a": 2, "z": 4})
    assert set(w) == {"X"}


def test_baseline_excludes_control_by_default():
    days = {"a": res("a", [("X", "long", 0.5)]), "control": res("control", [("Y", "long", 0.9)])}
    assert set(baseline_target_weights(days, {"a": 1.0, "control": 1.0})) == {"X"}
