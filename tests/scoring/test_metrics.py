import math

import pandas as pd
import pytest

from atlas.scoring.metrics import excess_return, hit_rate, rank_ic


def test_rank_ic_perfect_monotone_is_one():
    assert rank_ic([0.1, 0.2, 0.3, 0.4], [1, 5, 9, 20]) == pytest.approx(1.0)


def test_rank_ic_constant_convictions_is_nan():
    assert math.isnan(rank_ic([0.5, 0.5, 0.5, 0.5], [1, 2, 3, 4]))
    assert math.isnan(rank_ic([0.1, 0.2, 0.3], [0.01, 0.01, 0.01]))


def test_rank_ic_fewer_than_three_points_is_nan():
    assert math.isnan(rank_ic([0.1, 0.2], [1, 2]))
    assert math.isnan(rank_ic([], []))


def test_excess_return_subtracts_benchmark_and_costs():
    t = pd.Series([100, 101, 102, 103, 104, 110.0])  # +10% over 5 days
    b = pd.Series([200, 201, 202, 203, 204, 210.0])  # +5%
    assert excess_return(t, b, 5, 0.002) == pytest.approx(0.10 - 0.05 - 0.002)


def test_excess_return_nan_when_window_too_short():
    s = pd.Series([1.0, 2.0])
    assert math.isnan(excess_return(s, s, 5, 0.0))


def test_hit_rate():
    assert hit_rate([0.1, -0.1, 0.2, 0.0]) == pytest.approx(0.5)
    assert math.isnan(hit_rate([]))
