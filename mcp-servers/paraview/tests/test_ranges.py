"""The default colour range's percentiles, joined from a pvserver's ranks, without
ParaView.

Each rank sends evenly spaced order statistics of its values (numpy's linear
percentiles); ``ranges.pooled`` must give the whole field's 2nd and 98th percentiles.
"""

from __future__ import annotations

import random

import pytest

from paraview_mcp.node import ranges


def _numpy_like(ordered: list[float], percent: float) -> float:
    at = percent / 100 * (len(ordered) - 1)
    low = int(at)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (at - low)


def _row(values: list[float], samples: int = ranges.SAMPLES, stride: int = 1) -> dict:
    """What the rank script reports for ``values``."""
    ordered = sorted(values)
    picks = [_numpy_like(ordered, 100 * i / (samples - 1)) for i in range(samples)]
    return {"finite": float(len(ordered)), "quantiles": picks, "stride": float(stride)}


def test_a_percentile_is_numpys_linear_one() -> None:
    assert ranges.percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    assert ranges.percentile([5.0], 98.0) == 5.0
    assert ranges.percentile([0.0, 10.0], 2.0) == pytest.approx(0.2)


def test_one_rank_gives_the_exact_percentiles() -> None:
    values = [random.Random(4).gauss(0, 1) for _ in range(5000)]
    ordered = sorted(values)
    for percent in ranges.PERCENTILES:
        assert ranges.pooled([_row(values)], percent) == pytest.approx(
            _numpy_like(ordered, percent), abs=1e-12
        )


def test_ranks_join_to_the_whole_fields_percentiles() -> None:
    draw = random.Random(7)
    parts = [
        [draw.uniform(0, 1) for _ in range(30000)],
        [draw.uniform(0, 1) for _ in range(10000)] + [100.0] * 400,
        [],
    ]
    rows = [
        _row(p) if p else {"finite": 0.0, "quantiles": [float("nan")]} for p in parts
    ]
    whole = sorted(v for p in parts for v in p)
    for percent in ranges.PERCENTILES:
        assert ranges.pooled(rows, percent) == pytest.approx(
            _numpy_like(whole, percent), abs=2e-3
        )
    # The body at 100 is 1% of the field: the top of the range stays in the ramp.
    assert ranges.pooled(rows, 98.0) < 1.0


def test_a_strided_rank_weighs_as_the_values_it_stands_for() -> None:
    low = _row([0.0] * 100, stride=10)
    high = _row([1.0] * 1000)
    # 1000 zeros and 1000 ones: the median sits between them.
    assert 0.0 <= ranges.pooled([low, high], 50.0) <= 1.0
    assert ranges.pooled([low, high], 2.0) == pytest.approx(0.0)
    assert ranges.pooled([low, high], 98.0) == pytest.approx(1.0)


def test_no_values_gives_no_range() -> None:
    assert ranges.pooled([{"finite": 0.0, "quantiles": [0.0]}], 2.0) is None
    assert ranges.pooled([], 98.0) is None
