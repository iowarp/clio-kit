"""Joining a pvserver's per-rank statistics, without ParaView.

A remote measurement brings back one row per pvserver rank, never the dataset;
``data.combine`` must give the whole field's numbers from those rows.
"""

from __future__ import annotations

import math
import statistics

import pytest

from paraview_mcp.node import data


def _row(values: list[float], bins: int, low: float, high: float) -> dict:
    """What the rank script reports for ``values``, binned on [low, high]."""
    finite = [v for v in values if math.isfinite(v)]
    mean = statistics.fmean(finite) if finite else math.nan
    counts = [0] * bins
    for v in finite:
        at = bins - 1 if v == high else int((v - low) / (high - low) * bins)
        counts[at] += 1
    return {
        "count": float(len(values)),
        "finite": float(len(finite)),
        "mean": mean,
        "m2": sum((v - mean) ** 2 for v in finite),
        "min": min(finite) if finite else math.nan,
        "max": max(finite) if finite else math.nan,
        "median": statistics.median(finite) if finite else math.nan,
        "histogram": counts if bins > 1 else float(counts[0]),
    }


def test_ranks_join_to_the_whole_fields_mean_and_spread() -> None:
    parts = [[1.0, 2.0, 3.0], [10.0, 11.0], [4.0, math.nan, 5.0, 6.0]]
    whole = [v for p in parts for v in p if math.isfinite(v)]
    low, high = min(whole), max(whole)
    rows = [_row(p, 4, low, high) for p in parts]
    joined = data.combine(rows, 4, rows)
    assert joined["count"] == 9 and joined["finite"] == 8
    assert joined["mean"] == pytest.approx(statistics.fmean(whole))
    assert joined["std"] == pytest.approx(statistics.pstdev(whole))
    assert (joined["min"], joined["max"]) == (1.0, 11.0)
    assert joined["histogram"]["counts"] == [3, 2, 1, 2]
    assert joined["histogram"]["edges"] == pytest.approx([1.0, 3.5, 6.0, 8.5, 11.0])
    # Split across ranks, the median is the bin that holds it, not a guess.
    assert joined["median"] is None
    low_edge, high_edge = joined["median_between"]
    assert low_edge <= statistics.median(whole) <= high_edge


def test_one_rank_holding_the_field_gives_the_exact_median() -> None:
    rows = [_row([], 3, 0, 1), _row([3.0, 1.0, 2.0, 8.0], 3, 1.0, 8.0)]
    joined = data.combine(rows, 3)
    assert joined["median"] == 2.5
    assert joined["histogram"]["counts"] == [3, 0, 1]


def test_a_series_step_skips_the_shared_binning() -> None:
    rows = [_row([1.0, 2.0], 1, 1.0, 2.0), _row([5.0], 1, 5.0, 5.0)]
    joined = data.combine(rows, 1)
    assert joined["mean"] == pytest.approx(8 / 3)
    assert joined["median"] is None and "histogram" not in joined


def test_no_finite_values_is_said_plainly() -> None:
    assert data.combine([_row([math.nan], 2, 0, 1)], 2) == {"count": 1, "finite": 0}


def test_a_flat_field_bins_like_numpy() -> None:
    joined = data.combine([_row([2.0, 2.0], 2, 1.5, 2.5)], 2)
    assert joined["histogram"]["edges"] == [1.5, 2.0, 2.5]
    assert joined["std"] == 0.0
