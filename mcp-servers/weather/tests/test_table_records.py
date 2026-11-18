"""Tables in tool results carry missing values as null, not NaN."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from weather_mcp.implementation.common import table_records


def test_missing_landfall_and_total_become_null() -> None:
    frame = pd.DataFrame(
        {
            "object": [1, 2],
            "landfall": [pd.Timestamp("2022-12-23T00:00"), pd.NaT],
            "precip3h_region_mean_total": [80.2, np.nan],
            "start": pd.to_datetime(["2022-12-22", "2023-02-14"]),
        }
    )
    rows = table_records(frame)
    json.dumps(rows, allow_nan=False)
    assert rows[0]["landfall"] == "2022-12-23T00:00:00"
    assert rows[1]["landfall"] is None
    assert rows[1]["precip3h_region_mean_total"] is None
    assert rows[1]["start"] == "2023-02-14T00:00:00"
    assert isinstance(rows[0]["object"], int)


def test_limit_and_infinities() -> None:
    frame = pd.DataFrame({"value": [1.0, np.inf, -np.inf, 2.0]})
    rows = table_records(frame, limit=3)
    assert [r["value"] for r in rows] == [1.0, None, None]
