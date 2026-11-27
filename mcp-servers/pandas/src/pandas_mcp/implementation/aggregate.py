"""Read-only grouped aggregation that returns its table inline and writes nothing."""

import math
import os
from typing import Any, Dict, List, Optional

import pandas as pd

AGGREGATE_OPERATIONS = (
    "sum",
    "mean",
    "median",
    "min",
    "max",
    "count",
    "std",
    "var",
    "nunique",
    "first",
    "last",
)
PERIODS = {
    "year": "Y",
    "quarter": "Q",
    "month": "M",
    "week": "W",
    "day": "D",
}
MAX_GROUPS = 10000


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _bucket(df: pd.DataFrame, token: str, notes: List[str]) -> str:
    """Add a period column for "column:period" and return its name."""
    column, period = (part.strip() for part in token.split(":", 1))
    if column not in df.columns:
        raise ValueError(f"Group by column not found: {column!r}")
    if period not in PERIODS and period != "hour":
        raise ValueError(
            f"Unknown period {period!r} in {token!r}; use one of "
            f"{', '.join([*PERIODS, 'hour'])}"
        )
    stamps = pd.to_datetime(df[column], errors="coerce")
    unparsed = int(stamps.isna().sum() - df[column].isna().sum())
    if unparsed:
        notes.append(f"{unparsed} {column!r} values are not dates; grouped as null")
    name = f"{column}_{period}"
    if period == "hour":
        df[name] = stamps.dt.strftime("%Y-%m-%d %H:00")
    else:
        df[name] = stamps.dt.to_period(PERIODS[period]).astype("string")
    df.loc[stamps.isna(), name] = None
    return name


def aggregate(
    file_path: str,
    group_by: List[str],
    operations: Dict[str, str],
    filter_condition: Optional[str] = None,
    drop_duplicates: bool = False,
    sort_by: Optional[str] = None,
    descending: bool = False,
    max_groups: int = 1000,
) -> Dict[str, Any]:
    """Group a CSV and aggregate it without writing anything.

    ``group_by`` entries are column names or ``"column:period"`` with period
    year, quarter, month, week, day or hour. ``operations`` maps a column to one
    operation or a comma-separated list. Missing group keys form their own
    group so nothing disappears silently.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    if not group_by:
        raise ValueError("group_by needs at least one column")
    if not operations:
        raise ValueError("operations needs at least one column: operation pair")
    max_groups = max(1, min(int(max_groups), MAX_GROUPS))

    df = pd.read_csv(file_path)
    rows_read = len(df)
    notes: List[str] = []

    duplicates_dropped = 0
    if drop_duplicates:
        before = len(df)
        df = df.drop_duplicates()
        duplicates_dropped = before - len(df)

    rows_filtered_out = 0
    if filter_condition:
        try:
            kept = df.query(filter_condition)
        except Exception as exc:
            raise ValueError(f"Filter condition error: {exc}") from exc
        rows_filtered_out = len(df) - len(kept)
        df = kept.copy()

    keys = []
    for token in group_by:
        if ":" in token and token not in df.columns:
            keys.append(_bucket(df, token, notes))
        elif token in df.columns:
            keys.append(token)
        else:
            raise ValueError(f"Group by column not found: {token!r}")

    spec: Dict[str, List[str]] = {}
    for column, ops in operations.items():
        if column not in df.columns:
            raise ValueError(f"Aggregation column not found: {column!r}")
        names = [op.strip() for op in str(ops).split(",") if op.strip()]
        unknown = sorted(set(names) - set(AGGREGATE_OPERATIONS))
        if unknown or not names:
            raise ValueError(
                f"Unknown operation(s) {unknown or ops!r} for {column!r}; "
                f"use {', '.join(AGGREGATE_OPERATIONS)}"
            )
        spec[column] = names

    grouped = df.groupby(keys, dropna=False, sort=True)
    table = grouped.agg(spec)
    table.columns = [f"{column}_{op}" for column, op in table.columns]
    table.insert(0, "rows", grouped.size())
    table = table.reset_index()

    if sort_by:
        if sort_by not in table.columns:
            raise ValueError(
                f"sort_by {sort_by!r} is not an output column: {list(table.columns)}"
            )
        table = table.sort_values(sort_by, ascending=not descending, kind="stable")

    total = len(table)
    records = [
        {key: _plain(value) for key, value in row.items()}
        for row in table.head(max_groups).to_dict("records")
    ]
    result: Dict[str, Any] = {
        "success": True,
        "file_path": file_path,
        "group_by": keys,
        "columns": list(table.columns),
        "rows_read": rows_read,
        "duplicates_dropped": duplicates_dropped,
        "rows_filtered_out": rows_filtered_out,
        "rows_aggregated": len(df),
        "number_of_groups": total,
        "truncated": total > max_groups,
        "results": records,
        "message": f"Aggregated {len(df)} rows into {total} groups",
    }
    if notes:
        result["notes"] = notes
    return result
