"""
Data transformation capabilities including groupby, merge, and pivot operations.
"""

import pandas as pd
from pathlib import Path
import os
from typing import Any, Literal, Optional, List, Dict
from typing_extensions import TypedDict
import traceback


class GroupByInfo(TypedDict):
    """Parameters and outcome summary of a groupby aggregation."""

    group_by_columns: list[str]
    operations: dict[str, str]
    filter_condition: str | None
    number_of_groups: int
    original_rows: int
    aggregated_columns: list[str]


class GroupByOperationsResult(TypedDict):
    """Structured result for a successful groupby operation."""

    success: Literal[True]
    file_path: str
    output_file: str
    group_info: GroupByInfo
    results: list[dict[str, Any]]
    message: str


GROUPBY_OPERATIONS = ("count", "sum", "mean", "median", "std", "min", "max", "nunique")


def groupby_operations(
    file_path: str,
    group_by: List[str],
    operations: Dict[str, str],
    filter_condition: Optional[str] = None,
    output_file: Optional[str] = None,
    overwrite: bool = False,
) -> dict:
    """
    Perform groupby operations on data.

    Args:
        file_path: Path to the data file
        group_by: Columns to group by
        operations: Dictionary of column: operation pairs
        filter_condition: Optional filter condition

    Returns:
        Dictionary with groupby results
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "error_type": "FileNotFoundError",
            }

        # Load data
        df = pd.read_csv(file_path)

        # Apply filter if provided
        if filter_condition:
            try:
                df = df.query(filter_condition)
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Filter condition error: {str(e)}",
                    "error_type": "ValueError",
                }

        # Check if group_by columns exist
        missing_cols = [col for col in group_by if col not in df.columns]
        if missing_cols:
            return {
                "success": False,
                "error": f"Group by columns not found: {missing_cols}",
                "error_type": "ValueError",
            }

        # Group by operations
        grouped = df.groupby(group_by)

        # Apply aggregations (an unknown operation used to silently become mean)
        unknown_ops = sorted(set(operations.values()) - set(GROUPBY_OPERATIONS))
        if unknown_ops:
            return {
                "success": False,
                "error": f"Unknown aggregation operation(s): {unknown_ops}. "
                f"Valid operations: {', '.join(GROUPBY_OPERATIONS)}",
                "error_type": "ValueError",
            }
        agg_dict = {col: op for col, op in operations.items() if col in df.columns}

        # Perform aggregation
        result = grouped.agg(agg_dict)

        # Reset index to make group columns regular columns
        result = result.reset_index()

        # Convert to JSON-serializable format
        result_dict = result.to_dict("records")

        # Group information
        group_info = {
            "group_by_columns": group_by,
            "operations": operations,
            "filter_condition": filter_condition,
            "number_of_groups": len(result),
            "original_rows": len(df),
            "aggregated_columns": list(agg_dict.keys()),
        }

        # Save result
        output_path = output_file or str(
            Path(file_path).with_name(f"{Path(file_path).stem}_grouped.csv")
        )
        if Path(output_path).resolve() == Path(file_path).resolve() or (
            Path(output_path).exists() and os.path.samefile(output_path, file_path)
        ):
            raise ValueError("The output must not replace the input file")
        result.to_csv(output_path, index=False, mode="w" if overwrite else "x")

        return {
            "success": True,
            "file_path": file_path,
            "output_file": output_path,
            "group_info": group_info,
            "results": result_dict,
            "message": f"Grouped data into {len(result)} groups",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


def merge_datasets(
    left_file: str,
    right_file: str,
    join_type: str = "inner",
    left_on: Optional[str] = None,
    right_on: Optional[str] = None,
    on: Optional[str] = None,
) -> dict:
    """
    Merge two datasets.

    Args:
        left_file: Path to the left dataset
        right_file: Path to the right dataset
        join_type: Type of join (inner, outer, left, right)
        left_on: Column to join on in left dataset
        right_on: Column to join on in right dataset
        on: Column to join on (if same in both datasets)

    Returns:
        Dictionary with merge results
    """
    try:
        if not os.path.exists(left_file):
            return {
                "success": False,
                "error": f"Left file not found: {left_file}",
                "error_type": "FileNotFoundError",
            }

        if not os.path.exists(right_file):
            return {
                "success": False,
                "error": f"Right file not found: {right_file}",
                "error_type": "FileNotFoundError",
            }

        # Load datasets
        left_df = pd.read_csv(left_file)
        right_df = pd.read_csv(right_file)

        # Determine join keys
        if on:
            left_on = on
            right_on = on
        elif not left_on or not right_on:
            return {
                "success": False,
                "error": "Must specify either 'on' or both 'left_on' and 'right_on'",
                "error_type": "ValueError",
            }

        # Check if join columns exist
        if left_on not in left_df.columns:
            return {
                "success": False,
                "error": f"Left join column '{left_on}' not found in left dataset",
                "error_type": "ValueError",
            }

        if right_on not in right_df.columns:
            return {
                "success": False,
                "error": f"Right join column '{right_on}' not found in right dataset",
                "error_type": "ValueError",
            }

        # Perform merge
        merged_df = pd.merge(
            left_df,
            right_df,
            left_on=left_on,
            right_on=right_on,
            how=join_type,
            suffixes=("_left", "_right"),
        )

        # Merge statistics
        merge_stats = {
            "left_shape": left_df.shape,
            "right_shape": right_df.shape,
            "merged_shape": merged_df.shape,
            "join_type": join_type,
            "left_on": left_on,
            "right_on": right_on,
            "common_values": int(left_df[left_on].isin(right_df[right_on]).sum()),
            "left_only_values": int((~left_df[left_on].isin(right_df[right_on])).sum()),
            "right_only_values": int(
                (~right_df[right_on].isin(left_df[left_on])).sum()
            ),
        }

        # Save merged dataset
        output_path = str(
            Path(left_file).with_name(f"{Path(left_file).stem}_merged.csv")
        )
        merged_df.to_csv(output_path, index=False)

        # Convert to JSON-serializable format (limit to first 100 rows)
        merged_dict = merged_df.head(100).to_dict("records")

        return {
            "success": True,
            "left_file": left_file,
            "right_file": right_file,
            "output_file": output_path,
            "merge_stats": merge_stats,
            "merged_data": merged_dict,
            "message": f"Merged datasets: {merge_stats['left_shape']} + {merge_stats['right_shape']} = {merge_stats['merged_shape']}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }


def create_pivot_table(
    file_path: str,
    index: List[str],
    columns: Optional[List[str]] = None,
    values: Optional[List[str]] = None,
    aggfunc: str = "mean",
) -> dict:
    """
    Create a pivot table from data.

    Args:
        file_path: Path to the data file
        index: Columns to use as row index
        columns: Columns to use as column headers
        values: Columns to aggregate
        aggfunc: Aggregation function

    Returns:
        Dictionary with pivot table results
    """
    try:
        if not os.path.exists(file_path):
            return {
                "success": False,
                "error": f"File not found: {file_path}",
                "error_type": "FileNotFoundError",
            }

        # Load data
        df = pd.read_csv(file_path)

        # Check if index columns exist
        missing_cols = [col for col in index if col not in df.columns]
        if missing_cols:
            return {
                "success": False,
                "error": f"Index columns not found: {missing_cols}",
                "error_type": "ValueError",
            }

        # Check if column headers exist
        if columns:
            missing_cols = [col for col in columns if col not in df.columns]
            if missing_cols:
                return {
                    "success": False,
                    "error": f"Column header columns not found: {missing_cols}",
                    "error_type": "ValueError",
                }

        # Check if value columns exist
        if values:
            missing_cols = [col for col in values if col not in df.columns]
            if missing_cols:
                return {
                    "success": False,
                    "error": f"Value columns not found: {missing_cols}",
                    "error_type": "ValueError",
                }

        # None means every numeric column that is not already a pivot key
        if not values:
            keys = set(index) | set(columns or [])
            values = [
                col
                for col in df.select_dtypes(include="number").columns
                if col not in keys
            ]
            if not values:
                return {
                    "success": False,
                    "error": "No numeric value columns to aggregate",
                    "error_type": "ValueError",
                }

        # An empty cell has no observations: only a count of it is truly 0.
        pivot_table = pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=0 if aggfunc == "count" else None,
        )

        # Reset index to make it a regular DataFrame
        pivot_table = pivot_table.reset_index()

        # Flatten column names if they are MultiIndex
        if isinstance(pivot_table.columns, pd.MultiIndex):
            pivot_table.columns = [
                "_".join(col).strip() if col[1] != "" else col[0]
                for col in pivot_table.columns.values
            ]

        # Convert to JSON-serializable format
        pivot_dict = (
            pivot_table.astype(object)
            .where(pivot_table.notna(), None)
            .to_dict("records")
        )

        # Pivot table information
        pivot_info = {
            "index_columns": index,
            "column_headers": columns,
            "value_columns": values,
            "aggregation_function": aggfunc,
            "pivot_shape": pivot_table.shape,
            "original_shape": df.shape,
        }

        # Save pivot table
        output_path = str(
            Path(file_path).with_name(f"{Path(file_path).stem}_pivot.csv")
        )
        pivot_table.to_csv(output_path, index=False)

        return {
            "success": True,
            "file_path": file_path,
            "output_file": output_path,
            "pivot_info": pivot_info,
            "pivot_table": pivot_dict,
            "message": f"Created pivot table with shape {pivot_table.shape}",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
