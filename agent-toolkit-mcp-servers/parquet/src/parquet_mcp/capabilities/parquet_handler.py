"""Core Parquet file operations and handlers."""

import json
import os
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
from typing import Optional, List, Dict, Any


def _apply_filter(table: pa.Table, filter_dict: Optional[Dict[str, Any]]) -> pa.Table:
    """
    Apply a structured filter to a PyArrow table.

    Filter format (JSON):
    {
        "column": "column_name",
        "op": "less|greater|equal|not_equal|less_equal|greater_equal|is_null|is_valid|is_in",
        "value": <value> (not needed for is_null/is_valid),
        "values": [<list>] (for is_in operation)
    }

    Or for compound filters:
    {
        "and": [<filter1>, <filter2>, ...]
    }
    {
        "or": [<filter1>, <filter2>, ...]
    }
    {
        "not": <filter>
    }

    Args:
        table: PyArrow table
        filter_dict: Filter specification as dict

    Returns:
        Filtered table
    """
    if filter_dict is None:
        return table

    try:
        mask = _build_filter_mask(table, filter_dict)
        if mask is None:
            # If mask is None, it means the filter was invalid, so return unfiltered table
            return table
        return table.filter(mask)
    except Exception as e:
        # Return unfiltered table on filter error
        print(f"Filter error: {e}")
        return table


def _build_filter_mask(
    table: pa.Table, filter_dict: Dict[str, Any]
) -> Optional[pa.Array]:
    """
    Build a boolean mask from filter specification.

    Returns:
        Boolean array mask, or None if filter is invalid
    """
    # Handle logical operators
    if "and" in filter_dict:
        masks = [_build_filter_mask(table, f) for f in filter_dict["and"]]
        if any(m is None for m in masks):
            return None
        result = masks[0]
        for mask in masks[1:]:
            result = pc.and_(result, mask)
        return result

    if "or" in filter_dict:
        masks = [_build_filter_mask(table, f) for f in filter_dict["or"]]
        if any(m is None for m in masks):
            return None
        result = masks[0]
        for mask in masks[1:]:
            result = pc.or_(result, mask)
        return result

    if "not" in filter_dict:
        inner_mask = _build_filter_mask(table, filter_dict["not"])
        if inner_mask is None:
            return None
        return pc.invert(inner_mask)

    # Handle comparison operations
    if "column" not in filter_dict or "op" not in filter_dict:
        return None

    column_name = filter_dict["column"]
    op = filter_dict["op"]

    if column_name not in table.column_names:
        return None

    column = table[column_name]

    # Null checks
    if op == "is_null":
        return pc.is_null(column)
    elif op in ("is_valid", "is_not_null"):
        return pc.is_valid(column)

    # IN operation (support both "in" and "is_in")
    elif op in ("is_in", "in"):
        if "values" not in filter_dict:
            return None
        values = filter_dict["values"]
        return pc.is_in(column, value_set=pa.array(values))

    # Comparison operations require value
    if "value" not in filter_dict:
        return None

    value = filter_dict["value"]

    if op == "equal":
        return pc.equal(column, value)
    elif op == "not_equal":
        return pc.not_equal(column, value)
    elif op == "less":
        return pc.less(column, value)
    elif op == "less_equal":
        return pc.less_equal(column, value)
    elif op == "greater":
        return pc.greater(column, value)
    elif op == "greater_equal":
        return pc.greater_equal(column, value)

    return None


async def summarize(file_path: str) -> str:
    """
    Return a summary of the Parquet file's structure.

    Args:
        file_path: Path to the Parquet file

    Returns:
        JSON string with:
        - schema (column names, types, nullable flags)
        - num_rows: Total number of rows
        - num_row_groups: Number of row groups
        - file_size_bytes: Total file size
    """
    try:
        pq_file = pq.ParquetFile(file_path)
        schema = pq_file.schema_arrow
        metadata = pq_file.metadata

        summary = {
            "status": "success",
            "filename": file_path,
            "schema": {
                "columns": [
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": field.nullable,
                    }
                    for field in schema
                ],
            },
            "num_rows": metadata.num_rows,
            "num_row_groups": metadata.num_row_groups,
            "file_size_bytes": os.path.getsize(file_path),
        }

        return json.dumps(summary, indent=2)

    except FileNotFoundError:
        return json.dumps(
            {"status": "error", "message": f"File not found: {file_path}"}
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "message": f"Error summarizing Parquet file: {str(e)}"}
        )


def _estimate_slice_size(
    pq_file: pq.ParquetFile,
    start_row: int,
    end_row: int,
    columns: Optional[List[str]] = None,
) -> int:
    """
    Estimate the JSON serialized size of a slice without reading it.

    Args:
        pq_file: ParquetFile object
        start_row: Starting row index (inclusive)
        end_row: Ending row index (exclusive)
        columns: Optional list of columns to include

    Returns:
        Estimated size in bytes of JSON-serialized output
    """
    metadata = pq_file.metadata
    total_rows = metadata.num_rows

    # Validate bounds
    if start_row < 0 or end_row > total_rows or start_row >= end_row:
        return 0

    # Get schema to filter columns
    schema = pq_file.schema_arrow
    if columns is None:
        columns = [field.name for field in schema]
    else:
        # Validate columns exist
        valid_cols = {field.name for field in schema}
        columns = [c for c in columns if c in valid_cols]

    # Find row groups that overlap with [start_row, end_row)
    uncompressed_size = 0
    current_row = 0

    for rg_idx in range(metadata.num_row_groups):
        rg = metadata.row_group(rg_idx)
        rg_rows = rg.num_rows
        rg_start = current_row
        rg_end = current_row + rg_rows

        # Check if this row group overlaps with our range
        if rg_end > start_row and rg_start < end_row:
            # Calculate overlap
            overlap_start = max(rg_start, start_row)
            overlap_end = min(rg_end, end_row)
            overlap_rows = overlap_end - overlap_start

            # Sum uncompressed sizes for selected columns
            for col_idx in range(rg.num_columns):
                col = rg.column(col_idx)
                col_name = schema[col_idx].name
                if col_name in columns:
                    # Use uncompressed size as base (more accurate than compressed)
                    col_size = col.total_uncompressed_size
                    uncompressed_size += int((overlap_rows / rg_rows) * col_size)

        current_row = rg_end

    # Return the uncompressed binary size as estimate
    # This is conservative: actual JSON will be larger, but the final payload size
    # check will catch any oversized responses before sending to the client
    return uncompressed_size


async def read_slice(
    file_path: str,
    start_row: int,
    end_row: int,
    columns: Optional[List[str]] = None,
    filter_json: Optional[str] = None,
) -> str:
    """
    Read a horizontal slice of a Parquet file with optional column projection and filtering.

    Args:
        file_path: Path to the Parquet file
        start_row: Starting row index (inclusive, 0-based)
        end_row: Ending row index (exclusive)
        columns: Optional list of column names to include (all if None)
        filter_json: Optional JSON string with filter specification

    Filter format:
        Simple: {"column": "zenith", "op": "less", "value": 0.5}
        Compound: {"and": [{"column": "zenith", "op": "less", "value": 0.5},
                           {"column": "batch_id", "op": "equal", "value": 1}]}

    Returns:
        JSON string with status, schema, data, and shape
    """
    try:
        # Validate and coerce start_row and end_row types
        if isinstance(start_row, str):
            try:
                start_row = int(start_row)
            except ValueError:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"start_row must be an integer, got: '{start_row}'",
                    }
                )
        elif isinstance(start_row, float):
            # Convert float to int (common when parameters come from JSON)
            start_row = int(start_row)
        elif not isinstance(start_row, int):
            return json.dumps(
                {
                    "status": "error",
                    "message": f"start_row must be an integer, got type: {type(start_row).__name__}",
                }
            )

        if isinstance(end_row, str):
            try:
                end_row = int(end_row)
            except ValueError:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"end_row must be an integer, got: '{end_row}'",
                    }
                )
        elif isinstance(end_row, float):
            # Convert float to int (common when parameters come from JSON)
            end_row = int(end_row)
        elif not isinstance(end_row, int):
            return json.dumps(
                {
                    "status": "error",
                    "message": f"end_row must be an integer, got type: {type(end_row).__name__}",
                }
            )

        pq_file = pq.ParquetFile(file_path)
        metadata = pq_file.metadata
        total_rows = metadata.num_rows

        # Validate input parameters
        if start_row < 0:
            return json.dumps(
                {
                    "status": "error",
                    "message": "start_row must be >= 0",
                    "suggestion": f"Total rows available: {total_rows}",
                }
            )

        if end_row > total_rows:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"end_row ({end_row}) exceeds total rows ({total_rows})",
                    "suggestion": f"Use end_row <= {total_rows}",
                }
            )

        if start_row >= end_row:
            return json.dumps(
                {"status": "error", "message": "start_row must be less than end_row"}
            )

        # Validate columns if specified
        schema = pq_file.schema_arrow
        available_columns = {field.name for field in schema}

        if columns is not None:
            invalid_cols = [c for c in columns if c not in available_columns]
            if invalid_cols:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid columns: {invalid_cols}",
                        "available_columns": list(available_columns),
                        "suggestion": "Check column names and try again",
                    }
                )

        # Parse filter if provided
        filter_dict = None
        if filter_json:
            try:
                filter_dict = json.loads(filter_json)
            except json.JSONDecodeError:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Invalid filter JSON format",
                        "suggestion": "Provide valid JSON filter specification",
                    }
                )

        # Read the slice
        num_rows = end_row - start_row
        # First read the full table with column filtering
        table = pq.read_table(file_path, columns=columns)
        # Then slice to get the requested row range
        table = table.slice(offset=start_row, length=num_rows)

        # Apply filter if provided
        table = _apply_filter(table, filter_dict)
        rows_after_filter = len(table)

        # Convert to JSON-serializable format
        data = table.to_pylist()

        # Final size check (actual)
        max_size_bytes = 16384  # 16KB limit
        json_str = json.dumps(data)
        actual_size = len(json_str.encode("utf-8"))

        if actual_size > max_size_bytes:
            # Calculate suggested safe size based on actual payload
            rows_requested = end_row - start_row
            # Use rows_after_filter for bytes_per_row calculation since that's what's in the payload
            bytes_per_row = (
                actual_size / rows_after_filter
                if rows_after_filter > 0
                else actual_size / rows_requested
            )
            suggested_safe_rows = int(
                (max_size_bytes / bytes_per_row) * 0.9
            )  # 90% safety margin
            suggested_end_row = start_row + suggested_safe_rows

            return json.dumps(
                {
                    "status": "error",
                    "message": f"Actual payload exceeds limit: {actual_size} bytes (limit: {max_size_bytes})",
                    "suggestion": f"Try a smaller slice. Suggested: rows {start_row} to {suggested_end_row} ({suggested_safe_rows} rows)",
                    "metadata": {
                        "actual_size_bytes": actual_size,
                        "limit_bytes": max_size_bytes,
                        "rows_requested": rows_requested,
                        "rows_after_filter": rows_after_filter,
                        "bytes_per_row": round(bytes_per_row, 2),
                        "suggested_safe_rows": suggested_safe_rows,
                        "suggested_slice": {
                            "start_row": start_row,
                            "end_row": suggested_end_row,
                            "num_rows": suggested_safe_rows,
                        },
                    },
                }
            )

        # Build response
        response = {
            "status": "success",
            "file_path": file_path,
            "slice_info": {
                "start_row": start_row,
                "end_row": end_row,
                "requested_rows": num_rows,
                "rows_after_filter": rows_after_filter,
            },
            "schema": {
                "columns": [
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": field.nullable,
                    }
                    for field in table.schema
                ]
            },
            "data": data,
            "shape": {"rows": len(data), "columns": len(table.column_names)},
            "metadata": {
                "payload_size_bytes": actual_size,
                "total_rows_in_file": total_rows,
                "filter_applied": filter_dict is not None,
            },
        }

        if filter_dict is not None:
            response["metadata"]["filter_spec"] = filter_dict

        return json.dumps(response, indent=2)

    except FileNotFoundError:
        return json.dumps(
            {"status": "error", "message": f"File not found: {file_path}"}
        )
    except Exception as e:
        return json.dumps(
            {
                "status": "error",
                "message": f"Error reading slice from Parquet file: {str(e)}",
            }
        )


async def get_column_preview(
    file_path: str, column_name: str, start_index: int = 0, max_items: int = 100
) -> str:
    """
    Get a preview of values from a specific column.

    Args:
        file_path: Path to the Parquet file
        column_name: Name of the column to preview
        start_index: Starting index for pagination (default: 0)
        max_items: Maximum number of items to return (default: 100, max: 100)

    Returns:
        JSON string with:
        - On success: status, column_name, column_type, data (array), pagination info, metadata
        - On error: status, message, suggestion, and metadata with available_from_start_index
          (number of rows available from the requested start_index onwards)
    """
    try:
        pq_file = pq.ParquetFile(file_path)
        metadata = pq_file.metadata
        total_rows = metadata.num_rows
        schema = pq_file.schema_arrow

        # Validate column exists
        available_columns = {field.name for field in schema}
        if column_name not in available_columns:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Column not found: {column_name}",
                    "available_columns": list(available_columns),
                    "suggestion": "Check column name and try again",
                }
            )

        # Validate pagination parameters
        if start_index < 0:
            return json.dumps(
                {"status": "error", "message": "start_index must be >= 0"}
            )

        if start_index >= total_rows:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"start_index ({start_index}) exceeds total rows ({total_rows})",
                    "suggestion": f"Use start_index < {total_rows}",
                }
            )

        # Constrain max_items to 100
        max_items = min(max_items, 100)
        if max_items < 1:
            max_items = 1

        # Calculate end index
        end_index = min(start_index + max_items, total_rows)

        # Read the column data
        table = pq.read_table(file_path, columns=[column_name])
        column_data = table.column(column_name)

        # Slice to get the requested range
        column_slice = column_data.slice(
            offset=start_index, length=end_index - start_index
        )

        # Convert to Python list for JSON serialization
        data = column_slice.to_pylist()

        # Check payload size
        json_str = json.dumps(data)
        payload_size = len(json_str.encode("utf-8"))
        max_size_bytes = 16384  # 16KB limit

        if payload_size > max_size_bytes:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Payload exceeds limit: {payload_size} bytes (limit: {max_size_bytes})",
                    "suggestion": "Try reducing max_items",
                    "metadata": {
                        "column_name": column_name,
                        "total_values": total_rows,
                        "available_from_start_index": total_rows - start_index,
                        "payload_size_bytes": payload_size,
                        "limit_bytes": max_size_bytes,
                        "recommended_max_items": max(
                            1,
                            int(
                                (max_size_bytes / payload_size)
                                * (end_index - start_index)
                            ),
                        ),
                    },
                }
            )

        # Get column type info
        col_field = next((f for f in schema if f.name == column_name), None)
        col_type = str(col_field.type) if col_field else "unknown"

        # Build response
        response = {
            "status": "success",
            "file_path": file_path,
            "column_name": column_name,
            "column_type": col_type,
            "data": data,
            "pagination": {
                "start_index": start_index,
                "end_index": end_index,
                "num_items": len(data),
                "total_values": total_rows,
                "has_more": end_index < total_rows,
            },
            "metadata": {
                "payload_size_bytes": payload_size,
                "limit_bytes": max_size_bytes,
            },
        }

        return json.dumps(response, indent=2)

    except FileNotFoundError:
        return json.dumps(
            {"status": "error", "message": f"File not found: {file_path}"}
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "message": f"Error getting column preview: {str(e)}"}
        )


async def aggregate_column(
    file_path: str,
    column_name: str,
    operation: str,
    filter_json: Optional[str] = None,
    start_row: Optional[int] = None,
    end_row: Optional[int] = None,
) -> str:
    """
    Compute aggregate statistics on a column with optional filtering and range bounds.

    Args:
        file_path: Path to the Parquet file
        column_name: Name of the column to aggregate
        operation: Aggregation operation (min, max, mean, sum, count, std, count_distinct)
        filter_json: Optional JSON string with filter specification
        start_row: Optional starting row index for range constraint
        end_row: Optional ending row index for range constraint

    Returns:
        JSON string with aggregation result and metadata
    """
    try:
        pq_file = pq.ParquetFile(file_path)
        metadata = pq_file.metadata
        total_rows = metadata.num_rows
        schema = pq_file.schema_arrow

        # Validate column exists
        available_columns = {field.name for field in schema}
        if column_name not in available_columns:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Column not found: {column_name}",
                    "available_columns": list(available_columns),
                    "suggestion": "Check column name and try again",
                }
            )

        # Validate operation
        valid_operations = [
            "min",
            "max",
            "mean",
            "sum",
            "count",
            "std",
            "count_distinct",
        ]
        if operation not in valid_operations:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Invalid operation: {operation}",
                    "valid_operations": valid_operations,
                    "suggestion": "Use one of the supported aggregation operations",
                }
            )

        # Validate and coerce start_row and end_row types
        if start_row is not None:
            if isinstance(start_row, str):
                try:
                    start_row = int(start_row)
                except ValueError:
                    return json.dumps(
                        {
                            "status": "error",
                            "message": f"start_row must be an integer, got: '{start_row}'",
                        }
                    )
            elif isinstance(start_row, float):
                # Convert float to int (common when parameters come from JSON)
                start_row = int(start_row)
            elif not isinstance(start_row, int):
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"start_row must be an integer, got type: {type(start_row).__name__}",
                    }
                )

        if end_row is not None:
            if isinstance(end_row, str):
                try:
                    end_row = int(end_row)
                except ValueError:
                    return json.dumps(
                        {
                            "status": "error",
                            "message": f"end_row must be an integer, got: '{end_row}'",
                        }
                    )
            elif isinstance(end_row, float):
                # Convert float to int (common when parameters come from JSON)
                end_row = int(end_row)
            elif not isinstance(end_row, int):
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"end_row must be an integer, got type: {type(end_row).__name__}",
                    }
                )

        # Parse filter if provided
        filter_dict = None
        if filter_json:
            try:
                filter_dict = json.loads(filter_json)
            except json.JSONDecodeError:
                return json.dumps(
                    {
                        "status": "error",
                        "message": "Invalid filter JSON format",
                        "suggestion": "Provide valid JSON filter specification",
                    }
                )

        # Read table (full or range)
        if start_row is not None and end_row is not None:
            if start_row < 0 or end_row > total_rows or start_row >= end_row:
                return json.dumps(
                    {
                        "status": "error",
                        "message": f"Invalid range: start_row={start_row}, end_row={end_row}",
                        "suggestion": f"Use 0 <= start_row < end_row <= {total_rows}",
                    }
                )
            table = pq.read_table(file_path)
            table = table.slice(offset=start_row, length=end_row - start_row)
        else:
            table = pq.read_table(file_path)

        rows_before_filter = len(table)

        # Apply filter if provided
        table = _apply_filter(table, filter_dict)
        rows_after_filter = len(table)

        if rows_after_filter == 0:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No rows remain after filtering",
                    "metadata": {
                        "rows_before_filter": rows_before_filter,
                        "rows_after_filter": 0,
                        "filter_applied": filter_dict is not None,
                    },
                }
            )

        # Get column data
        column = table[column_name]

        # Compute aggregation
        try:
            if operation == "min":
                result = pc.min(column).as_py()
            elif operation == "max":
                result = pc.max(column).as_py()
            elif operation == "mean":
                result = pc.mean(column).as_py()
            elif operation == "sum":
                result = pc.sum(column).as_py()
            elif operation == "count":
                result = pc.count(column).as_py()
            elif operation == "std":
                result = pc.stddev(column).as_py()
            elif operation == "count_distinct":
                result = len(pc.unique(column))
            else:
                result = None

        except Exception as e:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"Error computing {operation}: {str(e)}",
                    "suggestion": f"Ensure column '{column_name}' has appropriate data type for {operation}",
                }
            )

        # Get column type
        col_field = next((f for f in schema if f.name == column_name), None)
        col_type = str(col_field.type) if col_field else "unknown"

        # Build response
        response = {
            "status": "success",
            "file_path": file_path,
            "column_name": column_name,
            "operation": operation,
            "result": result,
            "metadata": {
                "rows_processed": rows_after_filter,
                "rows_before_filter": rows_before_filter,
                "filter_applied": filter_dict is not None,
                "null_count": pc.count(column, mode="only_null").as_py(),
                "data_type": col_type,
            },
        }

        if filter_dict is not None:
            response["metadata"]["filter_spec"] = filter_dict

        if start_row is not None and end_row is not None:
            response["metadata"]["range_constraint"] = {
                "start_row": start_row,
                "end_row": end_row,
            }

        return json.dumps(response, indent=2)

    except FileNotFoundError:
        return json.dumps(
            {"status": "error", "message": f"File not found: {file_path}"}
        )
    except Exception as e:
        return json.dumps(
            {"status": "error", "message": f"Error computing aggregation: {str(e)}"}
        )
