"""FastMCP server for Apache Parquet files."""

from typing import Optional, List, Union
from fastmcp import FastMCP
from parquet_mcp.capabilities.parquet_handler import (
    summarize,
    read_slice,
    get_column_preview,
    aggregate_column,
)

mcp = FastMCP("parquet-mcp")


@mcp.tool(
    description="Return structured JSON with Parquet schema, row count, and file size"
)
async def summarize_tool(file_path: str) -> str:
    """
    Summarize a Parquet file's structure and metadata.

    Args:
        file_path: Path to the Parquet file

    Returns:
        JSON string with schema, row count, row groups, and file size
    """
    return await summarize(file_path)


@mcp.tool(
    description="Read a horizontal slice of a Parquet file with optional column projection and filtering"
)
async def read_slice_tool(
    file_path: str,
    start_row: int,
    end_row: int,
    columns: Optional[List[str]] = None,
    filter_json: Optional[str] = None,
) -> str:
    """
    Read a specific range of rows from a Parquet file with optional column filtering and row filtering.

    Args:
        file_path: Path to the Parquet file
        start_row: Starting row index (inclusive, 0-based)
        end_row: Ending row index (exclusive)
        columns: Optional list of column names to include (all columns if None)
        filter_json: Optional JSON string with filter specification
                     Example: '{"column": "zenith", "op": "less", "value": 0.5}'

    Returns:
        JSON string with status, schema, data, and shape information
    """
    return await read_slice(file_path, start_row, end_row, columns, filter_json)


@mcp.tool(
    description="Get a preview of values from a specific column with pagination support"
)
async def get_column_preview_tool(
    file_path: str, column_name: str, start_index: int = 0, max_items: int = 100
) -> str:
    """
    Get a preview of values from a named column in a Parquet file.

    Args:
        file_path: Path to the Parquet file
        column_name: Name of the column to preview
        start_index: Starting index for pagination (default: 0)
        max_items: Maximum number of items to return (default: 100, max: 100)

    Returns:
        JSON string with column values, type info, and pagination metadata
    """
    return await get_column_preview(file_path, column_name, start_index, max_items)


@mcp.tool(
    description="Compute aggregate statistics on a column with optional filtering"
)
async def aggregate_column_tool(
    file_path: str,
    column_name: str,
    operation: str,
    filter_json: Optional[str] = None,
    start_row: Optional[Union[int, float]] = None,
    end_row: Optional[Union[int, float]] = None,
) -> str:
    """
    Compute aggregate statistics on a column with optional filtering and range bounds.

    Args:
        file_path: Path to the Parquet file
        column_name: Name of the column to aggregate
        operation: Aggregation operation (min, max, mean, sum, count, std, count_distinct)
        filter_json: Optional JSON string with filter specification
                     Example: '{"column": "batch_id", "op": "equal", "value": 1}'
        start_row: Optional starting row index for range constraint
        end_row: Optional ending row index for range constraint

    Returns:
        JSON string with aggregation result and metadata
    """
    return await aggregate_column(
        file_path, column_name, operation, filter_json, start_row, end_row
    )


def main():
    """Start the Parquet MCP server."""
    import sys
    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    try:
        # Run with stdio transport (default for MCP)
        logger.info("Starting Parquet MCP server with stdio transport")
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
