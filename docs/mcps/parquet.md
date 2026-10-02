---
title: Parquet MCP
description: "MCP server for Apache Parquet files"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Parquet"
  icon="📋"
  category="Data Processing"
  description="MCP server for Apache Parquet files"
  version="2.2.5"
  actions={["summarize_tool", "read_slice_tool", "get_column_preview_tool", "aggregate_column_tool"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["parquet", "columnar-data", "data-analysis", "scientific-computing", "mcp", "llm-integration", "apache-arrow"]}
  license="BSD-3-Clause"
  tools={[{"name": "summarize_tool", "description": "Return Parquet schema, row count, and file size.", "function_name": "summarize_tool"}, {"name": "read_slice_tool", "description": "Read a row slice from a Parquet file with optional column projection and filtering.", "function_name": "read_slice_tool"}, {"name": "get_column_preview_tool", "description": "Preview values from a specific column with pagination.", "function_name": "get_column_preview_tool"}, {"name": "aggregate_column_tool", "description": "Compute aggregate statistics (min, max, mean, etc.) on a Parquet column.", "function_name": "aggregate_column_tool"}]}
>

{/* clio-kit:usage:start */}

### Inspect and summarize a Parquet file

Use `summarize_tool` to inspect schema and row count before reading values.
Select a column with `get_column_preview_tool`, or use `read_slice_tool` for a
bounded row range with column projection. Use `aggregate_column_tool` to compute
supported statistics without returning the complete column to the agent.

Paths refer to files accessible to the MCP process. Inspect the live input
schema for the accepted filter and aggregation arguments, and check the result
for errors or empty selections before interpreting it.

Invalid operators, missing columns, incompatible values and malformed nested
filters fail the request. They never return unfiltered data as a successful
filtered result. Omit `filter_json` or pass an empty string for no filter.

{/* clio-kit:usage:end */}

</MCPDetail>
