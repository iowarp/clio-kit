"""
MCP handlers for Parallel Sort server.
These handlers wrap all implementation for MCP protocol compliance.
"""

from typing import Dict, Any, List, Union, Optional
from fastmcp.exceptions import ToolError
from .implementation.sort_handler import sort_log_by_timestamp
from .implementation.statistics_handler import analyze_log_statistics
from .implementation.pattern_detection import detect_patterns
from .implementation.filter_handler import (
    filter_logs,
    filter_by_time_range,
    filter_by_log_level,
    filter_by_keyword,
    apply_filter_preset,
)
from .implementation.export_handler import (
    export_to_json,
    export_to_csv,
    export_to_text,
    export_summary_report,
)
from .implementation.parallel_processor import parallel_sort_large_file

# Bound inline replies even when no output file is requested.
PREVIEW_LINES = 100
PREVIEW_BYTES = 16_384


def _finish(
    result: Dict[str, Any],
    lines_key: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Raise implementation errors as tool errors; write output_file and bound the reply."""
    if result.get("error"):
        raise ToolError(str(result["error"]))
    if lines_key in result:
        lines = result[lines_key]
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                for line in lines:
                    f.write(line + "\n")
            result["output_file"] = output_file
            result["lines_written"] = len(lines)
        preview = []
        remaining = PREVIEW_BYTES
        for line in lines[:PREVIEW_LINES]:
            encoded = line.encode("utf-8")
            if len(encoded) > remaining:
                if remaining:
                    preview.append(encoded[:remaining].decode("utf-8", errors="ignore"))
                break
            preview.append(line)
            remaining -= len(encoded)
        if preview != lines:
            result[lines_key] = preview
            result["truncated"] = True
            result["lines_returned"] = len(preview)
            result["total_result_lines"] = len(lines)
            result["truncation_note"] = (
                f"{lines_key} is a preview limited to {PREVIEW_LINES} lines and {PREVIEW_BYTES} UTF-8 bytes. "
                + (
                    f"All {len(lines)} lines are in {output_file}."
                    if output_file
                    else "Set output_file to save the full result; it has not been saved."
                )
            )
    return result


async def sort_log_handler(
    file_path: str,
    output_file: Optional[str] = None,
    reverse: bool = False,
) -> Dict[str, Any]:
    """Handler wrapping the log sorting capability for MCP.

    Args:
        file_path: Path to the log file to sort.
        output_file: Path for sorted output file.
        reverse: Sort in descending order.
    """
    try:
        result = await sort_log_by_timestamp(file_path)
        # Apply reverse sorting if requested
        if reverse and "sorted_lines" in result:
            result["sorted_lines"] = list(reversed(result["sorted_lines"]))
        return _finish(result, "sorted_lines", output_file)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"sort_log failed: {e}") from e


async def parallel_sort_handler(
    file_path: str,
    output_file: str,
    chunk_size_mb: int = 100,
    num_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Handler wrapping the parallel sort capability for MCP.

    Args:
        file_path: Path to the large log file.
        output_file: Path for sorted output file.
        chunk_size_mb: Chunk size in MB.
        num_workers: Number of worker processes.
    """
    try:
        result = await parallel_sort_large_file(file_path, chunk_size_mb, num_workers)
        return _finish(result, "sorted_lines", output_file)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"parallel_sort failed: {e}") from e


async def analyze_statistics_handler(
    file_path: str, include_patterns: bool = True
) -> Dict[str, Any]:
    """Handler wrapping the statistics analysis capability for MCP.

    Args:
        file_path: Path to the log file.
        include_patterns: Include pattern analysis.
    """
    try:
        result = await analyze_log_statistics(file_path)
        if not include_patterns and "statistics" in result:
            result["statistics"].pop("message_analysis", None)
        return _finish(result)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"analyze_statistics failed: {e}") from e


async def detect_patterns_handler(
    file_path: str,
    pattern_types: Optional[List[Any]] = None,
    sensitivity: Optional[str] = None,
) -> Dict[str, Any]:
    """Handler wrapping the pattern detection capability for MCP.

    Args:
        file_path: Path to the log file.
        pattern_types: Types of patterns to detect.
        sensitivity: Detection sensitivity ('low', 'medium', 'high').
    """
    try:
        # Build detection config from individual parameters
        detection_config: Optional[Dict[str, Any]] = None
        if pattern_types is not None or sensitivity is not None:
            detection_config = {}
            if pattern_types is not None:
                detection_config["pattern_types"] = pattern_types
            if sensitivity is not None:
                sensitivity_thresholds = {"low": 4.0, "medium": 3.0, "high": 2.0}
                if sensitivity not in sensitivity_thresholds:
                    raise ToolError(
                        f"Unknown sensitivity {sensitivity!r}; use 'low', 'medium' or 'high'"
                    )
                detection_config["anomaly_threshold"] = sensitivity_thresholds[
                    sensitivity
                ]
        result = await detect_patterns(file_path, detection_config)
        return _finish(result)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"detect_patterns failed: {e}") from e


async def filter_logs_handler(
    file_path: str,
    filter_conditions: List[Dict[str, Any]],
    logical_operator: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Handler wrapping the log filtering capability for MCP.

    Args:
        file_path: Path to the log file.
        filter_conditions: List of filter condition dictionaries.
        logical_operator: Logical operator between filters ('AND', 'OR').
        output_file: Path for filtered output.
    """
    try:
        op = logical_operator if logical_operator is not None else "and"
        result = await filter_logs(file_path, filter_conditions, op)
        return _finish(result, "filtered_lines", output_file)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"filter_logs failed: {e}") from e


async def filter_time_range_handler(
    file_path: str,
    start_time: str,
    end_time: str,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Handler wrapping the time range filtering capability for MCP.

    Args:
        file_path: Path to the log file.
        start_time: Start timestamp.
        end_time: End timestamp.
        output_file: Path for filtered output.
    """
    try:
        result = await filter_by_time_range(file_path, start_time, end_time)
        return _finish(result, "filtered_lines", output_file)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"filter_time_range failed: {e}") from e


async def filter_level_handler(
    file_path: str,
    levels: Union[str, List[str]],
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Handler wrapping the log level filtering capability for MCP.

    Args:
        file_path: Path to the log file.
        levels: List of log levels to include.
        output_file: Path for filtered output.
    """
    try:
        result = await filter_by_log_level(file_path, levels)
        return _finish(result, "filtered_lines", output_file)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"filter_level failed: {e}") from e


async def filter_keyword_handler(
    file_path: str,
    keywords: Union[str, List[str]],
    case_sensitive: bool = False,
    logical_operator: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Handler wrapping the keyword filtering capability for MCP.

    Args:
        file_path: Path to the log file.
        keywords: List of keywords to search for.
        case_sensitive: Case sensitive matching.
        logical_operator: Operator between keywords ('AND', 'OR').
        output_file: Path for filtered output.
    """
    try:
        # Map logical_operator to match_all boolean
        if (logical_operator or "OR").upper() not in ("AND", "OR"):
            raise ToolError(
                f"Unknown logical_operator {logical_operator!r}; use 'AND' or 'OR'"
            )
        match_all = (logical_operator or "").upper() == "AND"
        result = await filter_by_keyword(file_path, keywords, case_sensitive, match_all)
        return _finish(result, "filtered_lines", output_file)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"filter_keyword failed: {e}") from e


async def filter_preset_handler(
    file_path: str,
    preset_name: str,
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Handler wrapping the filter preset capability for MCP.

    Args:
        file_path: Path to the log file.
        preset_name: Preset name to apply.
        output_file: Path for filtered output.
    """
    try:
        result = await apply_filter_preset(file_path, preset_name)
        return _finish(result, "filtered_lines", output_file)
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"filter_preset failed: {e}") from e


async def export_json_handler(
    data: Dict[str, Any], include_metadata: bool = True
) -> Dict[str, Any]:
    """Handler wrapping the JSON export capability for MCP."""
    try:
        return _finish(await export_to_json(data, include_metadata))
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"export_json failed: {e}") from e


async def export_csv_handler(
    data: Dict[str, Any], include_headers: bool = True
) -> Dict[str, Any]:
    """Handler wrapping the CSV export capability for MCP."""
    try:
        return _finish(await export_to_csv(data, include_headers))
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"export_csv failed: {e}") from e


async def export_text_handler(
    data: Dict[str, Any], include_summary: bool = True
) -> Dict[str, Any]:
    """Handler wrapping the text export capability for MCP."""
    try:
        return _finish(await export_to_text(data, include_summary))
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"export_text failed: {e}") from e


async def summary_report_handler(data: Dict[str, Any]) -> Dict[str, Any]:
    """Handler wrapping the summary report capability for MCP."""
    try:
        return _finish(await export_summary_report(data))
    except ToolError:
        raise
    except Exception as e:
        raise ToolError(f"summary_report failed: {e}") from e
