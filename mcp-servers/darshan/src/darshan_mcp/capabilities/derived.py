"""Pure derivations over parsed Darshan data (split out of darshan_parser)."""

import re
from typing import Any, Dict, List, Optional

from .native_text import job_runtime

# Field names ending in ``_mbps`` are kept for compatibility; the values are MiB/s.
_UNITS = {
    "bandwidth_mbps": "MiB/s (bytes / 1048576 / seconds)",
    "total_bandwidth_mbps": "total bytes read+written / (read time + write time)",
    "avg_read_bandwidth_mbps / avg_write_bandwidth_mbps": "bytes / job runtime",
}
_COMPARISON_METRICS = ("bandwidth", "iops", "file_count")


def _total_bandwidth_mbps(files: Dict[str, Any]) -> Optional[float]:
    """Total bytes moved per second of I/O time (read time + write time), MiB/s.

    The one definition of ``total_bandwidth_mbps``; None when the log carries
    no read/write timing.
    """
    records = [data for data in files.values() if isinstance(data, dict)]
    io_time = sum(r.get("read_time", 0) + r.get("write_time", 0) for r in records)
    if io_time <= 0:
        return None
    total = sum(r.get("bytes_read", 0) + r.get("bytes_written", 0) for r in records)
    return total / (1024 * 1024) / io_time


def resolution_error(time_resolution: str) -> Optional[str]:
    """Return why ``time_resolution`` is invalid, or None when it is valid."""
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(ms|s|m)", time_resolution.strip())
    if match and float(match.group(1)) > 0:
        return None
    return (
        f"Invalid time_resolution '{time_resolution}': expected a positive number "
        "with unit ms, s or m (e.g. '1s', '100ms')."
    )


def timeline_result(
    parsed_data: Dict[str, Any], time_resolution: str
) -> Dict[str, Any]:
    """The timing a Darshan summary log actually carries.

    Summary logs hold counters, not per-operation timestamps, so activity
    cannot be binned over time; that needs DXT trace data.
    """
    job_info = parsed_data.get("job", {})
    return {
        "success": True,
        "time_resolution": time_resolution,
        "timeline_available": False,
        "message": (
            "Binned I/O activity over time (peak periods, idle periods, I/O "
            "phases) is not available: Darshan summary logs record counters, "
            "not per-operation timestamps. That needs DXT trace data (run the "
            "job with DXT_ENABLE_IO_TRACE=1 and read it with "
            "darshan-dxt-parser). This log does provide the job start, end "
            "and runtime, and per module the first and last "
            "open/read/write/close timestamps in seconds since job start "
            "(analysis.module_activity_windows)."
        ),
        "analysis": {
            "total_duration": job_runtime(job_info),
            "start_time": job_info.get("start_time"),
            "end_time": job_info.get("end_time"),
            "module_activity_windows": parsed_data.get("activity_windows", {}),
            # Not computed (see message); None rather than an empty list so
            # "no data" is not read as "no peaks".
            "peak_periods": None,
            "idle_periods": None,
            "io_phases": None,
        },
    }


def compare_metrics(
    metrics_1: Optional[Dict[str, Any]],
    metrics_2: Optional[Dict[str, Any]],
    comparison_metrics: List[str],
) -> Dict[str, Any]:
    """Differences and a one-word summary per metric, or ``{"error": ...}``.

    Unknown metric names are an error; with ``None`` metrics only the names
    are validated.
    """
    unknown = [m for m in comparison_metrics if m not in _COMPARISON_METRICS]
    if unknown:
        return {
            "error": f"Unknown comparison metric(s): {', '.join(map(str, unknown))}. "
            f"Supported: {', '.join(_COMPARISON_METRICS)}."
        }
    result: Dict[str, Any] = {"differences": {}, "summary": {}}
    if metrics_1 is None or metrics_2 is None:
        return result

    def value(metrics: Dict[str, Any], metric: str) -> float:
        if metric == "file_count":
            return metrics.get("file_count", 0)
        key = "total_bandwidth_mbps" if metric == "bandwidth" else "total_iops"
        return metrics.get("overall_metrics", {}).get(key, 0)

    for metric in comparison_metrics:
        v1, v2 = value(metrics_1, metric), value(metrics_2, metric)
        result["differences"][metric] = {
            "log_1": v1,
            "log_2": v2,
            "difference": v2 - v1,
            "percent_change": ((v2 - v1) / v1 * 100) if v1 > 0 else 0,
        }
        result["summary"][metric] = (
            "unchanged"
            if v2 == v1
            else ("higher" if v2 > v1 else "lower") + " in log_2"
        )
    return result
