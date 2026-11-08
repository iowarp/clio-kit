"""Export selection and value formatting for tool responses."""

import logging
from typing import Any, Literal

import numpy as np
from fastmcp import Context
from fastmcp.exceptions import ToolError

logger = logging.getLogger(__name__)

# ponytail: fixed inline-response bound; make it a setting if clients need more.
MAX_VALUES = 1000


def json_safe(value: Any) -> Any:
    """Convert HDF5/NumPy values to strict-JSON types; NaN and infinity become null."""
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        if array.dtype.kind == "f":
            finite = np.isfinite(array)
            array = array.astype(object)
            array[~finite] = None
        elif array.dtype.kind == "S":
            array = np.char.decode(array, "utf-8", "replace")
        return array.tolist()
    return value if isinstance(value, (str, int, float, bool)) else str(value)


def format_values(data: Any) -> str:
    """Render values for a tool response, stating any truncation beyond MAX_VALUES."""
    array = np.asarray(data)
    if array.size <= MAX_VALUES:
        return f"Values: {array.tolist()}"
    return (
        f"Values (first {MAX_VALUES} of {array.size} in row-major order, "
        f"{array.size - MAX_VALUES} omitted; use read_partial_dataset for a slice): "
        f"{array.flat[:MAX_VALUES].tolist()}"
    )


async def select_export_format(
    ctx: Context | None, requested_format: Literal["csv", "json", "numpy"] | None
) -> Literal["csv", "json", "numpy"] | None:
    """Return a format, or None when a legacy caller declines the export."""
    # Modern MCP has no mid-call back-channel. Its caller supplies the format
    # explicitly; retain optional elicitation for legacy connections only.
    export_format = requested_format or "json"
    protocol_version = (
        getattr(ctx.request_context, "protocol_version", None) if ctx else None
    )
    if ctx and requested_format is None and protocol_version != "2026-07-28":
        try:
            format_result = await ctx.elicit(
                "What format should I export to?",
                response_type=["csv", "json", "numpy"],  # type: ignore[arg-type]
            )

            if format_result.action == "accept":
                export_format = format_result.data  # type: ignore[assignment]
            elif format_result.action == "decline":
                return None
            else:  # cancel
                raise ToolError("Export cancelled")
        except ToolError:
            raise
        except ValueError as e:
            logger.debug(f"Elicitation not supported: {e}")
        except Exception as e:
            logger.warning(f"Error during elicitation: {e}")
            import traceback

            logger.debug(traceback.format_exc())
            # Fall back to default format
            export_format = "json"

    return export_format
