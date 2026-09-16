"""Export selection for modern and legacy MCP connections."""

import logging
from typing import Literal

from fastmcp import Context
from fastmcp.exceptions import ToolError

logger = logging.getLogger(__name__)


async def select_export_format(
    ctx: Context | None, requested_format: Literal["csv", "json", "numpy"] | None
) -> str | None:
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
