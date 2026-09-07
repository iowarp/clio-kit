"""MRTR size-guard for data-returning tools (iowarp/clio-agent #1325 C2).

A data-returning tool (one whose PURPOSE is to hand rows/records directly to
the model — NOT a file→file op) calls :func:`guard_records_payload` with its
computed result. The guard gates on three things:

1. **Budget** — is the payload small enough to return without overflowing a
   local model's context?  If yes, returns the full result unchanged.
2. **Negotiation signal** — ``ctx.client_supports_extension(
   AGENT_ELICITATION_EXTENSION_ID)`` — *before* returning an
   ``InputRequiredResult``.  A client that cannot drive the MRTR loop
   (legacy or non-clio) is never handed one; it gets the full result instead,
   with a note.
3. **Re-invocation** — on the second round ``ctx.input_responses`` is not
   ``None``; the guard reads the agent's narrowing answer from it (plus the
   opaque ``request_state`` token it minted on round 1), applies the narrowing
   to the full frame, and returns only the bounded result.

Transport
---------
Uses the **InputRequiredResult guard pattern** (SEP-2322 / fastmcp >= 4.0):

- Round 1: return ``mcp_types.InputRequiredResult`` carrying a single
  ``ElicitRequest`` whose ``_meta`` is tagged ``{"x-clio-agent/audience":
  "agent"}`` so clio routes the question to the session's own agent (not the
  human).  ``requestState`` carries a JSON token clio echoes back verbatim.
- Round 2: ``ctx.input_responses["narrow"]`` is an ``ElicitResult``; its
  ``content`` carries the agent's narrowing choices.

Do NOT use ``ctx.elicit()`` — it raises ``ToolError`` on the 2026-07-28
modern protocol era (SEP-2577, ``_ELICIT_MODERN_ERROR`` in
``fastmcp/server/context.py``).

Extension constant
------------------
``AGENT_ELICITATION_EXTENSION_ID`` mirrors
``clio_agent.tools.mcp_extension_registry.AGENT_ELICITATION_EXTENSION_ID``
(``src/clio_agent/tools/mcp_extension_registry.py``). Do NOT import from
clio-agent — separate package; keep the constant local and in sync manually.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pandas as pd
from mcp import types as mcp_types

# ---------------------------------------------------------------------------
# Extension / audience constants
# ---------------------------------------------------------------------------

# Mirrors clio_agent.tools.mcp_extension_registry.AGENT_ELICITATION_EXTENSION_ID
# (src/clio_agent/tools/mcp_extension_registry.py).  Keep in sync manually —
# do NOT import from clio-agent (separate package).
AGENT_ELICITATION_EXTENSION_ID = "x-clio-agent/agent-driven-elicitation"

# _meta key + value clio reads to route an elicitation to the AGENT
# (clio_agent.gact.agent_elicitation.AGENT_AUDIENCE_META_KEY / _VALUE).
_AUDIENCE_META_KEY = "x-clio-agent/audience"
_AUDIENCE_META_VALUE = "agent"

# Key name used in the InputRequiredResult.input_requests dict (and therefore
# the key used to read ctx.input_responses on round 2).
_INPUT_REQUEST_KEY = "narrow"

# ---------------------------------------------------------------------------
# Budget defaults
# ---------------------------------------------------------------------------

# ~8 000 tokens × 4 chars/token = 32 000 chars — conservative enough to fit
# any local model without overflowing while still generous for typical results.
DEFAULT_MAX_TOKENS: int = 8_000
_CHARS_PER_TOKEN: int = 4


# ---------------------------------------------------------------------------
# Internal dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SizeEstimate:
    """Cheap, honest size estimate of a records payload."""

    rows: int
    cols: int
    serialized_bytes: int
    est_tokens: int

    def as_dict(self) -> dict[str, int]:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "serialized_bytes": self.serialized_bytes,
            "est_tokens": self.est_tokens,
        }


@dataclass
class _NarrowingAnswer:
    """Parsed narrowing answer from the agent's ElicitResult.content."""

    columns: str = ""
    top_n: int = 0
    query: str = ""
    group_by: str = ""
    agg: str = ""


# ---------------------------------------------------------------------------
# Public helpers — size estimation and narrowing
# ---------------------------------------------------------------------------


def estimate_records_size(records: list[dict[str, Any]]) -> SizeEstimate:
    """Estimate the payload size of a to-be-returned records list.

    Args:
        records: The list of row dicts the tool would return.

    Returns:
        A :class:`SizeEstimate` with rows, cols, serialized_bytes, est_tokens.
    """
    rows = len(records)
    cols = len(records[0]) if rows else 0
    try:
        serialized = json.dumps(records, default=str)
    except (TypeError, ValueError):
        serialized = str(records)
    serialized_bytes = len(serialized.encode("utf-8"))
    est_tokens = serialized_bytes // _CHARS_PER_TOKEN
    return SizeEstimate(
        rows=rows,
        cols=cols,
        serialized_bytes=serialized_bytes,
        est_tokens=est_tokens,
    )


def apply_narrowing(
    df: pd.DataFrame,
    narrowing: _NarrowingAnswer,
    fallback_top_n: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Apply an agent's narrowing answer to the full frame.

    Args:
        df: The full (untruncated) DataFrame.
        narrowing: Parsed agent answer.
        fallback_top_n: Hard row cap used when the agent supplied no explicit
            ``top_n`` and no aggregation — ensures the result is always bounded.

    Returns:
        A ``(frame, applied_ops)`` tuple where ``applied_ops`` documents what
        was done (for the ``applied`` field in the guard block).
    """
    applied: list[str] = []
    result = df

    if narrowing.query.strip():
        try:
            result = result.query(narrowing.query)
            applied.append(f"query({narrowing.query!r})")
        except Exception as exc:  # noqa: BLE001
            applied.append(f"query_failed({exc})")

    if narrowing.columns.strip():
        wanted = [c.strip() for c in narrowing.columns.split(",") if c.strip()]
        present = [c for c in wanted if c in result.columns]
        if present:
            result = result[present]
            applied.append(f"columns({present})")

    if narrowing.group_by.strip() and narrowing.agg.strip():
        keys = [c.strip() for c in narrowing.group_by.split(",") if c.strip()]
        keys = [c for c in keys if c in result.columns]
        if keys:
            try:
                result = getattr(result.groupby(keys), narrowing.agg)()
                result = result.reset_index()
                applied.append(f"groupby({keys}).{narrowing.agg}()")
                return result, applied  # aggregated frames are already small
            except Exception as exc:  # noqa: BLE001
                applied.append(f"groupby_failed({exc})")

    effective_top_n = narrowing.top_n if narrowing.top_n > 0 else fallback_top_n
    if len(result) > effective_top_n:
        result = result.head(effective_top_n)
        applied.append(f"top_n({effective_top_n})")

    return result, applied


# ---------------------------------------------------------------------------
# Core guard — the one-liner any data-returning tool calls
# ---------------------------------------------------------------------------


async def guard_records_payload(
    *,
    ctx: Any,
    full_df: pd.DataFrame,
    records: list[dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict[str, Any] | mcp_types.InputRequiredResult:
    """Size-guard a records payload; request agent narrowing via MRTR if needed.

    Call this from any data-returning tool *instead of* embedding the result
    directly.  The tool must be declared ``async``; ``ctx`` must be the FastMCP
    ``Context`` injected by the framework.

    Decision tree
    -------------
    * ``ctx.input_responses`` is not ``None`` (round 2):
        Read the agent's narrowing from ``ctx.input_responses[_INPUT_REQUEST_KEY]``,
        recover the token from ``ctx.request_state``, apply the narrowing,
        return the bounded result dict.
    * Within budget (round 1):
        Return the full result dict unchanged (``guarded=False``).
    * Over budget + extension absent (round 1):
        Return the full result dict with a note (``guarded=True``,
        ``action="extension_absent_full_returned"``).  The tool stays universal.
    * Over budget + extension present (round 1):
        Return an ``mcp_types.InputRequiredResult`` carrying a single
        ``ElicitRequest`` tagged ``audience=agent`` so clio routes to the
        session's own agent.

    Args:
        ctx: FastMCP ``Context`` (injected by the framework).
        full_df: The full, untruncated DataFrame (narrowing is applied to this,
            not to the pre-serialized ``records``).
        records: The payload the tool would otherwise return — used only for
            the size estimate on round 1.
        max_tokens: Token budget (default ~8 000 tokens ≈ 32 000 chars).

    Returns:
        On round 1 within-budget or extension-absent: a plain ``dict``.
        On round 1 over-budget + extension-present: ``InputRequiredResult``.
        On round 2: a plain ``dict`` with the narrowed ``records``.
    """
    # --- Round 2: agent has answered — apply narrowing and return. ----------
    if ctx is not None and ctx.input_responses is not None:
        return _apply_round2(ctx, full_df, max_tokens)

    # --- Round 1: compute budget. ------------------------------------------
    estimate = estimate_records_size(records)
    within_budget = estimate.est_tokens <= max_tokens

    if within_budget:
        return {
            "guarded": False,
            "estimate": estimate.as_dict(),
            "records": records,
        }

    # Over budget — check the negotiation signal.
    extension_present = (
        ctx is not None and ctx.client_supports_extension(AGENT_ELICITATION_EXTENSION_ID)
    )

    if not extension_present:
        # Generic / non-clio client: return the full result, with a typed note.
        # Never hand an InputRequiredResult to a client that can't drive the loop.
        return {
            "guarded": True,
            "action": "extension_absent_full_returned",
            "note": (
                f"Result is ~{estimate.est_tokens} tokens (budget {max_tokens}); "
                f"client did not advertise {AGENT_ELICITATION_EXTENSION_ID!r} so "
                "the full payload is returned."
            ),
            "estimate": estimate.as_dict(),
            "records": records,
        }

    # Extension present → return InputRequiredResult.
    return _build_input_required(estimate, full_df, max_tokens)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_input_required(
    estimate: SizeEstimate,
    full_df: pd.DataFrame,
    max_tokens: int,
) -> mcp_types.InputRequiredResult:
    """Construct the InputRequiredResult for round-1 over-budget + extension present."""
    cols = list(full_df.columns)

    message = (
        f"This result is too large to return in full without overflowing the "
        f"model context: {estimate.rows} rows × {estimate.cols} columns "
        f"(~{estimate.serialized_bytes:,} bytes, ~{estimate.est_tokens:,} tokens; "
        f"budget is {max_tokens:,} tokens). "
        f"Available columns: {cols}. "
        "Tell me how to narrow it by filling in ONE OR MORE of the fields below:\n"
        "  • top_n — keep only the first N rows (integer > 0)\n"
        "  • columns — comma-separated column subset to keep\n"
        "  • query — a pandas df.query() filter string (e.g. 'age > 40')\n"
        "  • group_by + agg — aggregate (e.g. group_by='dept', agg='count')\n"
        "I will apply it and return only the bounded result."
    )

    # Flat requestedSchema: all optional primitives (MCP elicitation-legal).
    requested_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "top_n": {
                "type": "integer",
                "description": "Keep only the first N rows (0 = use default cap).",
                "minimum": 0,
            },
            "columns": {
                "type": "string",
                "description": "Comma-separated column names to keep (empty = all).",
            },
            "query": {
                "type": "string",
                "description": "pandas df.query() filter string (empty = no filter).",
            },
            "group_by": {
                "type": "string",
                "description": (
                    "Comma-separated columns to group by (requires agg). "
                    "Empty = no aggregation."
                ),
            },
            "agg": {
                "type": "string",
                "description": (
                    "Aggregation function: count, sum, mean, min, max, std. "
                    "Only used when group_by is set."
                ),
            },
        },
        "additionalProperties": False,
    }

    elicit_params = mcp_types.ElicitRequestFormParams(
        meta={_AUDIENCE_META_KEY: _AUDIENCE_META_VALUE},
        message=message,
        requested_schema=requested_schema,
    )
    elicit_request = mcp_types.ElicitRequest(params=elicit_params)

    # request_state is an opaque string clio echoes back on round 2 via
    # ctx.request_state.  We encode the parameters needed to reconstruct
    # the budget (max_tokens) and the fallback cap.
    #
    # The cap derives from THIS payload's measured tokens-per-row (not a fixed
    # rows-per-token guess), so the decline fallback actually fits the budget
    # regardless of column count.
    tokens_per_row = max(1, estimate.est_tokens // max(1, estimate.rows))
    fallback_top_n = max(1, max_tokens // tokens_per_row)
    state_token = json.dumps(
        {
            "max_tokens": max_tokens,
            "fallback_top_n": fallback_top_n,
            "original_estimate": estimate.as_dict(),
        }
    )

    return mcp_types.InputRequiredResult(
        input_requests={_INPUT_REQUEST_KEY: elicit_request},
        request_state=state_token,
    )


def _apply_round2(
    ctx: Any,
    full_df: pd.DataFrame,
    default_max_tokens: int,
) -> dict[str, Any]:
    """Apply the agent's narrowing on round 2 and return the bounded result."""
    # Recover token from ctx.request_state (may be None if client didn't echo).
    state_token = ctx.request_state
    try:
        state = json.loads(state_token) if state_token else {}
    except (json.JSONDecodeError, TypeError):
        state = {}

    fallback_top_n = int(state.get("fallback_top_n", max(1, default_max_tokens // 10)))
    original_estimate = state.get("original_estimate", {})

    # Read the agent's answer.
    responses = ctx.input_responses  # dict[str, ElicitResult | ...]
    elicit_result = responses.get(_INPUT_REQUEST_KEY) if responses else None

    action = getattr(elicit_result, "action", None) if elicit_result else None
    if action != "accept":
        # Declined / cancelled / missing → hard-bound to fallback_top_n with reason.
        bounded = full_df.head(fallback_top_n).to_dict("records")
        return {
            "guarded": True,
            "action": "hard_truncated",
            "degrade_reason": f"elicitation_{action or 'missing'}",
            "original_estimate": original_estimate,
            "records": bounded,
        }

    narrowing = _parse_narrowing(getattr(elicit_result, "content", None))
    narrowed_df, applied = apply_narrowing(full_df, narrowing, fallback_top_n)
    narrowed_records = narrowed_df.to_dict("records")
    narrowed_estimate = estimate_records_size(narrowed_records)

    return {
        "guarded": True,
        "action": "narrowed_by_agent",
        "original_estimate": original_estimate,
        "narrowed_estimate": narrowed_estimate.as_dict(),
        "applied": applied,
        "records": narrowed_records,
    }


def _parse_narrowing(content: Any) -> _NarrowingAnswer:
    """Coerce ElicitResult.content into a _NarrowingAnswer."""
    if isinstance(content, dict):
        return _NarrowingAnswer(
            columns=str(content.get("columns", "") or ""),
            top_n=int(content.get("top_n", 0) or 0),
            query=str(content.get("query", "") or ""),
            group_by=str(content.get("group_by", "") or ""),
            agg=str(content.get("agg", "") or ""),
        )
    return _NarrowingAnswer()
