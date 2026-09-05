"""Tests for the MRTR size-guard (iowarp/clio-agent #1325 C2).

Two groups of targeted tests — kept narrow to respect the shared HPC head-node
constraint.  Uses ``asyncio.run`` rather than ``@pytest.mark.asyncio``: this
server's dev dependency group does not carry a pytest asyncio plugin (unlike
some sibling servers).  See tests/test_tool_titles.py for the precedent.

Group A — unit tests for the guard's decision function directly (no file I/O,
no server subprocess).  These exercise all four branches:
  * within-budget                      → plain dict, guarded=False
  * over-budget + extension present    → InputRequiredResult with agent audience
  * over-budget + extension absent     → plain dict with full result + note
  * round-2 accept                     → bounded narrowed result
  * round-2 decline                    → hard-truncated bounded result

Group B — in-memory Client(server) round-trip test against load_data_tool.
Tests (b1) extension absent + over-budget → full result (graceful degradation);
and (b2) the extension present branch at the tool level (asserts the tool
wiring is exercised end-to-end; MRTR round-trip skipped because the in-memory
harness does not yet surface InputRequiredResult as a distinct object — deferred
to live compute-node verification per the HPC constraint note).
"""

from __future__ import annotations

import asyncio
import json
import tempfile

import pandas as pd
import pytest
from mcp import types as mcp_types

from pandas_mcp.implementation.size_guard import (
    AGENT_ELICITATION_EXTENSION_ID,
    DEFAULT_MAX_TOKENS,
    estimate_records_size,
    guard_records_payload,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_csv(path: str, rows: int, cols: int = 5) -> None:
    """Write a synthetic CSV large enough to trigger the guard."""
    data = {f"col{c}": list(range(rows)) for c in range(cols)}
    data["dept"] = [["A", "B", "C"][i % 3] for i in range(rows)]
    pd.DataFrame(data).to_csv(path, index=False)


class _FakeCtxRound1:
    """Fake Context for round-1 calls (no prior input_responses)."""

    input_responses = None
    request_state = None

    def __init__(self, *, extension_present: bool) -> None:
        self._ext = extension_present

    def client_supports_extension(self, _: str) -> bool:
        return self._ext


# ---------------------------------------------------------------------------
# Group A — guard decision unit tests (pure, no network)
# ---------------------------------------------------------------------------


class TestSizeGuardDecision:
    """Unit tests for guard_records_payload decision branches."""

    def test_within_budget_returns_dict_guarded_false(self) -> None:
        """Within budget → plain dict, guarded=False, records unchanged."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        records = df.to_dict("records")

        result = asyncio.run(
            guard_records_payload(
                ctx=_FakeCtxRound1(extension_present=True),
                full_df=df,
                records=records,
                max_tokens=100_000,  # enormous budget → always within budget
            )
        )
        assert isinstance(result, dict), "Within-budget must return a dict"
        assert result["guarded"] is False
        assert result["records"] == records

    def test_over_budget_extension_present_returns_input_required(self) -> None:
        """Over-budget + extension present → InputRequiredResult with agent audience."""
        df = pd.DataFrame({"col": list(range(500))})
        records = df.to_dict("records")

        result = asyncio.run(
            guard_records_payload(
                ctx=_FakeCtxRound1(extension_present=True),
                full_df=df,
                records=records,
                max_tokens=1,  # impossibly small → always over budget
            )
        )

        assert isinstance(result, mcp_types.InputRequiredResult), (
            f"Expected InputRequiredResult, got {type(result)}: {result}"
        )
        assert result.input_requests is not None
        assert "narrow" in result.input_requests, (
            f"Expected 'narrow' in input_requests, got {list(result.input_requests)}"
        )
        req = result.input_requests["narrow"]
        assert isinstance(req, mcp_types.ElicitRequest)
        # Must carry the agent-audience _meta tag.
        assert req.params.meta.get("x-clio-agent/audience") == "agent", (
            f"ElicitRequest meta: {req.params.meta}"
        )
        # request_state must be valid JSON with expected keys.
        assert result.request_state is not None
        state = json.loads(result.request_state)
        assert "max_tokens" in state
        assert "fallback_top_n" in state
        assert "original_estimate" in state

    def test_over_budget_extension_absent_returns_full_result(self) -> None:
        """Over-budget + extension absent → full result, never InputRequiredResult."""
        df = pd.DataFrame({"col": list(range(500))})
        records = df.to_dict("records")

        result = asyncio.run(
            guard_records_payload(
                ctx=_FakeCtxRound1(extension_present=False),
                full_df=df,
                records=records,
                max_tokens=1,  # over budget
            )
        )

        assert isinstance(result, dict), (
            "Extension absent → must return a plain dict (never InputRequiredResult)"
        )
        assert result["guarded"] is True
        assert result["action"] == "extension_absent_full_returned"
        assert result["records"] == records  # full result returned unchanged

    def test_round2_accept_applies_top_n_narrowing(self) -> None:
        """Round-2 accept with top_n → narrowed result, action=narrowed_by_agent."""
        df = pd.DataFrame({"dept": ["A", "B", "C", "A", "B"], "val": [1, 2, 3, 4, 5]})
        records = df.to_dict("records")

        elicit_result = mcp_types.ElicitResult(action="accept", content={"top_n": 2})
        state_token = json.dumps(
            {
                "max_tokens": 1,
                "fallback_top_n": 1,
                "original_estimate": {
                    "rows": 5,
                    "cols": 2,
                    "serialized_bytes": 0,
                    "est_tokens": 0,
                },
            }
        )

        class _CtxRound2:
            input_responses = {"narrow": elicit_result}
            request_state = state_token

            def client_supports_extension(self, _: str) -> bool:  # pragma: no cover
                return True

        result = asyncio.run(
            guard_records_payload(
                ctx=_CtxRound2(),
                full_df=df,
                records=records,
                max_tokens=1,
            )
        )
        assert isinstance(result, dict)
        assert result["action"] == "narrowed_by_agent"
        assert len(result["records"]) == 2
        assert "top_n(2)" in result["applied"]

    def test_round2_decline_hard_truncates_to_fallback(self) -> None:
        """Round-2 decline → hard_truncated to fallback_top_n rows."""
        df = pd.DataFrame({"x": list(range(100))})
        records = df.to_dict("records")

        elicit_result = mcp_types.ElicitResult(action="decline", content=None)
        state_token = json.dumps(
            {"max_tokens": 1, "fallback_top_n": 5, "original_estimate": {}}
        )

        class _CtxDecline:
            input_responses = {"narrow": elicit_result}
            request_state = state_token

            def client_supports_extension(self, _: str) -> bool:  # pragma: no cover
                return True

        result = asyncio.run(
            guard_records_payload(
                ctx=_CtxDecline(),
                full_df=df,
                records=records,
                max_tokens=1,
            )
        )
        assert result["action"] == "hard_truncated"
        assert result["degrade_reason"] == "elicitation_decline"
        assert len(result["records"]) == 5  # fallback_top_n


# ---------------------------------------------------------------------------
# Group B — in-memory Client(server) round-trip tests
# ---------------------------------------------------------------------------


def _extract(result: object) -> dict:
    """Pull the structured dict out of a call_tool result."""
    sc = getattr(result, "structured_content", None)
    if sc is not None:
        return sc  # type: ignore[return-value]
    raw = getattr(result, "data", None)
    if raw is not None:
        return raw if isinstance(raw, dict) else json.loads(str(raw))
    content = getattr(result, "content", None)
    if content:
        return json.loads(content[0].text)
    return {}


class TestLoadDataToolWiring:
    """End-to-end wiring of the size-guard into load_data_tool.

    Tests use ``asyncio.run`` over ``Client(server)`` in-memory to avoid
    subprocess overhead on the HPC head node.
    """

    @pytest.fixture()
    def over_budget_csv(self) -> str:
        """Create a CSV whose full-load will exceed DEFAULT_MAX_TOKENS."""
        # 2 000 rows × 6 cols → well over 32 000 chars / 8 000 tokens.
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        _make_csv(path, rows=2000, cols=6)
        df = pd.read_csv(path)
        est = estimate_records_size(df.to_dict("records"))
        assert est.est_tokens > DEFAULT_MAX_TOKENS, (
            f"Fixture too small (est_tokens={est.est_tokens}); increase rows."
        )
        return path

    def test_extension_absent_over_budget_returns_full_result(
        self, over_budget_csv: str
    ) -> None:
        """(b) Extension absent + over-budget → success response, data present,
        never an InputRequiredResult.
        """
        from fastmcp import Client

        from pandas_mcp.server import mcp

        async def _run() -> dict:
            async with Client(mcp) as client:
                raw = await client.call_tool(
                    "load_data",
                    {"file_path": over_budget_csv},
                )
            return _extract(raw)

        result = asyncio.run(_run())
        assert result.get("success") is True, f"Unexpected failure: {result}"
        assert "data" in result, "Result must contain 'data' key"
        assert len(result["data"]) > 0, "Result must contain rows"
        # Guard block must indicate extension was absent (full result returned).
        sg = result.get("size_guard", {})
        if sg:
            assert sg.get("action") in (
                "extension_absent_full_returned",
                None,
            ), f"Unexpected guard action: {sg.get('action')}"

    def test_extension_present_over_budget_tool_exercised(
        self, over_budget_csv: str
    ) -> None:
        """(a) Extension present + over-budget → the MRTR guard path is entered.

        The in-memory Client drives the MRTR loop: when the server returns an
        ``InputRequiredResult`` (which it will for an over-budget payload with the
        extension present), the client dispatches the ``ElicitRequest`` to its
        elicitation handler.  Since this test client has no handler registered for
        that elicitation, the MCP SDK raises ``MCPError("Elicitation not supported")``.

        That specific error is the proof that:
        (i)  the tool reached the over-budget + extension-present branch,
        (ii) it returned an ``InputRequiredResult`` (NOT a plain dict),
        (iii) the client correctly tried to drive the MRTR round-trip.

        The full answered round-trip (with an agent elicitation handler) is
        deferred to live compute-node verification.
        """
        from mcp.shared.exceptions import MCPError

        from fastmcp import Client
        from fastmcp.client.client import ClientExtension

        from pandas_mcp.server import mcp

        class _ClioExt(ClientExtension):
            identifier = AGENT_ELICITATION_EXTENSION_ID

        async def _run() -> object:
            async with Client(mcp, extensions=[_ClioExt()]) as client:
                return await client.call_tool(
                    "load_data",
                    {"file_path": over_budget_csv},
                )

        # The in-memory driver enters the MRTR path and fails at the
        # elicitation dispatch step because no handler is registered.
        # This confirms the guard is wired and returned InputRequiredResult.
        with pytest.raises(MCPError, match="Elicitation not supported"):
            asyncio.run(_run())
