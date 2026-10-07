"""Regressions for the 2026-10 pre-release acceptance findings (no network)."""

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from arxiv_mcp.capabilities import category_search, date_search, text_search
from arxiv_mcp.capabilities.arxiv_base import generate_bibtex
from arxiv_mcp.server import mcp


async def _query(module, call) -> dict:
    """Run ``call`` with the arXiv HTTP seam mocked; return the params it sent."""
    with patch.object(
        module, "execute_arxiv_query", new=AsyncMock(return_value=[])
    ) as query:
        await call
    return query.call_args[0][0]


@pytest.mark.asyncio
async def test_multi_word_queries_stay_in_their_field():
    """``ti:a b`` only scopes ``a``; every word must be field-scoped."""
    cases = [
        (
            text_search.search_by_title("Attention Is All You Need"),
            # "is" is a stop word arXiv does not index
            "(ti:Attention AND ti:All AND ti:You AND ti:Need)",
        ),
        (
            text_search.search_by_abstract("burst buffer I/O"),
            "(abs:burst AND abs:buffer AND abs:I/O)",
        ),
        (
            text_search.search_papers_by_author("Yann LeCun"),
            "(au:Yann AND au:LeCun)",
        ),
        (text_search.search_papers_by_author("An"), "au:An"),
        (text_search.search_papers_by_author("Will Smith"), "(au:Will AND au:Smith)"),
        (text_search.search_by_title('"vision transformer" pruning'), None),
    ]
    for call, expected in cases:
        query = (await _query(text_search, call))["search_query"]
        assert query == (expected or '(ti:"vision transformer" AND ti:pruning)')


@pytest.mark.asyncio
async def test_search_arxiv_uses_cat_only_for_category_codes():
    topic = await _query(
        category_search, category_search.search_arxiv("diffusion models")
    )
    assert topic["search_query"] == "(all:diffusion AND all:models)"
    assert topic["sortBy"] == "relevance"
    for code in ("cs.LG", "astro-ph", "cond-mat.mes-hall"):
        params = await _query(category_search, category_search.search_arxiv(code))
        assert params["search_query"] == f"cat:{code}"
        assert params["sortBy"] == "submittedDate"


@pytest.mark.asyncio
async def test_invalid_date_is_rejected_before_querying():
    with pytest.raises(ValueError, match="expected YYYY-MM-DD"):
        await date_search.search_date_range("garbage", "2023-01-07")


def test_bibtex_keeps_old_style_archive_prefix_and_versionless_eprint():
    bibtex = generate_bibtex({"id": "http://arxiv.org/abs/hep-th/9901001v3"})
    assert "@article{hep-th/9901001v3," in bibtex
    assert "eprint = {hep-th/9901001}," in bibtex
    assert "url = {http://arxiv.org/abs/hep-th/9901001v3}" in bibtex


@pytest.mark.asyncio
async def test_failures_are_real_mcp_errors():
    """A failing tool must set is_error, not return an ``isError`` payload."""
    async with Client(mcp) as client:
        for tool, args in [
            ("export_to_bibtex", {"papers_json": "not json"}),
            ("search_arxiv", {"query": "cs.DC", "max_results": -1}),
            ("search_date_range", {"start_date": "garbage", "end_date": "2023-01-07"}),
        ]:
            with pytest.raises(ToolError):
                await client.call_tool(tool, args)
            result = await client.call_tool(tool, args, raise_on_error=False)
            assert result.is_error is True


@pytest.mark.asyncio
async def test_download_tools_are_not_read_only_and_version_is_release():
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        version = client.server_info.version
    for name in ("download_paper_pdf", "download_multiple_pdfs"):
        assert tools[name].annotations.readOnlyHint is False
    assert tools["get_pdf_url"].annotations.readOnlyHint is True
    assert version == "2.2.5"
