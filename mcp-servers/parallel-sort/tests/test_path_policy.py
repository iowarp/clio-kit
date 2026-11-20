"""Tool path arguments stay inside the allowed folders and out of hidden ones."""

import asyncio

import pytest
from fastmcp import Client

from parallel_sort_mcp.server import mcp

LOG = "2026-01-01 00:00:01 INFO started\n2026-01-01 00:00:02 ERROR failed\n"


def call(tool, arguments):
    async def exchange():
        async with Client(mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(exchange())


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "app.log").write_text(LOG)
    (work / ".env").write_text("TOKEN=secret-value\n")
    (tmp_path / "secret.txt").write_text("PRIVATE KEY\n")
    monkeypatch.chdir(work)
    monkeypatch.delenv("PARALLEL_SORT_MCP_ALLOWED_ROOTS")
    monkeypatch.delenv("CLIO_KIT_ARTIFACTS", raising=False)
    return work


def test_a_log_in_the_working_directory_is_analysed(workspace):
    assert not call("analyze_log_statistics", {"log_file": "app.log"}).is_error


@pytest.mark.parametrize(
    "path, reason", [("../secret.txt", "outside the folders"), (".env", "hidden files")]
)
def test_other_files_are_refused_before_the_tool_runs(workspace, path, reason):
    result = call("analyze_log_statistics", {"log_file": path})
    assert result.is_error
    text = result.content[0].text
    assert reason in text
    assert "secret-value" not in text and "PRIVATE KEY" not in text


def test_an_output_file_outside_is_refused(workspace, tmp_path):
    result = call(
        "sort_log_by_timestamp",
        {"log_file": "app.log", "output_file": str(tmp_path / "sorted.log")},
    )
    assert result.is_error
    assert not (tmp_path / "sorted.log").exists()
