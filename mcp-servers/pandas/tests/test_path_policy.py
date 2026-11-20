"""Tool path arguments stay inside the allowed folders and out of hidden ones."""

import asyncio
import os

import pytest
from fastmcp import Client

from pandas_mcp.server import mcp


#: The argument that names the file, per tool.
PATH_ARGUMENT = {"profile_csv": "data_path"}


def call(tool, arguments):
    async def exchange():
        async with Client(mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(exchange())


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A working directory holding data, with secrets inside and outside it."""
    work = tmp_path / "work"
    work.mkdir()
    rows = "".join(f"{i},{i * i % 7}\n" for i in range(12))
    (work / "data.csv").write_text("a,b\n" + rows)
    (work / ".env").write_text("TOKEN=secret-value\n")
    (tmp_path / "home" / ".ssh").mkdir(parents=True)
    (tmp_path / "home" / ".ssh" / "id_rsa").write_text("PRIVATE KEY\n")
    (tmp_path / "outside.csv").write_text("a,b\n3,4\n")
    monkeypatch.chdir(work)
    monkeypatch.delenv("PANDAS_MCP_ALLOWED_ROOTS")
    monkeypatch.delenv("CLIO_KIT_ARTIFACTS", raising=False)
    return work


@pytest.mark.parametrize("tool", ["load_data", "profile_csv", "statistical_summary"])
def test_a_file_in_the_working_directory_is_read(workspace, tool):
    argument = PATH_ARGUMENT.get(tool, "file_path")
    assert not call(tool, {argument: str(workspace / "data.csv")}).is_error


@pytest.mark.parametrize(
    "path, reason",
    [
        ("../outside.csv", "outside the folders"),
        ("../home/.ssh/id_rsa", "outside the folders"),
        (".env", "hidden files and folders"),
    ],
)
@pytest.mark.parametrize("tool", ["load_data", "profile_csv", "statistical_summary"])
def test_other_files_are_refused_before_the_tool_runs(workspace, tool, path, reason):
    result = call(tool, {PATH_ARGUMENT.get(tool, "file_path"): path})
    assert result.is_error
    text = result.content[0].text
    assert reason in text
    assert "secret-value" not in text and "PRIVATE KEY" not in text


def test_a_link_out_of_the_working_directory_is_refused(workspace, tmp_path):
    os.symlink(tmp_path / "outside.csv", workspace / "linked.csv")
    assert call("load_data", {"file_path": "linked.csv"}).is_error


def test_hidden_folders_are_refused_even_when_home_is_the_working_directory(
    workspace, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path / "home")
    result = call("load_data", {"file_path": ".ssh/id_rsa", "file_format": "csv"})
    assert result.is_error
    assert "hidden files and folders (.ssh)" in result.content[0].text


def test_configured_roots_and_the_artifacts_folder_are_allowed(
    workspace, tmp_path, monkeypatch
):
    target = str(tmp_path / "outside.csv")
    assert call("load_data", {"file_path": target}).is_error
    monkeypatch.setenv("PANDAS_MCP_ALLOWED_ROOTS", str(tmp_path))
    assert not call("load_data", {"file_path": target}).is_error
    monkeypatch.delenv("PANDAS_MCP_ALLOWED_ROOTS")
    monkeypatch.setenv("CLIO_KIT_ARTIFACTS", str(tmp_path))
    assert not call("load_data", {"file_path": target}).is_error


def test_output_files_are_held_to_the_same_folders(workspace, tmp_path):
    result = call(
        "save_data",
        {"data": {"a": [1]}, "file_path": str(tmp_path / "home" / "written.csv")},
    )
    assert result.is_error
    assert not (tmp_path / "home" / "written.csv").exists()
