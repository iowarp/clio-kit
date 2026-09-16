"""Archive retrieval must preserve records and distinguish reader failures."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastmcp.exceptions import ToolError
from chronomcp.capabilities.retrieve_handler import retrieve_interaction
from chronomcp.utils import helpers


@pytest.mark.asyncio
async def test_quoted_multiline_record_and_untrusted_names(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    expected = 'user: "quoted"\nassistant: preserved \\ content'
    monkeypatch.setattr(
        helpers,
        "run_reader",
        lambda args: ("CLIO_RECORD_JSON " + json.dumps(expected) + "\n", ""),
    )
    filename = await retrieve_interaction("../../outside", "story/../../other")
    assert Path(filename).parent == tmp_path
    assert Path(filename).read_text() == expected


def test_failed_reader_does_not_become_no_records(monkeypatch):
    monkeypatch.setattr(
        helpers.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(
            returncode=2, stdout="", stderr="archive cannot be opened"
        ),
    )
    with pytest.raises(ToolError, match="archive cannot be opened"):
        helpers.run_reader(["reader"])
