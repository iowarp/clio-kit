"""Regressions for the 2026-10 pre-release acceptance findings (no ChronoLog needed)."""

from unittest.mock import Mock, patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from chronomcp import server
from chronomcp.capabilities import retrieve_handler
from chronomcp.capabilities.start_handler import start_chronolog
from chronomcp.capabilities.stop_handler import stop_chronolog
from chronomcp.utils import config


@pytest.fixture
def client(monkeypatch):
    """A stand-in native client; session state is restored afterwards."""
    fake = Mock()
    fake.Connect.return_value = 0
    fake.CreateChronicle.return_value = 0
    fake.AcquireStory.return_value = (0, object())
    fake.ReleaseStory.return_value = 0
    fake.Disconnect.return_value = 0
    monkeypatch.setattr(config, "client", fake)
    for name in ("_active_chronicle", "_active_story", "_story_handle"):
        monkeypatch.setattr(config, name, getattr(config, name))
    return fake


@pytest.mark.asyncio
async def test_start_reuses_an_existing_chronicle(client):
    """CL_ERR_CHRONICLE_EXISTS (-6) is not a failure: go on to acquire the story."""
    client.CreateChronicle.return_value = -6
    result = await start_chronolog("LLM", "conversation")
    assert "ChronoLog session started" in result
    client.AcquireStory.assert_called_once()
    client.Disconnect.assert_not_called()


@pytest.mark.asyncio
async def test_start_still_fails_on_other_create_errors(client):
    client.CreateChronicle.return_value = -7  # CL_ERR_NO_KEEPERS
    with pytest.raises(ToolError, match="Failed to create chronicle 'LLM': -7"):
        await start_chronolog("LLM", "conversation")


@pytest.mark.asyncio
async def test_stop_clears_session_when_story_is_already_released(client):
    await start_chronolog("LLM", "conversation")
    client.ReleaseStory.return_value = -5  # CL_ERR_NOT_ACQUIRED
    assert "stopped" in await stop_chronolog()
    assert config._story_handle is None


def test_status_resource_is_truthful(client, monkeypatch):
    with patch.object(server.importlib.util, "find_spec", return_value=None):
        assert server.chronolog_status()["status"] == "client_unavailable"
    with patch.object(server.importlib.util, "find_spec", return_value=object()):
        assert server.chronolog_status()["status"] == "no_session"
        monkeypatch.setattr(config, "_story_handle", object())
        monkeypatch.setattr(config, "_active_chronicle", "c")
        monkeypatch.setattr(config, "_active_story", "s")
        status = server.chronolog_status()
    assert status["status"] == "session_active"
    assert status["session"] == {"chronicle": "c", "story": "s"}


@pytest.mark.asyncio
async def test_retrieve_returns_an_absolute_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "CONFIG_FILE", "conf.json")
    out = 'CLIO_RECORD_JSON "user: a, assistant: b"\n'
    monkeypatch.setattr(retrieve_handler.helpers, "run_reader", lambda cmd: (out, ""))
    path = await retrieve_handler.retrieve_interaction("c", "s")
    assert path.startswith(str(tmp_path.resolve()))
    assert open(path).read() == "user: a, assistant: b"


@pytest.mark.asyncio
async def test_server_info_reports_release_version():
    async with Client(server.mcp) as mcp_client:
        assert mcp_client.server_info.version == "2.0.3"
