"""``clio-kit mcp-server-identity <server>``: a server's code identity, no launch.

A client caching a server's tool listing needs to know when the served code
changed. An embedded server runs from its source and lock, so their hash (the
same identity the launcher keys its environment by) names exactly the tools it
serves: a cached listing is valid while the identity is unchanged.
"""

import json

from click.testing import CliRunner

import clio_kit
from clio_kit import get_servers_path, locked_server_project_identity


def _first_embedded_server() -> str:
    servers = sorted(
        p.name for p in get_servers_path().iterdir() if (p / "uv.lock").exists()
    )
    assert servers, "no embedded server with a lock in this checkout"
    return servers[0]


def test_it_prints_the_launch_identity_without_starting_the_server(monkeypatch) -> None:
    server_dir = _first_embedded_server()
    commands, dirs = clio_kit.auto_discover_mcps()
    name = next(key for key, directory in dirs.items() if directory == server_dir)
    started: list[object] = []
    monkeypatch.setattr(
        clio_kit, "_run_locked_local_server", lambda *a, **k: started.append(a)
    )

    result = CliRunner().invoke(clio_kit.main, ["mcp-server-identity", name])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == locked_server_project_identity(
        get_servers_path() / server_dir
    )
    assert started == []


def test_an_unknown_server_is_an_error() -> None:
    result = CliRunner().invoke(
        clio_kit.main, ["mcp-server-identity", "no-such-server"]
    )

    assert result.exit_code != 0
    assert "Unknown server" in result.output
