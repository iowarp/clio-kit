"""``clio-kit mcp-server-identity <server>``: an embedded server's code identity."""

import json
import sys

import click

from clio_kit import (
    auto_discover_mcps,
    get_servers_path,
    locked_server_project_identity,
    main,
)


@main.command("mcp-server-identity")
@click.argument("server")
def mcp_server_identity(server):
    """Print an embedded server's code identity as JSON, without starting it.

    The identity hashes the server's embedded source and lock -- the same key its
    locked runtime environment is addressed by -- so it names exactly the tools the
    server serves: a client may reuse a cached tool listing while it is unchanged.
    """

    server_command_map, dir_name_map = auto_discover_mcps()
    server_lower = server.lower()
    if server_lower not in server_command_map:
        click.echo(f"Error: Unknown server '{server}'")
        sys.exit(1)
    server_path = get_servers_path() / dir_name_map[server_lower]
    if not server_path.exists():
        click.echo(
            f"Error: server '{server}' is not embedded here; it has no code identity"
        )
        sys.exit(1)
    click.echo(json.dumps(locked_server_project_identity(server_path), sort_keys=True))
