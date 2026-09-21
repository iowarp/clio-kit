"""Report executable/configuration prerequisites separately from MCP connectivity."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import os
import shutil
import sys

import click

from clio_kit.protocol_probe import inspect_stdio


def server_prerequisites() -> dict:
    """Read checks from checkout metadata or the small released index; no fetch."""
    from clio_kit import get_servers_path
    from clio_kit.component_store import INDEX_FILE, catalogue
    from clio_kit.discovery import tomllib

    if INDEX_FILE.is_file():
        return {
            name: record.get("prerequisites", {})
            for name, record in catalogue()["servers"].items()
        }
    inventory = get_servers_path().parent / "mcp-server-versions.toml"
    if not inventory.is_file():
        return {}
    return tomllib.loads(inventory.read_text()).get("prerequisites", {})


@click.command("doctor")
@click.option(
    "--server",
    "servers",
    multiple=True,
    help="Check selected servers; repeat as needed. Defaults to all.",
)
@click.option(
    "--connect",
    is_flag=True,
    help="Also initialize real MCP sessions (requires verification extra).",
)
@click.option("--json", "as_json", is_flag=True)
def doctor_command(servers: tuple[str, ...], connect: bool, as_json: bool) -> None:
    """Check launcher prerequisites and optionally actual MCP connections.

    A connection confirms protocol availability, not that every scientific
    backend is configured or every tool result is numerically correct.
    """
    from clio_kit import list_available_servers

    available = list_available_servers()
    selected = list(servers) or available
    unknown = set(selected) - set(available)
    if unknown:
        raise click.ClickException(f"Unknown servers: {sorted(unknown)}")
    checks = []
    prerequisites = server_prerequisites()
    for server in selected:
        record: dict = {"server": server, "prerequisites": []}
        declared = prerequisites.get(server, {})
        for executable in declared.get("executables", []):
            record["prerequisites"].append(
                {"name": executable, "available": bool(shutil.which(executable))}
            )
        for variable in declared.get("environment", []):
            record["prerequisites"].append(
                {"name": variable, "available": bool(os.getenv(variable))}
            )
        if declared.get("note"):
            record["note"] = declared["note"]
        if connect:
            try:
                metadata = asyncio.run(
                    inspect_stdio(
                        sys.executable,
                        ["-c", "from clio_kit import cli; cli()", "mcp-server", server],
                    )
                )
                record.update(connected=True, tools=len(metadata["tools"]))
            except Exception as exc:
                record.update(connected=False, error=str(exc))
        checks.append(record)
    result = {
        "version": importlib.metadata.version("clio-kit"),
        "python": sys.version.split()[0],
        "uv": shutil.which("uv"),
        "servers": checks,
    }
    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        click.echo(
            f"CLIO Kit {result['version']}; Python {result['python']}; uv: {result['uv'] or 'MISSING'}"
        )
        for check in checks:
            missing = [p["name"] for p in check["prerequisites"] if not p["available"]]
            status = (
                "missing " + ", ".join(missing)
                if missing
                else "basic prerequisites checked"
            )
            if connect:
                status += (
                    "; connected"
                    if check.get("connected")
                    else f"; connection failed: {check.get('error')}"
                )
            click.echo(f"{check['server']}: {status}")
            if check.get("note"):
                click.echo(f"  {check['note']}")
    if not result["uv"] or any(
        any(not p["available"] for p in c["prerequisites"])
        or c.get("connected") is False
        for c in checks
    ):
        raise click.exceptions.Exit(1)
