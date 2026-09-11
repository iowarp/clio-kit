"""Run contributed locked servers and verify their actual protocol surface."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click

from clio_kit.discovery import read_server_descriptor
from clio_kit.protocol_probe import inspect_stdio


@click.group("server")
def server_group() -> None:
    """Run or inspect a Python, TypeScript/Node, or Go server descriptor."""


@server_group.command("run", context_settings={"ignore_unknown_options": True})
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def run_server(directory: Path, args: tuple[str, ...]) -> None:
    """Launch a contributed project from its source and dependency locks."""
    from clio_kit import (
        _run_locked_local_server,
        subprocess_env_with_github_https_rewrite,
    )

    try:
        descriptor = read_server_descriptor(directory)
        if descriptor is None:
            raise ValueError(f"{directory} needs clio-server.toml")
        _run_locked_local_server(
            directory.resolve(),
            descriptor["entry"],
            args,
            subprocess_env_with_github_https_rewrite(),
        )
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


@server_group.command("inspect")
@click.argument(
    "directory", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option("--output", type=click.Path(path_type=Path))
def inspect_server(directory: Path, output: Path | None) -> None:
    """Initialize a real server and list all its tools, resources and prompts."""
    try:
        result = asyncio.run(
            inspect_stdio(
                sys.executable,
                [
                    "-c",
                    "from clio_kit import cli; cli()",
                    "server",
                    "run",
                    str(directory.resolve()),
                ],
            )
        )
    except Exception as exc:
        raise click.ClickException(f"MCP inspection failed: {exc}") from exc
    text = json.dumps(result, indent=2) + "\n"
    if output:
        output.write_text(text)
    else:
        click.echo(text, nl=False)
