"""Manage the compiled meta-marketplace without regenerating MCP servers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import click

from clio_kit.federation import refresh_marketplace


@click.group("marketplace")
def marketplace_group() -> None:
    """Refresh external collections in a CLIO marketplace checkout."""


@marketplace_group.command("refresh")
@click.option("--root", default=".", type=click.Path(exists=True, path_type=Path))
def refresh(root: Path) -> None:
    """Fetch indexed collections and merge their plugins into marketplace.json."""
    try:
        result = refresh_marketplace(root.resolve())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise click.ClickException(f"Marketplace refresh failed: {exc}") from exc
    click.echo(
        f"Refreshed {len(result['marketplaces'])} collections; {len(result['imported_names'])} imported plugins."
    )
    click.echo(
        "Next: claude plugin marketplace update clio-kit; then update installed plugins and reload them."
    )
