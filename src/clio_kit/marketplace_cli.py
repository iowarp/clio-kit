"""Manage the compiled meta-marketplace without regenerating MCP servers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click

from clio_kit.federation import refresh_marketplace
from clio_kit.skills import SkillProblem


@click.group("marketplace")
def marketplace_group() -> None:
    """Sync local components or refresh external marketplace collections."""


@marketplace_group.command("sync")
@click.option("--root", default=".", type=click.Path(exists=True, path_type=Path))
def sync(root: Path) -> None:
    """Discover checkout components and update native and website catalogues.

    Runs the trusted checkout's generator, without fetching upstream updates or
    executing contributed MCPs/hooks. Website builds and CI run this automatically.
    """
    root = root.resolve()
    script = root / "scripts/generate_marketplace.py"
    if not script.is_file() or not (root / "mcp-server-versions.toml").is_file():
        raise click.ClickException("--root must point to a CLIO Kit source checkout")
    try:
        completed = subprocess.run(
            [sys.executable, str(script), "--root", str(root), "--website"],
            cwd=root,
        )
    except OSError as exc:
        raise click.ClickException(f"Catalogue sync failed: {exc}") from exc
    if completed.returncode:
        # The generator has already printed its one-line error.
        raise click.exceptions.Exit(completed.returncode)


@marketplace_group.command("refresh")
@click.option("--root", default=".", type=click.Path(exists=True, path_type=Path))
def refresh(root: Path) -> None:
    """Fetch indexed collections and merge their plugins into marketplace.json."""
    try:
        result = refresh_marketplace(root.resolve())
    except (OSError, ValueError, SkillProblem, subprocess.SubprocessError) as exc:
        raise click.ClickException(f"Marketplace refresh failed: {exc}") from exc
    click.echo(
        f"Refreshed {len(result['marketplaces'])} collections; {len(result['imported_names'])} imported plugins."
    )
    click.echo(
        "Next: claude plugin marketplace update clio-kit; then update installed plugins and reload them."
    )
