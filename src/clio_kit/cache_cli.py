"""The `clio-kit cache` command group over the private MCP runtime cache.

Split out of the launcher module, which the size ratchet holds at a fixed line
count precisely because it had grown into the place every new command landed.
The commands here read and reclaim the content-addressed environment tree that
:mod:`clio_kit.env_cache` maintains; the launcher keeps only the code that
actually starts a server.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import click

from clio_kit.env_cache import (
    CacheInUseError,
    collect_cache_gc,
    discover_servers,
    load_cache_policy,
    measure_cache_budget,
)


def clio_cache_root() -> Path:
    """Return the operator-configurable cache root used by child runtimes."""
    configured_cache = os.getenv("CLIO_KIT_CACHE_DIR")
    if configured_cache:
        return Path(configured_cache).expanduser().resolve()
    return (
        Path(os.getenv("XDG_CACHE_HOME", str(Path.home() / ".cache"))).expanduser()
        / "clio-kit"
    ).resolve()


@click.group("cache")
def cache_group() -> None:
    """Inspect and reclaim the private MCP runtime cache."""


@cache_group.command("gc")
@click.option(
    "--keep",
    type=int,
    default=None,
    help="Environments to keep per server (overrides CLIO_KIT_ENV_KEEP).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report what would be evicted without deleting anything.",
)
@click.option(
    "--component-keep",
    type=click.IntRange(min=1),
    default=None,
    help="Component versions to keep (overrides CLIO_KIT_COMPONENT_KEEP).",
)
@click.option(
    "--include-legacy",
    is_flag=True,
    help="Also select untracked legacy payloads; unknown consumers cannot be protected.",
)
def cache_gc(
    keep: int | None, dry_run: bool, component_keep: int | None, include_legacy: bool
) -> None:
    """Collapse every server to its newest N specs and prune the uv cache.

    This is the manual reclaim path for a box already polluted by unbounded
    environment history. It refuses to run while any environment is held by a
    live server, because deleting an environment mid-spawn corrupts the cache.
    """
    # Imported here rather than at module scope: the launcher imports this
    # module to register the group, so a top-level import would cycle.
    from clio_kit import uv_command

    from clio_kit.component_cache import component_retention

    try:
        component_keep = component_retention(component_keep)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    cache_root = clio_cache_root()
    policy = load_cache_policy()
    if keep is not None:
        if keep < 1:
            raise click.ClickException("--keep must be >= 1")
        policy = dataclasses.replace(policy, keep_per_server=keep)
    try:
        eviction, prune = collect_cache_gc(
            cache_root,
            policy=policy,
            uv_executable=uv_command(),
            dry_run=dry_run,
        )
    except CacheInUseError as exc:
        raise click.ClickException(str(exc)) from exc
    from clio_kit.component_cache import prune_component_cache

    try:
        components = prune_component_cache(
            keep=component_keep, dry_run=dry_run, include_legacy=include_legacy
        )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc
    budget = measure_cache_budget(cache_root, policy=policy)
    click.echo(
        json.dumps(
            {
                "dry_run": dry_run,
                "components": components,
                "keep_per_server": policy.keep_per_server,
                "evicted": [
                    {
                        "server": entry.server,
                        "hash_prefix": entry.hash_prefix,
                        "bytes_freed": entry.bytes_freed,
                    }
                    for entry in eviction.evicted
                ],
                "skipped_in_use": [
                    {"server": entry.server, "hash_prefix": entry.hash_prefix}
                    for entry in eviction.skipped_in_use
                ],
                "bytes_freed": eviction.bytes_freed,
                "uv_cache_prune": {
                    "ran": prune.ran,
                    "ok": prune.ok,
                    "reason": prune.reason,
                },
                "cache_total_bytes": budget.total_bytes,
                "over_budget": budget.over_budget,
            },
            sort_keys=True,
        )
    )


@cache_group.command("status")
def cache_status() -> None:
    """Print a machine-readable summary of the private runtime cache footprint."""
    cache_root = clio_cache_root()
    policy = load_cache_policy()
    budget = measure_cache_budget(cache_root, policy=policy)
    environments_root = cache_root / "mcp-environments"
    per_server: dict[str, int] = {}
    if environments_root.is_dir():
        for server in sorted(discover_servers(cache_root)):
            token = f"{server}-"
            per_server[server] = sum(
                1
                for child in environments_root.iterdir()
                if child.is_dir() and child.name.startswith(token)
            )
    click.echo(
        json.dumps(
            {
                "cache_root": str(cache_root),
                "total_bytes": budget.total_bytes,
                "max_bytes": budget.max_bytes,
                "over_budget": budget.over_budget,
                "keep_per_server": policy.keep_per_server,
                "environments_per_server": per_server,
            },
            sort_keys=True,
        )
    )


@cache_group.command("components")
@click.option(
    "--keep",
    type=click.IntRange(min=1),
    default=None,
    help="Component versions to keep (CLIO_KIT_COMPONENT_KEEP, default 2).",
)
@click.option(
    "--dry-run/--apply",
    default=True,
    help="Preview by default; --apply deletes selected payloads.",
)
@click.option(
    "--include-legacy",
    is_flag=True,
    help="Also select untracked legacy payloads; unknown consumers cannot be protected.",
)
def component_gc(keep: int | None, dry_run: bool, include_legacy: bool) -> None:
    """Preview or prune superseded, unreferenced component payloads."""
    from clio_kit.component_cache import prune_component_cache

    try:
        click.echo(
            json.dumps(
                prune_component_cache(
                    keep=keep, dry_run=dry_run, include_legacy=include_legacy
                ),
                sort_keys=True,
            )
        )
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


@cache_group.command("forget-project")
@click.option(
    "--config",
    type=click.Path(path_type=Path),
    required=True,
    help="Original absolute client configuration path.",
)
@click.option(
    "--dry-run/--apply",
    default=True,
    help="Preview by default; --apply releases this project's pins.",
)
@click.option(
    "--confirm-unused",
    is_flag=True,
    help="Confirm no configuration, backup or running process still uses these payloads.",
)
def forget_project_refs(config: Path, dry_run: bool, confirm_unused: bool) -> None:
    """Release project pins after retiring its component consumers."""
    from clio_kit.component_cache import forget_project

    if not dry_run and not confirm_unused:
        raise click.ClickException(
            "--apply requires --confirm-unused; live consumers may break after cleanup"
        )
    try:
        click.echo(json.dumps(forget_project(config, dry_run=dry_run), sort_keys=True))
    except (ValueError, OSError) as exc:
        raise click.ClickException(str(exc)) from exc


CACHE_GROUP = cache_group
