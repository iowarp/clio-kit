"""Run an installed reviewer/planner through its client's noninteractive command."""

from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess

import click

from clio_kit.client_install import safe_destination, tomllib


@click.command("run-agent")
@click.argument("name")
@click.option("--client", type=click.Choice(["codex"]), required=True)
@click.option(
    "--project",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--prompt", required=True, help="Task and evidence to give this installed agent."
)
@click.option(
    "--model", help="Optional model supported by the selected client/account."
)
def run_agent(name: str, client: str, project: Path, prompt: str, model: str | None):
    """Invoke an installed agent; keep its permissions and the client's authentication."""
    try:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise ValueError("Invalid agent name")
        project = project.resolve()
        path = safe_destination(project, f".codex/agents/{name}.toml")
        role = tomllib.loads(path.read_text())
        if role.get("name") != name or role.get("sandbox_mode") != "read-only":
            raise ValueError("Only installed read-only Codex agents are supported")
        # Explicit invocation also works in CLI builds without custom-role delegation.
        args = [
            "codex",
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "-c",
            'approval_policy="never"',
            "-c",
            "developer_instructions=" + json.dumps(role["developer_instructions"]),
        ]
        for server in role.get("mcp_servers", {}):
            args.extend(["-c", f"mcp_servers.{json.dumps(server)}.enabled=false"])
        if model:
            args.extend(["--model", model])
        args.append("-")
        result = subprocess.run(args, cwd=project, input=prompt, text=True, timeout=600)
        if result.returncode:
            raise click.ClickException(
                f"{client} agent exited with status {result.returncode}"
            )
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        raise click.ClickException(str(exc)) from exc
