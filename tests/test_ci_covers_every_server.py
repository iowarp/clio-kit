"""Every shipped server is verified by some CI job.

CI discovers servers by looking for ``pyproject.toml``, which is how a server
in another language would get no lint, no type check and no test run, and ship
entirely unverified. Nothing said so at the time -- the discovery step simply
returned a shorter list.

This is the gate rather than a second set of lanes: writing node and go lanes
before a node or go server exists would test nothing. It fails the moment a
server is added that no job covers, which is the point at which those lanes
have to be written.
"""

from __future__ import annotations

import re
from pathlib import Path

from clio_kit.discovery import read_server_descriptor

REPO = Path(__file__).resolve().parents[1]
SERVERS = REPO / "clio-kit-mcp-servers"
QUALITY_CONTROL = REPO / ".github" / "workflows" / "quality_control.yml"

# Servers excluded from the shared matrix that a dedicated workflow covers
# instead. Each entry names the workflow, so a stale exemption is visible.
DEDICATED_WORKFLOWS = {"chronolog": "test-chronomcp.yml"}


def _discovery_command() -> str:
    """The line in CI that decides which servers get verified at all."""
    workflow = QUALITY_CONTROL.read_text(encoding="utf-8")
    match = re.search(r"all_mcps=\$\((.+?)\)\n", workflow, re.S)
    assert match, "could not find the server discovery step in quality_control.yml"
    return match.group(1)


def test_ci_still_discovers_servers_by_their_python_manifest() -> None:
    """Pins the assumption the rest of this file reasons about.

    If discovery stops keying on pyproject.toml, the coverage test below is
    reasoning about a rule CI no longer uses, and must be rewritten with it.
    """
    assert "pyproject.toml" in _discovery_command()


def test_every_shipped_server_is_covered_by_a_ci_job() -> None:
    uncovered = []
    for server_dir in sorted(p for p in SERVERS.iterdir() if p.is_dir()):
        if server_dir.name.startswith("."):
            continue
        descriptor = read_server_descriptor(server_dir)
        runtime = descriptor["runtime"] if descriptor else "python"

        covered_by_matrix = (
            runtime == "python" and (server_dir / "pyproject.toml").is_file()
        )
        if covered_by_matrix or server_dir.name in DEDICATED_WORKFLOWS:
            continue
        uncovered.append(f"{server_dir.name} ({runtime})")

    assert not uncovered, (
        "these servers ship without any CI job verifying them: "
        f"{uncovered}. The shared matrix runs Python tools (ruff, mypy, "
        "pytest), so a server in another language needs its own lane in "
        ".github/workflows/ and an entry in DEDICATED_WORKFLOWS above."
    )


def test_every_dedicated_workflow_exemption_is_live() -> None:
    """A stale exemption would silently excuse a server nothing checks."""
    for server, workflow in DEDICATED_WORKFLOWS.items():
        assert (SERVERS / server).is_dir(), f"{server} no longer exists"
        assert (REPO / ".github" / "workflows" / workflow).is_file(), workflow
