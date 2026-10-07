#!/usr/bin/env python3
"""Check every external source route through every project installer and real MCP.

Uses isolated local Git/registry fixtures from the native marketplace acceptance
script. No public PR or model calls; model-driven evidence is a separate check.
"""

import argparse
import asyncio
import json
import os
from pathlib import Path

from clio_kit.client_install import CLIENTS, tomllib
from verify_external_contributions import Acceptance, NAME, ROOT


class ClientAcceptance(Acceptance):
    def route(self, route: str, source: dict, kind: str = "plugin"):
        catalogue = self.prepare_catalogue(route, source, kind)
        for client, (skills, configuration, key) in CLIENTS.items():
            project = catalogue / client
            args = [
                str(ROOT / ".venv/bin/clio-kit"),
                "plugin",
                "install",
                NAME,
                "--root",
                str(catalogue),
                "--client",
                client,
                "--project",
                str(project),
            ]
            label = f"{route}-{client}"
            self.run(label + "-install", args, ROOT)
            skill = project / skills / "clio-kit-scientific-debugging/SKILL.md"
            expected = (
                ROOT
                / "skills/clio-coder-skills/skills/clio-kit-scientific-debugging/SKILL.md"
            )
            assert skill.read_bytes() == expected.read_bytes()
            path = project / configuration
            original = path.read_bytes()
            config = (
                tomllib.loads(path.read_text())
                if client == "codex"
                else json.loads(path.read_text())
            )
            settings = config[key]["numerics"]
            if client == "opencode":
                settings = {
                    "command": settings["command"][0],
                    "args": settings["command"][1:],
                    "env": settings.get("environment", {}),
                }
            # Query the installed client configuration, not the source fixture.
            fixture = catalogue / (client + "-query")
            fixture.mkdir()
            (fixture / ".mcp.json").write_text(
                json.dumps({"mcpServers": {"numerics": settings}})
            )
            asyncio.run(asyncio.wait_for(self.query(fixture, label), timeout=30))
            self.run(
                label + "-offline-reinstall",
                args,
                ROOT,
                {**self.env, "CLIO_KIT_OFFLINE": "1"},
            )
            assert path.read_bytes() == original
            self.run(label + "-update", [*args, "--update", "--replace"], ROOT)
            self.run(
                label + "-remove",
                [
                    str(ROOT / ".venv/bin/clio-kit"),
                    "plugin",
                    "uninstall",
                    NAME,
                    "--client",
                    client,
                    "--project",
                    str(project),
                ],
                ROOT,
            )
            assert not skill.exists()
            after = (
                tomllib.loads(path.read_text())
                if client == "codex"
                else json.loads(path.read_text())
            )
            assert not after.get(key, {}).get("numerics")
            self.record(
                route=route,
                client=client,
                status="passed",
                stage="install-query-offline-update-remove",
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    acceptance = ClientAcceptance(args.output.resolve())
    # Isolated Git rewrites are inherited by the actual launcher subprocesses.
    old = dict(os.environ)
    try:
        os.environ.update(acceptance.env)
        acceptance.execute()
    finally:
        os.environ.clear()
        os.environ.update(old)
    return int(any(r.get("status") == "failed" for r in acceptance.records))


if __name__ == "__main__":
    raise SystemExit(main())
